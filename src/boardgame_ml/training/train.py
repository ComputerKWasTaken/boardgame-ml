"""Training entry point.

    python -m boardgame_ml.training.train --config configs/blackjack_maskable_ppo.yaml

What a run does:

1. Builds vectorized training envs and a *separate* eval env from the
   config (explicitly — no global state).
2. Trains the configured SB3 algorithm. Action masks flow through the
   ``ActionMasker`` wrapper into MaskablePPO for BOTH action selection
   and learning updates, so the distribution that acts is the
   distribution that learns (the legacy project's fatal mismatch).
3. Periodically evaluates the deterministic policy on the eval env and
   keeps the **best-by-eval** checkpoint (``best_model.zip``) — never a
   "best" picked from noisy, exploration-contaminated training returns.
4. Logs every training episode return to ``train_log.csv``, SB3's
   own logger diagnostics to ``diagnostics.csv`` (see
   ``training/diagnostics.py``), and writes a ``summary.json``
   (full config + best eval score) into the run dir.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import boardgame_ml
import gymnasium as gym
import numpy as np
import sb3_contrib
import stable_baselines3
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from boardgame_ml.agents.baselines import get_baseline
from boardgame_ml.envs import TWO_PLAYER_GAMES, make_env
from boardgame_ml.envs.base import validate_mask
from boardgame_ml.envs.two_player import TwoPlayerEnv
from boardgame_ml.training.algorithms import MODEL_CLASSES, ensure_mask_compatible
from boardgame_ml.training.config import ExperimentConfig
from boardgame_ml.training.diagnostics import DiagnosticsCallback
from boardgame_ml.training.self_play import (
    OpponentPool,
    SelfPlayConfig,
    SelfPlayWrapper,
    SnapshotCallback,
)


def mask_fn(env: gym.Env) -> np.ndarray:
    """ActionMasker callback: read (and validate) the raw game env's mask."""
    raw = env.unwrapped
    return validate_mask(raw.action_masks(), raw.action_space.n)


def config_hash(config: ExperimentConfig) -> str:
    """8-char hash of the canonicalized config JSON.

    Run directories carry this suffix so two runs that differ anywhere
    in their config (a swept learning rate, a different hyperparameter)
    never share — and overwrite — one directory.
    """
    canonical = json.dumps(config.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def library_versions() -> dict[str, str]:
    """Versions of everything that can change a run's numerics."""
    return {
        "boardgame_ml": boardgame_ml.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "gymnasium": gym.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def _maybe_mask(env: gym.Env) -> gym.Env:
    """Wrap in ``ActionMasker`` iff the game has masks to deliver.

    Action masks exist only for ``Discrete`` action spaces. A ``Box``
    game (continuous control) has no mask — every point of the box is
    legal — so the wrapper (whose ``mask_fn`` reads ``action_space.n``)
    must not be applied to it. This is the one place the trainer's
    wiring branches on the action-space kind.
    """
    if isinstance(env.action_space, gym.spaces.Discrete):
        return ActionMasker(env, mask_fn)
    return env


def _maybe_wrap_self_play(
    env: gym.Env,
    game: str,
    self_play_cfg: SelfPlayConfig | None,
    *,
    seed: int,
    pools: list[OpponentPool] | None = None,
    fixed_opponent=None,
    fixed_opponent_kind: str = "fixed",
) -> gym.Env:
    """Wrap a two-player game for self-play when the config asks.

    Training envs get a league-backed wrapper (each vec slot its own
    :class:`OpponentPool` over the shared snapshot files, registered
    into ``pools`` so the SnapshotCallback can feed them all). The
    eval env gets a wrapper with one *fixed* opponent — the game's
    heuristic baseline — so the eval curve and best-by-eval selection
    measure the learner against a constant yardstick, not a moving
    league. Single-agent configs pass through untouched.
    """
    if self_play_cfg is None:
        return env
    raw = env.unwrapped
    if not isinstance(raw, TwoPlayerEnv):
        raise ValueError(
            f"Config sets self_play for game {game!r}, which is not a "
            "two-player game; self-play settings apply only to "
            f"{sorted(TWO_PLAYER_GAMES)}"
        )
    if fixed_opponent is not None:
        return SelfPlayWrapper(
            env,
            fixed_opponent=fixed_opponent,
            fixed_opponent_kind=fixed_opponent_kind,
        )
    pool = OpponentPool(game, self_play_cfg, seed=seed)
    if pools is not None:
        pools.append(pool)
    return SelfPlayWrapper(env, pool)


def build_single_env(
    game: str, *, monitor: bool = True, env_kwargs: dict | None = None
) -> gym.Env:
    """One env (used for evaluation / final checks); masked if Discrete."""
    env = make_env(game, **(env_kwargs or {}))
    if monitor:
        env = Monitor(env)
    return _maybe_mask(env)


def build_vec_env(
    game: str,
    n_envs: int,
    seed: int,
    env_kwargs: dict | None = None,
    *,
    self_play_cfg: SelfPlayConfig | None = None,
    pools: list[OpponentPool] | None = None,
) -> DummyVecEnv:
    def make(rank: int):
        def _init() -> gym.Env:
            env = _maybe_wrap_self_play(
                make_env(game, **(env_kwargs or {})),
                game,
                self_play_cfg,
                seed=seed + rank,
                pools=pools,
            )
            env = Monitor(env)
            return _maybe_mask(env)

        return _init

    vec = DummyVecEnv([make(i) for i in range(n_envs)])
    vec.seed(seed)
    return vec


def build_model(config: ExperimentConfig, env: gym.Env):
    """Construct the SB3 model, with config values passed explicitly."""
    cls = MODEL_CLASSES[config.algorithm]
    return cls(
        config.policy,
        env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        seed=config.seed,
        verbose=0,
        policy_kwargs=config.policy_kwargs,
        **config.hyperparams,
    )


class EpisodeReturnLogger(BaseCallback):
    """Append every completed training episode's return to a CSV file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._fh = path.open("w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["timesteps", "episode_return", "episode_length"])

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                ep = info["episode"]
                self._writer.writerow([self.num_timesteps, ep["r"], ep["l"]])
                self._fh.flush()
        return True

    def close(self) -> None:
        self._fh.close()


def _build_eval_env(
    config: ExperimentConfig,
    self_play_cfg: SelfPlayConfig | None,
) -> DummyVecEnv:
    """The held-out eval env: single, separately seeded.

    For self-play configs the learner faces one fixed opponent — the
    game's heuristic baseline — on both seats (the wrapper randomizes
    the learner's seat per episode), so the eval mean return is a
    *win-rate-like* number against a constant yardstick: -1 is
    always losing, +1 is always winning, and best-by-eval selects
    for beating it from both chairs.
    """
    seed = config.seed + 10_000
    if self_play_cfg is None:
        return build_vec_env(
            config.game, 1, seed, env_kwargs=config.env_kwargs
        )
    heuristic = get_baseline(self_play_cfg.heuristic, config.game)

    def _init() -> gym.Env:
        env = SelfPlayWrapper(
            make_env(config.game, **config.env_kwargs),
            fixed_opponent=heuristic,
            fixed_opponent_kind="eval_heuristic",
        )
        return _maybe_mask(Monitor(env))

    vec = DummyVecEnv([_init])
    vec.seed(seed)
    return vec


def train(
    config: ExperimentConfig,
    *,
    extra_callbacks: list[BaseCallback] | None = None,
) -> dict:
    """Run one training experiment and return its summary dict.

    ``extra_callbacks`` lets embedders (the web cockpit's job runner)
    observe progress and request an early stop — a callback whose
    ``_on_step`` returns False ends ``learn`` gracefully, after which
    this function still saves ``final_model.zip`` and ``summary.json``
    and the EvalCallback's ``best_model.zip`` (if any eval ran) stays
    on disk. The CLI passes no extra callbacks; behavior is unchanged.
    """
    # Mask guard first, before any env/model construction: never pair
    # a mask-blind algorithm with a game that masks (legacy failure #1).
    ensure_mask_compatible(
        config.game,
        config.algorithm,
        allow_unmasked=config.allow_unmasked,
        env_kwargs=config.env_kwargs,
    )

    self_play_cfg = (
        SelfPlayConfig.from_dict(config.self_play) if config.self_play else None
    )
    if self_play_cfg is not None and config.game not in TWO_PLAYER_GAMES:
        raise ValueError(
            f"Config sets self_play for game {config.game!r}, which is "
            "not a two-player game; self-play settings apply only to "
            f"{sorted(TWO_PLAYER_GAMES)}"
        )

    run_dir = (
        Path(config.output_dir)
        / f"{config.game}_{config.algorithm}_s{config.seed}_{config_hash(config)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    pools: list[OpponentPool] = []
    train_env = build_vec_env(
        config.game,
        config.n_envs,
        config.seed,
        env_kwargs=config.env_kwargs,
        self_play_cfg=self_play_cfg,
        pools=pools,
    )
    eval_env = _build_eval_env(config, self_play_cfg)

    model = build_model(config, train_env)

    # MaskablePPO must be *evaluated* with masks too: SB3's plain
    # EvalCallback predicts unmasked, which was invisible while every
    # game's mask was trivially all-True (M1/M2) but is fatal for a
    # game with real illegal actions — an unmasked deterministic
    # prediction can pick a rejected action on the same state forever
    # and the eval episode never ends (found by Match 3, M3).
    # MaskableEvalCallback is the same callback, mask-aware; the
    # non-maskable algorithms keep the standard one.
    eval_callback_cls = (
        MaskableEvalCallback
        if MODEL_CLASSES[config.algorithm] is MaskablePPO
        else EvalCallback
    )
    eval_callback = eval_callback_cls(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(1, config.eval_freq // config.n_envs),
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        verbose=0,
    )
    logger = EpisodeReturnLogger(run_dir / "train_log.csv")
    # Diagnostics (M8): SB3's logger values (value loss, explained
    # variance, KL, clip fraction, ...) plus policy std / update size,
    # kept in diagnostics.csv — see training/diagnostics.py for why.
    diagnostics = DiagnosticsCallback(run_dir / "diagnostics.csv")
    callbacks: list[BaseCallback] = [eval_callback, logger, diagnostics]
    if self_play_cfg is not None:
        callbacks.append(SnapshotCallback(pools, run_dir, self_play_cfg))
    callbacks.extend(extra_callbacks or [])

    start = time.time()
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
    )
    elapsed = time.time() - start
    logger.close()
    model.save(str(run_dir / "final_model"))

    # Best eval score as recorded by EvalCallback (deterministic, held-out env).
    best_eval_mean: float | None = None
    npz_path = run_dir / "eval_logs" / "evaluations.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        best_eval_mean = float(np.max(np.mean(data["results"], axis=1)))

    summary = {
        "config": config.to_dict(),
        "config_hash": config_hash(config),
        "versions": library_versions(),
        "run_dir": str(run_dir),
        "best_eval_mean_return": best_eval_mean,
        "best_model": str(run_dir / "best_model.zip"),
        "final_model": str(run_dir / "final_model.zip"),
        "train_seconds": round(elapsed, 1),
    }
    if self_play_cfg is not None:
        pool_dir = run_dir / "pool"
        summary["pool_snapshots"] = (
            sorted(p.name for p in pool_dir.glob("*.zip"))
            if pool_dir.is_dir()
            else []
        )
        summary["self_play"] = self_play_cfg.to_dict()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    train_env.close()
    eval_env.close()
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train an agent from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML.")
    args = parser.parse_args(argv)
    config = ExperimentConfig.from_yaml(args.config)
    summary = train(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
