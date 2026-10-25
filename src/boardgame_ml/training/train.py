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
4. Logs every training episode return to ``train_log.csv`` and writes a
   ``summary.json`` (full config + best eval score) into the run dir.
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
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from boardgame_ml.envs import make_env
from boardgame_ml.envs.base import validate_mask
from boardgame_ml.training.algorithms import MODEL_CLASSES, ensure_mask_compatible
from boardgame_ml.training.config import ExperimentConfig


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


def build_single_env(
    game: str, *, monitor: bool = True, env_kwargs: dict | None = None
) -> gym.Env:
    """One masked env (used for evaluation / final checks)."""
    env = make_env(game, **(env_kwargs or {}))
    if monitor:
        env = Monitor(env)
    return ActionMasker(env, mask_fn)


def build_vec_env(
    game: str, n_envs: int, seed: int, env_kwargs: dict | None = None
) -> DummyVecEnv:
    def make(rank: int):
        def _init() -> gym.Env:
            env = Monitor(make_env(game, **(env_kwargs or {})))
            env = ActionMasker(env, mask_fn)
            return env

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

    run_dir = (
        Path(config.output_dir)
        / f"{config.game}_{config.algorithm}_s{config.seed}_{config_hash(config)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_vec_env(
        config.game, config.n_envs, config.seed, env_kwargs=config.env_kwargs
    )
    eval_env = build_vec_env(
        config.game, 1, config.seed + 10_000, env_kwargs=config.env_kwargs
    )

    model = build_model(config, train_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(1, config.eval_freq // config.n_envs),
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        verbose=0,
    )
    logger = EpisodeReturnLogger(run_dir / "train_log.csv")

    start = time.time()
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[eval_callback, logger, *(extra_callbacks or [])],
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
