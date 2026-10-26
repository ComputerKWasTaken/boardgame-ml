"""Background training jobs for the web cockpit.

A job wraps the framework's own :func:`boardgame_ml.training.train`
(it is *not* a reimplementation): the trainer accepts
``extra_callbacks``, and this module supplies two —

* :class:`ProgressCallback` records every finished episode's return
  (via the Monitor infos, exactly like the trainer's own CSV logger)
  and polls EvalCallback's ``evaluations.npz`` so the UI can show
  eval results and the best-so-far as they land;
* the same callback doubles as the **stop mechanism**: it returns
  ``False`` from ``_on_step`` once a stop is requested, which ends
  SB3's ``learn`` loop gracefully. ``train()`` then still saves
  ``final_model.zip`` and ``summary.json``, and the EvalCallback's
  ``best_model.zip`` (written during training whenever an eval
  improved) stays on disk — so stopping keeps the best checkpoint.
  Caveat: a job stopped before its *first* eval has no
  ``best_model.zip``, only ``final_model.zip``.

Jobs run on daemon threads inside the server process. This is a
local, single-user tool; there is deliberately no queue, no worker
pool, and no persistence of job state across server restarts (the
runs themselves persist in ``runs/``, as always).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from boardgame_ml.training.config import ExperimentConfig
from boardgame_ml.training.train import config_hash, train

# ---------------------------------------------------------------------------
# Presets and per-pair defaults (mirroring configs/*.yaml)
# ---------------------------------------------------------------------------

#: UI presets: total timesteps plus eval cadence. "standard" tracks
#: the shipped YAML configs' budgets; quick/long scale around them.
#: Each field is a lookup table keyed by "game:algorithm", then
#: "game", then "default" (see :func:`_preset_value`). Eval cadence
#: is per game because episode lengths differ by orders of magnitude:
#: the shipped YAMLs evaluate racing on 10 episodes (each up to
#: 1,800 steps) and the board/card games on hundreds.
PRESETS: dict[str, dict[str, dict[str, int]]] = {
    "quick": {
        "timesteps": {
            "default": 50_000,
            "blackjack": 20_000,
            "higher_or_lower": 10_000,
            "match3": 100_000,
            "flappy": 100_000,
            "connect_four": 300_000,
            "racing": 250_000,
            "racing:sac": 50_000,
            "gym_classic": 25_000,
        },
        "eval_freq": {
            "default": 5_000,
            "match3": 25_000,
            "racing": 10_000,
        },
            "eval_episodes": {"default": 200, "racing": 10, "connect_four": 40},
    },
    "standard": {
        "timesteps": {
            "default": 50_000,
            "blackjack": 100_000,
            "higher_or_lower": 60_000,
            "match3": 700_000,
            "flappy": 300_000,
            "connect_four": 2_000_000,
            "racing": 600_000,
            "racing:sac": 150_000,
            "gym_classic": 100_000,
        },
        "eval_freq": {
            "default": 10_000,
            "blackjack": 20_000,
            "higher_or_lower": 10_000,
            "match3": 25_000,
            "flappy": 25_000,
            "connect_four": 25_000,
            "racing": 25_000,
            "racing:sac": 10_000,
        },
        "eval_episodes": {
            "default": 500,
            "match3": 200,
            "flappy": 200,
            "connect_four": 100,
            "racing": 10,
        },
    },
    "long": {
        "timesteps": {
            "default": 100_000,
            "blackjack": 300_000,
            "higher_or_lower": 100_000,
            "match3": 1_000_000,
            "flappy": 600_000,
            "connect_four": 4_000_000,
            "racing": 1_500_000,
            "racing:sac": 300_000,
            "gym_classic": 1_000_000,
        },
        "eval_freq": {
            "default": 10_000,
            "blackjack": 20_000,
            "higher_or_lower": 10_000,
            "match3": 25_000,
            "flappy": 25_000,
            "connect_four": 25_000,
            "racing": 25_000,
            "racing:sac": 10_000,
        },
        "eval_episodes": {
            "default": 500,
            "match3": 200,
            "flappy": 200,
            "connect_four": 100,
            "racing": 10,
        },
    },
}


for _preset, _steps, _episodes in (("quick", 50_000, 10), ("standard", 250_000, 20), ("long", 1_000_000, 20)):
    PRESETS[_preset]["timesteps"]["pong"] = _steps
    PRESETS[_preset]["eval_freq"]["pong"] = 25_000
    PRESETS[_preset]["eval_episodes"]["pong"] = _episodes


def _preset_value(
    preset: str, field: str, game: str, algorithm: str
) -> int:
    """Resolve one preset field for a (game, algorithm) pair.

    Most specific key wins: ``"game:algorithm"``, then ``"game"``,
    then ``"default"`` (every preset table defines one).
    """
    table = PRESETS[preset][field]
    for key in (f"{game}:{algorithm}", game, "default"):
        if key in table:
            return table[key]
    raise KeyError(f"preset {preset!r} field {field!r} has no default")


#: Per-(game, algorithm) defaults mirroring the shipped
#: ``configs/*.yaml`` on every load-bearing field: ``gamma``,
#: ``policy``, ``policy_kwargs``, ``env_kwargs``, ``n_envs``,
#: ``learning_rate``, and the algorithm ``hyperparams``. Absent keys
#: mean the framework defaults (MlpPolicy, no kwargs, gamma 0.99).
#:
#: This table is the cockpit's half of the cockpit/CLI parity
#: contract: ``tests/test_web_parity.py`` compares it against the
#: YAML files themselves, so editing a shipped config's load-bearing
#: field without updating this table (or vice versa) fails the
#: suite. Pairs not listed build with framework defaults, subject
#: to the mask guard at job-creation time.
PAIR_DEFAULTS: dict[tuple[str, str], dict[str, Any]] = {
    ("pong", "ppo"): {
        "gamma": 0.995,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "hyperparams": {"n_steps": 256, "batch_size": 256, "n_epochs": 4, "ent_coef": 0.01, "gae_lambda": 0.95},
        "self_play": {"snapshot_interval": 25000, "snapshot_share": .7,
                      "random_share": .1, "heuristic_share": .2, "heuristic": "pong_intercept"},
    },
    ("blackjack", "maskable_ppo"): {
        "gamma": 0.99,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "hyperparams": {
            "n_steps": 1024,
            "batch_size": 256,
            "n_epochs": 10,
            "ent_coef": 0.005,
            "gae_lambda": 0.95,
        },
    },
    ("blackjack", "dqn"): {
        "gamma": 0.99,
        "n_envs": 4,
        "learning_rate": 5e-4,
        "hyperparams": {
            "buffer_size": 100_000,
            "batch_size": 128,
            "learning_starts": 1000,
            "train_freq": 4,
            "target_update_interval": 1000,
            "exploration_fraction": 0.25,
            "exploration_final_eps": 0.05,
        },
    },
    ("higher_or_lower", "maskable_ppo"): {
        "gamma": 0.99,
        "n_envs": 4,
        "learning_rate": 1e-3,
        "hyperparams": {
            "n_steps": 512,
            "batch_size": 256,
            "n_epochs": 4,
            "ent_coef": 0.0,
            "gae_lambda": 0.95,
        },
    },
    ("match3", "maskable_ppo"): {
        "gamma": 0.99,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "policy": "MultiInputPolicy",
        "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        "env_kwargs": {
            "board_rows": 8,
            "board_cols": 8,
            "num_gems": 6,
            "max_moves": 20,
        },
        "hyperparams": {
            "n_steps": 256,
            "batch_size": 512,
            "n_epochs": 4,
            "ent_coef": 0.01,
            "gae_lambda": 0.95,
        },
    },
    ("flappy", "ppo"): {
        "gamma": 0.99,
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hyperparams": {
            "n_steps": 256,
            "batch_size": 512,
            "n_epochs": 4,
            "ent_coef": 0.01,
            "gae_lambda": 0.95,
        },
    },
    ("connect_four", "maskable_ppo"): {
        "gamma": 0.99,
        "policy": "MlpPolicy",
        "policy_kwargs": {"net_arch": [256, 256]},
        "n_envs": 8,
        "learning_rate": 3e-4,
        "hyperparams": {
            "n_steps": 256,
            "batch_size": 512,
            "n_epochs": 4,
            "ent_coef": 0.02,
            "gae_lambda": 0.95,
        },
        # The league recipe (training/self_play.py), run-3 values
        # (see the YAML's header for the run-1/run-2 post-mortems).
        # Parity-tested like every other load-bearing field: a
        # cockpit self-play run with a different mixture or snapshot
        # cadence would be a different experiment than the shipped one.
        "self_play": {
            "snapshot_interval": 10000,
            "snapshot_share": 0.55,
            "random_share": 0.15,
            "heuristic_share": 0.2,
            "heuristic": "connect_four_greedy_noisy",
            "search_share": 0.1,
            "search": "connect_four_minimax",
            "pool_max": 24,
            "recency_exponent": 1.0,
            "uniform_floor": 0.25,
        },
    },
    ("racing", "ppo"): {
        # gamma 0.995 is the braking horizon (DESIGN.md, M5): at
        # 0.99 the car never learns to brake. Do not "simplify" this
        # back to the framework default — the parity test will fail,
        # which is the point.
        "gamma": 0.995,
        "n_envs": 8,
        "learning_rate": 3e-4,
        # log_std_init -1.0 is the M8 collapse fix (DESIGN.md):
        # at SB3's default std 1.0 the sampled policy PPO trains
        # is a crash machine and the deterministic artifact rots
        # after a transient peak; 0.37 keeps the two coupled.
        "policy_kwargs": {"net_arch": [256, 256], "log_std_init": -1.0},
        "env_kwargs": {"track_seed": 7},
        "hyperparams": {
            "n_steps": 256,
            "batch_size": 1024,
            "n_epochs": 4,
            "ent_coef": 0.005,
            "gae_lambda": 0.98,
        },
    },
    ("racing", "sac"): {
        "gamma": 0.995,  # braking horizon, same as the PPO run
        "n_envs": 1,  # SAC is update-bound on CPU; envs add nothing
        "learning_rate": 3e-4,
        "policy_kwargs": {"net_arch": [256, 256]},
        "env_kwargs": {"track_seed": 7},
        "hyperparams": {
            "buffer_size": 300_000,
            "batch_size": 256,
            "tau": 0.005,
            "ent_coef": "auto",
            "learning_starts": 5000,
            "train_freq": 1,
            "gradient_steps": 1,
        },
    },
}

#: Named **variants** of a (game, algorithm) pair: shipped configs
#: that deliberately differ from the pair's canonical run on some
#: load-bearing fields, keyed by the shipped YAML's filename stem.
#: M7's generalist racing configs are the first: same algorithm,
#: same recipe, but the env deals a fresh generator-v2 circuit per
#: episode (``randomize_track`` over the training seed range) instead
#: of the canonical seed-7 loop — a different experiment that happens
#: to share its pair. The parity suite routes a YAML whose stem is
#: a variant key through ``build_config(..., variant=stem)`` and a
#: YAML with no variant through the bare pair, so every shipped
#: config still has exactly one cockpit description that must match
#: it field-for-field. Variant fields *replace* pair fields.
PAIR_VARIANTS: dict[str, dict[str, Any]] = {
    "racing_sac_generalist": {
        "game": "racing",
        "algorithm": "sac",
        "env_kwargs": {
            "track_version": 2,
            "randomize_track": True,
            "track_seed_lo": 10_000,
            "track_seed_hi": 11_000,
        },
    },
    "racing_ppo_generalist": {
        "game": "racing",
        "algorithm": "ppo",
        "env_kwargs": {
            "track_version": 2,
            "randomize_track": True,
            "track_seed_lo": 10_000,
            "track_seed_hi": 11_000,
        },
    },
    "racing_sac_generalist_v3": {
        "game": "racing",
        "algorithm": "sac",
        "env_kwargs": {
            "track_version": 3,
            "randomize_track": True,
            "track_seed_lo": 70_000,
            "track_seed_hi": 71_000,
        },
    },
}

#: Algorithms the Train view offers per game, derived from
#: :data:`PAIR_DEFAULTS` (i.e. exactly the pairs with a shipped
#: config). Order is presentation order.
GAME_ALGORITHMS: dict[str, list[str]] = {}
for _game, _algorithm in PAIR_DEFAULTS:
    GAME_ALGORITHMS.setdefault(_game, []).append(_algorithm)

#: The algorithm preselected per game in the Train view. Defaults to
#: the first of :data:`GAME_ALGORITHMS`; racing overrides to SAC —
#: the recommended racing driver (its final model is statistically
#: its best model; PPO's slightly faster peak survives only via
#: best-by-eval checkpoint selection — see DESIGN.md, M5).
DEFAULT_ALGORITHM: dict[str, str] = {"racing": "sac"}


def default_algorithm(game: str) -> str:
    """The preselected algorithm for ``game`` in the Train view."""
    algorithms = GAME_ALGORITHMS[game]
    return DEFAULT_ALGORITHM.get(game, algorithms[0])


def pair_defaults_payload() -> dict[str, dict[str, Any]]:
    """JSON-friendly view of :data:`PAIR_DEFAULTS` for ``/api/meta``.

    Keyed ``"game:algorithm"``; the Train view renders these as its
    effective-config readout, so what the cockpit will actually run
    is visible before a job starts — no silent divergence.
    """
    payload: dict[str, dict[str, Any]] = {}
    for (game, algorithm), defaults in PAIR_DEFAULTS.items():
        payload[f"{game}:{algorithm}"] = {
            "gamma": defaults.get("gamma", 0.99),
            "policy": defaults.get("policy", "MlpPolicy"),
            "policy_kwargs": deepcopy(defaults.get("policy_kwargs", {})),
            "env_kwargs": deepcopy(defaults.get("env_kwargs", {})),
            "n_envs": defaults.get("n_envs", 4),
            "learning_rate": defaults.get("learning_rate", 3e-4),
            "hyperparams": deepcopy(defaults.get("hyperparams", {})),
            "self_play": deepcopy(defaults.get("self_play", {})),
        }
    return payload


def pair_variants_payload() -> dict[str, dict[str, Any]]:
    """JSON-friendly view of :data:`PAIR_VARIANTS` for ``/api/meta``:
    the named shipped variants, keyed by config stem, each with the
    pair it modifies and the effective env_kwargs a job built with
    that variant would carry."""
    payload: dict[str, dict[str, Any]] = {}
    for stem, spec in PAIR_VARIANTS.items():
        built = build_config(
            game=spec["game"], algorithm=spec["algorithm"], variant=stem
        )
        payload[stem] = {
            "game": spec["game"],
            "algorithm": spec["algorithm"],
            "env_kwargs": deepcopy(built.env_kwargs),
            "gamma": built.gamma,
            "n_envs": built.n_envs,
        }
    return payload


def build_config(
    *,
    game: str,
    algorithm: str = "maskable_ppo",
    preset: str = "standard",
    timesteps: int | None = None,
    learning_rate: float | None = None,
    seed: int = 0,
    output_dir: str = "runs",
    n_envs: int | None = None,
    eval_freq: int | None = None,
    eval_episodes: int | None = None,
    variant: str | None = None,
) -> ExperimentConfig:
    """Map a web train request onto an :class:`ExperimentConfig`.

    Explicit fields win; below them, the per-pair defaults table
    (:data:`PAIR_DEFAULTS`, mirroring the shipped YAML) supplies the
    load-bearing fields — gamma, policy, policy_kwargs, env_kwargs,
    n_envs, learning rate, hyperparams — and the preset supplies the
    budget and eval cadence. A ``variant`` (see :data:`PAIR_VARIANTS`)
    replaces individual pair fields with a named shipped variant's
    (e.g. M7's generalist env_kwargs) before any of that. Config
    validation (unknown game or algorithm, mask guard inputs,
    shadowing hyperparams) happens in ``ExperimentConfig`` itself.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}; available: {sorted(PRESETS)}"
        )
    pair = PAIR_DEFAULTS.get((game, algorithm), {})
    if variant is not None:
        if variant not in PAIR_VARIANTS:
            raise ValueError(
                f"Unknown config variant {variant!r}; "
                f"available: {sorted(PAIR_VARIANTS)}"
            )
        spec = PAIR_VARIANTS[variant]
        if (spec["game"], spec["algorithm"]) != (game, algorithm):
            raise ValueError(
                f"Variant {variant!r} is the "
                f"({spec['game']!r}, {spec['algorithm']!r}) config, not "
                f"({game!r}, {algorithm!r})"
            )
        pair = {
            **pair,
            **{
                key: value
                for key, value in spec.items()
                if key not in ("game", "algorithm")
            },
        }
    return ExperimentConfig(
        game=game,
        algorithm=algorithm,
        total_timesteps=(
            timesteps
            if timesteps is not None
            else _preset_value(preset, "timesteps", game, algorithm)
        ),
        seed=seed,
        n_envs=(
            n_envs if n_envs is not None else pair.get("n_envs", 4)
        ),
        learning_rate=(
            learning_rate
            if learning_rate is not None
            else pair.get("learning_rate", 3e-4)
        ),
        gamma=pair.get("gamma", 0.99),
        env_kwargs=dict(pair.get("env_kwargs", {})),
        policy=pair.get("policy", "MlpPolicy"),
        policy_kwargs=deepcopy(pair.get("policy_kwargs", {})),
        eval_freq=(
            eval_freq
            if eval_freq is not None
            else _preset_value(preset, "eval_freq", game, algorithm)
        ),
        eval_episodes=(
            eval_episodes
            if eval_episodes is not None
            else _preset_value(preset, "eval_episodes", game, algorithm)
        ),
        output_dir=output_dir,
        hyperparams=deepcopy(pair.get("hyperparams", {})),
        self_play=deepcopy(pair.get("self_play", {})),
    )


def run_dir_for(config: ExperimentConfig) -> Path:
    """The directory ``train()`` will write this config's run into.

    Same naming rule as the trainer (game_algorithm_s<seed>_<hash>);
    duplicated here only so a job can report its run dir up front.
    """
    return (
        Path(config.output_dir)
        / f"{config.game}_{config.algorithm}_s{config.seed}_{config_hash(config)}"
    )


# ---------------------------------------------------------------------------
# Progress / stop callback
# ---------------------------------------------------------------------------


class ProgressCallback(BaseCallback):
    """Feeds a :class:`TrainJob`'s live state; also the stop switch."""

    def __init__(self, job: "TrainJob", run_dir: Path) -> None:
        super().__init__()
        self._job = job
        self._npz = run_dir / "eval_logs" / "evaluations.npz"
        self._npz_stamp: tuple[int, int] | None = None

    def _on_step(self) -> bool:
        job = self._job
        job.timesteps_done = self.num_timesteps
        for info in self.locals["infos"]:
            if "episode" in info:
                ep = info["episode"]
                job.record_episode(self.num_timesteps, float(ep["r"]))
        self._poll_evals()
        return not job.stop_requested

    def _poll_evals(self) -> None:
        """Refresh eval history from EvalCallback's npz when it changes."""
        try:
            stat = self._npz.stat()
        except OSError:
            return
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp == self._npz_stamp:
            return
        self._npz_stamp = stamp
        try:
            data = np.load(self._npz)
            means = np.mean(data["results"], axis=1)
            timesteps = data["timesteps"]
        except Exception:
            return  # partially written file; next poll retries
        history = [
            {"timesteps": int(t), "mean_return": float(m)}
            for t, m in zip(timesteps, means)
        ]
        self._job.record_evals(history)


# ---------------------------------------------------------------------------
# Job record + manager
# ---------------------------------------------------------------------------

_MAX_RECENT_RETURNS = 600


@dataclass
class TrainJob:
    job_id: str
    config: ExperimentConfig
    run_dir: Path
    status: str = "queued"  # queued | running | completed | failed
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    stopped: bool = False
    timesteps_done: int = 0
    episodes_completed: int = 0
    latest_eval_mean: float | None = None
    best_eval_mean: float | None = None
    eval_history: list[dict] = field(default_factory=list)
    summary: dict | None = None
    _recent_returns: deque = field(
        default_factory=lambda: deque(maxlen=_MAX_RECENT_RETURNS)
    )
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        self._stop_event.set()

    def record_episode(self, timesteps: int, ep_return: float) -> None:
        with self._lock:
            self.episodes_completed += 1
            self._recent_returns.append(
                {"timesteps": timesteps, "episode_return": ep_return}
            )

    def record_evals(self, history: list[dict]) -> None:
        with self._lock:
            self.eval_history = history
            if history:
                self.latest_eval_mean = history[-1]["mean_return"]
                self.best_eval_mean = max(h["mean_return"] for h in history)

    def snapshot(self) -> dict:
        with self._lock:
            recent = list(self._recent_returns)
            evals = list(self.eval_history)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stopped": self.stopped,
            "game": self.config.game,
            "algorithm": self.config.algorithm,
            "config": self.config.to_dict(),
            "run_id": self.run_dir.name,
            "run_dir": str(self.run_dir),
            "timesteps_done": self.timesteps_done,
            "total_timesteps": self.config.total_timesteps,
            "episodes_completed": self.episodes_completed,
            "recent_returns": recent,
            "eval_history": evals,
            "latest_eval_mean": self.latest_eval_mean,
            "best_eval_mean": self.best_eval_mean,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "summary": self.summary,
        }


class JobManager:
    """Owns training jobs and their worker threads."""

    def __init__(self) -> None:
        self._jobs: dict[str, TrainJob] = {}
        self._lock = threading.Lock()

    def create(self, config: ExperimentConfig) -> TrainJob:
        job = TrainJob(
            job_id=uuid.uuid4().hex[:12],
            config=config,
            run_dir=run_dir_for(config),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        thread = threading.Thread(
            target=self._run, args=(job,), daemon=True,
            name=f"train-{job.job_id}",
        )
        thread.start()
        return job

    def _run(self, job: TrainJob) -> None:
        job.status = "running"
        job.started_at = time.time()
        try:
            summary = train(
                job.config,
                extra_callbacks=[ProgressCallback(job, job.run_dir)],
            )
            job.summary = summary
            if job.stop_requested:
                job.stopped = True
            if summary.get("best_eval_mean_return") is not None:
                job.best_eval_mean = float(summary["best_eval_mean_return"])
                job.latest_eval_mean = job.best_eval_mean
            job.status = "completed"
        except Exception as exc:  # surfaced to the UI, not swallowed
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> TrainJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def stop(self, job_id: str) -> TrainJob | None:
        job = self.get(job_id)
        if job is not None and job.status == "running":
            job.request_stop()
        return job
