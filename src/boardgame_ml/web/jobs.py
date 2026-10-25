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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from boardgame_ml.training.config import ExperimentConfig
from boardgame_ml.training.train import config_hash, train

# ---------------------------------------------------------------------------
# Presets and defaults (mirroring configs/*.yaml)
# ---------------------------------------------------------------------------

#: UI presets: total timesteps per game, plus eval cadence. "standard"
#: tracks the shipped YAML configs; quick/long scale around them.
PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "timesteps": {
            "blackjack": 20_000,
            "higher_or_lower": 10_000,
            "flappy": 100_000,
            "racing": 250_000,
        },
        "eval_freq": 5_000,
        "eval_episodes": 200,
    },
    "standard": {
        "timesteps": {
            "blackjack": 100_000,
            "higher_or_lower": 60_000,
            "flappy": 300_000,
            "racing": 750_000,
        },
        "eval_freq": 10_000,
        "eval_episodes": 500,
    },
    "long": {
        "timesteps": {
            "blackjack": 300_000,
            "higher_or_lower": 100_000,
            "flappy": 600_000,
            "racing": 1_500_000,
        },
        "eval_freq": 20_000,
        "eval_episodes": 500,
    },
}

_DEFAULT_TIMESTEPS = 50_000

#: Hyperparameters from the shipped configs, keyed by (game, algorithm).
#: Pairs not listed use SB3 defaults.
DEFAULT_HYPERPARAMS: dict[tuple[str, str], dict[str, Any]] = {
    ("blackjack", "maskable_ppo"): {
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "ent_coef": 0.005,
        "gae_lambda": 0.95,
    },
    ("blackjack", "dqn"): {
        "buffer_size": 100_000,
        "batch_size": 128,
        "learning_starts": 1000,
        "train_freq": 4,
        "target_update_interval": 1000,
        "exploration_fraction": 0.25,
        "exploration_final_eps": 0.05,
    },
    ("higher_or_lower", "maskable_ppo"): {
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 4,
        "ent_coef": 0.0,
        "gae_lambda": 0.95,
    },
    ("flappy", "ppo"): {
        "n_steps": 256,
        "batch_size": 512,
        "n_epochs": 4,
        "ent_coef": 0.01,
        "gae_lambda": 0.95,
    },
    ("racing", "ppo"): {
        "n_steps": 256,
        "batch_size": 1024,
        "n_epochs": 4,
        "ent_coef": 0.0,
        "gae_lambda": 0.95,
    },
}

DEFAULT_LEARNING_RATES: dict[tuple[str, str], float] = {
    ("blackjack", "maskable_ppo"): 3e-4,
    ("blackjack", "dqn"): 5e-4,
    ("higher_or_lower", "maskable_ppo"): 1e-3,
    ("flappy", "ppo"): 3e-4,
    ("racing", "ppo"): 3e-4,
}


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
) -> ExperimentConfig:
    """Map a web train request onto an :class:`ExperimentConfig`.

    Explicit fields win over the preset; the preset fills the rest.
    Config validation (unknown game/algorithm, mask guard inputs,
    shadowing hyperparams) happens in ``ExperimentConfig`` itself.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}; available: {sorted(PRESETS)}"
        )
    p = PRESETS[preset]
    key = (game, algorithm)
    return ExperimentConfig(
        game=game,
        algorithm=algorithm,
        total_timesteps=(
            timesteps
            if timesteps is not None
            else p["timesteps"].get(game, _DEFAULT_TIMESTEPS)
        ),
        seed=seed,
        n_envs=n_envs if n_envs is not None else 4,
        learning_rate=(
            learning_rate
            if learning_rate is not None
            else DEFAULT_LEARNING_RATES.get(key, 3e-4)
        ),
        eval_freq=eval_freq if eval_freq is not None else p["eval_freq"],
        eval_episodes=(
            eval_episodes if eval_episodes is not None else p["eval_episodes"]
        ),
        output_dir=output_dir,
        hyperparams=dict(DEFAULT_HYPERPARAMS.get(key, {})),
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
