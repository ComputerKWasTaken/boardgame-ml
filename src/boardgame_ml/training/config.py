"""Experiment configuration: typed, explicit, and passed by hand.

The legacy project died by a thousand config cuts — hyperparameters
were read through a global singleton from YAML paths that didn't exist,
so tuning silently did nothing. The structural fixes here:

* Config is a plain dataclass tree, constructed in one place
  (:meth:`ExperimentConfig.from_yaml`).
* It is passed **explicitly** into the trainer and from there into the
  model constructor. No globals, no singletons, no ambient state.
* ``tests/test_config.py`` contains a regression test asserting that a
  non-default ``learning_rate`` in a YAML file actually reaches the
  constructed model's optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from boardgame_ml.envs import GAMES
from boardgame_ml.training.algorithms import MODEL_CLASSES


@dataclass
class ExperimentConfig:
    """Everything needed to reproduce one training run."""

    game: str
    algorithm: str = "maskable_ppo"
    total_timesteps: int = 100_000
    seed: int = 0
    n_envs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    # Explicit opt-in to pairing a mask-blind algorithm (ppo/dqn) with
    # a game that masks out actions. Default False: the trainer and
    # evaluator refuse that pairing (see training/algorithms.py).
    allow_unmasked: bool = False
    # Passed verbatim to the game constructor at every env build site
    # (training, eval, probes), so games are parameterizable from YAML
    # (board size, move budget, player count, ...) with no code changes.
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    # SB3 policy class name and its kwargs — a board-shaped observation
    # needs CnnPolicy/MultiInputPolicy, selectable here instead of
    # requiring trainer surgery.
    policy: str = "MlpPolicy"
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    # Evaluated on a separate env with a deterministic policy; the
    # best-by-eval checkpoint is the run's product (never best-by
    # training window — see DESIGN.md).
    eval_episodes: int = 500
    eval_freq: int = 20_000  # in total environment timesteps (approx.)
    output_dir: str = "runs"
    # Algorithm-specific kwargs (n_steps, batch_size, buffer_size, ...)
    # passed verbatim to the SB3 constructor.
    hyperparams: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Fail fast, at load time, with errors that name the problem —
        # never as a raw KeyError/TypeError from inside SB3 mid-run.
        if self.algorithm not in MODEL_CLASSES:
            raise ValueError(
                f"Unknown algorithm {self.algorithm!r}; "
                f"available: {sorted(MODEL_CLASSES)}"
            )
        if self.game not in GAMES:
            raise ValueError(
                f"Unknown game {self.game!r}; available: {sorted(GAMES)}"
            )
        for name in ("total_timesteps", "n_envs", "eval_freq", "eval_episodes"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"Config field {name!r} must be positive, "
                    f"got {getattr(self, name)}"
                )
        # hyperparams are passed to the SB3 constructor *alongside* the
        # explicit top-level fields; a key naming one of them (or
        # 'verbose', also set by the trainer) would detonate as a
        # duplicate-kwarg TypeError at model construction.
        reserved = set(type(self).__dataclass_fields__) | {"verbose"}
        shadowed = sorted(set(self.hyperparams) & reserved)
        if shadowed:
            raise ValueError(
                f"hyperparams keys {shadowed} shadow top-level config "
                f"fields; set them at the top level of the config instead"
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "algorithm": self.algorithm,
            "total_timesteps": self.total_timesteps,
            "seed": self.seed,
            "n_envs": self.n_envs,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "allow_unmasked": self.allow_unmasked,
            "env_kwargs": dict(self.env_kwargs),
            "policy": self.policy,
            "policy_kwargs": dict(self.policy_kwargs),
            "eval_episodes": self.eval_episodes,
            "eval_freq": self.eval_freq,
            "output_dir": self.output_dir,
            "hyperparams": dict(self.hyperparams),
        }
