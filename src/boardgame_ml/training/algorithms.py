"""The single registry of learning algorithms — and the mask guard.

Two structural rules live here:

* **One registry.** ``MODEL_CLASSES`` is the only mapping from config
  algorithm names to SB3 classes. The config validator checks against
  its keys, the trainer builds from it, and the evaluator loads
  checkpoints through it, so the three can never drift apart (an
  unknown name fails at config load with a named error, never as a
  raw ``KeyError`` deep inside a run).

* **The mask guard.** Only :data:`MASK_AWARE_ALGORITHMS` consume action
  masks during learning. Pairing a mask-blind algorithm (PPO, DQN)
  with a game that ever masks out an action would silently train an
  *unmasked* policy — legacy failure #1, re-entering through the
  config file. :func:`ensure_mask_compatible` probes the game and
  refuses the pairing unless the user explicitly opted in with
  ``allow_unmasked: true``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO, SAC

from boardgame_ml.envs import make_env
from boardgame_ml.envs.base import validate_mask

MODEL_CLASSES = {"maskable_ppo": MaskablePPO, "ppo": PPO, "dqn": DQN, "sac": SAC}

#: Algorithms whose learning update consumes action masks. Everything
#: else in the registry is mask-blind and must pass the guard below.
MASK_AWARE_ALGORITHMS = frozenset({"maskable_ppo"})


def game_uses_masks(
    game: str,
    *,
    steps: int = 50,
    seed: int = 0,
    env_kwargs: dict[str, Any] | None = None,
) -> bool:
    """Probe ``game``: does any reachable state mask out an action?

    Plays ``steps`` seeded steps choosing uniformly among *legal*
    actions (resetting as episodes end) and reports whether any
    collected ``action_masks()`` was not all-True.

    Continuous-action (``Box``) games cannot mask at all — there is no
    finite action set to exclude from — so the probe short-circuits
    to ``False`` and the guard passes every registered algorithm for
    them (the all-True route, generalized).
    """
    env = make_env(game, **(env_kwargs or {}))
    if not isinstance(env.action_space, gym.spaces.Discrete):
        return False
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    ep_seed = seed
    for _ in range(steps):
        mask = validate_mask(env.action_masks(), env.action_space.n)
        if not mask.all():
            return True
        action = int(rng.choice(np.flatnonzero(mask)))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            ep_seed += 1
            env.reset(seed=ep_seed)
    return False


def ensure_mask_compatible(
    game: str,
    algorithm: str,
    *,
    allow_unmasked: bool = False,
    env_kwargs: dict[str, Any] | None = None,
) -> None:
    """Refuse to pair a mask-blind algorithm with a game that masks.

    Raises ``ValueError`` naming the game, the algorithm, and the
    explicit override (``allow_unmasked: true``) when the pairing
    would silently learn an unmasked policy.
    """
    if algorithm in MASK_AWARE_ALGORITHMS or allow_unmasked:
        return
    if game_uses_masks(game, env_kwargs=env_kwargs):
        raise ValueError(
            f"Game {game!r} masks out actions in some states, but algorithm "
            f"{algorithm!r} ignores action masks: training or evaluating it "
            f"would silently learn an unmasked policy (legacy failure #1). "
            f"Use algorithm 'maskable_ppo', or set allow_unmasked: true "
            f"in the config to override."
        )
