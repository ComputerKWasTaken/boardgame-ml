"""Shared base class for every game in the framework.

The contract (see DESIGN.md, "Adding a new game"):

Simultaneous two-player games are the exception to the single-agent API:
their raw joint-action/observation spaces are Tuples and rewards are pairs.
The protocol lives in ``envs.simultaneous``; its single-seat self-play
adapter, rather than the raw game, implements the Gym/SB3 contract below.

* A game is a :class:`gymnasium.Env`. Turn-based games use a
  ``Discrete`` action space; continuous-control games (M5's racing)
  use a ``Box`` — the contract's other rules are space-agnostic.
* ``reset(seed=...)`` must be seedable and reproducible.
* ``action_masks()`` returns a boolean array of shape ``(n_actions,)``
  marking which actions are legal *right now*. This is the sb3-contrib
  convention consumed by ``MaskablePPO`` via the ``ActionMasker`` wrapper.
  The mask is part of the environment's public API — it is never dropped,
  recomputed differently, or silently ignored between action selection
  and learning updates (that mismatch is what invalidated the legacy
  project's PPO objective). Masks exist only for ``Discrete`` spaces:
  for a ``Box`` game the mask machinery simply does not apply (every
  point of the box is a legal action by construction), and the mask
  guard passes any registered algorithm for such a game.
* ``render()`` returns a human-readable string when
  ``render_mode="human"``.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


def validate_mask(mask, n_actions: int) -> np.ndarray:
    """Coerce and validate an action mask — loudly.

    ``action_masks()`` implementations are game-author code, so every
    consumer (the ``ActionMasker`` wiring, the evaluator, the base
    class's own ``info`` dict) validates through here instead of
    trusting the return value:

    * lists/tuples and 0/1-valued numeric arrays are coerced to a bool
      ``np.ndarray`` of shape ``(n_actions,)``;
    * anything else — wrong shape, non-binary values, or an all-False
      mask (which would leave the agent with no legal action) — raises
      ``ValueError`` immediately, at the boundary where it happened.
    """
    arr = np.asarray(mask)
    if arr.shape != (n_actions,):
        raise ValueError(
            f"Action mask must have shape ({n_actions},), got shape {arr.shape}"
        )
    if arr.dtype != bool:
        if not bool(np.isin(arr, (0, 1)).all()):
            raise ValueError(
                f"Action mask values must be boolean or 0/1, "
                f"got dtype {arr.dtype} with values {np.unique(arr)!r}"
            )
        arr = arr.astype(bool)
    if not arr.any():
        raise ValueError(
            "Action mask allows no actions; at least one action must be legal"
        )
    return arr


class BaseGameEnv(gym.Env):
    """Base class all games inherit from. Subclasses must implement
    ``reset`` and ``step`` and set ``observation_space`` / ``action_space``
    in ``__init__`` — a ``gym.spaces.Discrete`` for turn-based games,
    a ``gym.spaces.Box`` for continuous-control games.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode

    # -- masking ---------------------------------------------------------
    def _require_discrete(self, what: str) -> None:
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError(
                f"{what} is only defined for Discrete action spaces; this "
                f"game's action space is {self.action_space!r}. Continuous-"
                f"action games have no action masks — every point of the "
                f"box is legal by construction."
            )

    def action_masks(self) -> np.ndarray:
        """Boolean mask of legal actions in the current state.

        Default: every action is legal. Games with state-dependent legality
        override this. Must always return shape ``(action_space.n,)`` and
        dtype ``bool``. Only defined for ``Discrete`` action spaces.
        """
        self._require_discrete("action_masks()")
        return np.ones(self.action_space.n, dtype=bool)

    def valid_actions(self) -> list[int]:
        """Indices of currently legal actions (convenience for agents)."""
        self._require_discrete("valid_actions()")
        mask = validate_mask(self.action_masks(), self.action_space.n)
        return [int(a) for a in np.flatnonzero(mask)]

    def _info(self, **extra) -> dict:
        """Standard info dict.

        For ``Discrete`` games it always carries the current (validated)
        mask; for ``Box`` games there is no mask to carry and the key is
        absent — consumers must not assume its presence for
        continuous-action games.
        """
        info: dict = {}
        if isinstance(self.action_space, gym.spaces.Discrete):
            info["action_mask"] = validate_mask(
                self.action_masks(), self.action_space.n
            )
        info.update(extra)
        return info

    # -- rendering --------------------------------------------------------
    def render(self) -> str | None:
        if self.render_mode == "human":
            return self._render_text()
        return None

    def _render_text(self) -> str:
        raise NotImplementedError
