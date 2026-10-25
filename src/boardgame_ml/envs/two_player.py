"""The two-player contract — a separate base for alternating games.

Milestone 6 adds the framework's first multiplayer games. The
single-agent contract (:class:`~boardgame_ml.envs.base.BaseGameEnv`)
does not fit them: with two decision-makers, "the agent" and "the
reward" are ambiguous until you say *whose turn it is* and *from
whose seat a result is scored*. This module is the answer — a
minimal, deliberately PettingZoo-*inspired* (not PettingZoo) contract
for **alternating, two-player, zero-sum** games:

* ``step(action)`` is always taken by the **player to move**.
  Alternation is the env's business; the caller never passes a
  player id.
* The observation returned by ``reset``/``step`` is always from the
  perspective of the player to move *next* — "my discs" and "their
  discs" swap seats every step, so a single policy can play either
  seat without ever seeing an absolute player id. (After a terminal
  step there is no next player; the obs is then from the perspective
  of the player who *would* have moved, i.e. the loser of a decisive
  game — consumers key off ``info``, not the obs, at termination.)
* The reward returned by ``step`` is from the perspective of the
  player who **just moved**: ``+1.0`` when their move wins the game,
  ``0.0`` for a draw or any non-terminal move. A mover can never be
  handed ``-1.0`` by their own move — losing happens on the
  *opponent's* move, which returns ``+1.0`` *to the opponent*. The
  other seat's result for any finished game is the negation of the
  winner's, which is what a single-agent adapter (the self-play
  wrapper in :mod:`boardgame_ml.training.self_play`) uses to turn a
  two-player episode into per-seat returns.
* ``info`` always carries ``current_player`` (0 or 1 — the player to
  move next, or the would-be mover after termination) and, at
  termination, ``winner`` (0, 1, or ``None`` for a draw). During play
  ``winner`` is present and ``None``.
* ``action_masks()`` is the mask for the player to move, under the
  same sb3-contrib convention (and the same validation) as the
  single-agent contract. Two-player games here are ``Discrete``;
  continuous-action multiplayer is out of scope for this contract.

**Why not PettingZoo's AEC API?** Three measured reasons, recorded
in DESIGN.md: (1) no new dependency — the framework's whole stack is
gymnasium + SB3, and this contract is ~100 lines; (2) AEC's
per-agent ``last()``/accumulated-reward bookkeeping answers a more
general question (N players, simultaneous moves, agents entering
and leaving) than alternating two-player games pose, and its
reward convention (rewards accrue to the agent who is *about to
act*) is exactly the seat confusion this contract exists to avoid;
(3) SB3 trains single-agent envs, so multiplayer training here goes
through an adapter either way — the adapter is simpler against this
contract than against AEC. The shape is AEC-*compatible in spirit*:
``current_player`` plays the role of AEC's ``agent_selection``, and
a future AEC adapter can wrap a :class:`TwoPlayerEnv` mechanically.
"""

from __future__ import annotations

import numpy as np

from boardgame_ml.envs.base import BaseGameEnv


class TwoPlayerEnv(BaseGameEnv):
    """Base class for alternating two-player zero-sum games.

    Subclasses implement ``reset``/``step`` against the contract in
    this module's docstring, and set :attr:`NUM_PLAYERS` (fixed at 2
    for this contract). Players are indexed 0 and 1; player 0 always
    moves first after a reset.
    """

    NUM_PLAYERS = 2

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self.current_player: int = 0
        self.winner: int | None = None

    def _info(self, **extra) -> dict:
        """The single-agent info dict, plus the two-player keys.

        ``current_player`` and ``winner`` are framework keys; a game
        may not shadow them through ``extra``.
        """
        for key in ("current_player", "winner"):
            if key in extra:
                raise ValueError(
                    f"info key {key!r} is reserved by the two-player "
                    "contract; the base class sets it"
                )
        info = super()._info(**extra)
        info["current_player"] = int(self.current_player)
        info["winner"] = None if self.winner is None else int(self.winner)
        return info

    def _other(self, player: int) -> int:
        return 1 - player


def board_from_obs(obs: np.ndarray) -> np.ndarray:
    """Decode a two-channel perspective observation into a signed board.

    The perspective-observation convention for board games in this
    framework (see :class:`~boardgame_ml.envs.connect_four.ConnectFourEnv`):
    ``obs[..., 0]`` marks the discs of the player to move, ``obs[..., 1]``
    the opponent's. The signed board uses +1 for the mover's discs and
    −1 for the opponent's — the form search/eval helpers (minimax,
    greedy) compute in. Shared by the baselines so the decoding lives
    in exactly one place.
    """
    obs = np.asarray(obs)
    return (obs[..., 0] - obs[..., 1]).astype(np.int8)


__all__ = ["TwoPlayerEnv", "board_from_obs"]
