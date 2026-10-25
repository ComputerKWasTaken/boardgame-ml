"""Connect Four — the framework's first multiplayer game (M6).

Rules (standard): two players alternate dropping a disc into one of
7 columns; it falls to the lowest empty cell of a 6-row board. Four
of your discs in a row — horizontal, vertical, or either diagonal —
wins. A full board (42 discs) with no four-in-a-row is a draw.

This game is played through the **two-player contract**
(:mod:`boardgame_ml.envs.two_player`): ``step`` is taken by the
player to move, the observation is always from the player to move's
perspective, and the step reward is the mover's result (+1 win,
0 draw / non-terminal). Player 0 (rendered ●, "red") always opens.

Observation (``Box(6, 7, 2)`` float32, binary): channel 0 marks the
discs of the **player to move**, channel 1 the opponent's. Two
channels rather than one signed plane, so ``MlpPolicy`` sees "mine"
and "theirs" as separate features without having to learn that −1
means enemy. Rows are indexed **top-first** (row 0 is the top of the
board, row 5 the floor discs land on), matching how the board is
drawn and read.

Actions: ``Discrete(7)`` — the column to drop into. The mask is
exactly the non-full columns. Stepping a full column raises
``ValueError``: unlike Match 3 (where a rejected swap keeps a
*timed* episode coherent), a game with no move budget has no state
in which "nothing happened" is meaningful — an illegal drop is a
caller bug, and masked agents never attempt one.

**Endings.** There is no truncation in this game: the board holds
at most 42 discs, so every episode terminates — by a win, or by the
42nd disc filling the board (a draw, checked after the win check so
a board-filling winning drop still wins). No time limit exists to
report as ``truncated``.

Win detection is **incremental**: only lines through the just-placed
disc can be new wins, so the check counts same-colour discs along
the four axes from that cell — O(1) per move instead of a full
board scan. Correctness (including border cases) is pinned by
tests against scripted games and a brute-force scanner.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from boardgame_ml.envs.two_player import TwoPlayerEnv

ROWS, COLS = 6, 7
CONNECT = 4
EMPTY, P0, P1 = 0, 1, 2  # board cell values (internal, absolute seats)

_AXES = ((0, 1), (1, 0), (1, 1), (1, -1))  # (drow, dcol): → ↓ ↘ ↙


def creates_win(board: np.ndarray, row: int, col: int) -> bool:
    """Does the disc at ``(row, col)`` complete a line of CONNECT?

    Incremental check used by the env on the just-placed disc, and
    by the baselines' search (which places/discards trial discs).
    ``board`` uses the absolute cell values (EMPTY/P0/P1).
    """
    value = board[row, col]
    if value == EMPTY:
        return False
    for dr, dc in _AXES:
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while (
                0 <= r < board.shape[0]
                and 0 <= c < board.shape[1]
                and board[r, c] == value
            ):
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= CONNECT:
            return True
    return False


def board_has_win(board: np.ndarray, value: int) -> bool:
    """Brute-force: does ``value`` have any line of CONNECT anywhere?

    The reference implementation the incremental check is tested
    against; also used to validate crafted draw boards in tests.
    """
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            if board[r, c] != value:
                continue
            for dr, dc in _AXES:
                end_r, end_c = r + (CONNECT - 1) * dr, c + (CONNECT - 1) * dc
                if 0 <= end_r < rows and 0 <= end_c < cols:
                    if all(
                        board[r + i * dr, c + i * dc] == value
                        for i in range(CONNECT)
                    ):
                        return True
    return False


class ConnectFourEnv(TwoPlayerEnv):
    """Alternating Connect Four. See module docstring for the contract."""

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self.action_space = gym.spaces.Discrete(COLS)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(ROWS, COLS, 2), dtype=np.float32
        )
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self._heights = np.zeros(COLS, dtype=np.int8)  # discs per column
        self._moves = 0
        self._last_move: int | None = None  # column of the most recent drop

    # -- helpers ----------------------------------------------------------
    @property
    def moves_played(self) -> int:
        return self._moves

    @property
    def last_move(self) -> int | None:
        return self._last_move

    def legal_columns(self) -> list[int]:
        return [c for c in range(COLS) if self._heights[c] < ROWS]

    def _obs(self) -> np.ndarray:
        """Perspective observation: channel 0 = player to move."""
        mine = P0 if self.current_player == 0 else P1
        theirs = P1 if self.current_player == 0 else P0
        obs = np.zeros((ROWS, COLS, 2), dtype=np.float32)
        obs[..., 0] = self.board == mine
        obs[..., 1] = self.board == theirs
        return obs

    def action_masks(self) -> np.ndarray:
        if self.winner is not None or self._moves >= ROWS * COLS:
            # Terminal state: no move will be taken from here, but
            # the framework-wide mask invariant (at least one legal
            # action -- the mask rides along in the terminal step's
            # info dict) still applies. Masked agents never act on a
            # terminal obs, so all-True is the honest convention.
            return np.ones(COLS, dtype=bool)
        return (self._heights < ROWS).astype(bool)

    # -- gym API ----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board[:] = EMPTY
        self._heights[:] = 0
        self._moves = 0
        self._last_move = None
        self.current_player = 0
        self.winner = None
        return self._obs(), self._info()

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        if self._heights[action] >= ROWS:
            raise ValueError(f"Column {action} is full")
        if self.winner is not None or self._moves >= ROWS * COLS:
            raise ValueError("The game is already over; reset to play again")

        mover = self.current_player
        value = P0 if mover == 0 else P1
        row = ROWS - 1 - int(self._heights[action])
        self.board[row, action] = value
        self._heights[action] += 1
        self._moves += 1
        self._last_move = action

        reward = 0.0
        terminated = False
        if creates_win(self.board, row, action):
            self.winner = mover
            reward = 1.0
            terminated = True
        elif self._moves >= ROWS * COLS:
            terminated = True  # full board, no win: draw (reward 0)

        # The turn passes even on the terminal move: the returned obs
        # and current_player describe the would-be next mover, per
        # the contract; consumers key game-end off `terminated` and
        # info["winner"], not off whose perspective the obs is.
        self.current_player = self._other(mover)
        return self._obs(), reward, terminated, False, self._info()

    def _render_text(self) -> str:
        glyphs = {EMPTY: " . ", P0: " ● ", P1: " ○ "}
        lines = [
            "".join(glyphs[int(v)] for v in row) for row in self.board
        ]
        lines.append(" " + "  ".join(str(c) for c in range(COLS)))
        if self.winner is not None:
            lines.append(f"Winner: player {self.winner}")
        elif self._moves >= ROWS * COLS:
            lines.append("Draw")
        else:
            lines.append(f"To move: player {self.current_player}")
        return "\n".join(lines)
