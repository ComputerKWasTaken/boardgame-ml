"""2048 — the roadmap's conclusive-answers game (item 1).

Single player, stochastic spawns, no seats and no opponent: pure
planning, so the result cannot be asterisked. The rules are the
standard ones (Gabriele Cirulli's 2048):

* 4x4 board. A move (up / right / down / left) slides every tile as
  far as it goes in that direction; two tiles of equal value that
  collide merge **once** into their double — a tile created by a
  merge cannot merge again in the same move
  (``[2,2,2,2] -> [4,4]``, ``[4,2,2] -> [4,4]``,
  ``[2,2,4] -> [4,4]``, never ``[8]``).
* A move that does not change the board is **illegal** — it is
  masked out, and (the Match 3 contract for unmasked callers)
  attempting one is a rejected no-op: no slide, no spawn, no score,
  the move is not consumed.
* After a legal move, one tile spawns in a uniformly random empty
  cell: a 2 with probability 0.9, a 4 with probability 0.1. A legal
  move always leaves at least one empty cell (a slide needs an
  empty to slide into; a merge frees one), so the spawn always has
  somewhere to go.
* The episode **terminates** when no legal move exists (a full
  board with no equal orthogonal neighbours). There is no move
  limit, so nothing here ever truncates.

Board representation: a ``(4, 4)`` int64 array of **exponents** —
0 is an empty cell, ``k`` is the tile ``2**k``. Exponents keep the
merge rule one comparison (``a == b``) and the score one shift
(``1 << (k + 1)``).

**Reward**: the game's own score gained on the move (the sum of the
values of the tiles the move created by merging). Nothing else —
no shaping, no survival bonus, no terminal bonus. The episode
return is exactly the final 2048 score, which is what every
yardstick below is measured in.

**Observation**: per-cell one-hot over exponents 0..15 — a
``(16, 4, 4)`` float tensor, channel ``e`` marking the cells whose
exponent is ``e`` (the Match 3 one-hot precedent: tile values are
categories with an exponential relationship, not scalars to feed an
MLP raw). Exponents above 15 (a 65536 tile — not reachable in
practice) are clipped into channel 15 for the observation only;
the board state keeps the true exponent.

Actions: ``Discrete(4)`` — 0=up, 1=right, 2=down, 3=left (the same
direction order as Match 3).

The engine lives here as pure functions on the board array
(:func:`slide_line`, :func:`apply_move`, :func:`legal_moves_mask`,
:func:`spawn_tile`) — the env, the greedy/expectimax baselines, and
the tests share exactly one implementation of the rules. For
search, the same ``slide_line`` also generates a 64-bit-int board
encoding with precomputed 16-bit row tables (:func:`board_to_int`,
:func:`move_int`); the tests cross-check the two engines against
each other on random boards, so the fast path cannot drift from
the rules the env plays by.

Terminal-state mask convention (the Connect Four precedent): once
the game is over, :meth:`Game2048Env.action_masks` returns all-True
— no move will ever be taken from a terminal observation, and the
framework invariant that every returned mask has at least one
legal action still holds. The true (empty) legal set of a locked
board is what :func:`legal_moves_mask` reports, and the tests use
that.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from boardgame_ml.envs.base import BaseGameEnv

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_NAMES_2048 = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}

BOARD_SIZE = 4
#: Observation channels: exponents 0..15 (0 = empty, 15 = the 32768
#: tile; larger exponents clip into channel 15 — see module docstring).
NUM_CHANNELS = 16
#: Spawn distribution: a fresh tile is a 2 (exponent 1) with this
#: probability, otherwise a 4 (exponent 2).
SPAWN_TWO_PROB = 0.9


# ---------------------------------------------------------------------------
# Engine: pure functions on an exponent board (shape (4, 4)).
# ---------------------------------------------------------------------------


def slide_line(line) -> tuple[list[int], int]:
    """Slide one length-4 line toward index 0, merging once per tile.

    ``line`` is a sequence of 4 exponents, index 0 being the edge the
    tiles move toward. Returns ``(new_line, gain)`` where ``gain`` is
    the score the merges produced (the sum of the created tiles'
    *values*). A tile created by a merge is final for this move: the
    scan consumes both parents and never revisits the product, so
    ``[1, 1, 2]`` (2,2,4) becomes ``[2, 2]`` (4,4), not ``[3]`` (8).
    """
    tiles = [int(v) for v in line if v != 0]
    out: list[int] = []
    gain = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged = tiles[i] + 1
            out.append(merged)
            gain += 1 << merged
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    out.extend([0] * (BOARD_SIZE - len(out)))
    return out, gain


def _line_cells(action: int, i: int) -> list[tuple[int, int]]:
    """The 4 cells of line ``i`` for ``action``, edge-first.

    Line ``i`` is row ``i`` for left/right moves and column ``i``
    for up/down moves; the cells are ordered from the edge the
    tiles slide toward, so :func:`slide_line` applies verbatim.
    """
    if action == LEFT:
        return [(i, c) for c in range(BOARD_SIZE)]
    if action == RIGHT:
        return [(i, c) for c in reversed(range(BOARD_SIZE))]
    if action == UP:
        return [(r, i) for r in range(BOARD_SIZE)]
    if action == DOWN:
        return [(r, i) for r in reversed(range(BOARD_SIZE))]
    raise ValueError(f"Invalid action {action}")


def apply_move(board: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
    """Apply ``action`` to ``board`` (not mutated).

    Returns ``(new_board, gain, changed)``: the board after sliding
    and merging (before any spawn), the score gained, and whether
    anything moved — a move with ``changed == False`` is illegal.
    """
    new_board = np.zeros_like(board)
    gain = 0
    for i in range(BOARD_SIZE):
        cells = _line_cells(action, i)
        line = [int(board[r, c]) for r, c in cells]
        slid, line_gain = slide_line(line)
        gain += line_gain
        for (r, c), v in zip(cells, slid):
            new_board[r, c] = v
    return new_board, gain, bool(not np.array_equal(new_board, board))


def legal_moves_mask(board: np.ndarray) -> np.ndarray:
    """Bool mask (4,): exactly the board-changing moves."""
    return np.array(
        [apply_move(board, a)[2] for a in range(4)], dtype=bool
    )


def spawn_tile(
    board: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int, int] | None]:
    """Spawn one tile on a copy of ``board``.

    Uniformly random empty cell; exponent 1 (a 2) with probability
    :data:`SPAWN_TWO_PROB`, else exponent 2 (a 4). Returns
    ``(new_board, (row, col, exponent))``, or ``(board, None)`` if
    the board has no empty cell (unreachable after a legal move —
    see the module docstring — but defined rather than assumed).
    """
    empties = np.argwhere(board == 0)
    if len(empties) == 0:
        return board, None
    r, c = (int(v) for v in empties[int(rng.integers(len(empties)))])
    exponent = 1 if rng.random() < SPAWN_TWO_PROB else 2
    new_board = board.copy()
    new_board[r, c] = exponent
    return new_board, (r, c, exponent)


def max_tile(board: np.ndarray) -> int:
    """The largest tile *value* on the board (0 for an empty board)."""
    top = int(board.max()) if board.size else 0
    return (1 << top) if top > 0 else 0


# ---------------------------------------------------------------------------
# Fast engine: the same rules on a 64-bit int (4 bits per cell).
#
# Cell (r, c) sits at bit shift 4 * (4*r + c) — row 0 is the lowest
# 16 bits, and within a row, column 0 is the lowest nibble. The row
# tables are generated from slide_line itself, so there is still
# exactly one statement of the merge rules in the repo.
# ---------------------------------------------------------------------------


def board_to_int(board: np.ndarray) -> int:
    """Encode an exponent board as a 64-bit int (see layout above)."""
    x = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            e = int(board[r, c])
            if e:
                x |= e << (4 * (4 * r + c))
    return x


def int_to_board(x: int) -> np.ndarray:
    """Decode a 64-bit int board back to the (4, 4) exponent array."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            board[r, c] = (x >> (4 * (4 * r + c))) & 0xF
    return board


_ROW_TABLES: tuple[list, list] | None = None


def _row_tables() -> tuple[list, list]:
    """(slide-left table, nibble-reverse table) over all 2**16 rows.

    Built lazily on first use (65536 rows is a fraction of a second,
    but the env never pays it — only the search baselines do).
    ``slide_left[row] = (new_row, gain)``; ``rev[row]`` is ``row``
    with its four nibbles in reverse order (for rightward slides).

    One saturation: a merge of two exponent-15 tiles would produce
    exponent 16, which a 4-bit cell cannot hold, so the product is
    stored as 15 (the gain is still the true 2**16). The array
    engine the env plays by is unbounded; the int engine saturates
    at the same exponent the observation clips to — a state with
    two adjacent 32768 tiles is not reachable in any practical game
    (the baselines' own play tops out orders of magnitude lower).
    """
    global _ROW_TABLES
    if _ROW_TABLES is not None:
        return _ROW_TABLES
    slide_left: list = [None] * 65536
    rev: list = [0] * 65536
    for row in range(65536):
        line = [(row >> (4 * c)) & 0xF for c in range(BOARD_SIZE)]
        slid, gain = slide_line(line)
        new_row = 0
        for c, v in enumerate(slid):
            new_row |= min(v, 15) << (4 * c)
        slide_left[row] = (new_row, gain)
        rev[row] = (
            ((row & 0xF) << 12)
            | ((row & 0xF0) << 4)
            | ((row & 0xF00) >> 4)
            | ((row >> 12) & 0xF)
        )
    _ROW_TABLES = (slide_left, rev)
    return _ROW_TABLES


def transpose_int(x: int) -> int:
    """Transpose the 4x4 nibble matrix (cell (r, c) <-> (c, r))."""
    out = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            e = (x >> (4 * (4 * r + c))) & 0xF
            if e:
                out |= e << (4 * (4 * c + r))
    return out


def _slide_rows_left(x: int) -> tuple[int, int]:
    slide_left, _ = _row_tables()
    out = 0
    gain = 0
    for r in range(BOARD_SIZE):
        new_row, row_gain = slide_left[(x >> (16 * r)) & 0xFFFF]
        out |= new_row << (16 * r)
        gain += row_gain
    return out, gain


def _slide_rows_right(x: int) -> tuple[int, int]:
    slide_left, rev = _row_tables()
    out = 0
    gain = 0
    for r in range(BOARD_SIZE):
        row = rev[(x >> (16 * r)) & 0xFFFF]
        new_row, row_gain = slide_left[row]
        out |= rev[new_row] << (16 * r)
        gain += row_gain
    return out, gain


def move_int(x: int, action: int) -> tuple[int, int, bool]:
    """``apply_move`` on the int encoding: (new_board, gain, changed)."""
    if action == LEFT:
        out, gain = _slide_rows_left(x)
    elif action == RIGHT:
        out, gain = _slide_rows_right(x)
    elif action == UP:
        moved, gain = _slide_rows_left(transpose_int(x))
        out = transpose_int(moved)
    elif action == DOWN:
        moved, gain = _slide_rows_right(transpose_int(x))
        out = transpose_int(moved)
    else:
        raise ValueError(f"Invalid action {action}")
    return out, gain, out != x


def legal_mask_int(x: int) -> np.ndarray:
    """Bool mask (4,): exactly the board-changing moves (int engine)."""
    return np.array([move_int(x, a)[2] for a in range(4)], dtype=bool)


def empty_cells_int(x: int) -> list[int]:
    """Bit shifts of the empty cells of an int board."""
    return [
        4 * i for i in range(16) if (x >> (4 * i)) & 0xF == 0
    ]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class Game2048Env(BaseGameEnv):
    """Slide, merge, spawn, survive. See the module docstring."""

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
        self.score = 0
        self.moves = 0
        self.game_over = False
        self._mask_cache: np.ndarray | None = None

    # -- public state -----------------------------------------------------
    @property
    def max_tile_value(self) -> int:
        return max_tile(self.board)

    def set_board(self, board, *, score: int = 0, moves: int = 0) -> None:
        """Load a specific board state (tests, tooling, state APIs).

        Any (4, 4) array of exponents 0..30 is a valid decision
        state — unlike Match 3 there is no between-moves invariant
        beyond the shape (a full locked board is a legitimate,
        terminal, state to load). ``game_over`` is recomputed.
        """
        arr = np.asarray(board, dtype=np.int64)
        if arr.shape != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"Board must have shape ({BOARD_SIZE}, {BOARD_SIZE}), "
                f"got {arr.shape}"
            )
        if arr.size and (arr.min() < 0 or arr.max() > 30):
            raise ValueError(
                f"Board exponents must be in [0, 30], got range "
                f"[{arr.min()}, {arr.max()}]"
            )
        self.board = arr.copy()
        self.score = int(score)
        self.moves = int(moves)
        self.game_over = not legal_moves_mask(self.board).any()
        self._mask_cache = None

    # -- helpers ------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        clipped = np.minimum(self.board, NUM_CHANNELS - 1)
        planes = np.eye(NUM_CHANNELS, dtype=np.float32)[clipped]
        return np.ascontiguousarray(planes.transpose(2, 0, 1))

    def _legal_mask(self) -> np.ndarray:
        if self._mask_cache is None:
            self._mask_cache = legal_moves_mask(self.board)
        return self._mask_cache

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
        self.score = 0
        self.moves = 0
        # Standard start: two tiles spawn on the empty board.
        for _ in range(2):
            self.board, _ = spawn_tile(self.board, self.np_random)
        self.game_over = False
        self._mask_cache = None
        return self._obs(), self._info(
            score=0, points=0, max_tile=self.max_tile_value,
            spawned=None, last_move=None,
        )

    def action_masks(self) -> np.ndarray:
        if self.game_over:
            # Terminal convention (see module docstring): no move
            # will be taken from here; all-True keeps the framework
            # mask invariant. The true legal set is empty and is
            # what legal_moves_mask() reports.
            return np.ones(4, dtype=bool)
        return self._legal_mask().copy()

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        if self.game_over:
            raise ValueError("The game is over; reset to play again")
        if not self._legal_mask()[action]:
            # Rejected move (the Match 3 contract): nothing slides,
            # nothing spawns, nothing scores, and the move is NOT
            # consumed. Masked agents never attempt this; the branch
            # exists so unmasked callers (check_env probes, human
            # error) get a defined no-op instead of corruption.
            return (
                self._obs(),
                0.0,
                False,
                False,
                self._info(
                    score=self.score,
                    points=0,
                    max_tile=self.max_tile_value,
                    illegal_action=True,
                    spawned=None,
                    last_move=None,
                ),
            )

        new_board, gain, _changed = apply_move(self.board, action)
        self.board, spawned = spawn_tile(new_board, self.np_random)
        assert spawned is not None  # a legal move always leaves a cell
        self.score += gain
        self.moves += 1
        self._mask_cache = None
        self.game_over = not self._legal_mask().any()
        info = self._info(
            score=self.score,
            points=gain,
            max_tile=self.max_tile_value,
            illegal_action=False,
            spawned=list(spawned),
            last_move={"action": action, "points": gain},
        )
        return self._obs(), float(gain), bool(self.game_over), False, info

    def _render_text(self) -> str:
        def cell(v: int) -> str:
            return f"{(1 << v):>6}" if v else "     ."

        lines = [
            " | ".join(cell(int(v)) for v in row) for row in self.board
        ]
        lines.append(
            f"2048 | score: {self.score} | max tile: "
            f"{self.max_tile_value} | moves: {self.moves}"
            + (" | game over" if self.game_over else "")
        )
        return "\n".join(lines)
