"""Match 3 — the framework's third game, and its final exam.

The legacy project's Match 3 was unlearnable for three reasons that
had nothing to do with the game's rules: gem types were fed to an MLP
as scalar IDs (a fake ordering with no spatial structure), PPO dropped
the action mask between sampling and updating, and legal-move
computation brute-force re-simulated the whole board constantly. The
rules themselves — match detection, cascades, gravity, refill — were
verified correct in the autopsy, so this module reimplements *those
rules* cleanly and changes everything around them:

* **Observation**: multi-channel one-hot planes — a ``(num_gems, rows,
  cols)`` float tensor in a Dict space, plus ``moves_left``. No scalar
  gem IDs anywhere: gem types are categories, and the tensor says so.
* **Legality**: :meth:`Match3Env.action_masks` returns the true legal
  set for the current board, computed once per state and cached until
  the state changes (the trainer's MaskablePPO consumes it through
  the usual ``ActionMasker`` path — masks can no longer be dropped).
* **Rewards**: the game's intrinsic score, and nothing else (below).

Rules:

* Board of ``board_rows`` x ``board_cols`` cells (default 8x8) with
  ``num_gems`` gem types (default 6). A move swaps a cell with an
  orthogonal neighbour; the swap is legal only if it creates at least
  one run of 3+ equal gems. Illegal swaps are **rejected**: no swap,
  no points, and the move is not consumed (the mask exists precisely
  so a masked agent never attempts one; rejection is the contract
  for unmasked callers such as ``check_env``'s random probes).
* Matches clear, gems above fall (gravity), empty cells refill with
  new random gems, and any matches the fall creates resolve as the
  next **cascade level**, automatically, within the same move.
* An episode lasts ``max_moves`` moves (default 20). The limit is a
  time limit, so reaching it ends the episode by **truncation**.
* The board is guaranteed playable: generation produces a board with
  no pre-existing matches and at least one legal move, and if a
  post-resolution board has no legal moves it is **reshuffled** (the
  board's gem multiset is permuted until it is match-free with a
  legal move; in the pathological case that never accepts, a fresh
  board is generated instead — documented in :meth:`_reshuffle`).

Scoring (the game's own, per move — this is the reward, undivided,
with no move penalty, no shaping, and no terminal bonus):

* Each cascade level groups its cleared gems into connected
  components (same-type, orthogonally connected runs merged where
  they cross). A component of size 3 / 4 / 5 / 6+ scores
  **10 / 25 / 50 / 100** points.
* The level's points are multiplied by the cascade level
  (first match x1, first cascade x2, and so on).
* The move's reward is the sum over its cascade levels.

Actions: ``Discrete(rows * cols * 4)``; action = ``cell * 4 + dir``
with ``cell = row * cols + col`` and directions 0=up, 1=right,
2=down, 3=left.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from boardgame_ml.envs.base import BaseGameEnv

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

#: Points for one matched component, by its size (6+ all score 100).
COMPONENT_SCORES = {3: 10, 4: 25, 5: 50}
COMPONENT_SCORE_LARGE = 100

DEFAULT_ROWS, DEFAULT_COLS, DEFAULT_GEMS, DEFAULT_MOVES = 8, 8, 6, 20


def component_score(size: int) -> int:
    """Points for one matched component of ``size`` gems."""
    if size >= 6:
        return COMPONENT_SCORE_LARGE
    return COMPONENT_SCORES.get(size, 0)


# ---------------------------------------------------------------------------
# Engine: pure functions on a board array (values in [0, num_gems)).
# The env, the greedy baseline, and the tests all share these — there is
# exactly one implementation of the game's rules.
# ---------------------------------------------------------------------------


def find_matches(board: np.ndarray) -> np.ndarray:
    """Boolean mask of every cell in a horizontal/vertical run of 3+."""
    matched = np.zeros(board.shape, dtype=bool)
    if board.shape[1] >= 3:
        run = (board[:, :-2] == board[:, 1:-1]) & (board[:, 1:-1] == board[:, 2:])
        matched[:, :-2] |= run
        matched[:, 1:-1] |= run
        matched[:, 2:] |= run
    if board.shape[0] >= 3:
        run = (board[:-2, :] == board[1:-1, :]) & (board[1:-1, :] == board[2:, :])
        matched[:-2, :] |= run
        matched[1:-1, :] |= run
        matched[2:, :] |= run
    return matched


def _runs_from_mask(board: np.ndarray, matched: np.ndarray) -> list:
    """Maximal runs (gem, cells) inside a matched mask.

    Every matched cell belongs to at least one run of length >= 3 by
    construction of :func:`find_matches`; runs are how components are
    counted without double-counting a run's cells. Segments are split
    by gem value: two different gems' runs can sit contiguously in
    one row/column of the mask (a 5-run directly above a 4-run), and
    treating the mask segment as one run would merge them into a
    phantom component.
    """
    rows, cols = board.shape
    runs: list = []
    for r in range(rows):
        c = 0
        while c < cols:
            if matched[r, c]:
                gem = int(board[r, c])
                start = c
                while c < cols and matched[r, c] and int(board[r, c]) == gem:
                    c += 1
                if c - start >= 3:
                    runs.append((gem, [(r, cc) for cc in range(start, c)]))
            else:
                c += 1
    for c in range(cols):
        r = 0
        while r < rows:
            if matched[r, c]:
                gem = int(board[r, c])
                start = r
                while r < rows and matched[r, c] and int(board[r, c]) == gem:
                    r += 1
                if r - start >= 3:
                    runs.append((gem, [(rr, c) for rr in range(start, r)]))
            else:
                r += 1
    return runs


def score_runs(runs: list) -> int:
    """Score a set of runs as connected components.

    Runs that share a cell (an L/T/cross match) are one component;
    runs sharing a cell always have the same gem (it is the cell's
    gem), so union by shared cells is exactly same-type orthogonal
    connectivity of the matched cells.
    """
    parent = list(range(len(runs)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict = {}
    for i, (_gem, cells) in enumerate(runs):
        for cell in cells:
            j = owner.setdefault(cell, i)
            if j != i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    components: dict = {}
    for i, (_gem, cells) in enumerate(runs):
        components.setdefault(find(i), set()).update(cells)
    return sum(component_score(len(cells)) for cells in components.values())


def score_matches(board: np.ndarray, matched: np.ndarray) -> int:
    """Points for a full-board matched mask (one cascade level)."""
    return score_runs(_runs_from_mask(board, matched))


def swap_first_cascade_score(
    board: np.ndarray, r1: int, c1: int, r2: int, c2: int
) -> int:
    """Points the swap (r1,c1)<->(r2,c2) scores at cascade level 1.

    Only runs through the two swapped cells can be new (the board is
    assumed match-free before the swap), so this inspects just those
    lines instead of rescanning the board — the greedy baseline's
    1-ply simulation. Does not mutate ``board``.
    """
    rows, cols = board.shape
    bl = board.tolist()
    bl[r1][c1], bl[r2][c2] = bl[r2][c2], bl[r1][c1]

    def run_cells(r: int, c: int, dr: int, dc: int):
        v = bl[r][c]
        cells = [(r, c)]
        rr, cc = r + dr, c + dc
        while 0 <= rr < rows and 0 <= cc < cols and bl[rr][cc] == v:
            cells.append((rr, cc))
            rr += dr
            cc += dc
        rr, cc = r - dr, c - dc
        while 0 <= rr < rows and 0 <= cc < cols and bl[rr][cc] == v:
            cells.append((rr, cc))
            rr -= dr
            cc -= dc
        return v, cells

    runs: list = []
    seen: set = set()
    for r, c in ((r1, c1), (r2, c2)):
        for dr, dc in ((0, 1), (1, 0)):
            gem, cells = run_cells(r, c, dr, dc)
            if len(cells) >= 3:
                key = frozenset(cells)
                if key not in seen:
                    seen.add(key)
                    runs.append((gem, cells))
    return score_runs(runs)


def resolve_board(
    board: np.ndarray, rng: np.random.Generator, num_gems: int
) -> tuple:
    """Resolve all matches on ``board`` (mutated in place).

    Clears matches, applies gravity per column, refills from ``rng``,
    and repeats while new matches form. Returns ``(points, levels)``:
    the intrinsic score (component table x cascade level) and how
    many cascade levels resolved.
    """
    rows, _cols = board.shape
    total = 0
    level = 0
    while True:
        matched = find_matches(board)
        if not matched.any():
            return total, level
        level += 1
        total += score_matches(board, matched) * level
        for c in range(board.shape[1]):
            kept = board[~matched[:, c], c]
            n_new = rows - kept.size
            new_col = np.empty(rows, dtype=board.dtype)
            if n_new:
                new_col[:n_new] = rng.integers(0, num_gems, size=n_new)
            new_col[n_new:] = kept
            board[:, c] = new_col


def _line_at_least_3(bl: list, rows: int, cols: int, r: int, c: int) -> bool:
    """True if the gem at (r, c) is in a run of 3+ on a list board."""
    v = bl[r][c]
    n = 1
    cc = c - 1
    while cc >= 0 and bl[r][cc] == v:
        n += 1
        cc -= 1
    cc = c + 1
    while cc < cols and bl[r][cc] == v:
        n += 1
        cc += 1
    if n >= 3:
        return True
    n = 1
    rr = r - 1
    while rr >= 0 and bl[rr][c] == v:
        n += 1
        rr -= 1
    rr = r + 1
    while rr < rows and bl[rr][c] == v:
        n += 1
        rr += 1
    return n >= 3


def legal_actions_mask(board: np.ndarray) -> np.ndarray:
    """The true legal set for a **match-free** board, as a bool mask.

    A swap is legal iff it creates a run of 3+; on a match-free board
    any run a swap creates must pass through one of the two swapped
    cells, so legality is checked locally per adjacent pair (112
    pairs on 8x8 — not a full re-simulation per candidate, the legacy
    engine's habit). Each unordered pair is tested once and marks
    both directed actions.
    """
    rows, cols = board.shape
    bl = board.tolist()
    mask = np.zeros(rows * cols * 4, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            for d in (RIGHT, DOWN):
                dr, dc = DIRS[d]
                r2, c2 = r + dr, c + dc
                if r2 >= rows or c2 >= cols:
                    continue
                if bl[r][c] == bl[r2][c2]:
                    continue  # swapping equals changes nothing
                bl[r][c], bl[r2][c2] = bl[r2][c2], bl[r][c]
                ok = _line_at_least_3(bl, rows, cols, r, c) or _line_at_least_3(
                    bl, rows, cols, r2, c2
                )
                bl[r][c], bl[r2][c2] = bl[r2][c2], bl[r][c]
                if ok:
                    mask[(r * cols + c) * 4 + d] = True
                    mask[(r2 * cols + c2) * 4 + (d + 2) % 4] = True
    return mask


def generate_board(
    rng: np.random.Generator, rows: int, cols: int, num_gems: int
) -> np.ndarray:
    """A fresh board: no pre-existing matches, >= 1 legal move.

    Cells are filled one at a time, never completing a run of 3 with
    the two cells to the left or above; if the finished board happens
    to have no legal move (rare), generation restarts on the same RNG
    stream, so a seed fully determines the board.
    """
    for _attempt in range(10_000):
        board = np.empty((rows, cols), dtype=np.int64)
        for r in range(rows):
            for c in range(cols):
                forbidden = set()
                if c >= 2 and board[r, c - 1] == board[r, c - 2]:
                    forbidden.add(int(board[r, c - 1]))
                if r >= 2 and board[r - 1, c] == board[r - 2, c]:
                    forbidden.add(int(board[r - 1, c]))
                choices = [g for g in range(num_gems) if g not in forbidden]
                board[r, c] = choices[int(rng.integers(len(choices)))]
        if legal_actions_mask(board).any():
            return board
    raise RuntimeError(
        f"Could not generate a playable {rows}x{cols} board with "
        f"{num_gems} gem types after 10000 attempts"
    )


def decode_action(action: int, cols: int) -> tuple:
    """Action index -> (row, col, direction)."""
    cell, direction = divmod(int(action), 4)
    row, col = divmod(cell, cols)
    return row, col, direction


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class Match3Env(BaseGameEnv):
    """Swap gems, match 3+, cascade. See the module docstring."""

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        board_rows: int = DEFAULT_ROWS,
        board_cols: int = DEFAULT_COLS,
        num_gems: int = DEFAULT_GEMS,
        max_moves: int = DEFAULT_MOVES,
    ) -> None:
        super().__init__(render_mode=render_mode)
        if board_rows < 3 or board_cols < 3:
            raise ValueError(
                f"Match 3 needs at least a 3x3 board, got "
                f"{board_rows}x{board_cols}"
            )
        if num_gems < 3:
            raise ValueError(f"Match 3 needs at least 3 gem types, got {num_gems}")
        if max_moves < 1:
            raise ValueError(f"max_moves must be positive, got {max_moves}")
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.num_gems = num_gems
        self.max_moves = max_moves

        self.action_space = gym.spaces.Discrete(board_rows * board_cols * 4)
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(num_gems, board_rows, board_cols),
                    dtype=np.float32,
                ),
                "moves_left": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )
        self.board = np.zeros((board_rows, board_cols), dtype=np.int64)
        self.score = 0
        self.moves_used = 0
        self._mask_cache: np.ndarray | None = None

    # -- public state -----------------------------------------------------
    @property
    def moves_left(self) -> int:
        return self.max_moves - self.moves_used

    def set_board(self, board, *, moves_used: int = 0, score: int = 0) -> None:
        """Load a specific board state (tests, tooling, state APIs).

        The board must satisfy the between-moves invariant — correct
        shape, gem values in range, and **match-free** (a state with
        unresolved matches is never a decision state). Whether it has
        a legal move is the caller's concern; a dead board stays dead
        until a move resolves (reshuffling happens after resolutions).
        """
        arr = np.asarray(board, dtype=np.int64)
        if arr.shape != (self.board_rows, self.board_cols):
            raise ValueError(
                f"Board must have shape ({self.board_rows}, {self.board_cols}), "
                f"got {arr.shape}"
            )
        if arr.size and (arr.min() < 0 or arr.max() >= self.num_gems):
            raise ValueError(
                f"Board gem values must be in [0, {self.num_gems}), "
                f"got range [{arr.min()}, {arr.max()}]"
            )
        if find_matches(arr).any():
            raise ValueError("Board has unresolved matches; not a decision state")
        self.board = arr.copy()
        self.moves_used = int(moves_used)
        self.score = int(score)
        self._mask_cache = None

    # -- helpers ------------------------------------------------------------
    def _obs(self) -> dict:
        planes = np.eye(self.num_gems, dtype=np.float32)[self.board]
        return {
            "board": np.ascontiguousarray(planes.transpose(2, 0, 1)),
            "moves_left": np.array(
                [self.moves_left / self.max_moves], dtype=np.float32
            ),
        }

    def _reshuffle(self) -> None:
        """Permute the board's gems until playable again.

        Preserves the gem multiset (standard match-3 practice). If
        1000 permutations never accept — a pathological multiset —
        fall back to a fresh generated board, which does not preserve
        the multiset; either way the result is match-free with at
        least one legal move.
        """
        flat = self.board.reshape(-1).copy()
        for _ in range(1000):
            self.np_random.shuffle(flat)
            candidate = flat.reshape(self.board.shape)
            if not find_matches(candidate).any() and legal_actions_mask(
                candidate
            ).any():
                self.board = candidate.copy()
                self._mask_cache = None
                return
        self.board = generate_board(
            self.np_random, self.board_rows, self.board_cols, self.num_gems
        )
        self._mask_cache = None

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = generate_board(
            self.np_random, self.board_rows, self.board_cols, self.num_gems
        )
        self.score = 0
        self.moves_used = 0
        self._mask_cache = None
        return self._obs(), self._info(score=0, points=0, cascades=0)

    def action_masks(self) -> np.ndarray:
        if self._mask_cache is None:
            self._mask_cache = legal_actions_mask(self.board)
        return self._mask_cache.copy()

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")
        mask = self.action_masks()
        if not mask[action]:
            # Rejected swap: nothing moves, nothing scores, and the
            # move is NOT consumed. Masked agents never attempt this;
            # the branch exists so unmasked callers (check_env probes,
            # human error) get a defined no-op instead of corruption.
            return (
                self._obs(),
                0.0,
                False,
                False,
                self._info(
                    score=self.score,
                    points=0,
                    cascades=0,
                    illegal_action=True,
                    reshuffled=False,
                ),
            )

        row, col, direction = decode_action(action, self.board_cols)
        dr, dc = DIRS[direction]
        row2, col2 = row + dr, col + dc
        self.board[row, col], self.board[row2, col2] = (
            self.board[row2, col2],
            self.board[row, col],
        )
        points, cascades = resolve_board(self.board, self.np_random, self.num_gems)
        self.score += points
        self.moves_used += 1
        self._mask_cache = None

        reshuffled = False
        new_mask = legal_actions_mask(self.board)
        if not new_mask.any():
            # Checked even on the episode-ending move: the framework
            # invariant is that every observable state has >= 1 legal
            # action (the returned info mask must be valid), and the
            # post-truncation board is never a decision state anyway.
            self._reshuffle()
            reshuffled = True
        else:
            # The post-move mask IS the new state's mask — cache it
            # instead of letting _info() compute it a second time.
            self._mask_cache = new_mask

        # The move limit is a time limit: truncation, never termination.
        truncated = self.moves_used >= self.max_moves
        info = self._info(
            score=self.score,
            points=points,
            cascades=cascades,
            illegal_action=False,
            reshuffled=reshuffled,
            last_move={
                "from": [row, col],
                "direction": direction,
                "to": [row2, col2],
                "points": points,
                "cascades": cascades,
            },
        )
        return self._obs(), float(points), False, bool(truncated), info

    def _render_text(self) -> str:
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        lines = ["   " + " ".join(str(c) for c in range(self.board_cols))]
        for r in range(self.board_rows):
            row = " ".join(chars[int(v)] for v in self.board[r])
            lines.append(f"{r}  {row}")
        lines.append(
            f"Match3 | score: {self.score} | moves left: {self.moves_left}"
        )
        return "\n".join(lines)
