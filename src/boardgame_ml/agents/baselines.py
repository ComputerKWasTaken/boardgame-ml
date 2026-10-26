"""Non-learning reference agents.

Baselines are the framework's proof-of-learning yardsticks: a trained
agent only counts as having learned if it beats :class:`RandomAgent`
and approaches the relevant heuristic agent. Discrete-game baselines
only ever choose actions that are legal under ``env.action_masks()``;
for continuous (``Box``) games every point of the box is legal and
"random" means a uniform sample of it.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from boardgame_ml.envs.base import BaseGameEnv
from boardgame_ml.envs import blackjack as bj
from boardgame_ml.envs import connect_four as c4
from boardgame_ml.envs import flappy as fb
from boardgame_ml.envs import game2048 as g48
from boardgame_ml.envs import higher_or_lower as hol
from boardgame_ml.envs import match3 as m3
from boardgame_ml.envs import racing as rc
from boardgame_ml.envs.two_player import board_from_obs


class BaselineAgent:
    """Common interface: ``select_action(env, obs)`` — an ``int`` for
    Discrete games, an action vector for Box games."""

    name = "baseline"
    games: tuple[str, ...] = ()

    def select_action(self, env: BaseGameEnv, obs: np.ndarray):
        raise NotImplementedError


class RandomAgent(BaselineAgent):
    """Uniform random: over currently legal actions (Discrete), or a
    uniform sample of the action box (continuous games)."""

    name = "random"
    games = (
        "blackjack",
        "higher_or_lower",
        "match3",
        "flappy",
        "racing",
        "connect_four",
        "pong",
        "game2048",
    )

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, env: BaseGameEnv, obs: np.ndarray):
        if isinstance(env.action_space, gym.spaces.Box):
            return self._rng.uniform(
                env.action_space.low, env.action_space.high
            ).astype(np.float32)
        return int(self._rng.choice(env.valid_actions()))


class BasicStrategyAgent(BaselineAgent):
    """Real blackjack basic strategy, restricted to hit/stand decisions.

    The standard basic-strategy chart also uses doubling and splitting;
    neither exists in the M1 environment, so this is the hit/stand
    projection of the chart:

    * Hard totals: hit <= 11; stand on 12 vs dealer 4-6 (else hit);
      stand on 13-16 vs dealer 2-6 (else hit); stand >= 17.
    * Soft totals (usable ace): hit soft <= 17; soft 18 stands vs
      dealer 2-8, hits vs 9/10/ace; stand soft >= 19.

    Reads the hand from the environment (not the normalized obs) so the
    table is expressed in real card values.
    """

    name = "basic_strategy"
    games = ("blackjack",)

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        assert isinstance(env, bj.BlackjackEnv)
        total, usable = bj.hand_value(env.player_hand)
        dealer_up = env.dealer_hand[0]
        dealer = 11 if dealer_up == 1 else dealer_up  # ace counts high in the chart

        if usable:
            if total <= 17:
                return bj.HIT
            if total == 18:
                return bj.STAND if dealer <= 8 else bj.HIT
            return bj.STAND

        if total <= 11:
            return bj.HIT
        if total == 12:
            return bj.STAND if 4 <= dealer <= 6 else bj.HIT
        if total <= 16:
            return bj.STAND if dealer <= 6 else bj.HIT
        return bj.STAND


class ProbabilityAgent(BaselineAgent):
    """Probability-greedy agent for Higher or Lower.

    With cards drawn uniformly from 1-13 (with replacement):

    * P(next > c) = (13 - c) / 13
    * P(next < c) = (c - 1) / 13
    * P(tie)      = 1 / 13  (scores 0 either way)

    so the expected reward of a guess is ``(wins - losses) / 13 >= 0``
    for the better side, and exactly 0 for card 7. Holding also scores 0
    and ends the episode, so it is never better than guessing — this
    agent **never holds**. Ties in EV (card 7) break toward HIGHER;
    the choice is arbitrary and EV-neutral, and is documented here
    rather than hidden.
    """

    name = "probability"
    games = ("higher_or_lower",)

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        # Recover the current card from the normalized observation.
        card = int(round(float(obs[0]) * (hol.MAX_CARD - hol.MIN_CARD))) + hol.MIN_CARD
        p_higher = (hol.MAX_CARD - card) / 13.0
        p_lower = (card - hol.MIN_CARD) / 13.0
        return hol.HIGHER if p_higher >= p_lower else hol.LOWER


class GreedyMatch3Agent(BaselineAgent):
    """1-ply greedy agent for Match 3 — the learnability yardstick.

    For every legal swap, simulate it and score **only the matches
    the swap itself creates** (cascade level 1) using the game's own
    component table — :func:`boardgame_ml.envs.match3.swap_first_cascade_score`,
    which inspects just the lines through the two swapped cells.
    Cascades after the first are luck (they depend on random refills),
    so ignoring them is both the documented simplification and the
    honest one: this agent maximizes immediate, knowable score.

    Deterministic: actions are scanned in ascending index order and
    only a strictly higher score replaces the incumbent, so ties
    break toward the lowest action index.
    """

    name = "greedy"
    games = ("match3",)

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        assert isinstance(env, m3.Match3Env)
        best_action: int | None = None
        best_points = -1
        for action in env.valid_actions():
            row, col, direction = m3.decode_action(action, env.board_cols)
            dr, dc = m3.DIRS[direction]
            points = m3.swap_first_cascade_score(
                env.board, row, col, row + dr, col + dc
            )
            if points > best_points:
                best_points = points
                best_action = action
        assert best_action is not None  # env guarantees >= 1 legal move
        return best_action


class Greedy2048Agent(BaselineAgent):
    """1-ply greedy agent for 2048 — the milestone's acceptance bar.

    For every legal move, simulate the slide+merge (the env module's
    int engine — the same ``slide_line`` rules the env plays by)
    and take the move with the largest **immediate score gain**.
    Nothing else is valued: not empty cells, not tile position, not
    what the spawn might do. That narrowness is the point — this is
    the strongest agent that looks exactly one ply deep, and the
    roadmap's bar is stated against it.

    Deterministic: actions are scanned in ascending index order and
    only a strictly higher gain replaces the incumbent, so ties
    break toward the lowest action index (up < right < down < left).
    """

    name = "greedy_1ply"
    games = ("game2048",)

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        assert isinstance(env, g48.Game2048Env)
        x = g48.board_to_int(env.board)
        best_action: int | None = None
        best_gain = -1
        for action in range(4):
            _new_x, gain, changed = g48.move_int(x, action)
            if changed and gain > best_gain:
                best_gain = gain
                best_action = action
        assert best_action is not None  # decision states have a legal move
        return best_action


#: Snake path over the 4x4 board for the expectimax heuristic: the
#: largest weight sits in the top-left corner and weights decrease
#: geometrically (ratio 1/4 per step) along the snake, so the
#: heuristic prices a board by how well its big tiles are stacked
#: along one monotone path — the classic gradient evaluation.
_SNAKE_CELLS = (
    (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 3), (1, 2), (1, 1), (1, 0),
    (2, 0), (2, 1), (2, 2), (2, 3),
    (3, 3), (3, 2), (3, 1), (3, 0),
)
_SNAKE_W = np.zeros((4, 4))
for _p, (_r, _c) in enumerate(_SNAKE_CELLS):
    _SNAKE_W[_r, _c] = 0.25 ** _p

_HEUR_TABLES: list[list[float]] | None = None


def _heur_tables() -> list[list[float]]:
    """Per-row-position heuristic tables over all 2**16 rows.

    ``tables[r][row]`` is row ``r``'s total contribution to the
    expectimax heuristic: the snake-weighted sum of its tiles'
    *values*, plus the empty-cell bonus for its empty cells (the
    bonus is per-cell, so it folds into the same tables — a board's
    heuristic is then exactly 4 lookups). Built lazily, once.
    """
    global _HEUR_TABLES
    if _HEUR_TABLES is not None:
        return _HEUR_TABLES
    tables: list[list[float]] = []
    for r in range(4):
        weights = _SNAKE_W[r]
        table = [0.0] * 65536
        for row in range(65536):
            total = 0.0
            for c in range(4):
                e = (row >> (4 * c)) & 0xF
                if e:
                    total += weights[c] * float(1 << e)
                else:
                    total += Expectimax2048Agent.EMPTY_WEIGHT
            table[row] = total
        tables.append(table)
    _HEUR_TABLES = tables
    return tables


class Expectimax2048Agent(BaselineAgent):
    """Depth-limited expectimax for 2048 — the strong yardstick.

    Expectimax over the game's own chance structure: move nodes
    maximize, chance nodes average over the actual spawn
    distribution (uniform empty cell; a 2 at 0.9, a 4 at 0.1 — no
    sampling, no pruned outcomes). One frontier convention keeps
    pure-Python search affordable: at depth 0 the position the
    last move produced is read directly, without enumerating the
    spawn that follows it (that enumeration would multiply leaf
    reads by ~2 x empties for a second-order correction — a
    spawn's heuristic effect is mostly the lost empty-cell bonus,
    roughly uniform across candidate moves). Every enumerated
    chance node is exact. Node values are in the game's score
    units: a move contributes its merge gain plus the expected
    value of the resulting position; a position with no legal
    moves is worth exactly 0 future points.

    Leaves use a fixed, documented heuristic (never tuned against
    the results it judges):

    * **Positional asset value** — the snake-gradient sum above:
      each tile contributes its value weighted by 0.25**(its
      position along the snake from the top-left corner). The sum
      is dominated by the largest tiles sitting early on the snake,
      which prices "big tile cornered, board monotone" in roughly
      score units (comparable to the gains the search adds).
    * **Breathing room** — +100 per empty cell (a frozen round
      number: about the merge value one more empty cell keeps
      reachable over the horizon), folded into the same tables.

    Depth is in *move plies* and adapts to the root's empty count,
    because chance branching (an outcome per empty cell) is what
    costs: >= 4 empties -> depth 2, <= 3 empties -> depth 3 (so few
    spawn outcomes remain that the extra ply is affordable exactly
    when the position is most forcing). A deterministic node budget
    (``NODE_BUDGET`` expansions per decision) caps the worst case:
    past it, frontier move nodes return the heuristic instead of
    expanding. A transposition cache keyed by (board, depth)
    persists across decisions within a game.

    Deterministic: fixed action order, strict improvement, no RNG.
    Ties break toward the lowest action index.
    """

    name = "expectimax"
    games = ("game2048",)

    EMPTY_WEIGHT = 100.0
    NODE_BUDGET = 40_000
    _CACHE_MAX = 300_000

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int], float] = {}
        self._nodes = 0

    # -- heuristic ------------------------------------------------------
    @staticmethod
    def heuristic(x: int) -> float:
        tables = _heur_tables()
        return (
            tables[0][x & 0xFFFF]
            + tables[1][(x >> 16) & 0xFFFF]
            + tables[2][(x >> 32) & 0xFFFF]
            + tables[3][(x >> 48) & 0xFFFF]
        )

    # -- search -----------------------------------------------------------
    @staticmethod
    def _depth_for(x: int) -> int:
        empties = len(g48.empty_cells_int(x))
        return 2 if empties >= 4 else 3

    def _value(self, x: int, depth: int) -> float:
        """Expected future score from board ``x``, ``depth`` plies deep."""
        key = (x, depth)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if depth == 0:
            return self.heuristic(x)
        self._nodes += 1
        if self._nodes > self.NODE_BUDGET:
            return self.heuristic(x)
        best: float | None = None
        for action in range(4):
            new_x, gain, changed = g48.move_int(x, action)
            if not changed:
                continue
            value = gain + self._chance_value(new_x, depth - 1)
            if best is None or value > best:
                best = value
        result = 0.0 if best is None else best  # no legal move: terminal
        if len(self._cache) < self._CACHE_MAX:
            self._cache[key] = result
        return result

    def _chance_value(self, x: int, depth: int) -> float:
        """Expected ``_value`` after one random spawn on ``x``.

        Frontier convention: at ``depth == 0`` the spawn is not
        enumerated — the heuristic reads the position the move
        produced (see the class docstring). Every chance node at
        depth >= 1 averages the full spawn distribution exactly.
        """
        if depth == 0:
            return self.heuristic(x)
        cells = g48.empty_cells_int(x)
        if not cells:
            return self._value(x, depth)
        total = 0.0
        for shift in cells:
            total += 0.9 * self._value(x | (1 << shift), depth)
            total += 0.1 * self._value(x | (2 << shift), depth)
        return total / len(cells)

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        assert isinstance(env, g48.Game2048Env)
        x = g48.board_to_int(env.board)
        self._nodes = 0
        depth = self._depth_for(x)
        best_action: int | None = None
        best_value = -float("inf")
        for action in range(4):
            new_x, gain, changed = g48.move_int(x, action)
            if not changed:
                continue
            value = gain + self._chance_value(new_x, depth - 1)
            if value > best_value:
                best_value = value
                best_action = action
        assert best_action is not None  # decision states have a legal move
        return best_action


class FlappyHeuristicAgent(BaselineAgent):
    """Threshold controller for Flappy Bird — the learnability yardstick.

    The rule, in full: aim a little *below* the next gap's centre,
    and flap only when below that aim line and not already rising
    fast::

        flap  iff  bird_y < gap_centre − 10  and  bird_vy <= 80

    Why these constants (chosen from the physics, lightly tuned by
    hand, then frozen): a flap sets vy = +300 against gravity
    −1000 px/s², so from the flap the bird rises
    300² / (2·1000) = 45 px before falling back. Aiming 10 px below
    centre puts the oscillation's peak ≈ 35 px above centre —
    inside the gap's ±38 px of clearance (half the 100 px gap,
    minus the 12 px bird radius) — and its trough ≈ 27 px below.
    The vy cap keeps the controller from stacking flaps while
    already rising, which is the classic way threshold rules pin
    themselves to the ceiling. It is still a memoryless threshold
    rule: it cannot anticipate a gap sequence, and its oscillation
    wastes clearance a learned policy should not. Measured: roughly
    9–11 pipes per episode (see README's M4 table).
    """

    name = "flappy_heuristic"
    games = ("flappy",)

    #: Aim this far below the gap centre (px); flap only at or below
    #: this upward velocity (px/s). See the class docstring.
    AIM_BELOW_CENTER = 10.0
    FLAP_VY_CAP = 80.0

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        assert isinstance(env, fb.FlappyBirdEnv)
        target = env.next_pipe().gap_center - self.AIM_BELOW_CENTER
        if env.bird_y < target and env.bird_vy <= self.FLAP_VY_CAP:
            return fb.FLAP
        return fb.GLIDE


class RacingHeuristicAgent(BaselineAgent):
    """Raycast wall-follower for Racing — the learnability yardstick.

    **Observation only**: this agent reads the same ray fan, speed,
    and lateral offset the learned policy sees (plus the env's
    physical *parameters* — max speed, max steering angle, ray fan
    geometry — for denormalization; parameters are not state, and it
    never touches the centerline, the car's position, or any other
    privileged track data).

    The rule, in full:

    * **Steer toward open space.** The desired heading offset is the
      distance-squared-weighted centroid of the ray fan (long rays
      dominate, so the car aims where the track opens up), corrected
      toward the centerline proportional to the lateral offset. The
      steering input is that desired offset expressed as a fraction
      of the maximum steering angle, times a gain.
    * **Speed follows the forward view.** The target speed scales
      with the openness of the forward rays (mean of the three
      centre rays) and shrinks when the desired turn is sharp; the
      throttle is a proportional controller on (target − actual)
      speed, so overspeed produces braking on its own.

    It is a reactive controller with no memory and no lookahead
    beyond the ray range: it circulates, slowly and wide, and its
    lap times are the bar a learned policy should clear by a wide
    margin. Constants were tuned lightly by hand (an eager first
    tuning lapped in ~16 s, which made the yardstick a rival rather
    than a reference; the frozen constants lap in ~23 s — steady,
    crash-free, and clearly beatable on speed), then frozen.
    """

    name = "racing_heuristic"
    games = ("racing",)

    STEER_GAIN = 1.15
    CENTER_GAIN = 0.45  # rad of heading bias per unit normalized offset
    FRONT_SLOW = 0.12  # forward-ray openness where target speed bottoms out
    FRONT_FAST = 0.85  # ...and where it saturates
    MIN_SPEED_FRAC = 0.16
    TURN_SLOWDOWN = 0.55
    SPEED_TOLERANCE = 8.0  # m/s of speed error per unit throttle

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> np.ndarray:
        assert isinstance(env, rc.RacingEnv)
        n = env.n_rays
        rays = obs[:n].astype(np.float64)
        speed = float(obs[n]) * env.max_speed
        offset = float(obs[n + 3])  # lateral offset / half-width

        weights = rays**2
        desired = float(
            (weights * env.ray_angles).sum() / max(float(weights.sum()), 1e-9)
        )
        desired -= self.CENTER_GAIN * offset
        steer = float(
            np.clip(
                self.STEER_GAIN * desired / env.max_steer, -1.0, 1.0
            )
        )

        centre = np.abs(env.ray_angles) <= np.radians(20.0)
        front = float(rays[centre].mean())
        frac = np.clip(
            (front - self.FRONT_SLOW) / (self.FRONT_FAST - self.FRONT_SLOW),
            self.MIN_SPEED_FRAC,
            1.0,
        )
        frac *= 1.0 - self.TURN_SLOWDOWN * min(
            1.0, abs(desired) / (env.ray_fov / 2)
        )
        target_speed = frac * env.max_speed
        throttle = float(
            np.clip((target_speed - speed) / self.SPEED_TOLERANCE, -1.0, 1.0)
        )
        return np.array([steer, throttle], dtype=np.float32)


# ---------------------------------------------------------------------------
# Connect Four machinery (shared by the greedy and minimax baselines)
#
# Both agents compute on a *flat signed board*: a list of 42 ints in
# row-major order (row 0 = top), +1 for the discs of the player to
# move, -1 for the opponent's -- decoded from the perspective
# observation by board_from_obs (envs/two_player.py), the one place
# that decoding lives. Search flips the board's sign each ply so the
# side to move is always +1 (negamax convention).
# ---------------------------------------------------------------------------

_C4_CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)  # strongest columns first


def _c4_windows() -> list[tuple[int, int, int, int]]:
    """All 69 four-cell lines of the 6x7 board, as flat index tuples."""
    windows = []
    rows, cols = c4.ROWS, c4.COLS
    for r in range(rows):
        for c in range(cols):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = tuple(
                    (r + i * dr) * cols + (c + i * dc)
                    for i in range(c4.CONNECT)
                )
                if all(
                    0 <= r + i * dr < rows and 0 <= c + i * dc < cols
                    for i in range(c4.CONNECT)
                ):
                    windows.append(cells)
    return windows


_C4_WINDOWS = _c4_windows()


def _c4_heights(flat: list[int]) -> list[int]:
    cols, rows = c4.COLS, c4.ROWS
    return [
        sum(1 for r in range(rows) if flat[r * cols + col] != 0)
        for col in range(cols)
    ]


def _c4_drop_row(heights: list[int], col: int) -> int:
    """Row a disc lands on in ``col`` (top-first indexing), or -1."""
    if heights[col] >= c4.ROWS:
        return -1
    return c4.ROWS - 1 - heights[col]


def _c4_creates_win(flat: list[int], row: int, col: int) -> bool:
    """Flat-board twin of connect_four.creates_win (same semantics)."""
    value = flat[row * c4.COLS + col]
    if value == 0:
        return False
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r, cc = row + sign * dr, col + sign * dc
            while (
                0 <= r < c4.ROWS
                and 0 <= cc < c4.COLS
                and flat[r * c4.COLS + cc] == value
            ):
                count += 1
                r += sign * dr
                cc += sign * dc
        if count >= c4.CONNECT:
            return True
    return False


def _c4_winning_columns(flat: list[int], heights: list[int]) -> list[int]:
    """Columns where the side to move (+1) wins by dropping now."""
    wins = []
    for col in _C4_CENTER_ORDER:
        row = _c4_drop_row(heights, col)
        if row < 0:
            continue
        idx = row * c4.COLS + col
        flat[idx] = 1
        if _c4_creates_win(flat, row, col):
            wins.append(col)
        flat[idx] = 0
    return wins


def _c4_positional_score(flat: list[int]) -> float:
    """Window-count evaluation from the +1 (side-to-move) perspective.

    Every 4-cell window scores by its composition: an open three is
    the game's real currency (one move from four), an open two is a
    down payment; mixed windows are dead and score 0. The opponent's
    open three is weighted slightly *above* one's own -- failing to
    answer a threat loses on the spot, while making one only
    threatens to. Centre-column discs get a small bonus (the centre
    participates in the most windows).
    """
    score = 0.0
    for w in _C4_WINDOWS:
        mine = theirs = 0
        for idx in w:
            v = flat[idx]
            if v == 1:
                mine += 1
            elif v == -1:
                theirs += 1
        if mine and theirs:
            continue
        if mine == 3:
            score += 40.0
        elif mine == 2:
            score += 8.0
        elif mine == 1:
            score += 1.0
        elif theirs == 3:
            score -= 45.0
        elif theirs == 2:
            score -= 9.0
        elif theirs == 1:
            score -= 1.0
    centre = c4.COLS // 2
    for r in range(c4.ROWS):
        v = flat[r * c4.COLS + centre]
        if v == 1:
            score += 4.0
        elif v == -1:
            score -= 4.0
    return score


class ConnectFourGreedyAgent(BaselineAgent):
    """1-ply tactical greedy for Connect Four -- the self-play anchor.

    The rule, in full (columns considered centre-out: 3, 2, 4, 1, 5,
    0, 6, which is also the tie-break everywhere):

    1. **Win** if a drop completes four.
    2. **Block** if the opponent has a drop that completes four
       (flip the board's sign and ask the same question for them).
    3. **Don't blunder**: discard drops after which the opponent has
       an immediate winning drop (the classic trap: your disc gives
       them the support cell their line was missing) -- unless every
       legal drop blunders.
    4. Otherwise play the drop with the best positional score
       (:func:`_c4_positional_score` of the resulting board).

    It sees only the observation (decoded by the shared
    perspective convention); it never touches the env's internals.
    """

    name = "connect_four_greedy"
    games = ("connect_four",)

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        flat = [int(v) for v in board_from_obs(obs).reshape(-1)]
        heights = _c4_heights(flat)

        wins = _c4_winning_columns(flat, heights)
        if wins:
            return wins[0]
        flipped = [-v for v in flat]
        blocks = _c4_winning_columns(flipped, heights)
        if blocks:
            return blocks[0]

        legal = [c for c in _C4_CENTER_ORDER if heights[c] < c4.ROWS]

        def gives_opponent_a_win(col: int) -> bool:
            row = _c4_drop_row(heights, col)
            idx = row * c4.COLS + col
            flat[idx] = 1
            heights[col] += 1
            reply = [-v for v in flat]
            danger = bool(_c4_winning_columns(reply, heights))
            flat[idx] = 0
            heights[col] -= 1
            return danger

        safe = [c for c in legal if not gives_opponent_a_win(c)]
        candidates = safe or legal

        best_col, best_score = candidates[0], -float("inf")
        for col in candidates:
            row = _c4_drop_row(heights, col)
            idx = row * c4.COLS + col
            flat[idx] = 1
            score = _c4_positional_score(flat)
            flat[idx] = 0
            if score > best_score:
                best_col, best_score = col, score
        return best_col


class ConnectFourGreedyNoisyAgent(ConnectFourGreedyAgent):
    """The greedy yardstick with shaky hands: with probability
    ``epsilon`` it drops in a uniformly random legal column instead
    of the greedy one.

    This is a *league* opponent, not a tournament yardstick. It
    exists because of what run 2 of the M6 training showed: a
    learner fed a large share of a fully deterministic opponent
    memorizes counter-lines to that opponent's exact replies
    (100% vs the clean greedy) and learns little that transfers
    (0% vs minimax). The noisy twin keeps the same tactics — wins,
    blocks, blunders are still punished 80% of the time — but its
    lines cannot be memorized, so the learner has to learn the
    tactics themselves. Seeded optionally, like ``RandomAgent``.
    """

    name = "connect_four_greedy_noisy"
    games = ("connect_four",)
    epsilon = 0.2

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        if self._rng.random() < self.epsilon:
            legal = np.flatnonzero(env.action_masks())
            return int(self._rng.choice(legal))
        return super().select_action(env, obs)


class ConnectFourMinimaxAgent(BaselineAgent):
    """Alpha-beta minimax for Connect Four -- the tournament's ceiling.

    Negamax over the signed flat board, centre-first move ordering,
    depth-limited (``depth`` plies, default 4) with
    :func:`_c4_positional_score` at the leaves. A win found in search
    scores ``WIN_SCORE - ply`` so faster wins (and slower losses)
    are preferred. Deterministic: fixed move order, strict
    improvement keeps the first best.

    It plays from the observation only, like every baseline; the
    search state is decoded from the obs, never read off the env.
    """

    name = "connect_four_minimax"
    games = ("connect_four",)

    WIN_SCORE = 10_000.0

    def __init__(self, depth: int = 4) -> None:
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        flat = [int(v) for v in board_from_obs(obs).reshape(-1)]
        heights = _c4_heights(flat)
        best_col, best_score = -1, -float("inf")
        alpha = -float("inf")
        for col in _C4_CENTER_ORDER:
            row = _c4_drop_row(heights, col)
            if row < 0:
                continue
            idx = row * c4.COLS + col
            flat[idx] = 1
            heights[col] += 1
            if _c4_creates_win(flat, row, col):
                score = self.WIN_SCORE
            elif all(h >= c4.ROWS for h in heights):
                score = 0.0
            else:
                child = [-v for v in flat]
                score = -self._negamax(
                    child, heights, self.depth - 1, -float("inf"), -alpha, 1
                )
            flat[idx] = 0
            heights[col] -= 1
            if score > best_score:
                best_col, best_score = col, score
            alpha = max(alpha, best_score)
        return best_col

    def _negamax(
        self,
        flat: list[int],
        heights: list[int],
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
    ) -> float:
        """Score for the side to move (+1) in this node."""
        if depth == 0:
            return _c4_positional_score(flat)
        best = -float("inf")
        for col in _C4_CENTER_ORDER:
            row = _c4_drop_row(heights, col)
            if row < 0:
                continue
            idx = row * c4.COLS + col
            flat[idx] = 1
            heights[col] += 1
            if _c4_creates_win(flat, row, col):
                score = self.WIN_SCORE - ply
            elif all(h >= c4.ROWS for h in heights):
                score = 0.0
            else:
                child = [-v for v in flat]
                score = -self._negamax(
                    child, heights, depth - 1, -beta, -alpha, ply + 1
                )
            flat[idx] = 0
            heights[col] -= 1
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best


class PongTrackerAgent(BaselineAgent):
    """Reactive ball-height follower; no predictive interception."""
    name = "pong_tracker"
    games = ("pong",)

    def select_action(self, env, obs):
        error = float(obs[1] - obs[4]) * env.height
        # Braking distance matters: changing direction never teleports.
        velocity = float(obs[5]) * env.paddle_speed
        error -= velocity * abs(velocity) / (2 * env.paddle_acceleration)
        return 2 if error > 5 else 0 if error < -5 else 1


class PongInterceptAgent(PongTrackerAgent):
    """Obs-only wall-bounce prediction, with situational power activation."""
    name = "pong_intercept"

    def select_action(self, env, obs):
        x, y, vx, vy = (float(obs[0]) * env.width, float(obs[1]) * env.height,
                        float(obs[2]) * env.ball_speed_cap, float(obs[3]) * env.ball_speed_cap)
        target = env.height / 2
        incoming = vx < -1e-6 and obs[11] == 0
        eta = float("inf")
        if incoming:
            face = env.paddle_inset + env.paddle_width / 2 + env.ball_radius
            eta = max(0.0, (face - x) / vx)
            span = env.height - 2 * env.ball_radius
            folded = (y + vy * eta - env.ball_radius) % (2 * span)
            target = env.ball_radius + min(folded, 2 * span - folded)
        projected = np.array(obs, copy=True)
        projected[1] = target / env.height
        action = super().select_action(env, projected)
        inventory = int(np.argmax(obs[12:16]))
        activate = (inventory == 1 and incoming and eta < .8) or (inventory == 2 and incoming and eta < .4) or (inventory == 3 and incoming and eta < .5 and abs(target - float(obs[4]) * env.height) > env.paddle_height / 2)
        return action + (3 if activate else 0)


BASELINES: dict[str, type[BaselineAgent]] = {
    cls.name: cls
    for cls in (
        RandomAgent,
        BasicStrategyAgent,
        ProbabilityAgent,
        GreedyMatch3Agent,
        Greedy2048Agent,
        Expectimax2048Agent,
        FlappyHeuristicAgent,
        RacingHeuristicAgent,
        PongTrackerAgent,
        PongInterceptAgent,
        ConnectFourGreedyAgent,
        ConnectFourGreedyNoisyAgent,
        ConnectFourMinimaxAgent,
    )
}


def get_baseline(name: str, game: str) -> BaselineAgent:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; available: {sorted(BASELINES)}")
    agent = BASELINES[name]()
    if game not in agent.games:
        raise ValueError(f"Baseline {name!r} does not apply to game {game!r}")
    return agent
