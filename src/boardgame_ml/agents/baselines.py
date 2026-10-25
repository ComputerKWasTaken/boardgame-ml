"""Non-learning reference agents.

Baselines are the framework's proof-of-learning yardsticks: a trained
agent only counts as having learned if it beats :class:`RandomAgent`
and approaches the relevant heuristic agent. Every baseline only ever
chooses actions that are legal under ``env.action_masks()``.
"""

from __future__ import annotations

import numpy as np

from boardgame_ml.envs.base import BaseGameEnv
from boardgame_ml.envs import blackjack as bj
from boardgame_ml.envs import flappy as fb
from boardgame_ml.envs import higher_or_lower as hol
from boardgame_ml.envs import match3 as m3


class BaselineAgent:
    """Common interface: ``select_action(env, obs) -> int``."""

    name = "baseline"
    games: tuple[str, ...] = ()

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        raise NotImplementedError


class RandomAgent(BaselineAgent):
    """Uniform random over currently legal actions."""

    name = "random"
    games = ("blackjack", "higher_or_lower", "match3", "flappy")

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
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


BASELINES: dict[str, type[BaselineAgent]] = {
    cls.name: cls
    for cls in (
        RandomAgent,
        BasicStrategyAgent,
        ProbabilityAgent,
        GreedyMatch3Agent,
        FlappyHeuristicAgent,
    )
}


def get_baseline(name: str, game: str) -> BaselineAgent:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; available: {sorted(BASELINES)}")
    agent = BASELINES[name]()
    if game not in agent.games:
        raise ValueError(f"Baseline {name!r} does not apply to game {game!r}")
    return agent
