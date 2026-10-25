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
from boardgame_ml.envs import higher_or_lower as hol


class BaselineAgent:
    """Common interface: ``select_action(env, obs) -> int``."""

    name = "baseline"
    games: tuple[str, ...] = ()

    def select_action(self, env: BaseGameEnv, obs: np.ndarray) -> int:
        raise NotImplementedError


class RandomAgent(BaselineAgent):
    """Uniform random over currently legal actions."""

    name = "random"
    games = ("blackjack", "higher_or_lower")

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


BASELINES: dict[str, type[BaselineAgent]] = {
    cls.name: cls for cls in (RandomAgent, BasicStrategyAgent, ProbabilityAgent)
}


def get_baseline(name: str, game: str) -> BaselineAgent:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; available: {sorted(BASELINES)}")
    agent = BASELINES[name]()
    if game not in agent.games:
        raise ValueError(f"Baseline {name!r} does not apply to game {game!r}")
    return agent
