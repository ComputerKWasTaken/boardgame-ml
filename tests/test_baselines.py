"""Baseline agent sanity: the heuristics must actually encode the
strategies they claim to, since trained agents are judged against them."""

import numpy as np

from boardgame_ml.agents.baselines import (
    BasicStrategyAgent,
    ProbabilityAgent,
    RandomAgent,
)
from boardgame_ml.envs.blackjack import BlackjackEnv, HIT, STAND
from boardgame_ml.envs.higher_or_lower import HigherOrLowerEnv, HIGHER, LOWER


def _blackjack_with(player, dealer_up):
    env = BlackjackEnv()
    env.reset(seed=0)
    env.player_hand = player
    env.dealer_hand = [dealer_up, 5]
    return env


def test_basic_strategy_hard_totals():
    agent = BasicStrategyAgent()
    assert agent.select_action(_blackjack_with([10, 6], 7), None) == HIT    # 16 vs 7
    assert agent.select_action(_blackjack_with([10, 6], 6), None) == STAND  # 16 vs 6
    assert agent.select_action(_blackjack_with([10, 2], 4), None) == STAND  # 12 vs 4
    assert agent.select_action(_blackjack_with([10, 2], 2), None) == HIT    # 12 vs 2
    assert agent.select_action(_blackjack_with([5, 6], 10), None) == HIT    # 11 always hits
    assert agent.select_action(_blackjack_with([10, 7], 10), None) == STAND  # 17 stands


def test_basic_strategy_soft_totals():
    agent = BasicStrategyAgent()
    assert agent.select_action(_blackjack_with([1, 6], 5), None) == HIT    # soft 17 hits
    assert agent.select_action(_blackjack_with([1, 7], 9), None) == HIT    # soft 18 vs 9
    assert agent.select_action(_blackjack_with([1, 7], 6), None) == STAND  # soft 18 vs 6
    assert agent.select_action(_blackjack_with([1, 8], 10), None) == STAND  # soft 19 stands
    assert agent.select_action(_blackjack_with([1, 7], 1), None) == HIT    # soft 18 vs ace


def test_probability_agent_picks_likely_side():
    agent = ProbabilityAgent()
    env = HigherOrLowerEnv()
    # obs[0] encodes the card: (card - 1) / 12
    assert agent.select_action(env, np.array([2 / 12, 0.5, 1.0], dtype=np.float32)) == HIGHER  # card 3
    assert agent.select_action(env, np.array([11 / 12, 0.5, 1.0], dtype=np.float32)) == LOWER  # card 12
    assert agent.select_action(env, np.array([0 / 12, 0.5, 1.0], dtype=np.float32)) == HIGHER  # card 1


def test_random_agent_only_picks_valid_actions():
    agent = RandomAgent(seed=0)
    env = BlackjackEnv()
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        assert agent.select_action(env, obs) in env.valid_actions()
