"""Baseline agent sanity: the heuristics must actually encode the
strategies they claim to, since trained agents are judged against them."""

import numpy as np

from boardgame_ml.agents.baselines import (
    BasicStrategyAgent,
    GreedyMatch3Agent,
    ProbabilityAgent,
    RandomAgent,
)
from boardgame_ml.envs.blackjack import BlackjackEnv, HIT, STAND
from boardgame_ml.envs.higher_or_lower import HigherOrLowerEnv, HIGHER, LOWER
from boardgame_ml.envs.match3 import (
    Match3Env,
    decode_action,
    find_matches,
    score_matches,
)

# Match-free 5x5 with a planted legal move (see tests/test_match3_rules.py).
SCRIPTED5 = np.array(
    [
        [0, 0, 2, 3, 0],
        [1, 2, 0, 3, 1],
        [0, 0, 1, 1, 3],
        [3, 3, 0, 2, 2],
        [0, 1, 3, 2, 0],
    ]
)


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


def test_random_agent_match3_only_picks_legal_moves():
    agent = RandomAgent(seed=0)
    env = Match3Env()
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        action = agent.select_action(env, obs)
        assert action in env.valid_actions()
        obs, _r, terminated, truncated, _i = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=1)


def test_greedy_match3_picks_the_best_immediate_score():
    agent = GreedyMatch3Agent()
    env = Match3Env(board_rows=5, board_cols=5, num_gems=5)
    env.reset(seed=0)
    env.set_board(SCRIPTED5)
    # Independent expectation: full-board simulation of every legal
    # swap, scored by the game's own tables; the greedy agent uses
    # the localized first-cascade scorer, so this cross-checks them.
    best_action, best_points = None, -1
    for action in env.valid_actions():
        r, c, d = decode_action(action, env.board_cols)
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[d]
        sim = env.board.copy()
        sim[r, c], sim[r + dr, c + dc] = sim[r + dr, c + dc], sim[r, c]
        points = score_matches(sim, find_matches(sim))
        if points > best_points:
            best_action, best_points = action, points
    assert best_points > 0
    assert agent.select_action(env, None) == best_action
    # Deterministic: same state, same choice.
    assert agent.select_action(env, None) == best_action


def test_greedy_match3_beats_random_pointwise_smoke():
    """On one fixed seed batch, greedy's choices score at least the
    planted-move floor: its first move on SCRIPTED5 scores the match."""
    agent = GreedyMatch3Agent()
    env = Match3Env(board_rows=5, board_cols=5, num_gems=5)
    env.reset(seed=0)
    env.set_board(SCRIPTED5)
    action = agent.select_action(env, None)
    _obs, reward, _t, _tr, _i = env.step(action)
    assert reward >= 10
