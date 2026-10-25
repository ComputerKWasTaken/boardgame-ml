"""Baseline agent sanity: the heuristics must actually encode the
strategies they claim to, since trained agents are judged against them."""

import numpy as np

from boardgame_ml.agents.baselines import (
    BasicStrategyAgent,
    ConnectFourGreedyAgent,
    ConnectFourMinimaxAgent,
    FlappyHeuristicAgent,
    GreedyMatch3Agent,
    ProbabilityAgent,
    RacingHeuristicAgent,
    RandomAgent,
)
from boardgame_ml.envs.blackjack import BlackjackEnv, HIT, STAND
from boardgame_ml.envs.connect_four import ConnectFourEnv
from boardgame_ml.envs.flappy import FLAP, GLIDE, FlappyBirdEnv, Pipe
from boardgame_ml.envs.higher_or_lower import HigherOrLowerEnv, HIGHER, LOWER
from boardgame_ml.envs.match3 import (
    Match3Env,
    decode_action,
    find_matches,
    score_matches,
)
from boardgame_ml.envs.racing import RacingEnv

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


def _flappy_with(bird_y, bird_vy, gap_bottom, gap_top):
    env = FlappyBirdEnv()
    env.reset(seed=0)
    env.bird_y = bird_y
    env.bird_vy = bird_vy
    env.pipes = [
        Pipe(x=200.0, gap_bottom=float(gap_bottom), gap_top=float(gap_top))
    ]
    return env


def test_flappy_heuristic_flaps_when_below_the_aim_line():
    agent = FlappyHeuristicAgent()
    # Gap centred at 450; bird at 256 is far below the aim line.
    assert agent.select_action(_flappy_with(256, 0, 400, 500), None) == FLAP


def test_flappy_heuristic_glides_when_above_the_aim_line():
    agent = FlappyHeuristicAgent()
    # Gap centred at 100; bird at 256 is well above it — let gravity work.
    assert agent.select_action(_flappy_with(256, 0, 50, 150), None) == GLIDE


def test_flappy_heuristic_does_not_stack_flaps_while_rising_fast():
    agent = FlappyHeuristicAgent()
    # Below the aim line but already rising faster than the cap:
    # gliding avoids the ceiling-pinning failure mode.
    assert agent.select_action(_flappy_with(256, 200, 400, 500), None) == GLIDE


def test_flappy_heuristic_beats_random():
    """The yardstick must clear the floor by a wide margin on a fixed
    seed batch: heuristic ~9-11 pipes/episode, random ~0."""

    def mean_score(agent, episodes=100, seed0=9000):
        env = FlappyBirdEnv()
        total = 0
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed0 + ep)
            done = False
            while not done:
                obs, _r, terminated, truncated, _i = env.step(
                    agent.select_action(env, obs)
                )
                done = terminated or truncated
            total += env.score
        return total / episodes

    heuristic_mean = mean_score(FlappyHeuristicAgent())
    random_mean = mean_score(RandomAgent(seed=1))
    assert heuristic_mean >= 3.0
    assert heuristic_mean > random_mean + 3.0


# -- Racing -------------------------------------------------------------------


def _racing_obs(env, rays, speed_frac=0.0, offset=0.0):
    """A synthetic racing observation: the heuristic reads only the
    obs, so its rule can be unit-tested without an episode."""
    obs = np.zeros(env.observation_space.shape, dtype=np.float32)
    obs[: env.n_rays] = rays
    obs[env.n_rays] = speed_frac  # speed / max_speed
    obs[env.n_rays + 3] = offset  # lateral offset / half-width
    return obs


def test_random_agent_samples_the_box_for_continuous_games():
    agent = RandomAgent(seed=0)
    env = RacingEnv()
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        action = agent.select_action(env, obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert env.action_space.contains(action)


def test_racing_heuristic_steers_toward_open_space():
    agent = RacingHeuristicAgent()
    env = RacingEnv()
    angles = env.ray_angles
    # Left side (positive ray angles) walled off, right side open:
    # the distance-weighted centroid points right -> steer right (< 0).
    rays = np.where(angles > 0, 0.08, 1.0).astype(np.float32)
    action = agent.select_action(env, _racing_obs(env, rays))
    assert action[0] < -0.1
    # Mirror image: steer left.
    rays = np.where(angles < 0, 0.08, 1.0).astype(np.float32)
    action = agent.select_action(env, _racing_obs(env, rays))
    assert action[0] > 0.1


def test_racing_heuristic_brakes_when_fast_and_blind():
    agent = RacingHeuristicAgent()
    env = RacingEnv()
    # Walls close on every side and already at 90% of top speed:
    # the speed controller must brake.
    rays = np.full(env.n_rays, 0.05, dtype=np.float32)
    action = agent.select_action(env, _racing_obs(env, rays, speed_frac=0.9))
    assert action[1] < 0.0
    # Open road ahead, standing still: full-ish throttle.
    rays = np.full(env.n_rays, 1.0, dtype=np.float32)
    action = agent.select_action(env, _racing_obs(env, rays, speed_frac=0.0))
    assert action[1] > 0.5


def test_racing_heuristic_circulates_and_random_does_not():
    """The yardstick, measured: on a fixed seed batch the heuristic
    laps the track repeatedly while random barely leaves the spawn."""

    def mean_progress(agent, episodes=12, seed0=3000):
        env = RacingEnv()
        total = 0.0
        laps = 0
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed0 + ep)
            done = False
            while not done:
                obs, _r, terminated, truncated, info = env.step(
                    agent.select_action(env, obs)
                )
                done = terminated or truncated
            total += info["progress_m"]
            laps += info["laps"]
        return total / episodes, laps / episodes

    heuristic_progress, heuristic_laps = mean_progress(RacingHeuristicAgent())
    random_progress, _ = mean_progress(RandomAgent(seed=1))
    assert heuristic_laps >= 2.0
    assert heuristic_progress > random_progress + 500.0


# ---------------------------------------------------------------------------
# Connect Four baselines
# ---------------------------------------------------------------------------


def _c4_env_with(moves):
    env = ConnectFourEnv()
    obs, _ = env.reset(seed=0)
    for col in moves:
        obs, _, _, _, _ = env.step(col)
    return env, obs


def test_connect_four_greedy_takes_the_immediate_win():
    # Player to move (P0) has three stacked in column 0; P1 parked in 6.
    env, obs = _c4_env_with([0, 6, 0, 6, 0, 6])
    assert ConnectFourGreedyAgent().select_action(env, obs) == 0


def test_connect_four_greedy_blocks_as_second_player():
    # Moves: P0 parks 6/5, P1 stacks column 1 three times; P0 to move
    # must block column 1.
    env, obs = _c4_env_with([6, 1, 5, 1, 6, 1])
    assert env.current_player == 0
    assert ConnectFourGreedyAgent().select_action(env, obs) == 1


def test_connect_four_minimax_takes_the_immediate_win():
    env, obs = _c4_env_with([0, 6, 0, 6, 0, 6])
    assert ConnectFourMinimaxAgent().select_action(env, obs) == 0


def test_connect_four_minimax_opens_in_the_centre():
    env, obs = _c4_env_with([])
    assert ConnectFourMinimaxAgent().select_action(env, obs) == 3


def test_connect_four_noisy_greedy_extremes():
    """The league's noisy twin: epsilon 0 = exactly the yardstick;
    epsilon 1 = always a legal random column (never the full one)."""
    from boardgame_ml.agents.baselines import ConnectFourGreedyNoisyAgent

    env, obs = _c4_env_with([0, 6, 0, 6, 0, 6])
    quiet = ConnectFourGreedyNoisyAgent(seed=1)
    quiet.epsilon = 0.0
    assert quiet.select_action(env, obs) == 0  # takes the win, like greedy

    loud = ConnectFourGreedyNoisyAgent(seed=1)
    loud.epsilon = 1.0
    env2, obs2 = _c4_env_with([3, 3, 3, 3, 3, 3])  # column 3 is full
    for _ in range(25):
        assert loud.select_action(env2, obs2) != 3
