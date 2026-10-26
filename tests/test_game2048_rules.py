"""2048 rules tests — the env mechanics are the heart of the milestone.

Everything here is stated against the env module's pure engine
(``slide_line`` / ``apply_move`` / ``legal_moves_mask`` /
``spawn_tile``) and the env built on it, plus the fast int engine
the baselines search with (cross-checked against the array engine
so the two can never drift) and the baselines themselves.

Boards are written as exponents: 0 is empty, k is the tile 2**k.
"""

from __future__ import annotations

import numpy as np
import pytest

from boardgame_ml.agents.baselines import get_baseline
from boardgame_ml.envs import game2048 as g48
from boardgame_ml.envs import make_env
from boardgame_ml.envs.game2048 import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Game2048Env,
    apply_move,
    legal_moves_mask,
    slide_line,
    spawn_tile,
)

B = lambda rows: np.array(rows, dtype=np.int64)  # noqa: E731


# ---------------------------------------------------------------------------
# slide_line: merge rules, one line at a time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected,gain",
    [
        # The canonical cases from the rules: pairs merge once.
        ([1, 1, 1, 1], [2, 2, 0, 0], 8),    # 2,2,2,2 -> 4,4
        ([2, 1, 1, 0], [2, 2, 0, 0], 4),    # 4,2,2 -> 4,4
        ([1, 1, 2, 0], [2, 2, 0, 0], 4),    # 2,2,4 -> 4,4 (NOT 8)
        ([1, 1, 1, 0], [2, 1, 0, 0], 4),    # first pair merges, third stays
        ([3, 3, 0, 0], [4, 0, 0, 0], 16),   # 8,8 -> 16
        ([2, 2, 3, 3], [3, 4, 0, 0], 24),   # two independent merges
        ([0, 1, 0, 1], [2, 0, 0, 0], 4),    # gaps close before merging
        ([0, 0, 0, 2], [2, 0, 0, 0], 0),    # lone tile slides, no gain
        ([1, 2, 1, 2], [1, 2, 1, 2], 0),    # alternating: nothing merges
        ([0, 0, 0, 0], [0, 0, 0, 0], 0),    # empty line
        ([5, 0, 0, 0], [5, 0, 0, 0], 0),    # already packed at the edge
    ],
)
def test_slide_line(line, expected, gain):
    out, got_gain = slide_line(line)
    assert out == expected
    assert got_gain == gain


def test_slide_line_never_chain_merges():
    """A tile created by a merge cannot merge again in the same move:
    2,2,4,4 -> 4,8 (the fresh 4 stays), never a single 16... and
    4,4,8 -> 8,8 where the fresh 8 does not absorb the old 8."""
    out, gain = slide_line([1, 1, 2, 2])
    assert out == [2, 3, 0, 0]
    assert gain == 4 + 8
    out, gain = slide_line([2, 2, 3, 0])
    assert out == [3, 3, 0, 0]
    assert gain == 8


# ---------------------------------------------------------------------------
# apply_move: all four directions on one asymmetric board
# ---------------------------------------------------------------------------

FOUR_WAY = B(
    [
        [1, 0, 1, 2],
        [0, 0, 2, 2],
        [3, 0, 0, 2],
        [3, 1, 0, 0],
    ]
)


def test_apply_move_left():
    new, gain, changed = apply_move(FOUR_WAY, LEFT)
    assert changed
    assert gain == 12
    assert new.tolist() == [
        [2, 2, 0, 0],
        [3, 0, 0, 0],
        [3, 2, 0, 0],
        [3, 1, 0, 0],
    ]


def test_apply_move_right():
    new, gain, changed = apply_move(FOUR_WAY, RIGHT)
    assert changed
    assert gain == 12
    assert new.tolist() == [
        [0, 0, 2, 2],
        [0, 0, 0, 3],
        [0, 0, 3, 2],
        [0, 0, 3, 1],
    ]


def test_apply_move_up():
    new, gain, changed = apply_move(FOUR_WAY, UP)
    assert changed
    assert gain == 24
    assert new.tolist() == [
        [1, 1, 1, 3],
        [4, 0, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]


def test_apply_move_down():
    new, gain, changed = apply_move(FOUR_WAY, DOWN)
    assert changed
    assert gain == 24
    assert new.tolist() == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 1, 2],
        [4, 1, 2, 3],
    ]


def test_apply_move_does_not_mutate():
    before = FOUR_WAY.copy()
    apply_move(FOUR_WAY, LEFT)
    assert np.array_equal(FOUR_WAY, before)


# ---------------------------------------------------------------------------
# Legality: the mask is exactly the board-changing moves
# ---------------------------------------------------------------------------

LOCKED = B(  # full board, no equal orthogonal neighbours: terminal
    [
        [1, 2, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ]
)

FULL_MERGEABLE = B(  # full board, one horizontal pair: only L/R legal
    [
        [1, 1, 3, 4],
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ]
)


def test_mask_on_locked_board_is_all_false():
    assert legal_moves_mask(LOCKED).tolist() == [False, False, False, False]


def test_mask_on_full_mergeable_board():
    mask = legal_moves_mask(FULL_MERGEABLE)
    # Action order is UP, RIGHT, DOWN, LEFT.
    assert mask.tolist() == [False, True, False, True]


def test_mask_matches_apply_move_on_random_boards():
    rng = np.random.default_rng(7)
    for _ in range(200):
        board = rng.integers(0, 6, size=(4, 4))
        mask = legal_moves_mask(board)
        for action in range(4):
            assert mask[action] == apply_move(board, action)[2]


def test_opening_board_mask():
    """A fresh board (two tiles) always has a legal move, and its
    mask is the pure legal set."""
    env = Game2048Env()
    env.reset(seed=3)
    mask = env.action_masks()
    assert mask.any()
    assert mask.tolist() == legal_moves_mask(env.board).tolist()


# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------


def test_locked_board_is_terminal():
    env = Game2048Env()
    env.set_board(LOCKED)
    assert env.game_over
    # Terminal mask convention (module docstring): action_masks()
    # reports all-True once the game is over; the pure function is
    # the one that reports the (empty) legal set.
    assert env.action_masks().all()
    with pytest.raises(ValueError, match="game is over"):
        env.step(UP)


def test_full_mergeable_board_is_not_terminal():
    env = Game2048Env()
    env.set_board(FULL_MERGEABLE)
    assert not env.game_over
    assert env.action_masks().tolist() == [False, True, False, True]


def test_almost_locked_board_with_a_gap_is_not_terminal():
    board = LOCKED.copy()
    board[3, 3] = 0  # one hole: the last column/row can slide into it
    assert legal_moves_mask(board).any()
    env = Game2048Env()
    env.set_board(board)
    assert not env.game_over


# ---------------------------------------------------------------------------
# Spawning
# ---------------------------------------------------------------------------


def test_spawn_lands_on_empty_cell_with_legal_value():
    rng = np.random.default_rng(0)
    board = B(
        [
            [1, 0, 2, 0],
            [0, 3, 0, 4],
            [5, 0, 6, 0],
            [0, 7, 0, 8],
        ]
    )
    for _ in range(200):
        new_board, spawned = spawn_tile(board, rng)
        assert spawned is not None
        r, c, e = spawned
        assert board[r, c] == 0  # was empty
        assert e in (1, 2)  # a 2 or a 4
        assert new_board[r, c] == e
        # Exactly one cell changed.
        assert np.count_nonzero(new_board != board) == 1


def test_spawn_value_distribution_seeded():
    """90% 2s / 10% 4s, over 20,000 seeded spawns (~5 sigma band)."""
    rng = np.random.default_rng(42)
    empty = np.zeros((4, 4), dtype=np.int64)
    twos = 0
    trials = 20_000
    for _ in range(trials):
        _, spawned = spawn_tile(empty, rng)
        assert spawned is not None
        if spawned[2] == 1:
            twos += 1
    share = twos / trials
    assert 0.885 < share < 0.915


def test_spawn_cells_are_uniform_seeded():
    """Each of the 16 cells is chosen ~equally often (seeded)."""
    rng = np.random.default_rng(1)
    empty = np.zeros((4, 4), dtype=np.int64)
    counts = np.zeros((4, 4), dtype=int)
    trials = 16_000
    for _ in range(trials):
        _, spawned = spawn_tile(empty, rng)
        counts[spawned[0], spawned[1]] += 1
    expected = trials / 16
    assert counts.min() > expected * 0.85
    assert counts.max() < expected * 1.15


def test_spawn_on_full_board_is_none():
    _, spawned = spawn_tile(LOCKED, np.random.default_rng(0))
    assert spawned is None


def test_spawn_only_after_a_board_changing_move():
    """An illegal (masked-out) action is a rejected no-op: no slide,
    NO spawn, no score, the move is not consumed."""
    env = Game2048Env()
    env.set_board(FULL_MERGEABLE)  # UP and DOWN change nothing here
    before = env.board.copy()
    _obs, reward, terminated, truncated, info = env.step(UP)
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["illegal_action"] is True
    assert info["spawned"] is None
    assert np.array_equal(env.board, before)
    assert env.score == 0 and env.moves == 0


def test_legal_move_spawns_exactly_one_tile():
    env = Game2048Env()
    env.set_board(FOUR_WAY)
    post_move, gain, _ = apply_move(FOUR_WAY, LEFT)
    _obs, reward, *_rest, info = env.step(LEFT)
    assert reward == float(gain) == info["points"]
    assert info["illegal_action"] is False
    spawned = info["spawned"]
    assert spawned is not None
    r, c, e = spawned
    assert post_move[r, c] == 0 and e in (1, 2)
    # The env's board is the post-move board plus exactly that spawn.
    expected = post_move.copy()
    expected[r, c] = e
    assert np.array_equal(env.board, expected)
    assert env.moves == 1 and env.score == gain


# ---------------------------------------------------------------------------
# Reset, observation, reward accounting
# ---------------------------------------------------------------------------


def test_reset_state():
    env = Game2048Env()
    obs, info = env.reset(seed=0)
    assert np.count_nonzero(env.board) == 2  # the standard two tiles
    assert set(np.unique(env.board)) <= {0, 1, 2}
    assert env.score == 0 and env.moves == 0 and not env.game_over
    assert info["score"] == 0
    assert obs.shape == (16, 4, 4) and obs.dtype == np.float32


def test_observation_is_one_hot_over_exponents():
    env = Game2048Env()
    env.set_board(FOUR_WAY)
    obs = env._obs()
    assert obs.shape == (16, 4, 4)
    assert obs.sum() == 16.0  # exactly one hot channel per cell
    for r in range(4):
        for c in range(4):
            assert obs[FOUR_WAY[r, c], r, c] == 1.0


def test_observation_clips_exponents_above_15():
    env = Game2048Env()
    board = np.zeros((4, 4), dtype=np.int64)
    board[0, 0] = 17  # a 131072 tile: state keeps it, obs clips it
    env.set_board(board)
    assert env.board[0, 0] == 17
    obs = env._obs()
    assert obs[15, 0, 0] == 1.0


def test_score_accumulates_merge_gains():
    env = Game2048Env()
    env.set_board(B([[1, 1, 0, 0]] * 1 + [[0, 0, 0, 0]] * 3), score=100)
    _obs, reward, *_ , info = env.step(LEFT)  # 2+2 -> 4: gain 4
    assert reward == 4.0
    assert env.score == 104 == info["score"]


def test_invalid_action_and_step_after_game_over_raise():
    env = Game2048Env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(4)


def test_games_end_to_end_and_terminate():
    """Random-legal play always terminates, with the final mask
    empty and the score equal to the sum of step rewards."""
    env = Game2048Env()
    rng = np.random.default_rng(5)
    env.reset(seed=5)
    total = 0.0
    steps = 0
    while not env.game_over:
        legal = np.flatnonzero(env.action_masks())
        action = int(legal[rng.integers(len(legal))])
        _obs, reward, terminated, truncated, _info = env.step(action)
        total += reward
        steps += 1
        assert not truncated
        assert terminated == env.game_over
        assert steps < 100_000  # sanity: games do end
    assert total == float(env.score)
    assert not legal_moves_mask(env.board).any()


# ---------------------------------------------------------------------------
# Seeded determinism
# ---------------------------------------------------------------------------


def _play_first_legal(seed: int, max_steps: int = 500):
    env = Game2048Env()
    env.reset(seed=seed)
    boards = [env.board.copy()]
    for _ in range(max_steps):
        if env.game_over:
            break
        action = int(np.flatnonzero(env.action_masks())[0])
        env.step(action)
        boards.append(env.board.copy())
    return boards, env.score


def test_seeded_determinism_full_trajectory():
    boards_a, score_a = _play_first_legal(123)
    boards_b, score_b = _play_first_legal(123)
    assert score_a == score_b
    assert len(boards_a) == len(boards_b)
    for a, b in zip(boards_a, boards_b):
        assert np.array_equal(a, b)


def test_different_seeds_usually_differ():
    boards_a, _ = _play_first_legal(1, max_steps=30)
    boards_b, _ = _play_first_legal(2, max_steps=30)
    assert not all(
        np.array_equal(a, b) for a, b in zip(boards_a, boards_b)
    )


# ---------------------------------------------------------------------------
# The int engine must be the same game (baselines search with it)
# ---------------------------------------------------------------------------


def test_int_engine_round_trip_and_transpose():
    rng = np.random.default_rng(11)
    for _ in range(200):
        board = rng.integers(0, 10, size=(4, 4))
        x = g48.board_to_int(board)
        assert np.array_equal(g48.int_to_board(x), board)
        xt = g48.transpose_int(x)
        assert np.array_equal(g48.int_to_board(xt), board.T)
        assert g48.transpose_int(xt) == x


def test_int_engine_matches_array_engine():
    rng = np.random.default_rng(13)
    for _ in range(300):
        board = rng.integers(0, 8, size=(4, 4))
        x = g48.board_to_int(board)
        for action in range(4):
            new_board, gain, changed = apply_move(board, action)
            new_x, gain_x, changed_x = g48.move_int(x, action)
            assert gain == gain_x and changed == changed_x
            assert np.array_equal(g48.int_to_board(new_x), new_board)
        assert np.array_equal(
            g48.legal_mask_int(x), legal_moves_mask(board)
        )


def test_row_table_saturates_exponent_16():
    """The documented saturation: 32768+32768 in the int engine
    yields a (clipped) 32768 with the true 65536 gain — the array
    engine the env plays by keeps the full exponent."""
    slide_left, _rev = g48._row_tables()
    row = 15 | (15 << 4)  # line [15, 15, 0, 0]
    new_row, gain = slide_left[row]
    assert gain == 65536
    assert new_row & 0xF == 15  # saturated, not wrapped/corrupted
    out, array_gain = slide_line([15, 15, 0, 0])
    assert out[0] == 16 and array_gain == 65536


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _env_with(board) -> Game2048Env:
    env = Game2048Env()
    env.set_board(board)
    return env


def test_greedy_picks_the_biggest_immediate_gain():
    agent = get_baseline("greedy_1ply", "game2048")
    env = _env_with(FOUR_WAY)  # gains: UP 24, DOWN 24, LEFT 12, RIGHT 12
    # UP and DOWN tie at 24 -> lowest action index wins: UP (0).
    assert agent.select_action(env, None) == UP


def test_greedy_tie_breaks_to_lowest_action_index():
    agent = get_baseline("greedy_1ply", "game2048")
    board = B(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    env = _env_with(board)
    gains = {a: apply_move(board, a)[1] for a in range(4)}
    assert len(set(gains.values())) == 1  # all four moves gain 8
    assert agent.select_action(env, None) == UP


def test_baselines_always_play_legal_moves():
    rng = np.random.default_rng(21)
    env = Game2048Env()
    for name in ("random", "greedy_1ply", "expectimax"):
        agent = get_baseline(name, "game2048")
        for game in range(3):
            env.reset(seed=100 + game)
            while not env.game_over:
                action = agent.select_action(env, None)
                assert env.action_masks()[action]
                env.step(action)


def test_expectimax_is_deterministic():
    env = _env_with(FOUR_WAY)
    a1 = get_baseline("expectimax", "game2048").select_action(env, None)
    a2 = get_baseline("expectimax", "game2048").select_action(env, None)
    assert a1 == a2


def test_expectimax_terminal_value_is_zero():
    agent = get_baseline("expectimax", "game2048")
    assert agent._value(g48.board_to_int(LOCKED), 2) == 0.0


def test_expectimax_heuristic_prefers_the_corner():
    """The snake gradient prices a big tile in the top-left corner
    (path position 0, weight 1) far above the same tile mid-board."""
    corner = np.zeros((4, 4), dtype=np.int64)
    corner[0, 0] = 11  # a 2048 tile in the corner
    center = np.zeros((4, 4), dtype=np.int64)
    center[1, 1] = 11
    agent = get_baseline("expectimax", "game2048")
    assert agent.heuristic(g48.board_to_int(corner)) > agent.heuristic(
        g48.board_to_int(center)
    )


def test_expectimax_beats_greedy_on_shared_seeds():
    """The yardstick ordering at small scale (the Games-card test
    asserts it over 10 games; the calibration report measures it
    over 100-1,000). Two paired games keep this test fast while
    still catching a broken search — the measured gap is ~4x."""

    def play(agent_name: str, seed: int) -> int:
        env = make_env("game2048")
        agent = get_baseline(agent_name, "game2048")
        env.reset(seed=seed)
        while True:
            _o, _r, terminated, _t, info = env.step(
                agent.select_action(env, None)
            )
            if terminated:
                return info["score"]

    for seed in (7, 8):
        assert play("expectimax", seed) > play("greedy_1ply", seed)
