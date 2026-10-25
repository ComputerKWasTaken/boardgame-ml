"""Rules tests for Connect Four (M6) — the two-player contract's game.

Win detection is incremental in the env (lines through the placed
disc); these tests pin it against scripted games in every direction
and against the brute-force reference scanner, including the border
cases where index arithmetic usually goes wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from boardgame_ml.envs import connect_four as c4
from boardgame_ml.envs import make_env


def _play(env, columns):
    """Play a scripted column sequence; return the final step tuple."""
    obs, info = env.reset(seed=0)
    step = None
    for col in columns:
        step = env.step(col)
        obs, info = step[0], step[4]
    return step


def _drop_pattern_board():
    """A full board with NO four-in-a-row anywhere (verified by the
    brute-force scanner in the test): pairs pattern, color =
    ((row//2 + col) % 2)."""
    board = np.zeros((c4.ROWS, c4.COLS), dtype=np.int8)
    for r in range(c4.ROWS):
        for c in range(c4.COLS):
            board[r, c] = c4.P0 if ((r // 2 + c) % 2 == 0) else c4.P1
    return board


def test_vertical_win():
    env = make_env("connect_four")
    obs, reward, terminated, truncated, info = _play(env, [0, 1, 0, 1, 0, 1, 0])
    assert terminated and not truncated
    assert reward == 1.0
    assert info["winner"] == 0


def test_horizontal_win_bottom_row():
    env = make_env("connect_four")
    _, reward, terminated, _, info = _play(env, [0, 4, 1, 5, 2, 6, 3])
    assert terminated and reward == 1.0 and info["winner"] == 0


def test_horizontal_win_at_right_border():
    env = make_env("connect_four")
    _, reward, terminated, _, info = _play(env, [3, 0, 4, 1, 5, 2, 6])
    assert terminated and reward == 1.0 and info["winner"] == 0


def test_vertical_win_in_last_column():
    env = make_env("connect_four")
    _, reward, terminated, _, info = _play(env, [6, 0, 6, 0, 6, 0, 6])
    assert terminated and reward == 1.0 and info["winner"] == 0


def test_diagonal_up_right_win():
    env = make_env("connect_four")
    # Builds (5,0) (4,1) (3,2) (2,3) for player 0; the last drop wins.
    _, reward, terminated, _, info = _play(
        env, [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3]
    )
    assert terminated and reward == 1.0 and info["winner"] == 0


def test_diagonal_up_left_win():
    env = make_env("connect_four")
    # Mirror image: (5,6) (4,5) (3,4) (2,3) for player 0.
    _, reward, terminated, _, info = _play(
        env, [6, 5, 5, 4, 4, 3, 4, 3, 3, 0, 3]
    )
    assert terminated and reward == 1.0 and info["winner"] == 0


def test_second_player_can_win_and_reward_is_movers():
    env = make_env("connect_four")
    # Player 1 stacks column 1 while player 0 harmlessly stacks col 6
    # (only three discs: 0 plays first, so it moves one fewer time).
    _, reward, terminated, _, info = _play(env, [6, 1, 6, 1, 6, 1, 5, 1])
    assert terminated and reward == 1.0  # the mover (player 1) won
    assert info["winner"] == 1


def test_no_win_no_termination_midgame():
    env = make_env("connect_four")
    _, reward, terminated, truncated, info = _play(env, [3, 3, 2, 4])
    assert not terminated and not truncated
    assert reward == 0.0 and info["winner"] is None


def test_mask_is_exactly_the_non_full_columns():
    env = make_env("connect_four")
    _play(env, [0, 0, 0, 0, 0, 0])  # fill column 0 with alternating discs
    mask = env.action_masks()
    assert mask.tolist() == [False, True, True, True, True, True, True]
    assert env.valid_actions() == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        env.step(0)


def test_terminal_mask_is_all_true_by_convention():
    env = make_env("connect_four")
    _play(env, [0, 1, 0, 1, 0, 1, 0])  # player 0 wins
    assert env.action_masks().all()


def test_draw_on_full_board():
    env = make_env("connect_four")
    board = _drop_pattern_board()
    assert not c4.board_has_win(board, c4.P0)
    assert not c4.board_has_win(board, c4.P1)
    # Load the pattern minus the top cell of column 0, then drop it.
    full = board.copy()
    board[0, 0] = c4.EMPTY
    env.reset(seed=0)
    env.board[:] = board
    env._heights[:] = [int((board[:, c] != c4.EMPTY).sum()) for c in range(7)]
    env._moves = 41
    env.current_player = 1 if full[0, 0] == c4.P1 else 0
    mover_disc = int(full[0, 0])
    assert (c4.P0 if env.current_player == 0 else c4.P1) == mover_disc
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert reward == 0.0 and info["winner"] is None
    assert env.moves_played == 42


def test_perspective_observation_swaps_with_the_mover():
    env = make_env("connect_four")
    obs, info = env.reset(seed=0)
    assert obs.sum() == 0
    assert info["current_player"] == 0 and info["winner"] is None

    obs, _, _, _, info = env.step(3)  # player 0 drops in column 3
    # Now player 1 is to move: player 0's disc is "the opponent's".
    assert info["current_player"] == 1
    assert obs[5, 3, 1] == 1.0 and obs[..., 0].sum() == 0

    obs, _, _, _, info = env.step(3)  # player 1 stacks on it
    assert info["current_player"] == 0
    assert obs[5, 3, 0] == 1.0  # player 0's disc is "mine" again
    assert obs[4, 3, 1] == 1.0  # player 1's disc is "theirs"


def test_stepping_a_finished_game_raises():
    env = make_env("connect_four")
    _play(env, [0, 1, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        env.step(2)


def test_incremental_win_check_matches_brute_force():
    """Fuzz: random legal games; after every move the incremental
    result (env termination) must agree with a full board scan."""
    rng = np.random.default_rng(7)
    env = make_env("connect_four")
    for game in range(30):
        env.reset(seed=game)
        while True:
            legal = env.valid_actions()
            col = int(rng.choice(legal))
            mover = env.current_player
            _, _, terminated, _, info = env.step(col)
            value = c4.P0 if mover == 0 else c4.P1
            brute = c4.board_has_win(env.board, value)
            assert terminated == (brute or env.moves_played == 42)
            if brute:
                assert info["winner"] == mover
            if terminated:
                break


def test_render_names_the_state():
    env = make_env("connect_four", render_mode="human")
    env.reset(seed=0)
    assert "To move: player 0" in env.render()
    _play(env, [0, 1, 0, 1, 0, 1, 0])
    assert "Winner: player 0" in env.render()
