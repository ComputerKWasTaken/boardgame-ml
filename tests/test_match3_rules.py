"""Match 3 engine and env contract tests.

Scripted boards are hand-verified; where a full legal set matters,
the expected set is computed by an independent brute force (copy the
board, swap, run the full-board match finder) — deliberately *not*
the localized legality/scoring code under test.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from boardgame_ml.envs.match3 import (
    DOWN,
    RIGHT,
    UP,
    Match3Env,
    component_score,
    decode_action,
    find_matches,
    legal_actions_mask,
    resolve_board,
    score_matches,
    score_runs,
    swap_first_cascade_score,
)

# Latin pattern: no two orthogonally adjacent cells are equal, so the
# board is match-free by inspection.
LATIN5 = np.array(
    [
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3],
        [0, 1, 2, 3, 4],
    ]
)

# Match-free 5x5 with a planted legal move: swapping (1,2)=0 down
# into (2,2) makes row 2 read 0,0,0,1,3.
SCRIPTED5 = np.array(
    [
        [0, 0, 2, 3, 0],
        [1, 2, 0, 3, 1],
        [0, 0, 1, 1, 3],
        [3, 3, 0, 2, 2],
        [0, 1, 3, 2, 0],
    ]
)


def brute_force_mask(board: np.ndarray) -> np.ndarray:
    """Legal set by full-board simulation of every directed swap."""
    rows, cols = board.shape
    mask = np.zeros(rows * cols * 4, dtype=bool)
    for action in range(rows * cols * 4):
        r, c, d = decode_action(action, cols)
        dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[d]
        r2, c2 = r + dr, c + dc
        if not (0 <= r2 < rows and 0 <= c2 < cols):
            continue
        sim = board.copy()
        sim[r, c], sim[r2, c2] = sim[r2, c2], sim[r, c]
        if find_matches(sim).any():
            mask[action] = True
    return mask


class StubRng:
    """Deterministic stand-in for np.random.Generator in resolve tests."""

    def __init__(self, values):
        self._values = list(values)
        self._pos = 0

    def integers(self, low, high=None, size=None):
        n = 1 if size is None else int(np.prod(size))
        out = np.array(self._values[self._pos : self._pos + n], dtype=np.int64)
        assert out.size == n, "StubRng exhausted"
        self._pos += n
        return out.reshape(size) if size is not None else out[0]


# ---------------------------------------------------------------------------
# Match detection
# ---------------------------------------------------------------------------


def test_find_matches_horizontal_and_vertical():
    board = LATIN5.copy()
    assert not find_matches(board).any()

    board[0, 0] = 2
    board[0, 1] = 2
    board[0, 2] = 2  # row 0 becomes 2,2,2,4,0
    matched = find_matches(board)
    assert matched[0, 0] and matched[0, 1] and matched[0, 2]
    assert matched.sum() == 3

    board = LATIN5.copy()
    board[2, 4] = 3
    board[3, 4] = 3
    board[4, 4] = 3  # column 4 rows 2-4 (was 2,3,4)
    matched = find_matches(board)
    assert matched[:, 4].tolist() == [False, False, True, True, True]
    assert matched.sum() == 3


def test_find_matches_run_of_four_marks_all_four():
    board = LATIN5.copy()
    board[0, 1] = 1
    board[0, 2] = 1
    board[0, 3] = 1  # row 0: 1,1,1,1,0
    matched = find_matches(board)
    assert matched[0].tolist() == [True, True, True, True, False]


# ---------------------------------------------------------------------------
# Scoring: component table, L-cross merge, cascade multiplier
# ---------------------------------------------------------------------------


def test_component_score_table():
    assert component_score(3) == 10
    assert component_score(4) == 25
    assert component_score(5) == 50
    assert component_score(6) == 100
    assert component_score(9) == 100
    assert component_score(2) == 0


def test_score_matches_l_cross_is_one_component():
    board = LATIN5.copy()
    # Horizontal 3 on row 2 (cols 0-2) + vertical 3 on col 2 (rows 2-4),
    # sharing cell (2,2): one L component of 5 gems -> 50, not 10 + 10.
    board[2, 0] = 2
    board[2, 1] = 2
    board[2, 2] = 2
    board[3, 2] = 2
    board[4, 2] = 2
    matched = find_matches(board)
    assert matched.sum() == 5
    assert score_matches(board, matched) == 50


def test_score_matches_four_run():
    board = LATIN5.copy()
    board[0, 1] = 1
    board[0, 2] = 1
    board[0, 3] = 1
    assert score_matches(board, find_matches(board)) == 25


def test_score_runs_large_component():
    runs = [(0, [(0, c) for c in range(6)])]
    assert score_runs(runs) == 100


def test_score_matches_stacked_runs_of_different_gems_stay_separate():
    # REGRESSION: column 0 holds a run of 1s directly above a run of
    # 2s. The matched mask is one contiguous segment in that column,
    # but they are two components (10 + 10), not one 6-gem component
    # (100). A mask-contiguity scan once merged them.
    board = np.array(
        [
            [1, 1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 0],
            [1, 3, 4, 5, 0, 1],
            [2, 4, 5, 0, 1, 2],
            [2, 5, 0, 1, 2, 3],
            [2, 0, 1, 2, 3, 4],
        ]
    )
    matched = find_matches(board)
    assert matched.sum() == 6
    assert score_matches(board, matched) == 20


# ---------------------------------------------------------------------------
# Resolution: gravity + refill, cascades x level — exact with a stub RNG
# ---------------------------------------------------------------------------


def test_resolve_board_gravity_and_refill_exact():
    board = LATIN5.copy()
    board[4, 1] = 2
    board[4, 2] = 2
    board[4, 3] = 2  # only match: row 4, cols 1-3 (three 2s)
    assert find_matches(board).sum() == 3

    points, levels = resolve_board(board, StubRng([3, 0, 4]), num_gems=5)
    assert (points, levels) == (10, 1)
    # Cleared cells are refilled per column (draws 3, 0, 4), survivors
    # keep their order as they fall.
    expected = np.array(
        [
            [1, 3, 0, 4, 0],
            [2, 2, 3, 4, 1],
            [3, 3, 4, 0, 2],
            [4, 4, 0, 1, 3],
            [0, 0, 1, 2, 4],
        ]
    )
    np.testing.assert_array_equal(board, expected)
    assert not find_matches(board).any()


def test_resolve_board_cascade_multiplier_exact():
    # Column 0 holds a vertical match of 5s (rows 1-3). Clearing it
    # drops scripted refills [1,3,2] on top of the surviving 2 at the
    # bottom, forming 2,2,2 at rows 2-4 -> a level-2 cascade.
    board = np.array(
        [
            [2, 2, 3, 4, 0],
            [5, 3, 4, 0, 1],
            [5, 4, 0, 1, 2],
            [5, 0, 1, 2, 3],
            [2, 1, 2, 3, 4],
        ]
    )
    assert find_matches(board).sum() == 3
    assert score_matches(board, find_matches(board)) == 10

    points, levels = resolve_board(board, StubRng([1, 3, 2, 0, 4, 1]), num_gems=6)
    assert levels == 2
    assert points == 10 * 1 + 10 * 2  # component 10 at level 1, 10 at level 2
    expected = np.array(
        [
            [0, 2, 3, 4, 0],
            [4, 3, 4, 0, 1],
            [1, 4, 0, 1, 2],
            [1, 0, 1, 2, 3],
            [3, 1, 2, 3, 4],
        ]
    )
    np.testing.assert_array_equal(board, expected)
    assert not find_matches(board).any()


def test_resolve_board_seeded_invariants():
    rng = np.random.default_rng(123)
    board = LATIN5.copy()
    board[4, 1] = 2
    board[4, 2] = 2
    board[4, 3] = 2
    points, levels = resolve_board(board, rng, num_gems=5)
    assert points >= 10 and levels >= 1
    assert board.shape == (5, 5)
    assert board.min() >= 0 and board.max() < 5
    assert not find_matches(board).any()


# ---------------------------------------------------------------------------
# Legality: mask == true legal set; localized score == full-board score
# ---------------------------------------------------------------------------


def test_legal_mask_matches_brute_force_on_scripted_board():
    assert not find_matches(SCRIPTED5).any()
    mask = legal_actions_mask(SCRIPTED5)
    np.testing.assert_array_equal(mask, brute_force_mask(SCRIPTED5))
    # The planted move, both directions.
    assert mask[(1 * 5 + 2) * 4 + DOWN]
    assert mask[(2 * 5 + 2) * 4 + UP]
    # Swapping two equal gems (0,0)<->(0,1) can never be legal.
    assert not mask[(0 * 5 + 0) * 4 + RIGHT]


def test_legal_mask_matches_brute_force_on_random_boards():
    env = Match3Env()
    for seed in range(10):
        env.reset(seed=seed)
        np.testing.assert_array_equal(
            env.action_masks(), brute_force_mask(env.board)
        )


def test_swap_score_consistent_with_full_board_scoring():
    env = Match3Env()
    rng = np.random.default_rng(7)
    for seed in range(6):
        env.reset(seed=100 + seed)
        for action in env.valid_actions():
            r, c, d = decode_action(action, env.board_cols)
            dr, dc = ((-1, 0), (0, 1), (1, 0), (0, -1))[d]
            local = swap_first_cascade_score(env.board, r, c, r + dr, c + dc)
            sim = env.board.copy()
            sim[r, c], sim[r + dr, c + dc] = sim[r + dr, c + dc], sim[r, c]
            full = score_matches(sim, find_matches(sim))
            assert local == full > 0
        assert rng is not None  # (rng kept for symmetry; boards from env seeds)


# ---------------------------------------------------------------------------
# Env contract
# ---------------------------------------------------------------------------


def test_check_env_default_and_small():
    check_env(Match3Env(), skip_render_check=True)
    check_env(
        Match3Env(board_rows=5, board_cols=5, num_gems=5),
        skip_render_check=True,
    )


def test_reset_always_offers_a_legal_move():
    env = Match3Env()
    for seed in range(50):
        obs, info = env.reset(seed=seed)
        assert not find_matches(env.board).any()
        assert info["action_mask"].any()
        assert obs["board"].shape == (6, 8, 8)
        # One-hot planes: exactly one hot channel per cell.
        np.testing.assert_array_equal(obs["board"].sum(axis=0), np.ones((8, 8)))
        assert obs["moves_left"][0] == pytest.approx(1.0)


def test_illegal_action_rejected_without_consuming_move():
    env = Match3Env(board_rows=5, board_cols=5, num_gems=5)
    env.reset(seed=0)
    env.set_board(SCRIPTED5)
    before = env.board.copy()
    illegal = int(np.argmax(~env.action_masks()))
    _obs, reward, terminated, truncated, info = env.step(illegal)
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["illegal_action"] is True
    assert env.moves_used == 0
    np.testing.assert_array_equal(env.board, before)


def test_out_of_space_action_raises():
    env = Match3Env()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(256)
    with pytest.raises(ValueError):
        env.step(-1)


def test_legal_move_scores_and_consumes():
    env = Match3Env(board_rows=5, board_cols=5, num_gems=5)
    env.reset(seed=0)
    env.set_board(SCRIPTED5)
    action = (1 * 5 + 2) * 4 + DOWN
    _obs, reward, terminated, truncated, info = env.step(action)
    assert reward >= 10  # at least the planted 3-match at level 1
    assert reward == info["points"]
    assert env.score == int(reward)  # score started at 0; reward is pure score
    assert env.moves_used == 1
    assert not terminated and not truncated
    assert info["illegal_action"] is False
    assert info["last_move"]["from"] == [1, 2]
    assert info["last_move"]["to"] == [2, 2]
    # Post-move board is settled and playable.
    assert not find_matches(env.board).any()
    assert env.action_masks().any()


def test_truncation_at_move_cap():
    env = Match3Env(max_moves=3)
    rng = np.random.default_rng(0)
    env.reset(seed=3)
    for step_i in range(3):
        action = int(rng.choice(env.valid_actions()))
        _obs, _r, terminated, truncated, _info = env.step(action)
        assert not terminated
        assert truncated == (step_i == 2)
    assert env.moves_used == 3


def test_full_episode_random_play_invariants():
    env = Match3Env()
    rng = np.random.default_rng(1)
    for seed in range(20):
        env.reset(seed=seed)
        done = False
        total = 0.0
        while not done:
            assert not find_matches(env.board).any()
            action = int(rng.choice(env.valid_actions()))
            _obs, reward, terminated, truncated, _info = env.step(action)
            total += reward
            done = terminated or truncated
        assert truncated and not terminated
        assert env.moves_used == env.max_moves
        assert total == env.score  # reward is exactly the score, nothing added


def test_set_board_validation():
    env = Match3Env(board_rows=5, board_cols=5, num_gems=5)
    with pytest.raises(ValueError):
        env.set_board(np.zeros((4, 5), dtype=int))  # wrong shape
    with pytest.raises(ValueError):
        env.set_board(np.full((5, 5), 9, dtype=int))  # gem out of range
    matched = LATIN5.copy()
    matched[0, 0:3] = 2
    with pytest.raises(ValueError):
        env.set_board(matched)  # unresolved matches are not a decision state


def test_constructor_validation():
    with pytest.raises(ValueError):
        Match3Env(board_rows=2)
    with pytest.raises(ValueError):
        Match3Env(num_gems=2)
    with pytest.raises(ValueError):
        Match3Env(max_moves=0)


def test_env_kwargs_change_the_game():
    env = Match3Env(board_rows=6, board_cols=7, num_gems=4, max_moves=5)
    obs, _info = env.reset(seed=0)
    assert env.action_space.n == 6 * 7 * 4
    assert obs["board"].shape == (4, 6, 7)
    assert env.board.max() < 4


def test_reshuffle_restores_playability_and_preserves_multiset():
    # Cyclic Latin 4x4: a dead board (verified below, not assumed).
    dead = np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
            [2, 3, 0, 1],
            [3, 0, 1, 2],
        ]
    )
    assert not find_matches(dead).any()
    env = Match3Env(board_rows=4, board_cols=4, num_gems=4)
    env.reset(seed=0)
    env.set_board(dead)
    if not legal_actions_mask(dead).any():
        env._reshuffle()
        assert not find_matches(env.board).any()
        assert legal_actions_mask(env.board).any()
        np.testing.assert_array_equal(
            np.sort(env.board.ravel()), np.sort(dead.ravel())
        )
    else:  # pragma: no cover - pattern turned out playable; nothing to test
        pytest.skip("cyclic Latin 4x4 has a legal move; pick another dead board")
