"""Environment compliance + action-mask correctness for every registered game."""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from boardgame_ml.envs import GAMES as _GAME_REGISTRY
from boardgame_ml.envs import make_env

# Parameterized from the registry itself: a game added to GAMES gets
# compliance tests automatically, with no second list to remember.
GAMES = sorted(_GAME_REGISTRY)


@pytest.mark.parametrize("game", GAMES)
def test_check_env(game):
    """The official Gymnasium checker must pass (contract compliance)."""
    check_env(make_env(game))


@pytest.mark.parametrize("game", GAMES)
def test_action_mask_shape_and_dtype(game):
    env = make_env(game)
    env.reset(seed=0)
    mask = env.action_masks()
    assert mask.shape == (env.action_space.n,)
    assert mask.dtype == bool
    # M1 games never mask anything out; the machinery is still exercised.
    assert mask.all()


@pytest.mark.parametrize("game", GAMES)
def test_mask_consistent_after_steps(game):
    env = make_env(game)
    obs, info = env.reset(seed=1)
    assert "action_mask" in info
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.valid_actions()[0])
        np.testing.assert_array_equal(info["action_mask"], env.action_masks())
        if terminated or truncated:
            break


@pytest.mark.parametrize("game", GAMES)
def test_seeded_reset_is_reproducible(game):
    env_a, env_b = make_env(game), make_env(game)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)
    for _ in range(3):
        step_a = env_a.step(0)
        step_b = env_b.step(0)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        assert step_a[1] == step_b[1]
        if step_a[2] or step_a[3]:
            break


def test_render_returns_text():
    for game in GAMES:
        env = make_env(game, render_mode="human")
        env.reset(seed=0)
        text = env.render()
        assert isinstance(text, str) and len(text) > 0
