"""Environment compliance + action-mask correctness for every registered game."""

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from boardgame_ml.envs import GAMES as _GAME_REGISTRY
from boardgame_ml.envs import make_env
from boardgame_ml.envs.simultaneous import SimultaneousTwoPlayerEnv
from boardgame_ml.training.self_play import SimultaneousSelfPlayWrapper
from boardgame_ml.agents.baselines import get_baseline

# Parameterized from the registry itself: a game added to GAMES gets
# compliance tests automatically, with no second list to remember.
GAMES = sorted(_GAME_REGISTRY)


def _is_discrete(env) -> bool:
    return isinstance(env.action_space, gym.spaces.Discrete)


def _a_step_action(env):
    """One fixed, always-legal action for stepping in generic tests."""
    if _is_discrete(env):
        return env.valid_actions()[0]
    if isinstance(env, SimultaneousTwoPlayerEnv):
        return (1, 1)
    return np.zeros(env.action_space.shape, dtype=np.float32)


@pytest.mark.parametrize("game", GAMES)
def test_check_env(game):
    """The official Gymnasium checker must pass (contract compliance)."""
    env = make_env(game)
    if isinstance(env, SimultaneousTwoPlayerEnv):
        env = SimultaneousSelfPlayWrapper(env, fixed_opponent=get_baseline("pong_intercept", game))
    check_env(env)


@pytest.mark.parametrize("game", GAMES)
def test_action_mask_shape_and_dtype(game):
    env = make_env(game)
    env.reset(seed=0)
    if not _is_discrete(env):
        # Continuous-action games have no mask machinery at all:
        # asking for one is a named error, and info carries no mask.
        with pytest.raises(TypeError):
            env.action_masks()
        return
    mask = env.action_masks()
    assert mask.shape == (env.action_space.n,)
    assert mask.dtype == bool
    if game == "match3":
        # The first game with real illegal actions: most swaps create
        # no match, so some actions must be masked out — and board
        # generation guarantees at least one legal move.
        assert mask.any()
        assert not mask.all()
    else:
        # M1 games never mask anything out; the machinery is still exercised.
        assert mask.all()


@pytest.mark.parametrize("game", GAMES)
def test_mask_consistent_after_steps(game):
    env = make_env(game)
    obs, info = env.reset(seed=1)
    if _is_discrete(env):
        assert "action_mask" in info
    else:
        assert "action_mask" not in info
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(_a_step_action(env))
        if _is_discrete(env):
            np.testing.assert_array_equal(info["action_mask"], env.action_masks())
        else:
            assert "action_mask" not in info
        if terminated or truncated:
            break


def _assert_obs_equal(a, b):
    """Observation equality that understands Dict observations."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    else:
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("game", GAMES)
def test_seeded_reset_is_reproducible(game):
    env_a, env_b = make_env(game), make_env(game)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    _assert_obs_equal(obs_a, obs_b)
    for _ in range(3):
        action = _a_step_action(env_a)
        step_a = env_a.step(action)
        step_b = env_b.step(action)
        _assert_obs_equal(step_a[0], step_b[0])
        assert step_a[1] == step_b[1]
        if step_a[2] or step_a[3]:
            break


def test_render_returns_text():
    for game in GAMES:
        env = make_env(game, render_mode="human")
        env.reset(seed=0)
        text = env.render()
        assert isinstance(text, str) and len(text) > 0
