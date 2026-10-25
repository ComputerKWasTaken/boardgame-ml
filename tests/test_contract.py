"""Contract tests for the mask machinery — the framework's founding trauma.

The shipped M1 games never mask anything, so without a purpose-built
dummy env the mask contract would only ever be tested where it cannot
fail. ``DummyMaskedEnv`` has *state-dependent* legality (exactly one
action illegal at a time, rotating with the step counter) and a
5-step episode limit reported as a truncation. Everything the
framework promises about masks is exercised against it here:

* gymnasium ``check_env`` compliance;
* ``info["action_mask"]`` always equals ``env.action_masks()``,
  including across episode boundaries;
* MaskablePPO, wired exactly as the trainer wires it, never executes
  a masked-out action;
* eval's ``predict_action`` respects the mask;
* the train/eval mask guards refuse mask-blind algorithms/checkpoints
  on a game that masks;
* ``validate_mask`` coerces benign inputs and rejects bad ones loudly.
"""

from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from boardgame_ml.envs import GAMES, make_env
from boardgame_ml.envs.base import BaseGameEnv, validate_mask
from boardgame_ml.eval.evaluate import predict_action, resolve_agent
from boardgame_ml.training.algorithms import ensure_mask_compatible, game_uses_masks
from boardgame_ml.training.train import mask_fn


class DummyMaskedEnv(BaseGameEnv):
    """Three actions; the illegal one rotates with the step counter.

    Episodes end by *truncation* after ``EPISODE_LENGTH`` steps.
    Illegal actions are tolerated (and recorded) so ``check_env`` and
    unmasked probes can step freely — the recording is the test
    instrumentation for "did anything ever execute a masked action".
    """

    EPISODE_LENGTH = 5

    def __init__(self, render_mode=None, executed=None, violations=None):
        super().__init__(render_mode=render_mode)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._steps = 0
        self.executed = executed if executed is not None else []
        self.violations = violations if violations is not None else []

    def action_masks(self) -> np.ndarray:
        mask = np.ones(3, dtype=bool)
        mask[self._steps % 3] = False
        return mask

    def _obs(self) -> np.ndarray:
        return np.array([self._steps / self.EPISODE_LENGTH], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        if not self.action_masks()[action]:
            self.violations.append(action)
        self.executed.append(action)
        self._steps += 1
        truncated = self._steps >= self.EPISODE_LENGTH
        return self._obs(), 0.0, False, bool(truncated), self._info()

    def _render_text(self) -> str:
        return f"DummyMasked | step {self._steps} | mask {self.action_masks()}"


@pytest.fixture()
def dummy_registered(monkeypatch):
    """Register the dummy in the game registry for the guard tests."""
    monkeypatch.setitem(GAMES, "dummy_masked", DummyMaskedEnv)
    return "dummy_masked"


# -- validate_mask ------------------------------------------------------


@pytest.mark.parametrize(
    "good",
    [
        [True, False, True],
        [1, 0, 1],
        np.array([1.0, 0.0, 1.0]),
        np.array([True, True, False]),
        (False, True, True),
    ],
)
def test_validate_mask_coerces_benign_inputs(good):
    out = validate_mask(good, 3)
    assert isinstance(out, np.ndarray)
    assert out.dtype == bool
    assert out.shape == (3,)
    assert out.any()


@pytest.mark.parametrize(
    "bad",
    [
        [True, False],  # wrong shape
        np.array([0.5, 1.0, 0.0]),  # non-binary values
        np.array([False, False, False]),  # no legal action
        ["yes", "no", "yes"],  # nonsense dtype
    ],
)
def test_validate_mask_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        validate_mask(bad, 3)


# -- the dummy env against the contract ---------------------------------


def test_dummy_check_env():
    check_env(DummyMaskedEnv())


def test_dummy_mask_rotates_and_is_state_dependent():
    env = DummyMaskedEnv()
    env.reset(seed=0)
    forbidden = set()
    for _ in range(DummyMaskedEnv.EPISODE_LENGTH):
        mask = env.action_masks()
        assert mask.dtype == bool and mask.shape == (3,)
        assert int((~mask).sum()) == 1  # exactly one action illegal
        forbidden.add(int(np.flatnonzero(~mask)[0]))
        env.step(env.valid_actions()[0])
    assert forbidden == {0, 1, 2}  # rotation visited every action


def test_dummy_episode_limit_is_truncation_not_termination():
    env = DummyMaskedEnv()
    env.reset(seed=0)
    for _ in range(DummyMaskedEnv.EPISODE_LENGTH - 1):
        _, _, terminated, truncated, _ = env.step(env.valid_actions()[0])
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(env.valid_actions()[0])
    assert truncated and not terminated


def test_info_mask_matches_method_across_episode_boundaries():
    env = DummyMaskedEnv()
    _, info = env.reset(seed=0)
    np.testing.assert_array_equal(info["action_mask"], env.action_masks())
    for ep in range(3):
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(env.valid_actions()[0])
            np.testing.assert_array_equal(info["action_mask"], env.action_masks())
            done = terminated or truncated
        _, info = env.reset(seed=ep + 1)
        np.testing.assert_array_equal(info["action_mask"], env.action_masks())


# -- MaskablePPO through the trainer's own wiring ------------------------


def test_maskable_ppo_never_executes_a_masked_action():
    executed: list[int] = []
    violations: list[int] = []

    def _init():
        env = DummyMaskedEnv(executed=executed, violations=violations)
        return ActionMasker(Monitor(env), mask_fn)

    vec = DummyVecEnv([_init])
    model = MaskablePPO(
        "MlpPolicy", vec, n_steps=64, batch_size=64, seed=0, verbose=0
    )
    model.learn(total_timesteps=300)
    vec.close()
    assert len(executed) >= 300
    assert violations == []


def test_predict_action_respects_dummy_mask():
    env = DummyMaskedEnv()
    model = MaskablePPO("MlpPolicy", env, n_steps=64, seed=0, verbose=0)
    for advance in range(DummyMaskedEnv.EPISODE_LENGTH):
        probe = DummyMaskedEnv()
        obs, _ = probe.reset(seed=0)
        for _ in range(advance):
            obs, *_ = probe.step(probe.valid_actions()[0])
        action = predict_action(model, probe, obs)
        assert probe.action_masks()[action]


# -- the mask guards ------------------------------------------------------


def test_game_uses_masks_probe(dummy_registered):
    assert game_uses_masks("dummy_masked") is True
    assert game_uses_masks("blackjack") is False
    assert game_uses_masks("higher_or_lower") is False


def test_train_guard_rejects_mask_blind_algorithm(dummy_registered):
    with pytest.raises(ValueError, match="masks out actions"):
        ensure_mask_compatible("dummy_masked", "ppo")
    with pytest.raises(ValueError, match="allow_unmasked"):
        ensure_mask_compatible("dummy_masked", "dqn")
    # Mask-aware algorithm, and explicit override: both fine.
    ensure_mask_compatible("dummy_masked", "maskable_ppo")
    ensure_mask_compatible("dummy_masked", "ppo", allow_unmasked=True)


def test_eval_guard_rejects_mask_blind_checkpoint(dummy_registered, tmp_path):
    ppo = PPO("MlpPolicy", DummyMaskedEnv(), verbose=0)
    ckpt = str(tmp_path / "ppo_dummy")
    ppo.save(ckpt)
    with pytest.raises(ValueError, match="masks out actions"):
        resolve_agent(ckpt, "dummy_masked")
    resolved = resolve_agent(ckpt, "dummy_masked", allow_unmasked=True)
    assert isinstance(resolved, PPO)


def test_eval_guard_accepts_maskable_checkpoint(dummy_registered, tmp_path):
    mppo = MaskablePPO("MlpPolicy", DummyMaskedEnv(), n_steps=64, verbose=0)
    ckpt = str(tmp_path / "mppo_dummy")
    mppo.save(ckpt)
    assert isinstance(resolve_agent(ckpt, "dummy_masked"), MaskablePPO)


def test_registered_games_constructible_via_make_env():
    # Sanity that the registry parametrization elsewhere rests on.
    for name in GAMES:
        assert make_env(name) is not None


# -- checkpoint <-> game consistency (hardening fix 7) --------------------


def test_checkpoint_space_mismatch_is_a_named_error(tmp_path):
    # A blackjack checkpoint (obs (3,), 2 actions) resolved against
    # higher_or_lower (obs (3,), 3 actions) must fail at the boundary
    # naming both spaces — not as a torch shape error inside SB3.
    mppo = MaskablePPO("MlpPolicy", make_env("blackjack"), n_steps=64, verbose=0)
    ckpt = str(tmp_path / "blackjack_ckpt")
    mppo.save(ckpt)
    with pytest.raises(ValueError, match="observation space"):
        resolve_agent(ckpt, "higher_or_lower")
    # The correct pairing resolves fine.
    assert isinstance(resolve_agent(ckpt, "blackjack"), MaskablePPO)
