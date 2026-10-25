"""Config plumbing tests — including THE regression test.

Legacy-project autopsy, finding #2: the old PPO agent read its
hyperparameters from a YAML section that did not exist, so every tuning
edit was silently ignored and the agent always ran on library defaults.
This test writes a YAML file with a deliberately absurd learning rate
and asserts it shows up in the constructed model's optimizer. If config
plumbing ever breaks again, this test goes red.
"""

import gymnasium as gym
import numpy as np
import pytest

from boardgame_ml.training.config import ExperimentConfig
from boardgame_ml.training.train import build_model, build_single_env


class _ImageObsEnv(gym.Env):
    """Minimal env with an image-shaped observation, for CnnPolicy tests."""

    def __init__(self) -> None:
        # 84x84: SB3's default NatureCNN stack needs >= ~36x36 inputs.
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "game: blackjack\nalgorithm: dqn\ntotal_timesteps: 1234\n"
        "seed: 7\neval_episodes: 99\n"
    )
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.game == "blackjack"
    assert cfg.algorithm == "dqn"
    assert cfg.total_timesteps == 1234
    assert cfg.seed == 7
    assert cfg.eval_episodes == 99


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("game: blackjack\nlerning_rate: 0.1\n")  # typo must not pass
    with pytest.raises(ValueError, match="Unknown config keys"):
        ExperimentConfig.from_yaml(path)


def test_learning_rate_reaches_model(tmp_path):
    """REGRESSION: a non-default learning_rate in YAML must reach the model."""
    path = tmp_path / "lr.yaml"
    path.write_text(
        "game: blackjack\nalgorithm: maskable_ppo\nlearning_rate: 0.123\n"
    )
    cfg = ExperimentConfig.from_yaml(path)
    model = build_model(cfg, build_single_env("blackjack"))
    optimizer_lrs = [group["lr"] for group in model.policy.optimizer.param_groups]
    assert len(optimizer_lrs) >= 1
    assert all(lr == pytest.approx(0.123) for lr in optimizer_lrs)
    # And the schedule agrees (initial value at progress 1.0).
    assert model.lr_schedule(1.0) == pytest.approx(0.123)


def test_parameterization_fields_roundtrip(tmp_path):
    """env_kwargs / policy / policy_kwargs load from YAML and serialize."""
    path = tmp_path / "params.yaml"
    path.write_text(
        "game: blackjack\n"
        "policy: CnnPolicy\n"
        "policy_kwargs:\n  features_dim: 64\n"
        "env_kwargs:\n  num_players: 2\n"
    )
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.policy == "CnnPolicy"
    assert cfg.policy_kwargs == {"features_dim": 64}
    assert cfg.env_kwargs == {"num_players": 2}
    as_dict = cfg.to_dict()
    assert as_dict["policy"] == "CnnPolicy"
    assert as_dict["policy_kwargs"] == {"features_dim": 64}
    assert as_dict["env_kwargs"] == {"num_players": 2}


def test_cnn_policy_reaches_model(tmp_path):
    """A YAML naming CnnPolicy must construct a CNN policy class.

    Before the parameterization fix the policy was hardcoded to
    MlpPolicy in the trainer, so a board-shaped game could not be
    trained without editing trainer code.
    """
    path = tmp_path / "cnn.yaml"
    path.write_text("game: blackjack\nalgorithm: maskable_ppo\npolicy: CnnPolicy\n")
    cfg = ExperimentConfig.from_yaml(path)
    model = build_model(cfg, _ImageObsEnv())
    # sb3-contrib names its variant MaskableActorCriticCnnPolicy; what
    # matters is that the CNN policy class was constructed, not the MLP.
    assert "Cnn" in type(model.policy).__name__


# -- Match 3 parameterization (M3): the YAML is the whole integration --


def test_match3_shipped_config_loads():
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "match3_maskable_ppo.yaml"
    )
    cfg = ExperimentConfig.from_yaml(path)
    assert cfg.game == "match3"
    assert cfg.algorithm == "maskable_ppo"
    assert cfg.policy == "MultiInputPolicy"
    assert cfg.env_kwargs == {
        "board_rows": 8,
        "board_cols": 8,
        "num_gems": 6,
        "max_moves": 20,
    }


def test_match3_policy_and_env_kwargs_reach_construction():
    """The M3 proof: MultiInputPolicy + env_kwargs arrive from YAML.

    Before the parameterization fix, the trainer hardcoded MlpPolicy
    and games took no constructor kwargs, so a Dict-observation game
    could not be trained at all without editing trainer code.
    """
    cfg = ExperimentConfig(
        game="match3",
        algorithm="maskable_ppo",
        policy="MultiInputPolicy",
        env_kwargs={"board_rows": 6, "board_cols": 7, "num_gems": 4, "max_moves": 9},
    )
    env = build_single_env(cfg.game, env_kwargs=cfg.env_kwargs)
    raw = env.unwrapped
    assert (raw.board_rows, raw.board_cols, raw.num_gems, raw.max_moves) == (
        6,
        7,
        4,
        9,
    )
    model = build_model(cfg, env)
    assert "MultiInput" in type(model.policy).__name__


# -- fail-fast validation (hardening fix 7) ------------------------------


def test_unknown_game_is_rejected():
    with pytest.raises(ValueError, match="Unknown game 'chess'"):
        ExperimentConfig(game="chess")


def test_unknown_algorithm_is_rejected_by_name():
    with pytest.raises(ValueError, match="Unknown algorithm 'a2c'"):
        ExperimentConfig(game="blackjack", algorithm="a2c")


@pytest.mark.parametrize(
    "field_name", ["total_timesteps", "n_envs", "eval_freq", "eval_episodes"]
)
def test_non_positive_counts_are_rejected(field_name):
    with pytest.raises(ValueError, match=f"'{field_name}' must be positive"):
        ExperimentConfig(game="blackjack", **{field_name: 0})


def test_hyperparams_shadowing_is_rejected():
    with pytest.raises(ValueError, match="shadow top-level config fields"):
        ExperimentConfig(game="blackjack", hyperparams={"learning_rate": 0.1})
    with pytest.raises(ValueError, match="shadow top-level config fields"):
        ExperimentConfig(game="blackjack", hyperparams={"policy_kwargs": {}})
