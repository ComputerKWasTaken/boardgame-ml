"""Calibration tripwire (M10) + the GymClassicEnv shim contract.

The full calibration (``python -m boardgame_ml.eval.calibration``)
trains CartPole at the Zoo's 100k budget and LunarLander at 1M —
far too slow for a test suite. This miniature runs the same code
path (suite definition -> ``train()`` -> independent evaluator) at
a smoke budget and asserts the pipeline still *learns*: a solved
or near-solved CartPole, decisively above the ~22 a random policy
scores. If a future training-code change breaks the plumbing the
way the M8 std bug broke racing PPO, this test goes red at
CI scale instead of two milestones later.

The learning bar is set from measurement, not hope: at 24k steps
the pipeline measured 197 (the steep part of the curve — too
fragile to assert on); at 30k and 40k steps it measured
500.0 ± 0.0 over deterministic episodes on this VM. The test uses
40k steps and demands only > 250 — half the cap, ~10x the random
score, with the whole 250-point gap as machine-variance margin.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym

from boardgame_ml.envs import make_env
from boardgame_ml.eval.calibration import SUITES, run_suite


def test_gym_classic_shim_passthrough():
    env = make_env("gym_classic", env_id="Acrobot-v1")
    raw = gym.make("Acrobot-v1")
    assert env.observation_space == raw.observation_space
    assert env.action_space == raw.action_space
    assert env.env_id == "Acrobot-v1"
    # Default env_id is CartPole (the probe/tripwire env).
    assert make_env("gym_classic").env_id == "CartPole-v1"


def test_calibration_suites_are_well_formed():
    for suite in SUITES.values():
        cfg = suite.build_config()
        assert cfg.game == "gym_classic"
        assert cfg.algorithm == "ppo"
        assert cfg.env_kwargs == {"env_id": suite.env_id}
        assert cfg.total_timesteps == suite.total_timesteps
        assert suite.reference_mean > suite.solved_bar


def test_cartpole_tripwire_learns(tmp_path):
    record = run_suite(
        "cartpole",
        seed=0,
        output_dir=str(tmp_path),
        total_timesteps=40_000,
        report_episodes=50,
    )
    # The pipeline produced a policy far above random play (~22).
    assert record["measured_mean_return"] > 250
    # Seed plumbing (M10 discipline): the run's summary records the
    # seed it was trained with, in the config it serializes.
    summary = json.loads((Path(record["run_dir"]) / "summary.json").read_text())
    assert summary["config"]["seed"] == 0
    assert summary["config"]["env_kwargs"] == {"env_id": "CartPole-v1"}
    assert summary["best_eval_mean_return"] is not None
