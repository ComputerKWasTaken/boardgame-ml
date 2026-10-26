"""Behavior cloning for Pong: labeling correctness and warm-start loading.

The two properties the BC experiment stands on:

1. Every logged (observation, action) pair is labeled with the
   teacher's action *for that exact per-seat observation* — seat 1's
   pairs carry seat 1's horizontally-reflected observation and the
   action the teacher picks from it. A reflection bug here would teach
   the clone a mirrored, self-defeating policy and invalidate the
   whole experiment, so it is tested both end-to-end (regenerate a
   small dataset and re-derive every label) and as a state property
   (a mirrored position yields the seat-1 observation, bit for bit).
2. A BC-trained checkpoint is an ordinary SB3 checkpoint: the
   trainer's ``init_checkpoint`` warm-start path loads it and the new
   config's scalars (learning rate) override the donor's.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from boardgame_ml.agents.baselines import get_baseline
from boardgame_ml.envs.pong import PongEnv
from boardgame_ml.training.behavior_cloning import (
    generate_demonstrations,
    train_behavior_clone,
)
from boardgame_ml.training.config import ExperimentConfig
from boardgame_ml.training.self_play import SelfPlayConfig
from boardgame_ml.training.train import build_model, build_vec_env

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_CONFIG = REPO_ROOT / "configs" / "pong_ppo.yaml"


@pytest.fixture(scope="module")
def tiny_dataset():
    arrays, _stats = generate_demonstrations(target_pairs=400, seed=424_242)
    return arrays


def test_generated_labels_are_teacher_actions(tiny_dataset):
    """Every pair — both seats, every source bucket — re-derives."""
    teacher = get_baseline("pong_intercept", "pong")
    env = PongEnv()
    views = (env.player_view(0), env.player_view(1))
    obs, actions, seats = (
        tiny_dataset["obs"],
        tiny_dataset["actions"],
        tiny_dataset["seats"],
    )
    assert len(actions) > 0
    assert set(np.unique(seats)) == {0, 1}  # both seats are represented
    for i in range(len(actions)):
        seat = int(seats[i])
        expected = int(teacher.select_action(views[seat], obs[i]))
        assert expected == int(actions[i]), (
            f"pair {i} (seat {seat}): logged {int(actions[i])}, "
            f"teacher says {expected}"
        )
    env.close()


def test_seat_reflection_mirrored_state():
    """observe(1) of a state == observe(0) of its horizontal mirror,
    and the teacher answers both views with the same action."""
    env = PongEnv()
    env.reset(seed=7)
    env.ball_x, env.ball_y = 700.0, 150.0
    env.ball_vx, env.ball_vy = -300.0, 120.0
    env.paddle_y = [200.0, 400.0]
    env.paddle_v = [50.0, -80.0]
    env.scores = [2, 1]
    env.inventory = [1, 3]
    env.effects = [[3.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    env.last_touch = 0
    env.pickup = [2, 300.0, 200.0, 4.0]
    env.elapsed = 30.0
    env.serve_remaining = 0.2
    env.spawn_remaining = 3.0
    seat1_obs = env.observe(1)

    mirror = PongEnv()
    mirror.reset(seed=7)
    mirror.ball_x, mirror.ball_y = env.width - 700.0, 150.0
    mirror.ball_vx, mirror.ball_vy = 300.0, 120.0
    mirror.paddle_y = [400.0, 200.0]
    mirror.paddle_v = [-80.0, 50.0]
    mirror.scores = [1, 2]
    mirror.inventory = [3, 1]
    mirror.effects = [[0.0, 0.0, 2.0], [3.0, 0.0, 0.0]]
    mirror.last_touch = 1
    mirror.pickup = [2, env.width - 300.0, 200.0, 4.0]
    mirror.elapsed = 30.0
    mirror.serve_remaining = 0.2
    mirror.spawn_remaining = 3.0
    seat0_obs = mirror.observe(0)

    np.testing.assert_allclose(seat0_obs, seat1_obs, atol=1e-6)
    teacher = get_baseline("pong_intercept", "pong")
    assert teacher.select_action(
        mirror.player_view(0), seat0_obs
    ) == teacher.select_action(env.player_view(1), seat1_obs)
    env.close()
    mirror.close()


def test_bc_checkpoint_loads_via_init_checkpoint(tiny_dataset, tmp_path):
    """BC-train briefly, save, and warm-start through build_model."""
    config = ExperimentConfig.from_yaml(SHIPPED_CONFIG)
    model, report = train_behavior_clone(
        config, tiny_dataset, epochs=1, batch_size=512, bc_learning_rate=1e-3
    )
    assert report["val_pairs"] > 0
    assert 0.0 <= report["per_epoch"][0]["val_accuracy"] <= 1.0
    ckpt = tmp_path / "bc_tiny"
    model.save(str(ckpt))

    fine_tune = replace(
        config, init_checkpoint=str(ckpt.with_suffix(".zip")), learning_rate=1e-4
    )
    self_play_cfg = SelfPlayConfig.from_dict(fine_tune.self_play)
    env = build_vec_env(
        fine_tune.game,
        fine_tune.n_envs,
        fine_tune.seed,
        env_kwargs=fine_tune.env_kwargs,
        self_play_cfg=self_play_cfg,
    )
    try:
        loaded = build_model(fine_tune, env)
        # The warm-start path accepted the BC checkpoint, and the new
        # config's learning rate overrode the donor's.
        assert float(loaded.learning_rate) == pytest.approx(1e-4)
        action, _ = loaded.predict(tiny_dataset["obs"][0], deterministic=True)
        assert 0 <= int(action) < 6
    finally:
        env.close()
