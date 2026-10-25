"""Flappy Bird rules: physics, collisions, scoring, endings.

State is scripted through the env's public attributes (bird_y,
bird_vy, pipes) — the same seam the blackjack tests use with hands.
"""

import numpy as np
import pytest

from boardgame_ml.envs.flappy import (
    BIRD_RADIUS,
    DISTANCE_REWARD_PER_STEP,
    FLAP,
    GLIDE,
    PIPE_REWARD,
    WORLD_HEIGHT,
    FlappyBirdEnv,
    Pipe,
)


def _env(**kwargs) -> FlappyBirdEnv:
    env = FlappyBirdEnv(**kwargs)
    env.reset(seed=0)
    return env


def _pipe(x, gap_bottom, gap_top) -> Pipe:
    return Pipe(x=float(x), gap_bottom=float(gap_bottom), gap_top=float(gap_top))


# -- physics -----------------------------------------------------------------


def test_gravity_integrates_downward():
    env = _env()
    for _ in range(4):
        env.step(GLIDE)
    # Semi-implicit Euler: vy gains gravity*dt per step, position
    # integrates the *new* velocity.
    assert env.bird_vy == pytest.approx(-200.0)  # -1000 * 0.05 * 4
    expected_y = WORLD_HEIGHT / 2 - (50 + 100 + 150 + 200) * 0.05
    assert env.bird_y == pytest.approx(expected_y)


def test_flap_sets_upward_velocity():
    env = _env()
    env.step(FLAP)
    # The flap *sets* vy to flap_velocity (300); one integration step
    # of gravity then applies: 300 - 1000*0.05 = 250, and the bird
    # rises by 250*0.05.
    assert env.bird_vy == pytest.approx(250.0)
    assert env.bird_y == pytest.approx(WORLD_HEIGHT / 2 + 12.5)


def test_flap_replaces_rather_than_adds_to_velocity():
    env = _env()
    env.bird_vy = -500.0  # falling fast
    env.step(FLAP)
    assert env.bird_vy == pytest.approx(250.0)  # not -500 + 300 - 50


def test_physics_substeps_refine_integration_not_decisions():
    env = _env(physics_substeps=4)
    env.step(GLIDE)
    # Total velocity change over the step is identical; the position
    # integral is the finer (trapezoid-like) one.
    assert env.bird_vy == pytest.approx(-50.0)
    assert env.bird_y == pytest.approx(WORLD_HEIGHT / 2 - 1.5625)


def test_pipes_scroll_at_pipe_speed():
    env = _env()
    x0 = env.next_pipe().x
    env.step(GLIDE)
    assert env.next_pipe().x == pytest.approx(x0 - 120.0 * 0.05)


def test_first_pipe_spawns_at_a_fair_distance():
    env = _env()
    pipe = env.next_pipe()
    # Right edge of the world + the documented run-up offset.
    assert pipe.x == pytest.approx(288.0 + 40.0)


# -- scoring -------------------------------------------------------------------


def test_distance_reward_is_paid_per_step_survived():
    env = _env()
    _obs, reward, terminated, _tr, _info = env.step(GLIDE)
    assert not terminated
    assert reward == pytest.approx(DISTANCE_REWARD_PER_STEP)


def test_pipe_scores_exactly_when_its_right_edge_passes_the_bird():
    env = _env()
    # Gap centred on the bird's altitude so the pass is clean.
    env.pipes = [_pipe(x=2.0, gap_bottom=206, gap_top=306)]
    # Step 1: right edge lands exactly on the bird's x (2-6+52 = 48):
    # not yet past, so no point — reward is the distance term alone.
    _o, reward, terminated, _t, info = env.step(GLIDE)
    assert not terminated
    assert env.score == 0 and info["score"] == 0
    assert reward == pytest.approx(DISTANCE_REWARD_PER_STEP)
    # Step 2: the edge is past (42 < 48): exactly one pipe point.
    _o, reward, terminated, _t, info = env.step(GLIDE)
    assert not terminated
    assert env.score == 1 and info["score"] == 1
    assert reward == pytest.approx(PIPE_REWARD + DISTANCE_REWARD_PER_STEP)
    # Step 3: a scored pipe never pays twice.
    _o, reward, _t2, _tr2, _i = env.step(GLIDE)
    assert env.score == 1
    assert reward == pytest.approx(DISTANCE_REWARD_PER_STEP)


# -- collisions ------------------------------------------------------------------


def test_pipe_collision_terminates():
    env = _env()
    # Gap low on the screen; the bird flies in the solid top section.
    env.pipes = [_pipe(x=40.0, gap_bottom=100, gap_top=200)]
    _o, _r, terminated, _t, _i = env.step(GLIDE)
    assert terminated


def test_gap_pass_through_survives_and_scores():
    env = _env()
    env.pipes = [_pipe(x=100.0, gap_bottom=206, gap_top=306)]
    for _ in range(25):
        # Pin the bird at gap centre each step (the public-state
        # seam) so the test isolates collision/scoring, not control.
        env.bird_y = 256.0
        env.bird_vy = 0.0
        _o, _r, terminated, _t, _i = env.step(GLIDE)
        assert not terminated
    assert env.score == 1


def test_ground_collision_terminates_and_bird_rests_on_the_boundary():
    env = _env()
    env.bird_y = 13.0
    env.bird_vy = -100.0
    _o, _r, terminated, _t, _i = env.step(GLIDE)
    assert terminated
    assert env.bird_y == pytest.approx(BIRD_RADIUS)


def test_ceiling_collision_terminates():
    env = _env()
    env.bird_y = WORLD_HEIGHT - 5.0
    _o, _r, terminated, _t, _i = env.step(FLAP)
    assert terminated
    assert env.bird_y == pytest.approx(WORLD_HEIGHT - BIRD_RADIUS)


def test_graze_at_exactly_one_radius_is_safe():
    env = _env()
    # Pipe's left edge exactly one radius from the bird's centre
    # horizontally, bird level with the top solid section: the
    # closest-point distance equals the radius, which is not a hit.
    env.pipes = [_pipe(x=48.0 + BIRD_RADIUS + 6.0, gap_bottom=100, gap_top=200)]
    # After one scroll step the edge is at 48 + radius exactly.
    _o, _r, terminated, _t, _i = env.step(GLIDE)
    assert not terminated


# -- endings -----------------------------------------------------------------------


def test_step_cap_truncates_rather_than_terminates():
    env = _env(max_steps=5)
    for i in range(5):
        _o, _r, terminated, truncated, _i = env.step(GLIDE)
        assert not terminated  # still airborne, far from any pipe
        assert truncated == (i == 4)


def test_invalid_constructor_kwargs_fail_fast():
    with pytest.raises(ValueError):
        FlappyBirdEnv(dt=0)
    with pytest.raises(ValueError):
        FlappyBirdEnv(physics_substeps=0)
    with pytest.raises(ValueError):
        FlappyBirdEnv(pipe_gap=10_000)
    with pytest.raises(ValueError):
        FlappyBirdEnv(pipe_interval=0)
    with pytest.raises(ValueError):
        FlappyBirdEnv(max_steps=0)


def test_observation_stays_in_space_through_a_crash():
    env = _env()
    env.bird_y = 20.0
    env.bird_vy = -400.0
    obs, _r, terminated, _t, _i = env.step(GLIDE)
    assert terminated
    assert env.observation_space.contains(obs)
