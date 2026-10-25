"""Racing rules: track geometry, raycasts, physics, progress scheme.

The progress tests matter most: the checkpoint scheme is the game's
reward, and its anti-farming properties (no pay for reversing, no
pay for re-covered ground, no teleport credit) are pinned here
rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from boardgame_ml.agents.baselines import RacingHeuristicAgent
from boardgame_ml.envs.racing import RacingEnv, generate_track

ZERO = np.array([0.0, 0.0], dtype=np.float32)


def _gentlest_point(track) -> int:
    """Centerline index with the least local curvature (for placing
    the car in controlled raycast/physics setups)."""
    ang = np.arctan2(track.tangents[:, 1], track.tangents[:, 0])
    turn = np.abs(np.diff(np.unwrap(np.concatenate([ang, ang[:1]]))))
    return int(np.argmin(turn))


def _place(env: RacingEnv, idx: int, *, offset: float = 0.0, speed: float = 0.0):
    """Put the car exactly on the centerline point ``idx`` (plus a
    lateral offset), heading along the tangent, at ``speed``."""
    track = env.track
    cx, cy = track.centerline[idx]
    tx, ty = track.tangents[idx]
    env.car_x = float(cx - ty * offset)
    env.car_y = float(cy + tx * offset)
    env.heading = float(np.arctan2(ty, tx))
    env.speed = float(speed)
    env._update_nearest(global_search=True)
    env.last_rays = env._raycast()
    return env


# -- track generation -----------------------------------------------------


def test_track_generation_is_reproducible():
    a = generate_track(7)
    b = generate_track(7)
    np.testing.assert_array_equal(a.centerline, b.centerline)
    np.testing.assert_array_equal(a.left_edge, b.left_edge)
    assert a.total_length == b.total_length
    c = generate_track(8)
    assert not np.array_equal(a.centerline, c.centerline)


def test_default_track_is_a_sane_closed_loop():
    track = generate_track(7)
    assert track.n_points == 192
    assert 250.0 < track.total_length < 400.0
    assert (track.seg_len > 0).all()
    # Checkpoints are ordered and cover the lap.
    assert len(track.checkpoint_idx) == 24
    assert track.checkpoint_s[0] == 0.0
    assert np.isclose(track.sector_len.sum(), track.total_length)
    # No hairpins: minimum turn radius comfortably exceeds the
    # half-width (an undrivable kink was the prototype's failure).
    ang = np.arctan2(track.tangents[:, 1], track.tangents[:, 0])
    turn = np.abs(np.diff(np.unwrap(np.concatenate([ang, ang[:1]]))))
    min_radius = track.seg_len.mean() / turn.max()
    assert min_radius >= 15.0
    # Distinct branches stay far apart: with points at least a sixth
    # of the lap away along the loop, edges never overlap.
    c = track.centerline
    m = track.n_points
    idx = np.arange(m)
    for i in range(0, m, 8):
        d = np.linalg.norm(c - c[i], axis=1)
        along = np.minimum(np.abs(idx - i), m - np.abs(idx - i))
        assert d[along > m // 6].min() > 2 * 6.0 + 10.0


# -- raycasts ---------------------------------------------------------------


def test_raycast_centered_car_sees_symmetric_edges():
    env = RacingEnv()
    env.reset(seed=0)
    _place(env, _gentlest_point(env.track))
    rays = env._raycast()
    # Side rays hit the edges one half-width away.
    assert rays[0] == pytest.approx(6.0, abs=0.75)
    assert rays[-1] == pytest.approx(6.0, abs=0.75)
    # The forward ray sees further than the side rays on a gentle bend.
    assert rays[env.n_rays // 2] > rays[0] + 3.0
    assert (rays <= env.ray_max_range).all()


def test_raycast_near_edge_reads_short_on_that_side():
    env = RacingEnv()
    env.reset(seed=0)
    _place(env, _gentlest_point(env.track), offset=5.0)  # 1 m from left edge
    rays = env._raycast()
    assert rays[-1] == pytest.approx(1.0, abs=0.75)  # left ray
    assert rays[0] == pytest.approx(11.0, abs=1.0)  # right ray


# -- physics ------------------------------------------------------------------


def test_throttle_accelerates_and_brake_reverses():
    env = RacingEnv()
    env.reset(seed=0)
    _place(env, _gentlest_point(env.track))
    for _ in range(40):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    v_fast = env.speed
    assert v_fast > 5.0
    # Full brake: speed falls, crosses zero, and reverses — capped.
    for _ in range(200):
        env.step(np.array([0.0, -1.0], dtype=np.float32))
        if env.speed <= -env.reverse_max_speed + 0.01:
            break
    assert env.speed < 0.0
    assert env.speed >= -env.reverse_max_speed - 1e-9


def test_steering_needs_speed_and_scales_with_it():
    env = RacingEnv()
    env.reset(seed=0)
    idx = _gentlest_point(env.track)
    # Stationary: full lock turns nothing (bicycle model).
    _place(env, idx, speed=0.0)
    h0 = env.heading
    for _ in range(10):
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert env.heading == pytest.approx(h0)
    # Rolling: the same steering input turns ~proportionally more at
    # twice the speed (small angles, grip not binding at these speeds).
    turns = []
    for speed in (5.0, 10.0):
        _place(env, idx, speed=speed)
        h0 = env.heading
        for _ in range(3):
            env.step(np.array([0.3, 0.0], dtype=np.float32))
        turns.append(abs(env.heading - h0))
    assert turns[0] > 0.0
    assert turns[1] == pytest.approx(2.0 * turns[0], rel=0.15)


def test_grip_caps_yaw_rate_at_speed():
    env = RacingEnv()
    env.reset(seed=0)
    _place(env, _gentlest_point(env.track), speed=30.0)
    h0 = env.heading
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    yaw = abs(env.heading - h0) / env.dt
    # Bicycle yaw would be 30/2.7 * tan(0.55) ~= 6.8 rad/s; grip caps
    # it at grip / speed = 14 / 30 ~= 0.47 rad/s.
    assert yaw == pytest.approx(env.grip / 30.0, rel=0.05)


def test_extreme_inputs_stay_finite_and_bounded():
    env = RacingEnv()
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    for _ in range(600):
        action = rng.uniform(-5.0, 5.0, size=2).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert abs(env.speed) <= env.max_speed + 1e-9
        if terminated or truncated:
            env.reset(seed=4)
    assert env.action_space.contains(
        np.clip(action, -1, 1).astype(np.float32)
    )


# -- progress: the checkpoint scheme -------------------------------------------


def test_return_decomposes_into_progress_plus_lap_bonuses():
    env = RacingEnv()
    agent = RacingHeuristicAgent()
    obs, _ = env.reset(seed=3011)
    total = 0.0
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(
            agent.select_action(env, obs)
        )
        total += reward
        done = terminated or truncated
    expected = info["progress_m"] + 100.0 * info["laps"]
    assert total == pytest.approx(expected, abs=1e-6)


def test_progress_is_monotonic_and_reversing_earns_nothing():
    env = RacingEnv()
    agent = RacingHeuristicAgent()
    obs, _ = env.reset(seed=3005)
    # Phase 1: heuristic driving banks real progress, monotonically.
    last_progress = 0.0
    for _ in range(400):
        obs, reward, terminated, truncated, info = env.step(
            agent.select_action(env, obs)
        )
        assert info["progress_m"] >= last_progress
        last_progress = info["progress_m"]
        assert not (terminated or truncated)
    assert last_progress > 30.0
    # Phase 2: brake, then reverse. While the car rolls forward to a
    # stop it may bank a little more; once it is actually moving
    # backwards the high-water mark must freeze and rewards must be
    # exactly 0 — reversing can neither un-earn nor re-earn ground.
    frozen = None
    backwards_steps = 0
    for _ in range(300):
        # Heuristic steering (keeps the car on the road) with the
        # throttle pinned to brake/reverse.
        steer = agent.select_action(env, obs)[0]
        obs, reward, terminated, truncated, info = env.step(
            np.array([steer, -1.0], dtype=np.float32)
        )
        assert info["progress_m"] >= last_progress
        last_progress = info["progress_m"]
        if env.speed < -0.5:
            backwards_steps += 1
            if frozen is None:
                frozen = info["progress_m"]
            assert reward == 0.0
            assert info["progress_m"] == frozen
        if terminated or truncated or backwards_steps >= 60:
            break
    assert backwards_steps > 20  # the phase really did reverse
    assert frozen is not None
    assert not terminated
    # Phase 3: drive forward again over the same ground. Until the
    # car passes the frozen high-water mark, rewards stay exactly 0 —
    # re-covered ground pays nothing.
    prepaid = 0.0
    crossed = False
    for _ in range(900):
        obs, reward, terminated, truncated, info = env.step(
            agent.select_action(env, obs)
        )
        if info["progress_m"] > frozen:
            crossed = True
            break
        prepaid += reward
        if terminated or truncated:
            break
    assert crossed
    assert prepaid == 0.0


def test_nearest_search_is_windowed_no_teleport_credit():
    env = RacingEnv()
    env.reset(seed=0)
    idx = env.nearest_idx
    # Teleport the car across the track to a geometrically close but
    # far-along point: the windowed search (what step() uses) must
    # keep the car's track position local — progress can only be
    # earned by driving through the checkpoints in order.
    far = (idx + env.track.n_points // 2) % env.track.n_points
    env.car_x, env.car_y = env.track.centerline[far]
    env._update_nearest()
    jump = abs(env.nearest_idx - idx) % env.track.n_points
    jump = min(jump, env.track.n_points - jump)
    assert jump <= 33  # NEAREST_WINDOW + slack


def test_lap_completion_pays_bonus_and_records_time():
    env = RacingEnv()
    agent = RacingHeuristicAgent()
    obs, _ = env.reset(seed=3002)
    lap_step_reward = None
    done = False
    for _ in range(1500):
        obs, reward, terminated, truncated, info = env.step(
            agent.select_action(env, obs)
        )
        if info["lap_completed"]:
            lap_step_reward = reward
            break
        done = terminated or truncated
        assert not done
    assert lap_step_reward is not None and lap_step_reward >= 100.0
    assert info["laps"] == 1
    assert 5.0 < info["last_lap_time"] < 60.0
    assert info["best_lap_time"] == info["last_lap_time"]


# -- endings ---------------------------------------------------------------------


def test_leaving_the_track_terminates():
    env = RacingEnv()
    env.reset(seed=0)
    _place(env, _gentlest_point(env.track), offset=env.half_width + 2.0)
    _, _, terminated, truncated, info = env.step(ZERO)
    assert terminated and not truncated
    assert info["off_track"] is True


def test_step_cap_truncates_not_terminates():
    env = RacingEnv(max_steps=50)
    env.reset(seed=0)
    for i in range(50):
        _, _, terminated, truncated, _ = env.step(ZERO)
    assert truncated and not terminated


# -- the Box contract -------------------------------------------------------------


def test_box_env_has_no_mask_machinery():
    env = RacingEnv()
    _, info = env.reset(seed=0)
    assert "action_mask" not in info
    with pytest.raises(TypeError):
        env.action_masks()
    with pytest.raises(TypeError):
        env.valid_actions()


def test_spawn_is_seedable_and_varies():
    positions = set()
    for seed in range(6):
        env = RacingEnv()
        env.reset(seed=seed)
        positions.add((round(env.car_x, 6), round(env.car_y, 6)))
    assert len(positions) > 1  # spawns differ across seeds
    a, b = RacingEnv(), RacingEnv()
    a.reset(seed=42)
    b.reset(seed=42)
    assert (a.car_x, a.car_y, a.heading) == (b.car_x, b.car_y, b.heading)
