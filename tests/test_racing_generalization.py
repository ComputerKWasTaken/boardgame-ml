"""Racing generalization (M7): generator v2 validity, the frozen v1
track, per-episode track re-rolls, and the exam harness.

The validity tests recompute the gates with code written for the
tests (a straightforward O(M^2) segment-intersection and a
circumradius pass) rather than trusting the generator's own accept
path — a generator that grades its own homework proves nothing.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from boardgame_ml.envs.racing import (
    RacingEnv,
    _circumradii,
    build_track,
    generate_track,
    generate_track_v2,
    track_shape_stats,
    track_violations,
)
from boardgame_ml.eval.racing_exam import EXAM_TRACK_SEEDS, run_exam

#: The canonical M5 track, fingerprinted before the v2 work began.
SEED7_SHA256 = "2a613e5a982e3de924eeec5fa85020b98f5bb7038cf9682c7236c24bea2c6500"
SEED7_LENGTH = 301.37528762670416


def _bruteforce_self_intersection(points: np.ndarray) -> bool:
    """Independent segment-pair crossing check (test-side, slow)."""
    m = len(points)

    def seg_cross(p1, p2, p3, p4) -> bool:
        d1 = np.cross(p2 - p1, p3 - p1)
        d2 = np.cross(p2 - p1, p4 - p1)
        d3 = np.cross(p4 - p3, p1 - p3)
        d4 = np.cross(p4 - p3, p2 - p3)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    for i in range(m):
        for j in range(i + 3, m):
            if i == 0 and j == m - 1:
                continue  # circular neighbours
            if seg_cross(
                points[i],
                points[(i + 1) % m],
                points[j],
                points[(j + 1) % m],
            ):
                return True
    return False


# -- the frozen v1 track ----------------------------------------------------


def test_seed7_track_is_bit_identical_to_m5():
    track = generate_track(7)
    digest = hashlib.sha256(track.centerline.tobytes()).hexdigest()
    assert digest == SEED7_SHA256
    assert track.total_length == pytest.approx(SEED7_LENGTH, abs=1e-9)
    assert track.n_points == 192
    # ...and through the dispatcher the env actually uses.
    via_build = build_track(7, version=1)
    np.testing.assert_array_equal(via_build.centerline, track.centerline)


# -- generator v2 validity ----------------------------------------------------


@pytest.mark.parametrize("seed", [50001, 50002, 50003, 50004, 50005, 10000, 10001])
def test_v2_tracks_pass_the_gates_as_written(seed):
    track = generate_track_v2(seed)
    assert track_violations(track.centerline) == []
    stats = track_shape_stats(track)
    assert stats["min_turn_radius_m"] >= 13.0
    assert 170.0 <= stats["length_m"] <= 950.0
    assert np.isclose(track.sector_len.sum(), track.total_length)


def test_v2_validity_across_fifty_seeds_independent_check():
    """50 seeds: the generator's own gate list is empty AND the
    test-side brute-force intersection check agrees, AND the
    separation/radius floors hold under test-side measurement."""
    for seed in range(20_000, 20_050):
        track = generate_track_v2(seed)
        pts = track.centerline
        assert track_violations(pts) == []
        assert not _bruteforce_self_intersection(pts)
        assert float(_circumradii(pts).min()) >= 13.0
        m = track.n_points
        idx = np.arange(m)
        along = np.abs(idx[:, None] - idx[None, :])
        along = np.minimum(along, m - along)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        assert d[along > m // 6].min() > 2 * 6.0


def test_v2_generation_is_deterministic_and_varies():
    a = generate_track_v2(4242)
    b = generate_track_v2(4242)
    np.testing.assert_array_equal(a.centerline, b.centerline)
    lengths = set()
    for seed in range(30_000, 30_012):
        t = generate_track_v2(seed)
        lengths.add(round(t.total_length, 3))
        assert not np.array_equal(t.centerline, a.centerline)
    assert len(lengths) >= 10  # seeds really do deal different circuits


def test_v2_tracks_are_not_perturbed_circles():
    """Character check: across a batch, the tightest corner spreads
    over the tight/flowing spectrum and most tracks have several
    distinct corners — the v1 failure mode was every track a
    rounded loop with min radius ~28 m."""
    min_radii = []
    corner_counts = []
    for seed in range(40_000, 40_020):
        stats = track_shape_stats(generate_track_v2(seed))
        min_radii.append(stats["min_turn_radius_m"])
        corner_counts.append(stats["corners_r40"])
    assert min(min_radii) < 17.0  # genuinely tight hairpins exist
    assert max(min_radii) > 21.0  # ...and flowing circuits too
    assert np.median(corner_counts) >= 4


# -- per-episode tracks ---------------------------------------------------------

_RANDOMIZED = dict(
    track_version=2,
    randomize_track=True,
    track_seed_lo=10_000,
    track_seed_hi=11_000,
)


def test_randomized_reset_is_deterministic_per_episode_seed():
    e1, e2 = RacingEnv(**_RANDOMIZED), RacingEnv(**_RANDOMIZED)
    e1.reset(seed=777)
    e2.reset(seed=777)
    assert e1.track_seed == e2.track_seed
    np.testing.assert_array_equal(e1.track.centerline, e2.track.centerline)
    assert (e1.car_x, e1.car_y, e1.heading) == (e2.car_x, e2.car_y, e2.heading)
    assert 10_000 <= e1.track_seed < 11_000


def test_randomized_reset_deals_different_tracks_for_different_seeds():
    env = RacingEnv(**_RANDOMIZED)
    seen = set()
    for seed in range(900, 912):
        env.reset(seed=seed)
        seen.add(env.track_seed)
        # The dealt track is a valid v2 circuit.
        assert track_violations(env.track.centerline) == []
    assert len(seen) >= 10


def test_fixed_mode_never_rerolls_the_track():
    env = RacingEnv(track_version=2, track_seed=50003)
    first = env.track.centerline.copy()
    for seed in range(5):
        env.reset(seed=seed)
        assert env.track_seed == 50003
        np.testing.assert_array_equal(env.track.centerline, first)


def test_randomized_mode_validates_its_range_and_version():
    with pytest.raises(ValueError, match="track_seed_lo"):
        RacingEnv(randomize_track=True, track_seed_lo=100, track_seed_hi=100)
    with pytest.raises(ValueError, match="track_version"):
        RacingEnv(track_version=3)


# -- the exam -------------------------------------------------------------------


def test_exam_harness_bookkeeping():
    result = run_exam(
        ["random"], [50001, 50002], episodes_per_track=3, seed_base=555_000
    )
    assert set(result["tracks"]) == {50001, 50002}
    agent = result["agents"]["random"]
    assert agent["pooled"]["episodes"] == 6
    for track_result in agent["per_track"].values():
        assert track_result["episodes"] == 3
        assert len(track_result["records"]) == 3
        assert 0.0 <= track_result["lap_rate"] <= 1.0
        assert 0.0 <= track_result["crash_rate"] <= 1.0
    # Random barely moves (the M5 finding): single-digit progress.
    assert agent["pooled"]["mean_progress_fraction"] < 0.2
    # The exam set constant is the published one and stays out of
    # the training range by construction.
    assert EXAM_TRACK_SEEDS == (50001, 50002, 50003, 50004, 50005)
    assert all(not (10_000 <= s < 11_000) for s in EXAM_TRACK_SEEDS)
