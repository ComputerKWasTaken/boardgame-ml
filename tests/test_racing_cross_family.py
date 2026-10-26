"""Racing cross-family transfer (M9): generator v3 validity and
character, and the cross-family exam's bookkeeping.

Generator v3 builds circuits the way tracks are drawn — straights
joined by constant-radius arcs, closed by an exact solve — behind
the *same* validity gates as v2. As with the M7 tests, the gates
are re-checked here with test-side code (a brute-force segment
intersection and independent radius/separation passes), not by
trusting the generator's own accept path.
"""

from __future__ import annotations

import numpy as np
import pytest

from boardgame_ml.envs.racing import (
    RacingEnv,
    _circumradii,
    build_track,
    generate_track_v2,
    generate_track_v3,
    track_violations,
)
from boardgame_ml.eval.racing_exam import run_exam

#: The held-out M9 exam set: generator-v3 seeds in neither the v3
#: training range [70000, 71000) nor any other run's range (the M7
#: exam is 50001–50005; the v2 training range is [10000, 11000)).
EXAM_V3_SEEDS = (60001, 60002, 60003, 60004, 60005)


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


def _straight_fraction(track) -> float:
    """Fraction of centerline samples on (numerically) straight
    elements: circumradius beyond 200 m. This is the v3 family's
    signature — piecewise-constant curvature with long exact zeros —
    where v2's spline curvature is merely *small* in places."""
    return float((_circumradii(track.centerline) > 200.0).mean())


# -- generator v3 validity ----------------------------------------------------


@pytest.mark.parametrize("seed", [*EXAM_V3_SEEDS, 70000, 70001])
def test_v3_tracks_pass_the_gates_as_written(seed):
    track = generate_track_v3(seed)
    assert track_violations(track.centerline) == []
    assert float(_circumradii(track.centerline).min()) >= 13.0
    assert 170.0 <= track.total_length <= 950.0
    assert np.isclose(track.sector_len.sum(), track.total_length)


def test_v3_validity_across_fifty_seeds_independent_check():
    """50 seeds: the generator's gate list is empty AND the
    test-side brute-force intersection check agrees, AND the
    separation/radius floors hold under test-side measurement."""
    for seed in range(90_000, 90_050):
        track = generate_track_v3(seed)
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


def test_v3_generation_is_deterministic_and_varies():
    a = generate_track_v3(60003)
    b = generate_track_v3(60003)
    np.testing.assert_array_equal(a.centerline, b.centerline)
    via_build = build_track(60003, version=3)
    np.testing.assert_array_equal(via_build.centerline, a.centerline)
    lengths = set()
    for seed in range(95_000, 95_012):
        t = generate_track_v3(seed)
        lengths.add(round(t.total_length, 3))
        assert not np.array_equal(t.centerline, a.centerline)
    assert len(lengths) >= 10  # seeds really do deal different circuits


def test_unknown_track_version_still_rejected():
    with pytest.raises(ValueError, match="Unknown track generator version"):
        build_track(1, version=4)


# -- family character: v3 is not v2 --------------------------------------------


def test_v3_family_character_differs_from_v2():
    """Same seed batch through both generators: v3's centerlines
    spend roughly half their samples on exactly-straight elements,
    v2's splines almost none — the families are measurably
    different constructions, not one recipe re-seeded. (Batch is
    seeded, hence deterministic; thresholds sit far inside the
    measured ranges on both sides.)"""
    v3_fracs, v2_fracs, v3_lengths = [], [], []
    for seed in range(80_000, 80_020):
        t3 = generate_track_v3(seed)
        t2 = generate_track_v2(seed)
        v3_fracs.append(_straight_fraction(t3))
        v2_fracs.append(_straight_fraction(t2))
        v3_lengths.append(t3.total_length)
    assert min(v3_fracs) > 0.30
    assert max(v2_fracs) < 0.30
    assert float(np.median(v3_fracs)) >= 0.40
    # v3's straights stretch its circuits: median length beyond
    # v2's 200-seed median (358 m) by a wide margin.
    assert float(np.median(v3_lengths)) > 420.0


# -- env integration ------------------------------------------------------------


def test_env_runs_on_v3_tracks():
    env = RacingEnv(track_version=3, track_seed=60002)
    assert env.track_seed == 60002
    obs, _info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    for _ in range(50):
        obs, _r, terminated, truncated, _i = env.step(
            env.action_space.sample()
        )
        if terminated or truncated:
            break
    assert obs.shape == env.observation_space.shape


def test_randomized_v3_reset_is_deterministic_and_valid():
    kwargs = dict(
        track_version=3,
        randomize_track=True,
        track_seed_lo=70_000,
        track_seed_hi=71_000,
    )
    e1, e2 = RacingEnv(**kwargs), RacingEnv(**kwargs)
    e1.reset(seed=4242)
    e2.reset(seed=4242)
    assert e1.track_seed == e2.track_seed
    np.testing.assert_array_equal(e1.track.centerline, e2.track.centerline)
    env = RacingEnv(**kwargs)
    for seed in range(300, 308):
        env.reset(seed=seed)
        assert 70_000 <= env.track_seed < 71_000
        assert track_violations(env.track.centerline) == []


# -- the cross-family exam --------------------------------------------------------


def test_cross_family_exam_bookkeeping():
    result = run_exam(
        ["random"],
        [60001, 60002],
        episodes_per_track=3,
        seed_base=555_000,
        track_version=3,
    )
    assert result["track_version"] == 3
    assert set(result["tracks"]) == {60001, 60002}
    agent = result["agents"]["random"]
    assert agent["pooled"]["episodes"] == 6
    for track_result in agent["per_track"].values():
        assert track_result["episodes"] == 3
        assert len(track_result["records"]) == 3
    # The exam seeds live outside the v3 training range.
    assert all(not (70_000 <= s < 71_000) for s in EXAM_V3_SEEDS)
