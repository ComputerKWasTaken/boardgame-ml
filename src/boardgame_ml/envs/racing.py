"""2D Racing — the framework's first *continuous-action* game (M5).

Flappy Bird (M4) made the world continuous but kept the decisions
discrete. Racing removes the last discrete crutch: the action is a
``Box(2)`` — steering and throttle, both continuous — so there is no
action mask at all (every point of the box is legal by construction)
and the learning algorithms are the continuous-control ones (PPO
here; SAC is registered for later tinkering). The time-slice model
from M4 carries over unchanged: one step is ``dt`` seconds, physics
integrates in substeps, and the observation ships the derivatives.

The track
---------
A closed loop generated deterministically from ``track_seed``: 12
control points on a rough circle (seeded radius/angle jitter), a
closed Catmull-Rom centerline sampled to 192 points, constant
half-width 6 m. The *track* is canonical — train, eval, and baselines
all face the same one (``track_seed`` is an ``env_kwargs`` entry in
the YAML config). What the episode seed randomizes is the **spawn**:
each reset places the car at a seeded centerline point with small
lateral/heading jitter. That one choice does three jobs: episodes
differ (a deterministic policy would otherwise replay one identical
trajectory every eval episode), training practices every corner from
step 0 instead of starving the far side of the track, and "a lap"
always means a full circuit from wherever you started.

**Generator v2 and per-episode tracks (M7).** The v1 generator above
is *frozen* — the M5 numbers must stay reproducible — but one
near-oval cannot distinguish "learned to drive" from "memorized a
track". ``track_version=2`` selects a separate procedural generator
(wide-variance control polygons behind validity gates: no
self-intersection, branch separation, a minimum-turn-radius floor —
see the v2 block below) that produces genuinely varied circuits:
hairpins, S-curves, long straights, 170–950 m. And with
``randomize_track=True``, ``reset()`` draws the episode's track seed
from ``[track_seed_lo, track_seed_hi)`` out of the episode's seeded
RNG stream and rebuilds the track before spawning — same episode
seed, same track and spawn; different seeds, different circuits.
Physics, observation, and reward are untouched: the raycast
observation was always track-agnostic geometry, and M7 is where
that bet is settled.

Car physics (arcade bicycle model)
----------------------------------
State: ``(x, y, heading, speed)`` — velocity aligned to the heading
(no lateral slip velocity; the arcade simplification, documented
rather than hidden). Per substep ``h``: steering angle
``δ = steer · max_steer``; yaw rate ``ω = speed / wheelbase · tan(δ)``
(a real bicycle model: no speed, no turn); throttle accelerates,
negative throttle brakes and then reverses at a limited speed;
linear drag caps the top speed. All constants arrive through
``env_kwargs``.

One piece of real racing physics is *not* simplified away: **grip**.
The yaw rate is capped so the lateral acceleration ``|speed · ω|``
never exceeds ``grip`` (default 14 m/s²). Without a grip limit,
velocity-aligned physics makes full throttle optimal everywhere —
cornering speed costs nothing — and "drive as fast as possible"
degenerates into "hold the throttle and point". With it, corner
speed is bounded by ``sqrt(grip · radius)`` (≈20 m/s on this
track's tightest corners, vs 32 m/s on the straights), so braking
for corners is a real, learnable skill. (Verified empirically: the
un-gripped prototype was lapping at the speed clamp within 30k
training steps; the grip cap is what makes this a driving problem.)

Observation (float32, normalized — derivatives + lookahead)
-----------------------------------------------------------
* ``n_rays`` raycasts fanned over ±90° from the car's nose, each the
  distance to the nearest track edge (ray vs. both edge polylines),
  normalized by ``ray_max_range`` — the spatial lookahead, Flappy's
  "distance to next pipe" grown into a fan;
* ``speed / max_speed`` — the derivative, as always;
* ``sin`` / ``cos`` of the heading relative to the centerline
  tangent at the nearest point;
* lateral offset from the centerline / half-width;
* the currently applied steering.

No pixels, no frame stacking: position-relative geometry + velocity
is the full state of these dynamics.

Progress and reward — checkpoints, paid once
--------------------------------------------
Progress is arc length along the centerline, tracked through an
ordered checkpoint scheme: 24 checkpoints divide the lap; a
checkpoint counter advances **monotonically** as the car's nearest
centerline point passes each next checkpoint in order, and credited
progress is the arc length of all passed sectors plus the forward
distance into the current sector, clamped at the sector's end.
Credited progress is a **high-water mark**: the per-step reward is
``Δcredited`` (metres), which is ≥ 0 by construction, so

* going backwards earns exactly 0 (the "wrong-way" choice: no
  negative progress rewards, and no way to re-earn ground by
  reversing and re-advancing — the high-water mark has already paid
  for it);
* cutting across the track cannot farm progress: the nearest-point
  search is windowed around the previous nearest point, checkpoint
  order cannot be skipped, and a "forward distance" that wraps most
  of the lap (the signature of reversing over a checkpoint line) is
  treated as *behind*, not ahead.

At fixed ``dt``, metres-per-step *is* speed: maximizing the reward
is minimizing lap time. Completing all 24 checkpoints pays a
**+100 lap bonus** and records the lap time; the episode continues
(more laps, more return) until the car leaves the track
(``|offset| > half_width`` → **termination**) or the step cap is hit
(``max_steps`` → **truncation**).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from boardgame_ml.envs.base import BaseGameEnv

#: Reward for completing one full lap (all checkpoints, in order).
LAP_BONUS = 100.0

#: Half-width of the window (in centerline points) for the local
#: nearest-point search. The car moves ≲1 point per step; 32 points
#: of slack is generous, and the window is what keeps "nearest" local
#: — a far branch of the track can never pose as the car's position.
NEAREST_WINDOW = 32


@dataclass
class Track:
    """A generated closed-loop track (all geometry precomputed)."""

    centerline: np.ndarray  # (M, 2) float64
    left_edge: np.ndarray  # (M, 2)
    right_edge: np.ndarray  # (M, 2)
    tangents: np.ndarray  # (M, 2) unit vectors along travel direction
    seg_len: np.ndarray  # (M,) length of segment i -> i+1 (wrapping)
    cumlen: np.ndarray  # (M,) arc length at point i
    total_length: float
    checkpoint_idx: np.ndarray  # (K,) centerline indices of checkpoints
    checkpoint_s: np.ndarray  # (K,) arc length at each checkpoint
    sector_len: np.ndarray  # (K,) arc length checkpoint k -> k+1

    @property
    def n_points(self) -> int:
        return int(self.centerline.shape[0])


def generate_track(
    seed: int,
    *,
    n_control_points: int = 12,
    samples_per_segment: int = 16,
    base_radius: float = 55.0,
    radius_jitter: float = 0.25,
    angle_jitter: float = 0.25,
    smooth_passes: int = 3,
    smooth_factor: float = 0.35,
    half_width: float = 6.0,
    n_checkpoints: int = 24,
) -> Track:
    """Build a closed loop deterministically from ``seed``.

    Control points sit on a rough circle (angles evenly spaced with
    seeded jitter strictly smaller than the spacing, so their order
    is preserved; radii jittered multiplicatively). A few Laplacian
    smoothing passes over the control polygon (each point pulled
    toward its neighbours' midpoint) remove the near-hairpin kinks
    raw jitter + spline overshoot would otherwise create — measured
    minimum turn radius rises from ~7 m (undrivable on a 12 m wide
    track) to ~25 m, while the loop keeps its organic, non-circular
    shape. A closed Catmull-Rom spline through the smoothed points
    is sampled uniformly in the spline parameter; arc lengths come
    from the sampled polyline. Identical seeds reproduce identical
    tracks, bit for bit (tested).
    """
    rng = np.random.default_rng(seed)
    n = n_control_points
    spacing = 2.0 * np.pi / n
    angles = spacing * np.arange(n) + rng.uniform(
        -angle_jitter, angle_jitter, size=n
    ) * spacing
    radii = base_radius * rng.uniform(
        1.0 - radius_jitter, 1.0 + radius_jitter, size=n
    )
    control = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    for _ in range(smooth_passes):
        midpoint = 0.5 * (
            np.roll(control, 1, axis=0) + np.roll(control, -1, axis=0)
        )
        control = control + smooth_factor * (midpoint - control)

    # Closed Catmull-Rom: segment i runs control[i] -> control[i+1]
    # with neighbours control[i-1], control[i+2] (indices wrap).
    ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    points = []
    for i in range(n):
        p0, p1, p2, p3 = (
            control[(i - 1) % n],
            control[i],
            control[(i + 1) % n],
            control[(i + 2) % n],
        )
        t = ts[:, None]
        seg = (
            0.5
            * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t**3
            )
        )
        points.append(seg)
    centerline = np.concatenate(points, axis=0)
    return _assemble_track(centerline, half_width, n_checkpoints)


def _assemble_track(
    centerline: np.ndarray, half_width: float, n_checkpoints: int
) -> Track:
    """Geometry derived from a sampled centerline: lengths, tangents,
    edge polylines, checkpoints. Shared by the v1 and v2 generators —
    pure functions of the centerline, in a fixed operation order, so
    both generators' outputs are bit-reproducible."""
    m = centerline.shape[0]

    nxt = np.roll(centerline, -1, axis=0)
    seg_vec = nxt - centerline
    seg_len = np.linalg.norm(seg_vec, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
    total_length = float(seg_len.sum())

    # Tangents by central differences (wrap); normals are the
    # left-hand perpendicular (-ty, tx).
    tangents = np.roll(centerline, -1, axis=0) - np.roll(centerline, 1, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    left_edge = centerline + normals * half_width
    right_edge = centerline - normals * half_width

    checkpoint_idx = np.array(
        [int(round(k * m / n_checkpoints)) % m for k in range(n_checkpoints)]
    )
    checkpoint_s = cumlen[checkpoint_idx]
    sector_len = (np.roll(checkpoint_s, -1) - checkpoint_s) % total_length

    return Track(
        centerline=centerline,
        left_edge=left_edge,
        right_edge=right_edge,
        tangents=tangents,
        seg_len=seg_len,
        cumlen=cumlen,
        total_length=total_length,
        checkpoint_idx=checkpoint_idx,
        checkpoint_s=checkpoint_s,
        sector_len=sector_len,
    )


# ---------------------------------------------------------------------------
# Track generator v2 (M7) — procedural circuits, validity-gated
#
# The v1 generator above is frozen: it produces the canonical M5
# tracks (seed 7 above all) and must never change, or every M5/M5b
# number stops being reproducible. v2 is a separate generator for a
# different job — not *a* track but a *distribution* of tracks, so a
# policy can be trained and examined on circuits it has never seen.
#
# Design: candidate circuits are control polygons with far wider
# variance than v1 (10–18 points, angular gaps 0.5–1.6× the mean,
# log-normal radii, base radius 42–72 m), sampled through the same
# closed Catmull-Rom spline. Raw variance at that scale produces
# undrivable kinks and self-intersections, so candidates pass
# through **validity gates** before acceptance:
#
# * *No self-intersection* — no two centerline segments cross unless
#   they are neighbours along the loop.
# * *Branch separation* — centerline points far apart along the loop
#   (> 1/6 of it) stay more than ``2 · half_width + 3 m`` apart, so
#   distinct branches' edges never overlap (the windowed
#   nearest-point search and the offset semantics rely on this).
# * *Minimum turn radius* — the tightest circumradius of the sampled
#   centerline must clear ``min_turn_radius`` (default 13 m). With
#   the grip cap, corner speed is ``sqrt(grip · r)`` ≈ 13.5 m/s at
#   13 m: a true hairpin, slow but physically makeable. Each attempt
#   draws a *target* radius in [min, 27 m] and repairs toward it, so
#   accepted tracks spread across the tight↔flowing spectrum
#   instead of all clustering at the gate.
# * *Sanity* — no degenerate segments; total length in [170, 950] m.
#
# A candidate that fails only the radius gate is *repaired* (light
# Laplacian passes on the control polygon — local character survives
# because the passes stop the moment the target is met); a candidate
# that fails a structural gate is rejected outright. After 40 failed
# attempts the variance dial turns down in stages (×0.7, ×0.45,
# ×0.25) toward the v1 regime, which passes essentially always, so
# generation terminates quickly in practice (measured: ~3 attempts
# and ~16 ms per seed) — and everything is a pure function of the
# seed: same seed, same track, bit for bit.
# ---------------------------------------------------------------------------

#: Default tightest-corner gate for v2 tracks (metres, centerline
#: circumradius). See the block comment above for the physics.
V2_MIN_TURN_RADIUS = 13.0

_V2_LENGTH_RANGE = (170.0, 950.0)
_V2_BRANCH_MARGIN = 3.0  # metres beyond 2 * half_width


def _circumradii(points: np.ndarray) -> np.ndarray:
    """Circumradius of every consecutive triple of points (wrapping).

    ``inf`` where the triple is (near-)collinear — straights have no
    curvature to gate on. This is the discrete curvature measure the
    v2 gates and shape statistics use.
    """
    a = np.roll(points, 1, axis=0)
    b = points
    c = np.roll(points, -1, axis=0)
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    area2 = np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (ab * bc * ca) / (2.0 * area2)
    r[area2 < 1e-12] = np.inf
    return r


def _far_pair_mask(m: int, min_along: int) -> np.ndarray:
    """(M, M) bool: pairs whose circular distance along the loop
    exceeds ``min_along`` points."""
    idx = np.arange(m)
    along = np.abs(idx[:, None] - idx[None, :])
    along = np.minimum(along, m - along)
    return along > min_along


def _self_intersects(points: np.ndarray) -> bool:
    """Any crossing between centerline segments that are not
    neighbours (circular gap > 2). Fully vectorized segment-pair
    intersection: for segments ``i`` and ``j`` solve
    ``a_i + t·e_i = a_j + u·e_j``; a crossing needs both parameters
    strictly inside (0, 1)."""
    m = points.shape[0]
    a = points
    e = np.roll(points, -1, axis=0) - a
    rel = a[None, :, :] - a[:, None, :]  # a_j - a_i, (i, j, 2)
    denom = e[:, None, 0] * e[None, :, 1] - e[:, None, 1] * e[None, :, 0]
    ok = np.abs(denom) > 1e-14
    den = np.where(ok, denom, 1.0)
    t = (rel[:, :, 0] * e[None, :, 1] - rel[:, :, 1] * e[None, :, 0]) / den
    u = (rel[:, :, 0] * e[:, None, 1] - rel[:, :, 1] * e[:, None, 0]) / den
    hit = ok & (t > 1e-9) & (t < 1 - 1e-9) & (u > 1e-9) & (u < 1 - 1e-9)
    hit &= _far_pair_mask(m, 2)
    return bool(hit.any())


def _branch_separation(points: np.ndarray) -> float:
    """Minimum distance between centerline points that are far apart
    along the loop (> 1/6 of it) — how close two distinct branches
    of the circuit come to each other."""
    m = points.shape[0]
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return float(d[_far_pair_mask(m, m // 6)].min())


def track_violations(
    centerline: np.ndarray,
    *,
    half_width: float = 6.0,
    min_turn_radius: float = V2_MIN_TURN_RADIUS,
) -> list[str]:
    """The v2 validity gates, as a list of failure tags ([] = valid).

    Order is cheap-to-expensive and the generator short-circuits on
    the same order; tests call this directly to verify the gates
    independently of the generator's accept path.
    """
    seg_len = np.linalg.norm(np.roll(centerline, -1, axis=0) - centerline, axis=1)
    total = float(seg_len.sum())
    failures: list[str] = []
    if bool((seg_len <= 0.05).any()):
        failures.append("degenerate_segment")
    if not (_V2_LENGTH_RANGE[0] <= total <= _V2_LENGTH_RANGE[1]):
        failures.append("length_out_of_range")
    if float(_circumradii(centerline).min()) < min_turn_radius:
        failures.append("min_turn_radius")
    if _self_intersects(centerline):
        failures.append("self_intersection")
    if _branch_separation(centerline) < 2.0 * half_width + _V2_BRANCH_MARGIN:
        failures.append("branch_separation")
    return failures


def track_shape_stats(track: Track) -> dict[str, float]:
    """Shape summary of a track, for generator batch reports/tests."""
    r = _circumradii(track.centerline)
    tight = (r < 40.0).astype(int)
    corners = int(((tight - np.roll(tight, 1)) == 1).sum())
    return {
        "length_m": float(track.total_length),
        "min_turn_radius_m": float(r.min()),
        "p10_turn_radius_m": float(np.percentile(r, 10)),
        "corners_r40": float(corners),
        "tight_fraction_r35": float((r < 35.0).mean()),
    }


def _v2_control_polygon(rng: np.random.Generator, variance: float) -> np.ndarray:
    """One raw candidate control polygon at the given variance dial
    (1.0 = full v2 variance; lower dials approach the v1 regime)."""
    n = int(rng.integers(10, 19))
    gaps = rng.uniform(0.5, 1.6, size=n) ** variance
    gaps *= 2.0 * np.pi / gaps.sum()
    phase = rng.uniform(0.0, 2.0 * np.pi)
    angles = phase + np.cumsum(gaps) - gaps
    base = rng.uniform(42.0, 72.0)
    log_r = rng.normal(0.0, 0.34 * variance, size=n)
    # One light smoothing of the radius profile itself: white-noise
    # radii zigzag the polygon; a smoothed profile flows — long
    # straights where the radius holds, bends where it sweeps.
    log_r = log_r + 0.35 * (0.5 * (np.roll(log_r, 1) + np.roll(log_r, -1)) - log_r)
    radii = base * np.exp(np.clip(log_r, -0.95 * variance, 0.65 * variance))
    return np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)


def _v2_sample(control: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Closed Catmull-Rom sampling (same scheme as v1, kept separate
    so v1's code path is never touched)."""
    n = control.shape[0]
    ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    points = []
    for i in range(n):
        p0, p1, p2, p3 = (
            control[(i - 1) % n],
            control[i],
            control[(i + 1) % n],
            control[(i + 2) % n],
        )
        t = ts[:, None]
        seg = (
            0.5
            * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t**2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t**3
            )
        )
        points.append(seg)
    return np.concatenate(points, axis=0)


def _laplacian_pass(control: np.ndarray, factor: float = 0.22) -> np.ndarray:
    midpoint = 0.5 * (np.roll(control, 1, axis=0) + np.roll(control, -1, axis=0))
    return control + factor * (midpoint - control)


def _generate_track_v2_impl(
    seed: int,
    *,
    half_width: float,
    n_checkpoints: int,
    min_turn_radius: float,
    samples_per_segment: int,
    max_attempts: int,
) -> tuple[Track, int]:
    """The v2 attempt loop; returns (track, attempts used)."""
    rng = np.random.default_rng(seed)
    for attempt in range(max_attempts):
        # Variance dial: full for the first 40 attempts, then staged
        # down toward the v1 regime, which passes essentially always.
        if attempt < 40:
            variance = 1.0
        elif attempt < 90:
            variance = 0.7
        elif attempt < 150:
            variance = 0.45
        else:
            variance = 0.25
        control = _v2_control_polygon(rng, variance)
        target = float(rng.uniform(min_turn_radius + 0.5, 27.0))
        for _repair in range(7):
            centerline = _v2_sample(control, samples_per_segment)
            # Gates run cheap-to-expensive: segment/length sanity
            # and the radius gate every iteration (repairs act on
            # the radius), the O(M^2) pair checks only once the
            # radius target is met — a candidate still failing the
            # radius never pays for them.
            seg_len = np.linalg.norm(
                np.roll(centerline, -1, axis=0) - centerline, axis=1
            )
            total = float(seg_len.sum())
            if bool((seg_len <= 0.05).any()) or not (
                _V2_LENGTH_RANGE[0] <= total <= _V2_LENGTH_RANGE[1]
            ):
                break  # structural: resample a fresh candidate
            if float(_circumradii(centerline).min()) < target:
                control = _laplacian_pass(control)  # radius repair
                continue
            if _self_intersects(centerline) or _branch_separation(
                centerline
            ) < 2.0 * half_width + _V2_BRANCH_MARGIN:
                break  # structural: resample a fresh candidate
            return (
                _assemble_track(centerline, half_width, n_checkpoints),
                attempt + 1,
            )
    raise RuntimeError(
        f"v2 track generation exhausted {max_attempts} attempts for "
        f"seed {seed} — the staged variance fallback should make this "
        "unreachable; treat it as a generator bug, not bad luck"
    )


def generate_track_v2(
    seed: int,
    *,
    half_width: float = 6.0,
    n_checkpoints: int = 24,
    min_turn_radius: float = V2_MIN_TURN_RADIUS,
    samples_per_segment: int = 16,
    max_attempts: int = 200,
) -> Track:
    """Build a varied, validity-gated circuit deterministically.

    See the v2 block comment above for the design: wide-variance
    candidates, repair-before-reject on the radius gate, structural
    rejection, staged variance fallback. Identical seeds reproduce
    identical tracks, bit for bit (tested).
    """
    track, _attempts = _generate_track_v2_impl(
        seed,
        half_width=half_width,
        n_checkpoints=n_checkpoints,
        min_turn_radius=min_turn_radius,
        samples_per_segment=samples_per_segment,
        max_attempts=max_attempts,
    )
    return track


def build_track(
    seed: int,
    *,
    version: int = 1,
    half_width: float = 6.0,
    n_checkpoints: int = 24,
    **v1_kwargs,
) -> Track:
    """Dispatch to the v1 (frozen, canonical) or v2 (procedural,
    validity-gated) generator. ``v1_kwargs`` are the v1 generator's
    shape parameters; v2 ignores them (its distribution is its own)."""
    if version == 1:
        return generate_track(
            seed, half_width=half_width, n_checkpoints=n_checkpoints, **v1_kwargs
        )
    if version == 2:
        return generate_track_v2(
            seed, half_width=half_width, n_checkpoints=n_checkpoints
        )
    raise ValueError(f"Unknown track generator version {version!r}; use 1 or 2")


class RacingEnv(BaseGameEnv):
    """2D racing: continuous steering/throttle on a fixed closed track.

    Constructor parameters are the physics/track knobs the YAML
    ``env_kwargs`` seam exists for; defaults are the documented game.
    """

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        track_seed: int = 7,
        track_version: int = 1,
        randomize_track: bool = False,
        track_seed_lo: int = 0,
        track_seed_hi: int = 100_000,
        n_control_points: int = 12,
        samples_per_segment: int = 16,
        base_radius: float = 55.0,
        radius_jitter: float = 0.25,
        angle_jitter: float = 0.25,
        half_width: float = 6.0,
        n_checkpoints: int = 24,
        dt: float = 0.05,
        physics_substeps: int = 2,
        wheelbase: float = 2.7,
        max_steer: float = 0.55,
        grip: float = 14.0,
        accel: float = 11.0,
        brake_decel: float = 18.0,
        max_speed: float = 32.0,
        reverse_max_speed: float = 6.0,
        drag: float = 0.32,
        n_rays: int = 11,
        ray_fov: float = np.pi,
        ray_max_range: float = 60.0,
        max_steps: int = 1800,
    ) -> None:
        super().__init__(render_mode=render_mode)
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if physics_substeps < 1:
            raise ValueError(
                f"physics_substeps must be >= 1, got {physics_substeps}"
            )
        if half_width <= 0:
            raise ValueError(f"half_width must be positive, got {half_width}")
        if n_rays < 3:
            raise ValueError(f"n_rays must be >= 3, got {n_rays}")
        if not 0 < ray_fov <= 2 * np.pi:
            raise ValueError(f"ray_fov must be in (0, 2π], got {ray_fov}")
        if ray_max_range <= 0:
            raise ValueError(
                f"ray_max_range must be positive, got {ray_max_range}"
            )
        if max_speed <= 0 or reverse_max_speed < 0:
            raise ValueError("speed limits must be positive")
        if grip <= 0:
            raise ValueError(f"grip must be positive, got {grip}")
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if n_checkpoints < 4:
            raise ValueError(f"n_checkpoints must be >= 4, got {n_checkpoints}")
        if track_version not in (1, 2):
            raise ValueError(
                f"track_version must be 1 (frozen canonical) or 2 "
                f"(procedural), got {track_version}"
            )
        if randomize_track and not 0 <= track_seed_lo < track_seed_hi:
            raise ValueError(
                f"randomize_track needs 0 <= track_seed_lo < track_seed_hi, "
                f"got [{track_seed_lo}, {track_seed_hi})"
            )

        self.track_seed = int(track_seed)
        self.track_version = int(track_version)
        self.randomize_track = bool(randomize_track)
        self.track_seed_lo = int(track_seed_lo)
        self.track_seed_hi = int(track_seed_hi)
        self._track_shape_kwargs = dict(
            n_control_points=n_control_points,
            samples_per_segment=samples_per_segment,
            base_radius=base_radius,
            radius_jitter=radius_jitter,
            angle_jitter=angle_jitter,
        )
        self.half_width = float(half_width)
        self.n_checkpoints = int(n_checkpoints)
        self.dt = float(dt)
        self.physics_substeps = int(physics_substeps)
        self.wheelbase = float(wheelbase)
        self.max_steer = float(max_steer)
        self.grip = float(grip)
        self.accel = float(accel)
        self.brake_decel = float(brake_decel)
        self.max_speed = float(max_speed)
        self.reverse_max_speed = float(reverse_max_speed)
        self.drag = float(drag)
        self.n_rays = int(n_rays)
        self.ray_fov = float(ray_fov)
        self.ray_max_range = float(ray_max_range)
        self.max_steps = int(max_steps)

        self._set_track(
            build_track(
                self.track_seed,
                version=self.track_version,
                half_width=half_width,
                n_checkpoints=n_checkpoints,
                **(self._track_shape_kwargs if self.track_version == 1 else {}),
            )
        )
        self.ray_angles = np.linspace(
            -self.ray_fov / 2, self.ray_fov / 2, self.n_rays
        )
        self._window_offsets = np.arange(
            -NEAREST_WINDOW, NEAREST_WINDOW + 1
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_rays + 5,), dtype=np.float32
        )

        # Car / episode state (public: renderers and tests read it).
        self.car_x = 0.0
        self.car_y = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.steer = 0.0  # last applied steering input, [-1, 1]
        self.throttle = 0.0  # last applied throttle input
        self.steps = 0
        self.laps = 0
        self.progress_m = 0.0  # credited high-water progress (metres)
        self.last_lap_time: float | None = None
        self.best_lap_time: float | None = None
        self.last_rays = np.ones(self.n_rays, dtype=np.float64)
        self._nearest_idx = 0
        self._offset = 0.0
        self._progress_offset = 0.0  # credited value at spawn (see reset)
        self._cp_global = 0  # index of last passed checkpoint
        self._cp_count = 0  # checkpoints passed since spawn
        self._sectors_credited = 0.0  # arc length of passed sectors
        self._lap_start_step = 0

    # -- geometry helpers ---------------------------------------------------
    def _set_track(self, track: Track) -> None:
        """Install a track and rebuild the derived raycast geometry.

        Called at construction and, in ``randomize_track`` mode, at
        every reset whose drawn track seed differs from the current
        one. Edge segments for raycasting: both edge polylines as
        (A, B) endpoint arrays; each loop closes within itself
        (segment i runs point i -> point i+1, wrapping). The raycast
        uses a stride-2 decimation of the edges: the chord-vs-curve
        deviation is ~0.1 m, immaterial for a distance observation,
        and it halves the per-step intersection work.
        """
        self.track = track
        stride = 2
        left_d = track.left_edge[::stride]
        right_d = track.right_edge[::stride]
        self._seg_a = np.concatenate([left_d, right_d])
        seg_b = np.concatenate(
            [np.roll(left_d, -1, axis=0), np.roll(right_d, -1, axis=0)]
        )
        self._seg_e = seg_b - self._seg_a

    def _update_nearest(self, *, global_search: bool = False) -> None:
        """Refresh the nearest centerline point, signed offset, distance.

        Windowed around the previous nearest index (see
        ``NEAREST_WINDOW``): the car's notion of "where on the track
        am I" can only move continuously, which is part of the
        progress scheme's cut/farm resistance.
        """
        centerline = self.track.centerline
        m = self.track.n_points
        if global_search:
            idxs = np.arange(m)
        else:
            idxs = (self._nearest_idx + self._window_offsets) % m
        pts = centerline[idxs]
        d2 = (pts[:, 0] - self.car_x) ** 2 + (pts[:, 1] - self.car_y) ** 2
        self._nearest_idx = int(idxs[int(np.argmin(d2))])
        cx, cy = centerline[self._nearest_idx]
        tx, ty = self.track.tangents[self._nearest_idx]
        # Signed lateral offset: positive = left of the travel direction.
        self._offset = float((self.car_x - cx) * -ty + (self.car_y - cy) * tx)

    @property
    def nearest_idx(self) -> int:
        return self._nearest_idx

    @property
    def offset(self) -> float:
        """Signed lateral offset from the centerline (metres)."""
        return self._offset

    def _raycast(self) -> np.ndarray:
        """Distances to the track edge along the ray fan (metres).

        Ray vs. every edge segment, all rays at once: for ray origin
        ``o`` and unit directions ``d``, solve ``o + t·d = a + u·e``
        per segment; a hit needs ``t > 0`` and ``u ∈ [0, 1]``, and the
        ray's distance is the smallest such ``t``. Rays that hit
        nothing within range report ``ray_max_range``.
        """
        a = self._seg_a
        e = self._seg_e
        ao_x = a[:, 0] - self.car_x  # (S,)
        ao_y = a[:, 1] - self.car_y
        angles = self.heading + self.ray_angles
        dx = np.cos(angles)[:, None]  # (R, 1)
        dy = np.sin(angles)[:, None]
        ex = e[:, 0][None, :]  # (1, S)
        ey = e[:, 1][None, :]
        denom = dx * ey - dy * ex  # cross(d, e), (R, S)
        safe = np.abs(denom) > 1e-12
        denom = np.where(safe, denom, 1.0)
        t = (ao_x[None, :] * ey - ao_y[None, :] * ex) / denom
        u = (ao_x[None, :] * dy - ao_y[None, :] * dx) / denom
        hit = safe & (t > 0.0) & (u >= 0.0) & (u <= 1.0)
        t = np.where(hit, t, np.inf)
        out = t.min(axis=1)
        return np.minimum(out, self.ray_max_range)

    # -- progress ---------------------------------------------------------
    def _credited_progress(self) -> tuple[float, bool]:
        """Checkpoint-anchored credited progress + lap-completion flag.

        Advances the monotonic checkpoint counter while the car's
        nearest point lies past the next checkpoint (a forward
        distance in ``[sector_len, total/2)`` — a larger wrap means
        the car is *behind* the last checkpoint, e.g. it reversed
        over the line, and must not advance). Returns the current
        credited arc length (passed sectors + clamped local forward
        distance) and whether this call completed a lap.
        """
        track = self.track
        s_local = float(track.cumlen[self._nearest_idx])
        lap_done = False
        while True:
            g = self._cp_global
            fwd = (s_local - float(track.checkpoint_s[g])) % track.total_length
            sector = float(track.sector_len[g])
            if sector <= fwd < 0.5 * track.total_length:
                self._sectors_credited += sector
                self._cp_global = (g + 1) % self.n_checkpoints
                self._cp_count += 1
                if self._cp_count % self.n_checkpoints == 0:
                    lap_done = True
                continue
            local = fwd if fwd < 0.5 * track.total_length else 0.0
            return self._sectors_credited + min(local, sector), lap_done

    # -- observations ---------------------------------------------------------
    def _obs(self) -> np.ndarray:
        ray_norm = np.clip(self.last_rays / self.ray_max_range, 0.0, 1.0)
        tx, ty = self.track.tangents[self._nearest_idx]
        rel = self.heading - float(np.arctan2(ty, tx))
        obs = np.concatenate(
            [
                ray_norm,
                [
                    np.clip(self.speed / self.max_speed, -1.0, 1.0),
                    np.sin(rel),
                    np.cos(rel),
                    np.clip(self._offset / self.half_width, -1.0, 1.0),
                    np.clip(self.steer, -1.0, 1.0),
                ],
            ]
        )
        return obs.astype(np.float32)

    def _info_dict(self, *, lap_completed: bool, off_track: bool) -> dict:
        lap_time = (self.steps - self._lap_start_step) * self.dt
        return self._info(
            progress_m=float(self.progress_m),
            progress_laps=float(self.progress_m / self.track.total_length),
            progress_fraction=float(
                (self.progress_m - self.laps * self.track.total_length)
                / self.track.total_length
            ),
            laps=int(self.laps),
            lap_completed=bool(lap_completed),
            last_lap_time=self.last_lap_time,
            best_lap_time=self.best_lap_time,
            current_lap_time=float(lap_time),
            speed=float(self.speed),
            offset=float(self._offset),
            off_track=bool(off_track),
        )

    # -- gym API --------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.randomize_track:
            # Per-episode circuit (M7): the track seed is the FIRST
            # draw from the episode's seeded RNG stream, before the
            # spawn draws — so an episode seed determines track and
            # spawn together, and different seeds deal different
            # circuits. Rebuild only on change (consecutive draws
            # can repeat a seed in a small range).
            episode_track_seed = int(
                self.np_random.integers(self.track_seed_lo, self.track_seed_hi)
            )
            if episode_track_seed != self.track_seed:
                self.track_seed = episode_track_seed
                self._set_track(
                    build_track(
                        episode_track_seed,
                        version=self.track_version,
                        half_width=self.half_width,
                        n_checkpoints=self.n_checkpoints,
                        **(
                            self._track_shape_kwargs
                            if self.track_version == 1
                            else {}
                        ),
                    )
                )
        track = self.track
        # Spawn: a seeded centerline point, tangent heading, small
        # lateral/heading jitter — the same track, a different start.
        spawn_idx = int(self.np_random.integers(0, track.n_points))
        lateral = float(self.np_random.uniform(-1.5, 1.5))
        cx, cy = track.centerline[spawn_idx]
        tx, ty = track.tangents[spawn_idx]
        self.car_x = float(cx - ty * lateral)
        self.car_y = float(cy + tx * lateral)
        self.heading = float(
            np.arctan2(ty, tx) + self.np_random.uniform(-0.05, 0.05)
        )
        self.speed = 0.0
        self.steer = 0.0
        self.throttle = 0.0
        self.steps = 0
        self.laps = 0
        self.progress_m = 0.0
        self.last_lap_time = None
        self.best_lap_time = None
        self._nearest_idx = spawn_idx
        self._update_nearest(global_search=True)
        # The last passed checkpoint is the one at or behind the spawn
        # sector; progress counts from there.
        k0 = int(self._nearest_idx * self.n_checkpoints // track.n_points)
        self._cp_global = k0
        self._cp_count = 0
        self._sectors_credited = 0.0
        self._lap_start_step = 0
        # Credited progress is measured *from the spawn*: the partial
        # sector the car spawns into is the zero point, so spawning
        # can never pre-pay progress.
        credited, _ = self._credited_progress()
        self._progress_offset = credited
        self.progress_m = 0.0
        self.last_rays = self._raycast()
        return self._obs(), self._info_dict(lap_completed=False, off_track=False)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (2,):
            raise ValueError(f"Racing action must have shape (2,), got {action.shape}")
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], -1.0, 1.0))
        self.steer = steer
        self.throttle = throttle

        h = self.dt / self.physics_substeps
        delta = steer * self.max_steer
        tan_delta = math.tan(delta)
        for _ in range(self.physics_substeps):
            # Longitudinal: throttle / brake-reverse / drag.
            a = -self.drag * self.speed
            if throttle >= 0.0:
                a += throttle * self.accel
            elif self.speed > 0.05:
                a += throttle * self.brake_decel  # braking (throttle < 0)
            else:
                a += throttle * self.accel * 0.6  # reverse, gentler
            self.speed = min(
                max(self.speed + a * h, -self.reverse_max_speed),
                self.max_speed,
            )
            # Bicycle yaw, grip-limited: no speed, no turn — and no
            # amount of steering beats the tyres. |v · ω| <= grip.
            yaw_rate = self.speed / self.wheelbase * tan_delta
            max_yaw = self.grip / max(abs(self.speed), 3.0)
            yaw_rate = min(max(yaw_rate, -max_yaw), max_yaw)
            self.heading += yaw_rate * h
            self.car_x += math.cos(self.heading) * self.speed * h
            self.car_y += math.sin(self.heading) * self.speed * h

        self.steps += 1
        self._update_nearest()
        self.last_rays = self._raycast()

        credited, lap_completed = self._credited_progress()
        credited -= self._progress_offset
        reward = max(0.0, credited - self.progress_m)
        self.progress_m = max(self.progress_m, credited)
        if lap_completed:
            lap_time = (self.steps - self._lap_start_step) * self.dt
            self.last_lap_time = float(lap_time)
            self.best_lap_time = (
                lap_time
                if self.best_lap_time is None
                else min(self.best_lap_time, float(lap_time))
            )
            self.laps += 1
            self._lap_start_step = self.steps
            reward += LAP_BONUS

        off_track = abs(self._offset) > self.half_width
        terminated = bool(off_track)
        truncated = bool(not terminated and self.steps >= self.max_steps)
        return (
            self._obs(),
            float(reward),
            terminated,
            truncated,
            self._info_dict(lap_completed=lap_completed, off_track=off_track),
        )

    def _render_text(self) -> str:
        kmh = self.speed * 3.6
        best = (
            f"{self.best_lap_time:.1f}s" if self.best_lap_time is not None else "—"
        )
        return (
            f"Racing | speed {kmh:5.1f} km/h | progress "
            f"{self.progress_m:7.1f} m (lap {self.laps + 1}) | "
            f"offset {self._offset:+.1f} m | best lap {best} | "
            f"step {self.steps}"
        )
