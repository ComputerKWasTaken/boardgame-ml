"""Flappy Bird — the framework's first *continuous* game (M4).

Every game before this one is turn-based: a step is one whole
decision (hit or stand, swap two gems). Flappy Bird's world moves
whether or not the agent is ready. The thesis this env tests (see
DESIGN.md, "Continuous games"): a Gymnasium ``step`` was never a
turn — it is a **time slice**. Shrink the slice until the dynamics
look continuous and the same contract, trainer, evaluator, and
viewer all keep working, unchanged.

Time and physics
-----------------
* The agent decides **once per step**; one step advances the world
  by ``dt`` seconds (default 1/20 s = 20 decisions per second).
* Physics integrates with semi-implicit Euler in ``physics_substeps``
  substeps of ``dt / physics_substeps``: velocity updates first, then
  position. Substeps refine the *physics*; the decision rate is
  always one action per ``dt``.
* All randomness (pipe gap placement) comes from the env's seeded
  RNG, so an episode is a deterministic function of (seed, actions) —
  tested, like every other game here.

The world (pixels, y measured **up from the ground**):

* width 288, height 512 (the classic cabinet proportions), bird
  pinned at x = 48 with radius 12; pipes are 52 wide with a 100 px
  gap, scrolling left at 120 px/s, left edges 180 px apart.
* Gravity −1000 px/s²; a flap *sets* vertical velocity to
  +300 px/s (it does not add to it — the classic feel, and what
  makes the dynamics a clean function of the last flap).

Observation (float32, all normalized — the design point)
-------------------------------------------------------
A single snapshot of positions is **not** a Markov state for a
continuous game: it hides how fast the bird is moving. This env
therefore ships the derivatives in the observation instead of
stacking frames:

* ``bird_y / height``                — where the bird is
* ``bird_vy / 800``, clipped to [−1, 1] — how fast it is moving
* ``(next_pipe_x − bird_x) / width`` — horizontal distance to the
  pipe the bird must thread next
* ``next_gap_top / height``          — the gap's upper edge
* ``next_gap_bottom / height``       — the gap's lower edge

Position + velocity + the target's geometry is the full state of
the dynamics, so a plain MLP on this vector sees everything a
frame-stack would have to infer.

Actions: ``Discrete(2)`` — 0 = glide, 1 = flap. Both are always
legal (nothing to mask: the all-True default from the base class).

Rewards — the game's own score, honestly counted
-------------------------------------------------
* **+1.0 per pipe cleared** — Flappy Bird's displayed score.
* **+0.01 per step survived** — a *distance* term, not shaping by
  stealth: the world scrolls at a constant speed, so steps survived
  are exactly proportional to horizontal distance travelled, and
  distance is a score this genre keeps (endless runners display it
  directly). It cannot be farmed — the only way to accumulate it is
  to keep not crashing while pipes keep coming. At 0.01/step it is
  worth 0.3 per pipe interval, strictly subordinate to the pipe
  point it accompanies. (A test pins this decomposition.)

Endings: crashing into a pipe, the ground, or the ceiling is a true
**termination**. The ``max_steps`` cap is a time limit, hence a
**truncation** — per the framework's contract rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym

from boardgame_ml.envs.base import BaseGameEnv

GLIDE, FLAP = 0, 1

# World geometry (pixels). y is measured up from the ground.
WORLD_WIDTH = 288.0
WORLD_HEIGHT = 512.0
BIRD_X = 48.0
BIRD_RADIUS = 12.0
PIPE_WIDTH = 52.0
#: Vertical margin kept clear when placing a gap centre, so gaps are
#: always reachable: centre ∈ [margin + gap/2, height − margin − gap/2].
GAP_MARGIN = 40.0
#: The first pipe spawns this far beyond the right edge — the bird
#: gets a fair run-up before the first decision matters.
FIRST_PIPE_OFFSET = 40.0
#: Scale for the velocity component of the observation (see docstring).
VY_NORM = 800.0
#: Numeric-safety clamp on vertical velocity (px/s).
VY_LIMIT = 1000.0

# Reward decomposition (pinned by tests; see module docstring).
PIPE_REWARD = 1.0
DISTANCE_REWARD_PER_STEP = 0.01


@dataclass
class Pipe:
    """One pipe pair: a solid column with a gap in it.

    ``x`` is the left edge; the gap spans ``[gap_bottom, gap_top]``
    in world y (up from the ground); the solid parts are
    ``[0, gap_bottom]`` and ``[gap_top, WORLD_HEIGHT]``.
    """

    x: float
    gap_bottom: float
    gap_top: float
    scored: bool = False

    @property
    def gap_center(self) -> float:
        return 0.5 * (self.gap_bottom + self.gap_top)


class FlappyBirdEnv(BaseGameEnv):
    """Flappy Bird: continuous dynamics, discrete decisions.

    Constructor parameters are the physics the YAML ``env_kwargs``
    seam exists for; defaults reproduce the documented game.
    """

    def __init__(
        self,
        render_mode: str | None = None,
        *,
        dt: float = 0.05,
        physics_substeps: int = 1,
        gravity: float = -1000.0,
        flap_velocity: float = 300.0,
        pipe_speed: float = 120.0,
        pipe_gap: float = 100.0,
        pipe_interval: float = 180.0,
        max_steps: int = 3600,
    ) -> None:
        super().__init__(render_mode=render_mode)
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if physics_substeps < 1:
            raise ValueError(
                f"physics_substeps must be >= 1, got {physics_substeps}"
            )
        if pipe_gap <= 0 or pipe_gap >= WORLD_HEIGHT - 2 * GAP_MARGIN:
            raise ValueError(f"pipe_gap {pipe_gap} leaves no room for a gap")
        if pipe_interval <= 0:
            raise ValueError(
                f"pipe_interval must be positive, got {pipe_interval}"
            )
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        self.dt = float(dt)
        self.physics_substeps = int(physics_substeps)
        self.gravity = float(gravity)
        self.flap_velocity = float(flap_velocity)
        self.pipe_speed = float(pipe_speed)
        self.pipe_gap = float(pipe_gap)
        self.pipe_interval = float(pipe_interval)
        self.max_steps = int(max_steps)

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 2.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # World state (public: renderers and heuristic baselines read
        # the env's real state, as they do for every game here).
        self.bird_y: float = WORLD_HEIGHT / 2
        self.bird_vy: float = 0.0
        self.pipes: list[Pipe] = []
        self.score: int = 0
        self.steps: int = 0

    # -- world construction -------------------------------------------------
    def _spawn_pipe(self, x: float) -> Pipe:
        lo = GAP_MARGIN + self.pipe_gap / 2
        hi = WORLD_HEIGHT - GAP_MARGIN - self.pipe_gap / 2
        center = float(self.np_random.uniform(lo, hi))
        return Pipe(
            x=float(x),
            gap_bottom=center - self.pipe_gap / 2,
            gap_top=center + self.pipe_gap / 2,
        )

    def next_pipe(self) -> Pipe:
        """The pipe the bird must thread next.

        The first pipe whose right edge has not yet passed the bird.
        Spawning keeps at least one such pipe ahead at all times.
        """
        for pipe in self.pipes:
            if pipe.x + PIPE_WIDTH >= BIRD_X:
                return pipe
        raise RuntimeError("no upcoming pipe — spawning invariant broken")

    # -- observations ---------------------------------------------------------
    def _obs(self) -> np.ndarray:
        pipe = self.next_pipe()
        return np.array(
            [
                self.bird_y / WORLD_HEIGHT,
                float(np.clip(self.bird_vy / VY_NORM, -1.0, 1.0)),
                (pipe.x - BIRD_X) / WORLD_WIDTH,
                pipe.gap_top / WORLD_HEIGHT,
                pipe.gap_bottom / WORLD_HEIGHT,
            ],
            dtype=np.float32,
        )

    # -- gym API --------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.bird_y = WORLD_HEIGHT / 2
        self.bird_vy = 0.0
        self.score = 0
        self.steps = 0
        self.pipes = [self._spawn_pipe(WORLD_WIDTH + FIRST_PIPE_OFFSET)]
        return self._obs(), self._info(score=self.score)

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        reward = DISTANCE_REWARD_PER_STEP

        # 1. The decision: a flap *sets* vertical velocity.
        if action == FLAP:
            self.bird_vy = self.flap_velocity

        # 2. Physics substeps (semi-implicit Euler: velocity, then
        #    position). The action is held across substeps — they
        #    refine integration, not the decision rate.
        h = self.dt / self.physics_substeps
        for _ in range(self.physics_substeps):
            self.bird_vy = float(
                np.clip(self.bird_vy + self.gravity * h, -VY_LIMIT, VY_LIMIT)
            )
            self.bird_y += self.bird_vy * h

        # 3. The world scrolls (linear — substep-independent).
        scroll = self.pipe_speed * self.dt
        for pipe in self.pipes:
            pipe.x -= scroll
        # Keep a pipe ahead: spawn the next as the last one enters.
        while self.pipes[-1].x <= WORLD_WIDTH:
            self.pipes.append(
                self._spawn_pipe(self.pipes[-1].x + self.pipe_interval)
            )
        self.pipes = [p for p in self.pipes if p.x + PIPE_WIDTH >= 0.0]

        # 4. Scoring: a pipe is cleared when its right edge passes
        #    the bird.
        for pipe in self.pipes:
            if not pipe.scored and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.scored = True
                self.score += 1
                reward += PIPE_REWARD

        self.steps += 1

        # 5. Endings. A crash terminates; the step cap truncates.
        terminated = self._crashed()
        truncated = not terminated and self.steps >= self.max_steps
        return (
            self._obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(score=self.score),
        )

    # -- collisions -------------------------------------------------------------
    def _crashed(self) -> bool:
        # Ground and ceiling: the bird comes to rest on the boundary.
        if self.bird_y - BIRD_RADIUS <= 0.0:
            self.bird_y = BIRD_RADIUS
            return True
        if self.bird_y + BIRD_RADIUS >= WORLD_HEIGHT:
            self.bird_y = WORLD_HEIGHT - BIRD_RADIUS
            return True
        # Pipes: circle vs the two solid rectangles of each pipe.
        for pipe in self.pipes:
            if self._circle_hits_rect(pipe.x, 0.0, PIPE_WIDTH, pipe.gap_bottom):
                return True
            if self._circle_hits_rect(
                pipe.x, pipe.gap_top, PIPE_WIDTH, WORLD_HEIGHT - pipe.gap_top
            ):
                return True
        return False

    def _circle_hits_rect(
        self, rx: float, ry: float, rw: float, rh: float
    ) -> bool:
        """True iff the bird circle strictly overlaps the rect.

        Closest-point test: a graze at exactly one radius of distance
        is safe (distance < radius is the collision).
        """
        cx = min(max(BIRD_X, rx), rx + rw)
        cy = min(max(self.bird_y, ry), ry + rh)
        dx = BIRD_X - cx
        dy = self.bird_y - cy
        return dx * dx + dy * dy < BIRD_RADIUS * BIRD_RADIUS

    def _render_text(self) -> str:
        pipe = self.next_pipe()
        return (
            f"FlappyBird | y: {self.bird_y:6.1f} | vy: {self.bird_vy:+6.1f} "
            f"| next pipe in {pipe.x - BIRD_X:5.1f}px "
            f"(gap {pipe.gap_bottom:.0f}–{pipe.gap_top:.0f}) "
            f"| score: {self.score} | steps: {self.steps}"
        )
