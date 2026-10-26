"""Simultaneous Pong with public state and collectible, activated abilities.

Coordinates use y down. Actions 0/1/2 = up/idle/down; 3/4/5 add activation.
Observations use each seat's left-facing perspective (self always at left).
Rewards are exclusively the zero-sum point score; the match timer is a rule,
not an external truncation. See docs/PONG.md for rules and experiments.
"""

from __future__ import annotations

import math
from typing import ClassVar

import gymnasium as gym
import numpy as np

from boardgame_ml.envs.simultaneous import SimultaneousTwoPlayerEnv

NONE, ENLARGE, BOOST, SHIELD = range(4)
POWER_NAMES = ("empty", "enlarge", "boost", "shield")
ACTION_NAMES = {
    i: ("up", "idle", "down")[i % 3] + (" + activate" if i >= 3 else "")
    for i in range(6)
}


class PongEnv(SimultaneousTwoPlayerEnv):
    metadata: ClassVar[dict] = {"render_modes": ["human"], "render_fps": 20}

    def __init__(
        self,
        render_mode=None,
        *,
        width=960.0,
        height=540.0,
        paddle_height=96.0,
        paddle_width=14.0,
        paddle_inset=32.0,
        paddle_speed=300.0,
        paddle_acceleration=1800.0,
        ball_radius=8.0,
        ball_speed=420.0,
        return_acceleration=1.04,
        normal_speed_cap=720.0,
        ball_speed_cap=900.0,
        max_angle_degrees=60.0,
        spin=0.25,
        dt=0.05,
        physics_substeps=6,
        points_to_win=5,
        match_seconds=90.0,
        serve_seconds=0.5,
        pickup_interval=6.0,
        pickup_lifetime=6.0,
        pickup_radius=14.0,
        enlarge_multiplier=1.5,
        enlarge_seconds=5.0,
        boost_multiplier=1.35,
        boost_seconds=6.0,
        shield_seconds=6.0,
        serve_angle=0.4,
        pickup_x_min=0.35,
        pickup_x_max=0.65,
        pickup_y_min=0.15,
        pickup_y_max=0.85,
    ):
        super().__init__(render_mode)
        values = locals().copy()
        values.pop("self")
        values.pop("render_mode")
        values.pop("__class__", None)
        for key, value in values.items():
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be finite and positive")
            setattr(self, key, value)
        if (
            int(physics_substeps) != physics_substeps
            or int(points_to_win) != points_to_win
        ):
            raise ValueError("physics_substeps and points_to_win must be integers")
        self.physics_substeps, self.points_to_win = (
            int(physics_substeps),
            int(points_to_win),
        )
        if not (
            0 < pickup_x_min < pickup_x_max < 1
            and 0 < pickup_y_min < pickup_y_max < 1
            and serve_angle < math.pi / 2
        ):
            raise ValueError("invalid pickup bounds or serve angle")
        if not (
            ball_speed <= normal_speed_cap <= ball_speed_cap
            and 0 < max_angle_degrees < 85
        ):
            raise ValueError("invalid ball speed caps or return angle")
        if (
            paddle_height * enlarge_multiplier >= height
            or 2 * (paddle_inset + paddle_width + ball_radius) >= width
            or 2 * ball_radius >= height
        ):
            raise ValueError("arena must contain enlarged paddles and ball")
        self.single_action_space = gym.spaces.Discrete(6)
        self.single_observation_space = gym.spaces.Box(
            -1.0, 1.0, (36,), dtype=np.float32
        )
        self.action_space = gym.spaces.Tuple(
            (self.single_action_space, self.single_action_space)
        )
        self.observation_space = gym.spaces.Tuple(
            (self.single_observation_space, self.single_observation_space)
        )

    def public_parameters(self):
        return {
            key: getattr(self, key)
            for key in (
                "width",
                "height",
                "paddle_height",
                "paddle_speed",
                "paddle_acceleration",
                "paddle_inset",
                "paddle_width",
                "ball_radius",
                "ball_speed_cap",
                "enlarge_multiplier",
                "points_to_win",
            )
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.elapsed = 0.0
        self.scores = [0, 0]
        self.inventory = [NONE, NONE]
        self.effects = [[0.0] * 3 for _ in range(2)]
        self.power_uses = [0, 0]
        self.pickups_collected = [0, 0]
        self.hits = [0, 0]
        self.winner = None
        self.done = False
        self.end_reason = None
        self._serve_direction = int(self.np_random.choice((-1, 1)))
        self.spawn_remaining = self.pickup_interval
        self._serve()
        return self._observations(), self._info()

    def _serve(self):
        self.paddle_y = [self.height / 2] * 2
        self.paddle_v = [0.0, 0.0]
        self.ball_x, self.ball_y = self.width / 2, self.height / 2
        angle = float(self.np_random.uniform(-self.serve_angle, self.serve_angle))
        self.ball_vx = self._serve_direction * self.ball_speed * math.cos(angle)
        self.ball_vy = self.ball_speed * math.sin(angle)
        self._serve_direction *= -1
        self.serve_remaining = self.serve_seconds
        self.effects = [[0.0] * 3 for _ in range(2)]
        self.pickup = None  # [kind, x, y, seconds remaining]
        self.last_touch = None

    def paddle_size(self, seat):
        return self.paddle_height * (
            self.enlarge_multiplier if self.effects[seat][0] > 0 else 1
        )

    def observe(self, seat):
        other = 1 - seat
        sign = 1 if seat == 0 else -1
        pickup = self.pickup
        obs = [
            (self.ball_x if seat == 0 else self.width - self.ball_x) / self.width,
            self.ball_y / self.height,
            sign * self.ball_vx / self.ball_speed_cap,
            self.ball_vy / self.ball_speed_cap,
            self.paddle_y[seat] / self.height,
            self.paddle_v[seat] / self.paddle_speed,
            self.paddle_y[other] / self.height,
            self.paddle_v[other] / self.paddle_speed,
            self.scores[seat] / self.points_to_win,
            self.scores[other] / self.points_to_win,
            max(0, 1 - self.elapsed / self.match_seconds),
            self.serve_remaining / self.serve_seconds,
        ]
        for player in (seat, other):
            obs.extend(float(self.inventory[player] == k) for k in range(4))
        for player in (seat, other):
            obs.extend(
                t / limit
                for t, limit in zip(
                    self.effects[player],
                    (self.enlarge_seconds, self.boost_seconds, self.shield_seconds),
                )
            )
        obs.extend(
            (
                (pickup[1] if seat == 0 else self.width - pickup[1]) / self.width,
                pickup[2] / self.height,
            )
            if pickup
            else (0.0, 0.0)
        )
        obs.extend(float((pickup[0] if pickup else NONE) == k) for k in range(4))
        obs.extend(
            (
                pickup[3] / self.pickup_lifetime if pickup else 0.0,
                self.spawn_remaining / self.pickup_interval,
                float(self.last_touch == seat),
                float(self.last_touch == other),
            )
        )
        return np.asarray(obs, dtype=np.float32)

    def _observations(self):
        return self.observe(0), self.observe(1)

    def _info(self, **extra):
        return dict(
            winner=self.winner,
            scores=list(self.scores),
            end_reason=self.end_reason,
            power_uses=list(self.power_uses),
            pickups_collected=list(self.pickups_collected),
            **extra,
        )

    def _activate(self, seat):
        kind = self.inventory[seat]
        if kind:
            self.inventory[seat] = NONE
            self.effects[seat][kind - 1] = (
                self.enlarge_seconds,
                self.boost_seconds,
                self.shield_seconds,
            )[kind - 1]
            self.power_uses[seat] += 1

    def _collect_segment(self, x, y, nx, ny):
        if (
            self.pickup is None
            or self.last_touch is None
            or self.inventory[self.last_touch] != NONE
        ):
            return
        kind, px, py, _ = self.pickup
        dx, dy = nx - x, ny - y
        length2 = dx * dx + dy * dy
        t = (
            max(0.0, min(1.0, ((px - x) * dx + (py - y) * dy) / length2))
            if length2
            else 0.0
        )
        if (x + t * dx - px) ** 2 + (y + t * dy - py) ** 2 <= (
            self.ball_radius + self.pickup_radius
        ) ** 2:
            self.inventory[self.last_touch] = int(kind)
            self.pickups_collected[self.last_touch] += 1
            self.pickup = None

    def _return_ball(self, seat):
        offset = (self.ball_y - self.paddle_y[seat]) / (
            self.paddle_size(seat) / 2 + self.ball_radius
        )
        angle = max(
            -1.0, min(1.0, offset + self.spin * self.paddle_v[seat] / self.paddle_speed)
        ) * math.radians(self.max_angle_degrees)
        speed = min(
            self.normal_speed_cap,
            math.hypot(self.ball_vx, self.ball_vy) * self.return_acceleration,
        )
        if self.effects[seat][BOOST - 1] > 0:
            speed = min(self.ball_speed_cap, speed * self.boost_multiplier)
            self.effects[seat][BOOST - 1] = 0.0
        self.ball_vx = (1 if seat == 0 else -1) * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)
        self.last_touch = seat
        self.hits[seat] += 1

    def _advance_ball(self, remaining):
        """Earliest swept event, then integrate leftover time after reflection."""
        eps = 1e-9
        while remaining > eps:
            candidates = []
            vx, vy = self.ball_vx, self.ball_vy
            if vy:
                wall = self.ball_radius if vy < 0 else self.height - self.ball_radius
                candidates.append(((wall - self.ball_y) / vy, "wall", -1))
            seat = 0 if vx < 0 else 1
            if vx:
                goal = self.ball_radius if seat == 0 else self.width - self.ball_radius
                candidates.append(((goal - self.ball_x) / vx, "goal", seat))
                face = self.paddle_inset + self.paddle_width / 2 + self.ball_radius
                if seat == 1:
                    face = self.width - face
                t = (face - self.ball_x) / vx
                if (
                    t >= -eps
                    and abs(self.ball_y + vy * t - self.paddle_y[seat])
                    <= self.paddle_size(seat) / 2 + self.ball_radius
                ):
                    candidates.append((t, "paddle", seat))
            events = [
                event for event in candidates if -eps <= event[0] <= remaining + eps
            ]
            t, kind, seat = (
                min(events, key=lambda e: e[0]) if events else (remaining, "none", -1)
            )
            t = max(0.0, min(t, remaining))
            nx, ny = self.ball_x + vx * t, self.ball_y + vy * t
            self._collect_segment(self.ball_x, self.ball_y, nx, ny)
            self.ball_x, self.ball_y = nx, ny
            remaining -= t
            if kind == "none":
                return None
            if kind == "wall":
                self.ball_vy *= -1
            elif kind == "paddle":
                self._return_ball(seat)
            elif self.effects[seat][SHIELD - 1] > 0:
                self.effects[seat][SHIELD - 1] = 0.0
                self.ball_vx *= -1
                self.last_touch = seat
            else:
                return 1 - seat
        return None

    def step(self, actions):
        if self.done:
            raise ValueError("Match is over; reset before stepping")
        if not self.action_space.contains(actions):
            raise ValueError(f"Expected two actions in [0, 5], got {actions!r}")
        for seat, action in enumerate(actions):
            if action >= 3:
                self._activate(seat)
        reward = [0.0, 0.0]
        h = self.dt / self.physics_substeps
        for _ in range(self.physics_substeps):
            duration = min(h, self.match_seconds - self.elapsed)
            if duration <= 1e-10:
                break
            for seat, action in enumerate(actions):
                target = ((int(action) % 3) - 1) * self.paddle_speed
                old_v = self.paddle_v[seat]
                v = old_v + max(
                    -self.paddle_acceleration * duration,
                    min(self.paddle_acceleration * duration, target - old_v),
                )
                half = self.paddle_size(seat) / 2
                y = self.paddle_y[seat] + (old_v + v) * duration / 2
                clamped = max(half, min(self.height - half, y))
                self.paddle_y[seat] = clamped
                self.paddle_v[seat] = 0.0 if y != clamped else v
            playing_time = max(0.0, duration - self.serve_remaining)
            self.serve_remaining = max(0.0, self.serve_remaining - duration)
            scorer = self._advance_ball(playing_time) if playing_time else None
            self.elapsed = min(self.match_seconds, self.elapsed + duration)
            for seat in (0, 1):
                self.effects[seat] = [
                    max(0.0, t - duration) for t in self.effects[seat]
                ]
            self.spawn_remaining = max(0.0, self.spawn_remaining - duration)
            if self.pickup is not None:
                self.pickup[3] -= duration
                if self.pickup[3] <= 0:
                    self.pickup = None
            if self.spawn_remaining <= 1e-10:
                if self.pickup is None:
                    self.pickup = [
                        int(self.np_random.integers(1, 4)),
                        float(
                            self.np_random.uniform(self.pickup_x_min, self.pickup_x_max)
                            * self.width
                        ),
                        float(
                            self.np_random.uniform(self.pickup_y_min, self.pickup_y_max)
                            * self.height
                        ),
                        self.pickup_lifetime,
                    ]
                self.spawn_remaining = self.pickup_interval
            if scorer is not None:
                self.scores[scorer] += 1
                reward[scorer] += 1.0
                reward[1 - scorer] -= 1.0
                if self.scores[scorer] >= self.points_to_win:
                    self.done, self.end_reason = True, "target_score"
                else:
                    self._serve()
                if self.done:
                    break
        if self.elapsed >= self.match_seconds - 1e-9:
            self.done, self.end_reason = True, "time_limit"
        if self.done:
            self.winner = (
                None
                if self.scores[0] == self.scores[1]
                else int(self.scores[1] > self.scores[0])
            )
        return self._observations(), tuple(reward), self.done, False, self._info()

    def _render_text(self):
        return f"Pong {self.scores[0]}:{self.scores[1]} | {max(0, self.match_seconds - self.elapsed):.1f}s | {POWER_NAMES[self.inventory[0]]} vs {POWER_NAMES[self.inventory[1]]}"
