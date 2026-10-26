"""Gymnasium classics behind the game contract (M10 calibration).

The framework's games are all home-grown, which creates a blind
spot: nothing anchors ``train()``'s numbers to the outside world.
The M8 autopsy showed what that costs — SB3's Gaussian-policy std
default quietly invalidated racing PPO for two milestones because
every number the pipeline produced was internally consistent and
externally unchecked.

:class:`GymClassicEnv` is the tripwire's adapter: a raw Gymnasium
classic-control env (CartPole, LunarLander, Acrobot, ...) presented
as a :class:`BaseGameEnv`, so the *unmodified* trainer — vectorized
envs, mask wiring, EvalCallback best-by-eval selection, diagnostics —
trains on it exactly as it trains on a home game, and
``eval/calibration.py`` can compare the result against published
RL Baselines3 Zoo numbers for the same env and algorithm.

The shim is deliberately thin: spaces pass through untouched, the
inner env owns all dynamics and seeding, and the default
``action_masks()`` (all-True — these games mask nothing) is
inherited, so the mask guard probes it as an unmasked game and
plain PPO passes. The Gymnasium env id arrives via ``env_kwargs``,
so one registry entry (``gym_classic``) covers the whole classic
suite and the run's config records exactly which env was trained.

Dependency note: LunarLander and the other Box2D classics need
``gymnasium[box2d]`` extras (on this VM: the ``box2d`` and
``pygame`` wheels) installed in the venv; CartPole / Acrobot /
Pendulum need nothing extra. Constructing a Box2D env without
them raises Gymnasium's own named dependency error.
"""

from __future__ import annotations

import gymnasium as gym

from boardgame_ml.envs.base import BaseGameEnv


class GymClassicEnv(BaseGameEnv):
    """A Gymnasium classic env, wrapped to satisfy the game contract."""

    def __init__(
        self,
        env_id: str = "CartPole-v1",
        render_mode: str | None = None,
    ) -> None:
        super().__init__(render_mode=render_mode)
        self.env_id = env_id
        self._env = gym.make(env_id)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Seed our own RNG bookkeeping too (gym.Env.reset sets
        # _np_random): gymnasium's env checker inspects the
        # *wrapper's* generator, not the inner env's.
        super().reset(seed=seed)
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, {**info, **self._info()}

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        # _info() attaches the (all-True) action mask for Discrete
        # games, as the contract requires of step infos; Box-action
        # classics (Pendulum) carry no mask key, also per contract.
        return obs, float(reward), terminated, truncated, {**info, **self._info()}

    def _render_text(self) -> str:
        return f"GymClassicEnv({self.env_id}) — a Gymnasium classic, proxied verbatim."

    def close(self) -> None:
        self._env.close()
