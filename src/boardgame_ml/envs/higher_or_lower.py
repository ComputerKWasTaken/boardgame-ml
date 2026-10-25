"""Higher or Lower — the framework's "hello world" game.

Rules (kept deliberately minimal and fully documented):

* A card with value 1-13 is face up. Each turn the agent guesses whether
  the *next* card is higher (action 0) or lower (action 1) than the
  current one, or holds (action 2) to end the episode immediately.
* Cards are drawn **with replacement** from a uniform 1-13 distribution,
  so every turn is an independent probability question with a known
  analytic answer — which is exactly why this game is the smoke test:
  the probability-greedy heuristic's score is computable, and a trained
  agent should approach it.
* An episode lasts at most ``MAX_TURNS`` guesses. Reaching the limit
  ends the episode by **truncation** (it is a time limit, not a
  terminal game state); choosing HOLD is a true termination.

Rewards are HONEST — no shaping:

* correct guess: +1
* wrong guess:   -1
* tie:            0 (the turn is still consumed; a tie is neither the
  agent's fault nor a success, so it is scored as neither)
* hold: ends the episode with 0 additional reward.

Observation (float32 Box, all components in [0, 1]):

* ``(current_card - 1) / 12``
* ``(score + MAX_TURNS) / (2 * MAX_TURNS)``   (score = net points)
* ``turns_remaining / MAX_TURNS``
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from boardgame_ml.envs.base import BaseGameEnv

HIGHER, LOWER, HOLD = 0, 1, 2
MAX_TURNS = 10
MIN_CARD, MAX_CARD = 1, 13


class HigherOrLowerEnv(BaseGameEnv):
    """Guess whether the next card is higher or lower. See module docstring."""

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self._card: int = MIN_CARD
        self._score: int = 0
        self._turns_used: int = 0

    # -- helpers ----------------------------------------------------------
    def _draw_card(self) -> int:
        return int(self.np_random.integers(MIN_CARD, MAX_CARD + 1))

    def _obs(self) -> np.ndarray:
        return np.array(
            [
                (self._card - MIN_CARD) / (MAX_CARD - MIN_CARD),
                (self._score + MAX_TURNS) / (2 * MAX_TURNS),
                (MAX_TURNS - self._turns_used) / MAX_TURNS,
            ],
            dtype=np.float32,
        )

    # -- gym API ----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._card = self._draw_card()
        self._score = 0
        self._turns_used = 0
        return self._obs(), self._info()

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        if action == HOLD:
            # Holding banks nothing and risks nothing; it simply stops.
            # (With these rewards a guess always has non-negative expected
            # value, so holding is dominated — see the heuristic agent —
            # but the action exists so the framework supports early-stop
            # semantics for games that do reward it.)
            return self._obs(), 0.0, True, False, self._info()

        next_card = self._draw_card()
        if next_card == self._card:
            reward = 0.0  # tie: turn consumed, no points either way
        elif (next_card > self._card) == (action == HIGHER):
            reward = 1.0
            self._score += 1
        else:
            reward = -1.0
            self._score -= 1

        self._card = next_card
        self._turns_used += 1
        # Reaching the turn limit is a time limit, i.e. a TRUNCATION,
        # not a termination: the state reached is not a terminal state
        # of the game, so value learning must bootstrap across it
        # (see DESIGN.md, "The game contract").
        truncated = self._turns_used >= MAX_TURNS
        return self._obs(), float(reward), False, bool(truncated), self._info()

    def _render_text(self) -> str:
        return (
            f"HigherOrLower | card: {self._card:2d} | score: {self._score:+d} "
            f"| turns left: {MAX_TURNS - self._turns_used}"
        )
