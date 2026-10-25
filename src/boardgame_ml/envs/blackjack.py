"""Blackjack — standard rules, minimal M1 scope, terminal-only rewards.

Rules:

* Infinite shoe: every card is drawn independently (with replacement).
  Ranks are uniform over 1-13; face cards count as 10, aces as 1/11.
  P(value 10) = 4/13, as in a real deck.
* Dealer stands on **all** 17s (including soft 17).
* No doubling, no splitting, no insurance (M1 keeps the action space to
  hit/stand so results are directly comparable to hit/stand basic
  strategy).
* A dealt natural (ace + ten-value) pays 3:2 unless the dealer also has
  a natural, which pushes. The natural is resolved on the first step —
  whatever action is passed is ignored, because there is no decision to
  make; ``info["is_natural"]`` flags this so evaluators can exclude the
  state from policy-agreement metrics.

Rewards are TERMINAL ONLY — +1 win, -1 loss, 0 push, +1.5 natural.
There is deliberately **no intermediate shaping of any kind**: shaping
that is not potential-based changes the optimal policy (in the legacy
project, blackjack shaping paid up to +1.4 for a single hit — more than
the +1.0 for winning the hand — so the agent learned the hints instead
of the game).

Observation (float32 Box, components in [0, 1]):

* ``player_sum / 32`` (best non-bust value, or raw sum when bust)
* ``usable_ace`` (1.0 if the player holds an ace counted as 11)
* ``dealer_upcard / 10`` (ace shown as 1 -> 0.1; ten-value -> 1.0)

Actions: 0 = stand, 1 = hit. Both are always legal, so the action mask
is all-True; masking machinery is still exercised through the base class.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from boardgame_ml.envs.base import BaseGameEnv

STAND, HIT = 0, 1


def hand_value(cards: list[int]) -> tuple[int, bool]:
    """Return ``(total, usable_ace)`` for a hand of card values (ace = 1).

    ``total`` is the best total <= 21 if one exists, otherwise the
    smallest (bust) total. ``usable_ace`` is True when an ace is being
    counted as 11 in that best total.
    """
    total = sum(cards)
    usable_ace = 1 in cards and total + 10 <= 21
    if usable_ace:
        total += 10
    return total, usable_ace


def is_natural(cards: list[int]) -> bool:
    """A natural blackjack: exactly two cards totalling 21."""
    return len(cards) == 2 and hand_value(cards)[0] == 21


class BlackjackEnv(BaseGameEnv):
    """Single-player blackjack vs. a fixed-rule dealer. See module docstring."""

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__(render_mode=render_mode)
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.player_hand: list[int] = []
        self.dealer_hand: list[int] = []

    # -- helpers ----------------------------------------------------------
    def _draw_card(self) -> int:
        """Draw a card value from the infinite shoe (ace=1, faces=10)."""
        rank = int(self.np_random.integers(1, 14))  # 1..13
        return 1 if rank == 1 else min(rank, 10)

    def _obs(self) -> np.ndarray:
        total, usable = hand_value(self.player_hand)
        return np.array(
            [
                min(total, 32) / 32.0,
                1.0 if usable else 0.0,
                self.dealer_hand[0] / 10.0,  # first dealer card is face-up
            ],
            dtype=np.float32,
        )

    def _dealer_play(self) -> None:
        """Dealer hits below 17 and stands on all 17s."""
        while hand_value(self.dealer_hand)[0] < 17:
            self.dealer_hand.append(self._draw_card())

    def _resolve_showdown(self) -> float:
        """Play out the dealer's hand and score the finished hand."""
        self._dealer_play()
        player_total, _ = hand_value(self.player_hand)
        dealer_total, _ = hand_value(self.dealer_hand)
        if dealer_total > 21:
            return 1.0
        if player_total > dealer_total:
            return 1.0
        if player_total < dealer_total:
            return -1.0
        return 0.0

    # -- gym API ----------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Deal order: player, dealer (up), player, dealer (hole).
        self.player_hand = [self._draw_card()]
        self.dealer_hand = [self._draw_card()]
        self.player_hand.append(self._draw_card())
        self.dealer_hand.append(self._draw_card())
        return self._obs(), self._info(is_natural=is_natural(self.player_hand))

    def step(self, action: int):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # A dealt natural is not a decision: resolve it immediately.
        if is_natural(self.player_hand):
            reward = 0.0 if is_natural(self.dealer_hand) else 1.5
            return self._obs(), float(reward), True, False, self._info(
                is_natural=True, outcome="natural"
            )

        if action == HIT:
            self.player_hand.append(self._draw_card())
            total, _ = hand_value(self.player_hand)
            if total > 21:
                return self._obs(), -1.0, True, False, self._info(outcome="bust")
            if total == 21:
                # No further decision exists; stand automatically.
                reward = self._resolve_showdown()
                return self._obs(), float(reward), True, False, self._info(
                    outcome="showdown"
                )
            return self._obs(), 0.0, False, False, self._info()

        # STAND
        reward = self._resolve_showdown()
        return self._obs(), float(reward), True, False, self._info(outcome="showdown")

    def _render_text(self) -> str:
        p_total, p_soft = hand_value(self.player_hand)
        soft = " soft" if p_soft else ""
        return (
            f"Blackjack | player: {self.player_hand} = {p_total}{soft} "
            f"| dealer shows: {self.dealer_hand[0]} (hole hidden)"
        )
