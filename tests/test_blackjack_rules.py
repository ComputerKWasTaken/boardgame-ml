"""Blackjack rules tests. Draws are stubbed via the env's public state
(player_hand / dealer_hand) and a scripted ``_draw_card`` queue, so each
test is deterministic and tests the rule, not the RNG."""

from collections import deque

import pytest

from boardgame_ml.envs.blackjack import BlackjackEnv, hand_value, is_natural


class ScriptedEnv(BlackjackEnv):
    """BlackjackEnv whose shoe deals a fixed queue of card values."""

    def __init__(self, queue):
        super().__init__()
        self._queue = deque(queue)

    def _draw_card(self):
        return self._queue.popleft()


def test_hand_value_usable_ace():
    assert hand_value([1, 6]) == (17, True)    # soft 17
    assert hand_value([1, 6, 5]) == (12, False)  # ace demoted to 1
    assert hand_value([1, 1]) == (12, True)    # one ace high, one low
    assert hand_value([10, 9]) == (19, False)
    assert hand_value([10, 9, 5]) == (24, False)  # bust: smallest total


def test_is_natural():
    assert is_natural([1, 10])
    assert not is_natural([1, 5, 5])  # 21 in three cards is not a natural
    assert not is_natural([10, 9])


def test_natural_pays_three_to_two():
    env = ScriptedEnv([1, 5, 10, 9])  # player A+10, dealer 5+9
    _, info = env.reset(seed=0)
    assert info["is_natural"]
    _, reward, terminated, _, _ = env.step(0)
    assert terminated and reward == pytest.approx(1.5)


def test_natural_vs_natural_pushes():
    env = ScriptedEnv([1, 1, 10, 10])  # both dealt naturals
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(1)  # action is ignored
    assert terminated and reward == 0.0


def test_dealer_stands_on_soft_17():
    # Player 19 stands; dealer A+6 = soft 17 must STAND, so player wins.
    # (Under hit-soft-17 the dealer would draw from the queued 2 and make
    # 19, pushing — this test pins the stand-on-all-17s rule.)
    env = ScriptedEnv([10, 1, 9, 6, 2])
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(0)
    assert terminated and reward == 1.0


def test_dealer_hits_sixteen():
    # Player 19 stands; dealer 10+6 = 16 must hit; queued 5 makes 21.
    env = ScriptedEnv([10, 10, 9, 6, 5])
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(0)
    assert terminated and reward == -1.0


def test_player_bust_loses_immediately():
    env = ScriptedEnv([10, 5, 6, 9, 10])  # player 16 hits into a 10
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(1)
    assert terminated and reward == -1.0


def test_push_on_equal_totals():
    env = ScriptedEnv([10, 10, 9, 9])  # 19 vs 19
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(0)
    assert terminated and reward == 0.0


def test_rewards_are_terminal_only():
    """No shaping: a non-terminal hit must return exactly 0."""
    env = ScriptedEnv([5, 5, 6, 9, 2])  # player 11 hits a 2 -> 13, alive
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(1)
    assert not terminated and reward == 0.0
