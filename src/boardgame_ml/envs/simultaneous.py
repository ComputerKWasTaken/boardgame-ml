"""Joint-action two-player protocol, separate from alternating TwoPlayerEnv.

reset -> ((obs0, obs1), info); step((a0, a1)) ->
((obs0, obs1), (r0, r1), terminated, truncated, info).
Raw games are multi-agent protocols; Gym/SB3 check the single-seat adapter.
Neither seat observes the other's current action before choosing its own.
"""

from __future__ import annotations

from boardgame_ml.envs.base import BaseGameEnv


class SimultaneousTwoPlayerEnv(BaseGameEnv):
    NUM_PLAYERS = 2

    def public_parameters(self) -> dict:
        """Physical constants available to obs-only reference agents."""
        raise NotImplementedError

    def player_view(self, seat: int) -> PlayerView:
        return PlayerView(self, seat)


class PlayerView(BaseGameEnv):
    """Prediction-only seat view: public parameters, per-seat spaces/mask.

    Baselines receive the same observation as policies, with no raw game
    reference (and no peek at an opponent action or future RNG state).
    """

    def __init__(self, game: SimultaneousTwoPlayerEnv, seat: int):
        super().__init__()
        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        self.action_space = game.single_action_space
        self.observation_space = game.single_observation_space
        for key, value in game.public_parameters().items():
            setattr(self, key, value)


def predict_joint(agents, env, observations):
    """Resolve both decisions from one unchanged pre-step state."""
    from boardgame_ml.eval.evaluate import predict_action

    return tuple(
        predict_action(agents[seat], env.player_view(seat), observations[seat])
        for seat in (0, 1)
    )
