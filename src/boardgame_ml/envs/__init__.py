"""Game registry. Adding a game = implement it + one line here."""

from __future__ import annotations

from typing import Any

from boardgame_ml.envs.base import BaseGameEnv
from boardgame_ml.envs.blackjack import BlackjackEnv
from boardgame_ml.envs.connect_four import ConnectFourEnv
from boardgame_ml.envs.flappy import FlappyBirdEnv
from boardgame_ml.envs.higher_or_lower import HigherOrLowerEnv
from boardgame_ml.envs.match3 import Match3Env
from boardgame_ml.envs.racing import RacingEnv
from boardgame_ml.envs.two_player import TwoPlayerEnv
from boardgame_ml.envs.simultaneous import SimultaneousTwoPlayerEnv
from boardgame_ml.envs.pong import PongEnv

GAMES: dict[str, type[BaseGameEnv]] = {
    "blackjack": BlackjackEnv,
    "higher_or_lower": HigherOrLowerEnv,
    "match3": Match3Env,
    "flappy": FlappyBirdEnv,
    "racing": RacingEnv,
    "connect_four": ConnectFourEnv,
    "pong": PongEnv,
}

#: Alternating and simultaneous two-player games. Training uses the matching
#: single-seat adapter; alternating rules remain in :class:`TwoPlayerEnv`.
#: Training these goes through the self-play
#: wrapper; evaluation goes through the tournament harness.
TWO_PLAYER_GAMES: frozenset[str] = frozenset(
    name for name, cls in GAMES.items() if issubclass(cls, (TwoPlayerEnv, SimultaneousTwoPlayerEnv))
)
SIMULTANEOUS_GAMES = frozenset(name for name, cls in GAMES.items() if issubclass(cls, SimultaneousTwoPlayerEnv))


def make_env(
    game: str, render_mode: str | None = None, **kwargs: Any
) -> BaseGameEnv:
    """Construct a game env; extra ``kwargs`` go to its constructor."""
    if game not in GAMES:
        raise ValueError(f"Unknown game {game!r}; available: {sorted(GAMES)}")
    return GAMES[game](render_mode=render_mode, **kwargs)


__all__ = [
    "BaseGameEnv",
    "TwoPlayerEnv",
    "SimultaneousTwoPlayerEnv",
    "PongEnv",
    "SIMULTANEOUS_GAMES",
    "BlackjackEnv",
    "HigherOrLowerEnv",
    "Match3Env",
    "FlappyBirdEnv",
    "RacingEnv",
    "ConnectFourEnv",
    "GAMES",
    "TWO_PLAYER_GAMES",
    "make_env",
]
