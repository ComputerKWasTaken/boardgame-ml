"""Game registry. Adding a game = implement it + one line here."""

from __future__ import annotations

from typing import Any

from boardgame_ml.envs.base import BaseGameEnv
from boardgame_ml.envs.blackjack import BlackjackEnv
from boardgame_ml.envs.flappy import FlappyBirdEnv
from boardgame_ml.envs.higher_or_lower import HigherOrLowerEnv
from boardgame_ml.envs.match3 import Match3Env
from boardgame_ml.envs.racing import RacingEnv

GAMES: dict[str, type[BaseGameEnv]] = {
    "blackjack": BlackjackEnv,
    "higher_or_lower": HigherOrLowerEnv,
    "match3": Match3Env,
    "flappy": FlappyBirdEnv,
    "racing": RacingEnv,
}


def make_env(
    game: str, render_mode: str | None = None, **kwargs: Any
) -> BaseGameEnv:
    """Construct a game env; extra ``kwargs`` go to its constructor."""
    if game not in GAMES:
        raise ValueError(f"Unknown game {game!r}; available: {sorted(GAMES)}")
    return GAMES[game](render_mode=render_mode, **kwargs)


__all__ = [
    "BaseGameEnv",
    "BlackjackEnv",
    "HigherOrLowerEnv",
    "Match3Env",
    "FlappyBirdEnv",
    "RacingEnv",
    "GAMES",
    "make_env",
]
