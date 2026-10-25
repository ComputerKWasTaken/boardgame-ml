"""Agents package: reference (non-learning) agents live here in M1.

Learning agents are Stable-Baselines3 models constructed in
``boardgame_ml.training.train`` — the framework deliberately does not
hand-roll RL algorithms (see DESIGN.md).
"""

from boardgame_ml.agents.baselines import (
    BASELINES,
    BasicStrategyAgent,
    ProbabilityAgent,
    RandomAgent,
    get_baseline,
)

__all__ = [
    "BASELINES",
    "BasicStrategyAgent",
    "ProbabilityAgent",
    "RandomAgent",
    "get_baseline",
]
