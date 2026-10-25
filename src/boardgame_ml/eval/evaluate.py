"""Evaluation: the framework's proof-of-learning harness.

    python -m boardgame_ml.eval.evaluate --game blackjack --agent random
    python -m boardgame_ml.eval.evaluate --game blackjack \
        --agent runs/blackjack_maskable_ppo_s0/best_model.zip --episodes 2000

An ``agent`` is either a baseline name (``random``, ``basic_strategy``,
``probability``) or a path to a saved SB3 checkpoint. Policies are
always evaluated greedily (deterministic) over fresh episode seeds.

For blackjack the report also includes **basic-strategy agreement**:
the fraction of genuine decision states (dealt naturals excluded —
there is no decision to make) in which the agent's action equals the
basic-strategy action. This gives a ground-truth yardstick that return
alone can't: a good blackjack agent should *agree with the chart*, not
just post a lucky mean.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from boardgame_ml.agents.baselines import BaselineAgent, BasicStrategyAgent, get_baseline
from boardgame_ml.envs import make_env
from boardgame_ml.envs.base import validate_mask
from boardgame_ml.training.algorithms import MODEL_CLASSES, game_uses_masks


def load_sb3_model(path: str):
    """Load a checkpoint, trying each registered algorithm class."""
    last_exc: Exception | None = None
    for cls in MODEL_CLASSES.values():
        try:
            return cls.load(path)
        except Exception as exc:  # wrong class for this checkpoint
            last_exc = exc
    raise ValueError(f"Could not load checkpoint {path!r}: {last_exc}")


def _algorithm_label(model) -> str:
    """Registry name for a loaded model's class (for error messages)."""
    for name, cls in MODEL_CLASSES.items():
        if isinstance(model, cls):
            return name
    return type(model).__name__


def _ensure_checkpoint_mask_compatible(
    model,
    agent_spec: str,
    game: str,
    allow_unmasked: bool,
    env_kwargs: dict | None = None,
) -> None:
    """The training mask guard, applied to a resolved checkpoint.

    A non-MaskablePPO checkpoint evaluated on a game that masks would
    silently play an unmasked policy. Refuse, unless explicitly
    overridden — same rule as ``training.algorithms``.
    """
    if isinstance(model, MaskablePPO) or allow_unmasked:
        return
    if game_uses_masks(game, env_kwargs=env_kwargs):
        raise ValueError(
            f"Game {game!r} masks out actions in some states, but checkpoint "
            f"{agent_spec!r} is a {_algorithm_label(model)!r} model, which "
            f"ignores action masks: evaluating it would silently score an "
            f"unmasked policy (legacy failure #1). Pass allow_unmasked=True "
            f"(--allow-unmasked) to override."
        )


def _check_checkpoint_spaces(
    model, agent_spec: str, game: str, env_kwargs: dict | None = None
) -> None:
    """Refuse a checkpoint whose spaces don't match the game.

    Without this, a mismatched pairing surfaces as a cryptic torch
    shape error deep inside SB3 — or, for some algorithms, as
    plausible-looking garbage metrics.
    """
    env = make_env(game, **(env_kwargs or {}))
    if (
        model.observation_space != env.observation_space
        or model.action_space != env.action_space
    ):
        raise ValueError(
            f"Checkpoint {agent_spec!r} expects observation space "
            f"{model.observation_space} and action space {model.action_space}, "
            f"but game {game!r} has observation space {env.observation_space} "
            f"and action space {env.action_space}"
        )


def resolve_agent(
    agent_spec: str,
    game: str,
    *,
    allow_unmasked: bool = False,
    env_kwargs: dict | None = None,
):
    """Return a BaselineAgent or a loaded SB3 model."""
    try:
        return get_baseline(agent_spec, game)
    except ValueError:
        if Path(agent_spec).exists() or Path(agent_spec + ".zip").exists():
            model = load_sb3_model(agent_spec)
            _check_checkpoint_spaces(model, agent_spec, game, env_kwargs)
            _ensure_checkpoint_mask_compatible(
                model, agent_spec, game, allow_unmasked, env_kwargs
            )
            return model
        raise


def predict_action(agent, env, obs) -> int:
    """Greedy action from a baseline or SB3 model in ``env``'s state.

    For MaskablePPO the current action mask is passed explicitly, so
    the evaluated policy is exactly the masked policy that was trained.
    """
    raw = env.unwrapped
    if isinstance(agent, BaselineAgent):
        return agent.select_action(raw, obs)
    if isinstance(agent, MaskablePPO):
        mask = validate_mask(raw.action_masks(), raw.action_space.n)
        action, _ = agent.predict(obs, deterministic=True, action_masks=mask)
    else:
        action, _ = agent.predict(obs, deterministic=True)
    return int(np.asarray(action).reshape(-1)[0])


def evaluate(
    game: str,
    agent_spec: str,
    episodes: int = 1000,
    seed: int = 12345,
    *,
    allow_unmasked: bool = False,
    env_kwargs: dict | None = None,
) -> dict:
    """Run ``episodes`` greedy episodes and return a metrics dict."""
    agent = resolve_agent(
        agent_spec, game, allow_unmasked=allow_unmasked, env_kwargs=env_kwargs
    )
    env = make_env(game, **(env_kwargs or {}))
    reference = BasicStrategyAgent() if game == "blackjack" else None

    returns: list[float] = []
    lengths: list[int] = []
    wins = losses = pushes = 0
    agree = agree_total = 0

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        steps = 0
        first_decision = True
        while not done:
            action = predict_action(agent, env, obs)
            if reference is not None:
                # Dealt naturals are not decisions; skip them.
                if not (first_decision and info.get("is_natural", False)):
                    agree_total += 1
                    if action == reference.select_action(env, obs):
                        agree += 1
            obs, reward, terminated, truncated, info = env.step(action)
            first_decision = False
            ep_return += float(reward)
            steps += 1
            done = terminated or truncated
        returns.append(ep_return)
        lengths.append(steps)
        if ep_return > 0:
            wins += 1
        elif ep_return < 0:
            losses += 1
        else:
            pushes += 1

    result = {
        "game": game,
        "agent": agent_spec,
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "win_rate": wins / episodes,
        "loss_rate": losses / episodes,
        "push_rate": pushes / episodes,
    }
    if reference is not None and agree_total:
        result["basic_strategy_agreement"] = agree / agree_total
        result["agreement_decisions"] = agree_total
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent.")
    parser.add_argument("--game", required=True)
    parser.add_argument(
        "--agent",
        required=True,
        help="Baseline name (random / basic_strategy / probability) or checkpoint path.",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--json", action="store_true", help="Print raw JSON only.")
    parser.add_argument(
        "--allow-unmasked",
        action="store_true",
        help="Allow evaluating a mask-blind checkpoint on a game that masks.",
    )
    args = parser.parse_args(argv)

    result = evaluate(
        args.game, args.agent, args.episodes, args.seed,
        allow_unmasked=args.allow_unmasked,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Game: {result['game']}   Agent: {result['agent']}")
        print(f"Episodes: {result['episodes']}")
        print(f"Mean return: {result['mean_return']:+.4f} (±{result['std_return']:.3f})")
        print(
            f"Win/Loss/Push: {result['win_rate']:.1%} / "
            f"{result['loss_rate']:.1%} / {result['push_rate']:.1%}"
        )
        if "basic_strategy_agreement" in result:
            print(
                f"Basic-strategy agreement: "
                f"{result['basic_strategy_agreement']:.1%} "
                f"({result['agreement_decisions']} decisions)"
            )


if __name__ == "__main__":
    main()
