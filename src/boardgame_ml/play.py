"""Watch an agent play, in text.

    python -m boardgame_ml.play --game blackjack \
        --agent runs/blackjack_maskable_ppo_s0/best_model.zip
    python -m boardgame_ml.play --game higher_or_lower --agent probability

This is deliberately the seed of two future features: a graphical
"watch it play" viewer, and a bot API (load checkpoint, hand it a
state, get a move). Both only need this loop's decision point —
``predict_action`` — wrapped differently.
"""

from __future__ import annotations

import argparse

from boardgame_ml.envs import make_env
from boardgame_ml.eval.evaluate import predict_action, resolve_agent

ACTION_NAMES = {
    "blackjack": {0: "STAND", 1: "HIT"},
    "higher_or_lower": {0: "HIGHER", 1: "LOWER", 2: "HOLD"},
}


def play(game: str, agent_spec: str, episodes: int = 1, seed: int = 0) -> None:
    agent = resolve_agent(agent_spec, game)
    env = make_env(game, render_mode="human")
    names = ACTION_NAMES.get(game, {})

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        print(f"--- Episode {ep + 1} ({game}, agent={agent_spec}) ---")
        print(env.render())
        total = 0.0
        done = False
        while not done:
            action = predict_action(agent, env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            print(
                f"  -> {names.get(action, action)} | reward {float(reward):+g} "
                f"| running total {total:+g}"
            )
            print(env.render())
            done = terminated or truncated
        print(f"Episode return: {total:+g}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Watch an agent play (text mode).")
    parser.add_argument("--game", required=True)
    parser.add_argument("--agent", required=True,
                        help="Baseline name or checkpoint path.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    play(args.game, args.agent, args.episodes, args.seed)


if __name__ == "__main__":
    main()
