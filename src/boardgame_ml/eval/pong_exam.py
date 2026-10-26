"""Paired-seed Pong diagnostic: activation enabled versus disabled.

python -m boardgame_ml.eval.pong_exam --checkpoint runs/.../best_model.zip
The opponent keeps its abilities in both conditions. This is a behavioral
diagnostic, not a causal proof of strategic use or a multi-seed result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from boardgame_ml.agents.baselines import RandomAgent, get_baseline
from boardgame_ml.envs.pong import PongEnv
from boardgame_ml.envs.simultaneous import predict_joint
from boardgame_ml.eval.evaluate import resolve_agent


def examine(checkpoint: str, games_per_seat: int = 50, seed: int = 1_000_000):
    if games_per_seat < 1:
        raise ValueError("games_per_seat must be positive")
    model = resolve_agent(checkpoint, "pong")
    result = {
        "checkpoint": checkpoint,
        "games_per_seat": games_per_seat,
        "seed": seed,
        "conditions": {},
    }
    for enabled in (True, False):
        condition = {}
        for opponent_name in ("random", "pong_intercept"):
            seats = []
            for seat in (0, 1):
                row = {
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "points_for": 0,
                    "points_against": 0,
                    "power_uses": 0,
                    "pickups": 0,
                    "steps": 0,
                }
                for episode in range(games_per_seat):
                    match_seed = seed + episode + 10_000 * seat
                    opponent = (
                        RandomAgent(seed=match_seed)
                        if opponent_name == "random"
                        else get_baseline(opponent_name, "pong")
                    )
                    agents = [opponent, opponent]
                    agents[seat] = model
                    env = PongEnv()
                    obs, info = env.reset(seed=match_seed)
                    while True:
                        actions = list(predict_joint(agents, env, obs))
                        if not enabled:
                            actions[seat] %= 3
                        obs, _, done, truncated, info = env.step(tuple(actions))
                        row["steps"] += 1
                        if done or truncated:
                            break
                    row[
                        "draws"
                        if info["winner"] is None
                        else "wins"
                        if info["winner"] == seat
                        else "losses"
                    ] += 1
                    row["points_for"] += env.scores[seat]
                    row["points_against"] += env.scores[1 - seat]
                    row["power_uses"] += env.power_uses[seat]
                    row["pickups"] += env.pickups_collected[seat]
                    env.close()
                row["mean_point_difference"] = (
                    row["points_for"] - row["points_against"]
                ) / games_per_seat
                seats.append(row)
            condition[opponent_name] = {"seat0": seats[0], "seat1": seats[1]}
            print(f"activation={enabled} vs {opponent_name}: {seats}", flush=True)
        result["conditions"]["enabled" if enabled else "disabled"] = condition
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games-per-seat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument("--output", required=True, help="JSON diagnostic output path")
    args = parser.parse_args()
    result = examine(args.checkpoint, args.games_per_seat, args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
