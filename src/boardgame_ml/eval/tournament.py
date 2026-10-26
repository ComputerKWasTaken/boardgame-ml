"""Tournaments: how competitive games are evaluated (M6).

A single-agent game's proof of learning is a mean return; a
two-player game's return only exists relative to an opponent, so
the unit of evidence is the **match**: a policy plays another
policy as *both* seats (Connect Four's first player has a real
advantage, and a pairing measured from one chair would hide it), and
the record is wins / draws / losses per seat.

    python -m boardgame_ml.eval.tournament \
        --game connect_four \
        --agent runs/connect_four_maskable_ppo_s0_*/best_model.zip \
        --opponents random connect_four_greedy connect_four_minimax \
        --games-per-seat 200

Contestants are anything :func:`boardgame_ml.eval.evaluate.predict_action`
can drive: a baseline name, or a checkpoint path. Yardstick-vs-
yardstick rows (``--include-yardstick-matches``) calibrate the
ladder itself -- a learner's 70% against greedy means more when you
can see greedy goes 0% against minimax.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from boardgame_ml.envs import SIMULTANEOUS_GAMES, make_env
from boardgame_ml.eval.evaluate import predict_action, resolve_agent


@dataclass
class MatchRecord:
    """W/D/L for one ordered pairing, from the row player's view."""

    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.games if self.games else 0.0

    @property
    def score_rate(self) -> float:
        """Win=1, draw=0.5: the single comparable strength number."""
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def add(self, other: "MatchRecord") -> None:
        self.wins += other.wins
        self.draws += other.draws
        self.losses += other.losses

    def to_dict(self) -> dict:
        return {
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "games": self.games,
            "win_rate": self.win_rate,
            "draw_rate": self.draw_rate,
            "loss_rate": self.loss_rate,
            "score_rate": self.score_rate,
        }


def play_match(
    game: str,
    agent_seat0,
    agent_seat1,
    *,
    seed: int = 0,
    env_kwargs: dict | None = None,
) -> int | None:
    """One game; seat 0 moves first. Returns the winner (0/1) or
    None for a draw. Agents are resolved contestants (baselines or
    loaded models), driven by ``predict_action`` exactly as in
    evaluation and watch sessions."""
    env = make_env(game, **(env_kwargs or {}))
    agents = (agent_seat0, agent_seat1)
    obs, info = env.reset(seed=seed)
    from boardgame_ml.envs.simultaneous import SimultaneousTwoPlayerEnv, predict_joint
    while True:
        if isinstance(env, SimultaneousTwoPlayerEnv):
            action = predict_joint(agents, env, obs)
        else:
            player = info["current_player"]
            action = predict_action(agents[player], env, obs)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.close()
            return info.get("winner")


def play_pairing(
    game: str,
    agent_a,
    agent_b,
    *,
    games_per_seat: int = 200,
    seed: int = 0,
    env_kwargs: dict | None = None,
) -> dict:
    """``games_per_seat`` games with A opening, the same with B
    opening. Returns A's record split by seat and combined.

    Contestants may be agent specs (baseline names / checkpoint
    paths — resolved here) or already-resolved agents.
    """
    if isinstance(agent_a, str):
        agent_a = resolve_agent(agent_a, game, env_kwargs=env_kwargs)
    if isinstance(agent_b, str):
        agent_b = resolve_agent(agent_b, game, env_kwargs=env_kwargs)
    as_first, as_second = MatchRecord(), MatchRecord()
    for i in range(games_per_seat):
        winner = play_match(
            game, agent_a, agent_b, seed=seed + i, env_kwargs=env_kwargs
        )
        if winner is None:
            as_first.draws += 1
        elif winner == 0:
            as_first.wins += 1
        else:
            as_first.losses += 1
    for i in range(games_per_seat):
        winner = play_match(
            game, agent_b, agent_a, seed=seed + games_per_seat + i,
            env_kwargs=env_kwargs,
        )
        if winner is None:
            as_second.draws += 1
        elif winner == 1:
            as_second.wins += 1
        else:
            as_second.losses += 1
    combined = MatchRecord()
    combined.add(as_first)
    combined.add(as_second)
    return {
        "as_first": as_first.to_dict(),
        "as_second": as_second.to_dict(),
        "combined": combined.to_dict(),
    }


def run_tournament(
    game: str,
    agent_spec: str,
    opponent_specs: list[str],
    *,
    games_per_seat: int = 200,
    seed: int = 1_000_000,
    include_yardstick_matches: bool = False,
    env_kwargs: dict | None = None,
) -> dict:
    """The learner against each opponent, plus (optionally) the
    yardsticks against each other, for ladder calibration."""
    agent = resolve_agent(agent_spec, game, env_kwargs=env_kwargs)
    opponents = {
        spec: resolve_agent(spec, game, env_kwargs=env_kwargs)
        for spec in opponent_specs
    }
    pairings: dict[str, dict] = {}
    for j, (spec, opp) in enumerate(opponents.items()):
        pairings[spec] = play_pairing(
            game,
            agent,
            opp,
            games_per_seat=games_per_seat,
            seed=seed + 10_000 * j,
            env_kwargs=env_kwargs,
        )
    yardsticks: dict[str, dict] = {}
    if include_yardstick_matches:
        specs = list(opponents)
        for a in range(len(specs)):
            for b in range(a + 1, len(specs)):
                key = f"{specs[a]} vs {specs[b]}"
                yardsticks[key] = play_pairing(
                    game,
                    opponents[specs[a]],
                    opponents[specs[b]],
                    games_per_seat=games_per_seat,
                    seed=seed + 500_000 + 10_000 * (a * len(specs) + b),
                    env_kwargs=env_kwargs,
                )
    # First-player advantage across the learner's pairings: of all
    # decisive games, how often did the seat that moved first win?
    # (as_first.wins are seat-0 wins in one half; as_second.losses
    # are seat-0 wins in the other.)
    fp_wins = sum(
        p["as_first"]["wins"] + p["as_second"]["losses"]
        for p in pairings.values()
    )
    decisive = sum(
        p["combined"]["wins"] + p["combined"]["losses"]
        for p in pairings.values()
    )
    return {
        "game": game,
        "agent": agent_spec,
        "games_per_seat": games_per_seat,
        "seed": seed,
        "pairings": pairings,
        "yardstick_matches": yardsticks,
        ("seat0_advantage" if game in SIMULTANEOUS_GAMES else "first_player_advantage"): {
            "decisive_games": decisive,
            ("seat0_wins" if game in SIMULTANEOUS_GAMES else "first_player_wins"): fp_wins,
            ("seat0_win_rate" if game in SIMULTANEOUS_GAMES else "first_player_win_rate"): (fp_wins / decisive) if decisive else 0.0,
        },
    }


def _format_record(rec: dict) -> str:
    return (
        f"W {rec['wins']:>4} / D {rec['draws']:>4} / L {rec['losses']:>4}"
        f"   win {rec['win_rate']:5.1%}  score {rec['score_rate']:5.1%}"
    )


def format_report(result: dict) -> str:
    lines = [
        f"Tournament — {result['game']} — agent: {result['agent']}",
        f"({result['games_per_seat']} games per seat, both seats played)",
    ]
    for spec, pairing in result["pairings"].items():
        lines.append(f"\nvs {spec}:")
        labels = ("seat 0", "seat 1") if result["game"] in SIMULTANEOUS_GAMES else ("as first", "as second")
        lines.append(f"  {labels[0]}: {_format_record(pairing['as_first'])}")
        lines.append(f"  {labels[1]}: {_format_record(pairing['as_second'])}")
        lines.append(f"  combined : {_format_record(pairing['combined'])}")
    for key, pairing in result.get("yardstick_matches", {}).items():
        lines.append(f"\nyardsticks — {key} (row = first-named):")
        lines.append(f"  combined : {_format_record(pairing['combined'])}")
    fpa = result.get("first_player_advantage")
    if result.get("seat0_advantage"):
        seat = result["seat0_advantage"]
        lines.append(f"\nseat-0 win share: {seat['seat0_win_rate']:.1%} of {seat['decisive_games']} decisive games (simultaneous play)")
    if fpa:
        lines.append(
            f"\nfirst-player advantage: the opening seat won "
            f"{fpa['first_player_win_rate']:.1%} of "
            f"{fpa['decisive_games']} decisive games"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Round-robin tournament for a two-player game."
    )
    parser.add_argument("--game", required=True)
    parser.add_argument(
        "--agent", required=True, help="Baseline name or checkpoint path."
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        required=True,
        help="Baseline names or checkpoint paths.",
    )
    parser.add_argument("--games-per-seat", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1_000_000)
    parser.add_argument(
        "--include-yardstick-matches",
        action="store_true",
        help="Also play the opponents against each other.",
    )
    parser.add_argument("--json", action="store_true", help="Print raw JSON.")
    args = parser.parse_args(argv)

    result = run_tournament(
        args.game,
        args.agent,
        args.opponents,
        games_per_seat=args.games_per_seat,
        seed=args.seed,
        include_yardstick_matches=args.include_yardstick_matches,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))


if __name__ == "__main__":
    main()
