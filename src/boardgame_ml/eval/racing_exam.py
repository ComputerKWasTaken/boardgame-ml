"""Racing generalization exam (M7): fixed held-out tracks, many agents.

    python -m boardgame_ml.eval.racing_exam \
        --agents runs/racing_sac_s0_d765d4d3/best_model.zip,racing_heuristic \
        --track-seeds 50001,50002,50003,50004,50005 --episodes 20

The M5 question — did the drivers learn *driving*, or did they
memorize one near-oval? — is an exam question, and an exam needs
fixed papers: a set of generator-v2 tracks whose seeds appear in no
training range, sat by every candidate under identical conditions
(same tracks, same episode-seed schedule, same spawn sequence).
Each agent is a checkpoint path or a baseline name, resolved through
the framework's usual ``resolve_agent``; the heuristic sits the same
exam as the **track-agnostic control** — it has never seen any track
either, so its score is what "reactively driving the geometry in
front of you" is worth, and the learned agents are judged against
it as much as against each other.

Per episode the harness records progress (metres and capped lap
fraction), laps completed, lap times, and whether the episode ended
in a crash (off-track termination) or the step cap. Aggregates are
reported per track and pooled; pooled is what the M7 acceptance
criteria read.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from boardgame_ml.envs import make_env
from boardgame_ml.envs.racing import track_shape_stats
from boardgame_ml.eval.evaluate import predict_action, resolve_agent

#: The held-out M7 exam set: generator-v2 seeds in neither the
#: training range [10000, 11000) nor any other run's range. Fixed in
#: code (and published in the README) so the exam cannot drift.
EXAM_TRACK_SEEDS: tuple[int, ...] = (50001, 50002, 50003, 50004, 50005)

#: Episode seeds are exam_base + track_index * 1000 + episode, so
#: every agent faces the same spawn sequence on the same paper.
EXAM_EPISODE_SEED_BASE = 777_000


def run_exam_episodes(
    agent,
    env,
    episodes: int,
    *,
    seed_base: int,
) -> dict:
    """Run ``episodes`` greedy episodes of ``agent`` on ``env``'s
    (fixed) track and return per-episode records + aggregates."""
    total_length = float(env.unwrapped.track.total_length)
    records = []
    for ep in range(episodes):
        obs, _info = env.reset(seed=seed_base + ep)
        done = False
        lap_times: list[float] = []
        info = _info
        while not done:
            action = predict_action(agent, env, obs)
            obs, _reward, terminated, truncated, info = env.step(action)
            if info.get("lap_completed"):
                lap_times.append(float(info["last_lap_time"]))
            done = terminated or truncated
        progress_m = float(info["progress_m"])
        records.append(
            {
                "progress_m": progress_m,
                "progress_fraction": min(progress_m, total_length) / total_length,
                "laps": int(info["laps"]),
                "crashed": bool(terminated),
                "lap_times": lap_times,
            }
        )
    all_lap_times = [t for r in records for t in r["lap_times"]]
    return {
        "episodes": episodes,
        "mean_progress_fraction": float(
            np.mean([r["progress_fraction"] for r in records])
        ),
        "mean_progress_m": float(np.mean([r["progress_m"] for r in records])),
        "lap_rate": float(np.mean([r["laps"] >= 1 for r in records])),
        "mean_laps": float(np.mean([r["laps"] for r in records])),
        "crash_rate": float(np.mean([r["crashed"] for r in records])),
        "mean_lap_time": float(np.mean(all_lap_times)) if all_lap_times else None,
        "best_lap_time": float(np.min(all_lap_times)) if all_lap_times else None,
        "records": records,
    }


def run_exam(
    agent_specs: list[str],
    track_seeds: list[int] | tuple[int, ...] = EXAM_TRACK_SEEDS,
    *,
    episodes_per_track: int = 20,
    seed_base: int = EXAM_EPISODE_SEED_BASE,
    track_version: int = 2,
) -> dict:
    """Sit every agent on every exam track; return the full result
    tree (per-agent per-track aggregates + pooled aggregates)."""
    # Build one env per track up front: track geometry + shape stats
    # are part of the exam paper and are reported with the results.
    tracks: dict[int, dict] = {}
    envs: dict[int, object] = {}
    for seed in track_seeds:
        env = make_env(
            "racing", track_seed=int(seed), track_version=track_version
        )
        envs[seed] = env
        raw = env.unwrapped
        tracks[seed] = {
            "seed": int(seed),
            "track_seed_actual": int(raw.track_seed),
            **track_shape_stats(raw.track),
        }

    agents: dict[str, dict] = {}
    for spec in agent_specs:
        per_track: dict[str, dict] = {}
        for i, seed in enumerate(track_seeds):
            env_kwargs = {"track_seed": int(seed), "track_version": track_version}
            agent = resolve_agent(spec, "racing", env_kwargs=env_kwargs)
            per_track[str(seed)] = run_exam_episodes(
                agent,
                envs[seed],
                episodes_per_track,
                seed_base=seed_base + i * 1000,
            )
        pooled_records = [
            r for t in per_track.values() for r in t["records"]
        ]
        pooled_laps = [t for r in pooled_records for t in r["lap_times"]]
        pooled = {
            "episodes": len(pooled_records),
            "mean_progress_fraction": float(
                np.mean([r["progress_fraction"] for r in pooled_records])
            ),
            "mean_progress_m": float(
                np.mean([r["progress_m"] for r in pooled_records])
            ),
            "lap_rate": float(
                np.mean([r["laps"] >= 1 for r in pooled_records])
            ),
            "mean_laps": float(np.mean([r["laps"] for r in pooled_records])),
            "crash_rate": float(
                np.mean([r["crashed"] for r in pooled_records])
            ),
            "mean_lap_time": (
                float(np.mean(pooled_laps)) if pooled_laps else None
            ),
            "best_lap_time": (
                float(np.min(pooled_laps)) if pooled_laps else None
            ),
        }
        agents[spec] = {"per_track": per_track, "pooled": pooled}
    return {
        "track_version": track_version,
        "episodes_per_track": episodes_per_track,
        "seed_base": seed_base,
        "tracks": tracks,
        "agents": agents,
    }


def _fmt(x: float | None, spec: str = ".1f") -> str:
    return "—" if x is None else format(x, spec)


def format_report(result: dict) -> str:
    lines = ["Racing generalization exam — pooled over exam tracks"]
    lines.append(
        f"{'agent':<58} {'prog%':>6} {'lap%':>6} {'crash%':>7} "
        f"{'laps/ep':>7} {'lap mean/best (s)':>18}"
    )
    for spec, data in result["agents"].items():
        p = data["pooled"]
        lap = (
            f"{_fmt(p['mean_lap_time'])}/{_fmt(p['best_lap_time'])}"
            if p["mean_lap_time"] is not None
            else "—"
        )
        lines.append(
            f"{spec:<58} {100 * p['mean_progress_fraction']:>6.1f} "
            f"{100 * p['lap_rate']:>6.1f} {100 * p['crash_rate']:>7.1f} "
            f"{p['mean_laps']:>7.2f} {lap:>18}"
        )
    lines.append("")
    lines.append("Per-track mean progress % / lap-completion %:")
    for seed, stats in result["tracks"].items():
        row = [f"  track {seed} (len {stats['length_m']:.0f} m, "
               f"minR {stats['min_turn_radius_m']:.0f} m):"]
        for spec, data in result["agents"].items():
            t = data["per_track"][str(seed)]
            row.append(
                f"{100 * t['mean_progress_fraction']:.0f}/"
                f"{100 * t['lap_rate']:.0f}"
            )
        lines.append("  ".join(row))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Racing generalization exam on held-out v2 tracks."
    )
    parser.add_argument(
        "--agents",
        required=True,
        help="Comma-separated checkpoint paths or baseline names.",
    )
    parser.add_argument(
        "--track-seeds",
        default=",".join(str(s) for s in EXAM_TRACK_SEEDS),
        help="Comma-separated generator track seeds (the exam paper).",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=EXAM_EPISODE_SEED_BASE)
    parser.add_argument(
        "--track-version",
        type=int,
        default=2,
        help="Track generator family to examine on (2 = the M7 exam; "
        "3 = the M9 cross-family exam).",
    )
    parser.add_argument(
        "--json-out", default=None, help="Also write the full result tree here."
    )
    args = parser.parse_args(argv)

    result = run_exam(
        [s.strip() for s in args.agents.split(",") if s.strip()],
        [int(s) for s in args.track_seeds.split(",") if s.strip()],
        episodes_per_track=args.episodes,
        seed_base=args.seed_base,
        track_version=args.track_version,
    )
    print(format_report(result))
    if args.json_out:
        slim = json.loads(json.dumps(result))
        for agent in slim["agents"].values():
            for track in agent["per_track"].values():
                track.pop("records", None)
        with open(args.json_out, "w") as fh:
            json.dump(slim, fh, indent=2)


if __name__ == "__main__":
    main()
