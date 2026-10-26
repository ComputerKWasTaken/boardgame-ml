"""Behavior cloning (BC) for Pong: imitate the intercept heuristic, then RL.

    python -m boardgame_ml.training.behavior_cloning generate \
        --output ~/workspace/boardgame-ml-pongbc/demos.npz --pairs 400000
    python -m boardgame_ml.training.behavior_cloning train \
        --config configs/pong_ppo.yaml \
        --dataset ~/workspace/boardgame-ml-pongbc/demos.npz \
        --output ~/workspace/boardgame-ml-pongbc/bc_model

Why this exists: four 250k-step PPO runs scored zero points against the
``pong_intercept`` baseline in 1,200 exam matches (docs/PONG.md). The
hypothesis under test is that a tabula-rasa MLP cannot discover
trajectory prediction from a sparse +/-1 point signal. BC replaces the
discovery problem with an imitation problem: the teacher is a
programmatic baseline, so demonstrations are unlimited and — crucially
for the DAgger slice — the teacher can label *any* visited state for
free, including states an epsilon-noised teacher drifts into.

Dataset contract (the part that must be exactly right): every logged
pair is ``(observation, action)`` where the observation is the
*per-seat* observation ``PongEnv.observe(seat)`` — already horizontally
reflected for seat 1, i.e. precisely what a policy sitting in that seat
consumes — and the action is the teacher's action *for that
observation*, computed through the seat's own ``player_view``. In the
DAgger slice the executed action is noised but the label is always the
true teacher action for the visited state.

The BC checkpoint is an ordinary SB3 checkpoint (``model.save()`` of a
model built by :func:`boardgame_ml.training.train.build_model`), so
the existing ``init_checkpoint`` warm-start path consumes it unchanged
for PPO fine-tuning. Only the policy (actor) is supervised; the value
head keeps its random init and is left for PPO to adapt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from boardgame_ml.agents.baselines import RandomAgent, get_baseline
from boardgame_ml.envs.pong import PongEnv
from boardgame_ml.training.config import ExperimentConfig

TEACHER_VS_TEACHER = 0
TEACHER_VS_TRACKER = 1
TEACHER_VS_RANDOM = 2
DAGGER = 3
SOURCE_NAMES = {
    TEACHER_VS_TEACHER: "teacher_vs_teacher",
    TEACHER_VS_TRACKER: "teacher_vs_tracker",
    TEACHER_VS_RANDOM: "teacher_vs_random",
    DAGGER: "dagger_noised_teacher",
}
# Bucket name per source code, fractions of the total pair target.
DEFAULT_COMPOSITION = {
    TEACHER_VS_TEACHER: 0.30,
    TEACHER_VS_TRACKER: 0.25,
    TEACHER_VS_RANDOM: 0.25,
    DAGGER: 0.20,
}


def _teacher_action(teacher, env: PongEnv, seat: int, obs: np.ndarray) -> int:
    """The teacher's action for ``obs`` — exactly the baseline path the
    framework itself uses (per-seat player view + the seat's own,
    already-reflected observation)."""
    return int(teacher.select_action(env.player_view(seat), obs))


def _play_match(
    env: PongEnv,
    *,
    seed: int,
    seat_kinds: tuple[str, str],
    source: int,
    teacher,
    epsilon: float,
    rng: np.random.Generator,
    collect: dict[str, list],
) -> dict:
    """Play one match, logging (obs, teacher_action) for labeled seats.

    A seat is *labeled* when its kind is ``teacher`` or
    ``noised_teacher``; the label is always the true teacher action for
    the visited state. The noised seat *executes* a uniform random
    action with probability ``epsilon`` (DAgger-style state coverage),
    which never changes its labels.
    """
    obs, _info = env.reset(seed=seed)
    labeled = [kind in ("teacher", "noised_teacher") for kind in seat_kinds]
    points = [0, 0]
    while True:
        actions = [0, 0]
        for seat in (0, 1):
            kind = seat_kinds[seat]
            if labeled[seat]:
                label = _teacher_action(teacher, env, seat, obs[seat])
                collect["obs"].append(obs[seat])
                collect["actions"].append(label)
                collect["seats"].append(seat)
                collect["sources"].append(source)
                if kind == "noised_teacher" and rng.random() < epsilon:
                    actions[seat] = int(rng.integers(0, 6))
                else:
                    actions[seat] = label
            elif kind == "tracker":
                actions[seat] = int(
                    get_baseline("pong_tracker", "pong").select_action(
                        env.player_view(seat), obs[seat]
                    )
                )
            elif kind == "random":
                actions[seat] = int(rng.integers(0, 6))
            else:  # pragma: no cover - guarded by the schedule below
                raise ValueError(f"Unknown seat kind {kind!r}")
        obs, _rewards, done, _truncated, info = env.step(tuple(actions))
        if done:
            points = list(env.scores)
            winner = info["winner"]
            break
    return {"winner": winner, "points": points, "labeled": labeled}


def generate_demonstrations(
    *,
    target_pairs: int = 400_000,
    seed: int = 777_000,
    epsilon: float = 0.1,
    composition: dict[int, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Roll out teacher matches and return (dataset arrays, stats).

    Buckets fill independently: matches in a bucket stop being started
    once the bucket's pair target is met (a match in flight always
    finishes, so buckets can overshoot by one match).
    """
    composition = composition or DEFAULT_COMPOSITION
    if set(composition) != set(SOURCE_NAMES):
        raise ValueError(f"composition keys must be {sorted(SOURCE_NAMES)}")
    if abs(sum(composition.values()) - 1.0) > 1e-9:
        raise ValueError("composition fractions must sum to 1")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    teacher = get_baseline("pong_intercept", "pong")
    rng = np.random.default_rng(seed)
    env = PongEnv()
    collect: dict[str, list] = {"obs": [], "actions": [], "seats": [], "sources": []}
    stats: dict[str, dict] = {
        SOURCE_NAMES[code]: {
            "matches": 0,
            "pairs": 0,
            "labeled_seat_wins": 0,
            "labeled_seat_losses": 0,
            "draws": 0,
            "points_for": 0,
            "points_against": 0,
        }
        for code in SOURCE_NAMES
    }
    # Matchup schedule per bucket: a function of the bucket's match
    # index, so teacher seats alternate for asymmetric matchups.
    schedules = {
        TEACHER_VS_TEACHER: lambda i: ("teacher", "teacher"),
        TEACHER_VS_TRACKER: lambda i: (
            ("teacher", "tracker") if i % 2 == 0 else ("tracker", "teacher")
        ),
        TEACHER_VS_RANDOM: lambda i: (
            ("teacher", "random") if i % 2 == 0 else ("random", "teacher")
        ),
        DAGGER: lambda i: (
            ("noised_teacher", "tracker")
            if i % 5 < 3
            else (("noised_teacher", "random") if i % 2 == 0 else ("random", "noised_teacher"))
        ),
    }
    match_seed = seed
    for code, fraction in composition.items():
        bucket_target = int(round(target_pairs * fraction))
        bucket = stats[SOURCE_NAMES[code]]
        match_index = 0
        while bucket["pairs"] < bucket_target:
            kinds = schedules[code](match_index)
            before = len(collect["obs"])
            result = _play_match(
                env,
                seed=match_seed,
                seat_kinds=kinds,
                source=code,
                teacher=teacher,
                epsilon=epsilon if code == DAGGER else 0.0,
                rng=rng,
                collect=collect,
            )
            match_seed += 1
            match_index += 1
            bucket["matches"] += 1
            bucket["pairs"] += len(collect["obs"]) - before
            for seat in (0, 1):
                if result["labeled"][seat]:
                    bucket["points_for"] += result["points"][seat]
                    bucket["points_against"] += result["points"][1 - seat]
                    if result["winner"] is None:
                        bucket["draws"] += 1
                    elif result["winner"] == seat:
                        bucket["labeled_seat_wins"] += 1
                    else:
                        bucket["labeled_seat_losses"] += 1
    env.close()
    arrays = {
        "obs": np.asarray(collect["obs"], dtype=np.float32),
        "actions": np.asarray(collect["actions"], dtype=np.int64),
        "seats": np.asarray(collect["seats"], dtype=np.int8),
        "sources": np.asarray(collect["sources"], dtype=np.int8),
    }
    return arrays, stats


def save_dataset(arrays: dict[str, np.ndarray], stats: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    meta = {
        "pairs": int(len(arrays["actions"])),
        "source_names": {str(k): v for k, v in SOURCE_NAMES.items()},
        "per_source_pairs": {
            SOURCE_NAMES[code]: int(np.sum(arrays["sources"] == code))
            for code in SOURCE_NAMES
        },
        "per_seat_pairs": {
            str(seat): int(np.sum(arrays["seats"] == seat)) for seat in (0, 1)
        },
        "buckets": stats,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return path


def train_behavior_clone(
    config: ExperimentConfig,
    arrays: dict[str, np.ndarray],
    *,
    epochs: int = 4,
    batch_size: int = 2048,
    bc_learning_rate: float = 1e-3,
    val_fraction: float = 0.05,
    split_seed: int = 0,
):
    """Supervised-train a fresh framework-built model's policy on demos.

    Returns ``(model, report)``. Cross-entropy over the 6 discrete
    actions (equivalently, negative log-prob under the policy's own
    categorical distribution, so the distribution that is trained is
    the distribution PPO will fine-tune). Gradients reach the actor
    network only; the value head is untouched by construction (the loss
    never evaluates it) and keeps its random init for PPO to adapt.
    """
    from boardgame_ml.training.self_play import SelfPlayConfig
    from boardgame_ml.training.train import build_model, build_vec_env

    obs = arrays["obs"].astype(np.float32)
    actions = arrays["actions"].astype(np.int64)
    n = len(actions)
    rng = np.random.default_rng(split_seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * val_fraction)))
    val_idx, train_idx = order[:n_val], order[n_val:]

    self_play_cfg = (
        SelfPlayConfig.from_dict(config.self_play) if config.self_play else None
    )
    env = build_vec_env(
        config.game,
        config.n_envs,
        config.seed,
        env_kwargs=config.env_kwargs,
        self_play_cfg=self_play_cfg,
    )
    model = build_model(config, env)
    policy = model.policy
    policy.set_training_mode(True)
    for group in policy.optimizer.param_groups:
        group["lr"] = bc_learning_rate
    optimizer = policy.optimizer
    device = policy.device

    def accuracy(indices: np.ndarray) -> float:
        policy.set_training_mode(False)
        correct = 0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                idx = indices[start : start + batch_size]
                batch = torch.as_tensor(obs[idx], device=device)
                dist = policy.get_distribution(batch)
                pred = dist.distribution.probs.argmax(dim=-1).cpu().numpy()
                correct += int(np.sum(pred == actions[idx]))
        policy.set_training_mode(True)
        return correct / max(1, len(indices))

    report: dict = {
        "pairs": n,
        "train_pairs": int(len(train_idx)),
        "val_pairs": int(len(val_idx)),
        "epochs": epochs,
        "batch_size": batch_size,
        "bc_learning_rate": bc_learning_rate,
        "val_accuracy_before": accuracy(val_idx),
        "per_epoch": [],
    }
    for epoch in range(epochs):
        shuffled = train_idx[rng.permutation(len(train_idx))]
        total_loss, batches = 0.0, 0
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start : start + batch_size]
            batch_obs = torch.as_tensor(obs[idx], device=device)
            batch_act = torch.as_tensor(actions[idx], device=device)
            dist = policy.get_distribution(batch_obs)
            loss = -dist.log_prob(batch_act).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        report["per_epoch"].append(
            {
                "epoch": epoch + 1,
                "mean_loss": total_loss / max(1, batches),
                "train_accuracy": accuracy(train_idx[:50_000]),
                "val_accuracy": accuracy(val_idx),
            }
        )
    env.close()
    return model, report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate teacher demonstrations.")
    gen.add_argument("--output", required=True)
    gen.add_argument("--pairs", type=int, default=400_000)
    gen.add_argument("--seed", type=int, default=777_000)
    gen.add_argument("--epsilon", type=float, default=0.1)

    tr = sub.add_parser("train", help="BC-train a policy from a dataset.")
    tr.add_argument("--config", required=True)
    tr.add_argument("--dataset", required=True)
    tr.add_argument("--output", required=True, help="Checkpoint path prefix.")
    tr.add_argument("--epochs", type=int, default=4)
    tr.add_argument("--batch-size", type=int, default=2048)
    tr.add_argument("--bc-lr", type=float, default=1e-3)

    args = parser.parse_args(argv)
    if args.command == "generate":
        start = time.time()
        arrays, stats = generate_demonstrations(
            target_pairs=args.pairs, seed=args.seed, epsilon=args.epsilon
        )
        path = save_dataset(arrays, stats, args.output)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        meta["generation_seconds"] = round(time.time() - start, 1)
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
    else:
        config = ExperimentConfig.from_yaml(args.config)
        data = np.load(args.dataset)
        arrays = {key: data[key] for key in ("obs", "actions", "seats", "sources")}
        start = time.time()
        model, report = train_behavior_clone(
            config,
            arrays,
            epochs=args.epochs,
            batch_size=args.batch_size,
            bc_learning_rate=args.bc_lr,
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        report["bc_seconds"] = round(time.time() - start, 1)
        report["checkpoint"] = str(out.with_suffix(".zip"))
        out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
