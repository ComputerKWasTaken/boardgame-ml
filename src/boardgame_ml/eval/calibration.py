"""Benchmark calibration (M10): the pipeline, measured against the world.

    python -m boardgame_ml.eval.calibration                 # all suites
    python -m boardgame_ml.eval.calibration --suite cartpole
    python -m boardgame_ml.eval.calibration --suite lunar_lander --json-out cal.json

Every other number in this project is self-referential: our envs,
our harness, our yardsticks. That is how the M8 std bug survived
two milestones — racing PPO's collapse was visible only because
its *shape* looked wrong, never because an external anchor said
the pipeline itself was miscalibrated. Calibration is the anchor:
train **our** ``train()`` (not a hand-rolled SB3 script — the
plumbing is the thing under test) on standard Gymnasium classics
through the :class:`~boardgame_ml.envs.gym_classic.GymClassicEnv`
shim, with the RL Baselines3 Zoo's canonical PPO hyperparameters,
and compare the result against the Zoo's published outcomes for
the same env and algorithm.

Suites (Zoo ``hyperparams/ppo.yml``, mirrored by the SB3 org's
trained-agent cards on Hugging Face):

* **CartPole-v1** — 100k steps, 8 envs. Zoo agent scores the
  500 maximum; the solved bar is a 475+ mean. One deviation,
  forced by the framework and recorded here rather than hidden:
  the Zoo decays the learning rate (``lin_0.001``) and clip range
  (``lin_0.2``) linearly, while ``ExperimentConfig`` carries a
  scalar learning rate (schedules would be trainer surgery), so
  the suite runs the constant 1e-3 / 0.2. Every other value is
  verbatim.
* **LunarLander** — 1M steps, 16 envs, scalar hyperparameters
  throughout (no deviation). Gymnasium 0.29 (this project's pin)
  registers the env as ``LunarLander-v2``; the Zoo renamed its
  entry to ``-v3`` when Gymnasium 1.0 renamed the env — same env,
  same hyperparameters, same published scores: Zoo-trained agents
  report 257.67 ± 24.70 and 262.07 ± 21.35 mean return on the hub;
  the env's own solved bar is 200. Needs the Box2D dependency
  (see ``envs/gym_classic.py``).

The verdict protocol per suite: after training, the run's
best-by-eval checkpoint is re-measured by the framework's own
independent evaluator (100 deterministic episodes, fixed seeds
the training never used) and read against the published figure:
**in ballpark** = at/above the solved bar and within a plausible
spread of the reference; anything else is named as what it is.
A miniature version of the CartPole suite runs in the pytest
suite (``tests/test_calibration.py``) as a tripwire, so a future
training-code change that breaks the pipeline's ability to learn
fails CI-sized, not milestone-sized.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from boardgame_ml.eval.evaluate import evaluate
from boardgame_ml.training.config import ExperimentConfig
from boardgame_ml.training.train import train

#: Fixed seed base for the independent post-training measurement —
#: deliberately far from every training/eval seed in the pipeline.
REPORT_SEED = 2_024_001
REPORT_EPISODES = 100


@dataclass(frozen=True)
class CalibrationSuite:
    """One calibration target: a Gymnasium classic, the Zoo's PPO
    recipe for it, and the published outcome to be read against."""

    name: str
    env_id: str
    total_timesteps: int
    n_envs: int
    learning_rate: float
    gamma: float
    eval_freq: int
    eval_episodes: int  # during training (best-by-eval selection)
    hyperparams: dict
    reference_mean: float
    reference_note: str
    solved_bar: float

    def build_config(
        self,
        *,
        seed: int = 0,
        output_dir: str = "runs",
        total_timesteps: int | None = None,
    ) -> ExperimentConfig:
        return ExperimentConfig(
            game="gym_classic",
            algorithm="ppo",
            total_timesteps=(
                self.total_timesteps if total_timesteps is None else total_timesteps
            ),
            seed=seed,
            n_envs=self.n_envs,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            env_kwargs={"env_id": self.env_id},
            eval_freq=self.eval_freq,
            eval_episodes=self.eval_episodes,
            output_dir=output_dir,
            hyperparams=dict(self.hyperparams),
        )


SUITES: dict[str, CalibrationSuite] = {
    "cartpole": CalibrationSuite(
        name="cartpole",
        env_id="CartPole-v1",
        total_timesteps=100_000,
        n_envs=8,
        learning_rate=1e-3,  # Zoo: lin_0.001 (see module docstring)
        gamma=0.98,
        eval_freq=10_000,
        eval_episodes=20,
        hyperparams={
            "n_steps": 32,
            "batch_size": 256,
            "n_epochs": 20,
            "gae_lambda": 0.8,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "vf_coef": 0.5,
        },
        reference_mean=500.0,
        reference_note=(
            "Zoo PPO agent scores the 500 cap (sb3/ppo-CartPole-v1 hub card)"
        ),
        solved_bar=475.0,
    ),
    "lunar_lander": CalibrationSuite(
        name="lunar_lander",
        env_id="LunarLander-v2",
        total_timesteps=1_000_000,
        n_envs=16,
        learning_rate=2.5e-4,
        gamma=0.999,
        eval_freq=50_000,
        eval_episodes=20,
        hyperparams={
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 4,
            "gae_lambda": 0.98,
            "ent_coef": 0.01,
        },
        reference_mean=260.0,
        reference_note=(
            "Zoo-trained agents on the hub: 257.67 ± 24.70 and "
            "262.07 ± 21.35 mean return; env solved bar is 200"
        ),
        solved_bar=200.0,
    ),
}


def run_suite(
    name: str,
    *,
    seed: int = 0,
    output_dir: str = "runs",
    total_timesteps: int | None = None,
    report_episodes: int = REPORT_EPISODES,
) -> dict:
    """Train one suite through ``train()`` and measure the product.

    Returns the calibration record: the training summary's best
    eval, an independent deterministic re-measurement of the
    best checkpoint, the published reference, and the verdict.
    """
    suite = SUITES[name]
    config = suite.build_config(
        seed=seed, output_dir=output_dir, total_timesteps=total_timesteps
    )
    summary = train(config)
    measured = evaluate(
        "gym_classic",
        summary["best_model"],
        episodes=report_episodes,
        seed=REPORT_SEED,
        env_kwargs={"env_id": suite.env_id},
    )
    mean = measured["mean_return"]
    in_ballpark = mean >= suite.solved_bar
    return {
        "suite": suite.name,
        "env_id": suite.env_id,
        "algorithm": "ppo",
        "seed": seed,
        "total_timesteps": config.total_timesteps,
        "run_dir": summary["run_dir"],
        "train_seconds": summary["train_seconds"],
        "best_eval_mean_return": summary["best_eval_mean_return"],
        "measured_mean_return": mean,
        "measured_std_return": measured["std_return"],
        "measured_episodes": report_episodes,
        "reference_mean": suite.reference_mean,
        "reference_note": suite.reference_note,
        "solved_bar": suite.solved_bar,
        "verdict": "in ballpark" if in_ballpark else "BELOW BAR",
    }


def format_report(results: list[dict]) -> str:
    lines = ["Benchmark calibration — train() vs published Zoo PPO results"]
    for r in results:
        lines.append(
            f"\n{r['env_id']} (seed {r['seed']}, {r['total_timesteps']} steps, "
            f"{r['train_seconds']:.0f}s)"
        )
        lines.append(
            f"  measured (best ckpt, {r['measured_episodes']} det. eps): "
            f"{r['measured_mean_return']:.1f} ± {r['measured_std_return']:.1f}"
        )
        lines.append(
            f"  published reference: {r['reference_mean']:.1f} — {r['reference_note']}"
        )
        lines.append(f"  verdict: {r['verdict']} (bar {r['solved_bar']:.0f})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate train() against published Zoo results."
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default=None,
        help="Run only this suite (default: all).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Override total_timesteps (non-canonical; for smoke runs).",
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    names = [args.suite] if args.suite else sorted(SUITES)
    results = [
        run_suite(
            name,
            seed=args.seed,
            output_dir=args.output_dir,
            total_timesteps=args.budget,
        )
        for name in names
    ]
    print(format_report(results))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
