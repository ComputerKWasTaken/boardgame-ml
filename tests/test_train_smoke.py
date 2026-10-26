"""End-to-end smoke test for train().

Unit tests cover envs, config plumbing, and model construction, but
before this test nothing exercised the full trainer integration —
vectorized envs x ActionMasker x EvalCallback x CSV logger x summary —
so an in-range SB3/gymnasium upgrade could break training while the
whole suite stayed green. This runs a real (tiny) training and asserts
every artifact the pipeline promises.
"""

from __future__ import annotations

import json
from pathlib import Path

from boardgame_ml.eval.evaluate import evaluate, load_sb3_model
from boardgame_ml.training.config import ExperimentConfig
from boardgame_ml.training.train import train


def test_train_end_to_end(tmp_path):
    cfg = ExperimentConfig(
        game="blackjack",
        algorithm="maskable_ppo",
        total_timesteps=256,
        n_envs=2,
        eval_freq=128,
        eval_episodes=10,
        seed=3,
        output_dir=str(tmp_path),
    )
    summary = train(cfg)
    run_dir = Path(summary["run_dir"])
    assert run_dir.parent == tmp_path
    assert "_" in run_dir.name  # carries the config-hash suffix

    summary_file = json.loads((run_dir / "summary.json").read_text())
    assert summary_file["best_eval_mean_return"] is not None
    assert summary_file["versions"]["boardgame_ml"]
    assert summary_file["versions"]["stable_baselines3"]

    # The best-by-eval checkpoint exists and loads through the eval path.
    model = load_sb3_model(summary["best_model"])
    assert model is not None

    # The episode logger produced a header plus at least one data row.
    lines = (run_dir / "train_log.csv").read_text().strip().splitlines()
    assert lines[0] == "timesteps,episode_return,episode_length"
    assert len(lines) >= 2

    # And the checkpoint evaluates through the public harness.
    metrics = evaluate("blackjack", summary["best_model"], episodes=5, seed=999)
    assert metrics["episodes"] == 5
    assert "mean_return" in metrics


def test_train_match3_with_masked_eval(tmp_path):
    """REGRESSION (M3): training a real-mask game must survive its evals.

    SB3's plain EvalCallback predicts *unmasked*; on Match 3 (the
    first game where most actions are illegal) an unmasked
    deterministic prediction can be a rejected swap on the same state
    forever, and the eval episode never ends — training hung at the
    first eval boundary. The trainer now uses MaskableEvalCallback
    for MaskablePPO. This test uses a small eval_freq so an eval
    definitely fires, and asserts the run completes with a best model.
    """
    cfg = ExperimentConfig(
        game="match3",
        algorithm="maskable_ppo",
        policy="MultiInputPolicy",
        total_timesteps=512,
        n_envs=2,
        eval_freq=128,
        eval_episodes=5,
        seed=3,
        output_dir=str(tmp_path),
    )
    summary = train(cfg)
    run_dir = Path(summary["run_dir"])
    assert Path(summary["best_model"]).exists()
    summary_file = json.loads((run_dir / "summary.json").read_text())
    assert summary_file["best_eval_mean_return"] is not None
    assert summary_file["best_eval_mean_return"] > 0


def test_train_racing_ppo_smoke(tmp_path):
    """A Box-action game trains through the standard pipeline: no
    ActionMasker (there are no masks), standard EvalCallback,
    best-by-eval artifacts on disk."""
    cfg = ExperimentConfig(
        game="racing",
        algorithm="ppo",
        total_timesteps=256,
        n_envs=2,
        eval_freq=128,
        eval_episodes=3,
        seed=3,
        output_dir=str(tmp_path),
    )
    summary = train(cfg)
    run_dir = Path(summary["run_dir"])
    assert Path(summary["best_model"]).exists()
    summary_file = json.loads((run_dir / "summary.json").read_text())
    assert summary_file["best_eval_mean_return"] is not None


def test_train_writes_diagnostics(tmp_path):
    """M8: every run keeps SB3's logger diagnostics on disk.

    Before this, value loss / explained variance / KL / clip
    fraction were dumped into the void (verbose=0, no output
    format), which is why racing PPO's collapse could be seen on
    the eval curve but never explained from artifacts. A tiny
    racing PPO run (Gaussian policy, so the callback-computed
    diag/* rows exist too) must leave a parseable diagnostics.csv
    carrying both channels."""
    import csv

    # Budget note: 2 envs x 2048 steps each guarantees every env
    # completes at least one episode (racing truncates at 1800),
    # so rollout/ep_rew_mean is always logged — at 1024 steps
    # whether an episode ends inside the run is trajectory luck.
    cfg = ExperimentConfig(
        game="racing",
        algorithm="ppo",
        total_timesteps=4096,
        n_envs=2,
        eval_freq=2048,
        eval_episodes=2,
        seed=3,
        output_dir=str(tmp_path),
        hyperparams={"n_steps": 256, "batch_size": 128, "n_epochs": 1},
    )
    summary = train(cfg)
    run_dir = Path(summary["run_dir"])
    path = run_dir / "diagnostics.csv"
    assert path.exists()
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["step", "key", "value"]
    keys = set()
    for step, key, value in rows[1:]:
        int(step)  # every row carries a numeric step...
        float(value)  # ...and a numeric value
        keys.add(key)
    # SB3 logger channel (via the KVWriter format) ...
    assert "train/value_loss" in keys
    assert "train/approx_kl" in keys
    assert "rollout/ep_rew_mean" in keys
    # ... and the callback-computed channel (SB3 logs none of these).
    assert "diag/policy_std" in keys
    assert "diag/param_delta" in keys
    assert "diag/learning_rate" in keys
    """SAC — registered in M5 as the sample-efficient continuous-
    control alternative — constructs and trains on the Box game
    through the same pipeline (a few hundred steps only: this pins
    the wiring, not SAC's driving)."""
    cfg = ExperimentConfig(
        game="racing",
        algorithm="sac",
        total_timesteps=300,
        n_envs=1,
        eval_freq=150,
        eval_episodes=3,
        seed=3,
        output_dir=str(tmp_path),
    )
    summary = train(cfg)
    assert Path(summary["best_model"]).exists()
