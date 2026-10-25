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
