"""Training diagnostics: SB3's own logger values, kept on disk.

The trainer has always logged episode returns (``train_log.csv``)
and eval scores (``evaluations.npz``), but the quantities that say
*why* a run went wrong — value loss, explained variance, approx KL,
clip fraction, entropy — were computed by SB3's logger and thrown
away (``verbose=0`` and no output format means ``Logger.dump``
clears them into the void every iteration).

This module keeps them. Two pieces, both attached by the trainer
for every run:

* :class:`DiagnosticsCSVFormat` — an SB3 logger output format
  (the canonical extension seam, the same one SB3's own
  ``CSVOutputFormat`` uses) that appends every dumped key/value
  to ``diagnostics.csv`` in long format (``step,key,value``).
* :class:`DiagnosticsCallback` — installs the format on the
  model's logger at training start and, once per rollout, adds
  the quantities SB3 does not log at all: the Gaussian policy's
  action std (``diag/policy_std``), the L2 norm of the policy
  parameter change since the previous rollout
  (``diag/param_delta`` — an update-size proxy for the gradient
  norms SB3 never exposes), and the optimizer's current learning
  rate (``diag/learning_rate``).

Why this exists (M8): racing PPO's peak-and-collapse was visible
for two milestones only as an eval curve; the mechanism question
— value chase? entropy death? KL blow-up? — was unanswerable
from the artifacts the runs left behind. Diagnostics are cheap
(one small CSV row set per rollout) and are now part of what a
run *is*, so the next collapse arrives pre-autopsied.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter, filter_excluded_keys


class DiagnosticsCSVFormat(KVWriter):
    """Logger output format writing ``step,key,value`` rows.

    Long format (one row per key) because the set of keys SB3 logs
    differs per algorithm family (on-policy ``train/*`` vs
    off-policy ``train/*`` plus ``rollout/*``) and grows between
    versions; a wide CSV would need the header-rewrite dance
    SB3's own CSV format does. Long format just appends.
    """

    def __init__(self, path: Path) -> None:
        self._fh = path.open("a", newline="")
        self._writer = csv.writer(self._fh)
        if path.stat().st_size == 0:
            self._writer.writerow(["step", "key", "value"])
            self._fh.flush()

    def write(self, key_values, key_excluded, step: int = 0) -> None:
        values = filter_excluded_keys(key_values, key_excluded, "csv")
        for key, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue  # strings (log lines etc.) are not diagnostics
            self._writer.writerow([step, key, number])
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _policy_std(model) -> float | None:
    """Mean action std of the policy's distribution, if it has one.

    PPO/MaskablePPO use a state-independent Gaussian whose
    ``log_std`` parameter lives on the policy's action
    distribution. Anything else (DQN's epsilon policy, SAC's
    state-dependent std head) returns None and the row is
    simply absent.
    """
    for owner in (getattr(model, "policy", None), getattr(model, "actor", None)):
        if owner is None:
            continue
        for holder in (owner, getattr(owner, "action_dist", None)):
            log_std = getattr(holder, "log_std", None)
            if isinstance(log_std, torch.Tensor):
                return float(log_std.detach().exp().mean())
    return None


class DiagnosticsCallback(BaseCallback):
    """Attach diagnostics capture to a model's training run.

    ``min_interval`` throttles the callback's own rows: on-policy
    algorithms roll out thousands of steps at a time so every
    rollout end is right, but off-policy algorithms "roll out"
    a handful of steps at a time and would drown the file (and
    pay a parameter-norm computation per env step).
    """

    def __init__(self, path: str | Path, min_interval: int = 512) -> None:
        super().__init__()
        self._path = Path(path)
        self._min_interval = min_interval
        self._format: DiagnosticsCSVFormat | None = None
        self._fh = None
        self._writer = None
        self._last_step = 0
        self._prev_params: list[torch.Tensor] | None = None

    # -- logger-format side -------------------------------------------------
    def _on_training_start(self) -> None:
        self._format = DiagnosticsCSVFormat(self._path)
        self.model.logger.output_formats.append(self._format)
        self._fh = self._path.open("a", newline="")
        self._writer = csv.writer(self._fh)
        self._snapshot_params()

    def _on_training_end(self) -> None:
        if self._format is not None:
            self._format.close()
        if self._fh is not None:
            self._fh.close()

    # -- callback-computed rows ---------------------------------------------
    def _snapshot_params(self) -> None:
        policy = getattr(self.model, "policy", None)
        if policy is None:
            self._prev_params = None
            return
        self._prev_params = [
            p.detach().clone() for p in policy.parameters()
        ]

    def _param_delta(self) -> float | None:
        policy = getattr(self.model, "policy", None)
        if policy is None or self._prev_params is None:
            return None
        total = 0.0
        for prev, cur in zip(self._prev_params, policy.parameters()):
            total += float(((cur.detach() - prev) ** 2).sum())
        return math.sqrt(total)

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._last_step < self._min_interval:
            return
        self._last_step = self.num_timesteps
        if self._writer is None:
            return
        rows: list[tuple[str, float]] = []
        std = _policy_std(self.model)
        if std is not None:
            rows.append(("diag/policy_std", std))
        delta = self._param_delta()
        if delta is not None:
            rows.append(("diag/param_delta", delta))
        self._snapshot_params()
        policy = getattr(self.model, "policy", None)
        optimizer = getattr(policy, "optimizer", None)
        if optimizer is not None and optimizer.param_groups:
            rows.append(
                ("diag/learning_rate", float(optimizer.param_groups[0]["lr"]))
            )
        for key, value in rows:
            self._writer.writerow([self.num_timesteps, key, value])
        self._fh.flush()

    def _on_step(self) -> bool:
        return True
