# boardgame-ml

A modular reinforcement-learning framework for teaching agents to play
board and card games. Implement a game behind one small contract,
train an agent on it, evaluate it against reference baselines, and
watch it play.

This is a ground-up rebuild of an earlier project. The concept survived
the autopsy; the code didn't. [DESIGN.md](DESIGN.md) documents the
architecture and maps each legacy failure (dropped action masks in the
PPO update, a config system that reached nothing, reward shaping that
changed the game being learned, "best" checkpoints chosen from
training noise) to the structural decision here that prevents it.

## Milestone 1

* Two games behind the contract: **Blackjack** (terminal-only rewards,
  dealer stands on all 17s, naturals pay 3:2) and **Higher or Lower**
  (honest +1/−1/0 per guess).
* Agents: Stable-Baselines3 **MaskablePPO** (default), PPO, and DQN,
  plus reference baselines — uniform-random, blackjack **basic
  strategy**, and a probability-greedy Higher-or-Lower player.
* Training with typed YAML configs passed explicitly into the trainer,
  periodic deterministic evaluation on a separate env, and
  **best-by-eval** checkpoint selection.
* An evaluator that reports mean return, win/loss/push rates, and (for
  blackjack) agreement with basic strategy — the proof-of-learning
  harness.

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # CPU PyTorch via PyPI

# Train (configs are in configs/)
.venv/bin/python -m boardgame_ml.training.train --config configs/blackjack_maskable_ppo.yaml
.venv/bin/python -m boardgame_ml.training.train --config configs/higher_or_lower_ppo.yaml

# Evaluate: baselines…
.venv/bin/python -m boardgame_ml.eval.evaluate --game blackjack --agent random
.venv/bin/python -m boardgame_ml.eval.evaluate --game blackjack --agent basic_strategy
# …or a trained checkpoint
.venv/bin/python -m boardgame_ml.eval.evaluate --game blackjack \
    --agent runs/blackjack_maskable_ppo_s0/best_model.zip --episodes 5000

# Watch it play (text mode)
.venv/bin/python -m boardgame_ml.play --game blackjack \
    --agent runs/blackjack_maskable_ppo_s0/best_model.zip

# Tests
.venv/bin/python -m pytest
```

## Web cockpit

A local web interface to the framework — browse games and runs,
train agents with live progress, and watch trained agents play —
served by one process on your own machine. Training stays
Python-side (this wraps the framework's trainer; it does not
reimplement it), and runs/checkpoints stay on disk in `runs/`. The
browser's localStorage holds only the personal layer: UI
preferences, last-used train settings, starred runs, saved replays,
and watch history.

```bash
.venv/bin/pip install -e ".[web]"     # FastAPI + uvicorn
cd web && npm install && npm run build && cd ..

.venv/bin/python -m boardgame_ml.web            # http://127.0.0.1:8765
.venv/bin/python -m boardgame_ml.web --open     # …and open the browser
# (or the console script: boardgame-ml-web)
```

Three views:

* **Games & Runs** — game cards with baseline reference numbers
  (the proof-of-learning yardsticks, measured live by the evaluator)
  and a table of runs from `runs/` with best-eval returns; star
  runs, and jump straight to watching any checkpoint or baseline.
* **Train** — game / algorithm / preset (quick · standard · long)
  with timesteps, learning-rate, and seed editable; a live chart of
  episode returns with deterministic eval means as they land (SSE
  stream from the training job). **Stop** ends training gracefully
  and keeps the best-by-eval checkpoint (a job stopped before its
  first eval keeps `final_model.zip` only).
* **Watch** — pick a run checkpoint or a baseline and watch it play
  live (WebSocket-streamed steps; play / pause / step / speed), with
  per-game boards, an action feed, and session stats. **Save
  replay** stores the session's event log in localStorage; replays
  re-play purely client-side, no server stepping needed.

Frontend development: run the backend as above, then
`cd web && npm run dev` — the Vite dev server on :5173 proxies
`/api` (HTTP + WebSocket) to :8765.

## M1 results — proof of learning

All numbers below are **independent evaluations**: greedy (deterministic)
policies over fresh episodes with seeds the training runs never saw,
produced by `boardgame_ml.eval.evaluate`. Baselines and trained agents
were measured with the same harness. After the post-M1 hardening pass
(see "Post-M1 hardening" below), every figure was re-measured in one
batch — 10,000 episodes, seed 777 — with the blackjack checkpoints
unchanged and Higher or Lower retrained under its truncation fix.

### Blackjack (mean return per hand)

| Agent | Mean return | Win / Loss / Push | Basic-strategy agreement |
|---|---|---|---|
| Random | −0.288 | 32.4% / 63.5% / 4.2% | 50.2% (≈ coin flip, as expected) |
| **Basic strategy** (heuristic) | **−0.016** | 43.7% / 47.6% / 8.8% | 100% (by construction) |
| **MaskablePPO**, 300k steps | **−0.039** | 42.7% / 48.9% / 8.4% | **85.1%** (14,487 decisions) |
| **DQN**, 200k steps | **−0.029** | 43.1% / 48.2% / 8.7% | **86.6%** (14,823 decisions) |

Both trained agents sit within ~0.02 of basic strategy in return
(−0.039 and −0.029 vs −0.016, per-agent SEM ≈ 0.01 — and the
basic-strategy figure itself moved by 0.02 between the original
5,000-episode measurement and this 10,000-episode one, so all three
are the same player within measurement noise) and agree with the chart
on 85–87% of decisions; their disagreements concentrate in states
where the chart's edge is smallest. The trained agents' figures
reproduce their M1 measurements to the third decimal. Note on
methodology: the run's *best eval window* during training read
+0.024 — that number selected the checkpoint, but it is the max of
many noisy 500-episode windows and is **not** the headline figure;
the honest estimate is the independent evaluation above.

### Higher or Lower (mean return per 10-turn episode)

| Agent | Mean return |
|---|---|
| Random | +0.01 |
| **Probability-greedy** (heuristic, analytically optimal) | **+4.96** |
| **MaskablePPO**, 60k steps (retrained) | **+4.95** |

The trained agent matches the analytically optimal policy (its true
expected return differs from the heuristic's by less than the
measurement error, SEM ≈ 0.03). The retrain followed the hardening
fix that reports the 10-turn limit as a truncation rather than a
termination; the measured return is unchanged within noise, as
expected — the optimal policy for this game is myopic, so corrected
end-of-episode bootstrapping has nothing to change.

## Post-M1 hardening

A foundation pass after M1 (see git history), aimed at the seams the
next milestones will stress:

* **Mask/algorithm guard** — a probe refuses to pair a mask-blind
  algorithm (PPO, DQN) with a game that ever masks out an action, at
  both train and eval time, unless explicitly overridden. One
  algorithm registry now serves config validation, training, and
  checkpoint loading; masks are validated (`validate_mask`) at every
  boundary.
* **Config parameterization + fail-fast validation** — games and
  policies are parameterizable from YAML (`env_kwargs`, `policy`,
  `policy_kwargs`); unknown games, non-positive counts, and
  `hyperparams` keys shadowing top-level fields fail at load with
  named errors, as do checkpoint/game space mismatches at eval time.
* **Contract tests** — a dummy env with state-dependent masks and a
  truncating step limit exercises the mask contract end-to-end,
  alongside an end-to-end `train()` smoke test (25 → 56 tests).
* **Run provenance** — run directories carry a hash of the config
  (sweeps can't overwrite each other) and `summary.json` records the
  library versions that produced each checkpoint.

## Adding a game

Five steps, detailed in [DESIGN.md](DESIGN.md): implement a
`BaseGameEnv` subclass with honest rewards, register it, add a
heuristic baseline, add a YAML config, add rules tests. Nothing in the
trainer, evaluator, or viewer changes.

## Layout

```
src/boardgame_ml/   envs/ agents/ training/ eval/ play.py, web/ (cockpit backend)
web/                React + Vite + TypeScript frontend (builds to web/dist/)
configs/            experiment YAMLs
tests/              contract, rules, config-plumbing, baseline, web API tests
runs/               training output (git-ignored)
```
