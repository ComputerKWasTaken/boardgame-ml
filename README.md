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

## Milestone 3 — Match 3

The third game is the one that was unlearnable in the legacy project,
added through the framework's public seams only (env module, registry
line, baselines, YAML, render builder, frontend board):

* **Match 3** — 8×8 board, 6 gem types, swap-to-match with gravity,
  refills, and cascade chains. Observation is a Dict of one-hot gem
  planes `(6, 8, 8)` + moves remaining (no scalar gem IDs); the
  256-action space carries a true legal-move mask, computed once per
  state and cached; the reward is the game's own score (component
  3/4/5/6+ → 10/25/50/100, × cascade level) with no shaping. Board
  size, gem count, and move budget are YAML `env_kwargs`; the policy
  is `MultiInputPolicy`, also from YAML.
* Baselines: uniform-random over legal moves, and a 1-ply **greedy**
  agent (maximizes the points the swap itself scores).
* The web cockpit gained a Match 3 board (colored tile grid, legend,
  cascade badge) and its Games & Runs card.
* One framework finding, fixed at the seam: SB3's `EvalCallback`
  predicts unmasked, which hangs forever on a game with real illegal
  actions — the trainer now uses sb3-contrib's `MaskableEvalCallback`
  for MaskablePPO. See [DESIGN.md](DESIGN.md#match-3-m3--the-frameworks-final-exam).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # CPU PyTorch via PyPI

# Train (configs are in configs/)
.venv/bin/python -m boardgame_ml.training.train --config configs/blackjack_maskable_ppo.yaml
.venv/bin/python -m boardgame_ml.training.train --config configs/higher_or_lower_ppo.yaml
.venv/bin/python -m boardgame_ml.training.train --config configs/match3_maskable_ppo.yaml

# Evaluate: baselines…
.venv/bin/python -m boardgame_ml.eval.evaluate --game blackjack --agent random
.venv/bin/python -m boardgame_ml.eval.evaluate --game blackjack --agent basic_strategy
.venv/bin/python -m boardgame_ml.eval.evaluate --game match3 --agent greedy
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

## M3 results — Match 3 (mean score per 20-move episode)

Independent evaluations with the same harness — 1,000 episodes,
per-episode seeds starting at 777000, which training never saw:

| Agent | Mean score | ± SEM |
|---|---|---|
| Random (legal moves only) | 568.3 | 6.7 |
| **Greedy** (1-ply, maximizes the swap's own points) | 571.6 | 6.2 |
| **MaskablePPO**, 700k steps (best-by-eval) | **701.8** | 8.2 |

Three honest observations:

* **The trained agent learned — decisively.** +23.5% over random is
  ~12 standard errors of the difference; there is no plausible world
  where this checkpoint is not a genuinely better player. It also
  beats the greedy yardstick by ~23%.
* **"Multiple-fold over random" was never on the table.** In this
  game a *legal* move scores by definition, so uniform-random legal
  play already banks ~570 points an episode, most of it cascade
  luck. The headroom above that floor is what learning can claim —
  and +23% of the total is a large share of it.
* **Greedy is a weak yardstick here, and the trained agent proves
  why.** Greedy's chosen moves score more *immediately* than
  anyone's (15.4 first-cascade points per move vs 11.5 for both
  random and the trained agent), yet greedy ties random over full
  episodes: its big clears break up the near-match structures that
  produce level-2+ cascades (random earns ~17.5 cascade points per
  move to greedy's ~12.8). The trained policy's immediate-match
  quality is exactly random's — its entire edge comes from the
  board it leaves behind. PPO found the positional structure that
  1-ply greed cannot see.

Training: 700,000 steps in 954 s (~734 steps/s, 2-core VM),
best-by-eval checkpoint selected at a deterministic-eval mean of
745.4 (a 200-episode window during training; the 1,000-episode
independent figure above is the headline).

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
