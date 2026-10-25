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

## Milestone 4 — Flappy Bird (first continuous game)

The framework's first game that isn't turn-based — the test of the
thesis that a Gymnasium step is a **time slice**, not a turn:

* **Flappy Bird** — side-view world (288×512), bird pinned at fixed x,
  pipes scrolling past; the agent decides 20 times per second whether
  to flap. Physics (gravity, flap velocity, pipe speed/gap/spacing,
  `dt`, integration substeps) is parameterized through `env_kwargs`.
  Nothing in the framework core changed to admit it.
* **Derivatives in the observation, not frame stacking** — the
  5-vector obs is bird height, **vertical velocity**, distance to the
  next pipe, and its gap edges: a Markov state for a dynamical system
  with no frame history.
* Rewards are the game's own score: +1 per pipe cleared, plus +0.01
  per step survived — a *distance* term (constant scroll speed makes
  steps proportional to distance), pinned by tests. Crashes
  terminate; the step cap truncates.
* Baselines: uniform-random (flaps ~50% of steps, scores zero pipes)
  and a hand-tuned **threshold heuristic** (flap when below the next
  gap's centre by a margin and not already rising fast).
* Trains with plain **PPO** — the mask is trivially all-True here, so
  the mask guard passes a mask-blind algorithm and the run exercises
  the framework's unmasked path end to end.
* The cockpit gained an SVG side-view renderer (bird, pipes, ground)
  and Flappy train presets. See
  [DESIGN.md](DESIGN.md#continuous-games-m4--a-step-is-a-time-slice).

## Milestone 5 — Racing (first continuous-action game)

The last discrete crutch removed: the agent doesn't pick from a
list of moves, it *drives* — a `Box(2)` of steering and throttle:

* **Racing** — an arcade-bicycle-model car on a fixed closed track
  (seeded Catmull-Rom loop, 301 m, canonical `track_seed` 7 in the
  YAML; episode seeds randomize only the spawn point). Physics runs
  in substeps at 20 decisions/s, with one piece of real racing
  physics kept: a **grip cap** on lateral acceleration, without
  which cornering is free and full throttle is trivially optimal
  (measured: the ungripped prototype lapped at the speed clamp).
  Corner speed ∝ √(grip · radius), so braking is a learnable skill.
* **Observation** — an 11-ray fan to the track edges (±90°) plus
  speed, heading vs the centerline tangent, lateral offset, and
  current steering: derivatives and lookahead, no pixels, no frame
  stacking.
* **Reward** — checkpoint progress in metres, paid once (ordered
  checkpoints + a high-water mark, so reversing and re-covered
  ground earn exactly 0 — pinned by tests), +100 per lap. Leaving
  the track terminates; the 90 s cap truncates.
* **The mask seam, generalized** — mask machinery is now explicitly
  Discrete-only: `Box` games are unmaskable, the guard passes any
  registered algorithm for them, and the evaluator returns full
  action vectors. **SAC** joined the algorithm registry
  (smoke-tested through the pipeline) as the sample-efficient
  alternative to PPO for later tinkering.
* Baselines: uniform-random over the box (barely leaves the spawn)
  and an obs-only **raycast wall-follower** heuristic (steers
  toward open space, laps steadily at ~23.5 s).
* The cockpit gained a top-down SVG renderer (track ribbon, ray
  fan, car with steering tick) and racing train presets. See
  [DESIGN.md](DESIGN.md#racing-m5--continuous-actions).

## Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # CPU PyTorch via PyPI

# Train (configs are in configs/)
.venv/bin/python -m boardgame_ml.training.train --config configs/blackjack_maskable_ppo.yaml
.venv/bin/python -m boardgame_ml.training.train --config configs/higher_or_lower_ppo.yaml
.venv/bin/python -m boardgame_ml.training.train --config configs/match3_maskable_ppo.yaml
.venv/bin/python -m boardgame_ml.training.train --config configs/racing_ppo.yaml

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
* **Train** — all five games, with the algorithms each game's
  shipped config supports; timesteps, learning-rate, and seed are
  editable on top of per-(game, algorithm) defaults that **mirror
  `configs/*.yaml`** (gamma, policy, env_kwargs, env counts,
  hyperparams), guarded by parity tests
  (`tests/test_web_parity.py`) that compare the cockpit's table
  against the YAML files themselves — a cockpit run is the shipped
  run, not a silent approximation of it. The form shows the
  effective config it will launch. For racing, **SAC is
  preselected**: it plateaus and stays there, while PPO's slightly
  faster peak survives only via the best-by-eval checkpoint. A
  live chart shows episode returns with deterministic eval means
  as they land (SSE stream from the training job). **Stop** ends
  training gracefully and keeps the best-by-eval checkpoint (a job
  stopped before its first eval keeps `final_model.zip` only).
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

## M4 results — Flappy Bird (mean per episode)

Independent evaluations with the same harness — 1,000 episodes,
per-episode seeds starting at 1000000, which training never saw.
Return decomposes exactly as the reward defines it:
pipes + 0.01 × steps survived.

| Agent | Pipes | Return | ± SEM | Episode length |
|---|---|---|---|---|
| Random | 0.00 | 0.26 | 0.00 | 26 steps |
| **Heuristic** (threshold controller) | 10.64 | 14.29 | 0.40 | 364 steps |
| **PPO**, 600k steps (best-by-eval) | **67.07** | **87.60** | 1.76 | 2,053 steps |

Honest observations:

* **The time-slice thesis holds.** The continuous game trained,
  evaluated, and checkpoint-selected through the exact machinery
  built for turn-based games — no core changes — and the agent
  learned decisively: 6.3× the pipes of a hand-tuned threshold
  controller that itself clears ~11 pipes an episode, and ~334×
  random's return (random scored **zero** pipes in 1,000 episodes).
* **The derivative did its job.** With velocity in the observation,
  a plain MLP on a 5-vector learned precise flap timing; no frame
  stacking was needed anywhere in the pipeline.
* **Some trained episodes hit the 3,600-step cap** (the max episode
  length above) — the policy can, on favourable gap sequences, fly
  to the time limit rather than to a crash.

Training: 600,000 steps in 1,219 s of wall clock on this VM. Raw
env stepping runs at several thousand steps/s; most of the wall time
late in training is the periodic 200-episode deterministic
evaluation, whose episodes stretch past 2,000 steps as the agent
improves. Best-by-eval checkpoint selected at an in-training eval
mean of 91.9; the 1,000-episode independent figure above is the
headline.

## M5 results — Racing (mean per episode, 90-second cap)

Independent evaluations with the same harness — 100 episodes,
per-episode seeds starting at 2000000, which training never saw.
The track is the canonical seed-7 loop (301 m); spawns are seeded
per episode, as in training.

| Agent | Return | ± SEM | Progress | Laps / ep | Lap time (mean / best) |
|---|---|---|---|---|---|
| Random (uniform box) | 9.5 | 0.6 | 3.2% of a lap | 0 | — |
| **Heuristic** (raycast wall-follower) | 1447.4 | 0.2 | 380.7% | 3.0 | 23.54 s / 23.1 s |
| **PPO**, 600k steps (best-by-eval) | **2971.6** | 0.3 | **753.7%** | **7.0** | **11.95 s / 11.35 s** |
| PPO, 600k steps (final model) | 146.1 | 1.2 | 48.5% | 0 | — |
| **SAC**, 150k steps (best-by-eval) | 2895.6 | 0.4 | 728.5% | 7.0 | 12.33 s / 12.05 s |
| **SAC**, 150k steps (final model) | 2891.3 | 0.3 | 727.1% | 7.0 | 12.36 s / 11.90 s |

Honest observations:

* **The trained agent learned to drive — decisively.** It
  completes at least one lap in **100%** of episodes (the success
  bar was 80%), never crashes in 700 evaluated laps, and laps at
  essentially **twice the heuristic's speed** (11.95 s vs 23.54 s
  mean lap; average speed ~25 m/s against the wall-follower's ~13).
  Random, by contrast, never completes a lap: with mean-zero
  throttle and a bicycle model (no speed, no turn), it barely
  leaves the spawn.
* **Gamma was the braking horizon.** At γ = 0.99 the identical
  setup converged to full-throttle-and-crash and never braked
  (eval ~140 through 150k steps) — braking's payoff lands seconds
  after the decision, outside the effective horizon. γ = 0.995
  (with gae_lambda 0.98) learned to lap within 50–100k steps.
* **And then PPO fell apart — the run's honest weakness.** Every
  high-γ run peaked (this one: eval 2971.4 at 75k steps) and then
  degraded back to the crash attractor (~150) for the remaining
  ~450k steps; a stabilizer variant (lr 1e-4, full batch,
  target-KL) collapsed earlier, not later. The shipped checkpoint
  is the **best-by-eval** one — the framework's checkpoint rule
  exists for exactly this — while `final_model.zip` is the
  collapsed policy and should not be used. The mechanism (return
  scale in the thousands is the leading suspect) is documented as
  an open problem in DESIGN.md; SAC, now registered and
  smoke-tested, is the documented next experiment.

Training: 600,000 steps in 475 s of wall clock on this VM
(~1,260 steps/s including the dense 25k-step evals; the bare env
steps at ~7,750/s). Best-by-eval checkpoint selected at an
in-training eval mean of 2971.4; the 100-episode independent
figure above (2971.6) reproduces it almost exactly.

**SAC follow-up (same day).** The documented next experiment —
SAC on the identical game, track, and reward, γ = 0.995 carried
over (`configs/racing_sac.yaml`; buffer 300k, batch 256,
τ = 0.005, entropy auto-tuned, 150k steps — the budget set by
SAC's measured ~37 steps/s on this VM vs PPO's ~1,260, for
4,068 s of wall clock). The verdict is in the eval curve: SAC
reached eval 2848 by 40k steps and then **stayed there** — twelve
plateau evals from 40k to 150k steps, mean 2868 ± 18, peak 2893
at 70k, final eval 2892.8. No collapse. The independent eval
(table rows above) makes the contrast concrete: SAC's *final*
model scores 2891.3 — statistically its best model — with the
same 100% lap completion and zero crashes as PPO's rescued peak,
while PPO's final model scores 146.1 and crashes in every
episode. PPO's peak remains ~2.6% ahead on return and ~3% a lap
(11.95 s vs 12.33 s), but that peak is a snapshot the framework's
checkpoint rule had to catch; SAC's number is a steady state the
run simply ends in. Sample-for-sample SAC also got there in
about half the environment steps (plateau at 40k vs PPO's peak
at 75k) — the cost is wall clock, not experience. See DESIGN.md.

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
