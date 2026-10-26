# boardgame-ml

**Powerup Pong:** simultaneous two-player physics, bounded paddle movement,
collect-and-activate abilities, PPO self-play, tournaments, and live/replay
Watch support. See [rules, integration and experiments](docs/PONG.md), or
train with `configs/pong_ppo.yaml`.

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

## Milestone 6 — Connect Four (first multiplayer game)

The framework's second contract arrives, and with it a different
kind of learning problem: there is no environment to beat, only an
opponent who is also learning.

* **Two-player contract** (`envs/two_player.py`) — alternating,
  zero-sum, perfect information. The observation is **always from
  the perspective of the player to move** ("my discs / your
  discs"); rewards are the **mover's** (+1 for a winning move, 0
  for a draw, never −1 — the other seat's result is derived by
  negation). Rolled by hand instead of adopting PettingZoo AEC:
  the framework needs one adapter, not N-agent generality, and no
  new dependency. Rationale in
  [DESIGN.md](DESIGN.md#multiplayer-m6--a-second-contract-an-adapter-a-league).
* **Connect Four** — 6×7, gravity, `Discrete(7)`, mask = non-full
  columns, two binary observation planes with the stock
  `MlpPolicy`; incremental win detection cross-checked against a
  brute-force scan by fuzz tests. A full board is the termination
  (draw) — no truncation exists in this game.
* **Self-play league** (`training/self_play.py`) — a wrapper
  presents the game to MaskablePPO as single-agent RL: the learner
  takes a seat each episode (biasable toward the harder second
  seat), opponents move inside `step()`, rewards arrive
  sign-corrected. Opponents are a **league**, not the latest
  checkpoint: frozen snapshots of the learner (saved every 10k
  steps, weighted toward recent, uniform floor so old styles
  never vanish, first snapshot a permanent anchor), small shares
  of random and a noisy greedy heuristic, and a minimax **search
  anchor** — the anti-cycle doctrine, documented in DESIGN.md.
* **Yardsticks + tournament** — a tactical greedy (win → block →
  don't blunder → positional) and an alpha-beta **minimax
  (depth 4)**. Competitive games get no scalar score: evaluation
  is a tournament (`eval/tournament.py`), every pairing played as
  **both seats**, W/D/L per seat plus the first-player-advantage
  split. Results below.
* The cockpit seats two agents in Watch (a checkpoint vs a
  baseline, or a mirror match), renders the rack with red/yellow
  discs and a last-move highlight, and trains Connect Four from
  the Train view with the league config carried by PAIR_DEFAULTS
  parity.

## Milestone 7 — Racing generalization (did it learn to drive, or a track?)

M5's drivers were trained and measured on a single circuit.
M7 asks what they actually learned — by building circuits they
have never seen, and an exam to grade everyone fairly.

* **Generator v2** (`generate_track_v2`) — star-shaped control
  polygons (10–18 points, jittered angles, log-normal radii)
  through a closed Catmull-Rom spline: straights, hairpins,
  S-curves, sweepers; lengths 216–528 m over a 200-seed batch.
  Every candidate passes documented **validity gates** — no
  self-intersection, branch separation beyond two track widths,
  and a minimum turn radius of 13 m tied to the car's grip cap —
  enforced by bounded repair + rejection resampling,
  deterministic per seed (200/200 seeds accepted, ~3.1 attempts
  average, ~16 ms per track). Generator v1 is **frozen**: seed 7
  is pinned by a golden-hash test, so all M5 numbers still
  reproduce exactly.
* **Per-episode tracks** — `RacingEnv(randomize_track=True)`
  deals a fresh circuit from the episode seed on every reset
  (same seed → same track and spawn). Physics, observation and
  reward untouched: the ray-fan obs was designed to be
  track-agnostic; this is the milestone that settles whether
  the policy is.
* **The transfer experiment** — the shipped M5 specialists,
  zero-shot on five unseen circuits: how much of M5 was
  memorization? (Table in the M7 results below.)
* **Generalists** — SAC and PPO retrained on the track
  distribution (fresh circuit per episode, seeds 10000–10999),
  same recipes as M5 otherwise, then graded on a **held-out
  exam set**: seeds **50001–50005**, 20 episodes per track, the
  heuristic riding along as the track-agnostic control.

## Milestone 8 — PPO collapse autopsy (the mean is not the policy)

M5's open weakness — racing PPO peaks at 75k steps, then rots
into the crash attractor while SAC holds its plateau — gets a
root cause, and it is not the one on file. The suspected
mechanism (return scale in the thousands swamping the value
loss) is **refuted**; the actual one is an exploration-noise
divergence between the policy PPO trains and the artifact
the framework ships. Full evidence in DESIGN.md's M8 section;
the M8 results below carry the tables.

* **Diagnostics first** — every run now writes
  `diagnostics.csv`: SB3's logger values (value loss,
  explained variance, approx KL, clip fraction, entropy)
  captured through a logger output format, plus policy std,
  per-rollout parameter-change norm and LR from a callback
  (`training/diagnostics.py`). M5's collapse was visible only
  as an eval curve because these values were being computed
  and discarded; they aren't, anymore.
* **Exact reproduction** — the shipped M5 recipe re-run under
  instrumentation lands on the original's best eval to the
  last digit (2971.3858907999997), peak, cliff and rot
  included.
* **The finding** — at the deterministic peak, the *sampled*
  policy (the one PPO optimizes) crashes in 20/20 episodes
  (return 131); by 600k the sampled policy has improved
  tenfold (1499) while the deterministic mean has rotted to
  148. SB3's Gaussian std starts at 1.0 — half the [−1, 1]
  action box, at 20 Hz — and receives almost no gradient
  (1.000 → 0.997 in 600k steps), so the shipped "driver" was
  the mean of a distribution whose samples couldn't drive.
* **The fix** — `log_std_init: -1.0` (std ≈ 0.37) in the
  racing PPO configs + cockpit defaults, parity-tested as
  usual. Value-side probes (`vf_coef` 0.05; separate pi/vf
  nets — the latter reproduces the collapse bit-for-bit)
  are recorded as refutations.

## Milestone 9 — Cross-family transfer (did it learn driving, or the family?)

M7's generalist aced an exam from its *own* generator family.
M9 asks the harder question with a second family:

* **Generator v3** (`generate_track_v3`) — segment-built
  circuits in the style of real track design: straights joined
  by constant-radius arcs, heading closed by construction
  (turn angles sum to +2π) and position closed by an exact
  linear solve on two straight lengths. Same validity gates
  as v2 (min radius ≥ 13 m, no self-intersection, branch
  separation) — different character, not impossible: over 200
  seeds, lengths 259–902 m (median 493, vs v2's 358) and a
  median **52% of the centerline exactly straight**, where
  v2's splines are curved everywhere.
* **The zero-shot transfer exam** — all five M7 agents, cold,
  on five held-out v3 circuits, seeds **60001–60005**,
  20 episodes each. (Tables in the M9 results below.)
* **The 2×2** — the F2 generalist's 62% triggered the full
  matrix: a generalist SAC trained on the v3 distribution
  (same recipe; training range 70000–70999), then both
  generalists sat both families' exams.



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

# Racing generalization exam (M7): agents vs held-out circuits 50001-50005
.venv/bin/python -m boardgame_ml.eval.racing_exam \
    --agents runs/racing_sac_s0_77baf5ad/best_model.zip,racing_heuristic

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

## M6 results — Connect Four (tournament, 200 games per seat)

A competitive game gets no scalar score: the learner plays every
opponent **as both first and second player**, 200 games per seat,
deterministic policies throughout. The yardsticks calibrate the
ladder first — minimax (depth 4) beats the greedy heuristic
400-0 and random 400-0; greedy beats random 400-0. Zero draws
occurred in any pairing below (Connect Four between win-seeking
players is decisive; a full-board draw is a test fixture, not an
outcome these policies produce).

**The shipped run (run 3 recipe) — best_model, 2M steps:**

| Opponent | As first (W-D-L) | As second (W-D-L) | Combined | Win rate |
|---|---|---|---|---|
| Random | 183-0-17 | 182-0-18 | 365-0-35 | 91.2% |
| Greedy heuristic | **200-0-0** | **0-0-200** | 200-0-200 | 50.0% |
| Minimax (depth 4) | **200-0-0** | **0-0-200** | 200-0-200 | 50.0% |
| Its own first snapshot | 200-0-0 | 200-0-0 | 400-0-0 | 100% |

Read the seat columns, not just the totals: against both real
yardsticks the **opening seat won literally every game** — the
policy is a perfect opener and a winless defender at this level
of opposition (its final_model twin measured the same shape:
91.7% / 50% / 50% / 100%). First-player advantage across all
1,600 games: 75.1% of decisive games went to the seat that moved
first.

**Acceptance, honestly:** beats its first snapshot 100% (bar
70%) ✓; not swept by minimax — 50% wins+draws (bar 25%) ✓;
beats random 91.2% (bar 95%) ✗; beats greedy 50% combined (bar
65%) ✗. Two of four bars cleared. The two misses are the same
miss — second-seat defense — seen from two rungs of the ladder.

**How it got there — four runs, four instructive failures:**

| Run | League recipe | vs Random | vs Greedy | vs Minimax d4 | vs first snapshot |
|---|---|---|---|---|---|
| 1 (1.5M steps) | snapshots 0.8, clean greedy 0.1, random 0.1; default net, ent 0.01 | 89.7% W | **0% W** | **0% W+D** | 100% W |
| 2 (2M steps) | clean greedy 0.3, 10k snapshots, [256,256], ent 0.02 | 92.0% W | **100% W** | **0% W+D** | 100% W |
| 3 (2M steps) | noisy greedy 0.2, minimax anchor 0.1, snapshots 0.55 | 91.2% W | 50% W (200-0 / 0-200 by seat) | 50% W+D (200-0 / 0-200) | 100% W |
| 4 (killed at 731k) | run 3 + second-seat bias 0.65, 2.5M budget | 94.0% W (@730k snapshot) | 100% W | 0% W+D | — |

* **Run 1 — tactics never arrived.** A league of near-copies of
  itself (80% snapshots) taught it to beat its own style; the 10%
  greedy slice was too thin to teach blocking. It lost all 400
  games to a heuristic that simply takes wins and blocks them.
* **Run 2 — memorization, not learning.** Tripling the *clean*
  greedy share produced a perfect 400-0 sweep of that greedy —
  by memorizing fork lines against its deterministic replies —
  while minimax, which sees the forks coming, swept it 0-400 in
  return. A deterministic yardstick at a large share is a
  textbook to memorize, not an opponent. (The run's process was
  killed by a VM restart at 1.85M/2M steps; its checkpoints were
  tournamented from disk regardless — snapshots make a run
  resumable-by-measurement even when it isn't resumable.)
* **Run 3 — a perfect seat split.** Replacing the clean greedy
  with its ε=0.2 noisy twin and adding minimax itself as a 10%
  league anchor (the *refuter*) produced a generalist: exactly
  50% against both yardsticks, split 200-0 as first player and
  0-200 as second. Connect Four is a first-player win under
  perfect play; the policy had become attack-complete and
  defense-deficient — and the ~9% it still dropped to *random*
  is the same hole (second-seat games let slip).
* **Run 4 — the obvious fix, refuted.** Aiming the budget at
  the weak seat (`second_seat_share` 0.65: the learner defends
  in ~2 of 3 episodes) brought the specialist back: its last
  snapshot (the run was killed by a VM restart at 731k of 2.5M
  steps) swept greedy 400-0 and was swept by minimax 0-400.
  Oversampling the reply seat oversamples exactly the context
  where 1-ply opponents are most exploitable by forks, so the
  specialist basin deepened faster than the minimax anchor
  could refute it. The shipped recipe is therefore run 3's; the
  seat-bias field remains in the schema (default an unbiased
  0.5), recorded as an unproven knob with a negative first
  result.

Training throughput is the self-play tax made concrete: ~815
steps/s for run 1's recipe, ~570 steps/s for run 3's (bigger
net, minimax anchor moves priced in Python). The bare env steps
orders of magnitude faster; the league *is* the compute.

## M7 results — Racing generalization (exam: 5 unseen circuits × 20 episodes)

**Generator v2, measured over 200 fresh seeds:** 200/200
accepted by the validity gates (no self-intersection, branch
separation > 2 track widths + 3 m, min turn radius ≥ 13 m),
mean 3.1 generation attempts per seed, ~16 ms per track.
Lengths 216–528 m (median 358), tightest-corner radius
13.6–29.0 m (median 18.6), 1–12 distinct corners (median 6).
The v1 seed-7 oval (301 m, min radius ~28 m) is frozen and
golden-hash tested — every M5 number still reproduces.

**The transfer experiment** — the shipped M5 drivers, trained
only on seed 7, zero-shot on the exam circuits:

| agent | progress | lap rate | crash rate | laps/ep | mean lap |
|---|---|---|---|---|---|
| Specialist SAC (M5b best) | 91.0% | 81% | 32% | 4.55 | 13.2 s |
| Specialist PPO (M5 best) | 81.9% | 45% | 88% | 1.24 | 14.7 s |
| Heuristic (control) | 100% | 100% | 0% | 3.00 | 26.1 s |

SAC had learned mostly driving; PPO's rescued peak was mostly
a memorized line. Both specialists die on exam track 50001 in
every episode (SAC still laps 25% of them first).

**The exam** — exam seeds **50001–50005** (published; disjoint
from the training range 10000–10999 and from every prior run):

| agent | progress | lap rate | crash rate | laps/ep | mean lap |
|---|---|---|---|---|---|
| **Generalist SAC** | **100%** | **100%** | **0%** | **6.20** | **13.4 s** |
| Specialist SAC (zero-shot) | 91.0% | 81% | 32% | 4.55 | 13.2 s |
| Specialist PPO (zero-shot) | 81.9% | 45% | 88% | 1.24 | 14.7 s |
| Generalist PPO | 70.5% | 26% | 89% | 0.88 | 13.2 s |
| Heuristic (control) | 100% | 100% | 0% | 3.00 | 26.1 s |

The generalist SAC — trained on a fresh circuit per episode,
eval-selected checkpoint (2,878 mean return over 20 random
circuits at 70k steps) — **laps every exam episode on every
track without a single crash**, at specialist pace. It matches
the heuristic's perfect robustness at half the lap time.
Per track it is uniformly 100/100; the specialists' weak spot
(50001) is its equal-best. The generalist PPO is the honest
failure in the table: worse than its own specialist despite
600k steps, crashing in ~9 of 10 episodes everywhere — the
M5 on-policy brittleness, expressed against a moving target.

**Training:** PPO generalist 600k steps in 353 s (~1,700
steps/s). SAC generalist: reached its plateau (~2,800 eval
return on fresh circuits) by ~30–70k steps and held it; the
run was killed by a VM restart at ≥170k of 200k steps (~85 min
wall, ~34 steps/s including 20-episode evals on random
tracks), and its best-by-eval checkpoint is the shipped
artifact, per the framework's checkpoint doctrine.

**Verdict.** M5's SAC driver had learned to drive — imperfectly
(a third of unseen-track episodes still ended in a crash).
M5's PPO driver had learned *a track*. And distribution
training closes the gap completely for SAC: within the
generator's family, the generalist is a driver, full stop —
the exam's hardest circuit for both specialists is routine
for it. Scope note: "driving" is proven within the generator
family (star-shaped circuits); cross-family transfer is
untested by design.

## M8 results — PPO collapse autopsy (racing, seed-7 track)

**Reproduction.** The shipped M5 recipe, re-run: best eval
**2971.4 @ 75k steps — identical to M5's original to full
float precision** — cliff to 531 @ 125k, then rot: 321, 209,
204, 185, 194, 172, 163, 168, 150, 155, 162, 163, 153, 158,
157, 150, 147, **135 @ 600k** (M5's final artifact: 146.1).
Meanwhile the run's *training* return climbs monotonically
the whole way (~29 → ~1200+): the two curves are grading
different policies.

**Mean vs sampled** (the reproduced run's own checkpoints,
20 episodes each, deterministic = the shipped artifact,
stochastic = the policy PPO actually optimized):

| checkpoint | deterministic | stochastic |
|---|---|---|
| best (75k) | 2972 · 140 laps · 0/20 crashes | 131 · 2 laps · **20/20 crashes** |
| final (600k) | 148 · 0 laps · 20/20 crashes | 1499 · 66 laps · 14/20 crashes |

**Diagnostics at the scene** (all 600k steps): policy std
1.000 → 0.997 (never anneals; entropy pinned at the std-1.0
value); approx-KL ≈ 0.0001 and clip fraction ≈ 0 per update;
explained variance ≈ 0.0–0.15 (the value function never fits
— in *any* run, healthy ones included); value loss grows
42 → ~1450 tracking the return scale.

**Probes** (one variable changed per run, 300k steps):

| probe | result |
|---|---|
| `vf_coef` 0.5 → 0.05 | deterministic eval flat ~140–210 for all 300k steps — no peak forms; value weight **refuted** |
| separate pi/vf nets | eval curve **bit-identical to baseline at all 12 points**, collapse included — shared-trunk interference **refuted** |
| `log_std_init` −1.0 (std 0.37) | monotonic climb 1000 → 2183; **peak == final**; no cliff — **causal** |

**Verification** (fixed recipe, full 600k steps): a
monotone-in-spirit climb — 1000, 1237, 1360, 1476, 1641,
1676, 1792, 1890, 2077, 2027, 2054, 2183, 2203, 2434, 2188,
2504, 2463, 2441 — peak **2858 @ 550k**, **final 2586 @
600k = 90% of peak**, every final-eval episode completing
multiple laps (min 2582). One disclosed blemish: a transient
visit to the crash attractor at 475–500k (311, 176) between
the 2441 plateau and the recovery (2663 @ 525k, new peak @
550k) — the baseline's cliff, deferred ~350k steps and, for
the first time, *recovered from*: at std 1.0 the attractor
is absorbing, at 0.37 the trained policy stays coupled
enough to climb out. PPO racing no longer needs the
best-checkpoint rescue; best and final are the same species
of driver.

**Generalist probe** (fixed recipe on the M7 track
distribution; probe run killed by a VM restart at ~174k of
300k steps, exam sat by its 150k best-by-eval checkpoint —
a quarter of M7's budget): random-circuit training eval
1659 @ 150k, vs the broken recipe's best of 645 across its
whole 600k run. On the M7 exam paper:

| agent (exam) | progress | lap rate | crash rate | laps/ep | mean lap |
|---|---|---|---|---|---|
| Generalist PPO, M8 fix (probe, 150k ckpt) | **100%** | **100%** | **0%** | 3.60 | 22.4 s |
| Generalist PPO, M7 (broken recipe, 600k) | 70.5% | 26% | 89% | 0.88 | 13.2 s |
| Generalist SAC, M7 (reference) | 100% | 100% | 0% | 6.20 | 13.4 s |

The fixed PPO generalist matches SAC's perfect robustness
line on every track — at a cautious, near-heuristic pace
rather than SAC's speed. M7's "PPO never became robust" was
substantially a verdict on the broken recipe; SAC remains
the better racing driver (robust *and* fast), and the
fixed PPO's full-budget pace is the open question.

## M9 results — Cross-family transfer

**Generator v3, measured over 200 fresh seeds:** 200/200
accepted by the same validity gates as v2, mean 9.1 attempts
per seed (~100 ms per track; the closure solve and structural
gates reject more candidates than v2's repair-first path).
Lengths 259–902 m (median 493), tightest-corner radius
14.5–38.2 m (median 16.8), 1–10 corners (median 5), and a
median 52% of centerline samples on exactly-straight
elements (circumradius > 200 m) — against v2's ~6%. The v1
seed-7 track remains frozen and golden-hash tested.

**The zero-shot exam** — exam seeds **60001–60005**
(published; disjoint from the v3 training range 70000–70999
and from every v2 range), 20 episodes per track. The same
agents' M7 (F2) exam scores alongside:

| agent | F2 progress | F2 lap rate | F3 progress | F3 lap rate | F3 crash rate | F3 laps/ep | F3 mean lap |
|---|---|---|---|---|---|---|---|
| Generalist SAC (F2-trained) | 100% | 100% | 81.9% | **62%** | 40% | 2.62 | 19.0 s |
| Specialist SAC (seed-7) | 91.0% | 81% | 22.5% | **0%** | 100% | 0.00 | — |
| Specialist PPO (seed-7) | 81.9% | 45% | 24.8% | **0%** | 100% | 0.00 | — |
| Generalist PPO (F2-trained) | 70.5% | 26% | 36.7% | **0%** | 100% | 0.00 | — |
| Heuristic (control) | 100% | 100% | 100% | **100%** | 0% | 1.60 | 43.3 s |

The F2 generalist's failures are concentrated, not diffuse:
100/100 on exam tracks 60003–60005, but 58% progress / 5%
laps on 60001 (774 m, min radius 14.6 m) and 52% / 5% on
60002 (680 m, 16.3 m) — the two longest, tightest circuits,
whose long straights feed hairpins at full speed.

**The 2×2** — a generalist SAC trained on the v3
distribution, both generalists on both exams (lap rate;
progress in parentheses):

| trained ↓ / exam → | F2 exam (50001–50005) | F3 exam (60001–60005) |
|---|---|---|
| F2 generalist SAC (M7) | 100% (100%) | 62% (81.9%) |
| F3 generalist SAC (M9) | **100% (100%)** | 63% (83.5%) |

The F3 generalist laps the *other* family's exam in every
episode without a crash (5.80 laps/ep, 14.8 s mean lap) —
and on its own family's exam it matches, not beats, the
F2 generalist (63% vs 62%), failing on the same two
circuits (59%/5% and 58%/10% per track). Home-family
training bought nothing measurable on the hard paper.

**Training:** F3 generalist SAC — attempt 1 (150k budget)
killed by a VM restart at ~16k steps; attempt 2 (100k
budget) killed by a second restart at 44,446 steps. The
shipped artifact is attempt 2's best-by-eval checkpoint
(40k steps; eval return 2,232.8 over 20 fresh v3 circuits;
curve 122.8 → 956.4 → 1,877.8 → 2,232.8, still rising
slowly), per the framework's checkpoint doctrine.

**Verdict.** Both, in layers. The specialists learned a
*track* (M7's verdict) and, it turns out, a *family's*
worth of habits at most — nothing they know survives a
change of construction recipe (0% laps). The generalists
learned **driving**: distribution training in either family
produces a driver who aces the other family's exam and
keeps ~62–63% on the harder F3 paper sight unseen. And the
F3 exam's two long, tight circuits are a wall for *every*
learned agent regardless of home family — while the
family-less heuristic laps them slowly — which says the
remaining gap is speed management at the
straight-into-hairpin combination, not family familiarity.
Scope note: "cross-family" here means two circuit
generators (splines vs segments) behind the same gates, car
and observation; open roads, width variation and traffic
remain untested by design.

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
