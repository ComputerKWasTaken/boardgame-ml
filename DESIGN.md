# Design — boardgame-ml

A modular reinforcement-learning framework for teaching agents to play
board and card games: implement a game, train an agent on it, watch it
play, and eventually serve the trained agent as a bot other programs
can play against.

This is a ground-up rebuild. No code was carried over from the legacy
project (`Machine-Learning-with-Board-Games`); several of its *ideas*
were — see "Lessons from the legacy project" below, which maps each
legacy failure to the structural decision here that prevents it.

## Architecture

```
src/boardgame_ml/
├── envs/
│   ├── base.py            BaseGameEnv: the game contract
│   ├── blackjack.py       BlackjackEnv
│   ├── higher_or_lower.py HigherOrLowerEnv
│   └── match3.py          Match3Env (one-hot planes, true action masks)
├── agents/
│   └── baselines.py       Random / basic-strategy / probability / greedy agents
├── training/
│   ├── config.py          ExperimentConfig dataclass tree (YAML-loaded)
│   └── train.py           SB3 trainer CLI
├── eval/
│   └── evaluate.py        Greedy evaluation + strategy-agreement metrics
├── play.py                Text "watch it play" viewer
└── web/                   Local web cockpit (FastAPI backend)
    ├── app.py             API: games / runs / train jobs / watch sessions
    ├── jobs.py            Background training jobs (the trainer + callbacks)
    ├── sessions.py        Watch sessions (env stepped by predict_action)
    └── render_state.py    Per-game JSON render payloads for the frontend
web/                       React + Vite + TypeScript frontend (repo root)
configs/                   One YAML per experiment
tests/                     Contract, rules, config, and baseline tests
runs/                      Training output (checkpoints, logs, summaries)
```

### The game contract

A game is a `BaseGameEnv` (a `gymnasium.Env`):

* `observation_space` — any Gymnasium space; M1 games use small
  normalized `Box` vectors.
* `action_space` — `Discrete(n)` for turn-based games (larger action
  spaces are fine as long as legality is expressible as a mask), or
  `Box` for continuous-control games (M5's racing), where legality
  is the box itself and the mask machinery does not apply.
* `action_masks() -> np.ndarray[bool]` of shape `(n,)` — which actions
  are legal **now**, following the sb3-contrib convention. The trainer
  wraps envs in `ActionMasker`, so MaskablePPO's policy is masked at
  action-selection time *and* the same mask is present when the
  learning update recomputes action probabilities.
* `reset(seed=...)` — fully seedable; identical seeds must reproduce
  identical episodes (tested).
* Step, move, and time limits are **truncations**, never terminations:
  when an episode ends only because a limit was reached, `step` returns
  `truncated=True` with `terminated=False`, so value learning
  bootstraps across the boundary instead of mistaking the state for a
  terminal one.
* `render()` with `render_mode="human"` — returns a text snapshot.

**Scope: `BaseGameEnv` is deliberately single-agent** — one
decision-maker acting against the environment. Multiplayer games
live behind a separate contract, `TwoPlayerEnv`
(`envs/two_player.py`, shipped in M6) — see "Multiplayer" below —
with an adapter (the self-play wrapper) presenting them to this
framework where practical.

### Agents

* **Learning agents** are Stable-Baselines3 algorithms (MaskablePPO by
  default; PPO and DQN available) constructed in one place
  (`training/train.py::build_model`) from the experiment config. The
  framework does not hand-roll RL algorithms.
* **The mask guard:** only MaskablePPO consumes action masks. At both
  train and eval time, a probe of the game refuses to pair a
  mask-blind algorithm with a game whose masks ever exclude an action
  — silently learning an unmasked policy was legacy failure #1 —
  unless the config explicitly sets `allow_unmasked: true`.
* **Baselines** (`agents/baselines.py`) are non-learning reference
  policies: uniform-random over legal actions, and per-game heuristics
  (blackjack basic strategy; higher-or-lower probability-greedy).
  They define what "learned" means: beat random, approach the heuristic.

### Training runs

`python -m boardgame_ml.training.train --config <yaml>`:

1. Builds `n_envs` vectorized training envs and a **separate** eval env.
2. Trains, evaluating the *deterministic* policy every `eval_freq`
   timesteps on the eval env.
3. Saves `best_model.zip` = the checkpoint with the best **eval** mean
   return, plus `final_model.zip`, `train_log.csv` (every training
   episode return), eval logs, and `summary.json` under
   `runs/<game>_<algorithm>_s<seed>_<config-hash>/`. The 8-character
   hash of the canonicalized config keeps runs with different configs
   in different directories (sweeps never overwrite each other), and
   `summary.json` records the full config plus the library versions
   (SB3 / sb3-contrib / gymnasium / torch / numpy) that produced it.

### Evaluation

`eval/evaluate.py` runs greedy episodes of any baseline or checkpoint
and reports mean return, win/loss/push rates, and — for blackjack —
agreement with basic strategy over genuine decision states. This is
deliberately separate from training: training curves are evidence of
activity, evaluation is evidence of learning.

### Cockpit/CLI config parity

There are two ways to build a training config — the CLI from
`configs/*.yaml`, and the web cockpit from `web/jobs.py`'s
`PAIR_DEFAULTS` table — and they must never quietly disagree (that
disagreement, config read from a place that doesn't say what you
think it says, is legacy failure #2 in another costume, and it
nearly shipped: the cockpit's original builder never set `gamma`,
so a cockpit racing run would have used 0.99 — the value under
which the car provably never learns to brake — and Match 3 wasn't
offered at all). The mechanism keeping them in agreement:

* `PAIR_DEFAULTS` mirrors every shipped YAML on the load-bearing
  fields (gamma, policy, policy_kwargs, env_kwargs, n_envs,
  learning rate, hyperparams); presets supply only budget and eval
  cadence, and explicit UI fields override the table field-by-field.
* `tests/test_web_parity.py` compares `build_config(game, algorithm)`
  against the YAML files themselves for every shipped pair, so
  editing either side without the other fails the suite.
* The Train view displays the effective config (algorithm, gamma,
  policy, env count, learning rate) before a job starts, so any
  residual divergence is visible rather than silent.

## Adding a new game (5 steps)

1. **Implement** `envs/<game>.py`: subclass `BaseGameEnv`, set the
   spaces, implement `reset`/`step`/`_render_text`, and override
   `action_masks()` if legality is state-dependent. Rewards must be
   honest: terminal win/loss (plus scoring intrinsic to the game);
   shaping only if it is potential-based and documented.
2. **Register** the class in `envs/__init__.py::GAMES`.
3. **Add a heuristic baseline** for it in `agents/baselines.py` if any
   reasonable hand-written policy exists — it becomes the proof-of-
   learning yardstick.
4. **Add a config** `configs/<game>_<algo>.yaml`.
5. **Add tests**: `check_env` compliance (parameterize in
   `tests/test_envs.py`) plus unit tests for the game's core rules,
   with scripted randomness where outcomes must be deterministic.

## Lessons from the legacy project

The legacy framework had a sound core idea (a small abstract game
interface) but its learning never worked, for reasons found in a
code-level autopsy. Each maps to a structural fix here:

| # | Legacy failure | Structural fix here |
|---|---|---|
| 1 | **Masks dropped between acting and learning.** The PPO agent sampled actions from a valid-action-masked distribution, then computed its update with unmasked log-probabilities. The importance ratio was wrong before training even started, so the objective being optimized wasn't PPO's. Worst exactly where masking mattered most (a 256-action game with ~90% illegal moves). | Masks are part of the environment contract (`action_masks()`), delivered by `ActionMasker` to a maintained implementation (sb3-contrib MaskablePPO) that uses the *same* mask for sampling and for the update. There is no code path in this repo that recomputes action probabilities without the mask. |
| 2 | **Config that reached nothing.** The PPO agent read hyperparameters from YAML keys that didn't exist and silently fell back to hardcoded defaults; tuning had no effect. | Config is a typed dataclass, loaded in one place, passed explicitly into the trainer and model constructor — no singletons or globals to bypass. Unknown YAML keys are *rejected*. `tests/test_config.py::test_learning_rate_reaches_model` is a regression test: an absurd learning rate in YAML must appear in the model's optimizer, or CI goes red. |
| 3 | **Reward shaping changed the game being learned.** Blackjack shaping paid up to +1.4 for a single "good" hit — more than the +1.0 for winning the hand — and fined correct basic-strategy plays. The agent diligently optimized the surrogate instead of blackjack. | Rewards are terminal-only and equal to the game's actual outcome (+1/−1/0, +1.5 natural; ±1/0 per card guess). Any future shaping must be potential-based (provably policy-invariant) and documented in the env module. A test asserts non-terminal blackjack rewards are exactly 0. |
| 4 | **"Best" checkpoints picked from training noise.** The best checkpoint was the best trailing window of *training* returns — exploration-contaminated, high-variance, and often just a lucky streak, so the saved "best" model was frequently not the best policy. | Checkpoints are selected by periodic **deterministic evaluation on a separate env** (`EvalCallback`); training returns are logged only as diagnostics in `train_log.csv`. |
| 5 | **An unlearnable representation** (a board game encoded as scalar category IDs fed to an MLP, erasing spatial structure). | Discharged in M3 (see "Match 3" below): the contract never forced flat vectors — observation spaces are per-game, and Match 3 exposes multi-channel one-hot planes through a Dict space with `MultiInputPolicy` selected from YAML, no trainer changes. |

Two legacy ideas were kept, deliberately: the small abstract game
interface (here in Gymnasium shape, so the ecosystem's tooling applies),
and per-game checkpoint organization with metadata (here: run dirs
with `summary.json`).

## Match 3 (M3) — the framework's final exam

Match 3 was the legacy project's worst casualty — the game where all
three learning failures stacked (scalar-ID representation, dropped
masks in a 256-action space ~90% illegal, brute-force legality). It
was added here through the public seams only: one env module, one
registry line, baselines, one YAML, one render-state builder, one
frontend board. What it exercised, and what it found:

**Representation.** The observation is a Dict: `board` — one-hot
planes `(num_gems, rows, cols)` float32, one channel per gem type —
plus `moves_left` (normalized scalar). No scalar gem IDs anywhere:
gem types are categories, and the tensor says so. The policy is
SB3's `MultiInputPolicy`, named **in the YAML** (`policy:` +
`policy_kwargs.net_arch`), and the game parameters arrive via
`env_kwargs` the same way — the M2 parameterization carrying exactly
the load it was built for. SB3's `CombinedExtractor` flattens the
planes into the MLP; at a fixed 8×8 the learned per-position weights
can express the local patterns the game turns on. A CNN extractor
(weight sharing, translation invariance) is the natural upgrade if a
future game needs larger or variable boards; it would be another
YAML-only change.

**Scoring (the reward, in full).** A move's reward is the game's own
score for that move — nothing added, divided, or shaped. Each cascade
level groups its cleared gems into connected components
(same-type, orthogonally connected — runs that cross or touch merge);
a component of 3 / 4 / 5 / 6+ gems scores **10 / 25 / 50 / 100**. The
level's points are multiplied by the cascade level (first match ×1,
first cascade ×2, …). No move penalty, no terminal bonus. (The
legacy engine's component scoring table is retained because it is
the game's intrinsic scoring — its *reward* code, which divided by
10, fined every move, and invented terminal bonuses, is not.)

**Legality, computed once.** `action_masks()` returns the true legal
set: a swap is legal iff it creates a run of 3+. On a match-free
board any run a swap creates passes through one of the two swapped
cells, so legality is checked locally per adjacent pair (112 pairs)
and the result is cached until the state changes — the trainer's
`ActionMasker` calls hit the cache, not a re-simulation. Measured:
the bare env steps at ~2,300 steps/s single-threaded, and the full
MaskablePPO run (8 vectorized envs, updates, periodic masked evals)
sustained 734 steps/s — 700k steps in 954 s on a 2-core VM.

**The dead-board rule.** Board generation fills cells without ever
completing a run of 3 and re-rolls until the board has at least one
legal move, so every reset is playable. If a post-resolution board
has no legal move, it is **reshuffled** — the board's gem multiset
is permuted until it is match-free with a legal move (fallback: a
fresh generated board). The check runs even on the episode-ending
move: the framework invariant is that *every observable state* has
at least one legal action, because the returned `info` mask must be
valid. Measured frequency under random play: ~1 reshuffle per 1,200
moves.

**Illegal actions are rejected, not punished.** Stepping an action
outside the mask is a no-op: no swap, no points, the move is not
consumed, `info["illegal_action"] = True`. A masked agent never
attempts one; the branch defines behavior for unmasked callers
(`check_env` probes, human error) — and it made the finding below
*visible* instead of silently corrupting episodes.

**The one trainer change (a modularity finding, not a patch-around).**
M3's first training run hung forever at the first evaluation
boundary. Cause: SB3's `EvalCallback` predicts **unmasked** — fine
while every game's mask was trivially all-True, fatal here: an
unmasked deterministic prediction can be a rejected swap on the same
state forever, so the eval episode never ends. sb3-contrib ships
`MaskableEvalCallback` for exactly this; the trainer now selects it
when the configured algorithm is MaskablePPO (the standard callback
otherwise). One class selection at the existing eval seam,
regression-tested in `tests/test_train_smoke.py`. Everything else
about M3 — env, masks, policy, parameterization, eval harness,
watch sessions — needed no framework changes.

**An honest word about the yardstick.** The greedy baseline plays
1-ply: it maximizes the first cascade's points (measured: 15.4
points/move vs random's 11.5). Over full episodes it does **not**
beat random (see README's M3 table) — its bigger clears disturb more
columns and break up the near-match structures that feed level-2+
cascades (random earns ~17.5 cascade points per move to greedy's
~12.8). In this scoring system, immediate-score greed is a weak
objective. The baseline stays, documented as what it is: the claim
M3 tests is *learnability* — a trained agent decisively beating
random — not superhuman play.

## Continuous games (M4) — a step is a time slice

Flappy Bird (`envs/flappy.py`) is the framework's first game whose
world moves in continuous time. Nothing in the core changed to admit
it: the experiment was whether the existing contract already *was* a
continuous-time contract. It was. A Gymnasium `step` was never a
turn — it is a **time slice**, and the turn-based games are just the
degenerate case where the world does nothing between decisions.

**The time-slice model.** The agent decides once per step; one step
advances the world by `dt` seconds (Flappy: 0.05 s = 20 decisions/s).
Inside the step, physics integrates with semi-implicit Euler in
`physics_substeps` substeps of `dt / physics_substeps`. Substeps
refine the *integration*; they never change the decision rate, and
the env validates `dt > 0`, `physics_substeps >= 1` at construction.
`dt`, substeps, gravity, flap velocity, pipe speed/gap/spacing all
arrive through the existing `env_kwargs` seam. Randomness (gap
placement) still comes only from the env's seeded RNG, so an episode
remains a deterministic function of (seed, actions) — the contract
tests check this for Flappy exactly as for the turn-based games.

**Derivatives in the observation, not frame stacking.** A snapshot
of positions is not a Markov state for a dynamical system: it hides
velocity. The Atari-era fix — stack the last k frames and let the
policy infer motion — spends model capacity re-deriving a quantity
the simulator already knows. Instead the observation *contains* the
derivative: Flappy's 5-vector is bird height, **vertical velocity**,
horizontal distance to the next pipe, and the next gap's top and
bottom edges, all normalized. Position + velocity + target geometry
is the full state of the dynamics, so a plain MLP sees everything a
frame stack would have to infer. Rule of thumb for future games:
ship the derivatives (velocities, angular rates, accelerations if
the dynamics need them), not the history.

**Reward and ending mapping.** The same honesty rules apply,
translated: rewards stay the game's *intrinsic* score. Flappy pays
+1.0 per pipe cleared (its displayed score) plus +0.01 per step
survived — a distance term, legitimate here because the world
scrolls at constant speed, so steps survived are exactly
proportional to distance travelled, a score this genre keeps; it
cannot be farmed except by not crashing while pipes keep coming,
and at 0.01/step it stays subordinate to the pipe point (0.3 per
pipe interval). The decomposition is pinned by tests. Endings keep
the contract's semantics: a crash is a genuine **termination**; the
step cap is a time limit, hence a **truncation**.

**Masks go quiet.** Both actions are always legal, so the mask is
trivially all-True and the mask guard passes a mask-blind algorithm:
Flappy therefore trains with plain **PPO**, exercising the
framework's unmasked path (standard `EvalCallback`) end to end —
the same guard that refused unmasked algorithms for Match 3
approving one here is the guard working, not being bypassed. The
hand yardstick is a threshold controller (flap when below the next
gap's centre by a margin and not already rising fast), documented
with its constants in `agents/baselines.py`.

**Measured** (independent eval, 1,000 episodes on seeds training
never saw; README's M4 table): PPO scores 67.1 ± 1.4 pipes per
episode against the heuristic's 10.6 ± 0.3 and random's 0.0 —
random did not clear a single pipe in 1,000 episodes — with mean
episode length 2,053 steps against the heuristic's 364. Some
trained episodes run to the 3,600-step cap: on favourable gap
sequences the policy flies to the time limit, not to a crash.

**What this implies for a racing game.** The remaining step is
continuous *actions*, and the seams are already shaped for it: the
contract never assumed `Discrete` (only the baselines and the text
renderer do, per game), so a `Box` action space (steering, throttle)
slots into the same env shape. The algorithm consequence is real,
though: masking has no continuous analogue, so the mask guard's
all-True route becomes the only route, and the algorithm menu moves
to continuous-control methods — PPO handles `Box` natively; SAC/TD3
would join the registry in `training/algorithms.py`. The observation
lesson carries over with interest: a racer needs derivatives *and*
lookahead — position, velocity, heading rate, plus raycast-style
track geometry (distances to walls ahead), the spatial analogue of
Flappy's "distance to next pipe + gap edges". Reward stays intrinsic
(checkpoints/lap progress, the genre's own score), crashes
terminate, the lap timer truncates.

## Racing (M5) — continuous actions

Racing (`envs/racing.py`) is the framework's first game with a
`Box` action space — steering and throttle, both in [−1, 1] — and
it landed the way M4 predicted: through the public seams (env
module, registry line, baselines, YAML, render builder, frontend
board), plus **one seam fix the game genuinely required**: the mask
machinery assumed `Discrete` everywhere (`BaseGameEnv._info()`
asserted it, the trainer wrapped every env in `ActionMasker`, the
guard probe called `action_masks()`, the evaluator collapsed
actions to `int`). The fix states the rule the M4 section
anticipated — *mask machinery applies only to Discrete spaces; a
Box game is unmaskable and the guard passes any registered
algorithm for it* — implemented at those four boundaries and
regression-tested (a Box env must train a smoke run with a
mask-blind algorithm; `predict_action` must return the full action
vector). SAC joined the algorithm registry the same way, as the
sample-efficient continuous-control alternative; it is smoke-tested
through the full pipeline but PPO is the primary (below).

**The track.** A closed loop from a seeded generator: 12 control
points on a jittered circle, Laplacian-smoothed, closed Catmull-Rom
centerline at 192 points, constant 6 m half-width, 301 m around for
the canonical `track_seed` 7. The smoothing is a measured fix, not
decoration: raw jitter + spline overshoot produced a ~7 m minimum
turn radius — a hairpin narrower than the track is wide — while
three smoothing passes raise it to ~28 m and keep the loop visibly
organic. The track is *fixed* (one canonical seed in the YAML);
episode seeds randomize only the **spawn point** on it, which
diversifies episodes for deterministic policies (otherwise every
eval episode replays one identical trajectory), lets training
practice every corner from step 0, and makes "a lap" mean a full
circuit from wherever the car started.

**Physics: the one realism that matters.** Arcade bicycle model —
velocity aligned to heading, yaw `= speed / wheelbase · tan(δ)` —
with one piece of real physics kept: a **grip cap** on lateral
acceleration (`|speed · ω| ≤ 14 m/s²`). The cap is there because the
ungripped prototype was measured, and it was degenerate: with
velocity-aligned physics and no grip limit, cornering costs
nothing, full throttle is optimal everywhere, and a barely-trained
policy was lapping at the 32 m/s speed clamp within 30k steps
(best lap 9.4 s, the physical ceiling). With grip, corner speed is
bounded by `sqrt(grip · radius)` — ≈20 m/s on the tightest corners
vs 32 m/s on straights — and braking becomes a real skill for the
reward to teach.

**Observation and reward.** The obs is Flappy's doctrine extended:
an 11-ray fan over ±90° (distances to the track edges, the spatial
lookahead), speed, sin/cos of heading vs the centerline tangent,
lateral offset, current steering — derivatives and geometry, no
pixels, no frame stack. The reward is progress itself: arc length
along the centerline through an **ordered-checkpoint scheme** — 24
checkpoints, a monotonic counter, credited progress paid once as a
high-water mark, so reversing earns exactly 0 and re-covered ground
pays nothing (both pinned by tests, along with the windowed
nearest-point search that stops corner-cutting from teleporting
the car's track position). A lap pays +100; leaving the track
terminates; the 1,800-step (90 s) cap truncates.

**Baselines.** Random is now a uniform sample of the box — it
barely leaves the spawn (3% of a lap per episode; the bicycle model
means steering without speed does nothing, and mean-zero throttle
keeps speed near zero). The heuristic is an obs-only raycast
wall-follower (steer toward the distance-weighted centroid of the
fan, correct toward the centerline, speed set by forward-ray
openness): a metronome that laps in ~23.5 s, crash-free. An eager
first tuning lapped in ~16 s; it was deliberately detuned — a
yardstick should be a reference, not a rival.

**Training dynamics — the honest part.** Two findings, both
measured on eval curves, neither hidden:

* *Gamma is the braking horizon.* At γ = 0.99 / gae_lambda 0.95,
  PPO converged to full-throttle-and-crash and stayed there (eval
  ~140 through 150k steps; trace shows throttle ≈ +1.0 always,
  never braking, dying at ~30 m/s on corner entry): the payoff for
  braking lands ~1–4 s after the decision, outside the effective
  horizon the advantages can see. At γ = 0.995 / 0.98 the same
  setup learns to lap within 50–100k steps.
* *Then it falls apart.* In every high-γ run the eval curve peaks
  (2835–2971) and then degrades — gradually in one run (2835 →
  1890 → 1084 → 215), off a cliff within one 50k eval window in
  others — settling back into the ~150-return crash attractor for
  the remaining ~450k steps. A stabilizer variant (lr 1e-4,
  full-batch 2048, target_kl 0.04) collapsed *earlier*, not later;
  the mechanism is not pinned down (return-scale value chase is
  the leading suspect) and is recorded here as an open weakness,
  not explained away. What makes the run's product trustworthy
  anyway is the framework's oldest design rule: **checkpoints are
  selected by deterministic eval, never by training windows** —
  the shipped model is the 75k-step peak (eval 2971.4), caught
  finely because `eval_freq` is set dense (25k) for this config.
  The final-model artifact, by contrast, is the collapsed policy;
  anyone reproducing this run should read `best_model.zip`, and
  the gap between the two is the current price of PPO at this
  reward scale. SAC — off-policy, replay-buffered, without this
  failure mode in principle — is registered and smoke-tested for
  exactly this reason; it is the documented next experiment, not
  a footnote.

* *The SAC experiment (run 2026-10-25).* The follow-up was run as
  a controlled comparison: identical env, track, reward, and
  γ = 0.995; only the algorithm changed (`configs/racing_sac.yaml`
  — buffer 300k ≥ the run, batch 256, lr 3e-4, τ 0.005,
  `ent_coef: "auto"`, `learning_starts` 5k, one env; 150k steps,
  a budget set by SAC's measured throughput, ~37 steps/s with
  evals vs PPO's ~1,264 — 4,068 s of wall clock). The eval curve
  answers the question the PPO collapse posed. SAC climbed
  17 → 149 → 525 → **2,848 at 40k steps** (PPO needed 75k to
  peak) and then held a plateau for the remaining 110k steps:
  twelve evals, mean 2,868, std 18 (0.6%), peak 2,893 at 70k,
  **final eval 2,892.8 — the end of training is the peak, within
  noise**. The independent 100-episode eval confirms it in the
  artifacts themselves: SAC best-by-eval 2,895.6, SAC final
  2,891.3 — the same driver — against PPO best 2,971.6 and PPO
  final 146.1 (0 laps, 100% crashes). Both SAC checkpoints keep
  PPO-best's 100% lap completion and zero crashes; PPO's rescued
  peak keeps a small edge (~2.6% return, ~3% per lap: 11.95 s vs
  12.33 s mean). Two conclusions. First, the collapse is an
  on-policy failure, not a property of this env or reward: the
  same return scale that accompanied PPO's degradation does not
  destabilize an off-policy learner, which narrows (without
  closing) the open mechanism question above. Second, the
  trade is explicit and measured: SAC buys a trustworthy final
  artifact — best-by-eval becomes a formality — at ~34× PPO's
  per-step cost, while needing about half the environment steps
  to reach comparable driving. For this framework's purposes
  (the run's product is the artifact you can ship), SAC is the
  better racing driver; PPO's absolute peak remains the faster
  lap, held by a checkpoint that dense evals had to catch.

**Measured** (independent eval, 100 episodes, seeds 2,000,000+;
README's M5 table): the trained policy completes a lap in **100%**
of episodes (7.0 laps per episode, zero crashes in 700 laps),
lapping in **11.95 s** on average against the heuristic's 23.54 s
and random's no-lap-at-all. Training: 600k steps in 475 s of wall
clock (~1,260 steps/s including evals; the bare env steps at
~7,750/s).

**Cockpit.** Render state ships the track polylines, car pose,
and the ray fan; the frontend draws a top-down SVG scene (track
ribbon, dashed centerline, ray lines, a car with a steering tick)
plus lap/progress/speed header stats, and Train presets for racing
were added. One transparency note: the Games card's baseline
reference for racing is measured over 100 episodes instead of
1,000 — racing episodes run up to 1,800 steps of simulation
(vs ~10–400 elsewhere), and the count ships in the payload.

## Multiplayer (M6) — a second contract, an adapter, a league

**The contract.** `BaseGameEnv` stays single-agent, untouched.
Multiplayer lives behind `TwoPlayerEnv` (`envs/two_player.py`):
alternating two-player, perfect-information, **zero-sum** games,
where `step(action)` is always a move by the player to move.

* **Perspective observations.** The observation is *always* written
  from the seat of the player to move — "my pieces / your pieces",
  never "player 0's / player 1's". One policy (or one baseline brain)
  therefore plays either seat unchanged, which is what makes
  self-play, tournaments and mirror matches free.
* **Mover-perspective rewards.** `step()` returns the reward of the
  player who *just moved*: `+1` for a winning move, `0` for a draw
  or any non-terminal move. A `−1` can never be returned by a step —
  the loser learns of the loss on the *next* step from the other
  side, or from the terminal `info["winner"]`. Whoever aggregates
  results over an episode (wrapper, tournament, cockpit) derives
  the other seat by negation; the convention is stated once, here,
  and the wrapper's sign handling is pinned by scripted tests from
  both seats.
* **`info` carries the seats:** `current_player` (to move next;
  the mover at termination) and, at termination only, `winner`
  (`0`, `1`, or `None` for a draw).
* `action_masks()` is the mask of the player to move, as usual. In
  a terminal state it returns all-True (SB3 validates masks at the
  terminal transition; there is no next decision to mask).
* Truncation is a game property, not a contract one: Connect Four
  cannot truncate at all — a full board *is* the termination.

**Why not PettingZoo AEC.** PettingZoo's AEC API (`last()` /
`agent_iter()`, per-agent reward accumulation, terminations dicts)
is the standard for this shape, and rolling our own is normally
the legacy project's mistake. Three reasons it is right here: (1)
the framework needs exactly one thing from a multiplayer env — a
way to present it to SB3 as single-agent — and the adapter below
is ~150 lines against PettingZoo's much larger surface; (2) no new
dependency for a two-player alternating subset that AEC's
generality (N agents, simultaneous moves, partial observability)
overshoots; (3) the perspective-observation convention — the part
that actually makes self-play work — is ours to state either way.
If a future game needs >2 players or hidden information, that is
the point to adopt AEC properly.

**Connect Four** (`envs/connect_four.py`): 6×7, gravity,
`Discrete(7)`, mask = non-full columns, observation `Box(6, 7, 2)`
(two binary planes, mover-first) with the framework's default
`MlpPolicy` — no Dict/MultiInput machinery needed. Win detection is
incremental from the placed disc (only lines through it can be
new), cross-checked against a brute-force scan by a fuzz test. A
full board without a win terminates as a draw.

**The adapter: self-play as single-agent RL.** `SelfPlayWrapper`
(`training/self_play.py`) wraps a `TwoPlayerEnv` as a plain
Gymnasium env: each episode the learner takes a uniformly random
seat; when the turn passes to the opponent, the wrapper executes
the opponent's move *inside* `step()` and returns the learner's
own terminal result, sign-corrected (`+1/0/−1`). From SB3's side
it is an ordinary masked env — MaskablePPO trains unchanged, and
`check_env` passes on the wrapper. The same wrapper with a *fixed*
opponent is the evaluation env: periodic eval plays the
config's named heuristic anchor from both seats, so
`best_model` selection means "best against a fixed, honest
yardstick", never "best against yesterday's self". (One honest
caveat the runs exposed: a *deterministic* eval yardstick
saturates — run 2's eval pinned at exactly +1.0 once the policy
had memorized it, and run 1's pinned at −1.0 while it lost every
game, in both cases making `best_model` an arbitrary early
checkpoint. The noisy anchor of the later recipes evals without
saturating; the tournament, not the eval curve, is the verdict.)

**The league (anti-cycle doctrine).** Pure latest-vs-latest
self-play *strategy-cycles*: the learner overfits to its current
opponent's quirks, forgets how to beat last month's style, and
can orbit a rock-paper-scissors loop forever while its
self-play score looks flat. The opponent pool is therefore a
*mixture*: a bulk share drawn from **frozen snapshots of the
learner itself**, saved every 10k steps by a callback, weighted
toward recent snapshots but with a uniform floor so old styles
never fully leave the mix; plus small constant shares of the
random agent and a noisy greedy heuristic, which anchor the
bottom of the skill ladder (and keep early training, before the
first snapshot exists, playable); plus, optionally, a **search
anchor** — a minimax baseline holding a share as the league's
refuter. The pool is capped (24); eviction
drops the oldest snapshot except the very first, which is a
permanent anchor — and the acceptance tournament plays the learner
against that first snapshot to prove the league actually taught
it something. Snapshots predict through the same masked
`predict_action` path as evaluation.

**Tournament doctrine.** A competitive game has no scalar score:
a single-agent "mean return" against one fixed opponent conflates
seat advantage with strength and hides *who* the policy loses to.
Evaluation is therefore a tournament (`eval/tournament.py`): the
learner plays every yardstick **as both first and second player**
(≥200 games per seat), reported as W/D/L per seat plus overall
win/draw/loss rates and the first-player-advantage split across
all decisive games. The yardsticks calibrate each other the same
way (minimax depth-4 beats the greedy heuristic from both seats;
greedy beats random from both seats), so the learner's numbers sit
on a measured ladder, not vibes.

**What the tournaments taught (four runs; full tables in the
README).** The league doctrine above survived contact with the
enemy, but three refinements were earned the hard way:

1. **A deterministic yardstick at a large share is a textbook,
   not an opponent.** Run 2 fed the learner 30% clean greedy and
   got a 400-0 sweep of that greedy — by memorizing fork lines
   against its deterministic replies — and a 0-400 loss to
   minimax, which sees the forks coming. The fix was not less
   heuristic but a *noisy* one: the greedy's ε-random twin
   (`connect_four_greedy_noisy`) keeps its tactics and removes
   the memorizable lines.
2. **A league needs a refuter.** Run 3 added minimax itself as a
   10% "search anchor" — an opponent whose whole job is to punish
   the shallow tactic the rest of the pool cannot see through.
   Specialization collapsed immediately: the policy went from
   0% to an even 50% against depth-4 search.
3. **Seats are different skills; measure them separately.**
   Run 3's combined 50% hid a *perfect* split — 200-0 as first
   player, 0-200 as second, against both yardsticks. The
   combined rate alone would have called it "even"; the
   per-seat table showed an attack-complete, defense-deficient
   policy (and explained its stubborn ~9% loss rate even to
   random: second-seat games let slip). The obvious response —
   oversample the defensive seat in training
   (`second_seat_share`) — was run 4, and it *refuted itself*:
   killed by a VM restart at 731k steps, its last snapshot had
   already re-specialized (greedy 400-0, minimax 0-400), because
   the reply seat is also where 1-ply opponents are most
   fork-exploitable. The shipped recipe is run 3's; the seat
   bias remains in the schema, defaulting to an unbiased 0.5,
   as a recorded negative result. What would actually move the
   second seat — deeper search anchors at a larger share, a
   value-pretrained defense, or an order of magnitude more
   steps — is future work, stated as such.

## Racing generalization (M7) — did it learn to drive, or a track?

M5's drivers were trained *and measured* on one generated
circuit (seed 7: a 301 m rounded loop, tightest radius ~28 m).
Nothing in that setup distinguishes "learned to drive" from
"learned this track" — the same policy that laps it in 11.95 s
might be a racing driver or a very fast playback. M7 builds the
instrument that can tell the difference: a track *distribution*
to train on, and a held-out *exam set* to be graded on.

**Generator v2** (`envs/racing.py::generate_track_v2`, selected
per track via `build_track(seed, version=2)`; v1 is frozen —
seed 7 is pinned by a golden sha256 test, so every M5 number
stays reproducible). A candidate is a star-shaped control
polygon: 10–18 points, angular gaps jittered U(0.5, 1.6),
radii log-normal (σ 0.34, one light smoothing pass, clipped to
[−0.95, +0.65] log units) around a base radius U(42, 72) m,
sampled as a closed Catmull-Rom spline at 16 samples per
segment. Star-shaped-by-construction keeps candidates sane
cheaply; the gates below do the real work.

**Validity gates** (`track_violations`, enforced inside the
generator by rejection/repair — never trusted, always
re-checkable): (a) no degenerate segments and total length in
[170, 950] m; (b) **minimum turn radius ≥ 13 m** by circumradius
over consecutive centerline triples — the threshold is tied to
the physics: the grip cap (14 m/s²) bounds cornering speed at
√(14·r), so a 13 m corner is drivable at ~13.5 m/s, while a
tighter one would be a wall the car cannot legally take;
(c) no crossings between non-adjacent centerline segments;
(d) **branch separation**: points more than 1/6 of a lap apart
along the loop stay more than `2·half_width + 3 m` apart in the
plane, so two passes of the circuit are two ribbons of road,
never one merged smear the raycasts cannot parse. Radius
failures are *repaired*, not resampled: Laplacian passes on the
control polygon (≤7) pull kinks out, aimed at a per-attempt
target radius drawn U(13.5, 27) m — so accepted tracks spread
across the tight/flowing spectrum instead of piling onto the
gate floor, and their character survives (v1's lesson was a
smoother that ironed a 7 m hairpin into a 28 m bend; v2's
repair is bounded and aimed, never a resample). Structural
failures resample; the radius variance stages down after
40/90/150 attempts; 200 attempts without an accept raises.
Measured over 200 fresh seeds: **200/200 accepted, mean 3.1
attempts, ~16 ms per seed**; lengths 216–528 m, min radius
13.6–29.0 m (median 18.6), 1–12 distinct corners per circuit
(median 6) — straights, hairpins, S-curves and sweepers, not
perturbed circles.

**Per-episode tracks.** `RacingEnv(randomize_track=True,
track_seed_lo, track_seed_hi)` draws the track seed *first*
from the episode RNG on every `reset()`, so an episode seed
determines (track, spawn) jointly and reproducibly; the dealt
track is rebuilt only when it changes. Default fixed-track
mode is untouched, as are physics, observation and reward —
M5's bet was that a ray-fan observation is track-agnostic, and
this mode is where the bet is settled.

**Doctrine: distribution vs exam.** M6's lesson — for
competitive games the tournament, not the eval curve, is the
verdict — has a single-player analogue here. A generalist's
training evals run on *random* tracks, so the curve mixes
policy quality with the luck of which circuits were dealt;
the grade comes from a fixed exam: five v2 tracks, seeds
**50001–50005**, disjoint from the training range
[10000, 11000) and from every prior run, 20 episodes each with
episode seeds shared across agents. Three controls frame the
result. The **specialists zero-shot** (the shipped M5 SAC and
PPO bests, trained only on seed 7) measure how much of M5 was
memorization. The **heuristic** — which reads only local
geometry and has never seen any track — measures whether the
task is reactively solvable at all, and sets the robustness
bar (it laps everything, slowly). A generalist is credited
with *driving* only if it beats the specialist's transfer by a
wide margin and laps the majority of exam episodes.

**Findings** (exam: 5 tracks × 20 episodes, shared episode
seeds; README's M7 tables). The **specialist SAC transferred
substantially**: 91% mean progress and an 81% lap-completion
rate zero-shot — most of M5 was driving, not playback. But 32%
of its episodes still ended in a crash, and one exam circuit
(50001) killed it in all 20 episodes (25% managed a lap first).
The **specialist PPO was the memorizer**: 45% laps, 88%
crashes — its rescued peak was a fast line around one oval.
The **generalist SAC aced the exam outright**: 100% progress,
**100% lap completion on every track, zero crashes in 100
episodes**, 6.2 laps per episode at 13.4 s mean — the
heuristic's perfect robustness at twice its speed, and the
specialists' pace without their fragility. Its training evals
tell the same story in the other direction: by 30k steps it
already averaged ~2,800 return over 20 *fresh* circuits per
eval, and held a ±3% plateau for the remaining 140k+ (the run
was killed by a VM restart at ≥170k of 200k steps; the
best-by-eval checkpoint — 2,878 at 70k — is the shipped
artifact, per doctrine). The **generalist PPO failed**: 26%
laps, 89% crashes, *worse* than its own specialist's zero-shot
despite a healthy-looking training curve. Its failure is
uniform across tracks (0% laps on the 14 m hairpin circuit,
100% episode-crash rates on four of five) — not one killer
corner but a driver that never became robust: on-policy
training against a distribution that changes every episode
never gave it a fixed target long enough to stabilize, the
same brittleness M5 saw as peak-and-collapse, expressed as
never-converging instead. The M5b verdict survives contact
with a distribution: SAC is the racing algorithm; PPO's racing
results, specialist and generalist alike, are snapshots.

One honest scope note: the exam circuits come from the same
generator family as the training distribution (star-shaped
control polygons). "Learned to drive" here means *within that
family* — genuinely unseen layouts, lengths and corner
sequences, but not cross-family transfer (no figure-eights,
no open roads), which the gates deliberately exclude.

## PPO collapse autopsy (M8) — the mean is not the policy

M5 left an open weakness: racing PPO peaks at ~75k steps and
then degrades into the crash attractor, and the framework
shipped the rescued peak. M7 inherited the question (its
generalist PPO failed worse). The standing suspicion, recorded
in M5, was *return-scale value chase*: returns in the
thousands, a value loss to match, swamping the policy. M8
reproduced the collapse under instrumentation, tested the
suspects one variable at a time, and found the suspicion
**wrong**. The mechanism is stranger and more useful.

**Reproduction.** The shipped M5 recipe, re-run with the new
diagnostics capture, reproduced the original run *exactly*:
best eval 2971.3858907999997 in both (same seed, same
versions — the pipeline is deterministic), peak at 75k steps,
cliff to 531 at 125k, slow rot to 135 by 600k (M5's final:
146.1). Every run also now writes `diagnostics.csv` (SB3's
logger values via a KVWriter format, plus policy std /
update size / LR from a callback — `training/diagnostics.py`);
before M8 those values were dumped into the void, which is
why this section couldn't have been written from M5's
artifacts.

**The divergence.** The diagnostics show two policies living
in one checkpoint. The *training* return (stochastic policy,
`rollout/ep_rew_mean`) climbs monotonically for all 600k steps
(29 → ~1200+) — training never stops "working". The
*deterministic* policy — the mean action, the thing
`EvalCallback` grades and the framework ships — is what peaks
and rots. Measured directly on the reproduced run's own
checkpoints (20 episodes each, seed-7 track):

| checkpoint | deterministic (mean) | stochastic (sampled) |
|---|---|---|
| best (75k) | 2972, 140 laps, 0/20 crashes | 131, 2 laps, **20/20 crashes** |
| final (600k) | 148, 0 laps, 20/20 crashes | 1499, 66 laps, 14/20 crashes |

At the celebrated peak, the policy PPO was actually training
could not finish a single episode; the shipped artifact was
the *mean* of that distribution, a driver nobody optimized.
By 600k the two have traded places: training genuinely
improved the sampled policy tenfold (131 → 1499) while the
mean rotted to the crash attractor. The "collapse" is the
moment the mean — drifting under gradients computed for a
different, noise-dominated objective — walks off the precise
line it had transiently occupied.

**Why the mean is unmoored: the exploration std is a frozen
constant.** SB3's Gaussian policy starts at `log_std_init`
(default 0 → std 1.0 on a [−1, 1] action box, at 20 decisions
a second) — and in this regime log_std receives essentially
no gradient: across 600k steps it moved 1.000 → 0.997
(entropy loss pinned at the std-1.0 value, −2.837, the whole
run). The initial std is therefore not a starting point but
the run's *permanent* exploration noise, and at 1.0 it is
half the entire control authority injected as jitter every
50 ms. Consequences cascade: sampled episodes are bimodal
(lucky laps vs early crashes), returns from any state are
dominated by the noise realization, the value function
cannot fit them (**explained variance ≈ 0.0–0.15 for the
entire run**), advantages inherit the noise, and the policy
gradient that remains (per-update approx-KL ≈ 0.0001, clip
fraction ≈ 0) slowly walks the mean toward whatever line is
*robust to std-1.0 jitter* — a worse driver, deterministically.

**The probes (one variable each, 300k steps, seed 0).**

| probe | change | eval trajectory | verdict |
|---|---|---|---|
| P1 | `vf_coef` 0.5 → 0.05 | flat ~140–210 for all 300k steps; never peaks | value-loss *weight* refuted: cutting it 10× prevented the peak instead of preserving it |
| P3 | separate pi/vf nets (no shared trunk) | **bit-identical to baseline at all 12 evals** — 2971 @ 75k, 531 @ 125k, 157 @ 300k | trunk interference refuted: with no shared trunk there is nothing for value gradients to corrupt, and the collapse is the same event on the same step |
| P4 | `log_std_init` −1.0 (std 0.37) | monotonic climb 1000 → 2183; **peak == final**; no cliff through the baseline's kill window | exploration-noise divergence confirmed as the causal variable |

Two corroborations. First, the value-scale story fails in
both directions: P4's healthy run carries value losses of the
same ~1000 magnitude as the collapsed one (and explained
variance still ≈ 0 — value learning on this env is
chronically weak in *every* run, healthy or not, so it cannot
be the differentiator). Second, the M5 "stabilizer" variant
(lr 1e-4, full batch, target_kl) collapsing *earlier* is
consistent: none of its three knobs touches the noise, and a
different random walk off the same cliff edge lands sooner
or later by chance. (`clip_range_vf` was not probed: P1 and
P3 jointly close the value-side space — weight and pathway —
and P4's single-variable sufficiency made a fourth probe
evidence-free.)

**Why SAC and the discrete games were immune.** SAC's std is
state-dependent and entropy-auto-tuned — its exploration
noise adapts against the objective instead of being frozen
at init — and its critic actually fits (replay + targets),
so its deterministic actor is graded on signal, not noise.
The categorical games have no mean/sample divergence at
all: deterministic eval takes the argmax of the same
distribution the sampler draws from, and a sampled action
is a legal move played once, not a 20 Hz control signal
drowned in jitter. The pathology needs all three of:
continuous Gaussian actions, a frozen init-scale std, and
a deterministic artifact graded apart from the sampled
policy.

**The fix.** `log_std_init: -1.0` (std ≈ 0.37) in both
racing PPO configs and the cockpit's `PAIR_DEFAULTS`
(parity-tested, as always). It lives at the config layer —
the framework's recipe seam — because the lesson is per-game
physics, not a framework universal: the right exploration
scale is a property of the control problem (action range,
decision rate, grip), and `policy_kwargs` is exactly where
SB3 exposes it. **Verification** (600k steps, fixed recipe,
seed 0): the eval curve climbs the whole run — 1000 @ 25k
through plateaus at ~2050 (250k) and ~2450 (425k) to a peak
of **2858 @ 550k**, with the **final model at 2586 — 90% of
peak, every eval episode completing multiple laps** (min
episode 2582). The best-checkpoint rescue is no longer
load-bearing: best and final are the same species of driver.
One honest qualification: the run *does* visit the crash
attractor once — 2441 @ 450k → 311 @ 475k → 176 @ 500k —
the baseline's cliff, deferred ~350k steps, and then it
**recovers**: 2663 @ 525k, a new peak at 550k. At std 1.0
that visit is absorbing (the baseline sat at ~150 for the
remaining 475k steps); at std 0.37 the sampled policy is
good enough that its data still contains driving, the
advantages still point home, and the mean finds its way
back within ~50k steps. The fix therefore does not repeal
the knife-edge — a drifting mean can still step off the
line — it demotes the collapse from terminal to transient
by keeping the trained policy coupled to the shipped one.
Diagnostics corroborate the mechanics: the run's largest
parameter jump (param-delta 1.58 @ ~465k) is the knock-off
event itself, and its largest policy updates by far
(approx-KL 0.185 and clip fraction 0.48 @ ~498k — against
run-typical ≈0.0002 / ≈0) fire *inside* the attractor,
real gradient signal doing the recovery work; the sampled
return craters to ~280 during the visit and climbs back
with the mean. `ep_rew_mean` ends at ~1150 against the
deterministic 2586 — coupled, not inverted.
Doctrine, stated once for future games: **the artifact you
ship must be the object you optimize** — if a framework
grades the deterministic mean, the sampling distribution
must stay coupled to that mean (exploration noise small
against control authority), or the eval curve grades a
policy that never trained. And: for Box-action PPO in SB3,
treat `log_std_init` as load-bearing — measure the std's
trajectory before believing it anneals.

**Still open.** (a) The transient dip: whether a lower
init std (−2.0) removes it or merely defers it past the
budget is untested; −1.0 was kept because it is the value
the evidence covers, and its climb is already slower than
the old transient peak's. (b) Explained variance ≈ 0 in
every racing PPO run means GAE is running on a decorative
value function (advantages ≈ discounted Monte-Carlo noise);
why the value head can't fit this env's returns — spawn
randomization plus lap-count discreteness are suspects —
is a separate investigation. (c) The M7 generalist-PPO
verdict, re-probed under the fix (300k-step probe, killed
by a VM restart at ~174k; the exam was sat by its
best-by-eval checkpoint from 150k): the probe's
random-circuit eval reached 1659 @ 150k — past the broken
recipe's 600k best of 645 — and on the M7 exam paper the
checkpoint scored **100% progress, 100% lap completion,
0% crashes on all five tracks**, the generalist SAC's
perfect robustness line, at a cautious pace (3.6 laps/ep,
22.4 s mean lap vs SAC's 6.2 / 13.4 s). So M7's "PPO
generalist failed" was substantially a verdict on the
broken recipe: with the noise fixed, even a quarter-budget
PPO probe is a robust generalist that trades pace for
survival. Full-budget pace is the open question, and the
M7 ranking (SAC the better racing driver) stands — SAC
remains both robust *and* fast, with no dip on record.

## Cross-family transfer (M9) — driving, or the family?

M7's verdict carried a scope note: its exam circuits were held
out but not *foreign* — one generator family (star-shaped
control polygons through a Catmull-Rom spline) wrote both the
training distribution and the exam paper. Nothing in that setup
can separate "learned to drive" from "learned the spline
recipe's habits". M9 builds the foreign paper.

**Generator v3** (`generate_track_v3`, selected by
`track_version=3`) constructs circuits the way real tracks are
drawn: as **elements** — straights joined by constant-radius
arcs. A candidate draws 5–10 corners (signed turn angles, ~1 in
5 reversed for S-curves, uniformly shifted so they sum to
exactly +2π; radii from a hairpin/sweeper mixture at 14.5–58 m;
straight lengths 14–62 m). The heading therefore closes by
construction, and the *position* closes by an exact solve: two
straight lengths are left as unknowns in the 2×2 system
`Σ element displacements = 0`, tried best-conditioned first —
the sampled loop closes to ~1e-13 m. The legality contract is
v2's, unchanged — the same `track_violations` gates (min
circumradius ≥ 13 m against the grip cap, no self-intersection,
branch separation) — so the family is different in character,
not impossible. Measured over 200 fresh seeds each:

| batch stat (200 seeds) | v2 (splines) | v3 (segments) |
|---|---|---|
| accepted | 200/200, mean 3.1 attempts | 200/200, mean 9.1 attempts |
| length | 216–528 m, median 358 | 259–902 m, median 493 |
| min turn radius | 13.6–29.0 m, median 18.6 | 14.5–38.2 m, median 16.8 |
| corners (r < 40 m) | 1–12, median 6 | 1–10, median 5 |
| samples on straights (r > 200 m) | median ~6% | **median 52%** |

The last row is the family's signature: v3's curvature profile
is a step function (exactly zero on straights, exactly 1/R on
arcs); v2's varies continuously and is never zero. v3's
circuits also run ~40% longer — straights stretch them — and
they deal a combination v2 rarely does: a long full-speed
straight feeding a 15 m hairpin, with a distinct braking zone.

**The zero-shot exam.** M7's five agents, cold, on five
held-out v3 circuits (seeds **60001–60005**, in no training
range; 20 episodes each, the M7 harness's conventions). The
F2 generalist's lap rate fell from 100% to **62%**, its crashes
concentrated on the two longest, tightest exam circuits
(60001: 774 m, min radius 14.6 m — 5% laps; 60002: 680 m,
16.3 m — 5%), while lapping the other three in every episode.
Both specialists and the generalist PPO scored **0% laps with
100% crash rates**. (That PPO is M7's shipped checkpoint; the
M8 autopsy above later showed what its family's failure was
made of — the zero-shot table here grades the artifacts M7
actually shipped.) The heuristic — which has no family to
learn — stayed at 100%/0% on both papers. A failure census of
the F2 generalist on the two hard circuits shows crashes
spread across the whole lap (progress at death from 8% to
100%), not one killer corner: repeated exposure to the hard
combination, not a single trap.

**Completing the 2×2.** The gap being large, a generalist SAC
was trained on the v3 distribution (the M7 SAC recipe
unchanged; training range [70000, 71000)) and both generalists
sat both exams:

| SAC generalist | F2 exam (laps) | F3 exam (laps) |
|---|---|---|
| trained on F2 (M7) | **100%** | 62% |
| trained on F3 (M9) | **100%** | 63% |

Three readings, in increasing order of importance:

1. **The skill transfers.** An agent trained only on
   straights-and-arcs laps the spline family's exam in 100% of
   episodes without a crash (5.80 laps/ep at 14.8 s, against
   the home agent's 6.20 at 13.4 s — a touch more cautious,
   flawless), and the F2 generalist keeps 62% on the foreign
   paper. That is not what "learned the family" predicts; the
   specialists show what family-bound actually looks like
   (0% laps, progress down ~60–70 points from their F2 exam).
2. **The F3 exam is the harder paper — for everyone.** Both
   generalists stall at ~62–63% on it, failing on the same
   two circuits; home-family training bought the F3
   generalist *nothing measurable* on its own exam over the
   F2 generalist (63% vs 62% laps, 83.5% vs 81.9% progress).
   The residual failures live at the full-speed-straight →
   hairpin combination both families' graduates meet as
   strangers, and the heuristic laps those circuits at half
   the learned agents' speed — they are drivable slowly, so
   the learned failures are speed-management failures, a
   property of the observation/physics package more than of
   either family's familiarity.
3. **Specialization is the specialists' disease.** The
   seed-7 drivers' zero-shot progress fell from 91%/82%
   (F2 exam) to 23%/25% (F3 exam): what one track teaches
   does not survive a change of construction recipe at all.
   Distribution training — in *either* family — is what
   bought transferable skill.

**The F3 training run, honestly.** It took three attempts'
worth of patience: the first (150k budget) was killed by a VM
restart at ~16k steps; the second (100k budget) reached
44,446 steps before a second restart killed it. The shipped
checkpoint is the second attempt's best-by-eval — the 40k
checkpoint, eval return 2,232.8 over 20 fresh v3 circuits
(curve: 122.8 → 956.4 → 1,877.8 → 2,232.8 — still rising,
decelerating), per the framework's best-by-eval doctrine;
the first attempt's run directory is preserved with a
`_dead16k` suffix, in the M7 style. A fully-plateaued F3 agent might squeeze more
from the two hard circuits — but the fully-mature F2
generalist scored the same 62% there, so undertraining is
not the obvious explanation for the wall.

**Scope, stated plainly.** Two families are not all families:
both generators still produce closed, constant-width, flat
circuits behind the same gates, driven by the same car
through the same ray-fan observation. What M9 establishes is
narrower and stronger than a slogan — the learned skill is
not an artifact of the spline recipe; it survives a change of
construction method, curvature profile, and length scale.
What it does not test: open roads, varying width, elevation,
traffic — the dimensions along which "driving in general"
would next be falsified.


## Foundations (M10) — calibration and three-seed replication

M10 adds no game and no algorithm. It pays down the two debts the
roadmap named as standing disciplines, because every claim before it
rested on the same two weaknesses: each headline number was a single
training seed, and nothing anchored the framework's own training
pipeline to the outside world — M8's exploration-std bug survived
two milestones precisely because there was no external anchor to
disagree with it.

**Part 1 — benchmark calibration.** The framework's `train()` (not a
hand-rolled SB3 script: the plumbing is what is under test) trains
PPO on two Gymnasium classics through a thin shim
(`envs/gym_classic.py`), using RL Baselines3 Zoo canonical
hyperparameters. Each run's best-by-eval checkpoint is then
re-measured by the framework's independent evaluator — 100
deterministic episodes at a fixed seed never used in training — and
read against the published Zoo figures.

| Suite | Budget | Measured (independent eval) | Published reference | Solved bar | Verdict |
|---|---|---|---|---|---|
| CartPole-v1 | 100k steps, 8 envs | **500.0 ± 0.0** | 500 (Zoo hub card) | 475 | in ballpark — the cap |
| LunarLander-v2 | 1M steps, 16 envs | **245.9 ± 32.7** | ~260 (hub agents 257.7, 262.1) | 200 | in ballpark |

Two naming/protocol notes. The Zoo's "LunarLander-v3" is the same
environment under gymnasium 1.0; under this project's gymnasium 0.29
it is `-v2`. And CartPole carries one recorded deviation: the Zoo
decays learning rate and clip range linearly, while
`ExperimentConfig` carries scalars, so the suite runs constant
1e-3 / 0.2 — it still pins the cap. The tripwire lives in
`tests/test_calibration.py`: a shortened CartPole run through
`train()` must clear 250, so the next std-class bug trips a test
instead of hiding inside a flagship result.

**Part 2 — three-seed replication of the four flagship claims.**
Existing seed-0 artifacts count as seed 0 (re-measured under the
replication protocol where a measurement harness is involved);
seeds 1–2 are fresh runs of the shipped recipes, identical protocol.

*Racing PPO, fixed recipe (M8's `log_std_init: -1.0`), 600k steps,
seed-7 track. Claim: final eval within ~10% of peak, no terminal
collapse.*

| Seed | Peak eval | Final eval | Final as % of peak |
|---|---|---|---|
| 0 | 2858 @ 550k | 2586 | 90.5% |
| 1 | 2916 @ 600k | 2916 | 100.0% |
| 2 | 2961 @ 600k | 2961 | 100.0% |

**Holds.** No terminal collapse in 3/3 seeds. Both non-trivial
seeds visited the crash attractor transiently (seed 0 fell to 176
at 500k, seed 2 to 1493 at 525k) and recovered to their peaks —
at the old std 1.0 the same visit was absorbing. The M8 fix is a
property of the recipe, not of one lucky run.

*Match 3, 700k recipe, 1,000-episode eval. Claim (M3): trained
701.8 vs greedy 571.6 / random 568.3.*

| Seed | Trained mean |
|---|---|
| 0 | 701.8 |
| 1 | 715.9 |
| 2 | 718.0 |

**Holds, decisively and tightly** — mean 711.9, spread 16.2 points
(≈2%). Baselines re-measured under the replication eval seed:
greedy 571.6, random 564.7 (M3's 568.3 was measured under a
different eval seed; the trained numbers are the comparison that
matters, and every seed clears greedy by 130+ points).

*Racing SAC generalist, F2 exam (held-out tracks 50001–50005,
20 episodes each). Claim (M7): 100% laps, 0% crashes.*

| Seed | Progress | Lap rate | Crash rate | Laps/ep | Mean lap |
|---|---|---|---|---|---|
| 0 | 100% | 100% | 0% | 6.2 | 13.38 s |
| 1 | 100% | 100% | 0% | 6.0 | 13.82 s |
| 2 | 100% | 100% | 0% | 5.8 | 14.17 s |

**Holds, maximally** — zero variance on the claim itself; only pace
varies, by ±0.4 s. Seed 0 is its original artifact (VM-killed at
~170k of 200k steps, best checkpoint at 70k); seeds 1–2 are clean
runs at the milestone's stated 150k budget, defensible because the
seed-0 curve had plateaued by 30–70k steps. Seed 1's road here was
not clean — attempts died at 20k, ~41.5k, and ~135k steps to VM
restarts before a clean run landed; the ~135k attempt's best
checkpoint independently exams at the same 100/100/0 line. The
citable seed-1 numbers are the clean run's.

*Connect Four, shipped league recipe, 2.00M steps; tournament per
M6 (200 games/seat vs random / greedy / minimax depth-4 / own first
snapshot). Claim (M6): random ~91%, greedy 50%, minimax 50%, FPA
75.1% — the perfect seat split.*

| Seed | Random | Greedy | Minimax | Snapshot₀ | FPA |
|---|---|---|---|---|---|
| 0 | 95.0% | 50.0% (200W as 1st, 0W as 2nd) | 50.0% (same split) | 100% | 75.5% |
| 1 | 94.8% | 50.0% (same split) | 50.0% (same split) | 100% | 76.1% |
| 2 | 92.2% | **100.0%** (wins both seats) | 50.0% wins, **0 losses** (200 draws as 2nd) | 100% | **58.4%** |

**Holds, with the milestone's most interesting caveat.** Seeds 0
and 1 reproduce M6's silhouette exactly — the opening seat wins
every game against both yardsticks, the second seat wins none,
and zero games are drawn anywhere. Seed 2 breaks it in the
direction M6 named as future work: it beats greedy from *both*
seats (400-0-0) and, as second player against minimax, draws all
200 games instead of losing them. The second-seat defense deficit
is therefore not a property of the recipe — it is bimodal across
seeds, and the shipped artifact happens to sit at the
non-defending end of the distribution. Random-opponent results
are tight across seeds (95.0 / 94.8 / 92.2; seed 0's small delta
from M6's 91.2% traces to the tournament seed base — the
deterministic pairings match exactly).

**Part 3 — chores.** `summary.json` backfilled for the M7 SAC
generalist run (`runs/racing_sac_s0_77baf5ad`), reconstructed from
its config hash, train log, and eval logs and marked
`"backfilled": true` — fields the artifacts cannot support
(`final_model`, `train_seconds`) are null rather than invented, so
the cockpit lists the project's best driver honestly. The M7/M9
exam papers are versioned into `eval/papers/` as first-class
artifacts.

**A methodology note, because it shaped this milestone.** M10 ran
through repeated VM restarts that killed in-flight runs (one SAC
seed three times; both new Connect Four seeds once each). Dead
attempts are preserved with `_dead<steps>` suffixes rather than
deleted, restarts were fresh runs of the same recipe, and every
number above was re-verified against the final on-disk artifacts
before being written here — monitoring snapshots taken mid-flight
proved unreliable across restarts and were never cited.

## 2048 (M11) — the conclusive-answers game

M11 adds 2048, the roadmap's first shortlist item: single-player,
stochastic spawns, no opponent — a result that cannot be asterisked.
The env stores exponents, not tile values; one slide/merge engine
exists in two cross-verified forms (an array engine the env plays by,
and a 64-bit int / row-table engine the baselines search with —
equivalence is test-pinned over all 65,536 row patterns). A move that
does not change the board is illegal and masked; unmasked attempts are
rejected no-ops, per the Match 3 contract. Reward is the game's own
merge score. Observation is per-cell one-hot over exponents 0–15.

Baselines are the point of the milestone: `greedy_1ply` (max immediate
gain) is the acceptance bar, and a depth-adaptive `expectimax`
(2 plies at ≥4 empty cells, 3 at ≤3; frozen snake-gradient positional
heuristic + 100/empty cell) is the yardstick — 17,388 mean over 120
games, 82.5% reaching 1024 and 28.3% reaching 2048, at ~3.2 s/game.

MaskablePPO (1M steps, seed 0, γ=0.995) **clears the bar at its peak
and then gives it back**: the eval curve rises to a best checkpoint at
400k whose honest 1,000-game mean is **3,354 vs greedy's 3,073**
(+9.1%; only non-expectimax 1024 tiles in the field), then declines
sustainedly to a final model at 1,511 — below greedy. Two doctrines
did their jobs here. Best-by-eval selection is what makes the claim
true at all (the shipped artifact is the 400k checkpoint, and saying
"PPO learns 2048 stably" would be false). And the verdict rests on
the 1,000-game protocol, not the callback's 3,925 — a winner's-curse
figure across 20 selection points, recorded here as the cautionary
example. Phase 2 (the afterstate-TD n-tuple agent) was gated on
Phase 1 missing the bar and did not run; the custom-algorithm seam
remains unbuilt, and expectimax's 5.7× margin stands as the measured
gap between search and model-free learning on this game. 378 tests.

* More games behind the same contract (the real test of modularity)
  — Match 3 landed in M3 (see above); Flappy Bird landed in M4 as
  the first continuous-dynamics game (see "Continuous games");
  racing landed in M5 as the first continuous-*action* game (see
  "Racing"); multiplayer landed in M6 (Connect Four + a self-play
  league, see "Multiplayer").
* ~~Graphical watch viewer~~ — **done**: the web cockpit
  (`src/boardgame_ml/web/` + `web/`) serves a local React UI from
  the same process as its FastAPI backend. It consumes the framework
  only through existing contracts — training runs the trainer (via
  an added optional `extra_callbacks` hook, used for live progress
  and graceful stop), watch sessions step envs with the evaluator's
  `predict_action`, and run browsing parses `summary.json` files.
  The per-game `render` payload (`web/render_state.py`) is the seam
  a new game extends to appear in the viewer.
* Bot API: load a checkpoint, accept a game state, return a move —
  `eval.evaluate::predict_action` is that core already.
