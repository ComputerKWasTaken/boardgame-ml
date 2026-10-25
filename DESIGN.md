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
(per-player hands, hidden information, turn order) do not fit this
base class and, when that milestone arrives, will live behind a
separate PettingZoo-style AEC contract, with adapters to this
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

## Roadmap beyond M1

* More games behind the same contract (the real test of modularity)
  — Match 3 landed in M3 (see above); Flappy Bird landed in M4 as
  the first continuous-dynamics game (see "Continuous games");
  racing landed in M5 as the first continuous-*action* game (see
  "Racing"); a multi-player game via a PettingZoo-style wrapper is
  the remaining frontier.
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
