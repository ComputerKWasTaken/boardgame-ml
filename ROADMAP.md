# Roadmap — boardgame-ml

*Where this project might go next. Written 2026-10-26, after M9. Nothing here is
scheduled; it's the idea board, with the question each idea answers, what it
costs, and what it depends on. House rules still apply: acceptance stated in
advance, the exam/tournament is the verdict (not the training curve), honest
failures are findings, push unless porked.*

## Where we are

Nine milestones in: six games (Higher or Lower, Blackjack, Match 3, Flappy,
Racing, Connect Four), three generator families for racing, a self-play league,
a local web cockpit, permanent training diagnostics, 265 tests. The framework's
core seams — `BaseGameEnv`, the two-player contract, the algorithm registry,
YAML recipes mirrored into the cockpit under parity tests — have survived every
game added so far with minimal changes. The open scientific threads are:
generalization (racing), seat bias (two-player games), and single-seed
evidence (everything).

---

## A. New games

### A1. 2048 — *the conclusive-answers game*
- **Question:** can the framework learn long-horizon planning under stochastic
  spawns, with no opponent and no seat effects contaminating the result?
- **Scope:** 4×4 env, Discrete(4) with no-op move masking, log-scaled tile
  observation; score = game score. Yardsticks: random-legal, greedy 1-ply, and
  an expectimax search agent (the minimax-equivalent for a stochastic game).
- **The interesting risk:** published experience says generic DQN/PPO plateaus
  around the 1024 tile. The literature's champion is TD learning over
  *afterstates* with n-tuple features — a natural first "bring your own
  algorithm" entry next to the SB3 registry, if SB3 plateaus as expected.
- **Acceptance sketch:** beat greedy 1-ply on mean score across ≥1,000 games;
  report the full tile-reached distribution, not just the mean.
- **Cost: M** (env is small; expectimax + possible custom algorithm are the work).

### A2. Reversi — *Connect Four's rematch on fair ground*
- **Question:** does the M6 self-play league recipe actually work when the game
  isn't structurally rigged for the first player?
- **Scope:** 8×8 env on the existing TwoPlayerEnv contract — zero framework
  changes expected, which is itself part of the test. Yardsticks: greedy
  (mobility + corners), alpha-beta minimax at a couple of depths.
- **Acceptance sketch:** first-player advantage measured and small (expect
  single digits, vs Connect Four's 75%); shipped agent beats greedy ≥65%
  combined seats and is not swept by minimax.
- **Cost: S–M** (league, tournament, and cockpit patterns all exist).

### A3. Tron (light-cycles) — *multiplayer with no first move*
- **Question:** what does self-play look like when the seat problem is removed
  structurally — simultaneous moves, no turns to alternate?
- **Scope:** needs a new simultaneous-move two-player contract (a sibling of
  TwoPlayerEnv, not a hack of it). Grid env, both trails grow, crash = loss,
  draw on simultaneous death. Cockpit renderer is simple SVG lines — and
  likely the best-looking Watch view in the project.
- **Cost: M** (the contract is the design work; the env is easy).

### A4. Snake — *the quick palate cleanser*
- **Question:** long-horizon survival planning in a discrete grid; also a
  stress test of "step limits are truncations, never terminations" at scale.
- **Scope:** single-player env, growing body, food spawns. Yardstick with
  character: a Hamiltonian-cycle follower is *provably* hard to beat on
  survival but slow and dumb on score — a great "robust vs fast" contrast of
  the same shape racing produced (heuristic vs SAC).
- **Cost: S.**

### A5. Tetris — *the stretch build*
- **Question:** can learned play approach hand-crafted feature play
  (Dellacherie-style) on a game where the observation is the whole strategy?
- **Scope:** full piece queue + hold is probably over-scope; a simplified
  10-wide well, 7-bag randomizer, line-clear scoring. The env is the biggest
  single-player build on this list, and the literature is humbling.
- **Cost: L. Park until A1/A2 land.**

### A6. Kuhn / Leduc poker — *imperfect information*
- **Question:** bluffing and belief states — the first game where the
  observation deliberately hides the state.
- **Scope:** tiny solved poker variants; needs agents that handle hidden
  information (the current registry assumes full observability; RecurrentPPO
  or explicit belief features are candidate routes).
- **Cost: M, research-flavored. Park behind A3's contract work.**

---

## B. Racing sequels

### B1. Mixed-family union generalist
- **Question:** does training on the v2+v3 union produce one driver that beats
  both family specialists on *both* exams — breadth over depth?
- **Scope:** SAC, shipped recipe, per-episode family mixing; sit both existing
  exams (50001–50005, 60001–60005). No new code beyond a mixing env kwarg.
- **Cost: S** (mostly training wall-clock; SAC is slow on CPU).

### B2. Braking-zone study
- **Question:** M9's residual failures are speed management — long straight
  into a 15 m hairpin. Is it perception (the 11-ray fan can't see far enough
  at full speed) or control (it sees, but can't plan the shed)?
- **Scope:** one-variable experiments: (a) add a curvature-lookahead
  observation feature, A/B on the two killer F3 tracks; (b) curriculum with
  progressively longer straights. Diagnostics.csv now exists — use it.
- **Cost: S–M. The most scientific item on this board.**

### B3. Fixed-PPO generalist, full budget
- **Question:** M8's probe (fixed recipe, killed at ~174k) aced robustness at
  parade pace — 22.4 s laps vs SAC's 13.4 s. Does PPO's pace ever arrive with
  a full 600k budget and the `log_std_init` fix?
- **Scope:** one long run + the M7 exam. Cheap in wall-clock (PPO is ~35×
  SAC's step rate); mostly a patience exercise.
- **Cost: S.**

### B4. Generator v4 — *new physics, not just new shapes*
- **Ideas:** variable track width; grip variation ("weather"); point-to-point
  rally stages (breaks the closed-loop assumption — checkpoints and reward
  need rethinking, in a good way).
- **Cost: M–L depending on appetite. Width variation is the cheap one.**

### B5. Traffic — *ghost opponents on track*
- **Question:** from time-trials to racing: slower heuristic cars as moving
  obstacles, overtake-or-follow decisions.
- **Cost: L. Moonshot; changes the game's nature. Park.**

---

## C. Framework science & hygiene

### C1. Multi-seed replication protocol
- **The uncomfortable truth:** every headline number in this project is one
  training seed. **Fix:** flagship recipes (racing SAC generalist, Connect
  Four league, Match 3) re-run at 3 seeds; report mean ± spread; make seed a
  first-class, always-logged config field. Turns anecdotes into results.
- **Cost: S in code, M in wall-clock.** High value per dollar.

### C2. Benchmark calibration
- **Question:** does our `train()` pipeline reproduce *published* SB3 results
  on standard Gymnasium classics (LunarLander, CartPole, Acrobot)?
- **Why:** the std bug hid for two milestones because nothing anchored our
  numbers to the outside world. A calibration suite is the tripwire.
- **Cost: S.**

### C3. Diagnostics, surfaced
- diagnostics.csv exists per run (M8) but the cockpit doesn't show it. Add
  the key curves (explained variance, KL, clip fraction, policy std) to the
  Runs view; flag the known pathologies (std pinned at init, EV ≈ 0) so the
  next collapse announces itself in the UI instead of in a post-mortem.
- **Cost: S–M.**

### C4. Known warts, swept
- The M7 SAC generalist run dir has no `summary.json`, so the cockpit Runs
  view can't list the project's best driver. Backfill/regenerate it.
- Exam papers (M7/M9 JSON) live in side folders; version them into the repo
  as first-class artifacts so results are reproducible from a clone.
- **Cost: S.**

### C5. Algorithm zoo expansion (as needed, not for its own sake)
- RecurrentPPO (prerequisite for A6), QR-DQN / dueling variants for the
  discrete games, A2C as a cheap on-policy control when PPO misbehaves.
  Each addition must arrive with a game that needs it — no shelf-ware.

---

## D. Cockpit & product

### D1. Human play mode
- Drive the car / place the tiles / make the moves against (or alongside)
  trained agents, keyboard-driven, in Watch. Fun, demo-able, and produces a
  human baseline — the one yardstick we don't have.
- **Cost: M.** Probably the highest delight-per-effort item here.

### D2. Face-off view
- Two agents, same track/board, side-by-side Watch with synced clocks.
  Makes every exam result *visible* instead of tabular.
- **Cost: S–M** (session machinery already streams one agent).

### D3. Tournament bracket view
- The tournament harness exists; give its results a bracket/leaderboard UI
  instead of JSON. Pairs naturally with A2 (Reversi).
- **Cost: S.**

### D4. Public showcase
- The repo is private and reads like a lab notebook. A curated public
  README (results tables, cockpit screenshots, the M8 autopsy as the
  centerpiece story) turns two weeks of evenings into a portfolio artifact.
- **Cost: S. Owner's call on going public — nothing ships without it.**

---

## E. Moonshots & parking lot

- **E1. Policy distillation → browser.** Distill the SAC generalist into a
  small net, export ONNX, run it live in the cockpit page. You'd race a
  neural net in your browser tab. (Pairs with D1.)
- **E2. Agent decathlon.** One rating system (Elo over tournament results)
  spanning every game and agent, tracked over time in the cockpit.
- **E3. Learned opponent modeling in self-play** — league members that adapt
  to *which* snapshot they're facing. Research-y; behind A2/A3 evidence.
- **Explicitly parked:** multi-car physics racing (B5 full version), Tetris
  with hold/queue (A5 deluxe), elevation/banking in the racing physics,
  anything involving an LLM playing the games (gimmick; against the house
  style).

---

## Suggested sequencing (opinionated, argue with it)

1. **A1 (2048)** — new territory, conclusive by construction, medium cost.
2. **B1 (union generalist)** in parallel — cheap, closes the racing
   generalization story.
3. **C1 (replication)** woven into both — every new flagship run goes
   3-seed from now on, so the debt stops growing.
4. **A2 (Reversi)** — the league's fair trial.
5. **B2 (braking-zone)** — the best pure-science item; do it while racing
   is still warm.
6. **D1 (human play)** — the reward for all of the above.
