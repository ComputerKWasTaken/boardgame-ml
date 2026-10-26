# Roadmap — boardgame-ml (trimmed)

*Cut down 2026-10-26 from the full idea board. The rule for surviving: an item
stays only if its answer changes what we believe, it unlocks something else,
or it's cheap and closes a story. Continuations for their own sake, garnish,
and process machinery were cut. House rules unchanged: acceptance stated in
advance, the exam/tournament is the verdict, honest failures are findings,
push unless porked.*

## State of play

M1–M9 done: six games, three racing track families, self-play league, web
cockpit, permanent diagnostics, the PPO std autopsy, cross-family transfer
measured. 265 tests. What remains below is the whole board — twelve items.

---

## The shortlist

### 1. 2048 — the conclusive-answers game
Single-player, stochastic spawns, no seats, no opponent: pure planning, and
the result can't be asterisked. Expectimax is the yardstick. Published
experience says SB3 plateaus around the 1024 tile; if it does, the literature's
champion — TD learning over afterstates with n-tuple features — becomes the
framework's first custom algorithm beside the SB3 registry. Acceptance:
beat greedy 1-ply on mean score over ≥1,000 games; report the tile-reached
distribution, not just the mean. **Cost: M. Unlocks: the custom-algorithm seam.**

**Status: DONE 2026-10-26.** MaskablePPO's best checkpoint (400k of
1M) beat greedy_1ply 3,354 vs 3,073 over 1,000 games; the run plateaus
and regresses after its peak (final model 1,511), and expectimax
(17,388; 28% reach 2048) shows how far search still outruns model-free
learning here. The n-tuple agent was not needed to clear the bar; the
custom-algorithm seam remains unbuilt.

### 2. Reversi, then AlphaZero-lite — the league's fair trial, then its ceiling
Reversi ports onto the existing two-player contract untouched, with a
first-player edge small enough to measure instead of drowning in. It answers
whether M6's league recipe works on a fair game. Second act, same board: an
MCTS + policy/value-net agent in the AlphaZero style — the one major algorithm
family the project hasn't touched, and the method whose search patches exactly
the tactical blindness PPO showed in Connect Four. It can sit the Connect Four
tournament too, as a guest. **Cost: S–M for Reversi; L for the MCTS agent.**

### 3. Union generalist — close the racing generalization story
One SAC trained on the v2+v3 mix, examined on both families' papers. Breadth
vs depth, settled. Rider, nearly free: the fixed-recipe PPO generalist at full
600k budget trains alongside (PPO is ~35× SAC's step rate) and answers whether
its pace ever arrives post-fix. **Cost: S, mostly wall-clock.**

### 4. Braking-zone study — the purest science left in racing
M9 isolated the deficit: shedding speed from a long straight into a 15 m
hairpin. Perception or control? One-variable A/Bs: a curvature-lookahead
observation feature vs baseline on the two killer F3 tracks; a straight-length
curriculum as the alternative hypothesis. **Cost: S–M.**

### 5. Tron — the one structural novelty
Simultaneous moves: no first move exists, so the seat problem dies by
construction. Needs a simultaneous-move contract (a sibling of TwoPlayerEnv).
Likely the best Watch view in the project. **Cost: M.**

### 6. Human play mode (+ a behavior-cloning rider)
Keyboard play in the cockpit against/alongside trained agents. Buys the one
yardstick we lack — a human baseline — and the data for one cheap rider:
clone Archer's driving, fine-tune the clone, race clone vs tabula-rasa RL.
**Cost: M. Highest delight-per-effort on the board.**

### 7. "What actually matters" ablation
Match 3's lesson, tested deliberately instead of anecdotally: one game,
three observation representations × shaped-vs-terminal reward, algorithms
held constant. If representation and reward dominate algorithm choice —
as the project's history keeps hinting — that's a finding worth having
cleanly. **Cost: S–M.**

### 8. Thumby port — the one moonshot
Distill a small trained policy (Flappy is the natural candidate) and run it
on a Thumby Color. Trained in the cloud, playing in your pocket. **Cost: S–M,
mostly distillation + a constrained runtime.**

### 9. Public showcase — whenever the owner says go
Curated public README: results tables, cockpit screenshots, the M8 autopsy
as the centerpiece. This project is quietly becoming a serious portfolio
piece for the job search; the showcase is an hour of curation, not a build.
**Cost: S. Nothing goes public without the owner's explicit call.**

### 10. Handicaps — modular action restrictions
Train agents with part of the action space pruned, as a config-layer
feature rather than per-game variants: static masks for Discrete games,
ANDed with the legality mask (Connect Four with no center column);
per-dimension clamps for Box spaces via one generic action wrapper (racing
with `throttle_min: 0` is a car with no brakes). The fields live in the
training YAML and the cockpit Train view, parity-tested like every other
knob, and the restriction is stamped into `summary.json` next to the seed
so runs stay honestly comparable. Flagship demo: brakeless racing — M9
named the residual deficit as speed management into hairpins, and a driver
that *cannot* brake has to learn speed discipline natively; its line
should look visibly different in Watch. Physics handicaps (a Pong paddle
that can only rise and falls back by gravity) are out of scope for the
module — that restriction lives in the world's response, not the action
set, and stays an env knob. Acceptance: the brakeless agent completes
laps on the F2 exam; lap time and crash rate reported against the
unrestricted generalist. **Cost: S. Item 7's sibling — ablation of the
action space instead of observation/reward.**

---

## Standing disciplines (not milestones — woven into everything)

- **Multi-seed replication.** *Established in M10 (2026-10-26): the
  four flagship claims were rerun at 3 seeds each — see DESIGN.md's
  M10 section for the tables and verdicts.* Every headline number so
  far is one training
  seed. From here: flagship runs go 3 seeds, results report the spread, and
  seed is stamped in every run's summary. The existing flagships get their
  reruns opportunistically, alongside new work — the debt stops growing
  first, then shrinks.
- **Benchmark calibration.** *Established in M10 (2026-10-26):
  `train()` reproduces the published Zoo PPO figures on CartPole-v1
  (500.0) and LunarLander-v2 (245.9 vs ~260); the tripwire smoke
  test lives in `tests/test_calibration.py`.* The train() pipeline must reproduce published
  SB3 results on LunarLander/CartPole. M8's std bug hid for two milestones
  because nothing anchored our numbers to the outside world; this is the
  tripwire, run whenever training code changes.

## Chores (folded into the next racing milestone, not roadmap items)

- ~~Backfill `summary.json` for the M7 SAC generalist run so the cockpit lists
  the project's best driver.~~ **Done in M10** — reconstructed from the
  run's artifacts and marked `"backfilled": true`.
- ~~Version the M7/M9 exam papers into the repo as first-class artifacts.~~
  **Done** (`eval/papers/`).

---

## What was cut, and why (veto any of these back in)

- **More racing generators (v4 shapes, width, weather, rally)** — family
  transfer is proven; more shapes are more-of-same. Physics randomization
  and dose-response curves: real questions, SAC-slow answers, nothing
  downstream depends on them.
- **Tetris, Sokoban, Klondike, Minesweeper, Mastermind, Snake, Hex,
  Checkers, Go, Gomoku, Quoridor, et al.** — each is either a known RL
  graveyard, a measuring exercise against solved yardsticks, or a variation
  on a game already on the shortlist. Snake survives only as a distillation
  donor if Flappy doesn't fit the Thumby.
- **Hidden-information games (poker, Hanabi, Battleship)** — a real axis,
  parked whole: it drags belief-state algorithm work behind it and no
  current thread needs it.
- **Quadcopter, parking, endless runners** — racing/Flappy cousins; the
  marginal insight doesn't pay for new physics.
- **Continual learning, non-transitivity census, exploration-noise sweep,
  failure-census tooling** — interesting trivia and process; no decision
  anywhere depends on their answers.
- **HPO sweep machinery, runs database, throughput project, CI** — process
  fluff. The hand recipes, folders + summary.json, and ad-hoc censuses have
  carried nine milestones fine.
- **Face-off view, tournament bracket UI, diagnostics charts** — cockpit
  garnish. Watch + tables already serve the need; human play (item 6) is
  the one cockpit addition that adds a capability.
- **One-brain multi-task net, cross-game Elo decathlon, browser ONNX
  agents, opponent modeling** — moonshots that lost to the Thumby port,
  which does "policy in a delightful place" better and cheaper.

## Suggested order

1. **2048** and **3 (union generalist)** in parallel — new territory plus
   a cheap closing chapter.
2. **Reversi → AlphaZero-lite** — the two-player story, fair then strong.
3. **Braking-zone** and **handicaps** while racing is warm; **Tron** when
   a new contract sounds fun.
4. **Human play** as the victory lap; **ablation** whenever a quiet week
   wants a clean result; **Thumby** when it sparks joy.
5. **Showcase** floats — it ships the day you say so.
