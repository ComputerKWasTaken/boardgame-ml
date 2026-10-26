# Powerup Pong

Pong is the first simultaneous two-player game in this framework. Both
players decide from the same pre-step state; only then does physics advance.
It uses no new runtime dependency.

## Rules and controls

A match ends at five points or 90 simulated seconds. The higher score wins;
equal scores at the timer are a draw. The timer is part of the rules and
ends by termination, not truncation. Each point pays +1 to its scorer and
−1 to the opponent. There are no other rewards.

Actions 0/1/2 mean up/idle/down; 3/4/5 mean the same movement plus activation.
An empty activation does nothing. All actions are legal, so the shipped
algorithm is ordinary PPO. Movement targets velocity, never position:
300 px/s speed, 1,800 px/s² acceleration, with braking and arena boundaries.
The deliberate paddle reset between points is separate from continuous
movement. Enlarging at an edge fits the larger paddle inside the arena.

The 960×540 arena advances at 20 decisions/s with six collision substeps.
Swept ball collisions integrate the rest of a substep after a reflection.
Ball speed starts at 420 px/s, rises by 4% per return to 720, and is capped at
900 when boosted. Impact offset and paddle velocity set the return angle,
bounded to ±60° of horizontal. The radius is 8 px; paddles start 96 px tall.
Every constructor parameter is available through YAML `env_kwargs`.

The first serve direction is seeded random; subsequent directions alternate.
Each point resets paddles, ball, active effects and arena pickup. Serving
pauses the ball for 0.5 s. Stored inventory survives points.

## Pickups and abilities

Every six seconds, if none is present, spawn one uniformly chosen pickup in
the central 30% of width and middle 70% of height. It lasts six seconds.
Ball contact collects it for the ball's last toucher. Untouched serves
cannot collect. A full inventory leaves the pickup available. Each player
can store one ability; activation consumes it immediately.
The spawn clock follows match time across points, so short rallies still
produce pickups. Scoring clears the current pickup without restarting that clock.

| Ability | Effect |
| --- | --- |
| Enlarge | Paddle is 1.5× taller for five seconds. |
| Boost | Arm a 1.35× faster next paddle return for six seconds; consumed on return, capped at 900 px/s. |
| Shield | Reflect one otherwise scoring ball within six seconds, then disappear; ownership becomes the shield owner's. |

Both activations happen before physics. Timers advance in simulation time,
so Watch speed does not change the rules. Multiple active ability types may
coexist after subsequent pickups; activating a type refreshes its duration.

## Observation and multiplayer contract

All state is public. The 36-element float32 observation is normalized;
horizontal coordinates and velocities are reflected for seat 1, so the
controlled paddle is always on the left. Vertical coordinates use y down.

| Indices | Contents |
| --- | --- |
| 0–3 | Ball x/y and velocity x/y |
| 4–7 | Own then opposing paddle y/velocity |
| 8–11 | Own/opponent score, remaining match fraction, serve countdown fraction |
| 12–19 | Own then opposing inventory one-hots: empty/enlarge/boost/shield |
| 20–25 | Own then opposing effect timer fractions: enlarge/boost/shield |
| 26–33 | Pickup x/y, pickup type one-hot, expiry fraction, next spawn fraction |
| 34–35 | Last toucher is self/opponent; both zero means no toucher |

`SimultaneousTwoPlayerEnv` is a sibling of the alternating `TwoPlayerEnv`.
Its reset returns `(obs0, obs1), info`; `step((action0, action1))` returns
the observation pair, reward pair, shared terminated/truncated flags, and
info containing absolute-seat `winner` and scores. Its raw spaces are
Tuple spaces. It is a multi-agent protocol, not a single-agent Gym env.

`SimultaneousSelfPlayWrapper` is the Gym/SB3 boundary: one seat's Discrete(6)
action, Box(36) observation and scalar point reward. It samples learner seats
and opponents through the existing league. The opponent uses the cached
pre-step observation, so it cannot react to the learner's new action.
`PlayerView` exposes prediction-only per-seat spaces and public physical
parameters to baselines, rather than the mutable raw game.

## Training, evaluation, and Watch

```powershell
.venv\Scripts\python.exe -m boardgame_ml.training.train --config configs/pong_ppo.yaml
.venv\Scripts\python.exe -m boardgame_ml.eval.tournament --game pong --agent runs/your_run/best_model.zip --opponents random pong_tracker pong_intercept --games-per-seat 50
.venv\Scripts\python.exe -m boardgame_ml.eval.pong_exam --checkpoint runs/your_run/best_model.zip --games-per-seat 50 --output runs/your_run/activation_exam.json
```

The shipped recipe uses four environments, an MLP PPO policy, gamma 0.995,
250k requested steps, snapshots every 25k, and a 70% snapshot / 10% random /
20% intercept opponent mixture. Every 25k steps, evaluate 20 matches against
the fixed intercept baseline. Best-checkpoint selection uses mean point
difference; tournament W/D/L is determined by the actual match winner.
The cockpit's quick preset requests 50k steps. Model artifacts remain in
ignored `runs/`; the config and reproduction instructions are versioned.

`pong_tracker` follows ball height and brakes; `pong_intercept` predicts
wall-bounced arrival height and uses abilities situationally. Both operate
under exactly the same acceleration and speed limits as learned policies.
They provide reference opponents, not an optimal-solution certificate.

The Games card lists available baselines without running a synchronous
tournament. Its empty metric cells mean unmeasured, not zero strength.
Watch supports baseline, checkpoint and mirror opponents. SVG rendering
shows the score, countdown, pickups, stored abilities and active timers.
Live and saved playback use the event's `tick_seconds` (0.05); old event
logs retain their 900 ms default. Pong step events add `joint_actions`
without changing the existing seat-0 `action` and `reward` fields. Opponent
run selections use `run:<run_id>:<best_model|final_model>` in the existing
`opponent` field; filesystem paths and baseline names remain supported.

Human keyboard play is not included.

## Learning check

On October 26, 2026, seed 0 requested 50,000 steps using the shipped recipe
with only `total_timesteps` changed. PPO completed 50,176 steps (whole
rollouts) in 158 seconds on this machine's CPU. Both 20-match evaluations
against the intercept baseline scored −5.0 mean point difference; the
best-by-eval checkpoint remains the first evaluation's checkpoint.

The local run is `runs/pong_ppo_s0_3d425dc3/`; its summary records versions,
configuration, snapshot list, and checkpoints. `activation_exam.json` in
that directory contains the full diagnostic. Reproduce the shorter run with:

```powershell
.venv\Scripts\python.exe -c "from dataclasses import replace; from boardgame_ml.training.config import ExperimentConfig; from boardgame_ml.training.train import train; train(replace(ExperimentConfig.from_yaml('configs/pong_ppo.yaml'), total_timesteps=50000))"
```

Evaluate the best checkpoint over 50 matches per seat against each opponent,
then repeat with only the learner's activation suppressed (`action % 3`).
Seed base is 1,000,000; seat 1 uses base + 10,000. Opponent randomness is
seeded afresh per match. The opponent retains abilities in both conditions.

| Opponent | Learner seat | Enabled W/D/L | Disabled W/D/L | Mean point difference, both conditions | Enabled ability uses |
| --- | --- | --- | --- | --- | --- |
| Random | 0 | 18/0/32 | 18/0/32 | −1.06 | 4 |
| Random | 1 | 14/0/36 | 14/0/36 | −1.20 | 4 |
| Intercept | 0 | 0/0/50 | 0/0/50 | −5.00 | 3 |
| Intercept | 1 | 0/0/50 | 0/0/50 | −5.00 | 3 |

Across all 400 diagnostic matches, enabling activation did not change the
win/loss totals or mean point difference. It did change some trajectories
and durations, and the enabled learner used 14 collected abilities. This
budget produced a weak policy (32% wins against random, none against
intercept), not strategic mastery or evidence of improvement over an
untrained policy. Disabling activation also changes future inventory and
trajectories, so this comparison is not a controlled causal ablation.

Validation: Pong rules and adapter tests, Gymnasium/SB3 checkers, existing
Connect Four self-play/tournament tests, web/config/contract regressions,
TypeScript production build, replay tick/legacy checks, and live browser
inspection of playback and saved replay. The expensive unrelated full-game
baseline-card evaluation is excluded from the regression command:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_pong.py tests/test_self_play.py tests/test_tournament.py tests/test_web.py tests/test_web_parity.py tests/test_contract.py tests/test_config.py tests/test_envs.py -q -k "not games_endpoint"
```
