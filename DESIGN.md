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
│   └── higher_or_lower.py HigherOrLowerEnv
├── agents/
│   └── baselines.py       Random / basic-strategy / probability agents
├── training/
│   ├── config.py          ExperimentConfig dataclass tree (YAML-loaded)
│   └── train.py           SB3 trainer CLI
├── eval/
│   └── evaluate.py        Greedy evaluation + strategy-agreement metrics
└── play.py                Text "watch it play" viewer
configs/                   One YAML per experiment
tests/                     Contract, rules, config, and baseline tests
runs/                      Training output (checkpoints, logs, summaries)
```

### The game contract

A game is a `BaseGameEnv` (a `gymnasium.Env`):

* `observation_space` — any Gymnasium space; M1 games use small
  normalized `Box` vectors.
* `action_space` — `Discrete(n)`. Larger action spaces are fine as
  long as legality is expressible as a mask.
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
| 5 | **An unlearnable representation** (a board game encoded as scalar category IDs fed to an MLP, erasing spatial structure). | Out of M1 scope, but the contract doesn't force flat vectors: observation spaces are per-game, and a future board game can expose multi-channel planes for a CNN policy without touching the trainer. |

Two legacy ideas were kept, deliberately: the small abstract game
interface (here in Gymnasium shape, so the ecosystem's tooling applies),
and per-game checkpoint organization with metadata (here: run dirs
with `summary.json`).

## Roadmap beyond M1

* More games behind the same contract (the real test of modularity),
  including a multi-player game via a PettingZoo-style wrapper.
* Graphical watch viewer (the text viewer in `play.py` isolates the
  decision point it will need).
* Bot API: load a checkpoint, accept a game state, return a move —
  `eval.evaluate::predict_action` is that core already.
