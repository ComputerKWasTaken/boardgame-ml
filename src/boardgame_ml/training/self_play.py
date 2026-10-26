"""Self-play: two-player games, trained as single-agent problems.

SB3 trains one policy against one env. A two-player game (the
contract in :mod:`boardgame_ml.envs.two_player`) becomes trainable
by *adapting* it, and this module is that adapter plus the league
that makes self-play actually teach:

* :class:`SelfPlayWrapper` presents a :class:`TwoPlayerEnv` as a
  single-agent env. The **learner** occupies one seat (chosen per
  episode; even by default, biasable toward the harder second seat
  via ``second_seat_share`` -- a policy that only ever plays one
  seat learns half a game).
  The other seat is driven by an **opponent policy**, sampled per
  episode from the league. The wrapper steps the underlying game
  twice per learner decision when needed (opponent's move, then the
  learner's), and **sign-corrects** every reward so the learner
  always sees its own result: ``+1`` when it wins, ``0`` for a draw,
  ``-1`` when the opponent's move wins the game (the two-player env
  reports that move's ``+1`` to the *opponent* -- the wrapper negates
  it, per the zero-sum convention).

* The **league** is the opponent distribution. Pure latest-vs-latest
  self-play is a documented failure mode for competitive games:
  strategies form cycles (A beats B, B beats C, C beats A) and the
  learner chases its own tail, improving against yesterday's policy
  while forgetting how to beat last month's. The fix used here is
  the classical one -- play against a **pool of frozen snapshots of
  the learner itself**, weighted toward recent ones but never
  exclusively (every snapshot keeps a floor of probability mass,
  so old strategies never fully leave the mix), plus fixed shares of
  :class:`RandomAgent` and the game's heuristic baseline, which
  anchor the bottom of the skill ladder. An optional **search
  anchor** (a minimax baseline, ``search_share``) can hold a share
  too: it is the league's refuter, the opponent that punishes
  shallow tactics the rest of the pool cannot see through. The very first snapshot
  (the near-random initial policy) is a **permanent anchor**: it is
  never evicted, so "can it still crush a beginner?" is always being
  asked.

* :class:`SnapshotCallback` grows the pool during training: every
  ``snapshot_interval`` learner steps it saves the current model to
  the run's ``pool/`` directory and registers it. Snapshots are
  loaded lazily and **cached** -- predicting through a snapshot is
  the wrapper's per-step cost, and an uncached load-per-move pool
  would tax training by orders of magnitude. The registry is capped
  (``pool_max``): the oldest non-anchor snapshots leave the sampling
  distribution as the pool slides forward, though their files stay
  in the run directory as artifacts.

Opponent policies predict through the framework's own
:func:`boardgame_ml.eval.evaluate.predict_action`, so a snapshot
plays exactly the masked, deterministic policy it would play under
evaluation -- the thing being trained against is the thing that was
measured, never an approximation of it.

Config: the league's knobs are first-class fields of
:class:`SelfPlayConfig`, populated from the experiment config's
``self_play`` block (see :class:`boardgame_ml.training.config.ExperimentConfig`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from boardgame_ml.envs.two_player import TwoPlayerEnv
from boardgame_ml.envs.base import BaseGameEnv
from boardgame_ml.envs.simultaneous import SimultaneousTwoPlayerEnv


@dataclass
class SelfPlayConfig:
    """The league's shape. Defaults are the shipped M6 recipe."""

    #: Timesteps between snapshots of the learner entering the pool.
    snapshot_interval: int = 25_000
    #: Opponent mixture (per episode): probability the opponent is a
    #: pool snapshot (the bulk), the random baseline, the game's
    #: heuristic baseline, or the search baseline. Must sum to 1
    #: (within tolerance).
    snapshot_share: float = 0.80
    random_share: float = 0.10
    heuristic_share: float = 0.10
    #: Name of the heuristic baseline for the game (must apply to it).
    heuristic: str = "connect_four_greedy"
    #: Optional second anchor: a *search* baseline (e.g. minimax),
    #: the opponent that refutes whatever shallow tactic the learner
    #: is currently exploiting. Off by default (``search_share`` 0);
    #: run 2 of M6 showed why it exists — a league with no refuter
    #: produced a policy that swept its heuristic 400-0 and was
    #: swept 0-400 by minimax (see the shipped YAML's header).
    search_share: float = 0.0
    search: str | None = None
    #: Probability the learner takes the *second* seat in a training
    #: episode (eval always plays 50/50). Connect Four's second seat
    #: is the harder skill — defense against an opponent with the
    #: initiative — and run 3's tournament showed a perfect seat
    #: split (200-0 opening, 0-200 replying): a policy can be
    #: attack-complete and defense-deficient at an even split, so
    #: the harder seat can be oversampled. 0.5 = unbiased.
    second_seat_share: float = 0.5
    #: Max pool snapshots kept on disk/in the registry. The first
    #: snapshot (index 0) is the permanent anchor and never evicts.
    pool_max: int = 24
    #: Recency weighting: snapshot rank r of n gets recency weight
    #: r**recency_exponent, mixed with a uniform floor (see
    #: OpponentPool.snapshot_weights) so recent opponents dominate
    #: but no snapshot's probability reaches zero.
    recency_exponent: float = 1.0
    uniform_floor: float = 0.25

    def __post_init__(self) -> None:
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        total = (
            self.snapshot_share
            + self.random_share
            + self.heuristic_share
            + self.search_share
        )
        if abs(total - 1.0) > 1e-6 or min(
            self.snapshot_share,
            self.random_share,
            self.heuristic_share,
            self.search_share,
        ) < 0:
            raise ValueError(
                "opponent shares (snapshot/random/heuristic/search) "
                f"must be non-negative and sum to 1, got {total}"
            )
        if self.search_share > 0 and not self.search:
            raise ValueError(
                "search_share > 0 requires naming the search baseline "
                "(`search:`), e.g. the game's minimax"
            )
        if not 0.0 <= self.second_seat_share <= 1.0:
            raise ValueError("second_seat_share must lie in [0, 1]")
        if self.pool_max < 1:
            raise ValueError("pool_max must be >= 1")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SelfPlayConfig":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown self_play keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name)
            for f in self.__dataclass_fields__.values()
        }


@dataclass
class _Snapshot:
    index: int
    path: Path
    steps: int
    anchor: bool = False
    _model: Any = field(default=None, repr=False)

    def model(self):
        if self._model is None:
            from boardgame_ml.eval.evaluate import load_sb3_model

            self._model = load_sb3_model(str(self.path))
        return self._model


class OpponentPool:
    """The league registry: baselines + frozen learner snapshots.

    Sampling is per *episode* (one opponent identity per game), which
    is what makes an episode a coherent match against one style of
    play rather than a blur of all of them.
    """

    def __init__(
        self, game: str, config: SelfPlayConfig, *, seed: int = 0
    ) -> None:
        from boardgame_ml.agents.baselines import get_baseline

        self.game = game
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._random_agent = get_baseline("random", game)
        self._heuristic_agent = get_baseline(config.heuristic, game)
        self._search_agent = (
            get_baseline(config.search, game) if config.search else None
        )
        self._snapshots: list[_Snapshot] = []
        self._next_index = 0

    # -- pool management --------------------------------------------------
    def add_snapshot(self, path: str | Path, *, steps: int) -> _Snapshot:
        snap = _Snapshot(
            index=self._next_index,
            path=Path(path),
            steps=steps,
            anchor=self._next_index == 0,
        )
        self._next_index += 1
        self._snapshots.append(snap)
        self._evict()
        return snap

    def _evict(self) -> None:
        """Slide the pool window: drop oldest non-anchor snapshots
        from the *registry*.

        Files are deliberately left on disk: each vectorized env
        holds its own pool over the same files and loads them
        lazily, so deleting a file could race another pool's first
        load of it. The registry is what bounds the opponent
        distribution; the run directory keeps the artifacts (the
        tournament reads the anchor back from there).
        """
        while len(self._snapshots) > self.config.pool_max:
            victim = next(s for s in self._snapshots if not s.anchor)
            self._snapshots.remove(victim)

    @property
    def snapshots(self) -> list[_Snapshot]:
        return list(self._snapshots)

    def snapshot_weights(self) -> np.ndarray:
        """Sampling weights over current snapshots (recency + floor).

        A convex mix of a uniform distribution (holding
        ``uniform_floor`` of the total mass) and a recency
        distribution proportional to (rank)**recency_exponent, where
        rank runs 1..n oldest-to-newest. Every snapshot keeps at
        least floor/n of the mass -- the anti-cycle floor.
        """
        n = len(self._snapshots)
        if n == 0:
            return np.zeros(0)
        ranks = np.arange(1, n + 1, dtype=np.float64)
        recency = ranks**self.config.recency_exponent
        recency /= recency.sum()
        uniform = np.full(n, 1.0 / n)
        w = (
            self.config.uniform_floor * uniform
            + (1.0 - self.config.uniform_floor) * recency
        )
        return w / w.sum()

    # -- sampling ---------------------------------------------------------
    def sample_opponent(self) -> tuple[str, Any]:
        """Draw one episode's opponent: (kind, agent-or-model).

        Kinds: ``"snapshot"``, ``"random"``, ``"heuristic"``,
        ``"search"``. Before the first snapshot exists the snapshot
        share falls back to the heuristic, so the pool is total
        from step 0.
        """
        r = self._rng.random()
        if r < self.config.random_share:
            return "random", self._random_agent
        r -= self.config.random_share
        if r < self.config.heuristic_share:
            return "heuristic", self._heuristic_agent
        r -= self.config.heuristic_share
        if r < self.config.search_share and self._search_agent is not None:
            return "search", self._search_agent
        if not self._snapshots:
            return "heuristic", self._heuristic_agent
        weights = self.snapshot_weights()
        pick = int(self._rng.choice(len(self._snapshots), p=weights))
        return "snapshot", self._snapshots[pick].model()


class SelfPlayWrapper(gym.Wrapper):
    """Present a :class:`TwoPlayerEnv` as a single-agent env.

    The wrapped env must be the *raw* game (mask access goes through
    ``env.unwrapped`` as elsewhere in the framework); the wrapper's
    own ``action_masks()`` delegates to the game whenever it is the
    learner's turn to be asked -- which is always, between wrapper
    steps: a wrapper step ends exactly when the game is waiting for
    the learner's next move (or the game is over).
    """

    def __init__(
        self,
        env: gym.Env,
        pool: OpponentPool | None = None,
        *,
        fixed_opponent: Any = None,
        fixed_opponent_kind: str = "fixed",
    ) -> None:
        super().__init__(env)
        raw = env.unwrapped
        if not isinstance(raw, TwoPlayerEnv):
            raise TypeError(
                "SelfPlayWrapper needs a TwoPlayerEnv, got "
                f"{type(raw).__name__}"
            )
        if pool is None and fixed_opponent is None:
            raise ValueError(
                "SelfPlayWrapper needs an OpponentPool or a fixed_opponent"
            )
        self.pool = pool
        self._fixed_opponent = fixed_opponent
        self._fixed_opponent_kind = fixed_opponent_kind
        self._raw: TwoPlayerEnv = raw
        self.learner_seat: int = 0
        self.opponent_kind: str | None = None
        self._opponent: Any = None
        self._seat_rng = np.random.default_rng()

    # -- mask plumbing ------------------------------------------------------
    def action_masks(self) -> np.ndarray:
        return self._raw.action_masks()

    # -- episode setup ----------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seat_rng = np.random.default_rng(seed)
        if self._fixed_opponent is not None:
            # Eval seats stay an honest 50/50 whatever the training
            # mixture says (see SelfPlayConfig.second_seat_share).
            self.learner_seat = int(self._seat_rng.integers(0, 2))
        else:
            assert self.pool is not None
            share = self.pool.config.second_seat_share
            self.learner_seat = int(self._seat_rng.random() < share)
        if self._fixed_opponent is not None:
            # Evaluation mode: one fixed opponent every episode, so
            # the eval curve measures the learner against a constant
            # yardstick instead of a moving league.
            self.opponent_kind = self._fixed_opponent_kind
            self._opponent = self._fixed_opponent
        else:
            assert self.pool is not None
            self.opponent_kind, self._opponent = self.pool.sample_opponent()
        obs, info = self.env.reset(seed=seed, options=options)
        if self.learner_seat == 1:
            # The opponent opens; play its move so the first obs the
            # learner is asked about is a state where it must move.
            obs, _, terminated, truncated, info = self._opponent_move(obs)
            assert not (terminated or truncated)  # one move can't end C4
        return obs, self._augment(info)

    # -- play ---------------------------------------------------------------
    def _opponent_move(self, obs):
        from boardgame_ml.eval.evaluate import predict_action

        action = predict_action(self._opponent, self.env, obs)
        return self.env.step(action)

    def step(self, action):
        # The learner's move.
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            # The env's reward is the mover's -- the mover was the
            # learner -- so it is already sign-correct (+1 win / 0 draw).
            return obs, float(reward), terminated, truncated, self._augment(info)
        # The opponent replies; if its move ends the game, the env's
        # +1 belongs to the opponent -- negate it for the learner.
        obs, opp_reward, terminated, truncated, info = self._opponent_move(obs)
        learner_reward = -float(opp_reward) if (terminated or truncated) else 0.0
        return obs, learner_reward, terminated, truncated, self._augment(info)

    def _augment(self, info: dict) -> dict:
        info = dict(info)
        info["learner_seat"] = self.learner_seat
        info["opponent"] = self.opponent_kind
        return info


class SimultaneousSelfPlayWrapper(BaseGameEnv):
    """Gym single-seat adapter; each learner step advances one joint tick.

    Unlike alternating self-play, nonterminal rewards are preserved.
    The opponent decides from the cached pre-step observation.
    """
    def __init__(self, env, pool=None, *, fixed_opponent=None, fixed_opponent_kind="fixed"):
        super().__init__()
        if not isinstance(env, SimultaneousTwoPlayerEnv):
            raise TypeError("Expected a simultaneous two-player game")
        if pool is None and fixed_opponent is None:
            raise ValueError("An opponent pool or fixed opponent is required")
        self.env, self.pool = env, pool
        self.action_space = env.single_action_space
        self.observation_space = env.single_observation_space
        self._fixed_opponent = fixed_opponent
        self._fixed_opponent_kind = fixed_opponent_kind
        self._seat_rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seat_rng = np.random.default_rng(seed)
        share = self.pool.config.second_seat_share if self.pool is not None else .5
        self.learner_seat = int(self._seat_rng.random() < share)
        if self._fixed_opponent is not None:
            self.opponent_kind, self._opponent = self._fixed_opponent_kind, self._fixed_opponent
        else:
            self.opponent_kind, self._opponent = self.pool.sample_opponent()
        self._observations, info = self.env.reset(seed=seed, options=options)
        return self._observations[self.learner_seat], self._augment(info)

    def step(self, action):
        from boardgame_ml.eval.evaluate import predict_action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid learner action: {action!r}")
        other = 1 - self.learner_seat
        reply = predict_action(self._opponent, self.env.player_view(other), self._observations[other])
        actions = [reply, reply]
        actions[self.learner_seat] = int(action)
        self._observations, rewards, terminated, truncated, info = self.env.step(tuple(actions))
        return self._observations[self.learner_seat], float(rewards[self.learner_seat]), terminated, truncated, self._augment(info)

    def _augment(self, info):
        return dict(info, learner_seat=self.learner_seat, opponent=self.opponent_kind,
                    action_mask=self.action_masks())

    def close(self):
        self.env.close()


def self_play_wrapper(env, *args, **kwargs):
    cls = SimultaneousSelfPlayWrapper if isinstance(env.unwrapped, SimultaneousTwoPlayerEnv) else SelfPlayWrapper
    return cls(env, *args, **kwargs)


class SnapshotCallback(BaseCallback):
    """Periodically freeze the learner into the opponent pool.

    Saves ``pool/snapshot_<index>_s<steps>.zip`` under the run dir
    every ``config.snapshot_interval`` *learner* timesteps and
    registers it with every vec env's pool. A snapshot at step 0 --
    the untrained policy -- is taken on training start: it is the
    pool's permanent anchor and the tournament's "did it learn
    anything at all?" yardstick.
    """

    def __init__(
        self,
        pools: list[OpponentPool],
        run_dir: str | Path,
        config: SelfPlayConfig,
    ) -> None:
        super().__init__()
        self._pools = pools
        self._pool_dir = Path(run_dir) / "pool"
        self._config = config
        self._counter = 0
        self._next_at = config.snapshot_interval

    def _on_training_start(self) -> None:
        self._snapshot(steps=0)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_at:
            self._snapshot(steps=self.num_timesteps)
            self._next_at += self._config.snapshot_interval
        return True

    def _snapshot(self, *, steps: int) -> None:
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        path = self._pool_dir / f"snapshot_{self._counter}_s{steps}"
        self.model.save(str(path))
        zip_path = path.with_suffix(".zip")
        for pool in self._pools:
            pool.add_snapshot(zip_path, steps=steps)
        self._counter += 1


__all__ = [
    "OpponentPool",
    "SelfPlayConfig",
    "SelfPlayWrapper",
    "SnapshotCallback",
    "SimultaneousSelfPlayWrapper",
    "self_play_wrapper",
]
