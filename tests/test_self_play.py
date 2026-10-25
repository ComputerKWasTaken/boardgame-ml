"""Self-play tests (M6): wrapper sign conventions, the league pool,
and an end-to-end training smoke run.

The sign tests use scripted opponents because the property under
test is bookkeeping, not learning: a forced win must reach the
learner as +1 from *either* seat, and a forced loss as -1 -- the
seat-dependent negation is exactly where a self-play adapter would
quietly corrupt the training signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from boardgame_ml.agents.baselines import BaselineAgent
from boardgame_ml.envs import make_env
from boardgame_ml.training.config import ExperimentConfig
from boardgame_ml.training.self_play import (
    OpponentPool,
    SelfPlayConfig,
    SelfPlayWrapper,
)
from boardgame_ml.training.train import train


class FixedColumnAgent(BaselineAgent):
    """Scripted opponent: always drops in the same column."""

    name = "fixed_column"
    games = ("connect_four",)

    def __init__(self, column: int) -> None:
        self.column = column

    def select_action(self, env, obs) -> int:
        return self.column


class ScriptedAgent(BaselineAgent):
    """Scripted opponent: cycles through a fixed column sequence.

    Needed for seat-1 win scenarios: a single-column opponent wins
    any pure stacking race against the second player (it moves
    first), so the script must spread its discs harmlessly.
    """

    name = "scripted"
    games = ("connect_four",)

    def __init__(self, columns: list[int]) -> None:
        self.columns = columns
        self._i = 0

    def select_action(self, env, obs) -> int:
        col = self.columns[self._i % len(self.columns)]
        self._i += 1
        return col


def _wrapper_with_seat(want_seat: int, opponent) -> SelfPlayWrapper:
    """A wrapper whose current episode has the learner in ``want_seat``.

    The seat is drawn at reset from the wrapper's seeded RNG; probe
    seeds until the draw matches (each seed is tried on a fresh
    wrapper; the kept episode is untouched past its reset).
    """
    for seed in range(64):
        wrapped = SelfPlayWrapper(
            make_env("connect_four"), fixed_opponent=opponent
        )
        wrapped.reset(seed=seed)
        if wrapped.learner_seat == want_seat:
            return wrapped
    raise AssertionError("no seed produced the wanted seat")


def test_learner_win_as_first_player_scores_plus_one():
    # Opponent parks in column 6; the learner stacks column 0.
    wrapped = _wrapper_with_seat(0, FixedColumnAgent(6))
    rewards = []
    done = False
    while not done:
        _, reward, terminated, truncated, info = wrapped.step(0)
        rewards.append(reward)
        done = terminated or truncated
    assert rewards[:-1] == [0.0] * (len(rewards) - 1)
    assert rewards[-1] == 1.0
    assert info["winner"] == 0 == wrapped.learner_seat


def test_learner_win_as_second_player_scores_plus_one():
    # Opponent (seat 0) spreads its discs over columns 5/6 (never
    # four anywhere); the learner stacks column 0 and completes the
    # vertical on its fourth drop.
    wrapped = _wrapper_with_seat(1, ScriptedAgent([5, 6]))
    done = False
    final = None
    while not done:
        _, reward, terminated, truncated, info = wrapped.step(0)
        final = reward
        done = terminated or truncated
    assert final == 1.0
    assert info["winner"] == 1 == wrapped.learner_seat


def test_learner_loss_as_second_player_scores_minus_one():
    # The opponent stacks column 0 and wins on its fourth disc; the
    # learner's discs (column 6) never threaten. The winning move
    # happens *inside* the wrapper's step — its +1 belongs to the
    # opponent, and the wrapper must hand the learner -1.
    wrapped = _wrapper_with_seat(1, FixedColumnAgent(0))
    rewards = []
    done = False
    while not done:
        _, reward, terminated, truncated, info = wrapped.step(6)
        rewards.append(reward)
        done = terminated or truncated
    assert rewards[-1] == -1.0
    assert info["winner"] == 0 != wrapped.learner_seat


def test_learner_loss_as_first_player_scores_minus_one():
    # The learner spreads its own discs harmlessly (5/6, never four)
    # while the opponent stacks column 0 and wins on its fourth
    # disc, inside the learner's fourth step.
    wrapped = _wrapper_with_seat(0, FixedColumnAgent(0))
    driver = ScriptedAgent([5, 6])
    rewards = []
    done = False
    while not done:
        _, reward, terminated, truncated, info = wrapped.step(
            driver.select_action(wrapped.env, None)
        )
        rewards.append(reward)
        done = terminated or truncated
    assert rewards[-1] == -1.0
    assert info["winner"] == 1 != wrapped.learner_seat


def test_wrapped_env_passes_check_env():
    wrapped = SelfPlayWrapper(
        make_env("connect_four"), fixed_opponent=FixedColumnAgent(3)
    )
    check_env(wrapped)


def test_wrapper_rejects_single_agent_games():
    with pytest.raises(TypeError):
        SelfPlayWrapper(
            make_env("blackjack"), fixed_opponent=FixedColumnAgent(0)
        )


def test_wrapper_info_carries_seat_and_opponent():
    wrapped = _wrapper_with_seat(0, FixedColumnAgent(6))
    _, info = wrapped.reset(seed=123)
    assert info["learner_seat"] == wrapped.learner_seat
    assert info["opponent"] == "fixed"


# -- the pool ---------------------------------------------------------------


def _pool(tmp_path, **overrides) -> OpponentPool:
    cfg = SelfPlayConfig(**overrides)
    return OpponentPool("connect_four", cfg, seed=0)


def test_pool_weights_sum_to_one_and_favor_recent(tmp_path):
    pool = _pool(tmp_path)
    for i in range(4):
        pool.add_snapshot(tmp_path / f"s{i}.zip", steps=i * 1000)
    w = pool.snapshot_weights()
    assert w.shape == (4,)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()  # the anti-cycle floor: nobody reaches zero
    assert w[-1] > w[0]  # recency tilt
    uniform_floor_share = 0.25 / 4
    assert w[0] >= uniform_floor_share - 1e-9


def test_pool_evicts_oldest_non_anchor(tmp_path):
    pool = _pool(tmp_path, pool_max=3)
    for i in range(6):
        pool.add_snapshot(tmp_path / f"s{i}.zip", steps=i)
    kept = pool.snapshots
    assert len(kept) == 3
    assert kept[0].index == 0 and kept[0].anchor  # the anchor never leaves
    assert [s.index for s in kept] == [0, 4, 5]  # window slid forward


def test_pool_without_snapshots_falls_back_to_heuristic(tmp_path):
    pool = _pool(
        tmp_path, snapshot_share=1.0, random_share=0.0, heuristic_share=0.0
    )
    kind, _ = pool.sample_opponent()
    assert kind == "heuristic"  # total from step 0: no dead share


def test_pool_config_validation():
    with pytest.raises(ValueError):
        SelfPlayConfig(snapshot_share=0.5, random_share=0.5, heuristic_share=0.5)
    with pytest.raises(ValueError):
        SelfPlayConfig.from_dict({"nonsense": 1})
    with pytest.raises(ValueError):
        SelfPlayConfig(snapshot_interval=0)
    # A search share without a named search baseline is a config bug.
    with pytest.raises(ValueError):
        SelfPlayConfig(
            snapshot_share=0.9, random_share=0.0, heuristic_share=0.0,
            search_share=0.1,
        )


def test_pool_samples_the_search_anchor(tmp_path):
    """With a search share set, the pool deals the minimax anchor at
    roughly that rate (the league's refuter must actually show up)."""
    pool = _pool(
        tmp_path,
        snapshot_share=0.0,
        random_share=0.0,
        heuristic_share=0.0,
        search_share=1.0,
        search="connect_four_minimax",
    )
    kind, agent = pool.sample_opponent()
    assert kind == "search"
    assert agent.name == "connect_four_minimax"


# -- training integration -----------------------------------------------------


def test_self_play_training_smoke(tmp_path):
    cfg = ExperimentConfig(
        game="connect_four",
        algorithm="maskable_ppo",
        total_timesteps=512,
        n_envs=2,
        eval_freq=256,
        eval_episodes=10,
        seed=0,
        output_dir=str(tmp_path),
        hyperparams={
            "n_steps": 128,
            "batch_size": 256,
            "n_epochs": 2,
            "ent_coef": 0.01,
            "gae_lambda": 0.95,
        },
        self_play={"snapshot_interval": 256, "heuristic": "connect_four_greedy"},
    )
    summary = train(cfg)
    run_dir = Path(summary["run_dir"])
    assert summary["best_eval_mean_return"] is not None
    # The league grew: the untrained anchor plus in-training snapshots.
    pool_files = sorted(p.name for p in (run_dir / "pool").glob("*.zip"))
    assert pool_files[0] == "snapshot_0_s0.zip"
    assert len(pool_files) >= 2
    assert summary["pool_snapshots"] == pool_files
    assert summary["self_play"]["snapshot_interval"] == 256
    disk_summary = json.loads((run_dir / "summary.json").read_text())
    assert disk_summary["config"]["self_play"]["heuristic"] == "connect_four_greedy"


def test_self_play_on_single_agent_game_is_refused(tmp_path):
    cfg = ExperimentConfig(
        game="blackjack",
        algorithm="maskable_ppo",
        total_timesteps=128,
        output_dir=str(tmp_path),
        self_play={"snapshot_interval": 64},
    )
    with pytest.raises(ValueError, match="not a two-player game"):
        train(cfg)


def test_second_seat_share_biases_training_seats_only(tmp_path):
    """second_seat_share steers the *training* seat draw: at 1.0 the
    learner always defends, at 0.0 always opens — while a
    fixed-opponent (eval) wrapper stays 50/50 by construction (its
    seats below come from the seeded probe, both seats reachable)."""
    cfg = SelfPlayConfig(second_seat_share=1.0)
    pool = OpponentPool("connect_four", cfg, seed=0)
    wrapped = SelfPlayWrapper(make_env("connect_four"), pool=pool)
    for seed in range(5):
        wrapped.reset(seed=seed)
        assert wrapped.learner_seat == 1

    cfg0 = SelfPlayConfig(second_seat_share=0.0)
    pool0 = OpponentPool("connect_four", cfg0, seed=0)
    wrapped0 = SelfPlayWrapper(make_env("connect_four"), pool=pool0)
    for seed in range(5):
        wrapped0.reset(seed=seed)
        assert wrapped0.learner_seat == 0

    with pytest.raises(ValueError):
        SelfPlayConfig(second_seat_share=1.5)
