"""Physics, simultaneous decisions, and end-to-end Pong integration."""

import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

from boardgame_ml.agents.baselines import BaselineAgent, get_baseline
from boardgame_ml.envs.pong import BOOST, ENLARGE, SHIELD, PongEnv
from boardgame_ml.envs.simultaneous import predict_joint
from boardgame_ml.eval.evaluate import resolve_agent
from boardgame_ml.eval.tournament import play_match, play_pairing
from boardgame_ml.training.config import ExperimentConfig
from boardgame_ml.training.self_play import SimultaneousSelfPlayWrapper
from boardgame_ml.training.train import train
from boardgame_ml.web.app import create_app
from boardgame_ml.web.sessions import SessionManager, WatchSession


def playing(**kwargs):
    env = PongEnv(**kwargs)
    env.reset(seed=42)
    env.serve_remaining = 0.0
    return env


class Idle(BaselineAgent):
    def select_action(self, env, obs):
        return 1


def imminent_miss(env, conceder=0):
    env.serve_remaining = 0.0
    env.ball_x = (
        env.ball_radius + 1 if conceder == 0 else env.width - env.ball_radius - 1
    )
    env.ball_y = env.height / 2
    env.ball_vx = -600 if conceder == 0 else 600
    env.ball_vy = 0.0


def test_seeded_actions_reproduce_complete_matches():
    a, b = PongEnv(match_seconds=10), PongEnv(match_seconds=10)
    for env in (a, b):
        env.reset(seed=9)
    rng = np.random.default_rng(2)
    for _ in range(200):
        actions = tuple(int(x) for x in rng.integers(6, size=2))
        sa, sb = a.step(actions), b.step(actions)
        np.testing.assert_array_equal(sa[0], sb[0])
        assert sa[1:] == sb[1:]
        assert sum(sa[1]) == 0
        assert a.observation_space.contains(sa[0])
        if sa[2]:
            break
    assert a.done


def test_mirrored_public_observations_include_inventory_and_ownership():
    env = playing()
    env.ball_x, env.ball_vx = 120, -600
    env.paddle_y = [100, 400]
    env.inventory = [BOOST, SHIELD]
    env.last_touch = 0
    env.pickup = [ENLARGE, 400, 200, 3]
    left, right = env._observations()
    assert left[0] + right[0] == pytest.approx(1)
    assert left[2] == -right[2]
    assert left[4] == right[6]
    np.testing.assert_array_equal(left[12:16], right[16:20])
    assert left[26] + right[26] == pytest.approx(1)
    assert tuple(left[34:]) == (1, 0)
    assert tuple(right[34:]) == (0, 1)


def test_acceleration_reversal_speed_and_displacement_limits():
    env = playing()
    # Isolate continuous motion from the explicitly allowed between-point reset.
    env.ball_vx = env.ball_vy = 0.0
    previous_y, previous_v = env.paddle_y[0], env.paddle_v[0]
    for action in [2] * 10 + [0] * 20 + [1] * 10:
        env.step((action, 1))
        assert abs(env.paddle_v[0]) <= env.paddle_speed
        assert abs(env.paddle_y[0] - previous_y) <= env.paddle_speed * env.dt + 1e-8
        # Away from boundaries, acceleration is bounded too.
        if env.paddle_height / 2 < env.paddle_y[0] < env.height - env.paddle_height / 2:
            assert (
                abs(env.paddle_v[0] - previous_v)
                <= env.paddle_acceleration * env.dt + 1e-8
            )
        previous_y, previous_v = env.paddle_y[0], env.paddle_v[0]
    assert (
        env.paddle_height / 2 <= env.paddle_y[0] <= env.height - env.paddle_height / 2
    )


def test_intercept_cannot_save_a_physically_unreachable_return():
    env = playing(points_to_win=1)
    env.paddle_y[0] = env.paddle_height / 2
    env.ball_x, env.ball_y = 70, env.height - 30
    env.ball_vx, env.ball_vy = -900, 0
    agent = get_baseline("pong_intercept", "pong")
    for _ in range(4):
        action = agent.select_action(env.player_view(0), env.observe(0))
        _, reward, done, _, info = env.step((action, 1))
        if done:
            break
    assert done and info["winner"] == 1 and reward == (-1, 1)
    assert env.paddle_y[0] < env.height / 2


@pytest.mark.parametrize("seat", [0, 1])
def test_swept_paddle_collision_at_maximum_speed(seat):
    env = playing(physics_substeps=1)
    face = env.paddle_inset + env.paddle_width / 2 + env.ball_radius
    env.ball_x = face + 20 if seat == 0 else env.width - face - 20
    env.ball_y = env.paddle_y[seat]
    env.ball_vx, env.ball_vy = (-900 if seat == 0 else 900), 0
    _, reward, done, _, _ = env.step((1, 1))
    assert not done and reward == (0, 0)
    assert env.ball_vx > 0 if seat == 0 else env.ball_vx < 0
    assert env.last_touch == seat


def test_spin_and_angle_speed_caps():
    stationary, moving = playing(), playing()
    for env in (stationary, moving):
        env.ball_y = env.paddle_y[0] + 20
        env.ball_vx, env.ball_vy = -900, 0
    moving.paddle_v[0] = moving.paddle_speed
    stationary._return_ball(0)
    moving._return_ball(0)
    assert moving.ball_vy > stationary.ball_vy
    assert math.hypot(moving.ball_vx, moving.ball_vy) <= moving.normal_speed_cap + 1e-8
    assert abs(math.atan2(moving.ball_vy, moving.ball_vx)) <= math.radians(60)


@pytest.mark.parametrize(
    "owner,full,expected",
    [(None, False, False), (0, False, True), (1, False, True), (0, True, False)],
)
def test_swept_pickup_ownership_and_full_inventory(owner, full, expected):
    env = playing(physics_substeps=1)
    env.last_touch = owner
    env.inventory = [SHIELD if full else 0, 0]
    env.ball_x, env.ball_y, env.ball_vx, env.ball_vy = 450, 270, 900, 0
    env.pickup = [BOOST, 475, 270, 3]
    env.step((1, 1))
    assert (env.pickup is None) == expected
    if expected:
        assert env.inventory[owner] == BOOST


def test_activation_both_seats_and_effect_expiry():
    env = playing()
    env.inventory = [ENLARGE, SHIELD]
    env.step((4, 4))
    assert env.inventory == [0, 0]
    assert env.paddle_size(0) == 144
    assert env.effects[1][2] == pytest.approx(6 - env.dt)
    for _ in range(121):
        env.step((1, 1))
    assert env.effects == [[0, 0, 0], [0, 0, 0]]
    assert env.paddle_size(0) == 96


def test_boost_consumes_next_return_and_respects_cap():
    env = playing()
    env.inventory[0] = BOOST
    env.ball_x, env.ball_y, env.ball_vx, env.ball_vy = 60, 270, -900, 0
    env.step((4, 1))
    assert math.hypot(env.ball_vx, env.ball_vy) == pytest.approx(900)
    assert env.effects[0][1] == 0 and env.inventory[0] == 0


def test_shield_reflects_once_and_assigns_touch():
    env = playing()
    env.inventory[0] = SHIELD
    imminent_miss(env)
    _, reward, _, _, _ = env.step((4, 1))
    assert reward == (0, 0) and env.ball_vx > 0
    assert env.last_touch == 0 and env.effects[0][2] == 0
    imminent_miss(env)
    _, reward, _, _, _ = env.step((1, 1))
    assert reward == (-1, 1)


def test_point_reset_preserves_inventory_and_alternates_serve():
    env = playing()
    original_direction = math.copysign(1, env.ball_vx)
    env.inventory = [BOOST, ENLARGE]
    env.effects[0][0] = 5
    env.pickup = [SHIELD, 400, 100, 3]
    env.spawn_remaining = 2.0
    imminent_miss(env)
    _, reward, done, truncated, _ = env.step((1, 1))
    assert reward == (-1, 1) and not done and not truncated
    assert env.inventory == [BOOST, ENLARGE] and env.pickup is None
    assert env.effects == [[0, 0, 0], [0, 0, 0]]
    assert env.serve_remaining > 0
    assert math.copysign(1, env.ball_vx) == -original_direction
    assert env.elapsed == pytest.approx(env.dt)
    assert env.spawn_remaining == pytest.approx(2.0 - env.dt)


def test_pickup_spawn_bounds_and_expiry():
    env = playing()
    env.spawn_remaining = env.dt / 2
    env.step((1, 1))
    kind, x, y, _ttl = env.pickup
    assert 1 <= kind <= 3 and 0.35 * env.width <= x <= 0.65 * env.width
    assert 0.15 * env.height <= y <= 0.85 * env.height
    env.pickup[3] = env.dt / 2
    env.spawn_remaining = 5
    env.step((1, 1))
    assert env.pickup is None


@pytest.mark.parametrize("scores,winner", [([2, 1], 0), ([1, 2], 1), ([2, 2], None)])
def test_match_timer_is_termination_and_uses_final_score(scores, winner):
    env = playing(match_seconds=0.05)
    env.scores = scores
    _, reward, done, truncated, info = env.step((1, 1))
    assert done and not truncated and info["winner"] == winner
    assert info["end_reason"] == "time_limit" and reward == (0, 0)
    with pytest.raises(ValueError, match="over"):
        env.step((1, 1))


@pytest.mark.parametrize("seat", [0, 1])
def test_adapter_preserves_nonterminal_point_rewards_for_each_seat(seat):
    raw = PongEnv()
    env = SimultaneousSelfPlayWrapper(raw, fixed_opponent=Idle())
    env.reset(seed=3)
    env.learner_seat = seat
    imminent_miss(raw, conceder=1 - seat)
    _, reward, done, _, info = env.step(1)
    assert reward == 1 and not done and info["learner_seat"] == seat
    imminent_miss(raw, conceder=seat)
    assert env.step(1)[1] == -1


def test_joint_prediction_reads_identical_pre_step_state():
    class Recording(Idle):
        def __init__(self):
            self.seen = []

        def select_action(self, env, obs):
            assert not hasattr(env, "ball_x")
            self.seen.append(obs.copy())
            return 2

    env = playing()
    obs = env._observations()
    agents = Recording(), Recording()
    actions = predict_joint(agents, env, obs)
    np.testing.assert_array_equal(agents[0].seen[0], obs[0])
    np.testing.assert_array_equal(agents[1].seen[0], obs[1])
    assert env.elapsed == 0 and actions == (2, 2)


def test_gym_and_sb3_single_seat_checkers():
    env = SimultaneousSelfPlayWrapper(PongEnv(), fixed_opponent=Idle())
    check_env(env, skip_render_check=True)
    sb3_check_env(env)


def test_tournament_and_watch_use_winner_instead_of_return(monkeypatch):
    import boardgame_ml.eval.tournament as module

    class FinalScore(PongEnv):
        def step(self, actions):
            self.done, self.winner = True, 1
            return self._observations(), (3.0, -3.0), True, False, self._info()

    monkeypatch.setattr(module, "make_env", lambda *args, **kwargs: FinalScore())
    assert play_match("pong", Idle(), Idle()) == 1
    record = play_pairing("pong", Idle(), Idle(), games_per_seat=2)
    assert record["combined"]["wins"] == 2 and record["combined"]["losses"] == 2
    session = WatchSession(
        game="pong", agent=Idle(), agent_label="idle", seed=0, opponent=Idle()
    )
    session.env = FinalScore()
    session._start_episode()
    session.step_once()
    assert session.stats["losses"] == 1 and session.stats["total_return"] == 3


def test_pong_card_does_not_run_baseline_evaluation(monkeypatch, tmp_path):
    import boardgame_ml.web.app as module

    monkeypatch.setattr(module, "GAMES", {"pong": PongEnv})
    monkeypatch.setattr(
        module,
        "evaluate",
        lambda *args, **kwargs: pytest.fail("Pong card must not run tournaments"),
    )
    client = TestClient(create_app(tmp_path))
    card = client.get("/api/games").json()[0]
    assert {b["name"] for b in card["baselines"]} == {
        "random",
        "pong_tracker",
        "pong_intercept",
    }
    assert all(b["mean_return"] is None for b in card["baselines"])


def test_websocket_joint_actions_and_replay_roundtrip(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/watch",
        json={"game": "pong", "agent": "pong_tracker", "opponent": "pong_intercept"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["initial"]["tick_seconds"] == 0.05
    with client.websocket_connect(f"/api/watch/{body['session_id']}") as ws:
        ws.send_json({"cmd": "step"})
        event = ws.receive_json()
        assert event["joint_actions"] == [1, 1]
        assert event["tick_seconds"] == 0.05
        render = event["render"]
        assert render["game"] == "pong" and len(render["paddles"]) == 2
        replay = json.loads(json.dumps({"events": [body["initial"], event]}))
        assert replay["events"][1]["joint_actions"] == [1, 1]


def test_training_snapshot_reload_and_watch_checkpoint(tmp_path):
    cfg = ExperimentConfig(
        game="pong",
        algorithm="ppo",
        total_timesteps=64,
        n_envs=2,
        eval_freq=32,
        eval_episodes=2,
        output_dir=str(tmp_path),
        env_kwargs={"match_seconds": 1.0},
        hyperparams={"n_steps": 16, "batch_size": 32, "n_epochs": 1},
        self_play={"heuristic": "pong_intercept", "snapshot_interval": 32},
    )
    summary = train(cfg)
    assert len(summary["pool_snapshots"]) >= 2
    model = resolve_agent(summary["best_model"], "pong", env_kwargs=cfg.env_kwargs)
    assert model.action_space.n == 6
    manager = SessionManager(tmp_path)
    from pathlib import Path

    run_id = Path(summary["run_dir"]).name
    session = manager.create(run_id=run_id, opponent=f"run:{run_id}:best_model")
    assert session.env.match_seconds == 1.0
    assert len(session.step_once()[0]["joint_actions"]) == 2


def test_live_playback_cadence_pause_step_and_speed(tmp_path):
    client = TestClient(create_app(tmp_path))
    body = client.post(
        "/api/watch", json={"game": "pong", "agent": "pong_tracker"}
    ).json()
    with client.websocket_connect(f"/api/watch/{body['session_id']}") as ws:
        ws.send_json({"cmd": "speed", "value": 2})
        ws.send_json({"cmd": "play"})
        events = [ws.receive_json() for _ in range(5)]
        # At 2x, four 50 ms simulation intervals take about 100 ms.
        # Wide tolerance handles scheduling; the old 900 ms cadence fails.
        assert 0.06 <= events[-1]["at"] - events[0]["at"] < 1.0
        ws.send_json({"cmd": "pause"})
        ws.send_json({"cmd": "state"})
        state = ws.receive_json()
        while state["type"] != "state":
            state = ws.receive_json()
        ws.send_json({"cmd": "step"})
        event = ws.receive_json()
        assert event["step_in_episode"] == state["step_in_episode"] + 1


def test_reports_identify_opponent_and_simultaneous_seats():
    from boardgame_ml.eval.evaluate import evaluate
    from boardgame_ml.eval.tournament import format_report, run_tournament

    result = evaluate(
        "pong", "pong_tracker", episodes=2, env_kwargs={"match_seconds": 0.05}
    )
    assert result["opponent"] == "pong_intercept" and result["push_rate"] == 1
    tournament = run_tournament(
        "pong",
        "pong_tracker",
        ["random"],
        games_per_seat=1,
        env_kwargs={"match_seconds": 0.05},
    )
    assert (
        "seat0_advantage" in tournament and "first_player_advantage" not in tournament
    )
    assert "seat 0" in format_report(tournament)


def test_text_play_uses_joint_actions(monkeypatch, capsys):
    from boardgame_ml import play as module

    monkeypatch.setattr(
        module,
        "make_env",
        lambda *args, **kwargs: PongEnv(render_mode="human", match_seconds=0.05),
    )
    module.play("pong", "pong_tracker")
    output = capsys.readouterr().out
    assert "P1: idle" in output and "P2: idle" in output and "Episode return" in output


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0},
        {"physics_substeps": 1.5},
        {"ball_speed": 1000},
        {"paddle_height": 600},
        {"spin": float("nan")},
    ],
)
def test_invalid_physics_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PongEnv(**kwargs)
