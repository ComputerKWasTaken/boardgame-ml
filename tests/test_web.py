"""Web cockpit API tests (FastAPI TestClient).

Covered: games metadata, run browsing against a fixture runs dir,
the training-job lifecycle with a tiny config (reaching "completed"
with artifacts on disk), error paths, and a watch session streaming
one full blackjack episode to completion.

The watch test drives a *baseline* agent (basic strategy) rather than
a checkpoint: session creation, the WebSocket protocol, and episode
termination are what's under test, and the baseline needs no trained
artifact. Checkpoint-backed sessions are exercised end-to-end in the
manual smoke run (see README) and indirectly here by watching the
tiny training job's own best_model.zip.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from boardgame_ml.web.app import create_app


@pytest.fixture()
def fixture_runs(tmp_path):
    """A runs dir with one complete and one summary-less run."""
    run = tmp_path / "blackjack_maskable_ppo_s0_deadbeef"
    run.mkdir()
    summary = {
        "config": {
            "game": "blackjack",
            "algorithm": "maskable_ppo",
            "total_timesteps": 1000,
            "seed": 0,
            "n_envs": 4,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "allow_unmasked": False,
            "env_kwargs": {},
            "policy": "MlpPolicy",
            "policy_kwargs": {},
            "eval_episodes": 10,
            "eval_freq": 500,
            "output_dir": str(tmp_path),
            "hyperparams": {},
        },
        "config_hash": "deadbeef",
        "versions": {"boardgame_ml": "0.1.0"},
        "run_dir": str(run),
        "best_eval_mean_return": -0.04,
        "best_model": str(run / "best_model.zip"),
        "final_model": str(run / "final_model.zip"),
        "train_seconds": 12.5,
    }
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "best_model.zip").write_bytes(b"fake")
    (run / "final_model.zip").write_bytes(b"fake")
    (run / "train_log.csv").write_text(
        "timesteps,episode_return,episode_length\n100,1.0,3\n"
    )
    (tmp_path / "interrupted_run").mkdir()
    return tmp_path


@pytest.fixture()
def client(fixture_runs):
    return TestClient(create_app(fixture_runs))


def test_games_endpoint(client):
    games = client.get("/api/games").json()
    by_name = {g["name"]: g for g in games}
    assert set(by_name) == {"blackjack", "higher_or_lower", "match3", "flappy"}
    bj = by_name["blackjack"]
    assert [a["name"] for a in bj["actions"]] == ["STAND", "HIT"]
    baseline_names = {b["name"] for b in bj["baselines"]}
    assert baseline_names == {"random", "basic_strategy"}
    for b in bj["baselines"]:
        assert b["episodes"] == 1000
        assert -1.6 <= b["mean_return"] <= 1.6

    m3 = by_name["match3"]
    assert m3["display_name"] == "Match 3"
    # 256 actions are summarized, not enumerated.
    assert len(m3["actions"]) == 1
    assert {b["name"] for b in m3["baselines"]} == {"random", "greedy"}
    for b in m3["baselines"]:
        assert b["episodes"] == 1000
        assert b["mean_return"] > 0  # match-3 scores are positive points
    # No greedy > random assertion: measured honestly, 1-ply greedy is
    # NOT stronger than random over full episodes in this game (its
    # bigger clears break up the near-matches that feed cascades).
    # See DESIGN.md's Match 3 section.


def test_runs_endpoints(client):
    runs = client.get("/api/runs").json()
    by_id = {r["run_id"]: r for r in runs}
    assert set(by_id) == {
        "blackjack_maskable_ppo_s0_deadbeef",
        "interrupted_run",
    }
    entry = by_id["blackjack_maskable_ppo_s0_deadbeef"]
    assert entry["game"] == "blackjack"
    assert entry["algorithm"] == "maskable_ppo"
    assert entry["seed"] == 0
    assert entry["config_hash"] == "deadbeef"
    assert entry["best_eval_mean_return"] == pytest.approx(-0.04)
    assert entry["checkpoints"] == ["best_model.zip", "final_model.zip"]
    assert entry["complete"] is True
    assert by_id["interrupted_run"]["complete"] is False

    detail = client.get("/api/runs/blackjack_maskable_ppo_s0_deadbeef").json()
    assert detail["train_log_rows"] == 1
    assert {f["name"] for f in detail["files"]} >= {
        "summary.json",
        "best_model.zip",
    }
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/..%2Fsecret").status_code in (404, 400)


def test_meta_endpoint(client):
    meta = client.get("/api/meta").json()
    assert "maskable_ppo" in meta["algorithms"]
    assert set(meta["presets"]) == {"quick", "standard", "long"}
    assert "basic_strategy" in meta["baselines"]


def test_train_job_lifecycle(tmp_path):
    client = TestClient(create_app(tmp_path))
    resp = client.post(
        "/api/train",
        json={
            "game": "higher_or_lower",
            "algorithm": "maskable_ppo",
            "timesteps": 128,
            "n_envs": 1,
            "eval_freq": 64,
            "eval_episodes": 5,
            "seed": 7,
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    status = {}
    deadline = time.time() + 120
    while time.time() < deadline:
        status = client.get(f"/api/train/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(0.25)
    assert status["status"] == "completed", status.get("error")
    assert status["episodes_completed"] > 0
    assert status["eval_history"], "expected at least one eval result"
    assert status["best_eval_mean"] is not None
    assert status["summary"]["best_eval_mean_return"] is not None

    run_dir = tmp_path / status["run_id"]
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "best_model.zip").exists()
    assert (run_dir / "final_model.zip").exists()

    # The run shows up in browsing, and its checkpoint can be watched.
    run_ids = {r["run_id"] for r in client.get("/api/runs").json()}
    assert status["run_id"] in run_ids
    watch = client.post(
        "/api/watch", json={"run_id": status["run_id"], "seed": 1}
    )
    assert watch.status_code == 200
    assert watch.json()["game"] == "higher_or_lower"


def test_train_request_validation(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.post("/api/train", json={"game": "nope"}).status_code == 400
    assert (
        client.post(
            "/api/train", json={"game": "blackjack", "algorithm": "nope"}
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/train", json={"game": "blackjack", "preset": "nope"}
        ).status_code
        == 400
    )
    assert client.get("/api/train/unknown").status_code == 404
    assert client.post("/api/train/unknown/stop").status_code == 404


def test_watch_session_streams_full_episode(client):
    resp = client.post(
        "/api/watch",
        json={"game": "blackjack", "agent": "basic_strategy", "seed": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    initial = body["initial"]
    assert initial["type"] == "state"
    render = initial["render"]
    assert render["game"] == "blackjack"
    assert len(render["player_hand"]) == 2
    assert render["dealer_hand"] == [render["dealer_upcard"]]
    assert render["dealer_revealed"] is False

    session_id = body["session_id"]
    seen_types = []
    final_step = None
    end_event = None
    with client.websocket_connect(f"/api/watch/{session_id}") as ws:
        for _ in range(30):
            ws.send_json({"cmd": "step"})
            while True:
                event = ws.receive_json()
                seen_types.append(event["type"])
                if event["type"] == "step":
                    final_step = event
                    break
                if event["type"] == "episode_end":
                    end_event = event
                    break
            if end_event is not None:
                break
    assert end_event is not None, f"no episode_end in {seen_types}"
    assert final_step["done"] is True
    assert final_step["render"]["dealer_revealed"] is True
    assert len(final_step["render"]["dealer_hand"]) >= 2
    stats = end_event["session_stats"]
    assert stats["episodes_completed"] == 1
    assert stats["wins"] + stats["losses"] + stats["pushes"] == 1


def test_watch_validation(client):
    # Baseline without a game.
    assert client.post("/api/watch", json={"agent": "random"}).status_code == 400
    # Baseline that does not apply to the game.
    assert (
        client.post(
            "/api/watch",
            json={"game": "higher_or_lower", "agent": "basic_strategy"},
        ).status_code
        == 400
    )
    # Unknown run.
    assert client.post("/api/watch", json={"run_id": "nope"}).status_code == 400
    # Neither agent nor run.
    assert client.post("/api/watch", json={}).status_code == 400


def test_frontend_not_built_page(client, monkeypatch):
    # Force the "dist missing" branch regardless of whether a local
    # build exists on disk: with no built frontend the root serves a
    # friendly how-to-build page, while the API keeps working.
    import boardgame_ml.web.app as app_module

    monkeypatch.setattr(app_module, "_frontend_dist", lambda: None)
    bare = TestClient(create_app(client.app.state.runs_dir))
    resp = bare.get("/")
    assert resp.status_code == 200
    assert "frontend has not been built" in resp.text


def test_frontend_served_when_built(fixture_runs, tmp_path, monkeypatch):
    import boardgame_ml.web.app as app_module

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>cockpit</body></html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    monkeypatch.setattr(app_module, "_frontend_dist", lambda: dist)
    built = TestClient(create_app(fixture_runs))
    assert "cockpit" in built.get("/").text
    assert built.get("/assets/app.js").status_code == 200
    # SPA fallback serves index for client-side routes, not for /api.
    assert "cockpit" in built.get("/some/route").text
    assert built.get("/api/definitely-not-a-route").status_code == 404
