"""FastAPI application for the boardgame-ml web cockpit.

One process serves the JSON API and the built React frontend:

* ``GET  /api/games`` — registered games + metadata + baseline
  reference numbers (computed once, cached).
* ``GET  /api/runs`` / ``GET /api/runs/{run_id}`` — run browsing over
  the ``runs/`` directory (the on-disk source of truth; nothing about
  runs lives in the browser or in server memory).
* ``POST /api/train`` / ``GET /api/train/{job_id}`` /
  ``GET /api/train/{job_id}/events`` (SSE) /
  ``POST /api/train/{job_id}/stop`` — background training jobs
  (see :mod:`boardgame_ml.web.jobs`).
* ``POST /api/watch`` + ``WS /api/watch/{session_id}`` — live watch
  sessions (see :mod:`boardgame_ml.web.sessions`).

``create_app(runs_dir)`` is the factory — tests point it at a fixture
runs directory; the launcher (``python -m boardgame_ml.web``) uses
``runs/`` under the current directory.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from boardgame_ml.agents.baselines import BASELINES
from boardgame_ml.envs import GAMES
from boardgame_ml.eval.evaluate import evaluate
from boardgame_ml.play import ACTION_NAMES
from boardgame_ml.training.algorithms import MODEL_CLASSES, ensure_mask_compatible
from boardgame_ml.web.jobs import (
    GAME_ALGORITHMS,
    PRESETS,
    JobManager,
    build_config,
    default_algorithm,
    pair_defaults_payload,
    pair_variants_payload,
)
from boardgame_ml.web.sessions import SessionManager

GAME_INFO: dict[str, dict[str, str]] = {
    "pong": {
        "display_name": "Powerup Pong",
        "observation": "Ball and paddle positions/velocities, scores, inventories, effects, pickups and timers (public, seat-relative)",
        "blurb": "Simultaneous two-player Pong. Accelerate bounded paddles; collect and activate enlarge, boost or shield. First to five or highest score after 90 seconds. +1 per point, −1 conceded. Baseline results depend on the opponent; use tournaments to measure strength.",
    },
    "blackjack": {
        "display_name": "Blackjack",
        "observation": (
            "Player total / 32 · usable-ace flag · dealer upcard / 10"
        ),
        "blurb": "Hit or stand against a dealer who stands on all 17s. "
        "Terminal-only rewards: +1 win, −1 loss, 0 push, +1.5 natural.",
    },
    "higher_or_lower": {
        "display_name": "Higher or Lower",
        "observation": "Current card · score · turns remaining (normalized)",
        "blurb": "Guess whether the next card (1–13, drawn with "
        "replacement) is higher or lower. +1 correct, −1 wrong, 0 tie; "
        "10 turns per episode.",
    },
    "match3": {
        "display_name": "Match 3",
        "observation": "One-hot gem planes (6, 8, 8) · moves remaining",
        "blurb": "Swap adjacent gems to match 3 or more. Matches "
        "clear, gems fall, and cascades chain with a level "
        "multiplier. 20 moves per episode; the reward is the game's "
        "own score (a matched component of 3/4/5/6+ gems scores "
        "10/25/50/100, times the cascade level).",
    },
    "flappy": {
        "display_name": "Flappy Bird",
        "observation": (
            "Bird height · vertical velocity · distance to next "
            "pipe · its gap edges (normalized)"
        ),
        "blurb": "The first continuous game: the world scrolls at "
        "20 decisions per second and the agent only chooses flap or "
        "glide. Velocity is in the observation, so no frame stacking "
        "is needed. +1 per pipe cleared, +0.01 per step of distance; "
        "crashing ends the episode.",
    },
    "racing": {
        "display_name": "Racing",
        "observation": (
            "11-ray fan to the track edges · speed · heading vs "
            "centerline tangent · lateral offset · steering"
        ),
        "blurb": "The first continuous-action game: a Box(2) of "
        "steering and throttle on a fixed closed track. Reward is "
        "checkpoint progress in metres (+100 per lap); leaving the "
        "track ends the episode. Lateral grip caps cornering speed, "
        "so braking for corners is a real, learnable skill.",
    },
    "connect_four": {
        "display_name": "Connect Four",
        "observation": (
            "Two binary planes (6, 7, 2): the discs of the player "
            "to move, and the opponent's"
        ),
        "blurb": "The first multiplayer game (M6): two players "
        "alternate dropping discs; four in a row wins, a full board "
        "draws. Rewards are the mover's: +1 win, 0 draw. The "
        "baseline rows below are *self-matches* (the named agent "
        "plays both seats), so their mean return is the share of "
        "games that end decisively — head-to-head strength lives in "
        "the tournament table (README), not on this card.",
    },
    "game2048": {
        "display_name": "2048",
        "observation": (
            "Per-cell one-hot over tile exponents 0-15: (16, 4, 4)"
        ),
        "blurb": "The classic sliding-tile game: slide and merge "
        "equal tiles (once per tile per move), a 2 (90%) or 4 (10%) "
        "spawns after every board-changing move, and the episode "
        "ends when no move changes the board. The reward is the "
        "game's own score, so the episode return is the final 2048 "
        "score. The greedy_1ply row is the roadmap's bar: the "
        "strongest agent that looks exactly one ply deep.",
    },
}

#: Baseline reference evaluations on the Games card. Every game is
#: measured over 1,000 episodes except racing (continuous episodes
#: are long), connect_four (each "episode" is a full two-player
#: self-match, and minimax moves are alpha-beta searches — 10
#: matches keep the card responsive; the count ships in the payload),
#: and game2048 (full games run ~1,000 moves and the expectimax
#: yardstick searches seconds per game — 10 games likewise).
_BASELINE_EPISODES: dict[str, int] = {
    "racing": 100,
    "connect_four": 10,
    "game2048": 10,
}

_BASE_STEP_SECONDS = 0.9  # watch-mode step interval at speed 1


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    game: str
    algorithm: str = "maskable_ppo"
    preset: str = "standard"
    timesteps: int | None = None
    learning_rate: float | None = None
    seed: int = 0
    n_envs: int | None = None
    eval_freq: int | None = None
    eval_episodes: int | None = None


class WatchRequest(BaseModel):
    game: str | None = None
    agent: str | None = None  # baseline name
    run_id: str | None = None
    checkpoint: str = "best_model"
    seed: int = 0
    # Two-player games only: a baseline name for seat 1. Omitted =
    # mirror match (seat 0's agent plays both sides).
    opponent: str | None = None


# ---------------------------------------------------------------------------
# Run browsing helpers
# ---------------------------------------------------------------------------


def _run_entry(run_dir: Path) -> dict:
    summary: dict | None = None
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            summary = None
    cfg = (summary or {}).get("config") or {}
    return {
        "run_id": run_dir.name,
        "game": cfg.get("game"),
        "algorithm": cfg.get("algorithm"),
        "seed": cfg.get("seed"),
        "config_hash": (summary or {}).get("config_hash"),
        "best_eval_mean_return": (summary or {}).get("best_eval_mean_return"),
        "train_seconds": (summary or {}).get("train_seconds"),
        "versions": (summary or {}).get("versions"),
        "config": cfg or None,
        "checkpoints": sorted(p.name for p in run_dir.glob("*.zip")),
        "complete": summary is not None,
        "created_at": run_dir.stat().st_mtime,
    }


def _frontend_dist() -> Path | None:
    """Locate the built frontend (``web/dist`` at the repo root)."""
    override = os.environ.get("BOARDGAME_ML_WEB_DIST")
    if override:
        path = Path(override)
        return path if (path / "index.html").exists() else None
    # src/boardgame_ml/web/app.py -> repo root is three parents up.
    repo_root = Path(__file__).resolve().parents[3]
    dist = repo_root / "web" / "dist"
    return dist if (dist / "index.html").exists() else None


_NOT_BUILT_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>boardgame-ml</title>
<style>body{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}
main{max-width:34rem;padding:2rem}code{background:#161b22;padding:.15em .4em;
border-radius:6px}pre{background:#161b22;padding:1rem;border-radius:8px;
overflow:auto}</style></head>
<body><main><h1>boardgame-ml cockpit</h1>
<p>The API is running, but the frontend has not been built yet.</p>
<pre>cd web
npm install
npm run build</pre>
<p>Then reload this page. For frontend development, use
<code>npm run dev</code> in <code>web/</code> instead — the Vite dev
server proxies <code>/api</code> to this backend.</p>
</main></body></html>"""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(runs_dir: str | Path = "runs") -> FastAPI:
    runs_path = Path(runs_dir)
    jobs = JobManager()
    sessions = SessionManager(runs_path)
    baseline_cache: dict[str, list[dict]] = {}

    app = FastAPI(title="boardgame-ml cockpit")
    app.state.runs_dir = runs_path
    app.state.jobs = jobs
    app.state.sessions = sessions

    # -- games ---------------------------------------------------------
    def _baselines_for(game: str) -> list[dict]:
        if game == "pong":
            return [{"name": cls.name, "mean_return": None, "win_rate": None, "episodes": 0}
                    for cls in BASELINES.values() if game in cls.games]
        if game not in baseline_cache:
            rows = []
            for cls in BASELINES.values():
                if game in cls.games:
                    result = evaluate(
                        game,
                        cls.name,
                        episodes=_BASELINE_EPISODES.get(game, 1000),
                        seed=20240,
                    )
                    rows.append(
                        {
                            "name": cls.name,
                            "mean_return": result["mean_return"],
                            "win_rate": result["win_rate"],
                            "episodes": result["episodes"],
                        }
                    )
            baseline_cache[game] = rows
        return baseline_cache[game]

    @app.get("/api/games")
    def list_games() -> list[dict]:
        games = []
        for name in GAMES:
            info = GAME_INFO.get(name, {})
            names = ACTION_NAMES.get(name, {})
            if len(names) > 24:
                # Games with hundreds of named actions (Match 3: 256)
                # summarize instead of flooding the card.
                actions = [
                    {"index": -1, "name": f"{len(names)} actions (see blurb)"}
                ]
            else:
                actions = [
                    {"index": i, "name": names.get(i, str(i))}
                    for i in sorted(names)
                ]
            games.append(
                {
                    "name": name,
                    "display_name": info.get("display_name", name),
                    "observation": info.get("observation", ""),
                    "blurb": info.get("blurb", ""),
                    "actions": actions,
                    "baselines": _baselines_for(name),
                }
            )
        return games

    @app.get("/api/meta")
    def meta() -> dict:
        return {
            "algorithms": sorted(MODEL_CLASSES),
            # Per-game algorithm choices for the Train view: exactly
            # the (game, algorithm) pairs with a shipped config, plus
            # the preselected default per game.
            "game_algorithms": GAME_ALGORITHMS,
            "default_algorithms": {
                game: default_algorithm(game) for game in GAME_ALGORITHMS
            },
            # The effective per-pair settings build_config will use
            # (mirroring configs/*.yaml — parity-tested), so the
            # Train view can show them instead of diverging silently.
            "pair_defaults": pair_defaults_payload(),
            # Named shipped variants of a pair (M7 generalists):
            # build_config(..., variant=stem) reproduces them.
            "pair_variants": pair_variants_payload(),
            "presets": {
                name: {
                    "timesteps": p["timesteps"],
                    "eval_freq": p["eval_freq"],
                    "eval_episodes": p["eval_episodes"],
                }
                for name, p in PRESETS.items()
            },
            "baselines": sorted(BASELINES),
        }

    # -- runs ----------------------------------------------------------
    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        if not runs_path.is_dir():
            return []
        entries = [
            _run_entry(d) for d in runs_path.iterdir() if d.is_dir()
        ]
        entries.sort(key=lambda e: e["created_at"], reverse=True)
        return entries

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict:
        run_dir = (runs_path / run_id).resolve()
        if run_dir.parent != runs_path.resolve() or not run_dir.is_dir():
            raise HTTPException(404, f"Unknown run {run_id!r}")
        entry = _run_entry(run_dir)
        entry["files"] = [
            {"name": p.name, "size": p.stat().st_size}
            for p in sorted(run_dir.iterdir())
            if p.is_file()
        ]
        log_path = run_dir / "train_log.csv"
        if log_path.exists():
            lines = log_path.read_text().strip().splitlines()
            entry["train_log_rows"] = max(0, len(lines) - 1)
        return entry

    # -- training jobs ---------------------------------------------------
    @app.post("/api/train")
    def start_train(req: TrainRequest) -> dict:
        try:
            config = build_config(
                game=req.game,
                algorithm=req.algorithm,
                preset=req.preset,
                timesteps=req.timesteps,
                learning_rate=req.learning_rate,
                seed=req.seed,
                output_dir=str(runs_path),
                n_envs=req.n_envs,
                eval_freq=req.eval_freq,
                eval_episodes=req.eval_episodes,
            )
            # Surface the mask guard as a 400 now, not as a failed job.
            ensure_mask_compatible(
                config.game,
                config.algorithm,
                allow_unmasked=config.allow_unmasked,
                env_kwargs=config.env_kwargs,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        job = jobs.create(config)
        return {"job_id": job.job_id, "run_id": job.run_dir.name}

    @app.get("/api/train/{job_id}")
    def train_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"Unknown job {job_id!r}")
        return job.snapshot()

    @app.get("/api/train/{job_id}/events")
    async def train_events(job_id: str) -> StreamingResponse:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"Unknown job {job_id!r}")

        async def gen():
            last: str | None = None
            while True:
                payload = json.dumps(job.snapshot(), default=str)
                if payload != last:
                    yield f"data: {payload}\n\n"
                    last = payload
                if job.status in ("completed", "failed"):
                    break
                await asyncio.sleep(0.5)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/api/train/{job_id}/stop")
    def train_stop(job_id: str) -> dict:
        job = jobs.stop(job_id)
        if job is None:
            raise HTTPException(404, f"Unknown job {job_id!r}")
        return {"job_id": job.job_id, "status": job.status, "stopping": True}

    # -- watch sessions --------------------------------------------------
    @app.post("/api/watch")
    def start_watch(req: WatchRequest) -> dict:
        try:
            session = sessions.create(
                game=req.game,
                agent=req.agent,
                run_id=req.run_id,
                checkpoint=req.checkpoint,
                seed=req.seed,
                opponent=req.opponent,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {
            "session_id": session.session_id,
            "game": session.game,
            "agent": session.agent_label,
            "opponent": session.opponent_label,
            "initial": session.state_event(),
        }

    @app.websocket("/api/watch/{session_id}")
    async def watch_ws(ws: WebSocket, session_id: str) -> None:
        session = sessions.get(session_id)
        if session is None:
            await ws.close(code=4404)
            return
        await ws.accept()
        send_lock = asyncio.Lock()

        async def send_events(events: list[dict]) -> None:
            async with send_lock:
                for event in events:
                    await ws.send_json(event)

        async def player() -> None:
            while True:
                if session.playing:
                    started = asyncio.get_running_loop().time()
                    await send_events(session.step_once())
                    await asyncio.sleep(
                        max(0.0, session.tick_seconds / max(session.speed, 0.05)
                            - (asyncio.get_running_loop().time() - started))
                    )
                else:
                    await asyncio.sleep(0.05)

        player_task = asyncio.create_task(player())
        try:
            while True:
                msg = await ws.receive_json()
                cmd = msg.get("cmd")
                if cmd == "play":
                    session.playing = True
                elif cmd == "pause":
                    session.playing = False
                elif cmd == "step":
                    if not session.playing:
                        await send_events(session.step_once())
                elif cmd == "speed":
                    try:
                        session.speed = min(
                            8.0, max(0.25, float(msg.get("value", 1.0)))
                        )
                    except (TypeError, ValueError):
                        pass
                elif cmd == "restart":
                    session.restart()
                    event = session.state_event()
                    session.event_log.append(event)
                    await send_events([event])
                elif cmd == "new_episode":
                    session.new_episode()
                    event = session.state_event()
                    session.event_log.append(event)
                    await send_events([event])
                elif cmd == "state":
                    await send_events([session.state_event()])
        except WebSocketDisconnect:
            pass
        finally:
            player_task.cancel()
            sessions.close(session_id)

    # -- static frontend ---------------------------------------------------
    dist = _frontend_dist()
    if dist is not None:
        assets = dist / "assets"
        if assets.is_dir():
            from fastapi.staticfiles import StaticFiles

            app.mount(
                "/assets", StaticFiles(directory=assets), name="assets"
            )

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(404, "Not found")
            candidate = dist / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    else:

        @app.get("/")
        def not_built() -> HTMLResponse:
            return HTMLResponse(_NOT_BUILT_PAGE)

    return app
