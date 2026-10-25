"""Watch sessions: a trained agent (or baseline) playing live.

A session owns one env and one resolved agent. The server steps the
env with the agent's greedy policy — the evaluator's own
:func:`boardgame_ml.eval.evaluate.predict_action`, so what you watch
is exactly the policy that was evaluated — and streams step events to
the browser over a WebSocket.

Protocol (server → client), all JSON:

* ``state`` — the session's current position (initial event, and the
  first event of each new episode): episode number, render object,
  session stats.
* ``step`` — one action taken: action index + name, reward, running
  episode return, the render object *after* the step (with hidden
  information revealed when the episode just ended).
* ``episode_end`` — episode return, step count, updated session stats.

Client → server control messages: ``play``, ``pause``, ``step``,
``speed`` (with ``value``), ``restart`` (new session seed sequence),
``new_episode`` (abandon the current episode and deal a fresh one).

Every event the server emits is also appended to the session's
``event_log``; the frontend's "Save replay" persists that log to
localStorage and can replay it purely client-side.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from boardgame_ml.envs import make_env
from boardgame_ml.eval.evaluate import predict_action, resolve_agent
from boardgame_ml.play import ACTION_NAMES


def _action_label(game: str, action) -> tuple[Any, str]:
    """(JSON-safe action, display name) for one taken action.

    Discrete actions serialize as-is and name through ``ACTION_NAMES``;
    continuous (``Box``) actions have no per-action names — they
    serialize as a list of floats and display as their components.
    """
    if isinstance(action, np.ndarray):
        flat = [float(a) for a in action.reshape(-1)]
        if game == "racing" and len(flat) == 2:
            return flat, f"steer {flat[0]:+.2f} · throttle {flat[1]:+.2f}"
        return flat, "[" + ", ".join(f"{v:+.2f}" for v in flat) + "]"
    names = ACTION_NAMES.get(game, {})
    return action, names.get(action, str(action))
from boardgame_ml.web.render_state import build_render


class WatchSession:
    """One agent playing one game, stepped by the server."""

    def __init__(
        self,
        *,
        game: str,
        agent,
        agent_label: str,
        seed: int = 0,
    ) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.game = game
        self.agent = agent
        self.agent_label = agent_label
        self.base_seed = seed
        self.env = make_env(game, render_mode="human")
        self.playing = False
        self.speed = 1.0
        self.event_log: list[dict] = []
        # Episode / session bookkeeping.
        self.episode = 0
        self.step_in_episode = 0
        self.episode_return = 0.0
        self._history: list[dict] = []  # higher_or_lower guess history
        self.stats = {
            "episodes_completed": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "total_return": 0.0,
            "mean_return": 0.0,
        }
        self._obs = None
        self._info: dict = {}
        self._start_episode()

    # -- episode lifecycle ------------------------------------------------
    def _start_episode(self) -> None:
        self.episode += 1
        self.step_in_episode = 0
        self.episode_return = 0.0
        self._history = []
        self._obs, self._info = self.env.reset(
            seed=self.base_seed + self.episode - 1
        )

    def restart(self) -> None:
        """Reset the whole session: stats cleared, episode 1 redealt."""
        self.episode = 0
        self.stats = {
            "episodes_completed": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "total_return": 0.0,
            "mean_return": 0.0,
        }
        self._start_episode()

    def new_episode(self) -> None:
        """Abandon the current episode (not scored) and deal a fresh one."""
        self._start_episode()

    # -- events -------------------------------------------------------------
    def _render(self, *, reveal: bool = False) -> dict:
        render = build_render(self.game, self.env, self._info, reveal=reveal)
        if self.game == "higher_or_lower":
            render["history"] = list(self._history)
        return render

    def state_event(self) -> dict:
        return {
            "type": "state",
            "session_id": self.session_id,
            "game": self.game,
            "agent": self.agent_label,
            "episode": self.episode,
            "step_in_episode": self.step_in_episode,
            "episode_return": self.episode_return,
            "render": self._render(),
            "session_stats": dict(self.stats),
            "at": time.time(),
        }

    def step_once(self) -> list[dict]:
        """Advance one action; returns the events to emit (1–3)."""
        render_before = self._render()
        action = predict_action(self.agent, self.env, self._obs)
        self._obs, reward, terminated, truncated, self._info = self.env.step(action)
        reward = float(reward)
        done = bool(terminated or truncated)
        self.step_in_episode += 1
        self.episode_return += reward
        action, action_name = _action_label(self.game, action)

        if self.game == "higher_or_lower":
            self._history.append(
                {
                    "card": render_before.get("current_card"),
                    "action": action,
                    "action_name": action_name,
                    "next_card": self._render().get("current_card"),
                    "reward": reward,
                }
            )

        events = [
            {
                "type": "step",
                "session_id": self.session_id,
                "game": self.game,
                "agent": self.agent_label,
                "episode": self.episode,
                "step_in_episode": self.step_in_episode,
                "action": action,
                "action_name": action_name,
                "reward": reward,
                "episode_return": self.episode_return,
                "done": done,
                "render": self._render(reveal=done),
                "session_stats": dict(self.stats),
                "at": time.time(),
            }
        ]
        if done:
            self._score_episode()
            events.append(
                {
                    "type": "episode_end",
                    "session_id": self.session_id,
                    "game": self.game,
                    "agent": self.agent_label,
                    "episode": self.episode,
                    "episode_return": self.episode_return,
                    "steps": self.step_in_episode,
                    "session_stats": dict(self.stats),
                    "at": time.time(),
                }
            )
            self._start_episode()
            events.append(self.state_event())
        self.event_log.extend(events)
        return events

    def _score_episode(self) -> None:
        s = self.stats
        s["episodes_completed"] += 1
        s["total_return"] += self.episode_return
        if self.episode_return > 0:
            s["wins"] += 1
        elif self.episode_return < 0:
            s["losses"] += 1
        else:
            s["pushes"] += 1
        s["mean_return"] = s["total_return"] / s["episodes_completed"]

    def close(self) -> None:
        self.env.close()


class SessionManager:
    """Owns live watch sessions (in-memory; local single-user tool)."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._sessions: dict[str, WatchSession] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        game: str | None = None,
        agent: str | None = None,
        run_id: str | None = None,
        checkpoint: str = "best_model",
        seed: int = 0,
    ) -> WatchSession:
        """Resolve the agent spec and open a session.

        ``run_id`` selects a checkpoint from a run directory (the game
        comes from the run's summary); ``agent`` is otherwise a
        baseline name (``random`` / ``basic_strategy`` /
        ``probability``) and ``game`` is required.
        """
        if run_id is not None:
            run_dir = self._run_dir(run_id)
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                raise ValueError(f"Run {run_id!r} has no summary.json")
            summary = json.loads(summary_path.read_text())
            game = summary["config"]["game"]
            ckpt = run_dir / f"{checkpoint}.zip"
            if not ckpt.exists():
                raise ValueError(
                    f"Run {run_id!r} has no checkpoint {checkpoint}.zip"
                )
            agent_obj = resolve_agent(str(ckpt), game)
            label = f"{run_id} · {checkpoint}"
        elif agent is not None:
            if game is None:
                raise ValueError("game is required when agent is a baseline")
            agent_obj = resolve_agent(agent, game)
            label = agent
        else:
            raise ValueError("Provide either run_id or agent")

        session = WatchSession(
            game=game, agent=agent_obj, agent_label=label, seed=seed
        )
        session.event_log.append(session.state_event())
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def _run_dir(self, run_id: str) -> Path:
        run_dir = (self.runs_dir / run_id).resolve()
        if run_dir.parent != self.runs_dir.resolve() or not run_dir.is_dir():
            raise ValueError(f"Unknown run {run_id!r}")
        return run_dir

    def get(self, session_id: str) -> WatchSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def close(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()
