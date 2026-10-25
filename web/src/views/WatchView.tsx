import { useEffect, useRef, useState } from "react";
import { api, watchSocketURL } from "../api";
import Board from "../components/Boards";
import { store } from "../storage";
import type {
  BlackjackRender,
  FlappyRender,
  HolRender,
  Match3Render,
  RacingRender,
  Replay,
  RunEntry,
  GameInfo,
  SessionStats,
  WatchEvent,
  WatchTarget,
} from "../types";

const BASE_MS = 900;

function fmt(v: number): string {
  return v >= 0 ? `+${v}` : `${v}`;
}

function feedLine(ev: WatchEvent): string {
  const head = `Ep ${ev.episode} · ${ev.action_name ?? ev.action}`;
  if (ev.game === "higher_or_lower" && ev.render) {
    const r = ev.render as HolRender;
    const last = r.history[r.history.length - 1];
    if (last) {
      return `${head}: ${last.card} → ${last.next_card} (${fmt(last.reward)})`;
    }
  }
  if (ev.game === "blackjack" && ev.render) {
    const r = ev.render as BlackjackRender;
    return `${head} — player ${r.player_total}${r.player_soft ? " soft" : ""} vs ${r.dealer_upcard}`;
  }
  if (ev.game === "match3" && ev.render) {
    const r = ev.render as Match3Render;
    const casc = r.last_move && r.last_move.cascades > 1 ? ` · cascade ×${r.last_move.cascades}` : "";
    return `${head} (${fmt(ev.reward ?? 0)}) — score ${r.score} · ${r.moves_left} moves left${casc}`;
  }
  if (ev.game === "flappy" && ev.render) {
    const r = ev.render as FlappyRender;
    return `${head} (${fmt(ev.reward ?? 0)}) — score ${r.score} · bird at y ${Math.round(r.bird_y)}`;
  }
  if (ev.game === "racing" && ev.render) {
    const r = ev.render as RacingRender;
    return `${head} (${fmt(ev.reward ?? 0)}) — lap ${r.laps + 1} · ${Math.round(r.progress_fraction * 100)}% · ${Math.round(r.speed * 3.6)} km/h`;
  }
  return `${head} (${fmt(ev.reward ?? 0)})`;
}

function Transport({
  playing,
  onToggle,
  onStep,
  speed,
  onSpeed,
  onRestart,
}: {
  playing: boolean;
  onToggle: () => void;
  onStep: () => void;
  speed: number;
  onSpeed: (v: number) => void;
  onRestart: () => void;
}) {
  return (
    <div className="transport">
      <button className="btn primary" onClick={onToggle}>
        {playing ? "Pause" : "Play"}
      </button>
      <button className="btn" onClick={onStep} disabled={playing}>
        Step
      </button>
      <button className="btn" onClick={onRestart}>
        Restart
      </button>
      <label className="speed-label">
        Speed
        <select value={speed} onChange={(e) => onSpeed(Number(e.target.value))}>
          {[0.5, 1, 2, 4].map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

function StatsBar({
  stats,
  episode,
  episodeReturn,
}: {
  stats: SessionStats | null;
  episode: number;
  episodeReturn: number;
}) {
  if (!stats) return null;
  return (
    <div className="stats-bar">
      <span>Episode {episode}</span>
      <span>Episode return {fmt(episodeReturn)}</span>
      <span>
        Session: {stats.episodes_completed} eps · W {stats.wins} / L{" "}
        {stats.losses} / P {stats.pushes} · mean{" "}
        {stats.episodes_completed
          ? stats.mean_return.toFixed(3)
          : "—"}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Live watch (server-driven session over WebSocket)
// ---------------------------------------------------------------------------

function LiveWatch({ target }: { target: WatchTarget }) {
  const [render, setRender] = useState<WatchEvent["render"]>(undefined);
  const [stats, setStats] = useState<SessionStats | null>(null);
  const [episode, setEpisode] = useState(1);
  const [episodeReturn, setEpisodeReturn] = useState(0);
  const [feed, setFeed] = useState<string[]>([]);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [label, setLabel] = useState("");
  const [game, setGame] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const logRef = useRef<WatchEvent[]>([]);

  useEffect(() => {
    let cancelled = false;
    logRef.current = [];
    setFeed([]);
    setSaved(false);
    setPlaying(false);

    const body: Record<string, unknown> =
      target.kind === "run"
        ? { run_id: target.runId, checkpoint: target.checkpoint, seed: 0 }
        : target.kind === "baseline"
          ? { game: target.game, agent: target.agent, seed: 0 }
          : {};
    api
      .startWatch(body)
      .then((resp) => {
        if (cancelled) return;
        setLabel(resp.agent);
        setGame(resp.game);
        applyEvent(resp.initial);
        logRef.current = [resp.initial];
        store.history.add({
          label: resp.agent,
          game: resp.game,
          at: Date.now(),
        });
        const ws = new WebSocket(watchSocketURL(resp.session_id));
        wsRef.current = ws;
        ws.onmessage = (msg) => {
          const ev = JSON.parse(msg.data) as WatchEvent;
          logRef.current.push(ev);
          applyEvent(ev);
        };
        ws.onerror = () => setError("WebSocket error");
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target.nonce]);

  const applyEvent = (ev: WatchEvent) => {
    if (ev.render) setRender(ev.render);
    if (ev.session_stats) setStats(ev.session_stats);
    setEpisode(ev.episode);
    setEpisodeReturn(ev.episode_return);
    if (ev.type === "step") {
      setFeed((f) => [feedLine(ev), ...f].slice(0, 60));
    }
    if (ev.type === "episode_end") {
      setFeed((f) =>
        [`Ep ${ev.episode} finished — return ${fmt(ev.episode_return)}`, ...f].slice(0, 60),
      );
    }
  };

  const send = (obj: Record<string, unknown>) =>
    wsRef.current?.send(JSON.stringify(obj));

  const toggle = () => {
    if (playing) {
      send({ cmd: "pause" });
      setPlaying(false);
    } else {
      send({ cmd: "play" });
      setPlaying(true);
    }
  };

  const saveReplay = () => {
    const replay: Replay = {
      id: Math.random().toString(36).slice(2, 10),
      name: `${game} · ${label}`,
      game,
      agent: label,
      createdAt: Date.now(),
      events: logRef.current,
    };
    store.replays.add(replay);
    setSaved(true);
  };

  return (
    <div className="watch-layout">
      <div className="panel">
        <div className="watch-head">
          <strong>{label || "Connecting…"}</strong>
          {error && <span className="error-box">{error}</span>}
        </div>
        {render && <Board render={render} />}
        <StatsBar stats={stats} episode={episode} episodeReturn={episodeReturn} />
        <Transport
          playing={playing}
          onToggle={toggle}
          onStep={() => send({ cmd: "step" })}
          speed={speed}
          onSpeed={(v) => {
            setSpeed(v);
            send({ cmd: "speed", value: v });
          }}
          onRestart={() => {
            logRef.current = [];
            setFeed([]);
            send({ cmd: "restart" });
          }}
        />
        <div className="form-actions">
          <button className="btn" onClick={saveReplay} disabled={logRef.current.length < 2 || saved}>
            {saved ? "Replay saved" : "Save replay"}
          </button>
          <button className="btn" onClick={() => send({ cmd: "new_episode" })}>
            New episode
          </button>
        </div>
        <p className="hint">
          Replays are stored in your browser (localStorage) and can be
          re-watched without the server.
        </p>
      </div>
      <div className="panel feed-panel">
        <h3>Action feed</h3>
        <ul className="feed">
          {feed.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Replay watch (purely client-side playback of a saved event log)
// ---------------------------------------------------------------------------

function ReplayWatch({ replay }: { replay: Replay }) {
  const events = replay.events;
  const [pos, setPos] = useState(1); // events consumed
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    setPos(1);
    setPlaying(false);
  }, [replay.id]);

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => {
      setPos((p) => {
        if (p >= events.length) {
          setPlaying(false);
          return p;
        }
        return p + 1;
      });
    }, BASE_MS / speed);
    return () => clearInterval(t);
  }, [playing, speed, events.length]);

  const consumed = events.slice(0, pos);
  const view =
    [...consumed].reverse().find((e) => e.render) ?? events[0];
  const stats =
    [...consumed].reverse().find((e) => e.session_stats)?.session_stats ??
    null;
  const feed = consumed
    .filter((e) => e.type === "step" || e.type === "episode_end")
    .map((e) =>
      e.type === "step"
        ? feedLine(e)
        : `Ep ${e.episode} finished — return ${fmt(e.episode_return)}`,
    )
    .reverse()
    .slice(0, 60);

  return (
    <div className="watch-layout">
      <div className="panel">
        <div className="watch-head">
          <strong>Replay · {replay.name}</strong>
          <span className="hint">
            {new Date(replay.createdAt).toLocaleString()} · step {pos}/
            {events.length}
          </span>
        </div>
        {view?.render && <Board render={view.render} />}
        <StatsBar
          stats={stats}
          episode={view?.episode ?? 1}
          episodeReturn={view?.episode_return ?? 0}
        />
        <Transport
          playing={playing}
          onToggle={() => {
            if (pos >= events.length) setPos(1);
            setPlaying(!playing);
          }}
          onStep={() => setPos((p) => Math.min(events.length, p + 1))}
          speed={speed}
          onSpeed={setSpeed}
          onRestart={() => {
            setPlaying(false);
            setPos(1);
          }}
        />
      </div>
      <div className="panel feed-panel">
        <h3>Action feed</h3>
        <ul className="feed">
          {feed.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Watch view: picker (when no target) + dispatch
// ---------------------------------------------------------------------------

interface Props {
  target: WatchTarget | null;
  onWatch: (t: WatchTarget) => void;
}

export default function WatchView({ target, onWatch }: Props) {
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [games, setGames] = useState<GameInfo[]>([]);
  const [replays, setReplays] = useState<Replay[]>(store.replays.get());

  useEffect(() => {
    Promise.all([api.runs(), api.games()])
      .then(([r, g]) => {
        setRuns(r);
        setGames(g);
      })
      .catch(() => undefined);
    setReplays(store.replays.get());
  }, [target]);

  let body: React.ReactNode = null;
  if (target?.kind === "run" || target?.kind === "baseline") {
    body = <LiveWatch key={target.nonce} target={target} />;
  } else if (target?.kind === "replay") {
    const replay = replays.find((r) => r.id === target.replayId);
    body = replay ? (
      <ReplayWatch key={target.nonce} replay={replay} />
    ) : (
      <p className="hint">That replay no longer exists in this browser.</p>
    );
  }

  return (
    <div className="view">
      <h2>Watch an agent play</h2>
      {body}
      <div className="watch-picker">
        <div className="panel">
          <h3>Trained runs</h3>
          {runs.filter((r) => r.checkpoints.length > 0).length === 0 && (
            <p className="hint">No runs with checkpoints yet.</p>
          )}
          <ul className="picker-list">
            {runs
              .filter((r) => r.checkpoints.length > 0)
              .map((r) => (
                <li key={r.run_id}>
                  <span className="mono">{r.run_id}</span>
                  <button
                    className="btn small"
                    onClick={() =>
                      onWatch({
                        kind: "run",
                        runId: r.run_id,
                        checkpoint: r.checkpoints.includes("best_model.zip")
                          ? "best_model"
                          : "final_model",
                        nonce: Date.now(),
                      })
                    }
                  >
                    Watch
                  </button>
                </li>
              ))}
          </ul>
          <h3>Baselines</h3>
          <ul className="picker-list">
            {games.flatMap((g) =>
              g.baselines.map((b) => (
                <li key={`${g.name}-${b.name}`}>
                  <span>
                    {g.display_name} · {b.name}
                  </span>
                  <button
                    className="btn small"
                    onClick={() =>
                      onWatch({
                        kind: "baseline",
                        game: g.name,
                        agent: b.name,
                        nonce: Date.now(),
                      })
                    }
                  >
                    Watch
                  </button>
                </li>
              )),
            )}
          </ul>
        </div>
        <div className="panel">
          <h3>Saved replays (this browser)</h3>
          {replays.length === 0 && (
            <p className="hint">
              No replays saved yet — watch a live session and hit
              "Save replay".
            </p>
          )}
          <ul className="picker-list">
            {replays.map((r) => (
              <li key={r.id}>
                <span>
                  {r.name}{" "}
                  <span className="hint">
                    {new Date(r.createdAt).toLocaleString()}
                  </span>
                </span>
                <span>
                  <button
                    className="btn small"
                    onClick={() =>
                      onWatch({
                        kind: "replay",
                        replayId: r.id,
                        nonce: Date.now(),
                      })
                    }
                  >
                    Replay
                  </button>{" "}
                  <button
                    className="btn small danger"
                    onClick={() => {
                      store.replays.remove(r.id);
                      setReplays(store.replays.get());
                    }}
                  >
                    Delete
                  </button>
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
