import { useEffect, useState } from "react";
import { api } from "../api";
import { store } from "../storage";
import type { GameInfo, RunEntry, WatchTarget } from "../types";

interface Props {
  onWatch: (t: WatchTarget) => void;
  refreshKey: number;
}

function fmtReturn(v: number | null): string {
  return v === null ? "—" : (v >= 0 ? `+${v.toFixed(3)}` : v.toFixed(3));
}

function fmtDate(epochSec: number): string {
  return new Date(epochSec * 1000).toLocaleString();
}

function bestCheckpoint(run: RunEntry): string | null {
  if (run.checkpoints.includes("best_model.zip")) return "best_model";
  if (run.checkpoints.includes("final_model.zip")) return "final_model";
  const first = run.checkpoints[0];
  return first ? first.replace(/\.zip$/, "") : null;
}

export default function GamesRuns({ onWatch, refreshKey }: Props) {
  const [games, setGames] = useState<GameInfo[]>([]);
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [starred, setStarred] = useState<string[]>(store.starredRuns.get());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    Promise.all([api.games(), api.runs()])
      .then(([g, r]) => {
        if (!live) return;
        setGames(g);
        setRuns(r);
      })
      .catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [refreshKey]);

  const toggleStar = (runId: string) => {
    const next = starred.includes(runId)
      ? starred.filter((s) => s !== runId)
      : [...starred, runId];
    setStarred(next);
    store.starredRuns.set(next);
  };

  return (
    <div className="view">
      {error && <div className="error-box">{error}</div>}

      <h2>Games</h2>
      <div className="game-cards">
        {games.map((g) => (
          <div key={g.name} className="game-card">
            <h3>{g.display_name}</h3>
            <p className="blurb">{g.blurb}</p>
            <p className="obs-line">Observation: {g.observation}</p>
            <p className="obs-line">
              Actions: {g.actions.map((a) => a.name).join(" · ")}
            </p>
            <table className="mini-table">
              <thead>
                <tr>
                  <th>Baseline</th>
                  <th>Mean return</th>
                  <th>Win rate</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {g.baselines.map((b) => (
                  <tr key={b.name}>
                    <td>{b.name}</td>
                    <td>{fmtReturn(b.mean_return)}</td>
                    <td>{(b.win_rate * 100).toFixed(1)}%</td>
                    <td>
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
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="hint">
              Baselines are the proof-of-learning yardsticks: beat random,
              approach the heuristic.
            </p>
          </div>
        ))}
      </div>

      <h2>Runs</h2>
      {runs.length === 0 && (
        <p className="hint">No runs yet — train one in the Train view.</p>
      )}
      {runs.length > 0 && (
        <table className="runs-table">
          <thead>
            <tr>
              <th></th>
              <th>Run</th>
              <th>Game</th>
              <th>Algorithm</th>
              <th>Best eval return</th>
              <th>Seed</th>
              <th>Date</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const ckpt = bestCheckpoint(r);
              return (
                <tr key={r.run_id} className={r.complete ? "" : "incomplete"}>
                  <td>
                    <button
                      className="star-btn"
                      title="Star run"
                      onClick={() => toggleStar(r.run_id)}
                    >
                      {starred.includes(r.run_id) ? "★" : "☆"}
                    </button>
                  </td>
                  <td className="mono">{r.run_id}</td>
                  <td>{r.game ?? "?"}</td>
                  <td>{r.algorithm ?? "?"}</td>
                  <td>{fmtReturn(r.best_eval_mean_return)}</td>
                  <td>{r.seed ?? "—"}</td>
                  <td>{fmtDate(r.created_at)}</td>
                  <td>
                    <button
                      className="btn small"
                      disabled={ckpt === null}
                      title={
                        ckpt === null
                          ? "No checkpoint in this run"
                          : `Watch ${ckpt}`
                      }
                      onClick={() =>
                        ckpt &&
                        onWatch({
                          kind: "run",
                          runId: r.run_id,
                          checkpoint: ckpt,
                          nonce: Date.now(),
                        })
                      }
                    >
                      Watch
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
