import { useEffect, useRef, useState } from "react";
import { api, trainEventsURL } from "../api";
import Chart from "../components/Chart";
import { store } from "../storage";
import type { JobSnapshot, MetaInfo, WatchTarget } from "../types";

interface Props {
  onWatch: (t: WatchTarget) => void;
  onRunFinished: () => void;
}

export default function TrainView({ onWatch, onRunFinished }: Props) {
  const [meta, setMeta] = useState<MetaInfo | null>(null);
  const [settings, setSettings] = useState(store.trainSettings.get());
  const [job, setJob] = useState<JobSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setError(String(e)));
    return () => esRef.current?.close();
  }, []);

  useEffect(() => {
    store.trainSettings.set(settings);
  }, [settings]);

  const set = (patch: Partial<typeof settings>) =>
    setSettings((s) => ({ ...s, ...patch }));

  const presetTimesteps =
    meta && settings.game
      ? meta.presets[settings.preset]?.timesteps[settings.game]
      : undefined;

  const start = async () => {
    setError(null);
    setJob(null);
    try {
      const body: Record<string, unknown> = {
        game: settings.game,
        algorithm: settings.algorithm,
        preset: settings.preset,
        seed: settings.seed,
      };
      if (settings.timesteps.trim() !== "")
        body.timesteps = Number(settings.timesteps);
      if (settings.learningRate.trim() !== "")
        body.learning_rate = Number(settings.learningRate);
      const { job_id } = await api.startTrain(body);
      const es = new EventSource(trainEventsURL(job_id));
      esRef.current = es;
      es.onmessage = (ev) => {
        const snap = JSON.parse(ev.data) as JobSnapshot;
        setJob(snap);
        if (snap.status === "completed" || snap.status === "failed") {
          es.close();
          if (snap.status === "completed") onRunFinished();
        }
      };
      es.onerror = () => {
        // Stream ended or errored; fall back to a final status fetch.
        api
          .trainStatus(job_id)
          .then((snap) => {
            setJob(snap);
            if (snap.status === "completed") onRunFinished();
          })
          .catch(() => undefined)
          .finally(() => es.close());
      };
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const stop = async () => {
    if (!job) return;
    try {
      await api.stopTrain(job.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const running = job?.status === "running" || job?.status === "queued";
  const progress =
    job && job.total_timesteps > 0
      ? Math.min(100, (job.timesteps_done / job.total_timesteps) * 100)
      : 0;

  return (
    <div className="view">
      <h2>Train an agent</h2>
      {error && <div className="error-box">{error}</div>}

      <div className="train-form panel">
        <label>
          Game
          <select
            value={settings.game}
            onChange={(e) => set({ game: e.target.value })}
            disabled={running}
          >
            <option value="blackjack">Blackjack</option>
            <option value="higher_or_lower">Higher or Lower</option>
            <option value="racing">Racing</option>
          </select>
        </label>
        <label>
          Algorithm
          <select
            value={settings.algorithm}
            onChange={(e) => set({ algorithm: e.target.value })}
            disabled={running}
          >
            {(meta?.algorithms ?? ["maskable_ppo"]).map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>
        <label>
          Preset
          <select
            value={settings.preset}
            onChange={(e) => set({ preset: e.target.value })}
            disabled={running}
          >
            {Object.keys(meta?.presets ?? { standard: 1 }).map((p) => (
              <option key={p} value={p}>
                {p}
                {meta?.presets[p]?.timesteps[settings.game]
                  ? ` (${meta.presets[p].timesteps[settings.game].toLocaleString()} steps)`
                  : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          Timesteps (override)
          <input
            type="number"
            min={1}
            placeholder={
              presetTimesteps ? String(presetTimesteps) : "preset default"
            }
            value={settings.timesteps}
            onChange={(e) => set({ timesteps: e.target.value })}
            disabled={running}
          />
        </label>
        <label>
          Learning rate (override)
          <input
            type="text"
            placeholder="framework default"
            value={settings.learningRate}
            onChange={(e) => set({ learningRate: e.target.value })}
            disabled={running}
          />
        </label>
        <label>
          Seed
          <input
            type="number"
            value={settings.seed}
            onChange={(e) => set({ seed: Number(e.target.value) })}
            disabled={running}
          />
        </label>
        <div className="form-actions">
          {!running ? (
            <button className="btn primary" onClick={start}>
              Start training
            </button>
          ) : (
            <button className="btn danger" onClick={stop}>
              Stop (keeps best checkpoint)
            </button>
          )}
        </div>
        <p className="hint">
          Mask-blind algorithms (ppo, dqn) are refused for games that mask
          actions — the framework's mask guard applies here too.
        </p>
      </div>

      {job && (
        <div className="panel">
          <div className="job-head">
            <strong>
              {job.game} · {job.algorithm}
            </strong>
            <span className={`status status-${job.status}`}>
              {job.status}
              {job.stopped ? " (stopped early)" : ""}
            </span>
            <span className="mono">{job.run_id}</span>
          </div>
          <div className="progress-bar">
            <div style={{ width: `${progress}%` }} />
          </div>
          <div className="job-stats">
            <span>
              {job.timesteps_done.toLocaleString()} /{" "}
              {job.total_timesteps.toLocaleString()} steps
            </span>
            <span>{job.episodes_completed.toLocaleString()} episodes</span>
            <span>
              latest eval{" "}
              {job.latest_eval_mean === null
                ? "—"
                : job.latest_eval_mean.toFixed(3)}
            </span>
            <span>
              best eval{" "}
              {job.best_eval_mean === null
                ? "—"
                : job.best_eval_mean.toFixed(3)}
            </span>
          </div>
          <Chart returns={job.recent_returns} evals={job.eval_history} />
          <div className="chart-legend">
            <span className="legend-line">episode return (moving avg)</span>
            <span className="legend-eval">● eval mean (deterministic)</span>
          </div>
          {job.error && <div className="error-box">{job.error}</div>}
          {job.status === "completed" && (
            <div className="form-actions">
              <button
                className="btn primary"
                onClick={() =>
                  onWatch({
                    kind: "run",
                    runId: job.run_id,
                    checkpoint: "best_model",
                    nonce: Date.now(),
                  })
                }
              >
                Watch this run
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
