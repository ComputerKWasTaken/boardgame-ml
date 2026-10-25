import type { GameInfo, JobSnapshot, MetaInfo, RunEntry } from "./types";

async function getJSON<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
  return resp.json() as Promise<T>;
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      detail = (await resp.json()).detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return resp.json() as Promise<T>;
}

export const api = {
  games: () => getJSON<GameInfo[]>("/api/games"),
  meta: () => getJSON<MetaInfo>("/api/meta"),
  runs: () => getJSON<RunEntry[]>("/api/runs"),
  run: (runId: string) => getJSON<RunEntry>(`/api/runs/${runId}`),
  startTrain: (body: Record<string, unknown>) =>
    postJSON<{ job_id: string; run_id: string }>("/api/train", body),
  trainStatus: (jobId: string) =>
    getJSON<JobSnapshot>(`/api/train/${jobId}`),
  stopTrain: (jobId: string) =>
    postJSON<{ job_id: string; status: string }>(
      `/api/train/${jobId}/stop`,
      {},
    ),
  startWatch: (body: Record<string, unknown>) =>
    postJSON<{
      session_id: string;
      game: string;
      agent: string;
      opponent: string | null;
      initial: import("./types").WatchEvent;
    }>("/api/watch", body),
};

export function watchSocketURL(sessionId: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/api/watch/${sessionId}`;
}

export function trainEventsURL(jobId: string): string {
  return `/api/train/${jobId}/events`;
}
