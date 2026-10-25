// The personal layer in browser localStorage.
//
// Runs and checkpoints live on disk in runs/ (the source of truth,
// served by the API). Only personal, per-browser state lives here:
// UI preferences, last-used train settings, starred runs, saved
// replays, and watch history.

import type { Replay } from "./types";

const PREFIX = "bgml.";

function getJSON<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw === null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

function setJSON(key: string, value: unknown): void {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    /* storage full/unavailable: personal layer is best-effort */
  }
}

export interface Prefs {
  lastView: string;
}

export interface TrainSettings {
  game: string;
  algorithm: string;
  preset: string;
  timesteps: string; // "" = preset default
  learningRate: string; // "" = framework default
  seed: number;
}

export interface HistoryEntry {
  label: string;
  game: string;
  at: number;
}

export const store = {
  prefs: {
    get: (): Prefs => getJSON<Prefs>("prefs", { lastView: "games" }),
    set: (p: Prefs) => setJSON("prefs", p),
  },
  trainSettings: {
    get: (): TrainSettings =>
      getJSON<TrainSettings>("trainSettings", {
        game: "blackjack",
        algorithm: "maskable_ppo",
        preset: "standard",
        timesteps: "",
        learningRate: "",
        seed: 0,
      }),
    set: (s: TrainSettings) => setJSON("trainSettings", s),
  },
  starredRuns: {
    get: (): string[] => getJSON<string[]>("starredRuns", []),
    set: (ids: string[]) => setJSON("starredRuns", ids),
  },
  replays: {
    get: (): Replay[] => getJSON<Replay[]>("replays", []),
    add: (r: Replay) => setJSON("replays", [r, ...store.replays.get()]),
    remove: (id: string) =>
      setJSON(
        "replays",
        store.replays.get().filter((r) => r.id !== id),
      ),
  },
  history: {
    get: (): HistoryEntry[] => getJSON<HistoryEntry[]>("watchHistory", []),
    add: (e: HistoryEntry) =>
      setJSON("watchHistory", [e, ...store.history.get()].slice(0, 25)),
  },
};
