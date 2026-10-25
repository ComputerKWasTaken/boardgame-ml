// Shared types mirroring the backend API payloads.

export interface GameAction {
  index: number;
  name: string;
}

export interface BaselineRef {
  name: string;
  mean_return: number;
  win_rate: number;
  episodes: number;
}

export interface GameInfo {
  name: string;
  display_name: string;
  observation: string;
  blurb: string;
  actions: GameAction[];
  baselines: BaselineRef[];
}

export interface RunEntry {
  run_id: string;
  game: string | null;
  algorithm: string | null;
  seed: number | null;
  config_hash: string | null;
  best_eval_mean_return: number | null;
  train_seconds: number | null;
  versions: Record<string, string> | null;
  config: Record<string, unknown> | null;
  checkpoints: string[];
  complete: boolean;
  created_at: number; // epoch seconds (dir mtime)
  files?: { name: string; size: number }[];
  train_log_rows?: number;
}

export interface PresetInfo {
  timesteps: Record<string, number>;
  eval_freq: number;
  eval_episodes: number;
}

export interface MetaInfo {
  algorithms: string[];
  presets: Record<string, PresetInfo>;
  baselines: string[];
}

export interface JobSnapshot {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  stopped: boolean;
  game: string;
  algorithm: string;
  config: Record<string, unknown>;
  run_id: string;
  timesteps_done: number;
  total_timesteps: number;
  episodes_completed: number;
  recent_returns: { timesteps: number; episode_return: number }[];
  eval_history: { timesteps: number; mean_return: number }[];
  latest_eval_mean: number | null;
  best_eval_mean: number | null;
  error: string | null;
  summary: Record<string, unknown> | null;
}

// ---- watch protocol ----

export interface BlackjackRender {
  game: "blackjack";
  text?: string;
  player_hand: number[];
  player_total: number;
  player_soft: boolean;
  dealer_upcard: number;
  dealer_hand: number[];
  dealer_total: number | null;
  dealer_revealed: boolean;
  is_natural: boolean;
  outcome: string | null;
}

export interface HolGuess {
  card: number | null;
  action: number;
  action_name: string;
  next_card: number | null;
  reward: number;
}

export interface HolRender {
  game: "higher_or_lower";
  text?: string;
  current_card: number;
  score: number;
  turns_used: number;
  turns_left: number;
  max_turns: number;
  history: HolGuess[];
}

export type RenderState = BlackjackRender | HolRender | { game: string; text?: string };

export interface SessionStats {
  episodes_completed: number;
  wins: number;
  losses: number;
  pushes: number;
  total_return: number;
  mean_return: number;
}

export interface WatchEvent {
  type: "state" | "step" | "episode_end";
  session_id?: string;
  game: string;
  agent?: string;
  episode: number;
  step_in_episode?: number;
  action?: number;
  action_name?: string;
  reward?: number;
  episode_return: number;
  steps?: number;
  done?: boolean;
  render?: RenderState;
  session_stats: SessionStats;
  at?: number;
}

export interface Replay {
  id: string;
  name: string;
  game: string;
  agent: string;
  createdAt: number;
  events: WatchEvent[];
}

/** What the Watch view should show. */
export type WatchTarget =
  | { kind: "run"; runId: string; checkpoint: string; nonce: number }
  | { kind: "baseline"; game: string; agent: string; nonce: number }
  | { kind: "replay"; replayId: string; nonce: number };
