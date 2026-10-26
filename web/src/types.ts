// Shared types mirroring the backend API payloads.

export interface GameAction {
  index: number;
  name: string;
}

export interface BaselineRef {
  name: string;
  mean_return: number | null;
  win_rate: number | null;
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
  eval_freq: Record<string, number>;
  eval_episodes: Record<string, number>;
}

/** Effective per-(game, algorithm) settings from the backend's
 * parity-tested defaults table (mirrors configs/*.yaml). */
export interface PairDefaults {
  gamma: number;
  policy: string;
  policy_kwargs: Record<string, unknown>;
  env_kwargs: Record<string, unknown>;
  n_envs: number;
  learning_rate: number;
  hyperparams: Record<string, unknown>;
  self_play: Record<string, unknown>;
}

export interface MetaInfo {
  algorithms: string[];
  game_algorithms: Record<string, string[]>;
  default_algorithms: Record<string, string>;
  pair_defaults: Record<string, PairDefaults>;
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

export interface Match3LastMove {
  from: [number, number];
  direction: number;
  to: [number, number];
  points: number;
  cascades: number;
}

export interface Match3Render {
  game: "match3";
  text?: string;
  board: number[][];
  gems: number;
  score: number;
  moves_left: number;
  moves_used: number;
  max_moves: number;
  last_move: Match3LastMove | null;
}

export interface FlappyPipe {
  x: number;
  width: number;
  gap_top: number;
  gap_bottom: number;
  scored: boolean;
}

export interface FlappyRender {
  game: "flappy";
  text?: string;
  bird_x: number;
  bird_y: number;
  bird_vy: number;
  bird_radius: number;
  pipes: FlappyPipe[];
  score: number;
  steps: number;
  width: number;
  height: number;
}

export interface RacingRender {
  game: "racing";
  text?: string;
  car: { x: number; y: number; heading: number; speed: number; steer: number };
  track: {
    centerline: [number, number][];
    left_edge: [number, number][];
    right_edge: [number, number][];
    total_length: number;
  };
  rays: number[];
  ray_angles: number[];
  ray_max_range: number;
  progress_fraction: number;
  progress_m: number;
  laps: number;
  current_lap_time: number | null;
  last_lap_time: number | null;
  best_lap_time: number | null;
  speed: number;
  offset: number;
  off_track: boolean;
  steps: number;
}

export interface ConnectFourRender {
  game: "connect_four";
  text?: string;
  board: number[][]; // 0 empty, 1 = player 0 (red), 2 = player 1 (yellow)
  rows: number;
  cols: number;
  current_player: number;
  winner: number | null;
  last_move: number | null;
  moves_played: number;
  game_over: boolean;
}

export interface PongRender {
  game: "pong";
  text?: string;
  width: number;
  height: number;
  ball: { x: number; y: number; radius: number };
  paddles: { x: number; y: number; height: number; width: number; inventory: string; effects: number[] }[];
  pickup: { kind: string; x: number; y: number; radius: number; remaining: number } | null;
  scores: number[];
  remaining: number;
  serve_remaining: number;
  winner: number | null;
  game_over: boolean;
  end_reason: string | null;
}

export interface Game2048Render {
  game: "game2048";
  text?: string;
  board: number[][]; // tile exponents per cell: 0 = empty, k = tile 2**k
  score: number;
  max_tile: number;
  moves: number;
  game_over: boolean;
  spawned: number[] | null; // [row, col, exponent] of the latest spawn
  last_move: { action: number; points: number } | null;
}

export type RenderState =
  | PongRender
  | BlackjackRender
  | HolRender
  | Match3Render
  | FlappyRender
  | RacingRender
  | ConnectFourRender
  | Game2048Render
  | { game: string; text?: string };

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
  opponent?: string | null;
  episode: number;
  step_in_episode?: number;
  action?: number | number[];
  action_name?: string;
  joint_actions?: [number, number];
  tick_seconds?: number;
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

/** What the Watch view should show. ``opponent`` is a baseline name
 * for seat 1 of a two-player game; omitted = mirror match. */
export type WatchTarget =
  | { kind: "run"; runId: string; checkpoint: string; opponent?: string; nonce: number }
  | { kind: "baseline"; game: string; agent: string; opponent?: string; nonce: number }
  | { kind: "replay"; replayId: string; nonce: number };
