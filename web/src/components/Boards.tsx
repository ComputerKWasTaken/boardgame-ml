// Game boards: render-state -> visual board, per game.

import type {
  BlackjackRender,
  FlappyRender,
  HolRender,
  Match3Render,
  RacingRender,
  RenderState,
} from "../types";

function cardLabel(v: number): string {
  return v === 1 ? "A" : String(v);
}

function CardTile({ value, faceDown }: { value?: number; faceDown?: boolean }) {
  if (faceDown) {
    return <div className="card-tile face-down">?</div>;
  }
  return <div className="card-tile">{value !== undefined ? cardLabel(value) : ""}</div>;
}

export function BlackjackBoard({ render }: { render: BlackjackRender }) {
  const dealerTiles = render.dealer_revealed
    ? render.dealer_hand
    : [render.dealer_upcard];
  return (
    <div className="board blackjack-board">
      <div className="hand-row">
        <div className="hand-label">
          Dealer
          {render.dealer_revealed && render.dealer_total !== null && (
            <span className="hand-total"> = {render.dealer_total}</span>
          )}
        </div>
        <div className="hand-cards">
          {dealerTiles.map((v, i) => (
            <CardTile key={i} value={v} />
          ))}
          {!render.dealer_revealed && <CardTile faceDown />}
        </div>
      </div>
      <div className="hand-row">
        <div className="hand-label">
          Agent
          <span className="hand-total">
            {" "}
            = {render.player_total}
            {render.player_soft ? " (soft)" : ""}
          </span>
          {render.is_natural && <span className="badge">natural</span>}
        </div>
        <div className="hand-cards">
          {render.player_hand.map((v, i) => (
            <CardTile key={i} value={v} />
          ))}
        </div>
      </div>
      {render.outcome && (
        <div className="board-outcome">Outcome: {render.outcome}</div>
      )}
    </div>
  );
}

export function HigherOrLowerBoard({ render }: { render: HolRender }) {
  return (
    <div className="board hol-board">
      <div className="hol-main">
        <div className="hol-card">{cardLabel(render.current_card)}</div>
        <div className="hol-meta">
          <div>
            Score <strong>{render.score > 0 ? `+${render.score}` : render.score}</strong>
          </div>
          <div>
            Turns left <strong>{render.turns_left}</strong> / {render.max_turns}
          </div>
        </div>
      </div>
      {render.history.length > 0 && (
        <div className="hol-history">
          {render.history.map((h, i) => (
            <span key={i} className="hol-chip" title={`guessed ${h.action_name}`}>
              {h.card !== null ? cardLabel(h.card) : "?"} {h.action === 0 ? "↑" : h.action === 1 ? "↓" : "■"}{" "}
              {h.next_card !== null ? cardLabel(h.next_card) : "?"}
              <em className={h.reward > 0 ? "pos" : h.reward < 0 ? "neg" : ""}>
                {h.reward > 0 ? "+1" : h.reward < 0 ? "−1" : "0"}
              </em>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export const GEM_COLORS = [
  "#e5534b",
  "#f0883e",
  "#e3b341",
  "#3fb950",
  "#58a6ff",
  "#bc8cff",
  "#f778ba",
  "#39c5cf",
];

export function gemLetter(v: number): string {
  return String.fromCharCode(65 + v); // A, B, C, ...
}

export function Match3Board({ render }: { render: Match3Render }) {
  const rows = render.board.length;
  const cols = rows > 0 ? render.board[0].length : 0;
  const swapped = new Set<string>();
  if (render.last_move) {
    swapped.add(render.last_move.from.join(","));
    swapped.add(render.last_move.to.join(","));
  }
  return (
    <div className="board match3-board">
      <div className="match3-header">
        <div>
          Score <strong>{render.score}</strong>
        </div>
        <div>
          Moves left <strong>{render.moves_left}</strong> / {render.max_moves}
        </div>
        {render.last_move && render.last_move.cascades > 1 && (
          <div className="badge">cascade ×{render.last_move.cascades}</div>
        )}
      </div>
      <div
        className="match3-grid"
        style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
      >
        {render.board.map((row, r) =>
          row.map((v, c) => (
            <div
              key={`${r}-${c}`}
              className={`match3-tile${swapped.has(`${r},${c}`) ? " swapped" : ""}`}
              style={{ background: GEM_COLORS[v % GEM_COLORS.length] }}
              title={`gem ${gemLetter(v)}`}
            >
              {gemLetter(v)}
            </div>
          )),
        )}
      </div>
      <div className="match3-legend">
        {Array.from({ length: render.gems }, (_, v) => (
          <span key={v} className="legend-chip">
            <span
              className="legend-swatch"
              style={{ background: GEM_COLORS[v % GEM_COLORS.length] }}
            />
            {gemLetter(v)}
          </span>
        ))}
      </div>
    </div>
  );
}

export function FlappyBoard({ render }: { render: FlappyRender }) {
  const { width, height } = render;
  // World coords have y up from the ground; SVG has y down. Flip the
  // whole scene with one transform and draw in world coordinates.
  return (
    <div className="board flappy-board">
      <div className="match3-header">
        <div>
          Score <strong>{render.score}</strong>
        </div>
        <div>
          Steps <strong>{render.steps}</strong>
        </div>
        <div>
          vy <strong>{render.bird_vy > 0 ? `+${Math.round(render.bird_vy)}` : Math.round(render.bird_vy)}</strong>
        </div>
      </div>
      <svg
        className="flappy-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Flappy Bird scene"
      >
        <rect x={0} y={0} width={width} height={height} className="flappy-sky" />
        <g transform={`translate(0 ${height}) scale(1 -1)`}>
          {render.pipes.map((p, i) => (
            <g key={i}>
              <rect
                x={p.x}
                y={0}
                width={p.width}
                height={p.gap_bottom}
                className="flappy-pipe"
              />
              <rect
                x={p.x}
                y={p.gap_top}
                width={p.width}
                height={height - p.gap_top}
                className="flappy-pipe"
              />
            </g>
          ))}
          <line x1={0} y1={0} x2={width} y2={0} className="flappy-ground" />
          <circle
            cx={render.bird_x}
            cy={render.bird_y}
            r={render.bird_radius}
            className="flappy-bird"
          />
        </g>
      </svg>
    </div>
  );
}

export function RacingBoard({ render }: { render: RacingRender }) {
  const { track, car } = render;
  // Bounds from the track geometry (world coords, y up). The whole
  // scene is drawn in world coordinates inside one flipped group,
  // like Flappy's board: SVG's y-down becomes the world's y-up.
  const all = [...track.left_edge, ...track.right_edge];
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const pad = 6;
  const minX = Math.min(...xs) - pad;
  const maxX = Math.max(...xs) + pad;
  const minY = Math.min(...ys) - pad;
  const maxY = Math.max(...ys) + pad;
  const ribbon =
    track.left_edge.map((p) => p.join(",")).join(" ") +
    " " +
    [...track.right_edge].reverse().map((p) => p.join(",")).join(" ");
  const centre = track.centerline.map((p) => p.join(",")).join(" ");
  const headingDeg = (car.heading * 180) / Math.PI;
  const kmh = Math.round(car.speed * 3.6);
  const lapTime = (v: number | null) => (v === null ? "—" : `${v.toFixed(1)}s`);
  return (
    <div className="board racing-board">
      <div className="match3-header">
        <div>
          Lap <strong>{render.laps + 1}</strong>
        </div>
        <div>
          Progress <strong>{Math.round(render.progress_fraction * 100)}%</strong>
        </div>
        <div>
          Speed <strong>{kmh} km/h</strong>
        </div>
        <div>
          Best lap <strong>{lapTime(render.best_lap_time)}</strong>
        </div>
        {render.off_track && <div className="badge">off track</div>}
      </div>
      <svg
        className="racing-svg"
        viewBox={`${minX} ${minY} ${maxX - minX} ${maxY - minY}`}
        role="img"
        aria-label="Racing track, top-down"
      >
        <g transform={`translate(0 ${minY + maxY}) scale(1 -1)`}>
          <polygon points={ribbon} className="racing-track" />
          <polyline points={centre} className="racing-center" />
          {render.rays.map((dist, i) => {
            const a = car.heading + render.ray_angles[i];
            return (
              <line
                key={i}
                x1={car.x}
                y1={car.y}
                x2={car.x + Math.cos(a) * dist}
                y2={car.y + Math.sin(a) * dist}
                className="racing-ray"
              />
            );
          })}
          <g transform={`translate(${car.x} ${car.y}) rotate(${headingDeg})`}>
            <rect x={-2.7} y={-1.15} width={5.4} height={2.3} rx={0.7} className="racing-car" />
            <rect x={0.4} y={-0.85} width={1.5} height={1.7} rx={0.4} className="racing-car-glass" />
            <line
              x1={0.2}
              y1={0}
              x2={0.2 + Math.cos(car.steer * 0.55) * 2.3}
              y2={Math.sin(car.steer * 0.55) * 2.3}
              className="racing-steer-line"
            />
          </g>
        </g>
      </svg>
    </div>
  );
}

export default function Board({ render }: { render: RenderState }) {
  if (render.game === "blackjack") {
    return <BlackjackBoard render={render as BlackjackRender} />;
  }
  if (render.game === "higher_or_lower") {
    return <HigherOrLowerBoard render={render as HolRender} />;
  }
  if (render.game === "match3") {
    return <Match3Board render={render as Match3Render} />;
  }
  if (render.game === "flappy") {
    return <FlappyBoard render={render as FlappyRender} />;
  }
  if (render.game === "racing") {
    return <RacingBoard render={render as RacingRender} />;
  }
  return (
    <div className="board">
      <pre>{render.text ?? JSON.stringify(render, null, 2)}</pre>
    </div>
  );
}
