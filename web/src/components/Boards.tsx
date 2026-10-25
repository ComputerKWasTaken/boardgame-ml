// Game boards: render-state -> visual board, per game.

import type { BlackjackRender, HolRender, RenderState } from "../types";

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

export default function Board({ render }: { render: RenderState }) {
  if (render.game === "blackjack") {
    return <BlackjackBoard render={render as BlackjackRender} />;
  }
  if (render.game === "higher_or_lower") {
    return <HigherOrLowerBoard render={render as HolRender} />;
  }
  return (
    <div className="board">
      <pre>{render.text ?? JSON.stringify(render, null, 2)}</pre>
    </div>
  );
}
