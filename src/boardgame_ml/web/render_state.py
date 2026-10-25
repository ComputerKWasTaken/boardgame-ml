"""Render state: the JSON the web frontend draws from.

The watch API never ships raw observations to the browser; it ships a
``render`` object per game — the same information a human player would
see at the table. The shape is deliberately generic::

    {"game": <name>, "text": <the env's text render>, ...game fields...}

so a new game (Match 3 added a ``board`` grid field this way) plugs
in by adding one builder here, with no protocol change. Game-specific
builders read the env's real state (hands, current card) rather than
decoding the normalized observation — the observation is the
*agent's* interface, not the viewer's.
"""

from __future__ import annotations

from typing import Any

from boardgame_ml.envs import blackjack as bj
from boardgame_ml.envs import higher_or_lower as hol


def _blackjack(env, info: dict, reveal: bool) -> dict:
    total, usable = bj.hand_value(env.player_hand)
    dealer_total = None
    dealer_hand = [env.dealer_hand[0]]  # only the upcard, while playing
    if reveal:
        dealer_hand = list(env.dealer_hand)
        dealer_total, _ = bj.hand_value(env.dealer_hand)
    return {
        "player_hand": list(env.player_hand),
        "player_total": total,
        "player_soft": bool(usable),
        "dealer_upcard": env.dealer_hand[0],
        "dealer_hand": dealer_hand,
        "dealer_total": dealer_total,
        "dealer_revealed": bool(reveal),
        "is_natural": bool(info.get("is_natural", False)),
        "outcome": info.get("outcome"),
    }


def _higher_or_lower(env, info: dict, reveal: bool) -> dict:
    return {
        "current_card": env._card,
        "score": env._score,
        "turns_used": env._turns_used,
        "turns_left": hol.MAX_TURNS - env._turns_used,
        "max_turns": hol.MAX_TURNS,
    }


def _match3(env, info: dict, reveal: bool) -> dict:
    return {
        "board": [[int(v) for v in row] for row in env.board],
        "gems": env.num_gems,
        "score": env.score,
        "moves_left": env.moves_left,
        "moves_used": env.moves_used,
        "max_moves": env.max_moves,
        "last_move": info.get("last_move"),
    }


_BUILDERS = {
    "blackjack": _blackjack,
    "higher_or_lower": _higher_or_lower,
    "match3": _match3,
}


def build_render(
    game: str,
    env,
    info: dict | None = None,
    *,
    reveal: bool = False,
) -> dict:
    """Build the render object for ``env``'s current state.

    ``reveal`` exposes hidden information (the blackjack dealer's hole
    card) — used for the final state of a finished episode. Games
    without a registered builder get a generic payload (text + raw
    observation), so the protocol degrades gracefully.
    """
    info = info or {}
    render: dict[str, Any] = {"game": game}
    text = env.render()
    if text is not None:
        render["text"] = text
    builder = _BUILDERS.get(game)
    if builder is not None:
        render.update(builder(env.unwrapped, info, reveal))
    return render
