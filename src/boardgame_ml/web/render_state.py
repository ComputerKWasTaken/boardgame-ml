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
from boardgame_ml.envs import connect_four as c4
from boardgame_ml.envs import flappy as fb
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


def _flappy(env, info: dict, reveal: bool) -> dict:
    # World coordinates as the env keeps them (y up from the ground);
    # the frontend's SVG flips the axis when drawing. The bird's x is
    # fixed — the world scrolls past it — so pipes carry positions.
    return {
        "bird_x": fb.BIRD_X,
        "bird_y": env.bird_y,
        "bird_vy": env.bird_vy,
        "bird_radius": fb.BIRD_RADIUS,
        "pipes": [
            {
                "x": p.x,
                "width": fb.PIPE_WIDTH,
                "gap_top": p.gap_top,
                "gap_bottom": p.gap_bottom,
                "scored": p.scored,
            }
            for p in env.pipes
        ],
        "score": env.score,
        "steps": env.steps,
        "width": fb.WORLD_WIDTH,
        "height": fb.WORLD_HEIGHT,
    }


def _racing(env, info: dict, reveal: bool) -> dict:
    # World coordinates (metres, y up); the frontend's SVG flips the
    # axis when drawing, like Flappy's. The track polylines ride
    # along on every event — one canonical track, ~200 points a side,
    # and the replay log stays self-contained this way.
    track = env.track

    def pts(arr) -> list[list[float]]:
        return [[float(x), float(y)] for x, y in arr]

    return {
        "car": {
            "x": env.car_x,
            "y": env.car_y,
            "heading": env.heading,
            "speed": env.speed,
            "steer": env.steer,
        },
        "track": {
            "centerline": pts(track.centerline),
            "left_edge": pts(track.left_edge),
            "right_edge": pts(track.right_edge),
            "total_length": track.total_length,
        },
        "rays": [float(v) for v in env.last_rays],
        "ray_angles": [float(a) for a in env.ray_angles],
        "ray_max_range": env.ray_max_range,
        "progress_fraction": info.get("progress_fraction", 0.0),
        "progress_m": info.get("progress_m", env.progress_m),
        "laps": info.get("laps", env.laps),
        "current_lap_time": info.get("current_lap_time"),
        "last_lap_time": info.get("last_lap_time"),
        "best_lap_time": info.get("best_lap_time"),
        "speed": env.speed,
        "offset": info.get("offset", env.offset),
        "off_track": bool(info.get("off_track", False)),
        "steps": env.steps,
    }


def _connect_four(env, info: dict, reveal: bool) -> dict:
    # Absolute seats (the viewer's board, not a player's obs):
    # 0 empty, 1 = player 0 (red), 2 = player 1 (yellow), top row
    # first, exactly as the env stores it.
    return {
        "board": [[int(v) for v in row] for row in env.board],
        "rows": c4.ROWS,
        "cols": c4.COLS,
        "current_player": info.get("current_player", env.current_player),
        "winner": info.get("winner", env.winner),
        "last_move": env.last_move,
        "moves_played": env.moves_played,
        "game_over": bool(
            env.winner is not None or env.moves_played >= c4.ROWS * c4.COLS
        ),
    }


def _pong(env, info: dict, reveal: bool) -> dict:
    from boardgame_ml.envs.pong import POWER_NAMES
    return {
        "width": env.width, "height": env.height,
        "ball": {"x": env.ball_x, "y": env.ball_y, "radius": env.ball_radius},
        "paddles": [{"x": env.paddle_inset if i == 0 else env.width - env.paddle_inset,
                     "y": env.paddle_y[i], "height": env.paddle_size(i), "width": env.paddle_width,
                     "inventory": POWER_NAMES[env.inventory[i]], "effects": list(env.effects[i])} for i in (0, 1)],
        "pickup": ({"kind": POWER_NAMES[env.pickup[0]], "x": env.pickup[1], "y": env.pickup[2],
                    "radius": env.pickup_radius, "remaining": env.pickup[3]} if env.pickup else None),
        "scores": list(env.scores), "remaining": max(0.0, env.match_seconds - env.elapsed),
        "serve_remaining": env.serve_remaining, "winner": env.winner,
        "game_over": env.done, "end_reason": env.end_reason,
    }


_BUILDERS = {
    "pong": _pong,
    "blackjack": _blackjack,
    "higher_or_lower": _higher_or_lower,
    "match3": _match3,
    "flappy": _flappy,
    "racing": _racing,
    "connect_four": _connect_four,
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
