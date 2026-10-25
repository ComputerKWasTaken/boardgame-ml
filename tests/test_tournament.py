"""Tournament harness tests (M6) on tiny game counts.

The harness's bookkeeping is what's under test: seat symmetry (a
pairing plays both chairs), W/D/L accounting, and that resolved
contestants are driven through the same predict_action path as
evaluation. The yardstick ladder itself (greedy >> random,
minimax >> greedy) is asserted at smoke scale here and measured at
full scale in the M6 tournament run.
"""

from __future__ import annotations

from boardgame_ml.agents.baselines import get_baseline
from boardgame_ml.eval.tournament import (
    MatchRecord,
    play_match,
    play_pairing,
    run_tournament,
)


def test_match_record_rates():
    rec = MatchRecord(wins=3, draws=1, losses=1)
    assert rec.games == 5
    assert rec.win_rate == 0.6
    assert rec.score_rate == 0.7


def test_play_match_reports_the_winner_seat():
    greedy = get_baseline("connect_four_greedy", "connect_four")
    random = get_baseline("random", "connect_four")
    assert play_match("connect_four", greedy, random, seed=5) == 0
    assert play_match("connect_four", random, greedy, seed=5) == 1


def test_pairing_plays_both_seats_and_accounts_all_games():
    greedy = get_baseline("connect_four_greedy", "connect_four")
    random = get_baseline("random", "connect_four")
    result = play_pairing(
        "connect_four", greedy, random, games_per_seat=5, seed=11
    )
    for seat in ("as_first", "as_second"):
        rec = result[seat]
        assert rec["games"] == 5
        assert rec["wins"] + rec["draws"] + rec["losses"] == 5
    combined = result["combined"]
    assert combined["games"] == 10
    # Greedy is deterministic and tactically sound; random is neither.
    assert combined["wins"] == 10


def test_minimax_beats_greedy_at_smoke_scale():
    minimax = get_baseline("connect_four_minimax", "connect_four")
    greedy = get_baseline("connect_four_greedy", "connect_four")
    result = play_pairing(
        "connect_four", minimax, greedy, games_per_seat=2, seed=3
    )
    assert result["combined"]["wins"] == 4


def test_run_tournament_shape():
    result = run_tournament(
        "connect_four",
        "connect_four_greedy",
        ["random"],
        games_per_seat=3,
        seed=99,
        include_yardstick_matches=False,
    )
    assert result["game"] == "connect_four"
    assert result["pairings"]["random"]["combined"]["games"] == 6
    assert result["yardstick_matches"] == {}
    fpa = result["first_player_advantage"]
    assert fpa["decisive_games"] == 6  # greedy sweeps random, no draws
    assert fpa["first_player_wins"] == 3  # seat 0 wins its 3, loses its 3
    assert fpa["first_player_win_rate"] == 0.5
