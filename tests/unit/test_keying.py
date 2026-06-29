"""Dynamic knockout signal keying: team-pair matching + match_id precedence."""

from __future__ import annotations

import numpy as np

from goalsignal.signals.base import OutcomeProbs
from goalsignal.signals.keying import PairIndex, normalize_team, pair_key
from goalsignal.signals.meta_ensemble import load_ensemble_config
from goalsignal.signals.pipeline import MatchSpec, build_signals, load_manual_inputs

# --- PairIndex ----------------------------------------------------------------


def test_normalize_and_pair_key():
    assert normalize_team("  Spain  ") == "spain"
    assert pair_key("Spain", "Germany") == pair_key("spain", " germany ")
    assert pair_key("A", "B") != pair_key("B", "A")


def test_pairindex_match_id_precedence_and_orientation():
    idx = PairIndex.build([("M1", "Alpha", "Beta", "payload")])
    # match_id wins, forward orientation.
    assert idx.resolve("M1", "Beta", "Alpha") == ("payload", 1)
    # forward pair.
    assert idx.resolve(None, "Alpha", "Beta") == ("payload", 1)
    # reverse pair -> orientation -1.
    assert idx.resolve(None, "Beta", "Alpha") == ("payload", -1)
    # miss.
    assert idx.resolve("nope", "X", "Y") == (None, 0)


def test_pairindex_forward_wins_over_reverse():
    # Two entries; a forward key must not be shadowed by another's reverse.
    idx = PairIndex.build(
        [("", "Alpha", "Beta", "fwd"), ("", "Beta", "Alpha", "other")]
    )
    payload, orient = idx.resolve(None, "Alpha", "Beta")
    assert payload == "fwd" and orient == 1


# --- build_signals dynamic matching -------------------------------------------


def _inputs(tmp_path):
    return load_manual_inputs(tmp_path, load_ensemble_config())


def test_market_attaches_by_team_pair_without_match_id(tmp_path):
    (tmp_path / "market_odds.csv").write_text(
        "team_a,team_b,team_a_odds,draw_odds,team_b_odds\n"
        "Alpha,Beta,1.80,3.50,4.50\n"
    )
    inputs = _inputs(tmp_path)
    fwd = build_signals(MatchSpec("dyn", "group", "Alpha", "Beta"), inputs)["market"]
    assert isinstance(fwd, OutcomeProbs) and fwd.home_win > fwd.away_win
    # Reverse fixture order -> flipped perspective.
    rev = build_signals(MatchSpec("dyn", "group", "Beta", "Alpha"), inputs)["market"]
    assert isinstance(rev, OutcomeProbs)
    np.testing.assert_allclose(rev.as_array(), fwd.flip().as_array())


def test_match_id_takes_precedence_over_team_pair(tmp_path):
    # Same teams, two different prices: one keyed by match_id, one pair-only.
    (tmp_path / "market_odds.csv").write_text(
        "match_id,team_a,team_b,team_a_odds,draw_odds,team_b_odds\n"
        "M1,Alpha,Beta,1.20,7.00,12.0\n"  # heavy Alpha favourite (match_id)
        ",Alpha,Beta,4.00,3.50,1.90\n"  # Beta favourite (pair-only)
    )
    inputs = _inputs(tmp_path)
    # A spec carrying match_id M1 must use the match_id row (Alpha favourite),
    # even though a team-pair row also exists.
    probs = build_signals(MatchSpec("M1", "knockout", "Alpha", "Beta"), inputs)["market"]
    assert probs.team_a_advances > 0.7  # the M1 heavy-favourite price


def test_expert_and_venue_attach_by_team_pair(tmp_path):
    (tmp_path / "expert_predictions.csv").write_text(
        "team_a,team_b,source_model,team_a_win_prob,draw_prob,team_b_win_prob,confidence\n"
        "Alpha,Beta,m,0.6,0.25,0.15,0.8\n"
    )
    (tmp_path / "venue_context.csv").write_text(
        "team_a,team_b,host_boost\nAlpha,Beta,80\n"
    )
    inputs = _inputs(tmp_path)
    sig = build_signals(MatchSpec("dyn", "group", "Alpha", "Beta"), inputs)
    assert sig["expert"] is not None and sig["expert"].home_win > sig["expert"].away_win
    venue = sig["venue_context"]
    assert venue is not None and venue.home_win > venue.away_win  # host boost favours A
    # Reverse order negates the directional advantage.
    rev = build_signals(MatchSpec("dyn", "group", "Beta", "Alpha"), inputs)["venue_context"]
    assert rev.home_win < rev.away_win  # boost now favours the other side


def test_existing_match_id_only_files_still_work():
    """The committed example files (match_id keyed) behave as before."""
    inputs = load_manual_inputs("data/manual", load_ensemble_config())
    sig = build_signals(MatchSpec("G01", "group", "Spain", "Germany"), inputs)
    assert sig["market"] is not None and sig["expert"] is not None
