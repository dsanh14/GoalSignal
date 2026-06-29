"""Unit tests for the external-signal layer (synthetic, fictional teams)."""

from __future__ import annotations

import numpy as np
import pytest

from goalsignal.signals.base import (
    AdvanceProbs,
    OutcomeProbs,
    advance_from_outcome,
    davidson_outcome,
    disagreement,
)
from goalsignal.signals.expert import (
    expert_consensus,
    load_expert_predictions,
)
from goalsignal.signals.market import (
    decimal_to_implied,
    load_market_odds,
    remove_overround,
)
from goalsignal.signals.meta_ensemble import (
    EnsembleConfig,
    MetaEnsemble,
    disagreement_vs_reference,
)
from goalsignal.signals.recent_form import form_signal, load_recent_form
from goalsignal.signals.squad_strength import load_squad_strength, squad_signal
from goalsignal.signals.venue_context import (
    VenueCoefficients,
    VenueContext,
    load_venue_context,
    venue_signal,
)

# --- base types ---------------------------------------------------------------


def test_outcome_probs_normalize_and_array():
    p = OutcomeProbs(2, 1, 1)  # un-normalized input
    np.testing.assert_allclose(p.as_array().sum(), 1.0)
    np.testing.assert_allclose(p.as_array(), [0.5, 0.25, 0.25])


def test_outcome_probs_reject_negative_and_zero():
    with pytest.raises(ValueError):
        OutcomeProbs(-0.1, 0.5, 0.6)
    with pytest.raises(ValueError):
        OutcomeProbs(0, 0, 0)


def test_davidson_monotone_and_symmetric():
    favoured = davidson_outcome(200)
    balanced = davidson_outcome(0)
    assert favoured.home_win > balanced.home_win > davidson_outcome(-200).home_win
    # Symmetry: reversing the advantage swaps home/away, draw unchanged.
    rev = davidson_outcome(-200)
    assert favoured.home_win == pytest.approx(rev.away_win)
    assert favoured.draw == pytest.approx(rev.draw)
    assert balanced.home_win == pytest.approx(balanced.away_win)


def test_davidson_nu_controls_draw():
    assert davidson_outcome(0, nu=2.0).draw > davidson_outcome(0, nu=0.5).draw
    assert davidson_outcome(0, nu=0.0).draw == pytest.approx(0.0)


def test_advance_from_outcome_splits_draw():
    adv = advance_from_outcome(OutcomeProbs(0.5, 0.2, 0.3), a_tiebreak_prob=0.5)
    assert adv.team_a_advances == pytest.approx(0.6)
    assert adv.team_b_advances == pytest.approx(0.4)


def test_disagreement_bounds():
    p = OutcomeProbs(0.5, 0.3, 0.2)
    assert disagreement(p, p) == pytest.approx(0.0)
    far = disagreement(OutcomeProbs(1, 1e-9, 1e-9), OutcomeProbs(1e-9, 1e-9, 1))
    assert far == pytest.approx(1.0, abs=1e-3)


# --- market -------------------------------------------------------------------


def test_decimal_to_implied_and_overround_removal():
    implied = decimal_to_implied([2.0, 4.0, 4.0])
    np.testing.assert_allclose(implied, [0.5, 0.25, 0.25])
    assert implied.sum() == pytest.approx(1.0)  # fair book
    skewed = decimal_to_implied([1.8, 3.5, 4.5])
    assert skewed.sum() > 1.0  # overround present
    for method in ("proportional", "power"):
        norm = remove_overround(skewed, method)
        assert norm.sum() == pytest.approx(1.0)
        assert np.all(norm > 0)


def test_decimal_to_implied_rejects_bad_odds():
    with pytest.raises(ValueError):
        decimal_to_implied([1.0, 2.0])


def test_load_market_odds_two_and_three_way(tmp_path):
    csv = tmp_path / "odds.csv"
    csv.write_text(
        "match_id,source,team_a_odds,draw_odds,team_b_odds,timestamp\n"
        "M1,book,2.10,3.40,3.60,2026-06-01\n"
        "K1,book,1.65,,2.30,2026-07-01\n"
        "BAD,book,0.5,3.0,2.0,2026-06-01\n"  # invalid odds -> skipped
    )
    errors: list[str] = []
    quotes = load_market_odds(csv, on_error=errors)
    assert set(quotes) == {"M1", "K1"}
    assert errors and "BAD" in errors[0]
    assert not quotes["M1"].two_way and quotes["K1"].two_way
    o = quotes["M1"].outcome()
    assert o.home_win > o.away_win
    adv = quotes["K1"].advance()
    assert adv.team_a_advances > adv.team_b_advances


def test_load_market_odds_missing_file_graceful(tmp_path):
    assert load_market_odds(tmp_path / "nope.csv") == {}
    with pytest.raises(FileNotFoundError):
        load_market_odds(tmp_path / "nope.csv", require=True)


# --- squad / form / venue -----------------------------------------------------


def test_squad_signal_favours_stronger_squad(tmp_path):
    csv = tmp_path / "squad.csv"
    csv.write_text(
        "team,total_squad_value,attacking_depth\n"
        "Alpha,1000,90\n"
        "Beta,300,60\n"
        "Gamma,650,75\n"
    )
    table = load_squad_strength(csv)
    sig = squad_signal(table, "Alpha", "Beta")
    assert sig is not None and sig.home_win > sig.away_win
    # Unknown team -> no signal.
    assert squad_signal(table, "Alpha", "Nobody") is None


def test_squad_signal_robust_to_sparse_fields(tmp_path):
    csv = tmp_path / "squad.csv"
    # Only one column present for each team; still standardizable across teams.
    csv.write_text("team,keeper_strength\nAlpha,90\nBeta,60\n")
    table = load_squad_strength(csv)
    assert squad_signal(table, "Alpha", "Beta") is not None


def test_form_signal_direction(tmp_path):
    csv = tmp_path / "form.csv"
    csv.write_text(
        "team,elo_adj_last5,xg_diff\nAlpha,0.5,0.6\nBeta,-0.3,-0.4\nGamma,0.1,0.0\n"
    )
    table = load_recent_form(csv)
    sig = form_signal(table, "Alpha", "Beta")
    assert sig is not None and sig.home_win > sig.away_win


def test_venue_signal_uses_present_fields_only():
    ctx = VenueContext(match_id="M1", host_boost=80.0)
    contexts = {"M1": ctx}
    sig = venue_signal(contexts, "M1")
    assert sig is not None and sig.home_win > sig.away_win
    # Empty context row -> no signal.
    assert venue_signal({"M2": VenueContext(match_id="M2")}, "M2") is None


def test_venue_advantage_components():
    ctx = VenueContext(
        match_id="M1",
        travel_km_a=0.0,
        travel_km_b=2000.0,
        rest_days_a=5.0,
        rest_days_b=3.0,
    )
    pts = ctx.advantage_points(VenueCoefficients())
    # team A travels less (+) and rests more (+): strictly positive.
    assert pts > 0


def test_load_venue_context_missing_file(tmp_path):
    assert load_venue_context(tmp_path / "nope.csv") == {}


# --- expert -------------------------------------------------------------------


def test_expert_validation_and_consensus(tmp_path):
    csv = tmp_path / "expert.csv"
    csv.write_text(
        "match_id,source_model,team_a_win_prob,draw_prob,team_b_win_prob,"
        "team_a_advance_prob,team_b_advance_prob,confidence,reasoning\n"
        "M1,a,0.5,0.3,0.2,,,0.8,strong\n"
        "M1,b,0.4,0.3,0.3,,,0.4,close\n"
        "K1,c,,,,0.7,0.3,0.6,favorite\n"
        "BAD,d,0.9,0.9,0.9,,,0.5,sums to far more than 1\n"
    )
    errors: list[str] = []
    preds = load_expert_predictions(csv, on_error=errors)
    assert "BAD" not in preds and errors
    cons = expert_consensus(preds["M1"])
    assert isinstance(cons, OutcomeProbs)
    # confidence-weighted toward source a (0.8 vs 0.4).
    assert cons.home_win > 0.45
    ko = expert_consensus(preds["K1"], knockout=True)
    assert isinstance(ko, AdvanceProbs) and ko.team_a_advances == pytest.approx(0.7)


# --- meta-ensemble ------------------------------------------------------------


def _config() -> EnsembleConfig:
    return EnsembleConfig(
        default_weights={"historical": 0.5, "market": 0.3, "expert": 0.2},
        model_versions={
            "baseline_historical": {"historical": 1.0},
            "market_only": {"market": 1.0},
        },
        signal_params={},
        market_overround_method="proportional",
        knockout_tiebreak_a_prob=0.5,
        disagreement_threshold=0.15,
    )


def test_meta_ensemble_blend_and_provenance():
    ens = MetaEnsemble(_config())
    signals = {
        "historical": OutcomeProbs(0.5, 0.3, 0.2),
        "market": OutcomeProbs(0.4, 0.3, 0.3),
        "expert": OutcomeProbs(0.6, 0.2, 0.2),
    }
    res = ens.blend(signals)
    np.testing.assert_allclose(res.probs.as_array().sum(), 1.0)
    assert res.missing == []
    assert res.used_weights["historical"] == pytest.approx(0.5)
    assert set(res.components) == set(signals)


def test_meta_ensemble_renormalizes_on_missing():
    ens = MetaEnsemble(_config())
    signals = {
        "historical": OutcomeProbs(0.5, 0.3, 0.2),
        "market": None,  # missing -> weight dropped, renormalized
        "expert": OutcomeProbs(0.6, 0.2, 0.2),
    }
    res = ens.blend(signals)
    assert res.missing == ["market"]
    # historical 0.5 / (0.5+0.2) and expert 0.2 / 0.7
    assert res.used_weights["historical"] == pytest.approx(0.5 / 0.7)
    assert res.used_weights["expert"] == pytest.approx(0.2 / 0.7)
    # blend equals the renormalized mix.
    expected = (
        0.5 / 0.7 * signals["historical"].as_array()
        + 0.2 / 0.7 * signals["expert"].as_array()
    )
    np.testing.assert_allclose(res.probs.as_array(), expected)


def test_meta_ensemble_all_missing_raises():
    ens = MetaEnsemble(_config())
    with pytest.raises(ValueError):
        ens.blend({"historical": None, "market": None, "expert": None})


def test_meta_ensemble_version_weights():
    ens = MetaEnsemble(_config())
    signals = {
        "historical": OutcomeProbs(0.5, 0.3, 0.2),
        "market": OutcomeProbs(0.1, 0.1, 0.8),
    }
    res = ens.blend(signals, version="market_only")
    np.testing.assert_allclose(res.probs.as_array(), signals["market"].as_array())
    assert res.missing == []  # historical not weighted in this version


def test_meta_ensemble_knockout_blend():
    ens = MetaEnsemble(_config())
    res = ens.blend_advance(
        {
            "historical": AdvanceProbs(0.6, 0.4),
            "market": AdvanceProbs(0.5, 0.5),
            "expert": AdvanceProbs(0.7, 0.3),
        }
    )
    assert isinstance(res.probs, AdvanceProbs)
    np.testing.assert_allclose(res.probs.as_array().sum(), 1.0)


def test_disagreement_vs_reference():
    signals = {
        "historical": OutcomeProbs(0.5, 0.3, 0.2),
        "market": OutcomeProbs(0.5, 0.3, 0.2),
        "expert": OutcomeProbs(0.2, 0.2, 0.6),
    }
    report = disagreement_vs_reference(signals, "historical")
    assert report.gaps["market"] == pytest.approx(0.0)
    worst = report.worst()
    assert worst is not None and worst[0] == "expert"


def test_real_ensemble_config_loads():
    """The shipped config/ensemble.yaml parses and exposes all versions."""
    from goalsignal.signals.meta_ensemble import load_ensemble_config

    cfg = load_ensemble_config()
    assert "final_ensemble" in cfg.model_versions
    assert cfg.default_weights["historical"] == pytest.approx(0.35)
    assert sum(cfg.default_weights.values()) == pytest.approx(1.0)
