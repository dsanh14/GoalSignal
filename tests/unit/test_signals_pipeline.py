"""Pipeline-level and CLI tests for the signal layer.

Complements ``test_signals.py`` with: end-to-end renormalization through
``load_manual_inputs``/``blend_match``, market-conversion edge cases, expert
(LLM) probability validation edge cases, ensemble provenance invariants, and
CLI smoke tests that exercise the README commands on the committed example data.
"""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from goalsignal.cli import app
from goalsignal.signals.base import AdvanceProbs, OutcomeProbs
from goalsignal.signals.expert import expert_consensus, load_expert_predictions
from goalsignal.signals.market import (
    decimal_to_implied,
    load_market_odds,
    remove_overround,
)
from goalsignal.signals.meta_ensemble import EnsembleConfig, MetaEnsemble
from goalsignal.signals.pipeline import (
    blend_match,
    build_signals,
    load_manual_inputs,
    load_matches,
)

runner = CliRunner()


# --- helpers ------------------------------------------------------------------


def _write(path, text: str):
    path.write_text(text)
    return path


def _product_config() -> EnsembleConfig:
    """The real six-signal product weights, decoupled from the YAML file."""
    return EnsembleConfig(
        default_weights={
            "historical": 0.35,
            "market": 0.25,
            "squad_strength": 0.15,
            "recent_form": 0.10,
            "expert": 0.10,
            "venue_context": 0.05,
        },
        model_versions={"final_ensemble": {
            "historical": 0.35,
            "market": 0.25,
            "squad_strength": 0.15,
            "recent_form": 0.10,
            "expert": 0.10,
            "venue_context": 0.05,
        }},
        signal_params={
            "davidson_scale": 400.0,
            "davidson_nu": 1.0,
            "squad_points_per_z": 60.0,
            "form_points_per_z": 40.0,
            "venue": {"travel_per_1000km": 8.0, "rest_per_day": 6.0, "timezone_per_hour": 3.0},
        },
        market_overround_method="proportional",
        knockout_tiebreak_a_prob=0.5,
        disagreement_threshold=0.15,
    )


# --- market odds conversion ---------------------------------------------------


def test_overround_value_is_booksum_minus_one(tmp_path):
    csv = _write(
        tmp_path / "odds.csv",
        "match_id,source,team_a_odds,draw_odds,team_b_odds,timestamp\n"
        "M1,b,2.0,4.0,4.0,t\n"  # fair book -> overround 0
        "M2,b,1.8,3.5,4.5,t\n",
    )
    quotes = load_market_odds(csv)
    assert quotes["M1"].overround() == pytest.approx(0.0, abs=1e-9)
    expected = float(decimal_to_implied([1.8, 3.5, 4.5]).sum() - 1.0)
    assert quotes["M2"].overround() == pytest.approx(expected)


def test_power_method_differs_from_proportional_under_margin():
    implied = decimal_to_implied([1.5, 4.5, 7.0])  # strong favourite, real margin
    prop = remove_overround(implied, "proportional")
    power = remove_overround(implied, "power")
    np.testing.assert_allclose(prop.sum(), 1.0)
    np.testing.assert_allclose(power.sum(), 1.0)
    # The two de-vig methods disagree when a margin is present.
    assert np.abs(prop - power).max() > 1e-4
    # With a positive margin k>1, so the power method accentuates the favourite
    # relative to proportional scaling (favourite-longshot correction).
    assert power[0] > prop[0]
    assert power[-1] < prop[-1]


def test_remove_overround_rejects_unknown_method():
    with pytest.raises(ValueError):
        remove_overround(np.array([0.5, 0.5]), "bogus")


def test_market_keeps_latest_quote_by_timestamp(tmp_path):
    csv = _write(
        tmp_path / "odds.csv",
        "match_id,source,team_a_odds,draw_odds,team_b_odds,timestamp\n"
        "M1,early,3.0,3.0,3.0,2026-06-01\n"
        "M1,late,1.5,4.0,6.0,2026-06-05\n",
    )
    quotes = load_market_odds(csv)
    assert quotes["M1"].source == "late"
    assert quotes["M1"].outcome().home_win > 0.5


# --- expert / LLM probability validation --------------------------------------


def test_expert_rejects_triple_sum_out_of_tolerance(tmp_path):
    csv = _write(
        tmp_path / "e.csv",
        "match_id,source_model,team_a_win_prob,draw_prob,team_b_win_prob,"
        "team_a_advance_prob,team_b_advance_prob,confidence,reasoning\n"
        "OK,m,0.50,0.30,0.20,,,0.9,fine\n"
        "NEAR,m,0.50,0.30,0.18,,,0.9,within tolerance -> normalized\n"
        "FAR,m,0.50,0.50,0.50,,,0.9,sums to 1.5\n",
    )
    errors: list[str] = []
    preds = load_expert_predictions(csv, on_error=errors)
    assert "OK" in preds and "NEAR" in preds and "FAR" not in preds
    assert any("FAR" in e for e in errors)
    # NEAR (sum 0.98) is accepted and renormalized to sum 1.
    np.testing.assert_allclose(preds["NEAR"][0].outcome.as_array().sum(), 1.0)


def test_expert_rejects_advance_pair_out_of_tolerance(tmp_path):
    csv = _write(
        tmp_path / "e.csv",
        "match_id,source_model,team_a_win_prob,draw_prob,team_b_win_prob,"
        "team_a_advance_prob,team_b_advance_prob,confidence,reasoning\n"
        "K1,m,,,,0.7,0.3,0.8,ok\n"
        "KBAD,m,,,,0.9,0.5,0.8,sums to 1.4\n",
    )
    errors: list[str] = []
    preds = load_expert_predictions(csv, on_error=errors)
    assert "K1" in preds and "KBAD" not in preds and errors


def test_expert_confidence_clamped_and_partial_rows(tmp_path):
    csv = _write(
        tmp_path / "e.csv",
        "match_id,source_model,team_a_win_prob,draw_prob,team_b_win_prob,"
        "team_a_advance_prob,team_b_advance_prob,confidence,reasoning\n"
        "M1,m,0.4,0.3,0.3,,,5.0,confidence above 1\n",  # clamped to 1.0
    )
    preds = load_expert_predictions(csv)
    pred = preds["M1"][0]
    assert pred.confidence == 1.0
    assert pred.outcome is not None and pred.advance is None
    # No knockout info -> knockout consensus is None.
    assert expert_consensus(preds["M1"], knockout=True) is None


def test_expert_row_with_no_complete_probs_skipped(tmp_path):
    csv = _write(
        tmp_path / "e.csv",
        "match_id,source_model,team_a_win_prob,draw_prob,team_b_win_prob,"
        "team_a_advance_prob,team_b_advance_prob,confidence,reasoning\n"
        "M1,m,0.4,,,,,0.5,incomplete triple and no pair\n",
    )
    errors: list[str] = []
    preds = load_expert_predictions(csv, on_error=errors)
    assert preds == {} and errors


# --- ensemble provenance + renormalization invariants -------------------------


def test_blend_used_weights_sum_to_one_and_skip_zero_weights():
    cfg = EnsembleConfig(
        default_weights={"historical": 0.6, "market": 0.4, "expert": 0.0},
        model_versions={},
        signal_params={},
        market_overround_method="proportional",
        knockout_tiebreak_a_prob=0.5,
        disagreement_threshold=0.15,
    )
    ens = MetaEnsemble(cfg)
    res = ens.blend(
        {
            "historical": OutcomeProbs(0.5, 0.3, 0.2),
            "market": OutcomeProbs(0.4, 0.3, 0.3),
            "expert": OutcomeProbs(0.9, 0.05, 0.05),  # present but weight 0
        }
    )
    assert "expert" not in res.used_weights  # zero weight excluded
    assert sum(res.used_weights.values()) == pytest.approx(1.0)
    assert set(res.components) == {"historical", "market"}


def test_is_flagged_threshold():
    cfg = _product_config()
    ens = MetaEnsemble(cfg)
    near = ens.blend(
        {"historical": OutcomeProbs(0.5, 0.3, 0.2), "market": OutcomeProbs(0.5, 0.3, 0.2)}
    )
    assert not ens.is_flagged(near)
    far = ens.blend(
        {"historical": OutcomeProbs(0.8, 0.1, 0.1), "market": OutcomeProbs(0.1, 0.1, 0.8)}
    )
    assert ens.is_flagged(far)


# --- pipeline-level renormalization (load -> build -> blend) ------------------


def test_pipeline_renormalizes_when_files_absent(tmp_path):
    """Only market + matches present; squad/form/venue/expert missing entirely."""
    _write(
        tmp_path / "market_odds.csv",
        "match_id,source,team_a_odds,draw_odds,team_b_odds,timestamp\n"
        "G01,b,2.10,3.40,3.60,t\n",
    )
    _write(
        tmp_path / "matches.csv",
        "match_id,stage,team_a,team_b,historical_home_win,historical_draw,historical_away_win\n"
        "G01,group,Alpha,Beta,0.46,0.27,0.27\n",
    )
    cfg = _product_config()
    inputs = load_manual_inputs(tmp_path, cfg)
    assert len(inputs.squad.teams) == 0 and len(inputs.form.teams) == 0
    ens = MetaEnsemble(cfg)
    spec = load_matches(tmp_path / "matches.csv")[0]
    result, _ = blend_match(spec, inputs, ens, version="final_ensemble")
    # Only historical + market available; the other four renormalize away.
    assert set(result.used_weights) == {"historical", "market"}
    assert sorted(result.missing) == [
        "expert",
        "recent_form",
        "squad_strength",
        "venue_context",
    ]
    assert result.used_weights["historical"] == pytest.approx(0.35 / 0.60)
    assert result.used_weights["market"] == pytest.approx(0.25 / 0.60)
    np.testing.assert_allclose(result.probs.as_array().sum(), 1.0)


def test_pipeline_partial_team_coverage_drops_only_that_signal(tmp_path):
    """A team missing from the squad file makes only the squad signal None."""
    _write(
        tmp_path / "squad_strength.csv",
        "team,total_squad_value,attacking_depth\nAlpha,1000,90\nGamma,500,70\n",
    )
    _write(
        tmp_path / "matches.csv",
        "match_id,stage,team_a,team_b,historical_home_win,historical_draw,historical_away_win\n"
        "G01,group,Alpha,Beta,0.46,0.27,0.27\n",  # Beta absent from squad file
    )
    cfg = _product_config()
    inputs = load_manual_inputs(tmp_path, cfg)
    spec = load_matches(tmp_path / "matches.csv")[0]
    signals = build_signals(spec, inputs)
    assert signals["squad_strength"] is None  # Beta missing -> no squad signal
    assert signals["historical"] is not None


def test_pipeline_knockout_reduces_adjustment_signals(tmp_path):
    """Adjustment signals become AdvanceProbs for a knockout match."""
    _write(
        tmp_path / "squad_strength.csv",
        "team,total_squad_value\nAlpha,1000\nBeta,300\nGamma,650\n",
    )
    _write(
        tmp_path / "matches.csv",
        "match_id,stage,team_a,team_b,"
        "historical_team_a_advances,historical_team_b_advances\n"
        "K01,knockout,Alpha,Beta,0.58,0.42\n",
    )
    cfg = _product_config()
    inputs = load_manual_inputs(tmp_path, cfg)
    spec = load_matches(tmp_path / "matches.csv")[0]
    result, signals = blend_match(spec, inputs, MetaEnsemble(cfg), version="final_ensemble")
    assert isinstance(result.probs, AdvanceProbs)
    assert isinstance(signals["squad_strength"], AdvanceProbs)
    assert isinstance(signals["historical"], AdvanceProbs)
    np.testing.assert_allclose(result.probs.as_array().sum(), 1.0)


# --- CLI smoke tests on the committed example data ----------------------------


def test_cli_signals_validate_runs():
    res = runner.invoke(app, ["signals", "validate"])
    assert res.exit_code == 0, res.output
    assert "market quotes:" in res.output


def test_cli_signals_market_runs():
    res = runner.invoke(app, ["signals", "market"])
    assert res.exit_code == 0, res.output
    assert "overround" in res.output


@pytest.mark.parametrize(
    "version",
    [
        "baseline_historical",
        "market_only",
        "squad_form_challenger",
        "llm_adjusted_challenger",
        "final_ensemble",
    ],
)
def test_cli_signals_blend_all_versions_run(version):
    res = runner.invoke(app, ["signals", "blend", "--version", version])
    assert res.exit_code == 0, res.output
    assert "weights:" in res.output


def test_cli_signals_blend_writes_csv(tmp_path):
    out = tmp_path / "blended.csv"
    res = runner.invoke(app, ["signals", "blend", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists()
    import pandas as pd

    df = pd.read_csv(out)
    assert {"match_id", "home_win", "draw", "away_win"} <= set(df.columns)


def test_cli_signals_disagreement_runs():
    res = runner.invoke(app, ["signals", "disagreement"])
    assert res.exit_code == 0, res.output
    assert "threshold" in res.output


def test_cli_signals_blend_unknown_version_fails():
    res = runner.invoke(app, ["signals", "blend", "--version", "does_not_exist"])
    assert res.exit_code != 0
