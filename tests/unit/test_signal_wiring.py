"""Tests for wiring the signal layer into the real model / API / tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goalsignal.signals.api import EnsemblePredictor
from goalsignal.signals.base import AdvanceProbs, OutcomeProbs
from goalsignal.signals.historical_adapter import (
    SOURCE_LIVE,
    SOURCE_UNAVAILABLE,
    LiveModelHistorical,
    UnavailableHistorical,
    advance_probs_from_adapter,
)
from goalsignal.signals.meta_ensemble import load_ensemble_config
from goalsignal.signals.pipeline import MatchSpec, load_manual_inputs
from goalsignal.tournament.model_adapter import RatingsGoalAdapter

# --- stubs --------------------------------------------------------------------


class _StubGoalModel:
    """Minimal goal model: Poisson score matrix from Elo-diff-driven lambdas."""

    def predict_expected_goals(self, frame: pd.DataFrame) -> np.ndarray:
        d = frame["elo_diff"].to_numpy(dtype=float) / 400.0
        return np.column_stack([np.exp(0.2 + 0.35 * d), np.exp(0.2 - 0.35 * d)])

    def score_matrix(self, lam_home: float, lam_away: float) -> np.ndarray:
        from scipy.stats import poisson

        h = poisson.pmf(np.arange(8), lam_home)
        a = poisson.pmf(np.arange(8), lam_away)
        m = np.outer(h, a)
        return m / m.sum()


class _StubLive:
    """Stand-in for LiveModel exposing the attributes the adapter needs."""

    def __init__(self, probs=(0.6, 0.25, 0.15), raise_outcome=False):
        self.ratings = {"Alpha": 1650.0, "Beta": 1450.0}
        self.goal_model = _StubGoalModel()
        self._probs = np.array(probs)
        self._raise = raise_outcome

    def feature_row(self, home, away, neutral):
        r_h = self.ratings.get(home, 1500.0)
        r_a = self.ratings.get(away, 1500.0)
        return pd.DataFrame(
            {"home_elo_pre": [r_h], "away_elo_pre": [r_a],
             "elo_diff": [r_h - r_a], "neutral": [bool(neutral)]}
        )

    def predict_outcome(self, frame):
        if self._raise:
            raise RuntimeError("model unavailable")
        return self._probs.reshape(1, 3)


# --- historical adapter -------------------------------------------------------


def test_historical_outcome_conversion():
    hist = LiveModelHistorical(_StubLive(probs=(0.6, 0.25, 0.15)))
    sig = hist.outcome("Alpha", "Beta", neutral=True)
    assert sig.source == SOURCE_LIVE and sig.outcome is not None
    np.testing.assert_allclose(sig.outcome.as_array(), [0.6, 0.25, 0.15])
    assert sig.advance is None and sig.available


def test_historical_advance_conversion_favours_stronger():
    hist = LiveModelHistorical(_StubLive())
    sig = hist.advance("Alpha", "Beta", neutral=True)
    assert sig.source == SOURCE_LIVE and isinstance(sig.advance, AdvanceProbs)
    np.testing.assert_allclose(sig.advance.as_array().sum(), 1.0)
    assert sig.advance.team_a_advances > sig.advance.team_b_advances  # Alpha stronger


def test_advance_probs_from_adapter_direct():
    adapter = RatingsGoalAdapter({"Alpha": 1650.0, "Beta": 1450.0}, _StubGoalModel())
    adv = advance_probs_from_adapter(adapter, "Alpha", "Beta")
    assert isinstance(adv, AdvanceProbs)
    np.testing.assert_allclose(adv.as_array().sum(), 1.0)
    assert adv.team_a_advances > 0.5


def test_missing_historical_returns_unavailable():
    hist = LiveModelHistorical(_StubLive(raise_outcome=True))
    sig = hist.outcome("Alpha", "Beta")
    assert sig.source == SOURCE_UNAVAILABLE and sig.outcome is None and not sig.available


def test_unavailable_historical_provider():
    null = UnavailableHistorical()
    assert null.outcome("A", "B").source == SOURCE_UNAVAILABLE
    assert null.advance("A", "B").advance is None


# --- ensemble prediction API --------------------------------------------------


@pytest.fixture
def empty_inputs(tmp_path):
    """ManualInputs from an empty dir: every manual signal is missing."""
    return load_manual_inputs(tmp_path, load_ensemble_config())


@pytest.fixture
def sample_inputs():
    """ManualInputs from the committed example files (G01-G03 covered)."""
    return load_manual_inputs("data/manual", load_ensemble_config())


def test_api_provenance_with_live_model(sample_inputs):
    predictor = EnsemblePredictor(sample_inputs, historical=LiveModelHistorical(_StubLive()))
    spec = MatchSpec("G01", "group", "Spain", "Germany", neutral=True)
    pred = predictor.predict_match_ensemble(spec, version="final_ensemble")
    assert pred.historical_source == SOURCE_LIVE
    assert isinstance(pred.probs, OutcomeProbs)
    assert "historical" in pred.components
    assert pred.used_weights and abs(sum(pred.used_weights.values()) - 1.0) < 1e-9
    assert 0.0 <= pred.disagreement <= 1.0
    assert isinstance(pred.flagged, bool)
    row = pred.to_row()
    assert {"match_id", "home_win", "draw", "away_win", "historical_source"} <= set(row)


def test_api_fixture_source_without_live_model(sample_inputs):
    predictor = EnsemblePredictor(sample_inputs)  # UnavailableHistorical
    spec = MatchSpec("G01", "group", "Spain", "Germany",
                     historical=OutcomeProbs(0.46, 0.27, 0.27))
    pred = predictor.predict(spec, version="final_ensemble")
    assert pred.historical_source == "fixture"
    assert "historical" in pred.components


def test_api_missing_manual_files_no_crash(empty_inputs):
    """Optional files absent: only the historical signal is present."""
    predictor = EnsemblePredictor(empty_inputs)
    spec = MatchSpec("X", "group", "Alpha", "Beta",
                     historical=OutcomeProbs(0.5, 0.3, 0.2))
    pred = predictor.predict(spec, version="final_ensemble")
    assert pred.components == ["historical"]
    # knockout_upset is configured in final_ensemble but never produced for a
    # group match, so it renormalizes away like the other absent signals.
    assert set(pred.missing) == {"market", "squad_strength", "recent_form",
                                 "expert", "venue_context", "knockout_upset"}
    np.testing.assert_allclose(pred.probs.as_array(), [0.5, 0.3, 0.2])


def test_api_knockout_and_type_guards(sample_inputs):
    predictor = EnsemblePredictor(sample_inputs, historical=LiveModelHistorical(_StubLive()))
    ko = MatchSpec("K01", "knockout", "Spain", "Germany")
    pred = predictor.predict_knockout_ensemble(ko, version="final_ensemble")
    assert isinstance(pred.probs, AdvanceProbs)
    with pytest.raises(ValueError):
        predictor.predict_match_ensemble(ko)
    grp = MatchSpec("G01", "group", "Spain", "Germany")
    with pytest.raises(ValueError):
        predictor.predict_knockout_ensemble(grp)


def test_api_batch_dataframe(sample_inputs):
    predictor = EnsemblePredictor(sample_inputs, historical=LiveModelHistorical(_StubLive()))
    specs = [
        MatchSpec("G01", "group", "Spain", "Germany"),
        MatchSpec("K01", "knockout", "Spain", "Germany"),
    ]
    df = predictor.predict_batch_ensemble(specs, version="final_ensemble")
    assert len(df) == 2
    assert {"match_id", "version", "historical_source", "disagreement"} <= set(df.columns)


# --- backtest + tuning --------------------------------------------------------


def test_ensemble_backtest_smoke(sample_inputs):
    from goalsignal.evaluation.ensemble_backtest import (
        load_backtest_table,
        run_ensemble_backtest,
    )
    from goalsignal.signals.meta_ensemble import MetaEnsemble

    table = load_backtest_table("data/manual/backtest_sample.example.csv")
    assert table.smoke and len(table.specs) == 3
    df = run_ensemble_backtest(table, sample_inputs, MetaEnsemble(sample_inputs.config))
    assert not df.empty
    assert {"baseline_historical", "final_ensemble"} <= set(df["version"])
    for col in ("log_loss", "mean_signals_used", "missing_signal_rate",
                "high_disagreement_count"):
        assert col in df.columns


def test_tune_weights_writes_artifact_and_preserves_config(tmp_path, sample_inputs):
    import hashlib

    from goalsignal.evaluation.ensemble_backtest import load_backtest_table
    from goalsignal.signals.tuning import tune_weights, write_tuned_weights
    from goalsignal.utils.paths import resolve

    config_path = resolve("config/ensemble.yaml")
    before = hashlib.sha256(config_path.read_bytes()).hexdigest()

    table = load_backtest_table("data/manual/backtest_sample.example.csv")
    result = tune_weights(table.specs, table.labels, sample_inputs, objective="log_loss")
    assert abs(sum(result.weights.values()) - 1.0) < 1e-9
    assert "log_loss" in result.validation_metrics

    out = tmp_path / "tuned.yaml"
    written = write_tuned_weights(result, out)
    assert written.exists()
    import yaml

    payload = yaml.safe_load(written.read_text())
    assert "tuned_weights" in payload and "validation_metrics_tuned" in payload

    # The human-readable default config must be untouched.
    after = hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert before == after


def test_tune_weights_brier_objective(sample_inputs):
    from goalsignal.evaluation.ensemble_backtest import load_backtest_table
    from goalsignal.signals.tuning import tune_weights

    table = load_backtest_table("data/manual/backtest_sample.example.csv")
    result = tune_weights(table.specs, table.labels, sample_inputs, objective="brier")
    assert result.objective == "brier"
    # Tuned objective should not be worse than the default weights' on validation.
    assert result.validation_metrics["brier"] <= result.baseline_metrics["brier"] + 1e-9
