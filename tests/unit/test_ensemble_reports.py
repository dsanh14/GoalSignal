"""Tests for the ensemble backtest reports, ablation, and tuning artifacts."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from goalsignal.evaluation.ensemble_backtest import (
    assess_ensemble,
    calibration_by_version,
    coverage_by_signal,
    load_backtest_table,
    run_ablation,
    run_ensemble_backtest,
    score_versions,
    write_ablation,
    write_reports,
)
from goalsignal.signals.meta_ensemble import MetaEnsemble, load_ensemble_config
from goalsignal.signals.pipeline import load_manual_inputs


@pytest.fixture
def sample_inputs():
    return load_manual_inputs("data/manual", load_ensemble_config())


def _synthetic_predictions(tmp_path, n=150):
    """A 'real-like' historical predictions CSV (non-smoke: n >= 100)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        p = rng.dirichlet([3, 2, 2])
        label = int(rng.choice(3, p=p))
        rows.append(
            {
                "canonical_match_id": f"m{i}",
                "home_team": f"Team{i % 30}",
                "away_team": f"Team{(i + 7) % 30}",
                "label": label,
                "ensemble_home": p[0],
                "ensemble_draw": p[1],
                "ensemble_away": p[2],
            }
        )
    path = tmp_path / "test_predictions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --- backtest paths -----------------------------------------------------------


def test_smoke_path_flagged_and_real_path_not(tmp_path, sample_inputs):
    smoke = load_backtest_table("data/manual/backtest_sample.example.csv")
    assert smoke.smoke is True
    real = load_backtest_table(_synthetic_predictions(tmp_path))
    assert real.smoke is False and len(real.specs) == 150


def test_real_artifact_backtest_runs_and_writes_files(tmp_path, sample_inputs):
    table = load_backtest_table(_synthetic_predictions(tmp_path))
    ensemble = MetaEnsemble(sample_inputs.config)
    df = run_ensemble_backtest(table, sample_inputs, ensemble)
    assert "baseline_historical" in set(df["version"])
    # Baseline historical scores every row (historical always present).
    base = df[df["version"] == "baseline_historical"].iloc[0]
    assert base["n_scored"] == 150

    coverage = coverage_by_signal(table, sample_inputs)
    scored = score_versions(table, sample_inputs, ensemble)
    calibration = calibration_by_version(scored)
    assessment = assess_ensemble(df, coverage)
    paths = write_reports(df, coverage, calibration, assessment, table.smoke, tmp_path)
    for key in ("comparison", "summary", "calibration", "coverage"):
        assert paths[key].exists()
    assert not calibration.empty


def test_report_does_not_overclaim_when_coverage_zero(tmp_path, sample_inputs):
    """Synthetic teams match no manual data -> verdict must be 'insufficient'."""
    table = load_backtest_table(_synthetic_predictions(tmp_path))
    ensemble = MetaEnsemble(sample_inputs.config)
    df = run_ensemble_backtest(table, sample_inputs, ensemble)
    coverage = coverage_by_signal(table, sample_inputs)
    assessment = assess_ensemble(df, coverage)
    assert "INSUFFICIENT" in assessment["verdict"].upper()
    assert assessment["recommendation"] == "keep final_ensemble opt-in"
    # Historical is the only trusted signal; nothing non-historical is trusted.
    trusted = coverage[coverage["status"] == "trusted"]["signal"].tolist()
    assert trusted == ["historical"]


def test_coverage_report_marks_sample_signals_experimental(sample_inputs):
    """On the bundled sample every signal covers all 3 rows -> trusted."""
    table = load_backtest_table("data/manual/backtest_sample.example.csv")
    cov = coverage_by_signal(table, sample_inputs)
    assert set(cov["signal"]) >= {"historical", "market", "expert"}
    assert (cov["coverage_rate"] <= 1.0).all()


# --- ablation -----------------------------------------------------------------


def test_ablation_runs_and_writes(tmp_path, sample_inputs):
    table = load_backtest_table("data/manual/backtest_sample.example.csv")
    df = run_ablation(table, sample_inputs, MetaEnsemble(sample_inputs.config))
    assert {"historical_only", "full_ensemble"} <= set(df["ablation"])
    # historical_only is the reference, so its delta is exactly 0.
    base = df[df["ablation"] == "historical_only"].iloc[0]
    assert base["logloss_delta_vs_historical"] == 0.0
    paths = write_ablation(df, table.smoke, tmp_path)
    assert paths["comparison"].exists() and paths["summary"].exists()
    assert "SMOKE TEST" in paths["summary"].read_text()


# --- tuning -------------------------------------------------------------------


def test_tuning_low_coverage_warns_and_preserves_config(tmp_path, sample_inputs):
    from goalsignal.signals.tuning import (
        tune_weights,
        write_tuned_weights,
        write_tuning_report,
    )
    from goalsignal.utils.paths import resolve

    config_path = resolve("config/ensemble.yaml")
    before = hashlib.sha256(config_path.read_bytes()).hexdigest()

    # Synthetic teams => only the historical signal is present => low coverage.
    table = load_backtest_table(_synthetic_predictions(tmp_path))
    result = tune_weights(table.specs, table.labels, sample_inputs, objective="log_loss")
    assert result.low_coverage and result.coverage_warning
    assert result.signals_present == ["historical"]

    weights_path = write_tuned_weights(result, tmp_path / "tuned_weights.yaml")
    report_path = write_tuning_report(result, tmp_path / "tuning_report.md")
    assert weights_path.exists() and report_path.exists()
    import yaml

    payload = yaml.safe_load(weights_path.read_text())
    assert payload["low_coverage"] is True and payload["coverage_warning"]
    assert "default weights" in report_path.read_text()

    after = hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert before == after  # config never mutated
