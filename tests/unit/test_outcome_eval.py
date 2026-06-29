"""Unit tests for outcome-first evaluation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from goalsignal.evaluation.outcome_eval import (
    binary_brier,
    binary_calibration_table,
    binary_log_loss,
    binary_summary,
    calibration_table,
    compare,
    format_comparison,
    uniform_baseline_logloss,
)


def test_calibration_table_perfect_model():
    # A model that always says "home" with prob 1, and home always wins.
    n = 50
    probs = np.tile([1.0, 0.0, 0.0], (n, 1))
    labels = np.zeros(n, dtype=int)
    rows = calibration_table(probs, labels)
    top = next(r for r in rows if r["outcome"] == "home_win" and r["bin"] == 9)
    assert top["mean_predicted"] == pytest.approx(1.0)
    assert top["empirical_frequency"] == pytest.approx(1.0)


def test_binary_metrics_perfect_and_uniform():
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1, 0, 1, 0])
    assert binary_log_loss(p, y) < 1e-6
    assert binary_brier(p, y) == 0.0
    half = np.full(4, 0.5)
    s = binary_summary(half, y)
    assert s["brier"] == pytest.approx(0.25)
    assert s["log_loss"] == pytest.approx(-np.log(0.5))


def test_binary_calibration_table_bins():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(int)  # well-calibrated by construction
    rows = binary_calibration_table(p, y, n_bins=5)
    assert all(r["count"] > 0 for r in rows)
    # mean predicted and empirical frequency should track within a bin.
    for r in rows:
        assert abs(r["mean_predicted"] - r["empirical_frequency"]) < 0.2


def test_compare_sorts_by_logloss():
    n = 100
    labels = np.zeros(n, dtype=int)
    good = np.tile([0.8, 0.1, 0.1], (n, 1))
    bad = np.tile([0.2, 0.4, 0.4], (n, 1))
    rows = compare({"bad": (bad, labels), "good": (good, labels)})
    assert rows[0]["model"] == "good"
    assert rows[0]["log_loss"] < rows[1]["log_loss"]
    table = format_comparison(rows)
    assert "good" in table and "logloss" in table


def test_uniform_baseline_constant():
    assert uniform_baseline_logloss() == pytest.approx(1.0986, abs=1e-3)
