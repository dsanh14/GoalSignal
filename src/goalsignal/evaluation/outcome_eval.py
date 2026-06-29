"""Outcome-first evaluation: per-class calibration, binary (advance) metrics,
and a model-comparison summary table.

These complement :mod:`goalsignal.evaluation.metrics` (log loss, Brier, RPS,
ECE, reliability) with the views the win/advance product needs:

* **Multiclass calibration tables** — per outcome (home/draw/away), binned
  predicted-vs-empirical frequency, the honest way to see *where* a model is
  mis-calibrated rather than a single ECE number.
* **Binary metrics for knockout advance probabilities** — log loss, Brier, and
  a calibration table over P(team A advances).
* **compare()** — a tidy table of the primary metrics across several models so a
  backtest can rank the baseline, market-only, challengers, and final ensemble.
"""

from __future__ import annotations

import numpy as np

from goalsignal.evaluation.metrics import brier_score, log_loss, summarize

_EPS = 1e-12

OUTCOME_LABELS = ("home_win", "draw", "away_win")


def calibration_table(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Per-class reliability table for a 3-way outcome model.

    For each outcome class and each predicted-probability bin, reports the mean
    predicted probability, the empirical frequency of that outcome, and the
    sample count. Rows with no samples are omitted.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels)
    rows: list[dict] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for cls, name in enumerate(OUTCOME_LABELS):
        p = probs[:, cls]
        y = (labels == cls).astype(float)
        bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            mask = bins == b
            if not mask.any():
                continue
            rows.append(
                {
                    "outcome": name,
                    "bin": b,
                    "bin_lo": float(edges[b]),
                    "bin_hi": float(edges[b + 1]),
                    "count": int(mask.sum()),
                    "mean_predicted": float(p[mask].mean()),
                    "empirical_frequency": float(y[mask].mean()),
                }
            )
    return rows


def binary_log_loss(p_a: np.ndarray, advanced_a: np.ndarray) -> float:
    """Log loss of P(team A advances) against binary outcomes (1 = A advanced)."""
    p = np.clip(np.asarray(p_a, dtype=float), _EPS, 1 - _EPS)
    y = np.asarray(advanced_a, dtype=float)
    return -float((y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def binary_brier(p_a: np.ndarray, advanced_a: np.ndarray) -> float:
    """Brier score of P(team A advances)."""
    p = np.asarray(p_a, dtype=float)
    y = np.asarray(advanced_a, dtype=float)
    return float(((p - y) ** 2).mean())


def binary_accuracy(p_a: np.ndarray, advanced_a: np.ndarray) -> float:
    p = np.asarray(p_a, dtype=float)
    y = np.asarray(advanced_a, dtype=float)
    return float(((p >= 0.5).astype(float) == y).mean())


def binary_calibration_table(
    p_a: np.ndarray, advanced_a: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Reliability table for the advance probability (predicted vs realized)."""
    p = np.asarray(p_a, dtype=float)
    y = np.asarray(advanced_a, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows: list[dict] = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "empirical_frequency": float(y[mask].mean()),
            }
        )
    return rows


def binary_summary(p_a: np.ndarray, advanced_a: np.ndarray) -> dict[str, float]:
    """Primary metrics for an advance-probability model."""
    return {
        "n": len(np.asarray(advanced_a)),
        "log_loss": binary_log_loss(p_a, advanced_a),
        "brier": binary_brier(p_a, advanced_a),
        "accuracy": binary_accuracy(p_a, advanced_a),
    }


def compare(models: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Build a comparison table (one row per model) for a 3-way backtest.

    Args:
        models: ``{model_name: (probs[n,3], labels[n])}``.

    Returns:
        Rows with the primary outcome metrics, sorted by ascending log loss
        (best first). ``accuracy`` is included only as a secondary metric.
    """
    rows = []
    for name, (probs, labels) in models.items():
        s = summarize(np.asarray(probs, dtype=float), np.asarray(labels))
        rows.append(
            {
                "model": name,
                "n": s["n"],
                "log_loss": s["log_loss"],
                "brier": s["brier"],
                "rps": s["rps"],
                "ece": s["ece"],
                "accuracy": s["accuracy"],
            }
        )
    rows.sort(key=lambda r: r["log_loss"])
    return rows


def format_comparison(rows: list[dict]) -> str:
    """Render :func:`compare` rows as a fixed-width text table."""
    if not rows:
        return "(no models to compare)"
    header = f"{'model':<26}{'n':>7}{'logloss':>10}{'brier':>9}{'rps':>8}{'ece':>8}{'acc':>8}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r['model']:<26}{r['n']:>7}{r['log_loss']:>10.4f}{r['brier']:>9.4f}"
            f"{r['rps']:>8.4f}{r['ece']:>8.4f}{r['accuracy']:>8.4f}"
        )
    return "\n".join(lines)


def uniform_baseline_logloss() -> float:
    """Log loss of the uniform 3-way forecast (``-ln(1/3) ≈ 1.0986``)."""
    return float(-np.log(1 / 3))


# Re-export the canonical 3-way metrics so callers have one import site.
__all__ = [
    "binary_accuracy",
    "binary_brier",
    "binary_calibration_table",
    "binary_log_loss",
    "binary_summary",
    "brier_score",
    "calibration_table",
    "compare",
    "format_comparison",
    "log_loss",
    "uniform_baseline_logloss",
]
