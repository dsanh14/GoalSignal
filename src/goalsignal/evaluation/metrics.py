"""Outcome-forecast evaluation metrics.

All metrics take probs of shape (n, 3) ordered [home, draw, away] and integer
labels in {0, 1, 2}. Log loss is the primary outcome metric; Brier the primary
calibration metric; RPS respects the W/D/L ordering.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def log_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    return -float(np.log(probs[np.arange(len(labels)), labels] + _EPS).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.eye(3)[labels]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def ranked_probability_score(probs: np.ndarray, labels: np.ndarray) -> float:
    cum_p = np.cumsum(probs, axis=1)
    cum_y = np.cumsum(np.eye(3)[labels], axis=1)
    return float(((cum_p - cum_y) ** 2)[:, :2].sum(axis=1).mean() / 2.0)


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """ECE of the max-probability prediction, sample-weighted over bins."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Per-bin confidence vs accuracy with sample counts (for honest plots)."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        rows.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "mean_confidence": float(conf[mask].mean()) if mask.any() else None,
                "empirical_accuracy": float(correct[mask].mean()) if mask.any() else None,
            }
        )
    return rows


def summarize(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "n": len(labels),
        "log_loss": log_loss(probs, labels),
        "brier": brier_score(probs, labels),
        "rps": ranked_probability_score(probs, labels),
        "accuracy": accuracy(probs, labels),
        "ece": expected_calibration_error(probs, labels),
    }


def block_bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    blocks: np.ndarray,
    metric=log_loss,
    n_resamples: int = 1000,
    seed: int = 7,
    level: float = 0.90,
) -> dict[str, float]:
    """Bootstrap CI resampling whole blocks (e.g. years) to respect dependence."""
    rng = np.random.default_rng(seed)
    unique = np.unique(blocks)
    index_by_block = {b: np.flatnonzero(blocks == b) for b in unique}
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_block[b] for b in chosen])
        stats[i] = metric(probs[idx], labels[idx])
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "point": float(metric(probs, labels)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "level": level,
        "n_resamples": n_resamples,
        "resampling_unit": "block",
        "seed": seed,
    }
