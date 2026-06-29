"""Validation-only ensemble weight tuning.

Fits ensemble weights to minimize log loss (default) or Brier on a validation
table, then writes them to a **separate** artifact. It never mutates
``config/ensemble.yaml`` — the human-readable defaults stay authoritative; tuned
weights are an opt-in input a future run can choose to load.

Leakage guard: tuning must only ever run on a validation split, never on the
test fold (the optimizer sees the labels). The caller is responsible for passing
validation data; this module does not touch the deployed model or the ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize

from goalsignal.signals.base import OutcomeProbs
from goalsignal.signals.meta_ensemble import EnsembleConfig
from goalsignal.signals.pipeline import ManualInputs, MatchSpec, build_signals
from goalsignal.utils.paths import resolve

_EPS = 1e-12
SIGNAL_ORDER = (
    "historical",
    "market",
    "squad_strength",
    "recent_form",
    "expert",
    "venue_context",
)


@dataclass
class TuningResult:
    """Tuned weights plus the validation metrics that justified them."""

    weights: dict[str, float]
    objective: str
    n: int
    validation_metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    signals_present: list[str]
    coverage: dict[str, float]
    low_coverage: bool
    coverage_warning: str | None


def _row_arrays(signals: dict) -> dict[str, np.ndarray]:
    return {
        name: probs.as_array()
        for name, probs in signals.items()
        if isinstance(probs, OutcomeProbs)
    }


def _blend_table(weight_map: dict[str, float], rows: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Blend each row with a global weight map, renormalizing over present signals."""
    out = np.empty((len(rows), 3))
    for i, avail in enumerate(rows):
        names = [n for n in avail if weight_map.get(n, 0.0) > 0.0]
        if not names:
            names = list(avail)  # fall back to uniform over available
            w = np.ones(len(names))
        else:
            w = np.array([weight_map[n] for n in names])
        w = w / w.sum()
        mix = sum(wi * avail[n] for wi, n in zip(w, names, strict=True))
        out[i] = mix / mix.sum()
    return out


def _objective_value(probs: np.ndarray, labels: np.ndarray, objective: str) -> float:
    if objective == "log_loss":
        return -float(np.log(probs[np.arange(len(labels)), labels] + _EPS).mean())
    if objective == "brier":
        onehot = np.eye(3)[labels]
        return float(((probs - onehot) ** 2).sum(axis=1).mean())
    raise ValueError(f"unknown objective {objective!r}; use 'log_loss' or 'brier'")


def tune_weights(
    specs: list[MatchSpec],
    labels: np.ndarray,
    inputs: ManualInputs,
    *,
    objective: str = "log_loss",
    config: EnsembleConfig | None = None,
) -> TuningResult:
    """Tune simplex weights over the present signals on validation data."""
    cfg = config or inputs.config
    labels = np.asarray(labels)
    rows = [_row_arrays(build_signals(spec, inputs)) for spec in specs]
    present = sorted({n for r in rows for n in r}, key=SIGNAL_ORDER.index)
    if not present:
        raise ValueError("no signals available on the validation table; cannot tune")

    n = len(labels)
    coverage = {name: sum(name in r for r in rows) / max(n, 1) for name in present}
    non_hist_cov = max(
        (c for name, c in coverage.items() if name != "historical"), default=0.0
    )
    low_coverage = n < 100 or non_hist_cov < 0.1
    warning = None
    if low_coverage:
        warning = (
            f"LOW COVERAGE: n={n} validation matches, max non-historical signal "
            f"coverage {non_hist_cov:.1%}. Tuned weights are unreliable; treat as a "
            "smoke test and re-run on a larger validation set with real signals."
        )

    start = np.array([max(cfg.default_weights.get(n, 0.0), 1e-3) for n in present])
    start = start / start.sum()

    def loss(w: np.ndarray) -> float:
        weight_map = dict(zip(present, w, strict=True))
        probs = _blend_table(weight_map, rows)
        return _objective_value(probs, labels, objective)

    res = minimize(
        loss,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(present),
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    w = np.clip(res.x, 0.0, 1.0)
    w = w / w.sum()
    tuned = dict(zip(present, (float(x) for x in w), strict=True))

    tuned_probs = _blend_table(tuned, rows)
    base_probs = _blend_table(
        {n: cfg.default_weights.get(n, 0.0) for n in present}, rows
    )

    def _metrics(probs: np.ndarray) -> dict[str, float]:
        return {
            "log_loss": round(_objective_value(probs, labels, "log_loss"), 4),
            "brier": round(_objective_value(probs, labels, "brier"), 4),
            "accuracy": round(float((probs.argmax(1) == labels).mean()), 4),
        }

    return TuningResult(
        weights=tuned,
        objective=objective,
        n=len(labels),
        validation_metrics=_metrics(tuned_probs),
        baseline_metrics=_metrics(base_probs),
        signals_present=present,
        coverage={k: round(v, 4) for k, v in coverage.items()},
        low_coverage=low_coverage,
        coverage_warning=warning,
    )


def write_tuned_weights(
    result: TuningResult,
    out: str | Path = "artifacts/ensemble/tuned_weights.yaml",
) -> Path:
    """Write tuned weights + justification to a separate artifact (never config)."""
    path = resolve(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tuned_weights": result.weights,
        "objective": result.objective,
        "n_validation_matches": result.n,
        "signals_present": result.signals_present,
        "signal_coverage": result.coverage,
        "low_coverage": result.low_coverage,
        "coverage_warning": result.coverage_warning,
        "validation_metrics_tuned": result.validation_metrics,
        "validation_metrics_default_weights": result.baseline_metrics,
        "note": (
            "Tuned on validation data only. Not applied automatically; "
            "config/ensemble.yaml remains the default. Load explicitly to use."
        ),
    }
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    return path


def write_tuning_report(
    result: TuningResult,
    out: str | Path = "artifacts/ensemble/tuning_report.md",
) -> Path:
    """Write a human-readable tuning report (before/after metrics + warning)."""
    path = resolve(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Ensemble weight tuning report", ""]
    if result.low_coverage:
        lines += [f"> **{result.coverage_warning}**", ""]
    lines += [
        f"- Objective: **{result.objective}**",
        f"- Validation matches: **{result.n}**",
        f"- Signals present: {result.signals_present}",
        "",
        "## Validation metrics (default weights vs tuned)",
        "",
        "| metric | default weights | tuned weights |",
        "| --- | --- | --- |",
    ]
    for metric in ("log_loss", "brier", "accuracy"):
        lines.append(
            f"| {metric} | {result.baseline_metrics[metric]} | "
            f"{result.validation_metrics[metric]} |"
        )
    lines += [
        "",
        "## Tuned weights",
        "",
        "| signal | weight | coverage |",
        "| --- | --- | --- |",
    ]
    for name in result.signals_present:
        lines.append(
            f"| {name} | {result.weights[name]:.3f} | "
            f"{result.coverage.get(name, 0.0):.1%} |"
        )
    lines += [
        "",
        "_config/ensemble.yaml is unchanged; tuned weights are written to a "
        "separate artifact and are never applied automatically._",
        "",
    ]
    path.write_text("\n".join(lines))
    return path
