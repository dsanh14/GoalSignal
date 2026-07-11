"""Chronological comparison of ensemble model versions (leakage-safe).

Compares fixed-weight ensemble versions (baseline_historical, market_only,
squad_form_challenger, llm_adjusted_challenger, final_ensemble) on identical
group-stage matches, reporting outcome-quality metrics plus signal coverage,
missing-signal rate, and high-disagreement-bucket performance.

Weights are **never tuned here** — every version uses its fixed configured
weights, so this is an honest out-of-sample comparison. The input table must
already carry the historical probabilities (from the live model or a prior
backtest's predictions); this module never refits a model, so it introduces no
leakage of its own. Use :mod:`goalsignal.signals.tuning` to tune weights, and
only ever on a validation split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from goalsignal.evaluation import metrics as M
from goalsignal.signals.base import OutcomeProbs
from goalsignal.signals.meta_ensemble import MetaEnsemble
from goalsignal.signals.pipeline import ManualInputs, MatchSpec, build_signals

DEFAULT_VERSIONS = (
    "baseline_historical",
    "market_only",
    "squad_form_challenger",
    "llm_adjusted_challenger",
    "final_ensemble",
)

# Triples that may carry the historical probabilities, in priority order.
_HISTORICAL_TRIPLES = (
    ("historical_home_win", "historical_draw", "historical_away_win"),
    ("ensemble_home", "ensemble_draw", "ensemble_away"),
    ("home_win", "draw", "away_win"),
)


@dataclass
class BacktestTable:
    """Group-stage backtest input: specs (with historical) + integer labels."""

    specs: list[MatchSpec]
    labels: np.ndarray
    smoke: bool  # True when the table is too small/sample-only to be conclusive


def load_backtest_table(path: str | Path) -> BacktestTable:
    """Load a backtest CSV, tolerating several column conventions.

    Requires a ``label`` column (0=home, 1=draw, 2=away) and a historical triple
    (one of ``historical_*``, ``ensemble_*``, or ``home_win/draw/away_win``).
    Team/id columns may be ``team_a``/``team_b`` or ``home_team``/``away_team``,
    ``match_id`` or ``canonical_match_id``.
    """
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("backtest CSV must have a 'label' column (0/1/2)")
    triple = next((t for t in _HISTORICAL_TRIPLES if all(c in df.columns for c in t)), None)
    if triple is None:
        raise ValueError(
            "backtest CSV must carry a historical probability triple "
            f"(one of {[t[0] for t in _HISTORICAL_TRIPLES]})"
        )
    id_col = "match_id" if "match_id" in df.columns else "canonical_match_id"
    a_col = "team_a" if "team_a" in df.columns else "home_team"
    b_col = "team_b" if "team_b" in df.columns else "away_team"

    specs: list[MatchSpec] = []
    labels: list[int] = []
    for i, row in df.iterrows():
        hist = OutcomeProbs(float(row[triple[0]]), float(row[triple[1]]), float(row[triple[2]]))
        specs.append(
            MatchSpec(
                match_id=str(row.get(id_col, f"row{i}")).strip(),
                stage="group",
                team_a=str(row[a_col]).strip(),
                team_b=str(row[b_col]).strip(),
                historical=hist,
            )
        )
        labels.append(int(row["label"]))
    return BacktestTable(specs=specs, labels=np.array(labels), smoke=len(specs) < 100)


def run_ensemble_backtest(
    table: BacktestTable,
    inputs: ManualInputs,
    ensemble: MetaEnsemble,
    versions: tuple[str, ...] = DEFAULT_VERSIONS,
) -> pd.DataFrame:
    """Score each version on the table; return one comparison row per version.

    Rows where a version has no available signal are skipped for that version
    (and counted), so e.g. ``market_only`` is scored only where market data
    exists. Versions with no scorable rows are omitted.
    """
    scored = score_versions(table, inputs, ensemble, versions)
    return pd.DataFrame([s.row for s in scored.values()])


@dataclass
class VersionScore:
    """Per-version blended probabilities, labels, and metric row."""

    probs: np.ndarray
    labels: np.ndarray
    high_disagreement: np.ndarray
    row: dict


def score_versions(
    table: BacktestTable,
    inputs: ManualInputs,
    ensemble: MetaEnsemble,
    versions: tuple[str, ...] = DEFAULT_VERSIONS,
) -> dict[str, VersionScore]:
    """Blend and score each version; shared by the table and the reports."""
    threshold = ensemble.config.disagreement_threshold
    signals_per_row = [build_signals(spec, inputs) for spec in table.specs]

    out: dict[str, VersionScore] = {}
    for version in versions:
        probs_list, labels_list = [], []
        n_available, n_missing, disagreements, skipped = [], [], [], 0
        for signals, label in zip(signals_per_row, table.labels, strict=True):
            try:
                result = ensemble.blend(signals, version=version)
            except ValueError:
                skipped += 1  # no weighted signal available for this row
                continue
            assert isinstance(result.probs, OutcomeProbs)
            probs_list.append(result.probs.as_array())
            labels_list.append(int(label))
            n_available.append(len(result.used_weights))
            n_missing.append(len(result.missing))
            disagreements.append(result.max_pairwise_disagreement)
        if not probs_list:
            continue
        probs = np.vstack(probs_list)
        labels = np.array(labels_list)
        summary = M.summarize(probs, labels)
        high = np.array(disagreements) >= threshold
        row = {
            "version": version,
            "n_scored": len(labels),
            "n_skipped_no_signal": skipped,
            "log_loss": round(summary["log_loss"], 4),
            "brier": round(summary["brier"], 4),
            "rps": round(summary["rps"], 4),
            "ece": round(summary["ece"], 4),
            "accuracy": round(summary["accuracy"], 4),
            "mean_signals_used": round(float(np.mean(n_available)), 3),
            "missing_signal_rate": round(float(np.mean(n_missing)), 3),
            "high_disagreement_count": int(high.sum()),
            "log_loss_high_disagreement": (
                round(M.log_loss(probs[high], labels[high]), 4) if high.any() else None
            ),
            "log_loss_low_disagreement": (
                round(M.log_loss(probs[~high], labels[~high]), 4) if (~high).any() else None
            ),
        }
        out[version] = VersionScore(probs, labels, high, row)
    return out


def coverage_by_signal(table: BacktestTable, inputs: ManualInputs) -> pd.DataFrame:
    """Per-signal coverage over the table, with a trust verdict.

    A signal needs enough coverage to be trusted: ``>=50%`` → trusted,
    ``>0%`` → experimental, ``0%`` → absent.
    """
    from goalsignal.signals.pipeline import SIGNAL_NAMES

    counts = dict.fromkeys(SIGNAL_NAMES, 0)
    for spec in table.specs:
        for name, probs in build_signals(spec, inputs).items():
            if probs is not None:
                counts[name] += 1
    n = max(len(table.specs), 1)
    rows = []
    for name in SIGNAL_NAMES:
        rate = counts[name] / n
        status = "trusted" if rate >= 0.5 else ("experimental" if rate > 0 else "absent")
        rows.append(
            {
                "signal": name,
                "n_covered": counts[name],
                "n_total": len(table.specs),
                "coverage_rate": round(rate, 4),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def calibration_by_version(scored: dict[str, VersionScore]) -> pd.DataFrame:
    """Per-class calibration table for every scored version (long format)."""
    from goalsignal.evaluation.outcome_eval import calibration_table

    rows = []
    for version, s in scored.items():
        for entry in calibration_table(s.probs, s.labels):
            rows.append({"version": version, **entry})
    return pd.DataFrame(rows)


def assess_ensemble(comparison: pd.DataFrame, coverage: pd.DataFrame) -> dict:
    """Compare final_ensemble to baseline_historical without overclaiming."""
    by_version = {r["version"]: r for r in comparison.to_dict("records")}
    base = by_version.get("baseline_historical")
    full = by_version.get("final_ensemble")
    experimental = coverage[coverage["status"] == "experimental"]["signal"].tolist()
    trusted = coverage[(coverage["status"] == "trusted") & (coverage["signal"] != "historical")][
        "signal"
    ].tolist()

    assessment = {
        "have_both": base is not None and full is not None,
        "trusted_non_historical_signals": trusted,
        "experimental_signals": experimental,
    }
    if base is None or full is None:
        assessment["verdict"] = "insufficient: baseline or final ensemble not scored"
        assessment["recommendation"] = "keep final_ensemble opt-in"
        return assessment

    # Did anything beyond the historical signal actually contribute?
    non_hist_coverage = coverage[coverage["signal"] != "historical"]["coverage_rate"].max()
    assessment.update(
        {
            "logloss_baseline": base["log_loss"],
            "logloss_final": full["log_loss"],
            "logloss_delta": round(full["log_loss"] - base["log_loss"], 4),
            "logloss_better": full["log_loss"] < base["log_loss"],
            "brier_baseline": base["brier"],
            "brier_final": full["brier"],
            "brier_delta": round(full["brier"] - base["brier"], 4),
            "brier_better": full["brier"] < base["brier"],
            "ece_baseline": base["ece"],
            "ece_final": full["ece"],
            "calibration_better": full["ece"] < base["ece"],
            "max_non_historical_coverage": round(float(non_hist_coverage), 4),
            "high_disagreement_worse": (
                full["log_loss_high_disagreement"] is not None
                and full["log_loss_low_disagreement"] is not None
                and full["log_loss_high_disagreement"] > full["log_loss_low_disagreement"]
            ),
        }
    )
    # Honest verdict: a tiny delta on near-zero coverage is not evidence.
    if non_hist_coverage < 0.05:
        assessment["verdict"] = (
            "INSUFFICIENT DATA: non-historical signal coverage is near zero, so "
            "final_ensemble is effectively the historical model. No conclusion."
        )
        assessment["recommendation"] = "keep final_ensemble opt-in"
    elif assessment["logloss_better"] and assessment["brier_better"]:
        assessment["verdict"] = "final_ensemble improves log loss and Brier on this data"
        assessment["recommendation"] = (
            "promising; validate on a larger out-of-sample set before promotion"
        )
    else:
        assessment["verdict"] = "final_ensemble does not clearly beat baseline_historical"
        assessment["recommendation"] = "keep final_ensemble opt-in"
    return assessment


def _summary_markdown(
    comparison: pd.DataFrame, coverage: pd.DataFrame, assessment: dict, smoke: bool
) -> str:
    lines = ["# Ensemble backtest summary", ""]
    if smoke:
        lines += [
            "> **SMOKE TEST** — small/sample data. Results are illustrative only "
            "and must not be read as evidence.",
            "",
        ]
    lines += ["## Is final_ensemble better than baseline_historical?", ""]
    if not assessment.get("have_both"):
        lines += [f"- {assessment['verdict']}", ""]
    else:
        a = assessment
        lines += [
            f"- **Log loss:** baseline {a['logloss_baseline']} vs final "
            f"{a['logloss_final']} (delta {a['logloss_delta']}) — "
            f"{'better' if a['logloss_better'] else 'not better'}",
            f"- **Brier:** baseline {a['brier_baseline']} vs final {a['brier_final']} "
            f"(delta {a['brier_delta']}) — {'better' if a['brier_better'] else 'not better'}",
            f"- **Calibration (ECE):** baseline {a['ece_baseline']} vs final "
            f"{a['ece_final']} — {'better' if a['calibration_better'] else 'not better'}",
            f"- **Max non-historical signal coverage:** {a['max_non_historical_coverage']:.1%}",
            f"- **High-disagreement matches worse?** "
            f"{'yes' if a['high_disagreement_worse'] else 'no'}",
            "",
            f"**Verdict:** {a['verdict']}",
            "",
            f"**Recommendation:** {a['recommendation']}",
            "",
        ]
    lines += ["## Signal coverage", ""]
    trusted = assessment.get("trusted_non_historical_signals", [])
    experimental = assessment.get("experimental_signals", [])
    lines += [
        f"- Trusted (>=50% coverage): {trusted or 'none'}",
        f"- Experimental (>0% but sparse): {experimental or 'none'}",
        "",
        "## Comparison table",
        "",
        _df_to_md(comparison),
        "",
    ]
    return "\n".join(lines)


def _df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra deps)."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *body])


def write_reports(
    comparison: pd.DataFrame,
    coverage: pd.DataFrame,
    calibration: pd.DataFrame,
    assessment: dict,
    smoke: bool,
    out_dir: str | Path = "artifacts/ensemble",
) -> dict[str, Path]:
    """Write all four backtest artifacts; return their paths."""
    from goalsignal.utils.paths import resolve

    base = resolve(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    comp = comparison.copy()
    comp.insert(0, "smoke_test", smoke)
    paths = {
        "comparison": base / "backtest_comparison.csv",
        "summary": base / "backtest_summary.md",
        "calibration": base / "calibration_by_version.csv",
        "coverage": base / "coverage_by_signal.csv",
    }
    comp.to_csv(paths["comparison"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    paths["summary"].write_text(_summary_markdown(comparison, coverage, assessment, smoke))
    return paths


# Ablation: each entry is historical anchored plus one signal group.
ABLATIONS: dict[str, list[str]] = {
    "historical_only": ["historical"],
    "historical+market": ["historical", "market"],
    "historical+squad_form": ["historical", "squad_strength", "recent_form"],
    "historical+venue": ["historical", "venue_context"],
    "historical+match_context": ["historical", "match_context"],
    "historical+expert": ["historical", "expert"],
    "full_ensemble": [
        "historical",
        "market",
        "squad_strength",
        "recent_form",
        "expert",
        "venue_context",
        "match_context",
    ],
}


def run_ablation(
    table: BacktestTable, inputs: ManualInputs, ensemble: MetaEnsemble
) -> pd.DataFrame:
    """Score historical-only vs historical+each-signal vs the full ensemble.

    Each ablation uses the default configured weights restricted to its signal
    subset (renormalized over what is available per match), so the deltas
    reflect each signal's marginal contribution at its product weight.
    """
    dw = ensemble.config.default_weights
    signals_per_row = [build_signals(spec, inputs) for spec in table.specs]
    rows = []
    for name, subset in ABLATIONS.items():
        wmap = {s: dw.get(s, 0.0) for s in subset}
        wmap["historical"] = dw.get("historical", 0.35) or 0.35  # always anchored
        probs_list, labels_list, n_used = [], [], []
        for signals, label in zip(signals_per_row, table.labels, strict=True):
            try:
                res = ensemble.blend(signals, weights=wmap)
            except ValueError:
                continue
            assert isinstance(res.probs, OutcomeProbs)
            probs_list.append(res.probs.as_array())
            labels_list.append(int(label))
            n_used.append(len(res.used_weights))
        if not probs_list:
            continue
        probs = np.vstack(probs_list)
        labels = np.array(labels_list)
        s = M.summarize(probs, labels)
        rows.append(
            {
                "ablation": name,
                "signals": "+".join(subset),
                "n_scored": len(labels),
                "log_loss": round(s["log_loss"], 4),
                "brier": round(s["brier"], 4),
                "ece": round(s["ece"], 4),
                "accuracy": round(s["accuracy"], 4),
                "mean_signals_used": round(float(np.mean(n_used)), 3),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and (df["ablation"] == "historical_only").any():
        base = df.loc[df["ablation"] == "historical_only", "log_loss"].iloc[0]
        df["logloss_delta_vs_historical"] = (df["log_loss"] - base).round(4)
        df["improves"] = df["logloss_delta_vs_historical"] < 0
    return df


def write_ablation(
    df: pd.DataFrame, smoke: bool, out_dir: str | Path = "artifacts/ensemble"
) -> dict[str, Path]:
    """Write the ablation comparison CSV and a short Markdown summary."""
    from goalsignal.utils.paths import resolve

    base = resolve(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.insert(0, "smoke_test", smoke)
    csv_path = base / "ablation_comparison.csv"
    md_path = base / "ablation_summary.md"
    out.to_csv(csv_path, index=False)

    lines = ["# Ensemble ablation summary", ""]
    if smoke:
        lines += ["> **SMOKE TEST** — sample data; deltas are not evidence.", ""]
    improving = df[df.get("improves", False)]["signals"].tolist() if "improves" in df else []
    lines += [
        "Each row is the historical model plus one signal group "
        "(negative delta = improvement over historical-only).",
        "",
        f"- Signals that improved log loss here: {improving or 'none'}",
        "",
        _df_to_md(df),
        "",
    ]
    md_path.write_text("\n".join(lines))
    return {"comparison": csv_path, "summary": md_path}


# Backwards-compatible single-file writer (kept for existing callers).
def write_comparison(
    df: pd.DataFrame,
    smoke: bool,
    out: str | Path = "artifacts/ensemble/backtest_comparison.csv",
) -> Path:
    """Write just the comparison table (with a clear smoke-test flag column)."""
    from goalsignal.utils.paths import resolve

    path = resolve(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df.insert(0, "smoke_test", smoke)
    df.to_csv(path, index=False)
    return path
