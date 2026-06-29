"""Construct an ensemble-driven simulation source from the live model.

Bridges the deployed :class:`~goalsignal.live.LiveModel` and the manual signal
files to an :class:`~goalsignal.tournament.ensemble_adapter.EnsembleGoalAdapter`
the tournament simulator can consume, and summarizes the per-matchup provenance
(coverage, missing signals, high-disagreement ties) for the run report.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from goalsignal.signals.api import EnsemblePredictor
from goalsignal.signals.historical_adapter import LiveModelHistorical
from goalsignal.signals.meta_ensemble import MetaEnsemble, load_ensemble_config
from goalsignal.signals.pipeline import MatchSpec, load_manual_inputs
from goalsignal.tournament.ensemble_adapter import EnsembleGoalAdapter
from goalsignal.tournament.model_adapter import RatingsGoalAdapter


def build_ensemble_adapter(
    live,
    *,
    version: str = "final_ensemble",
    manual_dir: str | Path = "data/manual",
) -> tuple[EnsembleGoalAdapter, EnsemblePredictor]:
    """Build an ensemble simulation adapter backed by the live historical model."""
    config = load_ensemble_config()
    inputs = load_manual_inputs(manual_dir, config)
    ensemble = MetaEnsemble(config)
    predictor = EnsemblePredictor(inputs, ensemble, LiveModelHistorical(live))
    base = RatingsGoalAdapter(live.ratings, live.goal_model)

    def blend_fn(home: str, away: str, neutral: bool, knockout: bool):
        spec = MatchSpec(
            match_id=f"{home}|{away}",
            stage="knockout" if knockout else "group",
            team_a=home,
            team_b=away,
            neutral=neutral,
        )
        prediction = predictor.predict(spec, version=version)
        return prediction.probs, prediction

    return EnsembleGoalAdapter(base, blend_fn), predictor


def ensemble_provenance_summary(adapter: EnsembleGoalAdapter, top_n: int = 10) -> dict:
    """Summarize per-matchup ensemble provenance for the simulation report."""
    preds = list(adapter.provenance.values())
    n = len(preds)
    missing_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    used_counter: Counter[str] = Counter()
    flagged = []
    for p in preds:
        missing_counter.update(p.missing)
        source_counter.update([p.historical_source])
        used_counter.update(p.components)
        if p.flagged:
            flagged.append((p.team_a, p.team_b, round(p.disagreement, 4)))
    flagged.sort(key=lambda t: -t[2])
    by_disagreement = sorted(preds, key=lambda p: -p.disagreement)[:top_n]
    return {
        "matchups_evaluated": n,
        "flagged_count": len(flagged),
        "flagged_matchups": flagged[:top_n],
        "historical_source_counts": dict(source_counter),
        "signal_usage_counts": dict(used_counter),
        "missing_signal_counts": dict(missing_counter),
        "highest_disagreement": [
            (p.team_a, p.team_b, round(p.disagreement, 4)) for p in by_disagreement
        ],
    }


def format_provenance_summary(summary: dict) -> str:
    """Render :func:`ensemble_provenance_summary` as readable text."""
    lines = [
        f"Ensemble matchups evaluated: {summary['matchups_evaluated']}",
        f"  historical source: {summary['historical_source_counts']}",
        f"  signal usage:      {summary['signal_usage_counts']}",
        f"  missing signals:   {summary['missing_signal_counts'] or 'none'}",
        f"  high-disagreement matchups (flagged): {summary['flagged_count']}",
    ]
    for home, away, gap in summary["flagged_matchups"]:
        lines.append(f"    FLAG {home} vs {away}  TVD={gap}")
    return "\n".join(lines)
