"""LLM / expert-judgment signal — a controlled, auditable input.

An LLM (or a human analyst) must never silently overwrite the statistical
forecast. Instead, structured predictions are entered into a CSV, validated,
and exposed as *one more signal* the meta-ensemble blends with a bounded
weight, plus a disagreement detector and an explanation (``reasoning``) field.

Input format — a CSV keyed by ``match_id`` (multiple rows per match allowed,
one per source model) with columns::

    match_id, source_model,
    team_a_win_prob, draw_prob, team_b_win_prob,   # group-stage triple
    team_a_advance_prob, team_b_advance_prob,       # knockout pair
    confidence,                                     # 0-1 weight for consensus
    reasoning                                       # free-text explanation

A row may carry only the group triple, only the knockout pair, or both. Triples
and pairs are validated to sum to ~1 (within tolerance) and then renormalized;
rows that are too far off are skipped (and reported via ``on_error``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from goalsignal.signals.base import AdvanceProbs, OutcomeProbs, disagreement

_SUM_TOLERANCE = 0.05


def _maybe_float(value) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    return float(s)


@dataclass(frozen=True)
class ExpertPrediction:
    """One source's structured prediction for a single match."""

    match_id: str
    source_model: str
    outcome: OutcomeProbs | None
    advance: AdvanceProbs | None
    confidence: float
    reasoning: str
    team_a: str | None = None  # team names enable dynamic team-pair matching
    team_b: str | None = None


def _validate_triple(values: list[float], match_id: str) -> OutcomeProbs:
    total = sum(values)
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValueError(
            f"match {match_id}: outcome probs sum to {total:.3f}, expected ~1.0"
        )
    return OutcomeProbs(*values)  # normalizes


def _validate_pair(values: list[float], match_id: str) -> AdvanceProbs:
    total = sum(values)
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise ValueError(
            f"match {match_id}: advance probs sum to {total:.3f}, expected ~1.0"
        )
    return AdvanceProbs(*values)  # normalizes


def load_expert_predictions(
    path: str | Path,
    *,
    require: bool = False,
    on_error: list[str] | None = None,
) -> dict[str, list[ExpertPrediction]]:
    """Load expert predictions into ``{match_id: [ExpertPrediction, ...]}``."""
    p = Path(path)
    if not p.exists():
        if require:
            raise FileNotFoundError(f"expert predictions file not found: {p}")
        return {}
    df = pd.read_csv(p, dtype=str).fillna("")
    if "match_id" not in df.columns and not {"team_a", "team_b"} <= set(df.columns):
        raise ValueError(
            "expert predictions CSV needs a 'match_id' column or 'team_a'/'team_b' columns"
        )

    out: dict[str, list[ExpertPrediction]] = {}
    for i, row in df.iterrows():
        match_id = str(row.get("match_id", "")).strip()
        name_a = str(row.get("team_a", "")).strip() or None
        name_b = str(row.get("team_b", "")).strip() or None
        key = match_id or (f"pair::{name_a}|{name_b}" if name_a and name_b else "")
        if not key:
            continue
        try:
            triple = [
                _maybe_float(row.get(c, ""))
                for c in ("team_a_win_prob", "draw_prob", "team_b_win_prob")
            ]
            pair = [
                _maybe_float(row.get(c, ""))
                for c in ("team_a_advance_prob", "team_b_advance_prob")
            ]
            outcome = (
                _validate_triple([float(v) for v in triple], match_id)
                if all(v is not None for v in triple)
                else None
            )
            advance = (
                _validate_pair([float(v) for v in pair], match_id)
                if all(v is not None for v in pair)
                else None
            )
            if outcome is None and advance is None:
                raise ValueError("no complete outcome or advance probabilities")
            conf = _maybe_float(row.get("confidence", ""))
            pred = ExpertPrediction(
                match_id=match_id or key,
                source_model=str(row.get("source_model", "")).strip() or "expert",
                outcome=outcome,
                advance=advance,
                confidence=1.0 if conf is None else float(np.clip(conf, 0.0, 1.0)),
                reasoning=str(row.get("reasoning", "")).strip(),
                team_a=name_a,
                team_b=name_b,
            )
        except (ValueError, TypeError) as exc:
            if on_error is not None:
                on_error.append(f"row {i} (match_id={match_id!r}): {exc}")
            continue
        out.setdefault(key, []).append(pred)
    return out


def expert_consensus(
    predictions: list[ExpertPrediction],
    *,
    knockout: bool = False,
) -> OutcomeProbs | AdvanceProbs | None:
    """Confidence-weighted consensus across sources for one match, or ``None``.

    Sources lacking the requested mode (group vs knockout) are ignored. Returns
    ``None`` if no source supplied the requested mode.
    """
    arrays = []
    weights = []
    for pred in predictions:
        probs = pred.advance if knockout else pred.outcome
        if probs is None:
            continue
        arrays.append(probs.as_array())
        weights.append(max(pred.confidence, 1e-6))
    if not arrays:
        return None
    blended = np.average(np.vstack(arrays), axis=0, weights=weights)
    return AdvanceProbs.from_array(blended) if knockout else OutcomeProbs.from_array(blended)


def expert_signal(
    predictions_by_match: dict[str, list[ExpertPrediction]],
    match_id: str,
    *,
    knockout: bool = False,
) -> OutcomeProbs | AdvanceProbs | None:
    """Consensus expert signal for one match, or ``None`` if absent."""
    preds = predictions_by_match.get(match_id)
    if not preds:
        return None
    return expert_consensus(preds, knockout=knockout)


def expert_disagreement(
    predictions_by_match: dict[str, list[ExpertPrediction]],
    match_id: str,
    reference: OutcomeProbs,
) -> float | None:
    """Total-variation gap between the expert consensus and a reference (model).

    Returns ``None`` if there is no group-stage expert consensus for the match.
    Useful as an explanation/triage feature: large gaps flag matches where the
    statistical model and human/LLM judgment diverge.
    """
    consensus = expert_signal(predictions_by_match, match_id, knockout=False)
    if consensus is None:
        return None
    assert isinstance(consensus, OutcomeProbs)
    return disagreement(consensus, reference)
