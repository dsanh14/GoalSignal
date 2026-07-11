"""Bounded, leakage-safe match-context adjustment signal.

This module turns late-breaking football information into a small adjustment
to an existing forecast.  It deliberately does not create a forecast from
context alone: the adjustment is anchored to the historical/market estimate,
then applied on the log-odds scale with component and total caps.

CSV rows are keyed by match id or team pair.  Every row must include
``available_at`` and ``kickoff_at``; information available at or after kickoff
is rejected.  Numeric columns are signed Elo-like points from team A's
perspective (positive favours A, negative favours B)::

    lineup_edge, availability_edge, goalkeeper_edge, fatigue_edge,
    match_quality_edge, tactical_edge, climate_edge

The columns are intentionally decomposed so audits can identify why a
prediction moved and so future backtests can ablate each evidence family.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from goalsignal.signals.base import AdvanceProbs, OutcomeProbs

CONTEXT_FIELDS = (
    "lineup_edge",
    "availability_edge",
    "goalkeeper_edge",
    "fatigue_edge",
    "match_quality_edge",
    "tactical_edge",
    "climate_edge",
)


@dataclass(frozen=True)
class MatchContextParams:
    """Caps and conversion scale for context evidence."""

    points_scale: float = 400.0
    component_cap: float = 25.0
    total_cap: float = 60.0
    max_probability_shift: float = 0.12

    @classmethod
    def from_mapping(cls, raw: dict | None) -> MatchContextParams:
        raw = raw or {}
        return cls(**{k: float(raw[k]) for k in cls.__dataclass_fields__ if k in raw})


@dataclass(frozen=True)
class MatchContext:
    """One timestamped, auditable bundle of pre-match evidence."""

    match_id: str
    available_at: datetime
    kickoff_at: datetime
    team_a: str | None = None
    team_b: str | None = None
    stage: str | None = None
    source: str = "manual"
    reason: str = ""
    lineup_edge: float | None = None
    availability_edge: float | None = None
    goalkeeper_edge: float | None = None
    fatigue_edge: float | None = None
    match_quality_edge: float | None = None
    tactical_edge: float | None = None
    climate_edge: float | None = None

    def raw_points(self, params: MatchContextParams) -> float:
        return sum(
            np.clip(float(value), -params.component_cap, params.component_cap)
            for name in CONTEXT_FIELDS
            if (value := getattr(self, name)) is not None
        )

    def advantage_points(self, params: MatchContextParams) -> float:
        return float(np.clip(self.raw_points(params), -params.total_cap, params.total_cap))

    def has_evidence(self) -> bool:
        return any(getattr(self, name) is not None for name in CONTEXT_FIELDS)


def _parse_time(value: object, *, label: str, row_number: int) -> datetime:
    try:
        timestamp = pd.Timestamp(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"match_context row {row_number}: invalid {label}") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"match_context row {row_number}: {label} must include timezone")
    return timestamp.to_pydatetime()


def load_match_context(path: str | Path, *, require: bool = False) -> dict[str, MatchContext]:
    """Load strict context rows, rejecting duplicates and future information."""
    p = Path(path)
    if not p.exists():
        if require:
            raise FileNotFoundError(f"match context file not found: {p}")
        return {}
    frame = pd.read_csv(p, dtype=str).fillna("")
    required = {"available_at", "kickoff_at"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"match_context CSV missing columns: {sorted(missing)}")
    if "match_id" not in frame.columns and not {"team_a", "team_b"} <= set(frame.columns):
        raise ValueError("match_context CSV needs match_id or team_a/team_b columns")

    contexts: dict[str, MatchContext] = {}
    for idx, row in frame.iterrows():
        row_number = idx + 2
        match_id = str(row.get("match_id", "")).strip()
        team_a = str(row.get("team_a", "")).strip() or None
        team_b = str(row.get("team_b", "")).strip() or None
        key = match_id or (f"pair::{team_a}|{team_b}" if team_a and team_b else "")
        if not key:
            raise ValueError(f"match_context row {row_number}: missing match identity")
        if key in contexts:
            raise ValueError(f"match_context row {row_number}: duplicate key {key!r}")
        available = _parse_time(row["available_at"], label="available_at", row_number=row_number)
        kickoff = _parse_time(row["kickoff_at"], label="kickoff_at", row_number=row_number)
        if available >= kickoff:
            raise ValueError(
                f"match_context row {row_number}: available_at must be before kickoff_at"
            )
        values: dict[str, float] = {}
        for name in CONTEXT_FIELDS:
            raw = str(row.get(name, "")).strip()
            if raw:
                try:
                    values[name] = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"match_context row {row_number}: {name} must be numeric"
                    ) from exc
                if not np.isfinite(values[name]):
                    raise ValueError(f"match_context row {row_number}: {name} must be finite")
        context = MatchContext(
            match_id=match_id or key,
            available_at=available,
            kickoff_at=kickoff,
            team_a=team_a,
            team_b=team_b,
            stage=str(row.get("stage", "")).strip() or None,
            source=str(row.get("source", "")).strip() or "manual",
            reason=str(row.get("reason", "")).strip(),
            **values,
        )
        if context.has_evidence() and not context.reason:
            raise ValueError(
                f"match_context row {row_number}: reason is required when evidence is present"
            )
        contexts[key] = context
    return contexts


def _bounded_shift(base: float, shifted: float, cap: float) -> float:
    return float(np.clip(shifted, max(0.0, base - cap), min(1.0, base + cap)))


def adjust_outcome(base: OutcomeProbs, points: float, params: MatchContextParams) -> OutcomeProbs:
    """Shift A-vs-B odds while preserving the base draw probability."""
    decisive = base.home_win + base.away_win
    if decisive <= 0.0 or points == 0.0:
        return base
    share = base.home_win / decisive
    logit = np.log(np.clip(share, 1e-9, 1 - 1e-9) / np.clip(1 - share, 1e-9, 1))
    shifted_share = 1.0 / (1.0 + np.exp(-(logit + np.log(10.0) * points / params.points_scale)))
    candidate_home = decisive * shifted_share
    home = _bounded_shift(base.home_win, candidate_home, params.max_probability_shift)
    return OutcomeProbs(home, base.draw, decisive - home)


def adjust_advance(base: AdvanceProbs, points: float, params: MatchContextParams) -> AdvanceProbs:
    """Apply the bounded edge to knockout advancement log odds."""
    if points == 0.0:
        return base
    p = np.clip(base.team_a_advances, 1e-9, 1 - 1e-9)
    logit = np.log(p / (1.0 - p))
    candidate = 1.0 / (1.0 + np.exp(-(logit + np.log(10.0) * points / params.points_scale)))
    a = _bounded_shift(base.team_a_advances, candidate, params.max_probability_shift)
    return AdvanceProbs(a, 1.0 - a)
