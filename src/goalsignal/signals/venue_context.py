"""Venue, travel, rest, and host-context signal for World Cup 2026.

2026 is co-hosted by the USA, Mexico, and Canada, so host advantage, altitude
and heat, travel across a continent, and rest differences all matter. This
module reads a per-match context CSV and turns the available fields into a
small Elo-like advantage for team A, then through the Davidson map into an
outcome signal.

Input format — a CSV keyed by ``match_id`` with any subset of (all optional,
all from team A's perspective; ``*_a`` / ``*_b`` are the two teams)::

    match_id,
    host_boost,          # signed points if team A (or B, negative) is a host
    crowd_advantage,     # signed regional/crowd advantage for team A, in points
    travel_km_a, travel_km_b,
    rest_days_a, rest_days_b,
    heat_disadvantage_a, # points team A loses to heat/altitude (e.g. Mexico City)
    timezone_shift_a, timezone_shift_b,  # |hours| of time-zone change pre-match

Each present field contributes through a configurable coefficient; absent
fields contribute nothing (no zero-fill of unknown context).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from goalsignal.signals.base import OutcomeProbs, davidson_outcome


@dataclass(frozen=True)
class VenueCoefficients:
    """Points-per-unit conversion for each context component."""

    travel_per_1000km: float = 8.0  # advantage to the *less*-travelled side
    rest_per_day: float = 6.0  # advantage to the *more*-rested side
    timezone_per_hour: float = 3.0  # advantage to the *less* time-shifted side


# Numeric context fields (everything except the identity/label columns).
_NUMERIC_FIELDS = (
    "host_boost",
    "crowd_advantage",
    "travel_km_a",
    "travel_km_b",
    "rest_days_a",
    "rest_days_b",
    "heat_disadvantage_a",
    "timezone_shift_a",
    "timezone_shift_b",
)


@dataclass(frozen=True)
class VenueContext:
    """Optional per-match context fields (any may be ``None``)."""

    match_id: str
    host_boost: float | None = None
    crowd_advantage: float | None = None
    travel_km_a: float | None = None
    travel_km_b: float | None = None
    rest_days_a: float | None = None
    rest_days_b: float | None = None
    heat_disadvantage_a: float | None = None
    timezone_shift_a: float | None = None
    timezone_shift_b: float | None = None
    team_a: str | None = None  # team names / stage enable dynamic matching
    team_b: str | None = None
    stage: str | None = None

    def advantage_points(self, coeffs: VenueCoefficients) -> float:
        """Signed Elo-like advantage for team A from the present fields."""
        pts = 0.0
        if self.host_boost is not None:
            pts += self.host_boost
        if self.crowd_advantage is not None:
            pts += self.crowd_advantage
        if self.heat_disadvantage_a is not None:
            pts -= self.heat_disadvantage_a
        if self.travel_km_a is not None and self.travel_km_b is not None:
            pts += coeffs.travel_per_1000km * (self.travel_km_b - self.travel_km_a) / 1000.0
        if self.rest_days_a is not None and self.rest_days_b is not None:
            pts += coeffs.rest_per_day * (self.rest_days_a - self.rest_days_b)
        if self.timezone_shift_a is not None and self.timezone_shift_b is not None:
            pts += coeffs.timezone_per_hour * (self.timezone_shift_b - self.timezone_shift_a)
        return pts

    def has_any(self) -> bool:
        """True if at least one *numeric* context field is populated."""
        return any(getattr(self, name) is not None for name in _NUMERIC_FIELDS)


def load_venue_context(
    path: str | Path, *, require: bool = False
) -> dict[str, VenueContext]:
    """Load a venue-context CSV into ``{key: VenueContext}``.

    Keyed by ``match_id`` when present, else by a synthetic team-pair key so the
    row can attach to dynamic knockout pairings.
    """
    p = Path(path)
    if not p.exists():
        if require:
            raise FileNotFoundError(f"venue context file not found: {p}")
        return {}
    df = pd.read_csv(p)
    if "match_id" not in df.columns and not {"team_a", "team_b"} <= set(df.columns):
        raise ValueError(
            "venue context CSV needs a 'match_id' column or 'team_a'/'team_b' columns"
        )
    out: dict[str, VenueContext] = {}
    for _, row in df.iterrows():
        match_id = str(row["match_id"]).strip() if "match_id" in df.columns else ""
        name_a = str(row["team_a"]).strip() if "team_a" in df.columns and pd.notna(
            row["team_a"]) else None
        name_b = str(row["team_b"]).strip() if "team_b" in df.columns and pd.notna(
            row["team_b"]) else None
        key = match_id or (f"pair::{name_a}|{name_b}" if name_a and name_b else "")
        if not key:
            continue
        kwargs: dict[str, float] = {}
        for col in _NUMERIC_FIELDS:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != "":
                kwargs[col] = float(row[col])
        stage = str(row["stage"]).strip() if "stage" in df.columns and pd.notna(
            row.get("stage")) else None
        out[key] = VenueContext(
            match_id=match_id or key, team_a=name_a, team_b=name_b, stage=stage, **kwargs
        )
    return out


def venue_signal(
    contexts: dict[str, VenueContext],
    match_id: str,
    *,
    coeffs: VenueCoefficients | None = None,
    scale: float = 400.0,
    nu: float = 1.0,
) -> OutcomeProbs | None:
    """Outcome signal from venue/travel/rest context, or ``None``.

    Returns ``None`` when no context row exists or the row has no populated
    fields — venue context is an *adjustment*, so with nothing to adjust the
    ensemble simply renormalizes without it.
    """
    ctx = contexts.get(match_id)
    if ctx is None or not ctx.has_any():
        return None
    return davidson_outcome(
        ctx.advantage_points(coeffs or VenueCoefficients()), scale=scale, nu=nu
    )
