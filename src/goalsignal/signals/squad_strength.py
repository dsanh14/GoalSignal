"""Squad-strength signal from a manually maintained file.

A simple, file-first approach: drop a CSV of per-team squad indicators into
``data/manual/`` and this module turns it into an outcome signal. Every field is
optional — the strength score is a weighted average of whichever standardized
indicators are present for a team, so a sparse file still produces a usable
(if weaker) signal.

Input format — a CSV keyed by ``team`` with any subset of these columns::

    team,
    total_squad_value,        # market value of the whole squad
    starting_xi_value,        # market value of the projected XI
    top5_league_minutes,      # minutes played in the top-5 leagues
    champions_league_minutes, # UCL minutes
    club_minutes_30d,         # recent club minutes (last 30 days)
    club_minutes_90d,         # recent club minutes (last 90 days)
    keeper_strength,          # 0-100 subjective/derived keeper rating
    attacking_depth,          # 0-100
    defensive_depth,          # 0-100
    missing_stars,            # count of unavailable key players (penalty)
    suspensions,              # count of suspended players (penalty)
    avg_age                   # informational; not scored by default

Direction and weight of each indicator are configurable; counts such as
``missing_stars`` and ``suspensions`` are penalties (higher = weaker).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd

from goalsignal.signals.base import OutcomeProbs, davidson_outcome

# (column, weight, higher_is_better). Weights are relative; only the columns
# present for a team contribute, and the contributing weights are renormalized.
DEFAULT_INDICATOR_WEIGHTS: tuple[tuple[str, float, bool], ...] = (
    ("total_squad_value", 1.0, True),
    ("starting_xi_value", 1.0, True),
    ("top5_league_minutes", 0.7, True),
    ("champions_league_minutes", 0.6, True),
    ("club_minutes_30d", 0.3, True),
    ("club_minutes_90d", 0.4, True),
    ("keeper_strength", 0.5, True),
    ("attacking_depth", 0.6, True),
    ("defensive_depth", 0.6, True),
    ("missing_stars", 0.8, False),
    ("suspensions", 0.5, False),
)

_SCORED_COLUMNS = tuple(name for name, _, _ in DEFAULT_INDICATOR_WEIGHTS)


@dataclass(frozen=True)
class SquadStrength:
    """Optional per-team squad indicators (any field may be ``None``)."""

    team: str
    total_squad_value: float | None = None
    starting_xi_value: float | None = None
    top5_league_minutes: float | None = None
    champions_league_minutes: float | None = None
    club_minutes_30d: float | None = None
    club_minutes_90d: float | None = None
    keeper_strength: float | None = None
    attacking_depth: float | None = None
    defensive_depth: float | None = None
    missing_stars: float | None = None
    suspensions: float | None = None
    avg_age: float | None = None

    def available(self) -> dict[str, float]:
        """Return the scored indicators that are present (excludes ``avg_age``)."""
        out: dict[str, float] = {}
        for col in _SCORED_COLUMNS:
            val = getattr(self, col)
            if val is not None:
                out[col] = float(val)
        return out


@dataclass
class SquadStrengthTable:
    """A loaded squad table with population statistics for standardization."""

    teams: dict[str, SquadStrength]
    weights: tuple[tuple[str, float, bool], ...] = field(
        default=DEFAULT_INDICATOR_WEIGHTS, repr=False
    )
    _mean: dict[str, float] = field(default_factory=dict, repr=False)
    _std: dict[str, float] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for col in _SCORED_COLUMNS:
            vals = [
                getattr(s, col) for s in self.teams.values() if getattr(s, col) is not None
            ]
            if len(vals) >= 2:
                self._mean[col] = float(np.mean(vals))
                std = float(np.std(vals))
                self._std[col] = std if std > 1e-9 else 0.0

    def strength_score(self, team: str) -> float | None:
        """Weighted average of standardized indicators (z-units), or ``None``.

        ``None`` is returned when the team is absent or none of its present
        indicators have a population standard deviation to standardize against.
        """
        squad = self.teams.get(team)
        if squad is None:
            return None
        num = 0.0
        wsum = 0.0
        signs = {col: good for col, _, good in self.weights}
        wmap = {col: w for col, w, _ in self.weights}
        for col, val in squad.available().items():
            std = self._std.get(col, 0.0)
            if std <= 0.0:
                continue
            z = (val - self._mean[col]) / std
            if not signs.get(col, True):
                z = -z
            w = wmap.get(col, 0.0)
            num += w * z
            wsum += w
        if wsum <= 0.0:
            return None
        return num / wsum


def load_squad_strength(
    path: str | Path, *, require: bool = False
) -> SquadStrengthTable:
    """Load a squad-strength CSV. A missing file yields an empty table."""
    p = Path(path)
    if not p.exists():
        if require:
            raise FileNotFoundError(f"squad strength file not found: {p}")
        return SquadStrengthTable(teams={})
    df = pd.read_csv(p)
    if "team" not in df.columns:
        raise ValueError("squad strength CSV must have a 'team' column")
    known = {f.name for f in fields(SquadStrength)} - {"team"}
    teams: dict[str, SquadStrength] = {}
    for _, row in df.iterrows():
        team = str(row["team"]).strip()
        if not team:
            continue
        kwargs: dict[str, float] = {}
        for col in known:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != "":
                kwargs[col] = float(row[col])
        teams[team] = SquadStrength(team=team, **kwargs)
    return SquadStrengthTable(teams=teams)


def squad_signal(
    table: SquadStrengthTable,
    home: str,
    away: str,
    *,
    points_per_z: float = 60.0,
    scale: float = 400.0,
    nu: float = 1.0,
) -> OutcomeProbs | None:
    """Outcome signal from the squad-strength difference, or ``None``.

    The standardized strength difference (``z_home - z_away``) is converted to
    an Elo-like advantage of ``points_per_z`` per z-unit and mapped through the
    Davidson model. Returns ``None`` if either team's score is unavailable.
    """
    sh = table.strength_score(home)
    sa = table.strength_score(away)
    if sh is None or sa is None:
        return None
    return davidson_outcome((sh - sa) * points_per_z, scale=scale, nu=nu)
