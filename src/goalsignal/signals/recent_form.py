"""Recent-form signal — Elo-adjusted, not raw.

Raw recent results are a biased signal: beating weak opponents inflates form and
friendlies are noisy. This module consumes *opponent-strength-adjusted* form
indicators (you supply them, or compute them upstream with
:mod:`goalsignal.features.d1`, which already builds leakage-safe opponent-
adjusted goal residuals and Elo-weighted form) and turns the home-vs-away form
difference into an outcome signal.

Input format — a CSV keyed by ``team`` with any subset of::

    team,
    elo_adj_last5,    # mean Elo-adjusted performance over the last 5 matches
    elo_adj_last10,   # ... over the last 10 matches
    gf_adj,           # goals for, adjusted for opponent strength
    ga_adj,           # goals against, adjusted for opponent strength (penalty)
    xg_diff           # xG differential per match, if available

The intent is that whoever produces these numbers already weights competitive
matches above friendlies and major tournaments above ordinary fixtures — the
helper :func:`weighted_form_mean` documents and implements that weighting for
callers computing form here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from goalsignal.signals.base import OutcomeProbs, davidson_outcome

# (column, weight, higher_is_better).
DEFAULT_FORM_WEIGHTS: tuple[tuple[str, float, bool], ...] = (
    ("elo_adj_last5", 1.0, True),
    ("elo_adj_last10", 0.7, True),
    ("gf_adj", 0.5, True),
    ("ga_adj", 0.5, False),
    ("xg_diff", 0.8, True),
)

_SCORED_COLUMNS = tuple(name for name, _, _ in DEFAULT_FORM_WEIGHTS)

# Default competition importance weights (competitive > friendly, majors most).
COMPETITION_WEIGHTS: dict[str, float] = {
    "world_cup": 2.0,
    "continental": 1.5,
    "qualification": 1.25,
    "other": 1.0,
    "friendly": 0.5,
}


def weighted_form_mean(values, competitions, *, weights: dict[str, float] | None = None) -> float:
    """Importance-weighted mean of per-match form values.

    Down-weights friendlies and up-weights major tournaments so callers who
    compute form here apply the same policy the file-based inputs are expected
    to follow. ``competitions`` are keys into ``weights`` (default
    :data:`COMPETITION_WEIGHTS`); unknown keys fall back to ``"other"``.
    """
    w = weights or COMPETITION_WEIGHTS
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        raise ValueError("cannot take a weighted mean of zero matches")
    ws = np.array([w.get(c, w.get("other", 1.0)) for c in competitions], dtype=float)
    return float(np.average(vals, weights=ws))


@dataclass(frozen=True)
class RecentForm:
    """Optional per-team form indicators (any field may be ``None``)."""

    team: str
    elo_adj_last5: float | None = None
    elo_adj_last10: float | None = None
    gf_adj: float | None = None
    ga_adj: float | None = None
    xg_diff: float | None = None

    def available(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for col in _SCORED_COLUMNS:
            val = getattr(self, col)
            if val is not None:
                out[col] = float(val)
        return out


@dataclass
class RecentFormTable:
    """Loaded form table with population statistics for standardization."""

    teams: dict[str, RecentForm]
    _mean: dict[str, float] = None  # type: ignore[assignment]
    _std: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._mean = {}
        self._std = {}
        for col in _SCORED_COLUMNS:
            vals = [
                getattr(s, col) for s in self.teams.values() if getattr(s, col) is not None
            ]
            if len(vals) >= 2:
                self._mean[col] = float(np.mean(vals))
                std = float(np.std(vals))
                self._std[col] = std if std > 1e-9 else 0.0

    def form_score(self, team: str) -> float | None:
        record = self.teams.get(team)
        if record is None:
            return None
        signs = {col: good for col, _, good in DEFAULT_FORM_WEIGHTS}
        wmap = {col: w for col, w, _ in DEFAULT_FORM_WEIGHTS}
        num = 0.0
        wsum = 0.0
        for col, val in record.available().items():
            std = self._std.get(col, 0.0)
            if std <= 0.0:
                continue
            z = (val - self._mean[col]) / std
            if not signs.get(col, True):
                z = -z
            num += wmap[col] * z
            wsum += wmap[col]
        if wsum <= 0.0:
            return None
        return num / wsum


def load_recent_form(path: str | Path, *, require: bool = False) -> RecentFormTable:
    """Load a recent-form CSV. A missing file yields an empty table."""
    p = Path(path)
    if not p.exists():
        if require:
            raise FileNotFoundError(f"recent form file not found: {p}")
        return RecentFormTable(teams={})
    df = pd.read_csv(p)
    if "team" not in df.columns:
        raise ValueError("recent form CSV must have a 'team' column")
    known = {f.name for f in fields(RecentForm)} - {"team"}
    teams: dict[str, RecentForm] = {}
    for _, row in df.iterrows():
        team = str(row["team"]).strip()
        if not team:
            continue
        kwargs: dict[str, float] = {}
        for col in known:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != "":
                kwargs[col] = float(row[col])
        teams[team] = RecentForm(team=team, **kwargs)
    return RecentFormTable(teams=teams)


def form_signal(
    table: RecentFormTable,
    home: str,
    away: str,
    *,
    points_per_z: float = 40.0,
    scale: float = 400.0,
    nu: float = 1.0,
) -> OutcomeProbs | None:
    """Outcome signal from the recent-form difference, or ``None``."""
    fh = table.form_score(home)
    fa = table.form_score(away)
    if fh is None or fa is None:
        return None
    return davidson_outcome((fh - fa) * points_per_z, scale=scale, nu=nu)
