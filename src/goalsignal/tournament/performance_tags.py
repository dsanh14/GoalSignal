"""Knockout performance tags — bounded, transparent nudges from live context.

``data/manual/knockout_performance_tags.csv`` records hand-entered evidence
about how each team advanced (penalty wins, extra-time fatigue, late
comebacks, dominant wins, ...). Each tag carries signed ``points``
(percentage-point units, same scale as the human-adjustments YAML) and a
required ``reason``. Tags earned in match N apply only to *later* knockout
matches (``match_number`` strictly greater), and per-team net nudges are
capped, so the layer produces bounded probability nudges — never rewrites.

This is an opinion layer: points are human judgments, not fitted
coefficients. Nothing here touches ``Datasets/``, the ledger, or the
deployed model.

Schema::

    team, match_number, tag, points, reason
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from goalsignal.tournament.knockout_results import MATCH_ROUNDS
from goalsignal.utils.paths import resolve

DEFAULT_TAGS_PATH = "data/manual/knockout_performance_tags.csv"

#: Per-tag point magnitude cap (same units as human-adjustment points).
MAX_TAG_POINTS = 10.0

#: Default cap on a team's *net* tag nudge for one future match.
DEFAULT_TAG_NUDGE_CAP = 6.0

#: tag -> (expected sign, human-adjustments category, optional modifier).
#: The category/modifier mapping keeps generated YAML entries inside the
#: existing, validated adjustment taxonomy.
TAG_DEFINITIONS: dict[str, tuple[int, str, str | None]] = {
    "dominant_win": (1, "tournament_form", "dominant_win_boost"),
    "narrow_win": (1, "tournament_form", None),
    "penalty_win": (1, "tournament_form", "penalty_survival_boost"),
    "extra_time_fatigue": (-1, "tournament_form", "extra_time_fatigue_penalty"),
    "late_comeback": (1, "tournament_form", "late_comeback_boost"),
    "late_collapse_warning": (-1, "tournament_form", "late_collapse_penalty"),
    "blew_lead": (-1, "tournament_form", "late_collapse_penalty"),
    "survived_pressure": (1, "tournament_form", None),
    "low_block_success": (1, "style_matchup", "low_block_vs_possession_boost"),
    "altitude_edge": (1, "venue", "altitude_boost"),
    "not_tested": (-1, "opponent_quality", "easy_path_penalty"),
    "battle_tested": (1, "opponent_quality", "battle_tested_boost"),
    "finishing_boost": (1, "tournament_form", None),
    "defensive_warning": (-1, "tournament_form", None),
}

REQUIRED_COLUMNS = ("team", "match_number", "tag", "points", "reason")


@dataclass(frozen=True)
class PerformanceTag:
    """One tagged observation about a team's knockout performance."""

    team: str
    match_number: int
    tag: str
    points: float
    reason: str

    @property
    def category(self) -> str:
        return TAG_DEFINITIONS[self.tag][1]

    @property
    def modifier(self) -> str | None:
        return TAG_DEFINITIONS[self.tag][2]


@dataclass(frozen=True)
class TagNudge:
    """A bounded net nudge for one team in one future match."""

    team: str
    points: float
    raw_points: float
    tags: tuple[PerformanceTag, ...]

    @property
    def capped(self) -> bool:
        return self.points != self.raw_points

    def reasons(self) -> str:
        return " | ".join(
            f"{t.points:+g} [{t.tag} M{t.match_number}] {t.reason}" for t in self.tags
        )


def load_performance_tags(
    path: str | Path = DEFAULT_TAGS_PATH, *, require: bool = False
) -> list[PerformanceTag]:
    """Load and validate performance tags. Missing file yields an empty list."""
    p = resolve(path)
    if not p.exists():
        if require:
            raise FileNotFoundError(f"performance tags file not found: {p}")
        return []
    frame = pd.read_csv(p, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    problems: list[str] = []
    tags: list[PerformanceTag] = []
    for i, raw in enumerate(frame.to_dict("records")):
        prefix = f"{Path(path).name} row {i + 1}"
        team = str(raw["team"]).strip()
        tag = str(raw["tag"]).strip()
        reason = str(raw["reason"]).strip()
        if not team:
            problems.append(f"{prefix}: 'team' is required")
        if not reason:
            problems.append(f"{prefix}: 'reason' is required")
        if tag not in TAG_DEFINITIONS:
            problems.append(
                f"{prefix}: unknown tag {tag!r} (supported: "
                + ", ".join(sorted(TAG_DEFINITIONS)) + ")"
            )
            continue
        try:
            number = int(str(raw["match_number"]).strip())
        except ValueError:
            problems.append(f"{prefix}: match_number must be an integer")
            continue
        if number not in MATCH_ROUNDS:
            problems.append(f"{prefix}: knockout match numbers are 73-104")
        try:
            points = float(str(raw["points"]).strip())
        except ValueError:
            problems.append(f"{prefix}: points must be numeric")
            continue
        if abs(points) > MAX_TAG_POINTS:
            problems.append(
                f"{prefix}: |points| {abs(points):g} exceeds cap {MAX_TAG_POINTS:g}"
            )
        expected_sign = TAG_DEFINITIONS[tag][0]
        if points * expected_sign < 0:
            problems.append(
                f"{prefix}: tag {tag!r} expects "
                f"{'non-negative' if expected_sign > 0 else 'non-positive'} points, "
                f"got {points:g}"
            )
        tags.append(
            PerformanceTag(
                team=team, match_number=number, tag=tag, points=points, reason=reason
            )
        )
    if problems:
        raise ValueError("invalid performance tags: " + "; ".join(problems))
    return tags


def tag_nudge(
    tags: list[PerformanceTag],
    team: str,
    match_number: int,
    *,
    cap: float = DEFAULT_TAG_NUDGE_CAP,
) -> TagNudge:
    """Bounded net nudge for ``team`` in a *future* match.

    Only tags earned strictly before ``match_number`` contribute (a tag never
    influences the match it was earned in), and the net sum is clamped to
    ``[-cap, +cap]``.
    """
    relevant = tuple(
        t for t in tags if t.team == team and t.match_number < match_number
    )
    raw = sum(t.points for t in relevant)
    return TagNudge(
        team=team,
        points=max(-cap, min(cap, raw)),
        raw_points=raw,
        tags=relevant,
    )
