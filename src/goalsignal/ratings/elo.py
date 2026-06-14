"""Sequential Elo ratings for national teams.

Matches are processed in strict chronological order (date, then source row for
same-day stability). Each row of the output timeline records the *pre-match*
ratings used for the update, which makes the timeline directly usable as a
leakage-free feature source: the pre-match rating for match i depends only on
matches that finished before it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from goalsignal.utils.paths import resolve


class ImportanceRule(BaseModel):
    pattern: str
    multiplier: float


class EloConfig(BaseModel):
    initial_rating: float = 1500.0
    scale: float = 400.0
    k_factor: float = 20.0
    home_advantage: float = 60.0
    importance: list[ImportanceRule] = Field(default_factory=list)
    default_importance: float = 1.0
    goal_difference_multiplier: str = "eloratings"
    shootout_policy: str = "draw"
    shootout_credit: float = 0.25
    unknown_regulation_outcome_fallback: str = "recorded_score"

    @classmethod
    def load(cls, path: str | Path = "config/elo.yaml") -> EloConfig:
        with open(resolve(path), encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


@dataclass
class EloResult:
    timeline: pd.DataFrame  # one row per played match, chronological
    final_ratings: dict[str, float]


def expected_home_score(r_home_adj: float, r_away: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(r_home_adj - r_away) / scale))


def _gd_multiplier(goal_diff: int, mode: str) -> float:
    if mode == "none":
        return 1.0
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def _importance(tournament: str, config: EloConfig) -> float:
    t = tournament.lower()
    for rule in config.importance:
        if rule.pattern in t:
            return rule.multiplier
    return config.default_importance


def _actual_score(row, config: EloConfig) -> float | None:
    """Home-team result S in [0, 1] for the rating update, or None to skip."""
    outcome = row.regulation_outcome
    if outcome == "unknown":
        if config.unknown_regulation_outcome_fallback != "recorded_score":
            return None
        h, a = row.home_score_recorded, row.away_score_recorded
        outcome = "home_win" if h > a else ("away_win" if h < a else "draw")

    base = {"home_win": 1.0, "draw": 0.5, "away_win": 0.0}[outcome]
    if row.shootout_played and config.shootout_policy == "winner_partial" and base == 0.5:
        credit = config.shootout_credit
        return 0.5 + credit if row.shootout_winner == row.home_team else 0.5 - credit
    return base


def compute_elo(matches: pd.DataFrame, config: EloConfig) -> EloResult:
    """Compute the full rating timeline over played matches.

    `matches` is the canonical match table (build_dataset output). Scheduled
    matches are ignored. Returns one timeline row per rated match.
    """
    played = matches[matches["status"] == "played"].sort_values(
        ["date", "source_row"], kind="stable"
    )

    ratings: dict[str, float] = {}
    rows: list[dict] = []
    for row in played.itertuples(index=False):
        r_home = ratings.get(row.home_team, config.initial_rating)
        r_away = ratings.get(row.away_team, config.initial_rating)
        is_neutral = bool(row.neutral) if row.neutral is not None else False
        home_adv = 0.0 if is_neutral else config.home_advantage
        expected = expected_home_score(r_home + home_adv, r_away, config.scale)

        actual = _actual_score(row, config)
        if actual is None:
            continue

        gd = int(abs(row.home_score_recorded - row.away_score_recorded))
        delta = (
            config.k_factor
            * _importance(row.tournament, config)
            * _gd_multiplier(gd, config.goal_difference_multiplier)
            * (actual - expected)
        )
        rows.append(
            {
                "canonical_match_id": row.canonical_match_id,
                "date": row.date,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "tournament": row.tournament,
                "neutral": is_neutral,
                "home_elo_pre": r_home,
                "away_elo_pre": r_away,
                "expected_home": expected,
                "actual_home": actual,
                "delta": delta,
                "home_elo_post": r_home + delta,
                "away_elo_post": r_away - delta,
            }
        )
        ratings[row.home_team] = r_home + delta
        ratings[row.away_team] = r_away - delta

    return EloResult(timeline=pd.DataFrame(rows), final_ratings=ratings)


def ratings_as_of(timeline: pd.DataFrame, cutoff: pd.Timestamp, config: EloConfig):
    """Team -> rating using only matches strictly before `cutoff`."""
    past = timeline[timeline["date"] < cutoff]
    ratings: dict[str, float] = {}
    for row in past.itertuples(index=False):
        ratings[row.home_team] = row.home_elo_post
        ratings[row.away_team] = row.away_elo_post
    return ratings
