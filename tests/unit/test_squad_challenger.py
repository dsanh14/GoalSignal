"""Synthetic tests for the squad scenario challenger."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from goalsignal.tournament.squad_challenger import (
    SquadChallengerConfig,
    SquadScenarioAdapter,
    score_team_features,
)


class FixedAdapter:
    def __init__(self):
        self.unrated_teams: set[str] = set()

    def expected_goals(self, home, away, neutral):
        return 1.5, 1.1

    def score_matrix(self, lam_home, lam_away):
        goals = np.arange(10)
        matrix = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
        return matrix / matrix.sum()


def _config() -> SquadChallengerConfig:
    return SquadChallengerConfig.load("config/squad_challenger_2026.yaml")


def _feature_rows() -> pd.DataFrame:
    base = {
        "identity_coverage": 1.0,
        "local_activity_coverage": 1.0,
        "valuation_coverage": 1.0,
        "goalkeeper_local_coverage": 1.0,
        "minimum_position_local_coverage": 1.0,
        "stale_valuation_proportion": 0.0,
        "minutes_30d_coverage": 1.0,
        "minutes_90d_coverage": 1.0,
        "minutes_180d_coverage": 1.0,
        "minutes_90d_total": 8000.0,
        "recently_active_90d": 23,
        "starts_30d_total": 60.0,
        "starts_90d_total": 180.0,
        "starts_180d_total": 360.0,
        "valuation_total": 500.0,
        "valuation_median": 20.0,
        "valuation_minutes_weighted": 30.0,
        "valuation_top_11": 350.0,
        "valuation_top_15": 430.0,
        "valuation_top_23": 500.0,
        "defender_minutes_90d": 2500.0,
        "midfielder_minutes_90d": 2500.0,
        "forward_minutes_90d": 2200.0,
        "defender_active_90d": 1.0,
        "midfielder_active_90d": 1.0,
        "forward_active_90d": 1.0,
        "goalkeeper_minutes_90d": 800.0,
        "goalkeeper_active_90d": 1.0,
        "top_11_minutes_90d": 6500.0,
        "top_15_minutes_90d": 7400.0,
        "top_23_minutes_90d": 8000.0,
        "next_4_minutes_90d": 900.0,
        "inactive_90d": 3,
    }
    strong = {"national_team": "Strong", **base}
    average = {"national_team": "Average", **base}
    for column in (
        "minutes_90d_total",
        "valuation_total",
        "valuation_median",
        "valuation_minutes_weighted",
        "valuation_top_11",
        "valuation_top_15",
        "valuation_top_23",
        "top_11_minutes_90d",
        "top_15_minutes_90d",
        "top_23_minutes_90d",
    ):
        strong[column] *= 1.3
        average[column] *= 0.8
    fallback = {
        "national_team": "Fallback",
        **base,
        "local_activity_coverage": 0.2,
        "valuation_coverage": 0.1,
        "goalkeeper_local_coverage": 0.0,
        "minimum_position_local_coverage": 0.2,
    }
    return pd.DataFrame([strong, average, fallback])


def test_scoring_shrinks_by_coverage_and_uses_exact_fallback():
    scored = score_team_features(_feature_rows(), _config()).set_index("national_team")

    assert scored.loc["Strong", "coverage_eligible"]
    assert scored.loc["Strong", "log_goal_adjustment"] > scored.loc[
        "Average", "log_goal_adjustment"
    ]
    assert scored.loc["Fallback", "fallback_used"]
    assert scored.loc["Fallback", "coverage_confidence"] == 0
    assert scored.loc["Fallback", "log_goal_adjustment"] == 0
    assert scored["log_goal_adjustment"].abs().max() <= 0.18


def test_adapter_preserves_probabilities_and_direction():
    scored = score_team_features(_feature_rows(), _config())
    base = FixedAdapter()
    challenger = SquadScenarioAdapter(base, scored)

    base_lams = base.expected_goals("Strong", "Average", True)
    squad_lams = challenger.expected_goals("Strong", "Average", True)
    probabilities = challenger.outcome_probabilities("Strong", "Average", True)

    assert squad_lams[0] > base_lams[0]
    assert squad_lams[1] < base_lams[1]
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_two_fallback_teams_reproduce_base_exactly():
    rows = _feature_rows()
    rows.loc[rows["national_team"].eq("Average"), [
        "local_activity_coverage",
        "valuation_coverage",
        "goalkeeper_local_coverage",
        "minimum_position_local_coverage",
    ]] = 0.0
    scored = score_team_features(rows, _config())
    base = FixedAdapter()
    challenger = SquadScenarioAdapter(base, scored)

    assert challenger.expected_goals("Fallback", "Average", True) == (
        base.expected_goals("Fallback", "Average", True)
    )
