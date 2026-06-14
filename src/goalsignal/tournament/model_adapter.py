"""Adapter exposing the trained goal model to the tournament simulator.

Wraps a ratings snapshot (as of the data cutoff) and a fitted goal model;
the simulator only ever asks for expected goals and a score matrix, keeping
tournament rules fully separate from model logic.
"""

from __future__ import annotations

import pandas as pd


class RatingsGoalAdapter:
    def __init__(
        self,
        ratings: dict[str, float],
        goal_model,
        default_rating: float = 1500.0,
    ):
        self.ratings = ratings
        self.goal_model = goal_model
        self.default_rating = default_rating
        self.unrated_teams: set[str] = set()

    def _rating(self, team: str) -> float:
        if team not in self.ratings:
            self.unrated_teams.add(team)
            return self.default_rating
        return self.ratings[team]

    def expected_goals(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        frame = pd.DataFrame(
            {
                "elo_diff": [self._rating(home) - self._rating(away)],
                "neutral": [bool(neutral)],
            }
        )
        lams = self.goal_model.predict_expected_goals(frame)
        return float(lams[0, 0]), float(lams[0, 1])

    def score_matrix(self, lam_home: float, lam_away: float):
        return self.goal_model.score_matrix(lam_home, lam_away)
