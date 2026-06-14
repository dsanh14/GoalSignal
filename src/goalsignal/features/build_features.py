"""Model input frame: canonical matches joined with pre-match Elo.

The Elo timeline stores the pre-match rating of both teams for every played
match, computed strictly from earlier matches, so joining it to the match
table yields a leakage-free feature frame. Outcome labels follow regulation
outcomes; the handful of matches with unknown regulation outcome are excluded
from supervised use and carry label -1.
"""

from __future__ import annotations

import pandas as pd

LABELS = {"home_win": 0, "draw": 1, "away_win": 2}
CLASS_NAMES = ["home_win", "draw", "away_win"]


def build_match_frame(matches: pd.DataFrame, elo_timeline: pd.DataFrame) -> pd.DataFrame:
    """Played matches with pre-match Elo features and outcome labels."""
    played = matches[matches["status"] == "played"]
    elo_cols = elo_timeline[
        ["canonical_match_id", "home_elo_pre", "away_elo_pre", "expected_home"]
    ]
    frame = played.merge(elo_cols, on="canonical_match_id", how="inner").copy()
    frame["neutral"] = frame["neutral"].fillna(False).astype(bool)
    frame["elo_diff"] = frame["home_elo_pre"] - frame["away_elo_pre"]
    frame["label"] = frame["regulation_outcome"].map(LABELS).fillna(-1).astype(int)
    frame = frame.sort_values(["date", "source_row"], kind="stable").reset_index(drop=True)
    return frame
