"""Live deployment pipeline for 2026 World Cup forecasting.

Mirrors the backtest protocol exactly: component models fit on matches before
the validation window, temperature calibration and ensemble weights fit on the
validation window (the final `val_years` years before the data cutoff), and
predictions generated only for fixtures after the cutoff. The data cutoff is
the day after the last played match, so a rebuilt dataset with new results
automatically produces a later cutoff and fresh ratings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from goalsignal.evaluation.backtest import ENSEMBLE_COMPONENTS
from goalsignal.features.build_features import build_match_frame
from goalsignal.models.baselines import EloDavidson
from goalsignal.models.calibration import TemperatureScaler
from goalsignal.models.dixon_coles import DixonColesModel
from goalsignal.models.ensemble import ConvexEnsemble
from goalsignal.models.outcome_classifier import MultinomialLogistic
from goalsignal.ratings.elo import EloConfig, compute_elo

MODEL_VERSION = "ensemble-v1"


@dataclass
class LiveModel:
    components: dict
    scalers: dict
    ensemble: ConvexEnsemble
    goal_model: DixonColesModel
    ratings: dict[str, float]
    cutoff: pd.Timestamp
    dataset_version: str
    diagnostics: dict
    # Distinct version per ingested-result state, e.g. "ensemble-v1+r1": new
    # predictions can never collide with or be confused for earlier frozen
    # forecasts (the ledger rejects duplicate fixture+model_version pairs).
    model_version: str = MODEL_VERSION

    def predict_outcome(self, feature_frame: pd.DataFrame) -> np.ndarray:
        calibrated = {
            name: self.scalers[name].transform(
                self.components[name].predict_proba(feature_frame)
            )
            for name in ENSEMBLE_COMPONENTS
        }
        return self.ensemble.predict_proba(calibrated)

    def feature_row(self, home: str, away: str, neutral: bool) -> pd.DataFrame:
        r_h = self.ratings.get(home, 1500.0)
        r_a = self.ratings.get(away, 1500.0)
        return pd.DataFrame(
            {
                "home_elo_pre": [r_h],
                "away_elo_pre": [r_a],
                "elo_diff": [r_h - r_a],
                "neutral": [bool(neutral)],
            }
        )


def score_summary(goal_model, feats: pd.DataFrame, k: int = 5) -> dict:
    """Exact-score quantities from a fitted goal model for one fixture.

    All scoreline information comes from the goal model's score matrix; the
    W/D/L ensemble plays no part here. `tail_mass` is the pre-normalization
    probability mass beyond the matrix grid, tracked, never discarded.
    """
    from goalsignal.models.poisson import market_probs, top_scorelines

    lam_h, lam_a = goal_model.predict_expected_goals(feats)[0]
    matrix = goal_model.score_matrix(float(lam_h), float(lam_a))
    return {
        "home_expected_goals": float(lam_h),
        "away_expected_goals": float(lam_a),
        "matrix": matrix,
        "max_goals": int(matrix.shape[0] - 1),
        "tail_mass": float(goal_model.poisson.last_tail_mass_)
        if hasattr(goal_model, "poisson")
        else float(goal_model.last_tail_mass_),
        "top_scorelines": top_scorelines(matrix, k),
        "markets": market_probs(matrix),
    }


def build_prediction_payload(live: LiveModel, row, revision_metadata: dict | None = None) -> dict:
    """Ledger payload for one scheduled fixture (schema identical to v1).

    `row` is a canonical-match itertuple. W/D/L probabilities come from the
    calibrated ensemble; expected goals and scorelines from the goal model.
    """
    feats = live.feature_row(row.home_team, row.away_team, bool(row.neutral))
    probs = live.predict_outcome(feats)[0]
    s = score_summary(live.goal_model, feats, k=5)
    payload = {
        "fixture_id": row.canonical_match_id,
        "home_team": row.home_team,
        "away_team": row.away_team,
        "tournament": row.tournament,
        "kickoff_timestamp": str(row.date.date()),
        "kickoff_time_known": False,
        "data_cutoff": str(live.cutoff.date()),
        "dataset_version": live.dataset_version,
        "model_version": live.model_version,
        "home_expected_goals": round(s["home_expected_goals"], 4),
        "away_expected_goals": round(s["away_expected_goals"], 4),
        "home_win_probability": round(float(probs[0]), 4),
        "draw_probability": round(float(probs[1]), 4),
        "away_win_probability": round(float(probs[2]), 4),
        "top_scorelines": [
            {"home": h, "away": a, "p": round(p, 4)}
            for h, a, p in s["top_scorelines"]
        ],
        "markets": {k: round(v, 4) for k, v in s["markets"].items()},
        "limitations": [
            "kickoff date known but not time; probabilities are "
            "regulation-time (90-minute) outcomes",
            "features limited to pre-match Elo and venue context",
        ],
    }
    payload.update(revision_metadata or {})
    return payload


def train_live_model(
    matches: pd.DataFrame,
    dataset_version: str,
    val_years: int = 3,
    elo_config: EloConfig | None = None,
) -> LiveModel:
    elo_config = elo_config or EloConfig.load()
    elo_result = compute_elo(matches, elo_config)
    frame = build_match_frame(matches, elo_result.timeline)
    labeled = frame[frame["label"] >= 0]

    cutoff = labeled["date"].max() + pd.Timedelta(days=1)
    val_start = cutoff - pd.DateOffset(years=val_years)
    train = labeled[labeled["date"] < val_start]
    val = labeled[(labeled["date"] >= val_start) & (labeled["date"] < cutoff)]

    components = {
        "elo_davidson": EloDavidson(elo_config).fit(train),
        "dixon_coles": DixonColesModel().fit(train),
        "multinomial_logistic": MultinomialLogistic().fit(train),
    }
    y_val = val["label"].to_numpy()
    val_probs, scalers = {}, {}
    for name, model in components.items():
        raw = model.predict_proba(val)
        scalers[name] = TemperatureScaler().fit(raw, y_val)
        val_probs[name] = scalers[name].transform(raw)
    ensemble = ConvexEnsemble().fit(val_probs, y_val)

    # Goal model used for score sampling is refit on everything before the
    # cutoff (it needs the freshest attack/defence signal and has no
    # calibration step of its own).
    goal_model = DixonColesModel().fit(labeled)

    # Ratings snapshot at the cutoff (includes every played match).
    ratings = dict(
        zip(
            elo_result.final_ratings.keys(),
            elo_result.final_ratings.values(),
            strict=True,
        )
    )
    return LiveModel(
        components=components,
        scalers=scalers,
        ensemble=ensemble,
        goal_model=goal_model,
        ratings=ratings,
        cutoff=cutoff,
        dataset_version=dataset_version,
        diagnostics={
            "train_matches": len(train),
            "val_matches": len(val),
            "val_window": [str(val_start.date()), str(cutoff.date())],
            "ensemble_weights": {k: float(v) for k, v in ensemble.weights_.items()},
            "temperatures": {k: s.temperature_ for k, s in scalers.items()},
            "dixon_coles_rho": goal_model.rho_,
        },
    )
