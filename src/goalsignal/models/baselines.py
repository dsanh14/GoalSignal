"""Transparent baseline outcome models.

All models share the interface: fit(frame) -> self, predict_proba(frame) ->
ndarray of shape (n, 3) ordered [home_win, draw, away_win]. Frames are the
feature frames from goalsignal.features.build_features (rows with label == -1
must be filtered by the caller before fitting).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from goalsignal.ratings.elo import EloConfig, expected_home_score

_EPS = 1e-12


class UniformBaseline:
    """Baseline 0: 1/3 each."""

    name = "uniform"

    def fit(self, frame: pd.DataFrame):
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full((len(frame), 3), 1.0 / 3.0)


class EmpiricalFrequency:
    """Baseline 1: global outcome frequencies from training data."""

    name = "empirical"

    def fit(self, frame: pd.DataFrame):
        counts = np.bincount(frame["label"].to_numpy(), minlength=3).astype(float)
        self.p_ = (counts + 1.0) / (counts.sum() + 3.0)  # add-one smoothing
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p_, (len(frame), 1))


class ContextFrequency:
    """Baseline 2: outcome frequencies conditioned on home vs neutral venue."""

    name = "context_frequency"

    def fit(self, frame: pd.DataFrame):
        self.p_ = {}
        for is_neutral in (False, True):
            sub = frame[frame["neutral"] == is_neutral]
            counts = np.bincount(sub["label"].to_numpy(), minlength=3).astype(float)
            self.p_[is_neutral] = (counts + 1.0) / (counts.sum() + 3.0)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.stack([self.p_[bool(n)] for n in frame["neutral"]])


class HigherRatedHeuristic:
    """Baseline 3: most of the mass on the higher-rated team.

    Probabilities are the training outcome frequencies conditioned on the sign
    of the Elo difference, so the heuristic stays probabilistic and honest.
    """

    name = "higher_rated"

    def fit(self, frame: pd.DataFrame):
        self.p_ = {}
        for sign in (-1, 0, 1):
            sub = frame[np.sign(frame["elo_diff"]).astype(int) == sign]
            counts = np.bincount(sub["label"].to_numpy(), minlength=3).astype(float)
            self.p_[sign] = (counts + 1.0) / (counts.sum() + 3.0)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        signs = np.sign(frame["elo_diff"]).astype(int)
        return np.stack([self.p_[int(s)] for s in signs])


class EloDavidson:
    """Baseline 4: Elo win expectancy extended with a Davidson draw parameter.

    With p = E (Elo expected home score incl. home advantage) and q = 1 - E:
        P(draw) ∝ nu * sqrt(p * q),  P(home) ∝ p,  P(away) ∝ q
    nu >= 0 is fit by maximum likelihood on the training frame. nu = 0 recovers
    the draw-free Elo model; larger nu concentrates mass on draws for evenly
    matched teams, which is the empirically observed pattern.
    """

    name = "elo_davidson"

    def __init__(self, elo_config: EloConfig | None = None):
        self.elo_config = elo_config or EloConfig()

    def _raw_expected(self, frame: pd.DataFrame) -> np.ndarray:
        adv = np.where(frame["neutral"], 0.0, self.elo_config.home_advantage)
        r_home = frame["home_elo_pre"].to_numpy() + adv
        r_away = frame["away_elo_pre"].to_numpy()
        return expected_home_score(r_home, r_away, self.elo_config.scale)

    @staticmethod
    def _probs(e: np.ndarray, nu: float) -> np.ndarray:
        p, q = e, 1.0 - e
        d = nu * np.sqrt(p * q)
        z = p + q + d
        return np.stack([p / z, d / z, q / z], axis=1)

    def fit(self, frame: pd.DataFrame):
        e = self._raw_expected(frame)
        y = frame["label"].to_numpy()

        def nll(nu: float) -> float:
            probs = self._probs(e, nu)
            return -float(np.log(probs[np.arange(len(y)), y] + _EPS).mean())

        res = minimize_scalar(nll, bounds=(1e-6, 5.0), method="bounded")
        self.nu_ = float(res.x)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self._probs(self._raw_expected(frame), self.nu_)
