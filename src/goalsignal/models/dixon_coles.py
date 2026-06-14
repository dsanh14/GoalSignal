"""Dixon-Coles low-score dependence correction (Model B).

Wraps the independent Poisson model and multiplies the (0,0), (1,0), (0,1),
(1,1) cells by the Dixon-Coles tau factor with dependence parameter rho, fit
by maximum likelihood on the training matches' exact regulation scores. The
corrected matrix is renormalized (the tau adjustment is only approximately
mass-preserving; renormalization keeps probabilities exact and is recorded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from goalsignal.models.poisson import PoissonGoalModel, outcome_probs

_EPS = 1e-12
_RHO_BOUNDS = (-0.15, 0.15)


def _tau(h: np.ndarray, a: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float):
    t = np.ones_like(lh, dtype=float)
    m00 = (h == 0) & (a == 0)
    m10 = (h == 1) & (a == 0)
    m01 = (h == 0) & (a == 1)
    m11 = (h == 1) & (a == 1)
    t = np.where(m00, 1.0 - lh * la * rho, t)
    t = np.where(m10, 1.0 + la * rho, t)
    t = np.where(m01, 1.0 + lh * rho, t)
    t = np.where(m11, 1.0 - rho, t)
    return np.maximum(t, _EPS)


class DixonColesModel:
    name = "dixon_coles"

    def __init__(self):
        self.poisson = PoissonGoalModel()

    def fit(self, frame: pd.DataFrame):
        self.poisson.fit(frame)
        eligible = frame[frame["strict_goal_model_eligible"]]
        lams = self.poisson.predict_expected_goals(eligible)
        lh, la = lams[:, 0], lams[:, 1]
        h = eligible["home_score_recorded"].to_numpy(dtype=float)
        a = eligible["away_score_recorded"].to_numpy(dtype=float)
        from scipy.stats import poisson as pois

        base_ll = pois.logpmf(h, lh) + pois.logpmf(a, la)

        def nll(rho: float) -> float:
            return -float((base_ll + np.log(_tau(h, a, lh, la, rho))).mean())

        res = minimize_scalar(nll, bounds=_RHO_BOUNDS, method="bounded")
        self.rho_ = float(res.x)
        return self

    def predict_expected_goals(self, frame: pd.DataFrame) -> np.ndarray:
        return self.poisson.predict_expected_goals(frame)

    def score_matrix(self, lam_home: float, lam_away: float) -> np.ndarray:
        m = self.poisson.score_matrix(lam_home, lam_away).copy()
        hh, aa = np.meshgrid(
            np.arange(m.shape[0]), np.arange(m.shape[1]), indexing="ij"
        )
        t = _tau(
            hh.astype(float),
            aa.astype(float),
            np.full_like(hh, lam_home, dtype=float),
            np.full_like(aa, lam_away, dtype=float),
            self.rho_,
        )
        m = m * t
        return m / m.sum()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        lams = self.predict_expected_goals(frame)
        out = np.empty((len(frame), 3))
        for i, (lh, la) in enumerate(lams):
            out[i] = outcome_probs(self.score_matrix(lh, la))
        return out
