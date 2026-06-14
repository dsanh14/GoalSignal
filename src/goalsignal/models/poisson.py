"""Independent Poisson goal model (Model A).

Each side's goals are Poisson with a log link on pre-match features:

    log(lambda) = b0 + b1 * (own_elo - opp_elo) / 400 + b2 * is_home_advantage

Fitting uses iteratively reweighted least squares on the stacked home/away
observations of the strict-goal-model-eligible training matches (regulation
scores only). Exact-score probabilities come from the outer product of the two
Poisson PMFs over a 0..max_goals grid; remaining tail mass is accounted for by
renormalization and reported, never silently discarded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

MAX_GOALS = 12
_ETA_CLIP = (-3.0, 3.0)  # log-intensity clip: lambda in [e^-3, e^3] ~ [0.05, 20]


def _design(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Stacked (home rows then away rows) design matrix and goal vector."""
    d = (frame["elo_diff"].to_numpy() / 400.0).astype(float)
    home_adv = (~frame["neutral"].to_numpy(dtype=bool)).astype(float)
    n = len(frame)
    x_home = np.column_stack([np.ones(n), d, home_adv])
    x_away = np.column_stack([np.ones(n), -d, np.zeros(n)])
    x = np.vstack([x_home, x_away])
    y = np.concatenate(
        [
            frame["home_score_recorded"].to_numpy(dtype=float),
            frame["away_score_recorded"].to_numpy(dtype=float),
        ]
    )
    return x, y


class PoissonGoalModel:
    name = "poisson"

    def fit(self, frame: pd.DataFrame, max_iter: int = 50, tol: float = 1e-10):
        eligible = frame[frame["strict_goal_model_eligible"]]
        x, y = _design(eligible)
        beta = np.zeros(x.shape[1])
        beta[0] = np.log(max(y.mean(), 1e-3))
        for _ in range(max_iter):
            eta = np.clip(x @ beta, *_ETA_CLIP)
            mu = np.exp(eta)
            grad = x.T @ (y - mu)
            hess = (x * mu[:, None]).T @ x
            step = np.linalg.solve(hess + 1e-8 * np.eye(len(beta)), grad)
            beta = beta + step
            if np.max(np.abs(step)) < tol:
                break
        self.beta_ = beta
        if not np.all(np.isfinite(beta)):
            raise RuntimeError("Poisson GLM failed to converge to finite coefficients")
        return self

    def predict_expected_goals(self, frame: pd.DataFrame) -> np.ndarray:
        """(n, 2) array of [lambda_home, lambda_away]."""
        x, _ = _design(frame.assign(home_score_recorded=0, away_score_recorded=0))
        eta = np.clip(x @ self.beta_, *_ETA_CLIP)
        lam = np.exp(eta)
        n = len(frame)
        return np.column_stack([lam[:n], lam[n:]])

    def score_matrix(self, lam_home: float, lam_away: float) -> np.ndarray:
        """(max_goals+1)^2 matrix of P(home=x, away=y), renormalized.

        The pre-normalization tail mass is stored in `self.last_tail_mass_`.
        """
        g = np.arange(MAX_GOALS + 1)
        ph = poisson.pmf(g, lam_home)
        pa = poisson.pmf(g, lam_away)
        m = np.outer(ph, pa)
        self.last_tail_mass_ = float(1.0 - m.sum())
        return m / m.sum()

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        lams = self.predict_expected_goals(frame)
        out = np.empty((len(frame), 3))
        for i, (lh, la) in enumerate(lams):
            m = self.score_matrix(lh, la)
            out[i] = outcome_probs(m)
        return out


def outcome_probs(matrix: np.ndarray) -> np.ndarray:
    """[P(home win), P(draw), P(away win)] from a score matrix."""
    home = float(np.tril(matrix, -1).sum())
    draw = float(np.trace(matrix))
    away = float(np.triu(matrix, 1).sum())
    return np.array([home, draw, away])


def market_probs(matrix: np.ndarray) -> dict[str, float]:
    """Derived goal-market probabilities from a score matrix."""
    g = np.arange(matrix.shape[0])
    totals = g[:, None] + g[None, :]
    return {
        "over_0_5": float(matrix[totals > 0.5].sum()),
        "over_1_5": float(matrix[totals > 1.5].sum()),
        "over_2_5": float(matrix[totals > 2.5].sum()),
        "over_3_5": float(matrix[totals > 3.5].sum()),
        "under_2_5": float(matrix[totals < 2.5].sum()),
        "both_teams_score": float(matrix[1:, 1:].sum()),
        "home_clean_sheet": float(matrix[:, 0].sum()),
        "away_clean_sheet": float(matrix[0, :].sum()),
    }


def top_scorelines(matrix: np.ndarray, k: int = 5) -> list[tuple[int, int, float]]:
    flat = [(int(h), int(a), float(matrix[h, a]))
            for h in range(matrix.shape[0]) for a in range(matrix.shape[1])]
    return sorted(flat, key=lambda t: t[2], reverse=True)[:k]
