"""Direct multinomial-logistic outcome model (softmax regression).

Features: signed Elo difference, home-advantage indicator, and absolute Elo
difference (lets draw probability fall with mismatch). Fit by L-BFGS on the
multiclass NLL with a small L2 penalty for identifiability. Implemented in
NumPy/SciPy to keep the base stack light.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_L2 = 1e-4


def _features(frame: pd.DataFrame) -> np.ndarray:
    d = frame["elo_diff"].to_numpy() / 400.0
    home_adv = (~frame["neutral"].to_numpy(dtype=bool)).astype(float)
    return np.column_stack([np.ones(len(frame)), d, np.abs(d), home_adv])


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MultinomialLogistic:
    name = "multinomial_logistic"

    def fit(self, frame: pd.DataFrame):
        x = _features(frame)
        y = frame["label"].to_numpy()
        n, f = x.shape
        onehot = np.eye(3)[y]

        def objective(w_flat: np.ndarray):
            w = w_flat.reshape(f, 3)
            p = _softmax(x @ w)
            nll = -np.log(p[np.arange(n), y] + 1e-12).mean() + _L2 * np.sum(w**2)
            grad = x.T @ (p - onehot) / n + 2 * _L2 * w
            return nll, grad.ravel()

        res = minimize(objective, np.zeros(f * 3), jac=True, method="L-BFGS-B")
        if not np.all(np.isfinite(res.x)):
            raise RuntimeError("multinomial logistic fit produced non-finite weights")
        self.w_ = res.x.reshape(f, 3)
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return _softmax(_features(frame) @ self.w_)
