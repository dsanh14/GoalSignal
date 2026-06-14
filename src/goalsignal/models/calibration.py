"""Probability calibration.

Temperature scaling: p_cal ∝ p^(1/T), the multiclass analogue of Platt
scaling on log-probabilities with a single parameter, which makes it safe to
fit on modest validation samples. T is fit by minimizing validation NLL and
must never be fit on test-period predictions.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-12


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.log(np.clip(probs, _EPS, 1.0)) / temperature
    logp -= logp.max(axis=1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureScaler:
    def fit(self, probs: np.ndarray, labels: np.ndarray):
        def nll(t: float) -> float:
            p = apply_temperature(probs, t)
            return -float(np.log(p[np.arange(len(labels)), labels] + _EPS).mean())

        res = minimize_scalar(nll, bounds=(0.2, 5.0), method="bounded")
        self.temperature_ = float(res.x)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return apply_temperature(probs, self.temperature_)
