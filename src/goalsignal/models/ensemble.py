"""Convex ensemble of component outcome models.

Weights live on the probability simplex (w_i >= 0, sum w_i = 1) and are fit by
minimizing the multiclass NLL of the weighted mixture on *validation-period*
predictions only. Component predictions are stored alongside the result so
every ensemble probability is reproducible from its parts.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12


class ConvexEnsemble:
    name = "ensemble"

    def fit(self, component_probs: dict[str, np.ndarray], labels: np.ndarray):
        names = sorted(component_probs)
        stack = np.stack([component_probs[n] for n in names])  # (k, n, 3)
        k = len(names)
        idx = np.arange(len(labels))

        def nll(w: np.ndarray) -> float:
            mix = np.tensordot(w, stack, axes=1)
            return -float(np.log(mix[idx, labels] + _EPS).mean())

        res = minimize(
            nll,
            np.full(k, 1.0 / k),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * k,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        )
        w = np.clip(res.x, 0.0, 1.0)
        self.weights_ = dict(zip(names, w / w.sum(), strict=True))
        return self

    def predict_proba(self, component_probs: dict[str, np.ndarray]) -> np.ndarray:
        missing = set(self.weights_) - set(component_probs)
        if missing:
            raise ValueError(f"missing component predictions: {sorted(missing)}")
        out = sum(
            w * component_probs[name] for name, w in self.weights_.items()
        )
        return out / out.sum(axis=1, keepdims=True)
