"""Configurable meta-ensemble that blends signals into the product output.

This is the top-level forecaster: it takes one probability object per signal
(some may be ``None``) and a weight map, and returns a single calibrated
distribution. Two product invariants are enforced here, not in the signal code:

1. **Configurable weights.** Weights come from ``config/ensemble.yaml`` (or any
   passed map), never hardcoded. Several named *model versions* (baseline,
   market-only, challengers, final ensemble) are supported for champion/
   challenger backtesting.
2. **Renormalization on missing signals.** When a signal has no value for a
   match, its weight is dropped and the remaining weights are renormalized, so
   a match with partial coverage still gets a proper distribution.

The blend is a weighted average of distributions (a linear opinion pool), which
keeps the output a valid probability vector and is order-invariant. The same
machinery handles 3-way group outcomes and 2-way knockout advancement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from goalsignal.signals.base import AdvanceProbs, OutcomeProbs, disagreement
from goalsignal.utils.paths import resolve

_DEFAULT_CONFIG = "config/ensemble.yaml"


@dataclass(frozen=True)
class EnsembleConfig:
    """Parsed ``config/ensemble.yaml``."""

    default_weights: dict[str, float]
    model_versions: dict[str, dict[str, float]]
    signal_params: dict
    market_overround_method: str
    knockout_tiebreak_a_prob: float
    disagreement_threshold: float

    def weights_for(self, version: str | None) -> dict[str, float]:
        """Weight map for a named model version (default = final product)."""
        if version is None or version == "default":
            return dict(self.default_weights)
        if version not in self.model_versions:
            raise KeyError(
                f"unknown model version {version!r}; "
                f"available: {sorted(self.model_versions)}"
            )
        return dict(self.model_versions[version])


def load_ensemble_config(path: str | Path = _DEFAULT_CONFIG) -> EnsembleConfig:
    """Load the meta-ensemble configuration from YAML."""
    with open(resolve(path)) as fh:
        raw = yaml.safe_load(fh)
    return EnsembleConfig(
        default_weights=dict(raw.get("default_weights", {})),
        model_versions={k: dict(v) for k, v in raw.get("model_versions", {}).items()},
        signal_params=dict(raw.get("signal_params", {})),
        market_overround_method=raw.get("market_overround_method", "proportional"),
        knockout_tiebreak_a_prob=float(raw.get("knockout_tiebreak_a_prob", 0.5)),
        disagreement_threshold=float(raw.get("disagreement_threshold", 0.15)),
    )


@dataclass(frozen=True)
class BlendResult:
    """Outcome of a single blend, with full provenance for reproducibility."""

    probs: OutcomeProbs | AdvanceProbs
    used_weights: dict[str, float]  # renormalized weights actually applied
    missing: list[str]  # configured signals with no value for this match
    components: dict[str, OutcomeProbs | AdvanceProbs]  # available signal probs
    max_pairwise_disagreement: float  # max TVD between any two available signals


def _pairwise_max_disagreement(components: dict[str, OutcomeProbs | AdvanceProbs]) -> float:
    items = list(components.values())
    worst = 0.0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i].as_array(), items[j].as_array()
            worst = max(worst, float(0.5 * np.abs(a - b).sum()))
    return worst


class MetaEnsemble:
    """Blend signals into a single calibrated distribution per match."""

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or load_ensemble_config()

    # -- internal generic blend -------------------------------------------------
    def _blend(
        self,
        signals: dict[str, OutcomeProbs | AdvanceProbs | None],
        weights: dict[str, float],
        rebuild,
    ) -> BlendResult:
        components: dict[str, OutcomeProbs | AdvanceProbs] = {}
        used: dict[str, float] = {}
        missing: list[str] = []
        for name, weight in weights.items():
            if weight <= 0.0:
                continue
            probs = signals.get(name)
            if probs is None:
                missing.append(name)
                continue
            components[name] = probs
            used[name] = weight
        if not used:
            raise ValueError(
                "no signals available to blend (every weighted signal was missing)"
            )
        total = sum(used.values())
        used = {k: v / total for k, v in used.items()}
        stacked = np.vstack([components[k].as_array() * used[k] for k in used])
        blended = rebuild(stacked.sum(axis=0))
        return BlendResult(
            probs=blended,
            used_weights=used,
            missing=missing,
            components=components,
            max_pairwise_disagreement=_pairwise_max_disagreement(components),
        )

    # -- public API -------------------------------------------------------------
    def blend(
        self,
        signals: dict[str, OutcomeProbs | None],
        *,
        version: str | None = None,
        weights: dict[str, float] | None = None,
    ) -> BlendResult:
        """Blend group-stage outcome signals into one :class:`OutcomeProbs`."""
        w = weights if weights is not None else self.config.weights_for(version)
        return self._blend(signals, w, lambda arr: OutcomeProbs.from_array(arr))

    def blend_advance(
        self,
        signals: dict[str, AdvanceProbs | None],
        *,
        version: str | None = None,
        weights: dict[str, float] | None = None,
    ) -> BlendResult:
        """Blend knockout advance signals into one :class:`AdvanceProbs`."""
        w = weights if weights is not None else self.config.weights_for(version)
        return self._blend(signals, w, lambda arr: AdvanceProbs.from_array(arr))

    def is_flagged(self, result: BlendResult) -> bool:
        """True if signals disagree beyond the configured review threshold."""
        return result.max_pairwise_disagreement >= self.config.disagreement_threshold


@dataclass
class SignalDisagreement:
    """Pairwise disagreement of each signal against a reference distribution."""

    reference: str
    gaps: dict[str, float] = field(default_factory=dict)

    def worst(self) -> tuple[str, float] | None:
        if not self.gaps:
            return None
        name = max(self.gaps, key=lambda k: self.gaps[k])
        return name, self.gaps[name]


def disagreement_vs_reference(
    signals: dict[str, OutcomeProbs | None],
    reference: str,
) -> SignalDisagreement:
    """Total-variation gap of every available signal against ``reference``.

    A focused disagreement detector for explanation/triage: e.g. "how far is the
    market (or the LLM) from the historical model on this match?".
    """
    ref = signals.get(reference)
    if ref is None:
        raise ValueError(f"reference signal {reference!r} is unavailable")
    gaps = {
        name: disagreement(ref, probs)
        for name, probs in signals.items()
        if probs is not None and name != reference
    }
    return SignalDisagreement(reference=reference, gaps=gaps)
