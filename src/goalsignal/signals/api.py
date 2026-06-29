"""Public match-level ensemble prediction API.

The single internal interface that CLI, simulation, and backtest code should use
to obtain a blended forecast for a match. It wires together the historical
adapter (real model), the manual signal files, and the configurable
meta-ensemble, and returns a result carrying full provenance.

Functions:

* :meth:`EnsemblePredictor.predict_match_ensemble` — group-stage W/D/L.
* :meth:`EnsemblePredictor.predict_knockout_ensemble` — knockout advancement.
* :meth:`EnsemblePredictor.predict_batch_ensemble` — a tidy DataFrame of many.

Each :class:`EnsemblePrediction` exposes the final probabilities, the model
version, the components used, the renormalized weights, the missing signals, the
disagreement score, whether the match was flagged, and where the historical
signal came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from goalsignal.signals.base import AdvanceProbs, OutcomeProbs
from goalsignal.signals.historical_adapter import (
    SOURCE_FIXTURE,
    SOURCE_UNAVAILABLE,
    LiveModelHistorical,
    UnavailableHistorical,
)
from goalsignal.signals.meta_ensemble import BlendResult, MetaEnsemble
from goalsignal.signals.pipeline import (
    ManualInputs,
    MatchSpec,
    build_signals,
)


@dataclass(frozen=True)
class EnsemblePrediction:
    """A blended forecast with full provenance."""

    match_id: str
    stage: str
    team_a: str
    team_b: str
    version: str
    probs: OutcomeProbs | AdvanceProbs
    components: list[str]
    used_weights: dict[str, float]
    missing: list[str]
    disagreement: float
    flagged: bool
    historical_source: str
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Flatten to a single dict suitable for a DataFrame row."""
        row = {
            "match_id": self.match_id,
            "stage": self.stage,
            "team_a": self.team_a,
            "team_b": self.team_b,
            "version": self.version,
            "components": "|".join(self.components),
            "weights": "|".join(f"{k}:{v:.3f}" for k, v in self.used_weights.items()),
            "missing_signals": "|".join(self.missing),
            "disagreement": round(self.disagreement, 4),
            "flagged": self.flagged,
            "historical_source": self.historical_source,
        }
        row.update({k: round(v, 4) for k, v in self.probs.to_dict().items()})
        return row


class EnsemblePredictor:
    """Blend historical + manual signals into match-level forecasts."""

    def __init__(
        self,
        inputs: ManualInputs,
        ensemble: MetaEnsemble | None = None,
        historical: LiveModelHistorical | UnavailableHistorical | None = None,
    ):
        self.inputs = inputs
        self.ensemble = ensemble or MetaEnsemble(inputs.config)
        # No live model => use whatever historical probs the spec/table carries.
        self.historical = historical or UnavailableHistorical()

    # -- core -------------------------------------------------------------------
    def _result_to_prediction(
        self,
        spec: MatchSpec,
        version: str,
        result: BlendResult,
        historical_source: str,
    ) -> EnsemblePrediction:
        return EnsemblePrediction(
            match_id=spec.match_id,
            stage=spec.stage,
            team_a=spec.team_a,
            team_b=spec.team_b,
            version=version,
            probs=result.probs,
            components=sorted(result.components),
            used_weights=result.used_weights,
            missing=result.missing,
            disagreement=result.max_pairwise_disagreement,
            flagged=self.ensemble.is_flagged(result),
            historical_source=historical_source,
        )

    def _resolve_historical(
        self, spec: MatchSpec
    ) -> tuple[MatchSpec, str]:
        """Fill the historical signal from the live model when available.

        Falls back to any historical probs already on the spec (fixture/sample
        data), else marks the historical signal unavailable. Returns the spec to
        blend (possibly with a refreshed ``historical``) and the provenance tag.
        """
        if isinstance(self.historical, UnavailableHistorical):
            source = SOURCE_FIXTURE if spec.historical is not None else SOURCE_UNAVAILABLE
            return spec, source
        sig = (
            self.historical.advance(spec.team_a, spec.team_b, spec.neutral)
            if spec.knockout
            else self.historical.outcome(spec.team_a, spec.team_b, spec.neutral)
        )
        probs = sig.advance if spec.knockout else sig.outcome
        if probs is None:
            # Live model could not produce one; keep the spec's own historical.
            source = SOURCE_FIXTURE if spec.historical is not None else sig.source
            return spec, source
        from dataclasses import replace

        return replace(spec, historical=probs), sig.source

    def predict(self, spec: MatchSpec, *, version: str = "final_ensemble") -> EnsemblePrediction:
        """Blend signals for one match spec (group or knockout)."""
        resolved, source = self._resolve_historical(spec)
        signals = build_signals(resolved, self.inputs)
        if spec.knockout:
            result = self.ensemble.blend_advance(signals, version=version)
        else:
            result = self.ensemble.blend(signals, version=version)
        return self._result_to_prediction(spec, version, result, source)

    # -- public, intention-revealing wrappers ----------------------------------
    def predict_match_ensemble(
        self, spec: MatchSpec, *, version: str = "final_ensemble"
    ) -> EnsemblePrediction:
        """Group-stage W/D/L prediction. Raises if given a knockout spec."""
        if spec.knockout:
            raise ValueError(f"{spec.match_id} is a knockout match; use predict_knockout_ensemble")
        return self.predict(spec, version=version)

    def predict_knockout_ensemble(
        self, spec: MatchSpec, *, version: str = "final_ensemble"
    ) -> EnsemblePrediction:
        """Knockout advancement prediction. Raises if given a group spec."""
        if not spec.knockout:
            raise ValueError(f"{spec.match_id} is a group match; use predict_match_ensemble")
        return self.predict(spec, version=version)

    def predict_batch_ensemble(
        self, specs: list[MatchSpec], *, version: str = "final_ensemble"
    ) -> pd.DataFrame:
        """Blend a batch of matches into a tidy DataFrame (one row per match)."""
        rows = [self.predict(spec, version=version).to_row() for spec in specs]
        return pd.DataFrame(rows)
