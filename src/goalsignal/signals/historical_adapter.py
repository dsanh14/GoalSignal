"""Historical signal from the real GoalSignal model.

The ``historical`` signal in the ensemble should come from the deployed
statistical model wherever possible, not a hand-entered file. This adapter
converts the existing model's outputs into the standardized signal types:

- group-stage W/D/L → :class:`~goalsignal.signals.base.OutcomeProbs` (from the
  calibrated convex ensemble, ``LiveModel.predict_outcome``);
- knockout advancement → :class:`~goalsignal.signals.base.AdvanceProbs`, derived
  from the fitted goal model's regulation / extra-time / penalty resolution
  (the same decomposition the tournament simulator uses).

No model logic is duplicated: the adapter only calls the trained model's public
methods. Every conversion carries provenance (``live_model`` / ``fixture`` /
``unavailable``) so callers can tell where the historical number came from, and
a prediction that cannot be produced is returned as missing rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import skellam

from goalsignal.signals.base import AdvanceProbs, OutcomeProbs
from goalsignal.tournament.model_adapter import RatingsGoalAdapter

# Provenance tags for the historical signal.
SOURCE_LIVE = "live_model"
SOURCE_FIXTURE = "fixture"
SOURCE_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class HistoricalSignal:
    """A historical signal value plus where it came from.

    Exactly one of ``outcome`` / ``advance`` is populated for an available
    signal; both are ``None`` when ``source`` is :data:`SOURCE_UNAVAILABLE`.
    """

    outcome: OutcomeProbs | None
    advance: AdvanceProbs | None
    source: str
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.outcome is not None or self.advance is not None


def advance_probs_from_adapter(
    adapter: RatingsGoalAdapter,
    home: str,
    away: str,
    neutral: bool = True,
) -> AdvanceProbs:
    """Knockout advance probabilities from a goal adapter's score model.

    Mirrors the simulator's resolution: regulation outcome from the score
    matrix, a drawn regulation decided by extra time (independent Skellam at one
    third intensity) and then a 50/50 shootout. Returns P(home advances) /
    P(away advances).
    """
    lam_home, lam_away = adapter.expected_goals(home, away, neutral)
    matrix = adapter.score_matrix(lam_home, lam_away)
    matrix = matrix / matrix.sum()
    reg_home = float(np.tril(matrix, -1).sum())
    reg_away = float(np.triu(matrix, 1).sum())
    draw = float(np.trace(matrix))
    et_home = float(1.0 - skellam.cdf(0, lam_home / 3, lam_away / 3))
    et_away = float(skellam.cdf(-1, lam_home / 3, lam_away / 3))
    et_draw = float(skellam.pmf(0, lam_home / 3, lam_away / 3))
    p_home = reg_home + draw * (et_home + 0.5 * et_draw)
    p_away = reg_away + draw * (et_away + 0.5 * et_draw)
    return AdvanceProbs(p_home, p_away)


class LiveModelHistorical:
    """Adapter turning a trained :class:`~goalsignal.live.LiveModel` into signals.

    Accepts any object exposing ``feature_row(home, away, neutral)``,
    ``predict_outcome(frame)``, ``ratings`` and ``goal_model`` — so a lightweight
    stub can stand in for tests without training the full pipeline.
    """

    def __init__(self, live, goal_adapter: RatingsGoalAdapter | None = None):
        self.live = live
        self.goal_adapter = goal_adapter or RatingsGoalAdapter(
            live.ratings, live.goal_model
        )

    def outcome(self, home: str, away: str, neutral: bool = True) -> HistoricalSignal:
        """Group-stage W/D/L from the calibrated live ensemble."""
        try:
            feats = self.live.feature_row(home, away, neutral)
            probs = self.live.predict_outcome(feats)[0]
            return HistoricalSignal(OutcomeProbs.from_array(probs), None, SOURCE_LIVE)
        except Exception as exc:  # pragma: no cover - defensive; report as missing
            return HistoricalSignal(None, None, SOURCE_UNAVAILABLE, repr(exc))

    def advance(self, home: str, away: str, neutral: bool = True) -> HistoricalSignal:
        """Knockout advancement from the fitted goal model."""
        try:
            adv = advance_probs_from_adapter(self.goal_adapter, home, away, neutral)
            return HistoricalSignal(None, adv, SOURCE_LIVE)
        except Exception as exc:  # pragma: no cover - defensive; report as missing
            return HistoricalSignal(None, None, SOURCE_UNAVAILABLE, repr(exc))


class UnavailableHistorical:
    """Null historical provider — always returns a missing signal.

    Used when no trained model is available (e.g. sample-only runs). The
    ensemble then renormalizes its weights across the remaining signals.
    """

    def outcome(self, home: str, away: str, neutral: bool = True) -> HistoricalSignal:
        return HistoricalSignal(None, None, SOURCE_UNAVAILABLE, "no live model")

    def advance(self, home: str, away: str, neutral: bool = True) -> HistoricalSignal:
        return HistoricalSignal(None, None, SOURCE_UNAVAILABLE, "no live model")
