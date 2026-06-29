"""External-signal layer for outcome-first forecasting.

GoalSignal's product output is calibrated *win/advance* probabilities, not exact
scorelines. This package provides a uniform abstraction — a **signal** — that
turns any information source (the historical statistical model, market odds,
squad strength, recent form, venue context, LLM/expert judgment) into the same
small probability object so they can be blended, compared, and renormalized.

- Group-stage matches use :class:`~goalsignal.signals.base.OutcomeProbs`
  (``home_win`` / ``draw`` / ``away_win``).
- Knockout matches use :class:`~goalsignal.signals.base.AdvanceProbs`
  (``team_a_advances`` / ``team_b_advances``).

Every loader is robust to missing optional fields and missing files: a signal
that has no information for a match returns ``None`` and the meta-ensemble
renormalizes the configured weights across the signals that *are* present.
"""

from __future__ import annotations

from goalsignal.signals.base import (
    AdvanceProbs,
    OutcomeProbs,
    advance_from_outcome,
    davidson_outcome,
    disagreement,
)

__all__ = [
    "AdvanceProbs",
    "OutcomeProbs",
    "advance_from_outcome",
    "davidson_outcome",
    "disagreement",
]
