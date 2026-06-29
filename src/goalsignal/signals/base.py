"""Core probability types and helpers shared by every signal.

Conventions (kept consistent with the rest of the codebase):

- Group-stage outcome order is ``[home_win, draw, away_win]`` mapped to integer
  labels ``[0, 1, 2]``. :meth:`OutcomeProbs.as_array` returns that order so the
  existing :mod:`goalsignal.evaluation.metrics` functions apply unchanged.
- "team_a" is the home team and "team_b" the away team for group matches; for
  knockout ties the labels are positional only (no venue is implied).

A *signal* is any callable/loader that yields, per match, one of these objects
or ``None`` when it has no information. Nothing here fits parameters to data or
touches match results, so signals carry no leakage risk on their own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OUTCOME_NAMES: tuple[str, str, str] = ("home_win", "draw", "away_win")
ADVANCE_NAMES: tuple[str, str] = ("team_a_advances", "team_b_advances")

_EPS = 1e-12


def _normalize(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if np.any(arr < -1e-9):
        raise ValueError(f"probabilities must be non-negative, got {values}")
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total <= _EPS:
        raise ValueError(f"probabilities sum to ~0 and cannot be normalized: {values}")
    return list(arr / total)


@dataclass(frozen=True)
class OutcomeProbs:
    """Calibrated group-stage outcome distribution (sums to 1)."""

    home_win: float
    draw: float
    away_win: float

    def __post_init__(self) -> None:
        h, d, a = _normalize([self.home_win, self.draw, self.away_win])
        object.__setattr__(self, "home_win", h)
        object.__setattr__(self, "draw", d)
        object.__setattr__(self, "away_win", a)

    def as_array(self) -> np.ndarray:
        """Return ``[home_win, draw, away_win]`` (label order 0, 1, 2)."""
        return np.array([self.home_win, self.draw, self.away_win], dtype=float)

    @classmethod
    def from_array(cls, arr) -> OutcomeProbs:
        a = np.asarray(arr, dtype=float).ravel()
        if a.size != 3:
            raise ValueError(f"expected 3 outcome probabilities, got {a.size}")
        return cls(float(a[0]), float(a[1]), float(a[2]))

    def to_dict(self) -> dict[str, float]:
        return {"home_win": self.home_win, "draw": self.draw, "away_win": self.away_win}

    def flip(self) -> OutcomeProbs:
        """Swap home/away perspective (draw unchanged).

        Used when a directional signal (e.g. market odds listed as team A vs
        team B) is matched to a fixture presented in the opposite order.
        """
        return OutcomeProbs(self.away_win, self.draw, self.home_win)


@dataclass(frozen=True)
class AdvanceProbs:
    """Knockout advancement distribution (sums to 1)."""

    team_a_advances: float
    team_b_advances: float

    def __post_init__(self) -> None:
        a, b = _normalize([self.team_a_advances, self.team_b_advances])
        object.__setattr__(self, "team_a_advances", a)
        object.__setattr__(self, "team_b_advances", b)

    def as_array(self) -> np.ndarray:
        """Return ``[team_a_advances, team_b_advances]``."""
        return np.array([self.team_a_advances, self.team_b_advances], dtype=float)

    @classmethod
    def from_array(cls, arr) -> AdvanceProbs:
        a = np.asarray(arr, dtype=float).ravel()
        if a.size != 2:
            raise ValueError(f"expected 2 advance probabilities, got {a.size}")
        return cls(float(a[0]), float(a[1]))

    def to_dict(self) -> dict[str, float]:
        return {
            "team_a_advances": self.team_a_advances,
            "team_b_advances": self.team_b_advances,
        }

    def flip(self) -> AdvanceProbs:
        """Swap team A / team B perspective."""
        return AdvanceProbs(self.team_b_advances, self.team_a_advances)


def davidson_outcome(
    advantage: float,
    *,
    scale: float = 400.0,
    nu: float = 1.0,
) -> OutcomeProbs:
    """Map a signed strength *advantage* to a W/D/L distribution.

    Uses the Davidson (1970) model — the same family as the Elo-Davidson
    baseline — so "adjustment" signals (squad, form, venue) that naturally
    produce a scalar edge can participate in the same probability space as the
    distribution signals (market, expert, historical model).

    Args:
        advantage: home-minus-away edge in Elo-like points (positive favours
            the home team / team A). 0 means a balanced tie.
        scale: Elo logistic scale (400 ⇒ a 400-point edge is a 10:1 strength
            ratio before the draw term).
        nu: draw propensity. Larger ``nu`` raises the draw probability; ``nu``
            of 0 collapses to a pure win/loss logistic.

    Returns:
        An :class:`OutcomeProbs`.
    """
    if nu < 0:
        raise ValueError("nu must be non-negative")
    r_home = 10.0 ** (advantage / (2.0 * scale))
    r_away = 10.0 ** (-advantage / (2.0 * scale))
    draw_term = nu * np.sqrt(r_home * r_away)
    denom = r_home + r_away + draw_term
    return OutcomeProbs(r_home / denom, draw_term / denom, r_away / denom)


def advance_from_outcome(
    outcome: OutcomeProbs,
    *,
    a_tiebreak_prob: float = 0.5,
) -> AdvanceProbs:
    """Convert a group-style W/D/L distribution into knockout advance odds.

    A knockout tie has no draws: a regulation/extra-time draw is decided by the
    remaining stages (extra time, then penalties). ``a_tiebreak_prob`` is the
    probability that team A survives a tie that reaches that point (0.5 by
    default — penalties are close to a coin flip). The full staged
    regulation/ET/shootout simulator lives in
    :mod:`goalsignal.tournament.knockout`; this is the closed-form reduction
    used when only an outcome distribution is available.
    """
    if not 0.0 <= a_tiebreak_prob <= 1.0:
        raise ValueError("a_tiebreak_prob must be in [0, 1]")
    a = outcome.home_win + a_tiebreak_prob * outcome.draw
    b = outcome.away_win + (1.0 - a_tiebreak_prob) * outcome.draw
    return AdvanceProbs(a, b)


def disagreement(a: OutcomeProbs, b: OutcomeProbs) -> float:
    """Total-variation distance between two distributions, in ``[0, 1]``.

    Used as a model-vs-market (or model-vs-expert) disagreement detector: 0 is
    perfect agreement, 1 is disjoint support. TVD is symmetric and bounded,
    which makes a readable threshold for "flag this match for review".
    """
    return float(0.5 * np.abs(a.as_array() - b.as_array()).sum())
