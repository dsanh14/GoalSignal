"""Drive the existing tournament simulator from blended ensemble probabilities.

The Monte Carlo simulator only ever asks a model for ``expected_goals`` and a
``score_matrix`` (group fixtures), and — for knockout ties — resolves a
6-bucket regulation/extra-time/penalty vector. This adapter wraps the normal
:class:`~goalsignal.tournament.model_adapter.RatingsGoalAdapter` (which supplies
the *scoreline shape*) and reweights its score matrix so the W/D/L marginals
match the ensemble's :class:`OutcomeProbs` for that matchup. For knockout ties it
also exposes ``advance_probs`` so the simulator's resolution matches the
ensemble's :class:`AdvanceProbs`.

Because the group W/D/L marginals are reweighted while the within-region
scoreline distribution is kept, goal-difference and goals-for tiebreakers keep
working and the entire vectorized simulator runs unchanged. This keeps the
historical (default) path byte-for-byte identical: a plain ``RatingsGoalAdapter``
has no ``advance_probs`` attribute, so the resolution code falls back to the
goal model.
"""

from __future__ import annotations

import numpy as np

from goalsignal.signals.base import OutcomeProbs

_EPS = 1e-12


def reweight_matrix(matrix: np.ndarray, outcome: OutcomeProbs) -> np.ndarray:
    """Rescale a score matrix so its W/D/L marginals equal ``outcome``.

    Home-win (lower triangle), draw (diagonal), and away-win (upper triangle)
    regions are each scaled to the target mass while their internal scoreline
    shape is preserved. Degenerate regions with ~0 base mass are left untouched;
    the result is renormalized to guard against that edge case.
    """
    m = matrix / matrix.sum()
    tril = np.tril(m, -1)
    triu = np.triu(m, 1)
    diag_vals = np.diag(m)
    w0, l0 = float(tril.sum()), float(triu.sum())
    d0 = float(diag_vals.sum())
    out = np.zeros_like(m)
    if w0 > _EPS:
        out += tril * (outcome.home_win / w0)
    if l0 > _EPS:
        out += triu * (outcome.away_win / l0)
    if d0 > _EPS:
        rows = np.arange(m.shape[0])
        out[rows, rows] += diag_vals * (outcome.draw / d0)
    total = out.sum()
    return out / total if total > _EPS else m


class EnsembleGoalAdapter:
    """A model adapter whose outcome marginals come from the meta-ensemble.

    Args:
        base: the underlying :class:`RatingsGoalAdapter` (scoreline shape +
            expected goals). Its ``unrated_teams`` set is surfaced unchanged.
        blend_fn: ``(home, away, neutral, knockout) -> (probs, provenance|None)``
            where ``probs`` is :class:`OutcomeProbs` for ``knockout=False`` and
            :class:`AdvanceProbs` otherwise. ``provenance`` (an
            :class:`~goalsignal.signals.api.EnsemblePrediction` or ``None``) is
            recorded per matchup for the run summary.
    """

    def __init__(self, base, blend_fn):
        self.base = base
        self.blend_fn = blend_fn
        self._stash: tuple[str, str, bool] | None = None
        self.provenance: dict[tuple[str, str], object] = {}

    @property
    def unrated_teams(self) -> set[str]:
        return self.base.unrated_teams

    def _record(self, home: str, away: str, provenance) -> None:
        if provenance is not None:
            self.provenance[(home, away)] = provenance

    def expected_goals(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        self._stash = (home, away, bool(neutral))
        return self.base.expected_goals(home, away, neutral)

    def score_matrix(self, lam_home: float, lam_away: float) -> np.ndarray:
        base_matrix = self.base.score_matrix(lam_home, lam_away)
        if self._stash is None:
            return base_matrix
        home, away, neutral = self._stash
        probs, provenance = self.blend_fn(home, away, neutral, False)
        self._record(home, away, provenance)
        return reweight_matrix(base_matrix, probs)

    def advance_probs(self, home: str, away: str) -> tuple[float, float]:
        """Ensemble P(home advances), P(away advances) for a knockout tie."""
        probs, provenance = self.blend_fn(home, away, True, True)
        self._record(home, away, provenance)
        return probs.team_a_advances, probs.team_b_advances
