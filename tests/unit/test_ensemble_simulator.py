"""The ensemble simulation source drives the real simulator on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goalsignal.signals.base import OutcomeProbs, advance_from_outcome, davidson_outcome
from goalsignal.tournament.bracket_2026 import GROUPS, OfficialBracket
from goalsignal.tournament.ensemble_adapter import EnsembleGoalAdapter, reweight_matrix
from goalsignal.tournament.full_simulator import (
    _pair_resolution_probabilities,
    check_full_invariants,
    simulate_full_tournament,
)
from goalsignal.tournament.model_adapter import RatingsGoalAdapter


class _StubGoalModel:
    def predict_expected_goals(self, frame: pd.DataFrame) -> np.ndarray:
        d = frame["elo_diff"].to_numpy(dtype=float) / 400.0
        return np.column_stack([np.exp(0.2 + 0.3 * d), np.exp(0.2 - 0.3 * d)])

    def score_matrix(self, lam_home: float, lam_away: float) -> np.ndarray:
        from scipy.stats import poisson

        h = poisson.pmf(np.arange(8), lam_home)
        a = poisson.pmf(np.arange(8), lam_away)
        m = np.outer(h, a)
        return m / m.sum()


# --- reweight_matrix ----------------------------------------------------------


def test_reweight_matrix_hits_target_marginals():
    base = _StubGoalModel().score_matrix(1.4, 1.1)
    target = OutcomeProbs(0.55, 0.25, 0.20)
    out = reweight_matrix(base, target)
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-9)
    home = float(np.tril(out, -1).sum())
    draw = float(np.trace(out))
    away = float(np.triu(out, 1).sum())
    np.testing.assert_allclose([home, draw, away], [0.55, 0.25, 0.20], atol=1e-9)


# --- advance_probs hook in the resolution vector ------------------------------


class _AdvanceModel:
    """A model exposing advance_probs to exercise the resolution hook."""

    def __init__(self, p_home):
        self.base = RatingsGoalAdapter({"A": 1550.0, "B": 1450.0}, _StubGoalModel())
        self.p_home = p_home

    def expected_goals(self, home, away, neutral):
        return self.base.expected_goals(home, away, neutral)

    def score_matrix(self, lh, la):
        return self.base.score_matrix(lh, la)

    def advance_probs(self, home, away):
        return self.p_home, 1.0 - self.p_home


def test_pair_resolution_respects_advance_probs():
    vec = _pair_resolution_probabilities("A", "B", _AdvanceModel(0.8))
    home_mass = vec[[0, 2, 4]].sum()
    away_mass = vec[[1, 3, 5]].sum()
    np.testing.assert_allclose(home_mass, 0.8, atol=1e-9)
    np.testing.assert_allclose(away_mass, 0.2, atol=1e-9)


def test_pair_resolution_default_unchanged_without_hook():
    """A plain RatingsGoalAdapter (no advance_probs) keeps original behavior."""
    adapter = RatingsGoalAdapter({"A": 1550.0, "B": 1450.0}, _StubGoalModel())
    vec = _pair_resolution_probabilities("A", "B", adapter)
    np.testing.assert_allclose(vec.sum(), 1.0, atol=1e-9)
    assert vec[[0, 2, 4]].sum() > vec[[1, 3, 5]].sum()  # stronger A advances more


# --- full simulator end-to-end with the ensemble adapter ----------------------


def _synthetic_groups_and_fixtures():
    from goalsignal.tournament.simulator import GroupFixture

    groups = {g: [f"{g}{i}" for i in range(1, 5)] for g in sorted(GROUPS)}
    fixtures = []
    for g, teams in groups.items():
        for i in range(4):
            for j in range(i + 1, 4):
                fixtures.append(
                    GroupFixture(
                        group=g,
                        home=teams[i],
                        away=teams[j],
                        fixture_id=f"{g}-{i}{j}",
                        neutral=True,
                        played=False,
                    )
                )
    return groups, fixtures


def test_simulate_full_tournament_with_ensemble_adapter():
    groups, fixtures = _synthetic_groups_and_fixtures()
    teams = [t for ts in groups.values() for t in ts]
    ratings = {t: 1500.0 + 6.0 * (hash(t) % 21 - 10) for t in teams}
    base = RatingsGoalAdapter(ratings, _StubGoalModel())

    def blend_fn(home, away, neutral, knockout):
        outcome = davidson_outcome(ratings[home] - ratings[away])
        provenance = {"home": home, "away": away, "knockout": knockout}
        if knockout:
            return advance_from_outcome(outcome), provenance
        return outcome, provenance

    adapter = EnsembleGoalAdapter(base, blend_fn)
    bracket = OfficialBracket.load()
    result = simulate_full_tournament(
        groups, fixtures, adapter, bracket, n_sims=120, seed=7
    )
    assert check_full_invariants(result) == []
    champ_total = sum(result.advancement_probs[t]["champion"] for t in result.teams)
    assert champ_total == pytest.approx(1.0, abs=1e-9)
    # The adapter recorded provenance for the matchups it evaluated.
    assert len(adapter.provenance) > 0
