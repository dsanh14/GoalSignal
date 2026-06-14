"""Unit tests for outcome and goal models (synthetic data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goalsignal.evaluation.metrics import (
    brier_score,
    expected_calibration_error,
    log_loss,
    ranked_probability_score,
)
from goalsignal.models.baselines import (
    ContextFrequency,
    EloDavidson,
    EmpiricalFrequency,
    UniformBaseline,
)
from goalsignal.models.calibration import TemperatureScaler
from goalsignal.models.dixon_coles import DixonColesModel
from goalsignal.models.ensemble import ConvexEnsemble
from goalsignal.models.outcome_classifier import MultinomialLogistic
from goalsignal.models.poisson import (
    PoissonGoalModel,
    market_probs,
    outcome_probs,
    top_scorelines,
)


@pytest.fixture(scope="module")
def frame():
    """Synthetic frame where higher-Elo teams genuinely score more."""
    rng = np.random.default_rng(42)
    n = 4000
    elo_diff = rng.normal(0, 200, n)
    neutral = rng.random(n) < 0.3
    lam_h = np.exp(0.1 + 0.4 * elo_diff / 400 + 0.25 * (~neutral))
    lam_a = np.exp(0.1 - 0.4 * elo_diff / 400)
    h = rng.poisson(lam_h)
    a = rng.poisson(lam_a)
    label = np.where(h > a, 0, np.where(h == a, 1, 2))
    return pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "home_elo_pre": 1500 + elo_diff / 2,
            "away_elo_pre": 1500 - elo_diff / 2,
            "neutral": neutral,
            "home_score_recorded": h,
            "away_score_recorded": a,
            "label": label,
            "strict_goal_model_eligible": True,
        }
    )


def _check_valid_probs(p):
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_baselines_produce_valid_probabilities(frame):
    for model in (UniformBaseline(), EmpiricalFrequency(), ContextFrequency()):
        p = model.fit(frame).predict_proba(frame)
        _check_valid_probs(p)


def test_elo_davidson_beats_uniform_on_signal(frame):
    davidson = EloDavidson().fit(frame)
    p = davidson.predict_proba(frame)
    _check_valid_probs(p)
    y = frame["label"].to_numpy()
    assert log_loss(p, y) < log_loss(UniformBaseline().predict_proba(frame), y)
    assert davidson.nu_ > 0


def test_poisson_recovers_generating_coefficients(frame):
    model = PoissonGoalModel().fit(frame)
    # Generating process: b0=0.1, b1=0.4, b2=0.25.
    assert model.beta_[0] == pytest.approx(0.1, abs=0.05)
    assert model.beta_[1] == pytest.approx(0.4, abs=0.06)
    assert model.beta_[2] == pytest.approx(0.25, abs=0.06)
    lams = model.predict_expected_goals(frame)
    assert np.all(np.isfinite(lams)) and np.all(lams > 0)


def test_score_matrix_is_distribution(frame):
    model = PoissonGoalModel().fit(frame)
    m = model.score_matrix(1.5, 1.1)
    assert m.shape == (13, 13)
    assert m.min() >= 0
    assert m.sum() == pytest.approx(1.0)
    assert abs(model.last_tail_mass_) < 1e-6  # tail negligible and recorded
    p = outcome_probs(m)
    assert p.sum() == pytest.approx(1.0)
    markets = market_probs(m)
    assert markets["over_2_5"] + markets["under_2_5"] == pytest.approx(1.0)
    top = top_scorelines(m, 3)
    assert len(top) == 3 and top[0][2] >= top[1][2] >= top[2][2]


def test_dixon_coles_rho_zero_on_independent_data(frame):
    # Data generated with independent Poissons: rho should be near zero.
    model = DixonColesModel().fit(frame)
    assert abs(model.rho_) < 0.03
    _check_valid_probs(model.predict_proba(frame.head(50)))


def test_multinomial_logistic_valid_and_better_than_uniform(frame):
    model = MultinomialLogistic().fit(frame)
    p = model.predict_proba(frame)
    _check_valid_probs(p)
    y = frame["label"].to_numpy()
    assert log_loss(p, y) < log_loss(UniformBaseline().predict_proba(frame), y)


def test_temperature_scaling_improves_overconfident_probs():
    rng = np.random.default_rng(0)
    n = 2000
    true_p = np.full((n, 3), 1 / 3)
    labels = np.array([rng.choice(3, p=tp) for tp in true_p])
    # Overconfident model: sharpen the uniform truth.
    overconfident = np.tile([0.7, 0.2, 0.1], (n, 1))
    scaler = TemperatureScaler().fit(overconfident, labels)
    assert scaler.temperature_ > 1.0  # softening
    cal = scaler.transform(overconfident)
    assert log_loss(cal, labels) < log_loss(overconfident, labels)


def test_ensemble_weights_on_simplex(frame):
    y = frame["label"].to_numpy()
    good = EloDavidson().fit(frame).predict_proba(frame)
    bad = UniformBaseline().predict_proba(frame)
    ens = ConvexEnsemble().fit({"good": good, "bad": bad}, y)
    w = np.array(list(ens.weights_.values()))
    assert np.all(w >= 0) and w.sum() == pytest.approx(1.0)
    # Majority of the mass goes to the informative model; some uniform mass
    # can legitimately survive as regularization.
    assert ens.weights_["good"] > 0.5
    mix = ens.predict_proba({"good": good, "bad": bad})
    assert log_loss(mix, y) <= log_loss(bad, y)
    assert log_loss(mix, y) <= log_loss(good, y) + 1e-9


def test_metrics_sanity():
    perfect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    labels = np.array([0, 1])
    assert log_loss(perfect, labels) < 1e-9
    assert brier_score(perfect, labels) == 0.0
    assert ranked_probability_score(perfect, labels) == 0.0
    assert expected_calibration_error(perfect, labels) == pytest.approx(0.0)
