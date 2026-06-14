"""Chronological expanding-window backtesting.

Protocol per fold (test year Y):

    train      = matches before (Y - val_years)-01-01
    validation = matches in [Y - val_years, Y)
    test       = matches in year Y

Component models fit on train only; temperature calibrators and ensemble
weights fit on validation predictions only; test predictions are generated
once and never modified. Components are deliberately *not* refit on
train+validation before testing so the calibrators match the distribution
they were fit on — a documented, conservative choice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from goalsignal.evaluation import metrics as M
from goalsignal.models.baselines import (
    ContextFrequency,
    EloDavidson,
    EmpiricalFrequency,
    HigherRatedHeuristic,
    UniformBaseline,
)
from goalsignal.models.calibration import TemperatureScaler
from goalsignal.models.dixon_coles import DixonColesModel
from goalsignal.models.ensemble import ConvexEnsemble
from goalsignal.models.outcome_classifier import MultinomialLogistic
from goalsignal.models.poisson import PoissonGoalModel
from goalsignal.utils.paths import resolve

CALIBRATED = ["elo_davidson", "poisson", "dixon_coles", "multinomial_logistic"]
ENSEMBLE_COMPONENTS = ["elo_davidson", "dixon_coles", "multinomial_logistic"]


def _make_models():
    return {
        "uniform": UniformBaseline(),
        "empirical": EmpiricalFrequency(),
        "context_frequency": ContextFrequency(),
        "higher_rated": HigherRatedHeuristic(),
        "elo_davidson": EloDavidson(),
        "poisson": PoissonGoalModel(),
        "dixon_coles": DixonColesModel(),
        "multinomial_logistic": MultinomialLogistic(),
    }


@dataclass
class FoldResult:
    year: int
    metrics: pd.DataFrame  # one row per model
    predictions: pd.DataFrame
    goal_metrics: pd.DataFrame
    diagnostics: dict


def _goal_metrics(model, name: str, test: pd.DataFrame) -> dict | None:
    strict = test[test["strict_goal_model_eligible"]]
    if len(strict) == 0:
        return None
    lams = model.predict_expected_goals(strict)
    h = strict["home_score_recorded"].to_numpy(dtype=float)
    a = strict["away_score_recorded"].to_numpy(dtype=float)
    nll_sum, top1, top3 = 0.0, 0, 0
    for i in range(len(strict)):
        m = model.score_matrix(lams[i, 0], lams[i, 1])
        hi, ai = int(h[i]), int(a[i])
        p_actual = m[hi, ai] if hi < m.shape[0] and ai < m.shape[1] else 1e-12
        nll_sum -= np.log(max(p_actual, 1e-12))
        order = np.argsort(m.ravel())[::-1]
        actual_flat = hi * m.shape[1] + ai if hi < m.shape[0] and ai < m.shape[1] else -1
        top1 += int(actual_flat == order[0])
        top3 += int(actual_flat in order[:3])
    return {
        "model": name,
        "n": len(strict),
        "scoreline_nll": nll_sum / len(strict),
        "home_goal_mae": float(np.abs(lams[:, 0] - h).mean()),
        "away_goal_mae": float(np.abs(lams[:, 1] - a).mean()),
        "total_goal_mae": float(np.abs(lams.sum(axis=1) - (h + a)).mean()),
        "top1_scoreline": top1 / len(strict),
        "top3_scoreline": top3 / len(strict),
    }


def run_fold(frame: pd.DataFrame, year: int, val_years: int = 3) -> FoldResult:
    val_start = pd.Timestamp(f"{year - val_years}-01-01")
    test_start = pd.Timestamp(f"{year}-01-01")
    test_end = pd.Timestamp(f"{year + 1}-01-01")

    labeled = frame[frame["label"] >= 0]
    train = labeled[labeled["date"] < val_start]
    val = labeled[(labeled["date"] >= val_start) & (labeled["date"] < test_start)]
    test = labeled[(labeled["date"] >= test_start) & (labeled["date"] < test_end)]
    if min(len(train), len(val), len(test)) < 50:
        raise ValueError(f"fold {year}: insufficient data "
                         f"(train={len(train)}, val={len(val)}, test={len(test)})")

    models = _make_models()
    val_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(train)
        val_probs[name] = model.predict_proba(val)
        test_probs[name] = model.predict_proba(test)

    y_val = val["label"].to_numpy()
    y_test = test["label"].to_numpy()

    diagnostics = {
        "davidson_nu": models["elo_davidson"].nu_,
        "dixon_coles_rho": models["dixon_coles"].rho_,
        "poisson_beta": [float(b) for b in models["poisson"].beta_],
        "temperatures": {},
        "ensemble_weights": {},
    }

    for name in CALIBRATED:
        scaler = TemperatureScaler().fit(val_probs[name], y_val)
        diagnostics["temperatures"][name] = scaler.temperature_
        val_probs[f"{name}_cal"] = scaler.transform(val_probs[name])
        test_probs[f"{name}_cal"] = scaler.transform(test_probs[name])

    ens = ConvexEnsemble().fit(
        {n: val_probs[f"{n}_cal"] for n in ENSEMBLE_COMPONENTS}, y_val
    )
    diagnostics["ensemble_weights"] = {k: float(v) for k, v in ens.weights_.items()}
    test_probs["ensemble"] = ens.predict_proba(
        {n: test_probs[f"{n}_cal"] for n in ENSEMBLE_COMPONENTS}
    )

    rows = [{"model": name, "year": year, **M.summarize(p, y_test)}
            for name, p in test_probs.items()]

    preds = test[
        ["canonical_match_id", "date", "home_team", "away_team", "tournament", "label"]
    ].copy()
    for name, p in test_probs.items():
        preds[f"{name}_home"] = p[:, 0]
        preds[f"{name}_draw"] = p[:, 1]
        preds[f"{name}_away"] = p[:, 2]

    goal_rows = []
    for name in ("poisson", "dixon_coles"):
        gm = _goal_metrics(models[name], name, test)
        if gm:
            goal_rows.append({**gm, "year": year})

    return FoldResult(
        year=year,
        metrics=pd.DataFrame(rows),
        predictions=preds,
        goal_metrics=pd.DataFrame(goal_rows),
        diagnostics=diagnostics,
    )


def run_backtest(
    frame: pd.DataFrame,
    start_year: int,
    end_year: int,
    val_years: int = 3,
    output_dir: str = "artifacts/reports/backtest",
) -> dict:
    folds = [run_fold(frame, y, val_years) for y in range(start_year, end_year + 1)]

    all_metrics = pd.concat([f.metrics for f in folds], ignore_index=True)
    all_preds = pd.concat([f.predictions for f in folds], ignore_index=True)
    all_goals = pd.concat([f.goal_metrics for f in folds], ignore_index=True)

    out = resolve(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(out / "metrics_by_fold.csv", index=False)
    all_goals.to_csv(out / "goal_metrics_by_fold.csv", index=False)
    all_preds.to_csv(out / "test_predictions.csv", index=False)

    # Pooled metrics with year-block bootstrap CIs for log loss.
    years = all_preds["date"].astype(str).str.slice(0, 4).to_numpy()
    y = all_preds["label"].to_numpy()
    model_names = sorted(
        {c[: -len("_home")] for c in all_preds.columns if c.endswith("_home")}
    )
    pooled = {}
    for name in model_names:
        p = all_preds[[f"{name}_home", f"{name}_draw", f"{name}_away"]].to_numpy()
        pooled[name] = {
            **M.summarize(p, y),
            "log_loss_ci": M.block_bootstrap_ci(p, y, years),
        }

    # Breakdown by competition type for the ensemble.
    def _bucket(t: str) -> str:
        tl = t.lower()
        if tl == "fifa world cup":
            return "world_cup"
        if "qualif" in tl:
            return "qualification"
        if tl == "friendly":
            return "friendly"
        return "other"

    buckets = all_preds["tournament"].map(_bucket).to_numpy()
    by_bucket = {}
    p_ens = all_preds[["ensemble_home", "ensemble_draw", "ensemble_away"]].to_numpy()
    for b in sorted(set(buckets)):
        mask = buckets == b
        by_bucket[b] = M.summarize(p_ens[mask], y[mask])

    summary = {
        "protocol": {
            "type": "expanding_window_yearly",
            "test_years": [start_year, end_year],
            "validation_years_before_test": val_years,
            "note": "components fit on train only; calibration and ensemble "
            "weights fit on validation predictions only",
        },
        "pooled": pooled,
        "ensemble_by_competition": by_bucket,
        "fold_diagnostics": {str(f.year): f.diagnostics for f in folds},
        "goal_metrics_pooled": {
            name: {
                k: float(v)
                for k, v in all_goals[all_goals["model"] == name]
                .drop(columns=["model", "year"])
                .mean()
                .items()
            }
            for name in all_goals["model"].unique()
        },
    }
    with open(out / "overall.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
