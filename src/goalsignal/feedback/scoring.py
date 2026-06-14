"""Post-match scoring of frozen predictions.

All probabilities are read from the immutable prediction payload. The
probability of the actual exact scoreline is taken from the stored
`top_scorelines` when present; otherwise it is left as None here and may be
filled only by a validated reconstruction of the frozen goal model (see
`reconstruct_scoreline_probability`) — never inferred from W/D/L
probabilities and never invented.
"""

from __future__ import annotations

import math

import numpy as np

_LABELS = {"home_win": 0, "draw": 1, "away_win": 2}
_EPS = 1e-12


def score_prediction(payload: dict, result: dict) -> dict:
    """Realized performance of one frozen prediction against one result."""
    probs = np.array(
        [
            payload["home_win_probability"],
            payload["draw_probability"],
            payload["away_win_probability"],
        ],
        dtype=float,
    )
    outcome = result["outcome"]
    label = _LABELS[outcome]
    onehot = np.eye(3)[label]
    hg = int(result["regulation_home_goals"])
    ag = int(result["regulation_away_goals"])

    top = payload.get("top_scorelines") or []
    actual_in_top = [
        k + 1 for k, s in enumerate(top) if s["home"] == hg and s["away"] == ag
    ]
    rank = actual_in_top[0] if actual_in_top else None
    stored_p = top[rank - 1]["p"] if rank else None
    predicted_class = int(probs.argmax())
    best = top[0] if top else None

    lam_h = payload.get("home_expected_goals")
    lam_a = payload.get("away_expected_goals")
    return {
        "fixture_id": payload["fixture_id"],
        "home_team": payload.get("home_team"),
        "away_team": payload.get("away_team"),
        "actual_home_goals": hg,
        "actual_away_goals": ag,
        "actual_outcome": outcome,
        "probability_of_actual_outcome": float(probs[label]),
        "log_loss": float(-math.log(max(probs[label], _EPS))),
        "brier": float(((probs - onehot) ** 2).sum()),
        "rps": float((((np.cumsum(probs) - np.cumsum(onehot)) ** 2)[:2]).sum() / 2.0),
        "predicted_outcome": ["home_win", "draw", "away_win"][predicted_class],
        "predicted_outcome_correct": predicted_class == label,
        "home_goal_abs_error": abs(lam_h - hg) if lam_h is not None else None,
        "away_goal_abs_error": abs(lam_a - ag) if lam_a is not None else None,
        "total_goal_abs_error": abs((lam_h + lam_a) - (hg + ag))
        if lam_h is not None and lam_a is not None
        else None,
        "predicted_total_goals": (lam_h + lam_a)
        if lam_h is not None and lam_a is not None
        else None,
        "actual_total_goals": hg + ag,
        "top_scoreline": f"{best['home']}-{best['away']}" if best else None,
        "exact_score_correct": bool(
            best and best["home"] == hg and best["away"] == ag
        ),
        "actual_scoreline_probability": stored_p,
        "actual_scoreline_probability_source": "stored_top_scorelines" if rank else None,
        "actual_in_top1": rank == 1,
        "actual_in_top3": rank is not None and rank <= 3,
        "actual_in_top5": rank is not None and rank <= 5,
        "model_version": payload.get("model_version"),
        "score_model_version": payload.get("score_model_version"),
        "data_cutoff": payload.get("data_cutoff"),
        "dataset_version": payload.get("dataset_version"),
        "prediction_timestamp": payload.get("prediction_timestamp"),
        "result_completed_at": result.get("completed_at"),
        "result_recorded_at": result.get("recorded_at"),
        "result_source": result.get("source"),
    }


def reconstruct_scoreline_probability(
    payload: dict, goal_model, feats, home_goals: int, away_goals: int
) -> tuple[float | None, str]:
    """P(actual scoreline) from a reconstructed frozen goal model, validated.

    The reconstruction is accepted only if it reproduces the stored expected
    goals and every stored top scoreline to their recorded precision;
    otherwise (None, reason) is returned and the value must be reported as
    unavailable — not guessed.
    """
    lams = goal_model.predict_expected_goals(feats)
    lam_h, lam_a = float(lams[0, 0]), float(lams[0, 1])
    if round(lam_h, 4) != payload["home_expected_goals"] or round(
        lam_a, 4
    ) != payload["away_expected_goals"]:
        return None, (
            f"reconstructed expected goals ({lam_h:.4f}, {lam_a:.4f}) do not match "
            f"stored ({payload['home_expected_goals']}, {payload['away_expected_goals']})"
        )
    matrix = goal_model.score_matrix(lam_h, lam_a)
    from goalsignal.models.poisson import top_scorelines as _top

    recomputed = [
        {"home": h, "away": a, "p": round(p, 4)} for h, a, p in _top(matrix, 5)
    ]
    if recomputed != payload.get("top_scorelines"):
        return None, "reconstructed top scorelines do not match stored payload"
    if home_goals >= matrix.shape[0] or away_goals >= matrix.shape[1]:
        return None, "actual score outside the model's score grid"
    return float(matrix[home_goals, away_goals]), "validated_reconstruction"
