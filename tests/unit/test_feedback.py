"""Validation of the post-match feedback workflow (synthetic data only)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from goalsignal.feedback.results import (
    apply_results_overlay,
    record_result,
    verify_results,
)
from goalsignal.feedback.scoring import score_prediction
from goalsignal.ledger.storage import append_predictions, list_entries, verify_ledger
from goalsignal.ratings.elo import EloConfig, compute_elo

RESULT = {
    "fixture_id": "fix1",
    "regulation_home_goals": 1,
    "regulation_away_goals": 1,
    "outcome": "draw",
    "completed_at": "2026-06-12",
    "recorded_at": "2026-06-13T00:00:00+00:00",
    "source": "manual-confirmed-result",
}

PAYLOAD = {
    "fixture_id": "fix1",
    "home_team": "Atlantis",
    "away_team": "Ruritania",
    "model_version": "test-v1",
    "data_cutoff": "2026-06-12",
    "dataset_version": "abc",
    "prediction_timestamp": "2026-06-13T00:26:18+00:00",
    "home_win_probability": 0.7579,
    "draw_probability": 0.1617,
    "away_win_probability": 0.0804,
    "home_expected_goals": 2.5731,
    "away_expected_goals": 0.7569,
    "top_scorelines": [
        {"home": 2, "away": 0, "p": 0.1185},
        {"home": 3, "away": 0, "p": 0.1016},
        {"home": 1, "away": 0, "p": 0.0899},
        {"home": 2, "away": 1, "p": 0.0897},
        {"home": 3, "away": 1, "p": 0.0769},
    ],
}


# --- result store -----------------------------------------------------------


def test_record_and_verify(tmp_path):
    store = tmp_path / "results.jsonl"
    entry = record_result("fix1", 1, 1, "2026-06-12", "manual", path=store)
    assert entry["payload"]["outcome"] == "draw"
    assert entry["payload"]["completed_at_time_known"] is False
    assert "recorded_at" in entry["payload"]
    assert verify_results(store) == []


def test_duplicate_result_rejected(tmp_path):
    store = tmp_path / "results.jsonl"
    record_result("fix1", 1, 1, "2026-06-12", "manual", path=store)
    with pytest.raises(ValueError, match="already exists"):
        record_result("fix1", 1, 1, "2026-06-12", "manual", path=store)
    with pytest.raises(ValueError, match="already exists"):
        record_result("fix1", 2, 0, "2026-06-12", "manual", path=store)  # conflicting
    assert len(store.read_text().splitlines()) == 1  # safe rejection, store unchanged


def test_correction_requires_reason_and_target(tmp_path):
    store = tmp_path / "results.jsonl"
    first = record_result("fix1", 2, 0, "2026-06-12", "manual", path=store)
    with pytest.raises(ValueError, match="reason"):
        record_result("fix1", 1, 1, "2026-06-12", "m", path=store,
                      corrects=first["entry_hash"])
    fixed = record_result(
        "fix1", 1, 1, "2026-06-12", "manual", path=store,
        corrects=first["entry_hash"], correction_reason="score typo",
    )
    assert fixed["payload"]["corrects"] == first["entry_hash"]
    assert verify_results(store) == []


def test_completed_before_kickoff_rejected(tmp_path):
    with pytest.raises(ValueError, match="before the kickoff"):
        record_result("fix1", 1, 1, "2026-06-11", "manual",
                      kickoff_date="2026-06-12", path=tmp_path / "r.jsonl")


def test_result_store_separate_from_ledger(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    store = tmp_path / "results.jsonl"
    append_predictions([dict(PAYLOAD)], ledger)
    before = ledger.read_bytes()
    record_result("fix1", 1, 1, "2026-06-12", "manual", path=store)
    assert ledger.read_bytes() == before  # prediction untouched, byte-for-byte
    assert verify_ledger(ledger) == []
    assert store.exists() and store != ledger


# --- scoring ----------------------------------------------------------------


def test_scoring_draw_metrics():
    r = score_prediction(PAYLOAD, RESULT)
    assert r["actual_outcome"] == "draw"
    assert r["probability_of_actual_outcome"] == pytest.approx(0.1617)
    assert r["log_loss"] == pytest.approx(-math.log(0.1617))
    expected_brier = 0.7579**2 + (0.1617 - 1) ** 2 + 0.0804**2
    assert r["brier"] == pytest.approx(expected_brier)
    assert r["predicted_outcome"] == "home_win"
    assert r["predicted_outcome_correct"] is False
    assert r["exact_score_correct"] is False
    assert r["top_scoreline"] == "2-0"
    assert r["home_goal_abs_error"] == pytest.approx(1.5731)
    assert r["away_goal_abs_error"] == pytest.approx(0.2431)
    assert r["total_goal_abs_error"] == pytest.approx(1.33)


def test_scoreline_probability_never_fabricated():
    # 1-1 is not among the stored top scorelines: must be None with no source,
    # never inferred from W/D/L probabilities.
    r = score_prediction(PAYLOAD, RESULT)
    assert r["actual_scoreline_probability"] is None
    assert r["actual_scoreline_probability_source"] is None
    assert r["actual_in_top1"] is False
    assert r["actual_in_top3"] is False
    assert r["actual_in_top5"] is False
    # When the actual score IS stored, the stored value is used verbatim.
    res20 = {**RESULT, "regulation_home_goals": 2, "regulation_away_goals": 0,
             "outcome": "home_win"}
    r2 = score_prediction(PAYLOAD, res20)
    assert r2["actual_scoreline_probability"] == 0.1185
    assert r2["actual_scoreline_probability_source"] == "stored_top_scorelines"
    assert r2["actual_in_top1"] and r2["exact_score_correct"]


# --- state update -----------------------------------------------------------


def _frame(rows):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["source_row"] = df.index + 2
    return df


def test_elo_draw_moves_ratings_toward_each_other():
    # Strong home favourite draws: home must lose rating, away must gain.
    matches = _frame([
        {
            "canonical_match_id": "m1", "date": "2026-06-12",
            "home_team": "Atlantis", "away_team": "Ruritania",
            "tournament": "FIFA World Cup", "neutral": False, "status": "played",
            "home_score_recorded": 1, "away_score_recorded": 1,
            "regulation_outcome": "draw", "shootout_played": False,
            "shootout_winner": None,
        }
    ])
    config = EloConfig(importance=[])
    result = compute_elo(matches, config)
    row = result.timeline.iloc[0]
    # Equal pre-ratings but home advantage -> expected > 0.5 -> draw penalizes home.
    assert row["delta"] < 0
    assert row["home_elo_post"] < row["home_elo_pre"]
    assert row["away_elo_post"] > row["away_elo_pre"]


def test_overlay_marks_only_target_fixture(synthetic_config):
    from goalsignal.data.build_dataset import build
    from goalsignal.data.loaders import load_all

    matches = build(load_all(synthetic_config), synthetic_config).matches
    scheduled = matches[matches["status"] == "scheduled"]
    assert len(scheduled) == 1
    fid = scheduled.iloc[0]["canonical_match_id"]
    results = {fid: {"regulation_home_goals": 1, "regulation_away_goals": 1,
                     "outcome": "draw"}}
    before_others = matches[matches["canonical_match_id"] != fid].copy()
    overlaid, n = apply_results_overlay(matches, results)
    assert n == 1
    row = overlaid[overlaid["canonical_match_id"] == fid].iloc[0]
    assert row["status"] == "played"
    assert row["regulation_outcome"] == "draw"
    assert row["strict_goal_model_eligible"]
    # Every other row is untouched.
    after_others = overlaid[overlaid["canonical_match_id"] != fid]
    pd.testing.assert_frame_equal(
        before_others.reset_index(drop=True), after_others.reset_index(drop=True)
    )
    # Original frame not mutated in place.
    assert matches[matches["canonical_match_id"] == fid].iloc[0]["status"] == "scheduled"


def test_overlay_rejects_unknown_and_played(synthetic_config):
    from goalsignal.data.build_dataset import build
    from goalsignal.data.loaders import load_all

    matches = build(load_all(synthetic_config), synthetic_config).matches
    with pytest.raises(ValueError, match="unknown fixture"):
        apply_results_overlay(matches, {"nope": {"regulation_home_goals": 1,
                                                 "regulation_away_goals": 0,
                                                 "outcome": "home_win"}})
    played_id = matches[matches["status"] == "played"].iloc[0]["canonical_match_id"]
    with pytest.raises(ValueError, match="already played"):
        apply_results_overlay(matches, {played_id: {"regulation_home_goals": 1,
                                                    "regulation_away_goals": 0,
                                                    "outcome": "home_win"}})


# --- immutability of earlier predictions -------------------------------------


def test_future_predictions_append_under_new_version_only(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_predictions([dict(PAYLOAD)], ledger)
    original_first_line = ledger.read_text().splitlines()[0]
    # Same fixture + same model version: rejected (no silent refresh).
    with pytest.raises(ValueError, match="refusing to overwrite"):
        append_predictions([dict(PAYLOAD)], ledger)
    # Refresh after a result must carry a new model version: append succeeds,
    # earlier entry byte-identical, chain intact.
    refreshed = {**PAYLOAD, "model_version": "test-v1+r1",
                 "data_cutoff": "2026-06-13"}
    append_predictions([refreshed], ledger)
    lines = ledger.read_text().splitlines()
    assert lines[0] == original_first_line
    assert len(lines) == 2
    assert verify_ledger(ledger) == []
    entries = list_entries(ledger)
    assert entries[0]["payload"]["model_version"] == "test-v1"
    assert entries[1]["payload"]["model_version"] == "test-v1+r1"


def test_summary_aggregation_consistency():
    r = score_prediction(PAYLOAD, RESULT)
    assert np.isfinite(r["log_loss"]) and np.isfinite(r["brier"]) and np.isfinite(r["rps"])
    assert 0 <= r["rps"] <= 1
