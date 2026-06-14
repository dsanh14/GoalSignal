"""Validation for exact-score prediction visibility (synthetic data only).

Proves: scorelines come from the goal model (not the W/D/L ensemble), score
quantities are numerically valid, display is read-only, and the ledger's
immutability is untouched by presentation code.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from goalsignal.ledger.display import (
    SCORE_MODEL_UNRECORDED,
    filter_entries,
    find_entry,
    flatten_entry,
    format_csv,
    format_json,
    format_table,
)
from goalsignal.ledger.storage import append_predictions, list_entries, verify_ledger
from goalsignal.live import score_summary
from goalsignal.models.poisson import PoissonGoalModel


@pytest.fixture(scope="module")
def goal_model():
    rng = np.random.default_rng(7)
    n = 2000
    elo_diff = rng.normal(0, 200, n)
    neutral = rng.random(n) < 0.5
    lam_h = np.exp(0.1 + 0.4 * elo_diff / 400 + 0.25 * (~neutral))
    lam_a = np.exp(0.1 - 0.4 * elo_diff / 400)
    frame = pd.DataFrame(
        {
            "elo_diff": elo_diff,
            "neutral": neutral,
            "home_score_recorded": rng.poisson(lam_h),
            "away_score_recorded": rng.poisson(lam_a),
            "strict_goal_model_eligible": True,
        }
    )
    return PoissonGoalModel().fit(frame)


@pytest.fixture
def feats():
    return pd.DataFrame({"elo_diff": [180.0], "neutral": [False]})


def test_score_summary_validity(goal_model, feats):
    s = score_summary(goal_model, feats, k=5)
    assert np.isfinite(s["home_expected_goals"]) and s["home_expected_goals"] > 0
    assert np.isfinite(s["away_expected_goals"]) and s["away_expected_goals"] > 0
    probs = [p for _, _, p in s["top_scorelines"]]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True)  # ordered by probability
    # Normalized matrix sums to one; the pre-normalization tail is tracked
    # and tiny rather than silently discarded.
    assert s["matrix"].sum() == pytest.approx(1.0)
    assert abs(s["tail_mass"]) < 1e-6
    assert s["max_goals"] == s["matrix"].shape[0] - 1


def test_scorelines_come_from_goal_model_not_wdl(goal_model, feats):
    s = score_summary(goal_model, feats, k=1)
    h, a, p = s["top_scorelines"][0]
    matrix = goal_model.score_matrix(
        s["home_expected_goals"], s["away_expected_goals"]
    )
    # The reported most-likely score is exactly the goal-model matrix argmax;
    # no W/D/L probability is involved anywhere in score_summary.
    assert matrix[h, a] == pytest.approx(p)
    assert (h, a) == np.unravel_index(matrix.argmax(), matrix.shape)


def _entry(fid, home="Atlantis", away="Ruritania", date="2030-01-01"):
    return {
        "fixture_id": fid,
        "home_team": home,
        "away_team": away,
        "kickoff_timestamp": date,
        "model_version": "test-v1",
        "home_win_probability": 0.5,
        "draw_probability": 0.3,
        "away_win_probability": 0.2,
        "home_expected_goals": 1.8,
        "away_expected_goals": 0.9,
        "top_scorelines": [
            {"home": 1, "away": 0, "p": 0.14},
            {"home": 2, "away": 0, "p": 0.11},
        ],
    }


@pytest.fixture
def ledger(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_predictions(
        [
            _entry("f1"),
            _entry("f2", home="Freedonia", away="Sylvania", date="2030-01-02"),
        ],
        path,
    )
    return path


def test_display_is_read_only_and_chain_survives(ledger):
    before = ledger.read_bytes()
    entries = list_entries(ledger)
    format_table(entries, top_scorelines=2)
    format_csv(entries)
    format_json(entries)
    flatten_entry(entries[0])
    assert ledger.read_bytes() == before  # byte-for-byte unchanged
    assert verify_ledger(ledger) == []


def test_flatten_reports_unrecorded_score_model(ledger):
    row = flatten_entry(list_entries(ledger)[0])
    assert row["likely_score"] == "1-0"
    assert row["likely_score_p"] == 0.14
    assert row["wdl_model"] == "test-v1"
    assert row["score_model"] == SCORE_MODEL_UNRECORDED  # honest: not backfilled


def test_filters(ledger):
    entries = list_entries(ledger)
    assert len(filter_entries(entries, team="sylv")) == 1
    assert len(filter_entries(entries, date="2030-01-01")) == 1
    assert len(filter_entries(entries, team="atlantis", date="2030-01-02")) == 0
    assert len(filter_entries(entries)) == 2


def test_find_entry_by_prefix(ledger):
    entries = list_entries(ledger)
    assert find_entry(entries, "f1")["payload"]["fixture_id"] == "f1"
    full_hash = entries[1]["entry_hash"]
    assert find_entry(entries, full_hash[:12])["entry_hash"] == full_hash
    assert find_entry(entries, "f") is None  # ambiguous -> no unique match


def test_formats(ledger):
    entries = list_entries(ledger)
    table = format_table(entries, top_scorelines=2)
    assert "Atlantis v Ruritania" in table and "1.80-0.90" in table
    assert "2-0" in table  # second scoreline shown
    csv_text = format_csv(entries)
    assert csv_text.splitlines()[0].startswith("date,home_team")
    assert len(csv_text.splitlines()) == 3
    payloads = json.loads(format_json(entries))
    assert len(payloads) == 2
    assert payloads[0]["top_scorelines"][0] == {"home": 1, "away": 0, "p": 0.14}
