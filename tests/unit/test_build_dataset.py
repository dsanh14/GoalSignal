"""Unit tests for canonical dataset construction on synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from goalsignal.data.build_dataset import build
from goalsignal.data.loaders import load_all
from goalsignal.data.schemas import SCOPE_REGULATION, SCOPE_UNKNOWN
from goalsignal.utils.hashing import canonical_match_id


@pytest.fixture
def result(synthetic_config):
    raw = load_all(synthetic_config)
    return build(raw, synthetic_config)


def _match(result, date: str, home: str):
    m = result.matches
    rows = m[(m["date"] == pd.Timestamp(date)) & (m["home_team"] == home)]
    assert len(rows) == 1, f"expected exactly one match for {date} {home}"
    return rows.iloc[0]


def test_row_accounting(result):
    # 10 raw rows: 1 duplicate excluded, 1 identical-teams excluded,
    # 1 unparseable-score excluded -> 7 canonical (6 played + 1 scheduled).
    assert result.stats["raw_rows"] == 10
    assert result.stats["canonical_matches"] == 7
    assert result.stats["played_matches"] == 6
    assert result.stats["scheduled_matches"] == 1
    reasons = set(result.exclusions["reason"])
    assert reasons == {
        "duplicate_canonical_identity",
        "identical_teams",
        "unparseable_score",
    }


def test_nonstandard_score_string_parses(result):
    row = _match(result, "2001-06-14", "Ruritania")
    assert row["away_score_recorded"] == 0  # raw value was "00"


def test_scheduled_fixture_retained_without_scores(result):
    row = _match(result, "2030-01-01", "Atlantis")
    assert row["status"] == "scheduled"
    assert pd.isna(row["home_score_recorded"])
    assert row["regulation_outcome"] == "unknown"
    assert not row["strict_goal_model_eligible"]


def test_date_aware_team_normalization(result):
    row = _match(result, "2002-03-01", "Ruritania")
    assert row["home_team_raw"] == "Old Ruritania"
    assert row["home_team"] == "Ruritania"


def test_shootout_scope_semantics(result):
    # Tied Mythic Cup match with shootout: regulation outcome is a known draw,
    # exact regulation score is unknown, strict training excluded.
    row = _match(result, "2001-06-10", "Atlantis")
    assert row["shootout_played"]
    assert row["shootout_winner"] == "Atlantis"
    assert row["recorded_score_scope"] == SCOPE_UNKNOWN
    assert row["regulation_outcome"] == "draw"
    assert not row["regulation_score_known"]
    assert not row["strict_goal_model_eligible"]
    assert row["strict_exclusion_reason"] == "shootout_score_scope_unknown"


def test_non_shootout_match_is_regulation_scope(result):
    row = _match(result, "2000-01-01", "Atlantis")
    assert row["recorded_score_scope"] == SCOPE_REGULATION
    assert row["regulation_home_score"] == 2
    assert row["regulation_outcome"] == "home_win"
    assert row["strict_goal_model_eligible"]


def test_shootout_reconciliation_statuses(result):
    recon = result.shootout_reconciliation.set_index("source_row")
    assert recon.loc[2, "status"] == "matched"
    # Shootout joined to a decisively-scored match (3-0): flagged inconsistent.
    assert recon.loc[3, "status"] == "matched_score_not_tied"
    # No corresponding result row.
    assert recon.loc[4, "status"] == "unmatched"
    # Winner is not one of the participants.
    assert recon.loc[5, "status"] == "winner_not_participant"


def test_decisive_knockout_capable_flagged_for_possible_extra_time(result):
    sus = result.suspicious_scope
    # The 2-1 Mythic Cup match has no shootout, so extra time cannot be ruled
    # out; the 3-0 match is NOT flagged because its (inconsistent) shootout
    # routes it to scope-unknown handling instead.
    flagged = sus[sus["reasons"].str.contains("possible_extra_time")]
    assert list(flagged["home_team"]) == ["Sylvania"]


def test_canonical_id_ignores_score():
    a = canonical_match_id("2000-01-01", "Atlantis", "Ruritania", "Friendly", "X", "Y")
    b = canonical_match_id("2000-01-01", "Atlantis", "Ruritania", "Friendly", "X", "Y")
    c = canonical_match_id("2000-01-02", "Atlantis", "Ruritania", "Friendly", "X", "Y")
    assert a == b != c


def test_no_silent_drops(result):
    accounted = result.stats["canonical_matches"] + result.stats["excluded_rows"]
    assert accounted == result.stats["raw_rows"]
