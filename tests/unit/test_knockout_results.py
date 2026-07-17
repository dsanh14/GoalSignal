"""Tests for the confirmed knockout results overlay loader (synthetic data)."""

from __future__ import annotations

import pytest

from goalsignal.tournament.knockout_results import (
    KnockoutResult,
    load_knockout_results,
)

HEADER = "match_number,round,team_a,team_b,score_a,score_b,aet,penalties,winner,notes"


def _write(tmp_path, *rows):
    path = tmp_path / "results.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_loads_regulation_extra_time_and_penalty_results(tmp_path):
    path = _write(
        tmp_path,
        "73,round_of_32,Astoria,Borduria,0,1,false,false,Borduria,",
        "74,round_of_32,Cascadia,Drachenland,1,1,true,true,Drachenland,\"pens 4-2\"",
        "75,round_of_32,Elbonia,Florin,2,1,true,false,Elbonia,\"won in ET\"",
    )
    results = load_knockout_results(path)
    assert set(results) == {73, 74, 75}
    assert results[73].decided_by == "regulation"
    assert results[73].winner == "Borduria"
    assert results[73].loser == "Astoria"
    assert results[74].decided_by == "penalties"
    assert results[74].notes == "pens 4-2"
    assert results[75].decided_by == "extra_time"


def test_winner_only_rows_allow_blank_scores(tmp_path):
    path = _write(
        tmp_path,
        "80,round_of_32,Astoria,Borduria,,,false,false,Astoria,\"score not recorded\"",
    )
    result = load_knockout_results(path)[80]
    assert result.score_a is None and result.score_b is None
    assert result.winner == "Astoria"


def test_missing_file_yields_empty_overlay_unless_required(tmp_path):
    assert load_knockout_results(tmp_path / "nope.csv") == {}
    with pytest.raises(FileNotFoundError):
        load_knockout_results(tmp_path / "nope.csv", require=True)


@pytest.mark.parametrize(
    "row, message",
    [
        ("73,round_of_32,Astoria,Borduria,0,1,false,false,Cascadia,",
         "winner"),
        ("73,round_of_32,Astoria,Borduria,1,1,false,true,Astoria,",
         "penalties=true requires aet=true"),
        ("73,round_of_32,Astoria,Borduria,1,1,false,false,Astoria,",
         "cannot end level"),
        ("73,round_of_32,Astoria,Borduria,2,0,false,false,Borduria,",
         "contradicts the score"),
        ("73,round_of_32,Astoria,Borduria,2,1,true,true,Astoria,",
         "drawn score"),
        ("73,final,Astoria,Borduria,0,1,false,false,Borduria,",
         "does not match the official round"),
        ("42,round_of_32,Astoria,Borduria,0,1,false,false,Borduria,",
         "73-104"),
        ("73,round_of_32,Astoria,Astoria,0,1,false,false,Astoria,",
         "must differ"),
        ("73,round_of_32,Astoria,Borduria,2,,false,false,Astoria,",
         "both scores"),
        ("73,round_of_32,Astoria,Borduria,-1,0,false,false,Borduria,",
         "non-negative"),
        ("73,round_of_32,Astoria,Borduria,0,1,maybe,false,Borduria,",
         "true/false"),
    ],
)
def test_invalid_rows_are_rejected(tmp_path, row, message):
    path = _write(tmp_path, row)
    with pytest.raises(ValueError, match=message):
        load_knockout_results(path)


def test_duplicate_match_numbers_rejected(tmp_path):
    path = _write(
        tmp_path,
        "73,round_of_32,Astoria,Borduria,0,1,false,false,Borduria,",
        "73,round_of_32,Astoria,Borduria,0,1,false,false,Borduria,",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_knockout_results(path)


def test_missing_columns_rejected(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("match_number,round\n73,round_of_32\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_knockout_results(path)


def test_repository_default_results_file_is_valid():
    """The tracked manual overlay must always load cleanly."""
    results = load_knockout_results()
    assert results, "tracked knockout_results_2026.csv should not be empty"
    for result in results.values():
        assert isinstance(result, KnockoutResult)
        assert result.round in {"round_of_32", "round_of_16", "quarterfinal", "semifinal"}
        assert 73 <= result.match_number <= 104
    # Penalty wins recorded with score semantics (level after ET).
    penalty_winners = {
        r.winner for r in results.values() if r.decided_by == "penalties"
    }
    assert {"Paraguay", "Morocco", "Egypt", "Switzerland"} <= penalty_winners
    assert results[98].winner == "Spain"
    assert results[101].winner == "Spain"
