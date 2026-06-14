"""Tests for the real-data source modules (FIFA ingest, WC validation,
player-data audit). Synthetic fixtures only — never the real datasets.
"""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from goalsignal.data.sources.config import (
    FifaRankingsConfig,
    PlayerDataConfig,
    SourcePathError,
    validate_source_path,
)
from goalsignal.data.sources.fifa_ingest import (
    as_of_fifa,
    load_fifa_historical,
    quality_report,
)
from goalsignal.data.sources.fifa_wc_validation import validate as wc_validate
from goalsignal.data.sources.player_data import (
    PlayerDataSource,
    PlayerDataUnavailable,
    build_coverage,
    build_inventory,
    temporal_audit,
)
from goalsignal.data.sources.readiness import team_alias_audit

# --- config: two separate FIFA files + path validation ----------------------


def test_fifa_config_has_two_separate_files():
    cfg = FifaRankingsConfig.load()
    assert cfg.path_env == "FIFA_RANKINGS_PATH"
    assert cfg.wc_teams_path_env == "FIFA_WC_TEAMS_PATH"
    assert cfg.path_env != cfg.wc_teams_path_env


def test_validate_source_path_errors(tmp_path):
    with pytest.raises(SourcePathError, match="not configured"):
        validate_source_path("", kind="file")
    with pytest.raises(SourcePathError, match="does not exist"):
        validate_source_path(str(tmp_path / "nope.csv"), kind="file")
    d = tmp_path / "adir"
    d.mkdir()
    with pytest.raises(SourcePathError, match="expected a file"):
        validate_source_path(str(d), kind="file")
    f = tmp_path / "x.txt"
    f.write_text("hi")
    with pytest.raises(SourcePathError, match="unsupported extension"):
        validate_source_path(str(f), kind="file", extensions=(".csv",))


# --- FIFA ingestion: real schema, rank reconstruction -----------------------


def _fifa_csv(tmp_path, rows):
    p = tmp_path / "ranking_fifa_historical.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_fifa_schema_detection_and_rank_reconstruction(tmp_path):
    # Two releases; ranks reconstructed by points desc. Note no rank column.
    rows = [
        {"team": "Brazil", "total_points": 1800, "date": "2018-06-07",
         "id": "id1", "id_num": 1, "team_short": "BRA"},
        {"team": "Germany", "total_points": 1850, "date": "2018-06-07",
         "id": "id1", "id_num": 1, "team_short": "GER"},
        {"team": "Spain", "total_points": 1700, "date": "2018-06-07",
         "id": "id1", "id_num": 1, "team_short": "ESP"},
        {"team": "Brazil", "total_points": 1820, "date": "2022-10-06",
         "id": "id2", "id_num": 2, "team_short": "BRA"},
    ]
    df, manifest = load_fifa_historical(_fifa_csv(tmp_path, rows))
    r2018 = df[df["ranking_release_date"] == "2018-06-07"].set_index("team")
    assert int(r2018.loc["Germany", "fifa_rank"]) == 1  # highest points -> rank 1
    assert int(r2018.loc["Brazil", "fifa_rank"]) == 2
    assert int(r2018.loc["Spain", "fifa_rank"]) == 3
    assert manifest["source"] == "fifa_rankings" and manifest["row_count"] == 4


def test_fifa_deterministic_tie_policy(tmp_path):
    rows = [
        {"team": "A", "total_points": 1000, "date": "2020-01-01"},
        {"team": "B", "total_points": 1000, "date": "2020-01-01"},
        {"team": "C", "total_points": 900, "date": "2020-01-01"},
    ]
    df, _ = load_fifa_historical(_fifa_csv(tmp_path, rows))
    by_team = df.set_index("team")
    # Standard competition ranking: tied A,B both rank 1; C gets rank 3.
    assert int(by_team.loc["A", "fifa_rank"]) == 1
    assert int(by_team.loc["B", "fifa_rank"]) == 1
    assert int(by_team.loc["C", "fifa_rank"]) == 3


def test_fifa_missing_points_preserved_not_zeroed(tmp_path):
    rows = [
        {"team": "A", "total_points": 1000, "date": "2020-01-01"},
        {"team": "B", "total_points": "", "date": "2020-01-01"},  # missing points
    ]
    df, _ = load_fifa_historical(_fifa_csv(tmp_path, rows))
    b = df[df["team"] == "B"].iloc[0]
    assert pd.isna(b["fifa_points"])  # not zero-filled
    assert pd.isna(b["fifa_rank"])  # rank not inferred when points absent
    q = quality_report(df)
    assert q["missing_points"] == 1


def test_fifa_as_of_no_future_release(tmp_path):
    rows = [
        {"team": "Brazil", "total_points": 1800, "date": "2018-06-07"},
        {"team": "Brazil", "total_points": 1820, "date": "2022-10-06"},
    ]
    df, _ = load_fifa_historical(_fifa_csv(tmp_path, rows))
    # Exact as-of: before the 2022 WC uses the 2018 release for a 2019 match.
    got = as_of_fifa(df, "Brazil", "2019-01-01")
    assert got["ranking_release_date"] == "2018-06-07"
    # Future release never selected; before any release -> None.
    assert as_of_fifa(df, "Brazil", "2017-01-01") is None
    # A match after the last release uses the last release, flagged by staleness.
    stale = as_of_fifa(df, "Brazil", "2026-06-01")
    assert stale["ranking_release_date"] == "2022-10-06"
    assert stale["days_since_release"] > 1000


# --- WC validation ----------------------------------------------------------


def test_wc_validation_classification(tmp_path):
    fifa_rows = [
        {"team": "Brazil", "total_points": 1800, "date": "2018-06-07"},
        {"team": "Germany", "total_points": 1850, "date": "2018-06-07"},
    ]
    fifa, _ = load_fifa_historical(_fifa_csv(tmp_path, fifa_rows))
    wc = pd.DataFrame({
        "tournament_year": [2018, 2018, 2018],
        "team": ["Germany", "Brazil", "Narnia"],
        "confederation": ["UEFA", "CONMEBOL", "X"],
        "published_pre_tournament_rank": [1, 5, 10],
        "source_row": [2, 3, 4],
    })
    wc["normalized_team"] = wc["team"].str.lower()
    frame, summary = wc_validate(wc, fifa)
    by_team = frame.set_index("team")
    assert by_team.loc["Germany", "classification"] == "exact_match"  # rec 1 == pub 1
    assert by_team.loc["Brazil", "classification"] == "large_discrepancy"  # rec 2 vs pub 5
    assert by_team.loc["Narnia", "classification"] == "unmatched_team"
    assert summary["total"] == 3


# --- team alias audit -------------------------------------------------------


def test_team_alias_audit_candidates_not_auto_accepted(tmp_path):
    fifa = pd.DataFrame({
        "team": ["Brazil", "Korea Republic", "Wakanda"],
        "normalized_team": ["brazil", "korea republic", "wakanda"],
    })
    audit = team_alias_audit(fifa, {"Brazil", "South Korea"}, out_dir=str(tmp_path))
    assert audit["exact_match"] == 1  # Brazil
    assert audit["alias_assisted_candidates"] == 1  # Korea Republic -> South Korea
    assert audit["unmatched"] == 1  # Wakanda
    cands = pd.read_csv(tmp_path / "team_source_alias_candidates.csv")
    # Alias suggestions are CANDIDATES, never auto-accepted.
    assert set(cands["review_status"]) <= {"candidate", "unmatched"}


# --- player data: read-only CSV.gz directory --------------------------------


def _tm_dir(tmp_path):
    d = tmp_path / "tm"
    d.mkdir()

    def _gz(name, df):
        with gzip.open(d / f"{name}.csv.gz", "wt", encoding="utf-8") as f:
            df.to_csv(f, index=False)

    _gz("players", pd.DataFrame({
        "player_id": [1, 2], "name": ["A", "B"], "date_of_birth": ["1990-01-01", "1992-02-02"],
        "position": ["Attack", "Defender"], "country_of_citizenship": ["Brazil", "Spain"],
        "current_club_id": [10, 11], "international_caps": [50, 0],
        "market_value_in_eur": [1000000, 500000]}))
    _gz("appearances", pd.DataFrame({
        "appearance_id": [1, 2], "game_id": [100, 101], "player_id": [1, 1],
        "date": ["2018-06-01", "2020-01-01"], "minutes_played": [90, 45],
        "goals": [1, 0], "assists": [0, 1], "competition_id": ["GB1", "GB1"]}))
    _gz("player_valuations", pd.DataFrame({
        "player_id": [1, 1], "date": ["2017-01-01", "2019-01-01"],
        "market_value_in_eur": [800000, 1200000]}))
    return d


def test_player_data_missing_path(monkeypatch):
    monkeypatch.delenv("PLAYER_DATA_PATH", raising=False)
    with pytest.raises(PlayerDataUnavailable, match="not configured"):
        PlayerDataSource.resolve_from_env()


def test_player_data_readonly_inventory_and_unchanged(tmp_path):
    d = _tm_dir(tmp_path)
    before = {p.name: p.read_bytes() for p in d.glob("*.gz")}
    src = PlayerDataSource(d, PlayerDataConfig())
    assert not src.is_duckdb
    assert set(src.table_names()) == {"players", "appearances", "player_valuations"}
    inv = build_inventory(src)
    assert inv["tables"]["players"]["rows"] == 2
    assert "player_id" in inv["tables"]["players"]["candidate_id_fields"]
    # Source files unchanged (read-only).
    after = {p.name: p.read_bytes() for p in d.glob("*.gz")}
    assert before == after


def test_player_data_temporal_classification(tmp_path):
    src = PlayerDataSource(_tm_dir(tmp_path), PlayerDataConfig())
    audit = temporal_audit(src)
    players = audit["classifications"]["players"]
    assert players["international_caps"] == "current_state_unsafe"
    assert players["market_value_in_eur"] == "current_state_unsafe"
    assert players["date_of_birth"] == "static_identity"
    assert audit["classifications"]["appearances"]["minutes_played"] == "dated_observation"
    assert audit["classifications"]["player_valuations"]["date"] == "dated_observation"


def test_player_data_coverage(tmp_path):
    src = PlayerDataSource(_tm_dir(tmp_path), PlayerDataConfig())
    cov = build_coverage(src)
    assert cov["players"]["total"] == 2
    assert cov["players"]["with_current_international_caps_gt0"] == 1  # only player A
    assert cov["appearances"]["total"] == 2
    assert cov["player_valuations"]["date_min"] == "2017-01-01"


def test_dated_observation_supports_cutoff(tmp_path):
    """A dated valuation/appearance can be filtered to before a cutoff (leakage-safe)."""
    src = PlayerDataSource(_tm_dir(tmp_path), PlayerDataConfig())
    vals = src.read_table("player_valuations")
    vals["date"] = pd.to_datetime(vals["date"])
    before_2018 = vals[vals["date"] < pd.Timestamp("2018-01-01")]
    assert len(before_2018) == 1  # only the 2017 valuation precedes the cutoff
