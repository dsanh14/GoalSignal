"""Milestone A contract tests for the enrichment source layer.

These cover the parts implemented in Milestone A: schema validation, provenance
/availability rules, deterministic manifests, the leakage-safe FIFA as-of join,
player identity resolution, deterministic travel math, config loading, optional
-dependency guarding, and the rule that ingestion is deferred. No source is
ingested and no network call is made.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from goalsignal.data.sources import (
    EventSourceAdapter,
    RankingSourceAdapter,
    SourceAdapter,
)
from goalsignal.data.sources.api_football import (
    SUPPORTED_ENDPOINTS,
    ApiFootballAdapter,
)
from goalsignal.data.sources.base import (
    AvailabilityStatus,
    FeatureAvailabilityError,
    ForecastStage,
    LineupStatus,
    MilestoneNotImplementedError,
    ProvenanceEnvelope,
    SourceValidationError,
    assert_available_before,
    require_optional_dependency,
)
from goalsignal.data.sources.config import (
    ApiFootballConfig,
    EnrichmentConfig,
    FifaRankingsConfig,
    PlayerFeaturesConfig,
    SourcesConfig,
    StatsBombConfig,
)
from goalsignal.data.sources.fifa_rankings import FifaRankingsAdapter, as_of_ranking
from goalsignal.data.sources.manifests import (
    build_snapshot_manifest,
    compute_snapshot_id,
    write_manifest,
)
from goalsignal.data.sources.players import normalize_player_name, resolve_player
from goalsignal.data.sources.schemas import (
    FeatureRecord,
    FifaRankingRecord,
    LineupRecord,
)
from goalsignal.data.sources.statsbomb import StatsBombAdapter
from goalsignal.data.sources.throttle import RateLimiter
from goalsignal.data.sources.venues import altitude_change_m, haversine_km


def _prov(available_at="2026-06-01T12:00:00+00:00", release="2026-06-01"):
    return {
        "source": "test",
        "source_record_id": "r1",
        "retrieved_at": "2026-06-10T00:00:00+00:00",
        "available_at": available_at,
        "source_snapshot_hash": "deadbeef",
        "schema_version": 1,
    }


# --- protocol conformance ---------------------------------------------------


def test_adapters_conform_to_protocols():
    assert isinstance(StatsBombAdapter(), SourceAdapter)
    assert isinstance(StatsBombAdapter(), EventSourceAdapter)
    assert isinstance(FifaRankingsAdapter(), RankingSourceAdapter)
    # api-football is a client-based source (live access via ApiFootballClient);
    # its adapter exposes the lightweight name/role/coverage surface.
    adapter = ApiFootballAdapter()
    assert adapter.name == "api_football" and adapter.role == "live_fixtures"


# --- provenance + availability (rules 3, 4) ---------------------------------


def test_provenance_envelope_is_frozen_and_complete():
    p = ProvenanceEnvelope.model_validate(_prov())
    assert p.source_snapshot_hash == "deadbeef"
    with pytest.raises(Exception):  # noqa: B017 - frozen model rejects mutation
        p.source = "other"


def test_assert_available_before_rejects_future_info():
    pred = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    assert_available_before(datetime(2026, 6, 1, 11, 0, tzinfo=UTC), pred)  # ok
    assert_available_before(pred, pred)  # equality allowed
    with pytest.raises(FeatureAvailabilityError, match="future information"):
        assert_available_before(datetime(2026, 6, 1, 13, 0, tzinfo=UTC), pred)


# --- schema validation ------------------------------------------------------


def test_fifa_ranking_schema_valid_and_invalid():
    rec = FifaRankingRecord.model_validate(
        {"team": "Brazil", "rank": 1, "points": 1837.0,
         "ranking_release_date": "2026-04-03", "provenance": _prov(release="2026-04-03")}
    )
    assert rec.rank == 1
    # available_at before release is impossible.
    with pytest.raises(ValueError, match="not knowable before"):
        FifaRankingRecord.model_validate(
            {"team": "Brazil", "rank": 1, "points": 1837.0,
             "ranking_release_date": "2026-06-02",
             "provenance": _prov(available_at="2026-06-01T12:00:00+00:00")}
        )
    # rank must be >= 1.
    with pytest.raises(ValueError):
        FifaRankingRecord.model_validate(
            {"team": "X", "rank": 0, "points": 1.0,
             "ranking_release_date": "2026-01-01", "provenance": _prov()}
        )


def test_fifa_adapter_validate_reports_errors():
    adapter = FifaRankingsAdapter()
    good = adapter.validate(
        [{"team": "Spain", "rank": 2, "points": 1800.0,
          "ranking_release_date": "2026-04-03", "provenance": _prov(release="2026-04-03")}]
    )
    assert good[0]["team"] == "Spain"
    with pytest.raises(SourceValidationError, match="record 0 failed"):
        adapter.validate([{"team": "Spain"}])


def test_lineup_confirmed_requires_eleven_and_keeps_status_separate():
    base = {
        "source_fixture_id": "f1", "formation": "4-3-3",
        "starting_xi": [f"p{i}" for i in range(11)], "provenance": _prov(),
    }
    confirmed = LineupRecord.model_validate({**base, "lineup_status": "confirmed"})
    assert confirmed.lineup_status is LineupStatus.CONFIRMED
    # Expected lineup may be partial.
    expected = LineupRecord.model_validate(
        {**base, "lineup_status": "expected", "starting_xi": ["p0", "p1"]}
    )
    assert expected.lineup_status is LineupStatus.EXPECTED
    # Confirmed with != 11 starters is rejected.
    with pytest.raises(ValueError, match="exactly 11"):
        LineupRecord.model_validate(
            {**base, "lineup_status": "confirmed", "starting_xi": ["p0"]}
        )


def test_feature_record_missing_is_not_zero():
    ok = FeatureRecord.model_validate(
        {"fixture_id": "f1", "prediction_timestamp": "2026-06-01T00:00:00+00:00",
         "feature_name": "x", "feature_value": 1.2, "feature_source": "s",
         "feature_available_at": "2026-05-30T00:00:00+00:00"}
    )
    assert ok.missing is False
    missing = FeatureRecord.model_validate(
        {"fixture_id": "f1", "prediction_timestamp": "2026-06-01T00:00:00+00:00",
         "feature_name": "x", "feature_value": None, "feature_source": "s",
         "feature_available_at": None, "missing": True}
    )
    assert missing.missing is True
    # A "missing" feature carrying a zero value is rejected (no silent zero-fill).
    with pytest.raises(ValueError, match="must not carry a value"):
        FeatureRecord.model_validate(
            {"fixture_id": "f1", "prediction_timestamp": "2026-06-01T00:00:00+00:00",
             "feature_name": "x", "feature_value": 0.0, "feature_source": "s",
             "feature_available_at": None, "missing": True}
        )


# --- manifests --------------------------------------------------------------


def test_snapshot_id_deterministic_and_content_sensitive():
    a = compute_snapshot_id("statsbomb", "competitions.json", "hashA", 1, {"k": 1})
    b = compute_snapshot_id("statsbomb", "competitions.json", "hashA", 1, {"k": 1})
    c = compute_snapshot_id("statsbomb", "competitions.json", "hashB", 1, {"k": 1})
    assert a == b != c
    assert len(a) == 16


def test_manifest_write_and_no_silent_overwrite(tmp_path):
    m = build_snapshot_manifest(
        source="statsbomb", role="event_enrichment", endpoint_or_url="competitions.json",
        available_at_semantics="match completion", license="StatsBomb Open Data",
        attribution="StatsBomb", content_hash="hashA", row_count=10, schema_version=1,
        cache_path="data/external/statsbomb/competitions.json",
    )
    p = write_manifest(m, directory=str(tmp_path))
    assert p.exists()
    write_manifest(m, directory=str(tmp_path))  # identical re-write is a no-op
    tampered = m.model_copy(update={"row_count": 99})
    with pytest.raises(FileExistsError, match="different content"):
        write_manifest(tampered.model_copy(update={"snapshot_id": m.snapshot_id}),
                       directory=str(tmp_path))


# --- FIFA as-of join (temporal test 17) -------------------------------------


def test_as_of_ranking_never_returns_future():
    import pandas as pd

    df = pd.DataFrame(
        {
            "team": ["Brazil", "Brazil", "Brazil"],
            "rank": [3, 2, 1],
            "points": [1800, 1820, 1840],
            "ranking_release_date": ["2026-02-01", "2026-04-01", "2026-06-19"],
        }
    )
    got = as_of_ranking(df, "Brazil", "2026-06-10")
    assert got["rank"] == 2  # latest release before the match
    assert got["days_since_ranking_release"] == 70
    # A match before any release returns None, never the nearest future ranking.
    assert as_of_ranking(df, "Brazil", "2026-01-01") is None
    # A release exactly on match day is excluded (may post-date kickoff).
    assert as_of_ranking(df, "Brazil", "2026-06-19")["rank"] == 2
    assert as_of_ranking(df, "Spain", "2026-06-10") is None


# --- player identity (tests 12-16) ------------------------------------------


CANDIDATES = [
    {"canonical_player_id": "P1", "full_name": "Luka Modric", "date_of_birth": "1985-09-09",
     "nationality": "Croatia", "club": "Real Madrid", "source_player_ids": {"sb": "100"}},
    {"canonical_player_id": "P2", "full_name": "Luka Modric", "date_of_birth": "2003-01-01",
     "nationality": "Croatia", "club": "Lokomotiva", "source_player_ids": {"sb": "200"}},
]


def test_resolve_by_source_id():
    out = resolve_player({"source_player_ids": {"sb": "200"}}, CANDIDATES, "sb")
    assert out["status"] == "matched" and out["canonical_player_id"] == "P2"


def test_name_only_is_ambiguous_not_merged():
    out = resolve_player({"full_name": "Luka Modrić"}, CANDIDATES, "sb")
    assert out["status"] == "ambiguous" and out["canonical_player_id"] is None


def test_dob_disambiguates():
    out = resolve_player(
        {"full_name": "Luka Modric", "date_of_birth": "1985-09-09"}, CANDIDATES, "sb"
    )
    assert out["status"] == "matched" and out["canonical_player_id"] == "P1"


def test_nationality_club_disambiguates():
    out = resolve_player(
        {"full_name": "Luka Modric", "nationality": "Croatia", "club": "Lokomotiva"},
        CANDIDATES, "sb",
    )
    assert out["status"] == "matched" and out["canonical_player_id"] == "P2"


def test_unmatched_when_no_name():
    out = resolve_player({"full_name": "Unknown Person"}, CANDIDATES, "sb")
    assert out["status"] == "unmatched"
    assert normalize_player_name("Luka Modrić") == "luka modric"


# --- travel determinism (test 27) -------------------------------------------


def test_haversine_is_deterministic_and_known():
    # London (~51.5,-0.13) to Paris (~48.85,2.35) is ~340 km.
    d1 = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
    d2 = haversine_km(51.5074, -0.1278, 48.8566, 2.3522)
    assert d1 == d2
    assert 330 < d1 < 350
    assert haversine_km(0, 0, 0, 0) == 0.0
    assert altitude_change_m(0, 2240) == 2240  # sea level to Mexico City


# --- config loading ---------------------------------------------------------


def test_all_configs_load_from_repo():
    assert isinstance(SourcesConfig.load(), SourcesConfig)
    af = ApiFootballConfig.load()
    assert af.auth_header == "x-apisports-key"
    assert af.base_url == "https://v3.football.api-sports.io"
    assert af.daily_request_limit == 100
    assert af.credential_env == "FOOTBALL_DATA_API_KEY"
    assert StatsBombConfig.load().optional_dependency == "statsbombpy"
    assert FifaRankingsConfig.load().expected_columns[0] == "team"
    assert PlayerFeaturesConfig.load().position_groups[0] == "goalkeeper"
    assert EnrichmentConfig.load().enabled is False


def test_sources_config_credential_gating(monkeypatch):
    cfg = SourcesConfig.load()
    af = next(s for s in cfg.sources if s.name == "api_football")
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    assert af.is_configured() is False
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "secret")
    assert af.is_configured() is True


# --- rate limiter (offline) -------------------------------------------------


def test_rate_limiter_sliding_window():
    clock = {"t": 0.0}
    rl = RateLimiter(2, now=lambda: clock["t"])
    assert rl.allow()
    rl.record()
    rl.record()
    assert rl.allow() is False
    assert rl.wait_time() == pytest.approx(60.0)
    clock["t"] = 61.0
    assert rl.allow() is True
    assert rl.wait_time() == 0.0


def test_api_football_supported_endpoints():
    # injuries is a real API-Football endpoint (its World Cup *population* is
    # measured separately); a made-up endpoint is not supported.
    assert "fixtures" in SUPPORTED_ENDPOINTS
    assert "injuries" in SUPPORTED_ENDPOINTS
    adapter = ApiFootballAdapter()
    assert adapter.is_supported("fixtures") is True
    assert adapter.is_supported("not_a_real_endpoint") is False


# --- legacy adapter placeholders / optional deps ----------------------------


def test_legacy_adapter_placeholders_redirect_to_milestone_b_entrypoints():
    # The Milestone A adapter stubs remain placeholders; real ingestion now
    # lives in dedicated loader/client classes (see test_sources_ingestion.py).
    with pytest.raises(MilestoneNotImplementedError):
        StatsBombAdapter().load()
    with pytest.raises(MilestoneNotImplementedError):
        FifaRankingsAdapter().load()


def test_optional_dependency_guard_is_actionable():
    with pytest.raises(MilestoneNotImplementedError, match="uv sync --extra"):
        require_optional_dependency("a_package_that_does_not_exist_xyz", "enrichment")


def test_forecast_stage_enum():
    assert ForecastStage.EARLY.value == "early"
    assert AvailabilityStatus.SUSPENDED.value == "suspended"
