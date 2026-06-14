"""Milestone B ingestion tests (offline, synthetic data only).

Covers the API-Sports / API-Football client (via a fake transport — no
network), StatsBomb offline loading/aggregation, FIFA rankings ingestion,
fixture linking, and security/regression invariants. No live network test runs
here; the live probe is marked separately and skipped by default.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from goalsignal.data.sources.api_football import (
    ApiFootballClient,
    AuthError,
    MalformedResponseError,
    MissingApiKeyError,
    PlanLimitationError,
    RateLimitError,
    RequestTimeoutError,
    WrongHostError,
    parse_envelope,
)
from goalsignal.data.sources.api_football_normalize import (
    normalize_fixtures,
    normalize_injuries,
    normalize_leagues,
    normalize_lineups,
    normalize_predictions_benchmark,
)
from goalsignal.data.sources.cache import read_raw_snapshot, write_raw_snapshot
from goalsignal.data.sources.config import ApiFootballConfig
from goalsignal.data.sources.http_client import (
    FakeTransport,
    HttpResponse,
    TransportTimeout,
    redact_headers,
)
from goalsignal.data.sources.linking import link_fixtures, link_summary

FAKE_KEY = "SECRETKEY_DO_NOT_LEAK_123"
AUTH_HEADER = "x-apisports-key"


def _envelope(response, *, errors=None, results=None):
    return {
        "get": "x", "parameters": {}, "errors": errors or [],
        "results": results if results is not None else (
            len(response) if isinstance(response, list) else 1),
        "paging": {"current": 1, "total": 1}, "response": response,
    }


def _resp(obj, status=200, headers=None):
    body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    return HttpResponse(status, {"Content-Type": "application/json", **(headers or {})}, body)


def _client(transport, tmp_path, sleeps=None, **kw):
    cfg = ApiFootballConfig(cache_dir=str(tmp_path / "af"))
    sleeps = sleeps if sleeps is not None else []
    return ApiFootballClient(
        cfg, transport=transport, api_key=FAKE_KEY,
        sleep=lambda s: sleeps.append(s), **kw,
    )


# --- API-Football client (1-12) ---------------------------------------------


def test_base_url_and_auth_header(tmp_path):
    t = FakeTransport([_resp(_envelope([{"id": 1}]))])
    client = _client(t, tmp_path)
    client.leagues()
    # Sent to the API-Sports host with the x-apisports-key header (redacted here).
    assert t.calls[0]["url"].startswith("https://v3.football.api-sports.io/")
    assert AUTH_HEADER in t.calls[0]["headers"]
    assert t.calls[0]["headers"][AUTH_HEADER] == "***REDACTED***"


def test_missing_key_raises(tmp_path):
    client = _client(FakeTransport([]), tmp_path)
    client._api_key = ""
    with pytest.raises(MissingApiKeyError, match="no API key"):
        client.leagues()


def test_key_never_sent_to_another_host(tmp_path):
    # Host-lock: a config pointed at a different host has a different allowed
    # host, so a base_url mismatch is impossible; assert the guard rejects a
    # foreign URL directly.
    client = _client(FakeTransport([]), tmp_path)
    with pytest.raises(WrongHostError, match="only"):
        client._check_host("https://evil.example.com/leagues")
    # And the real call only ever targets the configured host.
    t = FakeTransport([_resp(_envelope([]))])
    c2 = _client(t, tmp_path)
    c2.leagues()
    assert "api-sports.io" in t.calls[0]["url"]


def test_secret_redaction_in_transport_and_cache(tmp_path):
    t = FakeTransport([_resp(_envelope([{"id": 1}]))])
    client = _client(t, tmp_path)
    client.leagues()
    assert t.calls[0]["headers"][AUTH_HEADER] == "***REDACTED***"
    assert FAKE_KEY not in json.dumps(t.calls)
    files = list((tmp_path / "af" / "raw").rglob("*.json"))
    blob = "\n".join(p.read_text() for p in files)
    assert FAKE_KEY not in blob
    request_meta = json.loads(next(p for p in files if p.name == "request.json").read_text())
    assert AUTH_HEADER not in json.dumps(request_meta)


def test_successful_response_and_cache(tmp_path):
    t = FakeTransport([_resp(_envelope([{"league": {"id": 1, "name": "World Cup"}}]))])
    client = _client(t, tmp_path)
    data, manifest = client.leagues({"search": "World Cup"})
    assert data["response"][0]["league"]["name"] == "World Cup"
    assert manifest["source"] == "api_football"


def test_auth_error_via_envelope(tmp_path):
    # API-Sports returns HTTP 200 with an errors field on a bad key.
    t = FakeTransport([_resp(_envelope([], errors={"token": "Error/Missing application key"}))])
    with pytest.raises(AuthError):
        _client(t, tmp_path).leagues()


def test_auth_error_via_http_status(tmp_path):
    t = FakeTransport([_resp({}, status=403)])
    with pytest.raises(AuthError):
        _client(t, tmp_path).leagues()


def test_plan_limitation_classified(tmp_path):
    t = FakeTransport([_resp(_envelope(
        [], errors={"plan": "Free plans do not have access to this season"}))])
    with pytest.raises(PlanLimitationError):
        _client(t, tmp_path).fixtures({"league": 1, "season": 2026})


def test_rate_limit_via_status_429(tmp_path):
    t = FakeTransport([_resp({}, status=429)])
    with pytest.raises(RateLimitError):
        _client(t, tmp_path).leagues()


def test_timeout_then_exhausted(tmp_path):
    def boom(*a):
        raise TransportTimeout("slow")

    t = FakeTransport([boom, boom, boom, boom])
    with pytest.raises(RequestTimeoutError):
        _client(t, tmp_path).leagues()


def test_malformed_json(tmp_path):
    t = FakeTransport([HttpResponse(200, {"Content-Type": "application/json"}, b"{bad")])
    with pytest.raises(MalformedResponseError):
        _client(t, tmp_path).leagues()


def test_envelope_validation_rejects_non_envelope(tmp_path):
    t = FakeTransport([_resp(["not", "an", "envelope"])])
    with pytest.raises(MalformedResponseError):
        _client(t, tmp_path).leagues()


def test_daily_limit_accounting_and_quota_stop(tmp_path):
    # daily_limit 5, reserve 3 -> only 2 usable requests.
    cfg = ApiFootballConfig(cache_dir=str(tmp_path / "af"), daily_request_limit=5,
                            daily_request_reserve=3, cache_first=False)
    t = FakeTransport([_resp(_envelope([{"id": i}])) for i in range(5)])
    client = ApiFootballClient(cfg, transport=t, api_key=FAKE_KEY, sleep=lambda s: None)
    client.leagues({"x": 1})
    client.leagues({"x": 2})
    assert client.usage.current() == 2
    from goalsignal.data.sources.throttle import DailyQuotaExceeded
    with pytest.raises(DailyQuotaExceeded):
        client.leagues({"x": 3})


def test_cache_first_avoids_duplicate_request(tmp_path):
    t = FakeTransport([_resp(_envelope([{"id": 1}]))])  # only ONE scripted response
    client = _client(t, tmp_path)
    client.leagues({"search": "World Cup"})
    used_after_first = client.usage.current()
    # Second identical call is served from cache: no new transport call, no quota.
    data, _manifest = client.leagues({"search": "World Cup"})
    assert data["response"][0]["id"] == 1
    assert len(t.calls) == 1  # transport hit exactly once
    assert client.usage.current() == used_after_first


def test_refresh_forces_new_request(tmp_path):
    t = FakeTransport([_resp(_envelope([{"id": 1}])), _resp(_envelope([{"id": 1}]))])
    client = _client(t, tmp_path)
    client.leagues({"search": "x"})
    client.leagues({"search": "x"}, refresh=True)
    assert len(t.calls) == 2


def test_cache_write_and_replay_no_overwrite(tmp_path):
    base = str(tmp_path / "af")
    body = json.dumps(_envelope([{"id": 1}])).encode()
    m1 = write_raw_snapshot(
        source="api_football", role="live_fixtures", endpoint="leagues",
        safe_path="leagues", params={}, response_body=body, response_headers={},
        available_at_semantics="x", schema_version=1, license="L", attribution="A",
        base_dir=base,
    )
    replay = read_raw_snapshot(base, m1["snapshot_id"])
    assert replay["response"]["response"][0]["id"] == 1
    assert AUTH_HEADER not in json.dumps(replay["request"])
    m2 = write_raw_snapshot(
        source="api_football", role="live_fixtures", endpoint="leagues",
        safe_path="leagues", params={}, response_body=body, response_headers={},
        available_at_semantics="x", schema_version=1, license="L", attribution="A",
        base_dir=base,
    )
    assert m1["snapshot_id"] == m2["snapshot_id"]


def test_redact_headers_helper():
    red = redact_headers({"x-apisports-key": "k", "Accept": "json"})
    assert red["x-apisports-key"] == "***REDACTED***" and red["Accept"] == "json"


# --- API-Football normalization (11-15) -------------------------------------


def test_league_discovery_and_coverage_flags():
    envelope = _envelope([{
        "league": {"id": 1, "name": "World Cup", "type": "Cup"},
        "country": {"name": "World"},
        "seasons": [{"year": 2026, "current": True,
                     "coverage": {"fixtures": {"lineups": True, "events": True},
                                  "standings": True, "injuries": False, "predictions": True}}],
    }])
    df = normalize_leagues(envelope, "snap", "t")
    assert df.iloc[0]["league_id"] == 1 and df.iloc[0]["country"] == "World"
    assert bool(df.iloc[0]["cov_fixtures_lineups"]) is True
    assert bool(df.iloc[0]["cov_injuries"]) is False  # WC has no injuries coverage


def test_fixture_normalization():
    envelope = _envelope([{
        "fixture": {"id": 100, "date": "2026-06-20T18:00:00+00:00",
                    "status": {"short": "FT", "long": "Match Finished"},
                    "venue": {"name": "Stadium", "city": "City"}},
        "league": {"id": 1, "season": 2026, "round": "Group A"},
        "teams": {"home": {"id": 1, "name": "Brazil"}, "away": {"id": 2, "name": "Serbia"}},
        "goals": {"home": 2, "away": 0},
    }])
    df = normalize_fixtures(envelope, "snap", "t")
    assert df.iloc[0]["provider_fixture_id"] == 100
    assert df.iloc[0]["home_team"] == "Brazil" and df.iloc[0]["home_goals"] == 2
    assert df.iloc[0]["status_short"] == "FT"
    assert df.iloc[0]["source_snapshot_id"] == "snap"


def test_lineup_normalization():
    envelope = _envelope([{
        "team": {"id": 1, "name": "Brazil"}, "formation": "4-3-3",
        "startXI": [{"player": {"id": 5, "name": "Neymar", "number": 10, "pos": "F"}}],
        "substitutes": [{"player": {"id": 9, "name": "Sub", "number": 20, "pos": "M"}}],
    }])
    df = normalize_lineups(envelope, "snap", "t")
    assert set(df["role"]) == {"start", "bench"}
    starter = df[df["role"] == "start"].iloc[0]
    assert starter["player_name"] == "Neymar" and starter["formation"] == "4-3-3"


def test_injury_normalization():
    envelope = _envelope([{
        "player": {"id": 5, "name": "Player"}, "team": {"id": 1, "name": "Brazil"},
        "fixture": {"id": 100}, "type": "Missing Fixture", "reason": "Knee Injury",
    }])
    df = normalize_injuries(envelope, "snap", "t")
    assert df.iloc[0]["player_name"] == "Player" and df.iloc[0]["reason"] == "Knee Injury"
    # Empty injuries response yields an empty frame (absence, not zero).
    assert normalize_injuries(_envelope([]), "snap", "t").empty


def test_predictions_stored_as_benchmark_only():
    envelope = _envelope([{
        "predictions": {"winner": {"id": 1, "name": "Brazil"},
                        "percent": {"home": "60%", "draw": "25%", "away": "15%"},
                        "advice": "Brazil"},
    }])
    df = normalize_predictions_benchmark(envelope, 100, "snap", "t")
    assert df.iloc[0]["predicted_winner"] == "Brazil"
    # The usage tag makes it impossible to mistake for a training feature.
    assert df.iloc[0]["usage"] == "external_benchmark"


def test_parse_envelope_helper():
    resp, results = parse_envelope(_envelope([{"a": 1}, {"a": 2}]))
    assert results == 2 and len(resp) == 2


# --- StatsBomb (13-20) ------------------------------------------------------


def _statsbomb_tree(tmp_path):
    root = tmp_path / "sb"
    (root / "data" / "matches" / "43").mkdir(parents=True)
    (root / "data" / "lineups").mkdir(parents=True)
    (root / "data" / "events").mkdir(parents=True)
    (root / "data" / "competitions.json").write_text(json.dumps([
        {"competition_id": 43, "season_id": 3, "competition_name": "FIFA World Cup",
         "season_name": "2018", "country_name": "International", "competition_gender": "male"},
    ]))
    (root / "data" / "matches" / "43" / "3.json").write_text(json.dumps([
        {"match_id": 7, "match_date": "2018-06-20",
         "competition": {"competition_name": "FIFA World Cup"},
         "season": {"season_name": "2018"},
         "home_team": {"home_team_name": "Brazil"},
         "away_team": {"away_team_name": "Serbia"}, "home_score": 2, "away_score": 0},
    ]))
    (root / "data" / "lineups" / "7.json").write_text(json.dumps([
        {"team_name": "Brazil", "lineup": [{"player_id": 1, "player_name": "Neymar",
                                            "jersey_number": 10}]},
        {"team_name": "Serbia", "lineup": [{"player_id": 2, "player_name": "Tadic",
                                            "jersey_number": 10}]},
    ]))
    (root / "data" / "events" / "7.json").write_text(json.dumps([
        {"type": {"name": "Shot"}, "team": {"name": "Brazil"},
         "play_pattern": {"name": "Regular Play"},
         "shot": {"statsbomb_xg": 0.5, "type": {"name": "Open Play"},
                  "outcome": {"name": "Goal"}}},
        {"type": {"name": "Shot"}, "team": {"name": "Brazil"},
         "play_pattern": {"name": "From Free Kick"},
         "shot": {"statsbomb_xg": 0.8, "type": {"name": "Penalty"},
                  "outcome": {"name": "Goal"}}},
        {"type": {"name": "Substitution"}, "team": {"name": "Serbia"}},
    ]))
    return root


def test_statsbomb_missing_path(monkeypatch):
    from goalsignal.data.sources.statsbomb import StatsBombDataUnavailable, resolve_statsbomb_path

    monkeypatch.delenv("STATSBOMB_DATA_PATH", raising=False)
    with pytest.raises(StatsBombDataUnavailable, match="open-data"):
        resolve_statsbomb_path()


def test_statsbomb_load_and_aggregate(tmp_path):
    from goalsignal.data.sources.statsbomb import StatsBombLoader

    loader = StatsBombLoader(_statsbomb_tree(tmp_path))
    comps, h1 = loader.load_competitions()
    assert comps.iloc[0]["competition_name"] == "FIFA World Cup"
    matches, _ = loader.load_matches(43, 3)
    assert matches.iloc[0]["statsbomb_match_id"] == 7
    lineup, _ = loader.load_lineup(7)
    assert set(lineup["team"]) == {"Brazil", "Serbia"}
    events, _ = loader.load_events(7)
    agg = loader.aggregate_events(events).set_index("team")
    assert agg.loc["Brazil", "shots"] == 2
    assert agg.loc["Brazil", "xg"] == pytest.approx(1.3)
    assert agg.loc["Brazil", "np_xg"] == pytest.approx(0.5)  # penalty excluded
    assert agg.loc["Brazil", "goals"] == 2
    assert agg.loc["Serbia", "substitutions"] == 1
    # Deterministic manifest input: same file -> same hash.
    _, h1b = loader.load_competitions()
    assert h1 == h1b


def test_statsbomb_coverage_and_malformed(tmp_path):
    from goalsignal.data.sources.statsbomb import StatsBombLoader

    loader = StatsBombLoader(_statsbomb_tree(tmp_path))
    cov = loader.coverage()
    assert cov["competitions"] == 1 and cov["international_competitions"] == 1
    # Malformed JSON raises clearly.
    (loader.data / "competitions.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        loader.load_competitions()


# --- FIFA rankings (21-27) --------------------------------------------------


def _fifa_csv(tmp_path, rows):
    p = tmp_path / "fifa.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_fifa_load_validate_and_aliases(tmp_path):
    from goalsignal.data.sources.fifa_rankings import load_rankings, validate_rankings

    # Source uses 'country_full'/'rank_date'/'total_points' -> mapped to canonical.
    p = _fifa_csv(tmp_path, [
        {"country_full": "Brazil", "rank": 1, "total_points": 1840.0, "rank_date": "2026-04-03"},
        {"country_full": "Brazil", "rank": 1, "total_points": 1840.0, "rank_date": "2026-04-03"},
        {"country_full": "Spain", "rank": 2, "total_points": "", "rank_date": "2026-04-03"},
    ])
    df, h = load_rankings(p)
    assert "team" in df.columns and df.iloc[0]["team"] == "Brazil"
    q = validate_rankings(df)
    assert q["duplicate_team_date"] == 2  # both Brazil rows flagged
    assert q["missing_points"] == 1  # Spain blank points
    assert q["invalid_ranks"] == 0
    assert len(h) == 64


def test_fifa_as_of_exact_and_future_rejected():
    from goalsignal.data.sources.fifa_rankings import as_of_ranking

    df = pd.DataFrame({
        "team": ["Brazil"] * 3, "rank": [3, 2, 1], "points": [1800, 1820, 1850],
        "ranking_release_date": ["2026-02-01", "2026-04-01", "2026-07-01"],
    })
    # Exact as-of: latest strictly before the match.
    assert as_of_ranking(df, "Brazil", "2026-05-01")["rank"] == 2
    # Future ranking (2026-07-01) never chosen even though closer to a July match.
    assert as_of_ranking(df, "Brazil", "2026-06-15")["rank"] == 2
    assert as_of_ranking(df, "Brazil", "2026-01-01") is None


def test_fifa_reports_written(tmp_path):
    from goalsignal.data.sources.fifa_rankings import load_rankings, write_fifa_reports

    p = _fifa_csv(tmp_path, [
        {"team": "Brazil", "rank": 1, "points": 1840.0, "ranking_release_date": "2026-04-03"},
        {"team": "Atlantis", "rank": 2, "points": 1700.0, "ranking_release_date": "2026-04-03"},
    ])
    df, h = load_rankings(p)
    out = tmp_path / "reports"
    q = write_fifa_reports(df, h, canonical_teams={"Brazil"}, out_dir=str(out))
    assert (out / "fifa_rankings_coverage.csv").exists()
    assert (out / "fifa_rankings_quality.json").exists()
    unmatched = pd.read_csv(out / "fifa_rankings_unmatched_teams.csv")
    assert list(unmatched["team"]) == ["Atlantis"]  # not in canonical set
    assert q["canonical_team_link_rate"] == 0.5


# --- fixture linking (28-33) ------------------------------------------------


def _canonical():
    return pd.DataFrame({
        "canonical_match_id": ["c1", "c2", "c3"],
        "date": ["2026-06-20", "2026-06-20", "2026-06-21"],
        "home_team": ["Brazil", "Brazil", "Spain"],
        "away_team": ["Serbia", "Switzerland", "Italy"],
    })


def test_linking_classes():
    src = pd.DataFrame({
        "source_fixture_id": ["a", "b", "c", "d"],
        "match_date": ["2026-06-20", "2026-06-21", "2026-06-20", "2026-06-30"],
        "home_team": ["Brazil", "Italy", "BRAZIL", "Narnia"],
        "away_team": ["Serbia", "Spain", "Switzerland", "Oz"],
    })
    links = link_fixtures(src, _canonical(), source_name="football_data")
    by_id = links.set_index("source_fixture_id")
    assert by_id.loc["a", "link_type"] == "exact" and by_id.loc["a", "canonical_match_id"] == "c1"
    assert by_id.loc["b", "link_type"] == "reversed"  # Italy v Spain ~ Spain v Italy
    assert by_id.loc["c", "link_type"] == "exact"  # "BRAZIL" casefolds to brazil -> c2
    assert by_id.loc["c", "canonical_match_id"] == "c2"
    assert by_id.loc["d", "link_type"] == "unmatched"
    # No source fixture is linked to more than one canonical id.
    assert links["canonical_match_id"].dropna().is_unique


def test_linking_ambiguous_same_day():
    # Two canonical Brazil-? on the same date won't collide unless same opponent;
    # build a genuine ambiguity: duplicate canonical identity.
    canon = pd.DataFrame({
        "canonical_match_id": ["c1", "c2"],
        "date": ["2026-06-20", "2026-06-20"],
        "home_team": ["Brazil", "Brazil"],
        "away_team": ["Serbia", "Serbia"],
    })
    src = pd.DataFrame({"source_fixture_id": ["a"], "match_date": ["2026-06-20"],
                        "home_team": ["Brazil"], "away_team": ["Serbia"]})
    links = link_fixtures(src, canon, source_name="sb")
    assert links.iloc[0]["link_type"] == "ambiguous"
    assert links.iloc[0]["canonical_match_id"] is None


def test_linking_deterministic_and_summary():
    src = pd.DataFrame({"source_fixture_id": ["a"], "match_date": ["2026-06-20"],
                        "home_team": ["Brazil"], "away_team": ["Serbia"]})
    a = link_fixtures(src, _canonical(), source_name="x")
    b = link_fixtures(src, _canonical(), source_name="x")
    pd.testing.assert_frame_equal(a, b)
    summ = link_summary(a)
    assert summ["exact"] == 1 and summ["link_rate"] == 1.0


# --- security / regression (34-39) ------------------------------------------


def test_env_is_gitignored():
    gitignore = (__import__("pathlib").Path(__file__).resolve().parents[2] / ".gitignore")
    assert ".env" in gitignore.read_text().splitlines()


def test_no_api_key_in_generated_cache(tmp_path):
    t = FakeTransport([_resp(_envelope([{"id": 1}]))])
    _client(t, tmp_path).leagues({"search": "x"})
    for p in (tmp_path / "af").rglob("*"):
        if p.is_file():
            assert FAKE_KEY not in p.read_text(errors="ignore")


def test_base_pipeline_imports_without_enrichment(monkeypatch):
    # Core forecasting must import and run with no enrichment env configured.
    for var in ("FOOTBALL_DATA_API_KEY", "STATSBOMB_DATA_PATH", "FIFA_RANKINGS_PATH"):
        monkeypatch.delenv(var, raising=False)
    from goalsignal.live import score_summary  # noqa: F401
    from goalsignal.models.poisson import PoissonGoalModel  # noqa: F401

    assert True


def test_ledger_and_result_store_still_verify():
    from goalsignal.feedback.results import verify_results
    from goalsignal.ledger.storage import verify_ledger
    from goalsignal.utils.paths import resolve

    if resolve("artifacts/predictions/ledger.jsonl").exists():
        assert verify_ledger() == []
    if resolve("artifacts/results/results.jsonl").exists():
        assert verify_results() == []
