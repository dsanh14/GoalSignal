"""Normalize API-Football v3 envelopes into typed CSV tables.

Pure functions: envelope dict -> pandas DataFrame. Every row keeps the
API-Football source id, source snapshot id, retrieved_at, endpoint, and schema
version. CSV is used so the base install needs no pyarrow.

Provider PREDICTIONS are normalized separately and clearly labelled as an
external benchmark; they are never mixed into GoalSignal training features.
"""

from __future__ import annotations

import pandas as pd

SCHEMA_VERSION = 1


def _prov(snapshot_id: str, retrieved_at: str, endpoint: str) -> dict:
    return {
        "source": "api_football",
        "source_snapshot_id": snapshot_id,
        "retrieved_at": retrieved_at,
        "endpoint": endpoint,
        "schema_version": SCHEMA_VERSION,
    }


def _response(envelope: dict) -> list:
    resp = envelope.get("response")
    return resp if isinstance(resp, list) else ([] if resp is None else [resp])


def normalize_leagues(envelope: dict, snapshot_id: str, retrieved_at: str) -> pd.DataFrame:
    rows = []
    for item in _response(envelope):
        league = item.get("league", {}) or {}
        country = item.get("country", {}) or {}
        for season in item.get("seasons", []) or [{}]:
            cov = season.get("coverage", {}) or {}
            fx = cov.get("fixtures", {}) or {}
            rows.append({
                "league_id": league.get("id"), "league_name": league.get("name"),
                "type": league.get("type"), "country": country.get("name"),
                "season": season.get("year"), "season_current": season.get("current"),
                "cov_fixtures_events": fx.get("events"),
                "cov_fixtures_lineups": fx.get("lineups"),
                "cov_fixtures_statistics": fx.get("statistics_fixtures"),
                "cov_fixtures_players": fx.get("statistics_players"),
                "cov_standings": cov.get("standings"),
                "cov_players": cov.get("players"),
                "cov_injuries": cov.get("injuries"),
                "cov_predictions": cov.get("predictions"),
                "cov_odds": cov.get("odds"),
                **_prov(snapshot_id, retrieved_at, "leagues"),
            })
    return pd.DataFrame(rows)


def normalize_fixtures(envelope: dict, snapshot_id: str, retrieved_at: str) -> pd.DataFrame:
    rows = []
    for item in _response(envelope):
        fx = item.get("fixture", {}) or {}
        lg = item.get("league", {}) or {}
        teams = item.get("teams", {}) or {}
        goals = item.get("goals", {}) or {}
        status = fx.get("status", {}) or {}
        venue = fx.get("venue", {}) or {}
        rows.append({
            "provider_fixture_id": fx.get("id"), "date_utc": fx.get("date"),
            "status_short": status.get("short"), "status_long": status.get("long"),
            "league_id": lg.get("id"), "season": lg.get("season"), "round": lg.get("round"),
            "venue_name": venue.get("name"), "venue_city": venue.get("city"),
            "home_team": (teams.get("home") or {}).get("name"),
            "home_team_id": (teams.get("home") or {}).get("id"),
            "away_team": (teams.get("away") or {}).get("name"),
            "away_team_id": (teams.get("away") or {}).get("id"),
            "home_goals": goals.get("home"), "away_goals": goals.get("away"),
            "available_at_semantics": "result available when status_short in (FT,AET,PEN)",
            **_prov(snapshot_id, retrieved_at, "fixtures"),
        })
    return pd.DataFrame(rows)


def normalize_standings(envelope: dict, snapshot_id: str, retrieved_at: str) -> pd.DataFrame:
    rows = []
    for item in _response(envelope):
        lg = item.get("league", {}) or {}
        for group in lg.get("standings", []) or []:
            for entry in group:
                team = entry.get("team", {}) or {}
                allp = entry.get("all", {}) or {}
                g = allp.get("goals", {}) or {}
                rows.append({
                    "league_id": lg.get("id"), "season": lg.get("season"),
                    "group": entry.get("group"), "rank": entry.get("rank"),
                    "team": team.get("name"), "team_id": team.get("id"),
                    "points": entry.get("points"), "goals_diff": entry.get("goalsDiff"),
                    "played": allp.get("played"), "win": allp.get("win"),
                    "draw": allp.get("draw"), "lose": allp.get("lose"),
                    "goals_for": g.get("for"), "goals_against": g.get("against"),
                    **_prov(snapshot_id, retrieved_at, "standings"),
                })
    return pd.DataFrame(rows)


def normalize_lineups(envelope: dict, snapshot_id: str, retrieved_at: str) -> pd.DataFrame:
    rows = []
    for item in _response(envelope):
        team = item.get("team", {}) or {}
        formation = item.get("formation")
        for role, key in (("start", "startXI"), ("bench", "substitutes")):
            for p in item.get(key, []) or []:
                player = p.get("player", {}) or {}
                rows.append({
                    "team": team.get("name"), "team_id": team.get("id"),
                    "formation": formation, "role": role,
                    "player_id": player.get("id"), "player_name": player.get("name"),
                    "number": player.get("number"), "position": player.get("pos"),
                    "available_at_semantics": "confirmed lineup from announcement (~1h pre-KO)",
                    **_prov(snapshot_id, retrieved_at, "fixtures/lineups"),
                })
    return pd.DataFrame(rows)


def normalize_injuries(envelope: dict, snapshot_id: str, retrieved_at: str) -> pd.DataFrame:
    rows = []
    for item in _response(envelope):
        player = item.get("player", {}) or {}
        team = item.get("team", {}) or {}
        fx = item.get("fixture", {}) or {}
        rows.append({
            "player_id": player.get("id"), "player_name": player.get("name"),
            "team": team.get("name"), "team_id": team.get("id"),
            "provider_fixture_id": fx.get("id"),
            "type": item.get("type"), "reason": item.get("reason"),
            "available_at_semantics": "injury report as of retrieval; never used post-kickoff",
            **_prov(snapshot_id, retrieved_at, "injuries"),
        })
    return pd.DataFrame(rows)


def normalize_fixture_players(envelope: dict, snapshot_id: str, retrieved_at: str) -> pd.DataFrame:
    rows = []
    for item in _response(envelope):
        team = item.get("team", {}) or {}
        for p in item.get("players", []) or []:
            player = p.get("player", {}) or {}
            stats = (p.get("statistics") or [{}])[0]
            games = stats.get("games", {}) or {}
            goals = stats.get("goals", {}) or {}
            rows.append({
                "team": team.get("name"), "team_id": team.get("id"),
                "player_id": player.get("id"), "player_name": player.get("name"),
                "minutes": games.get("minutes"), "position": games.get("position"),
                "rating": games.get("rating"),
                "goals": goals.get("total"), "assists": goals.get("assists"),
                **_prov(snapshot_id, retrieved_at, "fixtures/players"),
            })
    return pd.DataFrame(rows)


def normalize_fixture_events(envelope: dict, snapshot_id: str, retrieved_at: str) -> pd.DataFrame:
    rows = []
    for item in _response(envelope):
        team = item.get("team", {}) or {}
        player = item.get("player", {}) or {}
        rows.append({
            "minute": (item.get("time", {}) or {}).get("elapsed"),
            "team": team.get("name"), "team_id": team.get("id"),
            "player": player.get("name"), "type": item.get("type"),
            "detail": item.get("detail"),
            **_prov(snapshot_id, retrieved_at, "fixtures/events"),
        })
    return pd.DataFrame(rows)


def normalize_predictions_benchmark(
    envelope: dict, fixture_id: int, snapshot_id: str, retrieved_at: str
) -> pd.DataFrame:
    """Provider predictions — BENCHMARK ONLY, never a training feature.

    The output is tagged `usage="external_benchmark"` so it cannot be mistaken
    for a GoalSignal feature table.
    """
    rows = []
    for item in _response(envelope):
        pred = item.get("predictions", {}) or {}
        percent = pred.get("percent", {}) or {}
        winner = pred.get("winner", {}) or {}
        rows.append({
            "provider_fixture_id": fixture_id,
            "predicted_winner": winner.get("name"),
            "percent_home": percent.get("home"), "percent_draw": percent.get("draw"),
            "percent_away": percent.get("away"),
            "advice": pred.get("advice"),
            "usage": "external_benchmark",
            **_prov(snapshot_id, retrieved_at, "predictions"),
        })
    return pd.DataFrame(rows)
