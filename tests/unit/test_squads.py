"""Squad-source, identity, activity, valuation, and readiness tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from goalsignal.data.sources.config import SquadDataConfig
from goalsignal.data.sources.schemas import ExpectedLineupInputRecord
from goalsignal.data.sources.squads import (
    assert_squads_available_at,
    build_feature_readiness,
    build_historical_valuations,
    build_player_activity,
    link_squad_players,
    load_official_extract,
    load_seed_link_candidates,
    load_squads,
    position_group,
    reconcile_official_extract,
    revalidate_seed_links,
)


def _squad_rows():
    return [
        {
            "snapshot_date": "2026-06-10",
            "group": "K",
            "national_team": "Portugal",
            "player_name": "João Teste",
            "date_of_birth": "2000-01-02",
            "position": "Midfielder",
            "club": "FC Example",
            "shirt_number": "8",
            "squad_status": "selected",
            "source_name": "Portuguese Football Federation",
            "source_url_or_reference": "https://example.invalid/official",
            "source_publication_date": "2026-06-09",
            "source_player_id": "",
            "notes": "",
        },
        {
            "snapshot_date": "2026-06-10",
            "group": "K",
            "national_team": "Portugal",
            "player_name": "Rui Keeper",
            "date_of_birth": "",
            "position": "Goalkeeper",
            "club": "",
            "shirt_number": "",
            "squad_status": "reserve",
            "source_name": "Portuguese Football Federation",
            "source_url_or_reference": "https://example.invalid/official",
            "source_publication_date": "2026-06-09",
            "source_player_id": "22",
            "notes": "",
        },
    ]


def _write_squads(tmp_path, rows=None, bom=False):
    path = tmp_path / "squads.csv"
    pd.DataFrame(rows or _squad_rows()).to_csv(
        path, index=False, encoding="utf-8-sig" if bom else "utf-8"
    )
    return path


def test_squad_config_paths_are_optional(monkeypatch):
    config = SquadDataConfig.load()
    monkeypatch.delenv(config.squads_path_env, raising=False)
    assert config.squads_path_env == "FIFA_2026_SQUADS_PATH"
    assert config.availability_path_env == "FIFA_2026_AVAILABILITY_PATH"
    assert config.player_aliases_path_env == "FIFA_2026_PLAYER_ALIASES_PATH"


def test_bom_safe_squad_load_preserves_raw_and_manifest(tmp_path):
    path = _write_squads(tmp_path, bom=True)
    before = path.read_bytes()
    retrieved = datetime(2026, 6, 11, tzinfo=UTC)
    frame, first, quality = load_squads(
        path, canonical_teams={"Portugal"}, retrieved_at=retrieved
    )
    _, second, _ = load_squads(
        path, canonical_teams={"Portugal"}, retrieved_at=retrieved
    )
    assert path.read_bytes() == before
    assert frame.iloc[0]["raw_player_name"] == "João Teste"
    assert frame.iloc[0]["canonical_team"] == "Portugal"
    assert quality["rows"] == 2
    assert first == second
    assert len(first["content_hash"]) == 64


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("player_name", "", "player_name missing"),
        ("source_name", "", "source_name missing"),
        ("source_publication_date", "bad", "invalid source_publication_date"),
        ("group", "Z", "invalid World Cup groups"),
    ],
)
def test_squad_required_field_and_group_validation(tmp_path, field, value, match):
    rows = _squad_rows()
    rows[0][field] = value
    with pytest.raises(ValueError, match=match):
        load_squads(_write_squads(tmp_path, rows), canonical_teams={"Portugal"})


def test_duplicate_player_rejected(tmp_path):
    rows = _squad_rows()
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match="duplicated canonical player"):
        load_squads(_write_squads(tmp_path, rows), canonical_teams={"Portugal"})


def test_team_alias_and_publication_cutoff(tmp_path):
    rows = _squad_rows()
    rows[0]["national_team"] = "Türkiye"
    rows[1]["national_team"] = "Türkiye"
    frame, _, _ = load_squads(
        _write_squads(tmp_path, rows), canonical_teams={"Turkey"}
    )
    assert set(frame["canonical_team"]) == {"Turkey"}
    assert_squads_available_at(frame, "2026-06-09")
    with pytest.raises(ValueError, match="not published"):
        assert_squads_available_at(frame, "2026-06-08")


def _players():
    return pd.DataFrame(
        [
            {
                "player_id": 10,
                "name": "Joao Teste",
                "date_of_birth": "2000-01-02",
                "country_of_citizenship": "Portugal",
                "current_club_name": "FC Example",
                "position": "Midfielder",
            },
            {
                "player_id": 22,
                "name": "Different Keeper",
                "date_of_birth": "1995-01-01",
                "country_of_citizenship": "Portugal",
                "current_club_name": "Other",
                "position": "Goalkeeper",
            },
        ]
    )


def test_identity_source_id_dob_and_accent_matching(tmp_path):
    squads, _, _ = load_squads(
        _write_squads(tmp_path), canonical_teams={"Portugal"}
    )
    links = link_squad_players(squads, _players())
    by_name = links.set_index("player_name")
    assert by_name.loc["João Teste", "match_method"] == "name_date_of_birth"
    assert by_name.loc["João Teste", "transfermarkt_player_id"] == "10"
    assert by_name.loc["Rui Keeper", "match_method"] == "exact_source_id"
    assert by_name.loc["Rui Keeper", "transfermarkt_player_id"] == "22"


def test_reviewed_alias_precedes_other_matches(tmp_path):
    squads, _, _ = load_squads(
        _write_squads(tmp_path), canonical_teams={"Portugal"}
    )
    aliases = pd.DataFrame(
        [
            {
                "national_team": "Portugal",
                "squad_player_name": "João Teste",
                "transfermarkt_player_id": "22",
                "review_status": "reviewed",
                "notes": "human reviewed",
            }
        ]
    )
    links = link_squad_players(squads.iloc[[0]], _players(), aliases)
    assert links.iloc[0]["match_method"] == "reviewed_alias"
    assert links.iloc[0]["transfermarkt_player_id"] == "22"


def test_official_extract_reconciliation_and_shift_normalization(tmp_path):
    squads, _, _ = load_squads(
        _write_squads(tmp_path), canonical_teams={"Portugal"}
    )
    extract = pd.DataFrame(
        [
            {
                "group": row["group"],
                "national_team": row["national_team"],
                "shirt_number": row["shirt_number"],
                "position": row["position"],
                "fifa_player_name": row["player_name"],
                "name_on_shirt": row["date_of_birth"],
                "date_of_birth": row["club"],
                "club": "",
                "source_pdf_page": "1",
                "source_url": "https://example.invalid/fifa",
            }
            for row in _squad_rows()
        ]
    )
    path = tmp_path / "extract.csv"
    extract.to_csv(path, index=False)
    report, summary = reconcile_official_extract(squads, load_official_extract(path))
    assert summary["matched_rows"] == 2
    assert not report["classification"].eq("substantive_discrepancy").any()


def test_seed_link_revalidation_accepts_exact_and_rejects_stale(tmp_path):
    squads, _, _ = load_squads(
        _write_squads(tmp_path), canonical_teams={"Portugal"}
    )
    candidates = pd.DataFrame(
        [
            {
                "national_team": "Portugal",
                "official_player_name": "João Teste",
                "date_of_birth": "2000-01-02",
                "transfermarkt_player_id": "10",
                "transfermarkt_name": "Joao Teste",
                "link_status": "exact_dob_name",
                "source_url": "https://example.invalid/fifa",
            },
            {
                "national_team": "Portugal",
                "official_player_name": "Rui Keeper",
                "date_of_birth": "1995-01-01",
                "transfermarkt_player_id": "999",
                "transfermarkt_name": "Rui Keeper",
                "link_status": "exact_dob_name",
                "source_url": "https://example.invalid/fifa",
            },
        ]
    )
    path = tmp_path / "candidates.csv"
    candidates.to_csv(path, index=False)
    report, summary = revalidate_seed_links(
        squads, load_seed_link_candidates(path), _players()
    )
    assert summary["accepted_deterministic"] == 1
    assert set(report["classification"]) == {
        "accepted deterministic",
        "rejected stale",
    }


def test_name_only_match_is_not_accepted(tmp_path):
    rows = _squad_rows()[:1]
    rows[0]["date_of_birth"] = ""
    rows[0]["club"] = ""
    rows[0]["position"] = "Defender"
    squads, _, _ = load_squads(
        _write_squads(tmp_path, rows), canonical_teams={"Portugal"}
    )
    links = link_squad_players(squads, _players())
    assert links.iloc[0]["match_class"] == "ambiguous"
    assert links.iloc[0]["canonical_player_id"] == ""


def _links():
    return pd.DataFrame(
        [
            {
                "snapshot_date": "2026-06-10",
                "national_team": "Portugal",
                "player_name": "Player",
                "canonical_player_id": "tm:10",
                "transfermarkt_player_id": "10",
            }
        ]
    )


def test_activity_windows_cutoff_target_and_missing_minutes():
    appearances = pd.DataFrame(
        [
            {
                "game_id": 1,
                "player_id": 10,
                "date": "2026-06-10",
                "minutes_played": 90,
                "goals": 1,
                "assists": 0,
                "yellow_cards": 0,
                "red_cards": 0,
            },
            {
                "game_id": 2,
                "player_id": 10,
                "date": "2026-05-01",
                "minutes_played": None,
                "goals": 0,
                "assists": 1,
                "yellow_cards": 1,
                "red_cards": 0,
            },
            {
                "game_id": 3,
                "player_id": 10,
                "date": "2026-06-15",
                "minutes_played": 90,
                "goals": 5,
                "assists": 0,
                "yellow_cards": 0,
                "red_cards": 0,
            },
        ]
    )
    lineups = pd.DataFrame(
        [
            {"game_id": 1, "player_id": 10, "date": "2026-06-10", "type": "starting_lineup"},
            {"game_id": 2, "player_id": 10, "date": "2026-05-01", "type": "substitutes"},
        ]
    )
    out = build_player_activity(
        _links(), appearances, lineups, cutoff="2026-06-15", target_game_id=1
    ).iloc[0]
    assert out["appearances_30d"] == 0
    assert pd.isna(out["minutes_90d"])
    assert out["starts_90d"] == 0
    assert out["goals_90d"] == 0
    assert out["days_since_last_appearance"] == 45


def test_activity_days_since_last_and_window_counts():
    appearances = pd.DataFrame(
        [
            {
                "game_id": 1, "player_id": 10, "date": "2026-06-10",
                "minutes_played": 90, "goals": 1, "assists": 0,
                "yellow_cards": 0, "red_cards": 0,
            },
            {
                "game_id": 2, "player_id": 10, "date": "2026-01-01",
                "minutes_played": 45, "goals": 0, "assists": 0,
                "yellow_cards": 0, "red_cards": 0,
            },
        ]
    )
    lineups = pd.DataFrame(
        [{"game_id": 1, "player_id": 10, "date": "2026-06-10", "type": "starting_lineup"}]
    )
    out = build_player_activity(
        _links(), appearances, lineups, cutoff="2026-06-15"
    ).iloc[0]
    assert out["minutes_30d"] == 90
    assert out["minutes_180d"] == 135
    assert out["starts_30d"] == 1
    assert out["days_since_last_appearance"] == 5


def test_historical_valuation_on_or_before_cutoff_and_missing():
    links = pd.concat(
        [
            _links(),
            pd.DataFrame(
                [{
                    "snapshot_date": "2026-06-10", "national_team": "Portugal",
                    "player_name": "Missing", "canonical_player_id": "tm:11",
                    "transfermarkt_player_id": "11",
                }]
            ),
        ],
        ignore_index=True,
    )
    values = pd.DataFrame(
        [
            {"player_id": 10, "date": "2026-05-01", "market_value_in_eur": 100},
            {"player_id": 10, "date": "2026-06-15", "market_value_in_eur": 999},
        ]
    )
    out = build_historical_valuations(
        links, values, cutoff="2026-06-15", source_snapshot_id="snap"
    ).set_index("player_name")
    assert out.loc["Player", "historical_valuation"] == 999
    assert out.loc["Player", "valuation_age_days"] == 0
    assert bool(out.loc["Missing", "available"]) is False
    assert pd.isna(out.loc["Missing", "historical_valuation"])


def test_position_groups_and_expected_lineup_placeholder():
    assert position_group("Goalkeeper") == "goalkeeper"
    assert position_group("Central Defender") == "defender"
    assert position_group("Attacking Midfield") == "midfielder"
    record = ExpectedLineupInputRecord.model_validate(
        {
            "fixture_id": "f1",
            "prediction_cutoff": "2026-06-15T00:00:00+00:00",
            "national_team": "Portugal",
            "canonical_player_id": "tm:10",
            "position_group": "midfielder",
            "selected_in_squad": True,
            "candidate_starter_probability": None,
        }
    )
    assert record.candidate_starter_probability is None
    with pytest.raises(ValueError, match="must remain missing"):
        ExpectedLineupInputRecord.model_validate(
            {
                **record.model_dump(),
                "candidate_starter_probability": 0.8,
            }
        )


def test_readiness_without_squad_is_explicitly_blocked():
    readiness = build_feature_readiness(
        squad_available=False,
        identity_link_rate=None,
        statsbomb_available=False,
        international_lineup_ready=False,
    )
    assert readiness["families"]["recent club minutes"] == (
        "blocked by missing squad source"
    )
    assert readiness["families"]["confirmed lineups"] == "blocked by provider plan"
    assert readiness["families"]["injuries"] == "unsupported"
