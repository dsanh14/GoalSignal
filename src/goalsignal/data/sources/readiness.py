"""Team-alias audit, entity-linking candidate scaffolding, and source-readiness
classification — all from REAL ingested data and prior audit artifacts.

Nothing here auto-accepts a fuzzy match. Alias and player/club candidates are
written for human review with `review_status="candidate"`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from goalsignal.data.sources.linking import normalize_team
from goalsignal.utils.paths import resolve

# Deterministic, reviewed FIFA↔canonical alias suggestions (candidates only;
# date-aware/former-name handling stays in the canonical normalizer).
KNOWN_FIFA_ALIASES = {
    "korea republic": "South Korea", "korea dpr": "North Korea",
    "usa": "United States", "ir iran": "Iran",
    "côte d'ivoire": "Ivory Coast", "cote d'ivoire": "Ivory Coast",
    "china pr": "China", "republic of ireland": "Ireland",
    "czechia": "Czech Republic", "cabo verde": "Cape Verde",
    "the gambia": "Gambia", "türkiye": "Turkey", "turkiye": "Turkey",
    "congo dr": "DR Congo", "kyrgyz republic": "Kyrgyzstan",
}


def team_alias_audit(fifa_df: pd.DataFrame, canonical_teams: set,
                     out_dir: str = "artifacts/reports") -> dict:
    """Compare FIFA team names to canonical; write alias candidates for review."""
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    canon_norm = {normalize_team(t): t for t in canonical_teams}
    fifa_teams = fifa_df[["team", "normalized_team"]].drop_duplicates()

    rows = []
    exact = 0
    alias_assisted = 0
    unmatched = 0
    for r in fifa_teams.itertuples(index=False):
        if r.normalized_team in canon_norm:
            exact += 1
            continue
        suggestion = KNOWN_FIFA_ALIASES.get(r.normalized_team)
        if suggestion and normalize_team(suggestion) in canon_norm:
            alias_assisted += 1
            rows.append({"fifa_team": r.team, "normalized": r.normalized_team,
                         "suggested_canonical": suggestion, "method": "known_alias",
                         "review_status": "candidate"})
        else:
            unmatched += 1
            rows.append({"fifa_team": r.team, "normalized": r.normalized_team,
                         "suggested_canonical": "", "method": "none",
                         "review_status": "unmatched"})
    pd.DataFrame(rows, columns=["fifa_team", "normalized", "suggested_canonical",
                                "method", "review_status"]).to_csv(
        out / "team_source_alias_candidates.csv", index=False)
    total = len(fifa_teams)
    return {
        "fifa_teams": int(total),
        "exact_match": int(exact),
        "alias_assisted_candidates": int(alias_assisted),
        "unmatched": int(unmatched),
        "exact_match_rate": round(exact / total, 4) if total else None,
        "note": "alias suggestions are CANDIDATES for review; never auto-accepted",
    }


def write_player_identity_scaffolding(out_dir: str = "artifacts/reports") -> None:
    """Write empty-but-headed entity-candidate reports + the matching rule doc.

    Large-scale player matching is deferred (Phase 12). We emit the report
    schema and the deterministic matching hierarchy so the next milestone has a
    defined, non-fuzzy starting point.
    """
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=[
        "source", "source_player_id", "name", "normalized_name", "date_of_birth",
        "nationality", "club", "position", "match_method", "review_status",
    ]).to_csv(out / "player_identity_candidates.csv", index=False)
    pd.DataFrame(columns=["name", "reason", "n_candidates", "review_status"]).to_csv(
        out / "player_identity_conflicts.csv", index=False)
    pd.DataFrame(columns=["source", "source_player_id", "name", "reason"]).to_csv(
        out / "player_unmatched.csv", index=False)
    pd.DataFrame(columns=[
        "source", "source_club_id", "name", "normalized_name", "country",
        "suggested_canonical", "review_status",
    ]).to_csv(out / "club_identity_candidates.csv", index=False)


# Source-readiness classification for each planned feature family.
READINESS_STATES = (
    "ready", "restricted_subset", "insufficient_coverage", "blocked_missing_data",
    "blocked_plan", "temporally_unsafe", "unsupported",
)


def build_source_readiness(out_dir: str = "artifacts/reports") -> dict:
    """Classify feature families from the real audit artifacts on disk."""
    out = resolve(out_dir)

    def _exists(name):
        return (out / name).exists()

    fifa_ready = _exists("fifa_rankings_quality.json")
    tm_ready = _exists("transfermarkt_table_inventory.json")
    af_cov = out / "api_football_coverage.json"
    af = json.loads(af_cov.read_text()) if af_cov.exists() else {}

    families = {
        "fifa_points_rank": {
            "state": "ready" if fifa_ready else "blocked_missing_data",
            "detail": "real timeline 1992-2024, 335 releases; leakage-safe as-of join. "
            "Ends 2024 -> for 2026 the latest release is ~620 days stale (flag it).",
        },
        "fifa_elo_disagreement": {
            "state": "ready" if fifa_ready else "blocked_missing_data",
            "detail": "derivable from FIFA rank/points vs internal Elo, as-of cutoff.",
        },
        "statsbomb_xg": {"state": "blocked_missing_data",
                         "detail": "no StatsBomb data present (STATSBOMB_DATA_PATH unset)."},
        "statsbomb_shots": {"state": "blocked_missing_data", "detail": "see statsbomb_xg."},
        "statsbomb_lineup_continuity": {"state": "blocked_missing_data",
                                        "detail": "see statsbomb_xg."},
        "player_minutes": {
            "state": "restricted_subset" if tm_ready else "blocked_missing_data",
            "detail": "Transfermarkt club appearances (1.89M, dated, 2012-2026) -> "
            "safe with cutoff as a CLUB-form proxy; not international minutes.",
        },
        "player_starts": {
            "state": "restricted_subset" if tm_ready else "blocked_missing_data",
            "detail": "game_lineups type=starting_lineup (club), dated -> cutoff-safe proxy.",
        },
        "club_strength": {
            "state": "restricted_subset" if tm_ready else "blocked_missing_data",
            "detail": "clubs.total_market_value is current-state; dated player_valuations "
            "(508k, 2000-2026) give a cutoff-safe club/player value proxy.",
        },
        "historical_valuations": {
            "state": "ready" if tm_ready else "blocked_missing_data",
            "detail": "player_valuations are dated -> cutoff-safe.",
        },
        "expected_lineup_strength": {
            "state": "insufficient_coverage",
            "detail": "needs international squad/lineup history; Transfermarkt national-team "
            "games are sparse (670) and lineups are club-centric.",
        },
        "confirmed_lineups": {
            "state": "blocked_plan",
            "detail": "API-Football confirmed lineups for 2026 WC are plan-locked (Free).",
        },
        "goalkeeper_strength": {"state": "insufficient_coverage",
                                "detail": "GK-specific international data not available."},
        "injuries": {"state": "unsupported",
                     "detail": "API-Football World Cup injuries cov=false; no other source."},
        "suspensions": {"state": "unsupported", "detail": "no source provides suspensions."},
        "rest": {"state": "ready",
                 "detail": "derivable from canonical match dates (internal); no new source."},
        "travel": {"state": "insufficient_coverage",
                   "detail": "needs venue coordinates; not yet ingested."},
        "altitude": {"state": "insufficient_coverage",
                     "detail": "needs venue altitude reference; not yet ingested."},
        "native_recent_form": {"state": "ready",
                               "detail": "derivable from canonical results (internal)."},
    }
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "states_legend": list(READINESS_STATES),
        "families": families,
        "api_football": {"auth_verified": af.get("auth_verified"),
                         "subscription_plan": af.get("subscription_plan"),
                         "world_cup_2026": "plan_locked_on_free"},
    }
    (out / "source_readiness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = ["# Source Readiness", "", f"Generated: {summary['generated_at']}", "",
          "| Feature family | State | Detail |", "| --- | --- | --- |"]
    for fam, info in families.items():
        md.append(f"| {fam} | **{info['state']}** | {info['detail']} |")
    (out / "source_readiness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pd.DataFrame([{"feature_family": k, "state": v["state"]} for k, v in families.items()]).to_csv(
        out / "enrichment_coverage.csv", index=False)
    return summary
