"""World Cup pre-tournament rank validation (wc_teams.csv vs reconstructed FIFA).

Compares the published pre-tournament rank in `wc_teams.csv` (year, team,
confederation, rank) against the rank reconstructed from the FIFA timeline at
the latest release before each World Cup's opening date. Neither dataset is
modified; discrepancies are reported, never silently reconciled.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from goalsignal.data.sources.linking import normalize_team
from goalsignal.utils.hashing import sha256_file
from goalsignal.utils.paths import resolve

# World Cup opening dates (documented). The pre-tournament cutoff is the latest
# FIFA release strictly before the opening.
WC_OPENING_DATES = {
    1994: "1994-06-17", 1998: "1998-06-10", 2002: "2002-05-31",
    2006: "2006-06-09", 2010: "2010-06-11", 2014: "2014-06-12",
    2018: "2018-06-14", 2022: "2022-11-20", 2026: "2026-06-11",
}


def load_wc_teams(path: str | Path) -> tuple[pd.DataFrame, str]:
    path = resolve(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = pd.DataFrame({
        "tournament_year": pd.to_numeric(raw.get("year"), errors="coerce"),
        "team": raw.get("team", "").astype(str).str.strip(),
        "confederation": raw.get("confederation"),
        "published_pre_tournament_rank": pd.to_numeric(raw.get("rank"), errors="coerce"),
        "source_row": range(2, len(raw) + 2),
    })
    df["normalized_team"] = df["team"].map(normalize_team)
    return df, sha256_file(path)


def validate(wc_df: pd.DataFrame, fifa_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (per-team validation frame, summary dict)."""
    rel = pd.to_datetime(fifa_df["ranking_release_date"], errors="coerce")
    fifa = fifa_df.assign(_rel=rel)
    rows = []
    for r in wc_df.itertuples(index=False):
        year = int(r.tournament_year) if pd.notna(r.tournament_year) else None
        opening = WC_OPENING_DATES.get(year)
        rec_rank = None
        rec_date = None
        cls = None
        if opening is None:
            cls = "tournament_cutoff_unknown"
        else:
            cutoff = pd.Timestamp(opening)
            sub = fifa[(fifa["normalized_team"] == r.normalized_team) & (fifa["_rel"] < cutoff)]
            if sub.empty:
                # distinguish "team never in FIFA file" from "no release before cutoff"
                anyteam = fifa[fifa["normalized_team"] == r.normalized_team]
                cls = "unmatched_team" if anyteam.empty else "ranking_release_unavailable"
            else:
                last = sub.sort_values("_rel").iloc[-1]
                rec_rank = int(last["fifa_rank"]) if pd.notna(last["fifa_rank"]) else None
                rec_date = last["ranking_release_date"]
        pub = (int(r.published_pre_tournament_rank)
               if pd.notna(r.published_pre_tournament_rank) else None)
        if cls is None and rec_rank is not None and pub is not None:
            diff = abs(rec_rank - pub)
            cls = ("exact_match" if diff == 0
                   else ("small_discrepancy" if diff <= 2 else "large_discrepancy"))
        elif cls is None:
            cls = "missing_rank"
        diff_val = abs(rec_rank - pub) if rec_rank is not None and pub is not None else None
        rows.append({
            "tournament_year": year, "team": r.team, "confederation": r.confederation,
            "published_rank": pub, "reconstructed_rank": rec_rank,
            "reconstructed_release_date": rec_date,
            "rank_diff": diff_val,
            "classification": cls,
        })
    out = pd.DataFrame(rows)
    by_year = {}
    for year, grp in out.groupby("tournament_year"):
        by_year[str(int(year))] = {str(k): int(v)
                                   for k, v in grp["classification"].value_counts().items()}
    summary = {
        "total": len(out),
        "by_classification": {str(k): int(v)
                              for k, v in out["classification"].value_counts().items()},
        "by_year": by_year,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "discrepancies arise from FIFA's official ranking methodology vs "
        "reconstructed standard-competition ranking, team aliases, and release "
        "timing; neither dataset was modified",
    }
    return out, summary


def write_reports(wc_df, fifa_df, out_dir: str = "artifacts/reports") -> dict:
    import json

    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame, summary = validate(wc_df, fifa_df)
    frame.to_csv(out / "fifa_world_cup_rank_validation.csv", index=False)
    (out / "fifa_world_cup_rank_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary
