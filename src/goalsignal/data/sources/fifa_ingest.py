"""Real-schema ingestion of the historical FIFA ranking timeline.

Adapts to the actual file `ranking_fifa_historical.csv`:
    team, total_points, date, id, id_num, team_short

There is NO per-team rank column, so rank is RECONSTRUCTED within each release
by sorting points descending. Tie policy (documented, deterministic): standard
competition ranking — a team's rank is `1 + (number of teams in the release
with strictly greater points)`; display order among equal-points teams is by
team name for reproducibility. Missing points are never replaced with zero and
rank is never inferred from points where points are absent.

The raw CSV is never modified. The leakage-safe as-of join selects only the
most recent release on or before the cutoff — never a future release.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from goalsignal.data.sources.linking import normalize_team
from goalsignal.data.sources.manifests import build_snapshot_manifest
from goalsignal.utils.hashing import sha256_file
from goalsignal.utils.paths import resolve

LICENSE = "User-provided; verify FIFA terms before redistribution"
ATTRIBUTION = "FIFA/Coca-Cola World Ranking"
SCHEMA_VERSION = 1

NORMALIZED_COLUMNS = [
    "ranking_release_date", "team", "normalized_team", "team_code",
    "fifa_points", "fifa_rank", "source_release_id", "source_row",
    "source_snapshot_id",
]


def load_fifa_historical(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load and normalize the FIFA timeline; reconstruct rank per release.

    Returns (normalized_df, manifest_dict). The manifest is content-derived and
    records file hash, size, rows, columns, date range, and snapshot id.
    """
    path = resolve(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    content_hash = sha256_file(path)

    dates = pd.to_datetime(raw.get("date"), errors="coerce")
    points = pd.to_numeric(raw.get("total_points"), errors="coerce")
    df = pd.DataFrame({
        "ranking_release_date": dates,
        "team": raw.get("team", pd.Series([""] * len(raw))).astype(str).str.strip(),
        "team_code": raw.get("team_short", pd.Series([None] * len(raw))),
        "fifa_points": points,
        "source_release_id": raw.get("id_num", raw.get("id", pd.Series([None] * len(raw)))),
        "source_row": range(2, len(raw) + 2),
    })
    df["normalized_team"] = df["team"].map(normalize_team)

    # Reconstruct rank within each release (standard competition ranking), only
    # for rows with parseable points; rows with missing points get rank <NA>.
    df["fifa_rank"] = pd.NA
    for _, idx in df.groupby(df["ranking_release_date"]).groups.items():
        block = df.loc[idx]
        valid = block[block["fifa_points"].notna()].sort_values(
            ["fifa_points", "team"], ascending=[False, True]
        )
        pts = valid["fifa_points"].to_numpy()
        # standard competition rank: 1 + count of strictly greater points
        ranks = [1 + int((pts > p).sum()) for p in pts]
        df.loc[valid.index, "fifa_rank"] = ranks

    manifest = build_snapshot_manifest(
        source="fifa_rankings", role="ranking",
        endpoint_or_url=str(path.name),
        available_at_semantics="ranking knowable from its release date onward",
        license=LICENSE, attribution=ATTRIBUTION, content_hash=content_hash,
        row_count=len(raw), schema_version=SCHEMA_VERSION, cache_path=str(path),
        coverage_period_start=str(dates.min().date()) if dates.notna().any() else None,
        coverage_period_end=str(dates.max().date()) if dates.notna().any() else None,
        notes=[f"file_size_bytes={path.stat().st_size}",
               f"columns={list(raw.columns)}"],
    )
    df["source_snapshot_id"] = manifest.snapshot_id
    df["ranking_release_date"] = df["ranking_release_date"].dt.strftime("%Y-%m-%d")
    return df[NORMALIZED_COLUMNS], manifest.model_dump()


def quality_report(df: pd.DataFrame) -> dict:
    dates = pd.to_datetime(df["ranking_release_date"], errors="coerce")
    pts = pd.to_numeric(df["fifa_points"], errors="coerce")
    dup = df.duplicated(subset=["team", "ranking_release_date"], keep=False)
    # normalized collisions: same normalized name, different raw team, same release
    norm_coll = (
        df.groupby(["ranking_release_date", "normalized_team"])["team"].nunique()
    )
    return {
        "rows": len(df),
        "invalid_dates": int(dates.isna().sum()),
        "missing_team_names": int((df["team"].str.strip() == "").sum()),
        "missing_points": int(pts.isna().sum()),
        "negative_or_invalid_points": int((pts < 0).sum()),
        "duplicate_team_release": int(dup.sum()),
        "release_count": int(dates.dropna().nunique()),
        "teams": int(df["team"].nunique()),
        "date_min": str(dates.min().date()) if dates.notna().any() else None,
        "date_max": str(dates.max().date()) if dates.notna().any() else None,
        "normalized_team_collisions": int((norm_coll > 1).sum()),
        "limitation": "timeline ends in 2024; cannot provide valid live 2026 "
        "FIFA values — 2026 forecasts must not use a future/absent release",
    }


def release_summary(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["ranking_release_date"], errors="coerce")
    return (
        df.assign(_d=dates)
        .groupby("ranking_release_date")
        .agg(team_count=("team", "nunique"),
             release_id=("source_release_id", "first"),
             max_points=("fifa_points", "max"),
             min_points=("fifa_points", "min"))
        .reset_index()
        .sort_values("ranking_release_date")
    )


def as_of_fifa(df: pd.DataFrame, team: str, match_date) -> dict | None:
    """Latest FIFA release for `team` strictly before `match_date` (no leak)."""
    match_date = pd.Timestamp(match_date)
    rel = pd.to_datetime(df["ranking_release_date"], errors="coerce")
    sub = df[(df["team"] == team) & (rel < match_date)].copy()
    if sub.empty:
        return None
    sub["_rel"] = pd.to_datetime(sub["ranking_release_date"])
    row = sub.sort_values("_rel").iloc[-1]
    return {
        "team": team,
        "fifa_points": float(row["fifa_points"]) if pd.notna(row["fifa_points"]) else None,
        "fifa_rank": int(row["fifa_rank"]) if pd.notna(row["fifa_rank"]) else None,
        "ranking_release_date": row["ranking_release_date"],
        "days_since_release": int((match_date - row["_rel"]).days),
    }


def write_reports(
    df: pd.DataFrame, manifest: dict, canonical_teams: set | None = None,
    out_dir: str = "artifacts/reports",
) -> dict:
    """Write the four FIFA ranking reports; returns the quality dict."""
    import json

    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    quality = quality_report(df)
    quality["snapshot_id"] = manifest["snapshot_id"]
    quality["content_hash"] = manifest["content_hash"]
    quality["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    dates = pd.to_datetime(df["ranking_release_date"], errors="coerce")
    coverage = (
        df.assign(year=dates.dt.year)
        .groupby("year")
        .agg(rows=("team", "size"), teams=("team", "nunique"),
             releases=("ranking_release_date", "nunique"))
        .reset_index()
    )
    coverage.to_csv(out / "fifa_rankings_coverage.csv", index=False)
    release_summary(df).to_csv(out / "fifa_release_summary.csv", index=False)

    unmatched = pd.DataFrame(columns=["team", "normalized_team", "reason"])
    link_rate = None
    if canonical_teams is not None:
        canon_norm = {normalize_team(t) for t in canonical_teams}
        teams = df[["team", "normalized_team"]].drop_duplicates()
        miss = teams[~teams["normalized_team"].isin(canon_norm)]
        unmatched = miss.assign(reason="no_canonical_team_match")
        link_rate = round(1 - len(miss) / max(len(teams), 1), 4)
    unmatched.to_csv(out / "fifa_rankings_unmatched_teams.csv", index=False)
    quality["canonical_team_link_rate"] = link_rate

    (out / "fifa_rankings_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    return quality
