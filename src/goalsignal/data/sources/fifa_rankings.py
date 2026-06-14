"""Historical FIFA rankings adapter (contract + leakage-safe as-of join).

FIFA rankings are user-provided (no official free historical API). The user
points `FIFA_RANKINGS_PATH` at a local CSV. The date-aware as-of join here is
pure and testable now: it returns the latest ranking strictly released before
a match date, never a future ranking.
"""

from __future__ import annotations

import pandas as pd
from pydantic import ValidationError

from goalsignal.data.sources.base import (
    CoverageReport,
    MilestoneNotImplementedError,
    SourceValidationError,
)
from goalsignal.data.sources.config import FifaRankingsConfig
from goalsignal.data.sources.schemas import FifaRankingRecord


def as_of_ranking(
    rankings: pd.DataFrame, team: str, match_date, *, strictly_before: bool = True
) -> dict | None:
    """Latest ranking for `team` released before `match_date` (no future leak).

    `rankings` must have columns: team, rank, points, ranking_release_date.
    Returns None when no ranking precedes the match (never the nearest future
    ranking). With `strictly_before=True` a ranking released exactly on the
    match date is excluded (it may post-date kickoff).
    """
    match_date = pd.Timestamp(match_date)
    sub = rankings[rankings["team"] == team].copy()
    if sub.empty:
        return None
    sub["ranking_release_date"] = pd.to_datetime(sub["ranking_release_date"])
    if strictly_before:
        sub = sub[sub["ranking_release_date"] < match_date]
    else:
        sub = sub[sub["ranking_release_date"] <= match_date]
    if sub.empty:
        return None
    row = sub.sort_values("ranking_release_date").iloc[-1]
    return {
        "team": team,
        "rank": int(row["rank"]),
        "points": float(row["points"]),
        "ranking_release_date": row["ranking_release_date"].date().isoformat(),
        "days_since_ranking_release": int(
            (match_date - row["ranking_release_date"]).days
        ),
    }


class FifaRankingsAdapter:
    name = "fifa_rankings"
    role = "ranking"

    def __init__(self, config: FifaRankingsConfig | None = None):
        self.config = config or FifaRankingsConfig()

    def validate(self, records: list[dict]) -> list[dict]:
        out = []
        for i, rec in enumerate(records):
            try:
                out.append(FifaRankingRecord.model_validate(rec).model_dump(mode="json"))
            except ValidationError as exc:
                raise SourceValidationError(
                    f"FIFA ranking record {i} failed schema validation: {exc}"
                ) from exc
        return out

    def load(self) -> pd.DataFrame:
        raise MilestoneNotImplementedError(
            "FIFA rankings CSV ingestion lands in Milestone B. It will read "
            f"${self.config.path_env} and validate columns "
            f"{self.config.expected_columns}."
        )

    def as_of(self, team: str, match_date) -> dict | None:
        raise MilestoneNotImplementedError(
            "Stateful as-of lookups require ingested data (Milestone B). Use the "
            "pure `as_of_ranking(df, team, match_date)` function for testing the "
            "leakage-safe join now."
        )

    def build_manifest(self):
        raise MilestoneNotImplementedError(
            "Manifest building requires ingested content (Milestone B); use "
            "`goalsignal.data.sources.manifests.build_snapshot_manifest` directly."
        )

    def report_coverage(self) -> CoverageReport:
        return CoverageReport(
            source=self.name,
            rows=0,
            notes=["no data ingested (Milestone A defines contracts only)"],
        )


# --- offline loader + quality reports (Milestone B) -------------------------
import json  # noqa: E402
import os  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from goalsignal.utils.hashing import sha256_file  # noqa: E402
from goalsignal.utils.paths import resolve  # noqa: E402

CANONICAL_COLUMNS = (
    "ranking_release_date", "team", "rank", "ranking_points",
    "previous_rank", "confederation",
)


class FifaRankingsUnavailable(MilestoneNotImplementedError):
    """No local FIFA rankings file configured."""


def resolve_fifa_path(config: FifaRankingsConfig | None = None) -> Path:
    config = config or FifaRankingsConfig()
    raw = os.environ.get(config.path_env, "")
    if not raw:
        raise FifaRankingsUnavailable(
            f"FIFA rankings not configured. Set ${config.path_env} to a CSV with "
            "columns: team, rank, points, ranking_release_date (extra columns are "
            "mapped via config). FIFA rankings are optional; other sources continue."
        )
    path = Path(raw)
    if not path.exists():
        raise FifaRankingsUnavailable(f"${config.path_env}={raw} does not exist.")
    return path


# Default mapping from common source column names to canonical names.
DEFAULT_COLUMN_MAP = {
    "team": "team", "country_full": "team", "country": "team",
    "rank": "rank", "rank_date": "ranking_release_date",
    "ranking_release_date": "ranking_release_date",
    "points": "ranking_points", "total_points": "ranking_points",
    "ranking_points": "ranking_points",
    "previous_rank": "previous_rank", "previous_points": None,
    "confederation": "confederation",
}


def load_rankings(path: str | Path, column_map: dict | None = None) -> tuple[pd.DataFrame, str]:
    """Load a FIFA rankings CSV through a documented mapping layer."""
    path = Path(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    cmap = column_map or DEFAULT_COLUMN_MAP
    renamed = {}
    for col in raw.columns:
        target = cmap.get(col, col if col in CANONICAL_COLUMNS else None)
        if target:
            renamed[col] = target
    df = raw.rename(columns=renamed)
    df = df[[c for c in df.columns if c in CANONICAL_COLUMNS]].copy()
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["source_row"] = range(2, len(df) + 2)
    return df, sha256_file(path)


def validate_rankings(df: pd.DataFrame) -> dict:
    """Quality checks; returns a JSON-serializable report (no mutation)."""
    dates = pd.to_datetime(df["ranking_release_date"], errors="coerce")
    ranks = pd.to_numeric(df["rank"], errors="coerce")
    points = pd.to_numeric(df["ranking_points"], errors="coerce")
    dup = df.duplicated(subset=["team", "ranking_release_date"], keep=False)
    return {
        "rows": len(df),
        "unparseable_dates": int(dates.isna().sum()),
        "invalid_ranks": int((ranks.isna() | (ranks < 1)).sum()),
        "missing_points": int(points.isna().sum()),
        "duplicate_team_date": int(dup.sum()),
        "teams": int(df["team"].nunique()),
        "release_dates": int(dates.dropna().nunique()),
        "date_min": dates.min().date().isoformat() if dates.notna().any() else None,
        "date_max": dates.max().date().isoformat() if dates.notna().any() else None,
    }


def write_fifa_reports(df: pd.DataFrame, snapshot_hash: str, canonical_teams: set | None = None,
                       out_dir: str = "artifacts/reports") -> dict:
    """Write fifa_rankings_coverage.csv, _quality.json, _unmatched_teams.csv."""
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    quality = validate_rankings(df)
    quality["snapshot_hash"] = snapshot_hash
    quality["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    dates = pd.to_datetime(df["ranking_release_date"], errors="coerce")
    coverage = (
        df.assign(year=dates.dt.year)
        .groupby("year", dropna=True)
        .agg(rows=("team", "size"), teams=("team", "nunique"))
        .reset_index()
    )
    coverage.to_csv(out / "fifa_rankings_coverage.csv", index=False)
    (out / "fifa_rankings_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    unmatched = pd.DataFrame(columns=["team", "reason"])
    link_rate = None
    if canonical_teams is not None:
        src_teams = set(df["team"].dropna().unique())
        missing = sorted(src_teams - canonical_teams)
        unmatched = pd.DataFrame({"team": missing, "reason": "no_canonical_team_match"})
        link_rate = round(1 - len(missing) / max(len(src_teams), 1), 4)
    unmatched.to_csv(out / "fifa_rankings_unmatched_teams.csv", index=False)
    quality["canonical_team_link_rate"] = link_rate
    return quality
