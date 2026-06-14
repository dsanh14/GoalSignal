"""Frozen June 11, 2026 FIFA ranking snapshot for the World Cup field."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from goalsignal.data.sources.linking import normalize_team
from goalsignal.data.sources.manifests import build_snapshot_manifest
from goalsignal.utils.hashing import sha256_file
from goalsignal.utils.paths import resolve

RELEASE_DATE = "2026-06-11"
SCHEMA_VERSION = 1
GROUPS = list("ABCDEFGHIJKL")
ALIASES = {
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "czechia": "Czech Republic",
}


def _canonical_name(name: str, canonical: set[str]) -> str | None:
    alias = ALIASES.get(normalize_team(name))
    if alias in canonical:
        return alias
    by_norm = {normalize_team(team): team for team in canonical}
    return by_norm.get(normalize_team(name))


def load_current_fifa(
    path: str | Path, canonical_teams: set[str]
) -> tuple[pd.DataFrame, dict, dict]:
    """Load, normalize, and strictly validate the frozen 48-team snapshot."""
    path = resolve(path)
    raw_bytes_hash = sha256_file(path)
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    raw.columns = [str(c).lstrip("\ufeff").strip() for c in raw.columns]
    expected = ["group", "team", "fifa_rank"]
    if list(raw.columns) != expected:
        raise ValueError(f"expected columns {expected}, got {list(raw.columns)}")

    df = raw.copy()
    df["group"] = df["group"].str.strip()
    df["team"] = df["team"].str.strip()
    df["fifa_rank"] = pd.to_numeric(df["fifa_rank"], errors="coerce")
    df["normalized_team"] = df["team"].map(normalize_team)
    df["canonical_team"] = df["team"].map(lambda x: _canonical_name(x, canonical_teams))
    df["ranking_release_date"] = RELEASE_DATE
    df["source_row"] = range(2, len(df) + 2)

    errors = []
    if len(df) != 48:
        errors.append(f"expected 48 rows, got {len(df)}")
    if sorted(df["group"].unique()) != GROUPS:
        errors.append(f"groups must be A-L, got {sorted(df['group'].unique())}")
    sizes = df.groupby("group").size()
    if len(sizes) != 12 or not (sizes == 4).all():
        errors.append(f"each group must contain four teams: {sizes.to_dict()}")
    if df[["group", "team", "fifa_rank"]].eq("").any().any():
        errors.append("missing values found")
    if df["team"].duplicated().any() or df[["group", "team"]].duplicated().any():
        errors.append("duplicate team or (group, team)")
    ranks = df["fifa_rank"]
    if ranks.isna().any() or (ranks <= 0).any() or not np.all(ranks == ranks.astype(int)):
        errors.append("FIFA ranks must be positive integers")
    if df["canonical_team"].isna().any():
        errors.append("unmatched canonical teams: " + ", ".join(
            df.loc[df["canonical_team"].isna(), "team"].tolist()
        ))
    linked = df["canonical_team"].dropna()
    if linked.duplicated().any():
        errors.append("duplicate canonical mappings")
    if errors:
        raise ValueError("; ".join(errors))

    df["fifa_rank"] = df["fifa_rank"].astype(int)
    manifest_obj = build_snapshot_manifest(
        source="fifa_current_2026",
        role="current_world_cup_ranking_snapshot",
        endpoint_or_url=path.name,
        available_at_semantics=f"available from {RELEASE_DATE}; never before release",
        license="User-provided; verify FIFA terms before redistribution",
        attribution="FIFA/Coca-Cola World Ranking",
        content_hash=raw_bytes_hash,
        row_count=len(raw),
        schema_version=SCHEMA_VERSION,
        cache_path=str(path.resolve()),
        coverage_period_start=RELEASE_DATE,
        coverage_period_end=RELEASE_DATE,
        notes=[f"file_size_bytes={path.stat().st_size}", f"columns={list(raw.columns)}"],
    )
    manifest = manifest_obj.model_dump()
    manifest.update({
        "filename": path.name,
        "resolved_path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "columns": list(raw.columns),
        "release_date": RELEASE_DATE,
        "source_type": "frozen_current_snapshot",
        "ingestion_timestamp": manifest["retrieval_timestamp"],
    })
    df["source_snapshot_id"] = manifest["snapshot_id"]
    quality = {
        "valid": True,
        "rows": len(df),
        "groups": int(df["group"].nunique()),
        "teams": int(df["team"].nunique()),
        "release_date": RELEASE_DATE,
        "snapshot_id": manifest["snapshot_id"],
        "sha256": raw_bytes_hash,
    }
    return df, manifest, quality


def available_as_of(match_date) -> bool:
    return pd.Timestamp(RELEASE_DATE) < pd.Timestamp(match_date)


def compare_with_elo(df: pd.DataFrame, ratings: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    out = df[["group", "team", "canonical_team", "fifa_rank"]].copy()
    out["goalsignal_elo"] = out["canonical_team"].map(ratings)
    if out["goalsignal_elo"].isna().any():
        raise ValueError("missing live Elo for: " + ", ".join(
            out.loc[out["goalsignal_elo"].isna(), "canonical_team"]
        ))
    out["goalsignal_elo_rank"] = (
        out["goalsignal_elo"].rank(method="min", ascending=False).astype(int)
    )
    out["fifa_rank_minus_elo_rank"] = out["fifa_rank"] - out["goalsignal_elo_rank"]
    out["absolute_rank_disagreement"] = out["fifa_rank_minus_elo_rank"].abs()
    out["normalized_fifa_strength"] = (
        out["fifa_rank"].max() - out["fifa_rank"]
    ) / (out["fifa_rank"].max() - out["fifa_rank"].min())
    out["normalized_elo_strength"] = (
        out["goalsignal_elo"] - out["goalsignal_elo"].min()
    ) / (out["goalsignal_elo"].max() - out["goalsignal_elo"].min())
    d = out["normalized_fifa_strength"] - out["normalized_elo_strength"]
    out["standardized_disagreement"] = (d - d.mean()) / d.std(ddof=0)
    out["fifa_rates_higher"] = out["fifa_rank_minus_elo_rank"] < 0
    out["elo_rates_higher"] = out["fifa_rank_minus_elo_rank"] > 0
    spear = spearmanr(out["fifa_rank"], out["goalsignal_elo_rank"])
    pear = pearsonr(out["normalized_fifa_strength"], out["normalized_elo_strength"])
    fifa_top = set(out.nsmallest(10, "fifa_rank")["canonical_team"])
    elo_top = set(out.nsmallest(10, "goalsignal_elo_rank")["canonical_team"])
    summary = {
        "snapshot_id": df["source_snapshot_id"].iloc[0],
        "spearman_rank_correlation": float(spear.statistic),
        "spearman_p_value": float(spear.pvalue),
        "pearson_strength_correlation": float(pear.statistic),
        "pearson_p_value": float(pear.pvalue),
        "top_10_overlap": len(fifa_top & elo_top),
        "largest_fifa_over_elo": out.nsmallest(10, "fifa_rank_minus_elo_rank")[
            ["canonical_team", "fifa_rank_minus_elo_rank"]
        ].to_dict("records"),
        "largest_elo_over_fifa": out.nlargest(10, "fifa_rank_minus_elo_rank")[
            ["canonical_team", "fifa_rank_minus_elo_rank"]
        ].to_dict("records"),
        "group_averages": out.groupby("group").agg(
            average_fifa_rank=("fifa_rank", "mean"),
            average_elo=("goalsignal_elo", "mean"),
        ).reset_index().to_dict("records"),
    }
    return out, summary


def write_reports(df: pd.DataFrame, manifest: dict, quality: dict, comparison=None) -> None:
    out = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fifa_current_2026_quality.json").write_text(
        json.dumps({**quality, "generated_at": datetime.now(UTC).isoformat()}, indent=2),
        encoding="utf-8",
    )
    df.to_csv(out / "fifa_current_2026_coverage.csv", index=False)
    df[df["canonical_team"].isna()][["team", "normalized_team"]].to_csv(
        out / "fifa_current_2026_unmatched_teams.csv", index=False
    )
    (out / "fifa_current_2026_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if comparison:
        table, summary = comparison
        table.to_csv(out / "fifa_current_2026_vs_elo.csv", index=False)
        (out / "fifa_current_2026_vs_elo.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
