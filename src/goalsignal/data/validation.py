"""Data-quality reporting.

Assembles the audit artifacts required by the project specification from a
BuildResult plus the raw frames. Reports describe problems; they never mutate
or filter data themselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from goalsignal.data.build_dataset import BuildResult
from goalsignal.data.schemas import DataConfig
from goalsignal.utils.paths import resolve


@dataclass
class QualityReport:
    summary: dict
    goalscorer_coverage: pd.DataFrame


def _goalscorer_coverage(
    goalscorers: pd.DataFrame, matches: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Per-year coverage of scorer events against recorded match goals.

    A match with recorded goals but no scorer rows is *uncovered*, not
    goalless: absence of scorer data must never be read as absence of goals.
    """
    gs = goalscorers.copy()
    gs["date_parsed"] = pd.to_datetime(gs["date"], errors="coerce")
    bad_dates = int(gs["date_parsed"].isna().sum())
    gs = gs[gs["date_parsed"].notna()]
    gs["year"] = gs["date_parsed"].dt.year

    missing_scorer = int((gs["scorer"].str.strip() == "").sum())
    bad_bool = int(
        (~gs["own_goal"].str.strip().str.lower().isin(["true", "false", ""])).sum()
        + (~gs["penalty"].str.strip().str.lower().isin(["true", "false", ""])).sum()
    )
    dup_events = int(
        gs.duplicated(
            subset=[c for c in ["date", "home_team", "away_team", "team", "scorer", "minute"]
                    if c in gs.columns],
            keep="first",
        ).sum()
    )

    played = matches[matches["status"] == "played"].copy()
    played["year"] = played["date"].dt.year
    played["total_goals"] = played["home_score_recorded"] + played["away_score_recorded"]

    # Join scorer events to matches on (date, raw home, raw away).
    event_counts = (
        gs.groupby(["date", "home_team", "away_team"]).size().rename("scorer_events")
    )
    keyed = played.set_index(
        [played["date"].dt.strftime("%Y-%m-%d"), "home_team_raw", "away_team_raw"]
    )
    keyed.index.names = ["date", "home_team", "away_team"]
    joined = keyed.join(event_counts, how="left")
    joined["scorer_events"] = joined["scorer_events"].fillna(0).astype(int)

    per_year = (
        joined.groupby("year")
        .agg(
            played_matches=("canonical_match_id", "count"),
            matches_with_goals=("total_goals", lambda s: int((s > 0).sum())),
            matches_with_scorer_rows=("scorer_events", lambda s: int((s > 0).sum())),
            recorded_goals=("total_goals", "sum"),
            scorer_events=("scorer_events", "sum"),
        )
        .reset_index()
    )
    # Nullable Float64: years with no scoring matches have undefined coverage.
    per_year["coverage_of_scoring_matches"] = (
        per_year["matches_with_scorer_rows"] / per_year["matches_with_goals"].replace(0, pd.NA)
    ).astype("Float64").round(4)

    unmatched_events = int(gs.shape[0] - joined["scorer_events"].sum())
    mismatched = joined[
        (joined["scorer_events"] > 0) & (joined["scorer_events"] != joined["total_goals"])
    ]
    summary = {
        "events": len(goalscorers),
        "events_unparseable_date": bad_dates,
        "events_missing_scorer_name": missing_scorer,
        "events_malformed_boolean": bad_bool,
        "duplicate_events": dup_events,
        "events_not_joined_to_played_match": unmatched_events,
        "matches_event_count_mismatch": len(mismatched),
    }
    return per_year, summary


def build_quality_report(
    raw: dict[str, pd.DataFrame], result: BuildResult, config: DataConfig
) -> QualityReport:
    matches = result.matches
    played = matches[matches["status"] == "played"]

    coverage_df, gs_summary = _goalscorer_coverage(raw["goalscorers"], matches)

    recon_status = result.shootout_reconciliation["status"].value_counts().to_dict()
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": config.schema_version,
        "results": result.stats,
        "missingness": {
            "neutral_unparseable": int(matches["neutral"].isna().sum()),
            "empty_city": int((matches["city"] == "").sum()),
            "empty_country": int((matches["country"] == "").sum()),
        },
        "score_scope": {
            "scope_counts": played["recorded_score_scope"].value_counts().to_dict(),
            "strict_goal_model_eligible": int(played["strict_goal_model_eligible"].sum()),
            "strict_exclusions": played[~played["strict_goal_model_eligible"]][
                "strict_exclusion_reason"
            ]
            .value_counts()
            .to_dict(),
        },
        "shootouts": {
            "rows": len(raw["shootouts"]),
            "reconciliation": {k: int(v) for k, v in recon_status.items()},
        },
        "goalscorers": gs_summary,
        "former_names": {
            "rows": len(raw["former_names"]),
            "mapping_issues": len(result.name_mapping_issues),
        },
        "exclusions_by_reason": result.exclusions["reason"].value_counts().to_dict()
        if len(result.exclusions)
        else {},
    }
    return QualityReport(summary=summary, goalscorer_coverage=coverage_df)


def _quality_markdown(summary: dict) -> str:
    r = summary["results"]
    lines = [
        "# Data Quality Report",
        "",
        f"Generated: {summary['generated_at_utc']} (schema v{summary['schema_version']})",
        "",
        "## Results",
        "",
        f"- Raw rows: {r['raw_rows']}",
        f"- Canonical matches: {r['canonical_matches']} "
        f"({r['played_matches']} played, {r['scheduled_matches']} scheduled)",
        f"- Excluded rows: {r['excluded_rows']} (see excluded_matches.csv)",
        f"- Date range: {r['date_min']} to {r['date_max']}",
        f"- Teams: {r['teams']}; tournaments: {r['tournaments']}",
        f"- Strict 90-minute goal-model eligible: {r['strict_goal_model_eligible']}",
        f"- Suspicious-scope rows flagged: {r['suspicious_scope_rows']} "
        "(see suspicious_scope_matches.csv)",
        "",
        "## Score scope",
        "",
        "The source records scores including extra time but excluding penalty",
        "shootouts. Extra time is provable only for matches with a shootout;",
        "those are excluded from strict 90-minute exact-score training.",
        "Decisive results in knockout-capable tournaments are flagged as an",
        "upper bound on undetectable extra-time contamination — flagged, not",
        "excluded.",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in summary["score_scope"]["scope_counts"].items()]
    lines += [
        "",
        "## Shootouts",
        "",
        f"- Rows: {summary['shootouts']['rows']}",
    ]
    lines += [
        f"- {k}: {v}" for k, v in sorted(summary["shootouts"]["reconciliation"].items())
    ]
    lines += [
        "",
        "## Goalscorers",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in summary["goalscorers"].items()]
    lines += [
        "",
        "## Exclusions",
        "",
    ]
    excl = summary["exclusions_by_reason"]
    lines += [f"- {k}: {v}" for k, v in excl.items()] if excl else ["- none"]
    lines.append("")
    return "\n".join(lines)


def write_reports(
    raw: dict[str, pd.DataFrame], result: BuildResult, config: DataConfig
) -> tuple[Path, QualityReport]:
    """Write all audit artifacts; returns the reports directory and report."""
    report = build_quality_report(raw, result, config)
    out = resolve(config.output.reports_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "data_quality.json", "w", encoding="utf-8") as f:
        json.dump(report.summary, f, indent=2, ensure_ascii=False)
    (out / "data_quality.md").write_text(_quality_markdown(report.summary), encoding="utf-8")
    result.exclusions.to_csv(out / "excluded_matches.csv", index=False)
    result.duplicate_identities.to_csv(out / "duplicate_matches.csv", index=False)
    result.suspicious_scope.to_csv(out / "suspicious_scope_matches.csv", index=False)
    result.shootout_reconciliation.to_csv(out / "shootout_reconciliation.csv", index=False)
    report.goalscorer_coverage.to_csv(out / "goalscorer_coverage.csv", index=False)
    result.name_mapping_issues.to_csv(out / "former_name_conflicts.csv", index=False)
    return out, report
