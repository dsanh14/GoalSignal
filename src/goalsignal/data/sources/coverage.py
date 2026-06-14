"""Real source-coverage reporting.

Coverage states are never collapsed to a Boolean. Each source reports one of:

    not_configured        no credential/path supplied
    unsupported           the feature family is not provided by the source
    unavailable           configured but could not be reached/authenticated
    configured_but_empty  configured/reachable but no data ingested yet
    partially_covered     some data present
    present               data present

Only states derivable from the *actual* environment and ingested artifacts are
reported; nothing is fabricated.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from goalsignal.data.sources.cache import list_snapshots
from goalsignal.data.sources.config import (
    ApiFootballConfig,
    FifaRankingsConfig,
    StatsBombConfig,
)
from goalsignal.data.sources.env import has_env
from goalsignal.utils.paths import resolve


def _api_football_state(cfg: ApiFootballConfig) -> dict:
    configured = has_env(cfg.credential_env)
    snapshots = list_snapshots(cfg.cache_dir)
    probe_path = resolve("artifacts/reports/api_football_probe.json")
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else None
    if not configured:
        state = "not_configured"
    elif snapshots:
        state = "present"
    elif probe and probe.get("auth_verified") is False:
        state = "unavailable"  # key present but provider rejected it
    elif probe and probe.get("auth_verified") is True:
        state = "configured_but_empty"  # auth ok, no data ingested yet
    else:
        state = "not_yet_tested"
    return {
        "source": "api_football",
        "vendor": "API-Sports",
        "role": "live_fixtures",
        "state": state,
        "auth_configured": configured,
        "auth_verified": bool(probe and probe.get("auth_verified")),
        "cached_snapshots": len(snapshots),
        "injuries_endpoint": "exists; World Cup/free-plan population measured, not assumed",
        "probe": probe,
    }


def _statsbomb_state(cfg: StatsBombConfig) -> dict:
    configured = has_env(cfg.data_path_env)
    normalized = resolve("data/external/statsbomb/normalized")
    has_norm = normalized.exists() and any(normalized.glob("*.csv"))
    state = "present" if has_norm else ("configured_but_empty" if configured else "not_configured")
    return {
        "source": "statsbomb",
        "role": "event_enrichment",
        "state": state,
        "configured": configured,
        "note": "international coverage is sparse; missing coverage is not a team signal",
    }


def _fifa_state(cfg: FifaRankingsConfig) -> dict:
    configured = has_env(cfg.path_env)
    quality = resolve("artifacts/reports/fifa_rankings_quality.json")
    has_data = quality.exists()
    state = "present" if has_data else ("configured_but_empty" if configured else "not_configured")
    return {"source": "fifa_rankings", "role": "ranking", "state": state,
            "configured": configured}


def build_source_coverage(out_dir: str = "artifacts/reports") -> dict:
    """Compute coverage states and write summary json/md + enrichment_coverage.csv."""
    fd = _api_football_state(ApiFootballConfig.load())
    sb = _statsbomb_state(StatsBombConfig.load())
    fifa = _fifa_state(FifaRankingsConfig.load())
    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": [fd, sb, fifa],
        "legend": [
            "not_configured", "unsupported", "unavailable",
            "configured_but_empty", "partially_covered", "present",
        ],
    }
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_coverage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_markdown(summary, out / "source_coverage_summary.md")
    # Per-source state lives here; feature-family readiness writes
    # enrichment_coverage.csv separately (see readiness.build_source_readiness).
    _write_enrichment_csv(summary, out / "source_state_coverage.csv")
    return summary


def _write_markdown(summary: dict, path: Path) -> None:
    lines = ["# Source Coverage Summary", "",
             f"Generated: {summary['generated_at']}", "",
             "| Source | Role | State | Notes |", "| --- | --- | --- | --- |"]
    for s in summary["sources"]:
        note = ""
        if s["source"] == "api_football":
            note = (f"vendor={s['vendor']}, auth_configured={s['auth_configured']}, "
                    f"auth_verified={s['auth_verified']}")
        elif s["source"] == "statsbomb":
            note = s["note"]
        lines.append(f"| {s['source']} | {s['role']} | **{s['state']}** | {note} |")
    lines += ["", "States are not Booleans: see the legend in the JSON summary."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_enrichment_csv(summary: dict, path: Path) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "role", "state"])
        for s in summary["sources"]:
            w.writerow([s["source"], s["role"], s["state"]])
