"""Versioning and reporting for result-aware World Cup simulations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from goalsignal.utils.hashing import sha256_json
from goalsignal.utils.paths import resolve


def simulation_version(
    result_hash: str,
    model_version: str,
    snapshot_id: str | None,
    bracket_hash: str | None = None,
) -> str:
    return sha256_json({
        "result_store_hash": result_hash,
        "model_version": model_version,
        "fifa_snapshot_id": snapshot_id,
        "bracket_hash": bracket_hash,
    })[:16]


def result_frame(result) -> pd.DataFrame:
    rows = []
    for group, teams in result.groups.items():
        for team in teams:
            pp = result.position_probs[team]
            rows.append({
                "group": group,
                "team": team,
                "expected_points": round(result.expected_points[team], 4),
                "p_first": round(pp[0], 6),
                "p_second": round(pp[1], 6),
                "p_third": round(pp[2], 6),
                "p_fourth": round(pp[3], 6),
                "p_best_third_advance": round(result.best_third_probs[team], 6),
                "p_reach_round_of_32": round(result.advance_probs[team], 6),
                "mc_se_advance": round(
                    result.mc_standard_error(result.advance_probs[team]), 7
                ),
            })
    return pd.DataFrame(rows)


def write_group_reports(
    frame: pd.DataFrame,
    official_groups: dict[str, list[str]],
    previous: pd.DataFrame | None = None,
) -> list[Path]:
    out = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for letter in ("B", "C", "D"):
        teams = official_groups[letter]
        report = frame[frame["team"].isin(teams)].copy()
        report["group"] = letter
        report["change_from_previous_r32"] = pd.NA
        if previous is not None and set(teams).issubset(set(previous["team"])):
            old = previous.set_index("team")["p_reach_round_of_32"]
            report["change_from_previous_r32"] = (
                report["p_reach_round_of_32"] - report["team"].map(old)
            )
        report = report.sort_values("p_reach_round_of_32", ascending=False)
        csv_path = out / f"group_{letter.lower()}_live_update.csv"
        md_path = out / f"group_{letter.lower()}_live_update.md"
        report.to_csv(csv_path, index=False)
        headers = list(report.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in report.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        md_path.write_text(
            f"# Group {letter} live update\n\n" + "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        paths.extend([csv_path, md_path])
    return paths


def write_versioned_simulation(
    frame: pd.DataFrame, metadata: dict, version: str
) -> Path:
    out = resolve(Path("artifacts/simulations") / version)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "wc2026_group_stage.csv"
    meta_path = out / "wc2026_group_stage_meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("result_store_hash") != metadata["result_store_hash"]:
            raise FileExistsError("stale simulation directory has a different result hash")
    frame.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out
