"""Read-only presentation helpers for the prediction ledger.

These functions never write: they flatten, filter, and format stored payloads.
W/D/L probabilities come from the ensemble (`model_version`); expected goals
and scorelines were produced by the deployed goal model at prediction time.
v1 payloads do not name the score model — that is displayed honestly as
unrecorded rather than backfilled into immutable entries.
"""

from __future__ import annotations

import csv
import io
import json

SCORE_MODEL_UNRECORDED = "unrecorded-in-v1-payload"


def flatten_entry(entry: dict) -> dict:
    """One display row per ledger entry."""
    p = entry["payload"]
    top = p.get("top_scorelines") or []
    best = top[0] if top else None
    return {
        "date": p.get("kickoff_timestamp", ""),
        "home_team": p.get("home_team", ""),
        "away_team": p.get("away_team", ""),
        "home_xg": p.get("home_expected_goals"),
        "away_xg": p.get("away_expected_goals"),
        "likely_score": f"{best['home']}-{best['away']}" if best else "",
        "likely_score_p": best["p"] if best else None,
        "p_home": p.get("home_win_probability"),
        "p_draw": p.get("draw_probability"),
        "p_away": p.get("away_win_probability"),
        "wdl_model": p.get("model_version", ""),
        "score_model": p.get("score_model_version", SCORE_MODEL_UNRECORDED),
        "fixture_id": p.get("fixture_id", ""),
        "entry_hash": entry.get("entry_hash", ""),
        "result_store_hash": p.get("result_store_hash"),
        "active_result_count": p.get("active_result_count"),
        "revision": p.get("revision", p.get("model_version", "")),
    }


def filter_entries(
    entries: list[dict], team: str | None = None, date: str | None = None
) -> list[dict]:
    out = entries
    if team is not None:
        t = team.casefold()
        out = [
            e
            for e in out
            if t in e["payload"].get("home_team", "").casefold()
            or t in e["payload"].get("away_team", "").casefold()
        ]
    if date is not None:
        out = [e for e in out if e["payload"].get("kickoff_timestamp", "") == date]
    return out


def latest_entries(entries: list[dict]) -> list[dict]:
    """Latest appended immutable revision for each fixture."""
    latest: dict[str, dict] = {}
    for entry in entries:
        latest[entry["payload"].get("fixture_id", "")] = entry
    return list(latest.values())


def model_version_entries(entries: list[dict], model_version: str | None) -> list[dict]:
    if model_version is None:
        return entries
    return [e for e in entries if e["payload"].get("model_version") == model_version]


def find_entry(entries: list[dict], prediction_id: str) -> dict | None:
    """Locate an entry by entry-hash prefix or fixture-id prefix (unique)."""
    matches = [
        e
        for e in entries
        if e.get("entry_hash", "").startswith(prediction_id)
        or e["payload"].get("fixture_id", "").startswith(prediction_id)
    ]
    return matches[0] if len(matches) == 1 else None


def _pct(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def format_table(entries: list[dict], top_scorelines: int = 1) -> str:
    lines = [
        f"{'Date':<11} {'Match':<42} {'xG':<11} {'Likely':<7} "
        f"{'Score P':<8} {'H':<7} {'D':<7} {'A':<7}"
    ]
    for e in entries:
        r = flatten_entry(e)
        match = f"{r['home_team']} v {r['away_team']}"
        xg = (
            f"{r['home_xg']:.2f}-{r['away_xg']:.2f}"
            if r["home_xg"] is not None and r["away_xg"] is not None
            else "n/a"
        )
        lines.append(
            f"{r['date']:<11} {match:<42} {xg:<11} {r['likely_score']:<7} "
            f"{_pct(r['likely_score_p']):<8} {_pct(r['p_home']):<7} "
            f"{_pct(r['p_draw']):<7} {_pct(r['p_away']):<7}"
        )
        if top_scorelines > 1:
            for s in (e["payload"].get("top_scorelines") or [])[:top_scorelines]:
                lines.append(
                    f"{'':<11}   {s['home']}-{s['away']}  {_pct(s['p'])}"
                )
    return "\n".join(lines)


def format_csv(entries: list[dict]) -> str:
    rows = [flatten_entry(e) for e in entries]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else [])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def format_json(entries: list[dict], top_scorelines: int = 5) -> str:
    payloads = []
    for e in entries:
        p = dict(e["payload"])
        if p.get("top_scorelines"):
            p["top_scorelines"] = p["top_scorelines"][:top_scorelines]
        p["entry_hash"] = e.get("entry_hash", "")
        payloads.append(p)
    return json.dumps(payloads, indent=2, ensure_ascii=False)
