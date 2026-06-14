"""Append-only match-result store, kept strictly separate from predictions.

Results live in their own hash-chained JSONL (`artifacts/results/results.jsonl`),
never inside the prediction ledger and never inside the read-only `Datasets/`
directory. A fixture may have exactly one active result; conflicting entries
are rejected unless recorded through the audited correction workflow, which
appends a new entry referencing the entry it corrects (nothing is ever
rewritten).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from goalsignal.utils.hashing import sha256_text
from goalsignal.utils.paths import resolve

GENESIS = "0" * 64
DEFAULT_RESULTS_PATH = "artifacts/results/results.jsonl"


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def list_results(path: str | Path = DEFAULT_RESULTS_PATH) -> list[dict]:
    return _read(resolve(path))


def active_results(path: str | Path = DEFAULT_RESULTS_PATH) -> dict[str, dict]:
    """fixture_id -> latest active result payload (corrections supersede)."""
    out: dict[str, dict] = {}
    for entry in list_results(path):
        out[entry["payload"]["fixture_id"]] = entry["payload"]
    return out


def result_store_hash(path: str | Path = DEFAULT_RESULTS_PATH) -> str:
    store = resolve(path)
    if not store.exists():
        return sha256_text("")
    import hashlib

    return hashlib.sha256(store.read_bytes()).hexdigest()


def _outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def record_result(
    fixture_id: str,
    home_goals: int,
    away_goals: int,
    completed_at: str,
    source: str,
    kickoff_date: str | None = None,
    path: str | Path = DEFAULT_RESULTS_PATH,
    corrects: str | None = None,
    correction_reason: str | None = None,
    match_date: str | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
) -> dict:
    """Append one result entry; returns the stored entry.

    Duplicate or conflicting results for a fixture are rejected unless this is
    an explicit correction (`corrects` = entry_hash of the superseded entry,
    with a mandatory reason).
    """
    if home_goals < 0 or away_goals < 0:
        raise ValueError("goals must be non-negative")
    completed = pd.to_datetime(completed_at, utc=True, errors="coerce")
    if pd.isna(completed):
        raise ValueError(f"unparseable completed-at timestamp: {completed_at!r}")
    if kickoff_date is not None and completed.date() < pd.Timestamp(kickoff_date).date():
        raise ValueError(
            f"completed-at {completed_at} is before the kickoff date {kickoff_date}"
        )

    store = resolve(path)
    store.parent.mkdir(parents=True, exist_ok=True)
    existing = _read(store)
    prior = [e for e in existing if e["payload"]["fixture_id"] == fixture_id]
    if prior and corrects is None:
        raise ValueError(
            f"a result for fixture {fixture_id[:12]}… already exists "
            f"(entry {prior[-1]['entry_hash'][:12]}…); use the audited correction "
            "workflow (result correct) to supersede it"
        )
    if corrects is not None:
        if not correction_reason:
            raise ValueError("corrections require an explicit --reason")
        targets = [e for e in existing if e["entry_hash"] == corrects]
        if not targets:
            raise ValueError(f"corrects target {corrects[:12]}… not found in store")
        if targets[0]["payload"]["fixture_id"] != fixture_id:
            raise ValueError("correction target belongs to a different fixture")
        if prior[-1]["entry_hash"] != corrects:
            raise ValueError("correction must supersede the current active result")

    payload = {
        "fixture_id": fixture_id,
        "regulation_home_goals": int(home_goals),
        "regulation_away_goals": int(away_goals),
        "outcome": _outcome(home_goals, away_goals),
        "completed_at": completed_at,
        "completed_at_time_known": "T" in completed_at,
        "timestamp_precision": "datetime" if "T" in completed_at else "date",
        "time_known": "T" in completed_at,
        "source": source,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if match_date is not None:
        payload["match_date"] = match_date
    if home_team is not None:
        payload["home_team"] = home_team
    if away_team is not None:
        payload["away_team"] = away_team
    if corrects is not None:
        payload["corrects"] = corrects
        payload["correction_reason"] = correction_reason

    prev_hash = existing[-1]["entry_hash"] if existing else GENESIS
    entry = {
        "payload": payload,
        "prev_hash": prev_hash,
        "entry_hash": sha256_text(prev_hash + _canonical(payload)),
    }
    with open(store, "a", encoding="utf-8") as f:
        f.write(_canonical(entry) + "\n")
    return entry


def verify_results(path: str | Path = DEFAULT_RESULTS_PATH) -> list[str]:
    """Integrity violations in the result store (empty = intact)."""
    problems = []
    prev_hash = GENESIS
    for i, entry in enumerate(_read(resolve(path))):
        if entry.get("prev_hash") != prev_hash:
            problems.append(f"result entry {i}: chain break")
        expected = sha256_text(entry.get("prev_hash", "") + _canonical(entry["payload"]))
        if entry.get("entry_hash") != expected:
            problems.append(f"result entry {i}: payload hash mismatch")
        prev_hash = entry.get("entry_hash", "")
    return problems


def apply_results_overlay(
    matches: pd.DataFrame, results: dict[str, dict]
) -> tuple[pd.DataFrame, int]:
    """Mark scheduled fixtures as played using recorded results.

    Returns (new frame, number applied). The canonical dataset itself is never
    touched; this is an in-memory view used to advance ratings, cutoffs, and
    simulations. Results for unknown or already-played fixtures raise rather
    than being silently ignored.
    """
    if not results:
        return matches, 0
    matches = matches.copy()
    idx_by_fixture = {
        row.canonical_match_id: i
        for i, row in enumerate(matches.itertuples(index=False))
    }
    applied = 0
    for fixture_id, r in results.items():
        if fixture_id not in idx_by_fixture:
            raise ValueError(f"recorded result for unknown fixture {fixture_id[:12]}…")
        i = idx_by_fixture[fixture_id]
        if matches.iloc[i]["status"] == "played":
            raise ValueError(
                f"fixture {fixture_id[:12]}… is already played in the dataset; "
                "conflicting recorded result"
            )
        hg, ag = r["regulation_home_goals"], r["regulation_away_goals"]
        matches.iloc[i, matches.columns.get_loc("status")] = "played"
        matches.iloc[i, matches.columns.get_loc("home_score_recorded")] = hg
        matches.iloc[i, matches.columns.get_loc("away_score_recorded")] = ag
        matches.iloc[i, matches.columns.get_loc("regulation_home_score")] = hg
        matches.iloc[i, matches.columns.get_loc("regulation_away_score")] = ag
        matches.iloc[i, matches.columns.get_loc("regulation_score_known")] = True
        matches.iloc[i, matches.columns.get_loc("regulation_outcome")] = r["outcome"]
        matches.iloc[i, matches.columns.get_loc("recorded_score_scope")] = "regulation"
        matches.iloc[i, matches.columns.get_loc("extra_time_played")] = False
        matches.iloc[i, matches.columns.get_loc("extra_time_status_known")] = True
        matches.iloc[i, matches.columns.get_loc("shootout_played")] = False
        matches.iloc[i, matches.columns.get_loc("strict_goal_model_eligible")] = True
        matches.iloc[i, matches.columns.get_loc("strict_exclusion_reason")] = None
        applied += 1
    return matches, applied
