"""Append-only, hash-chained prediction ledger.

Entries are serialized as canonical JSON (sorted keys, compact separators) in
a JSONL file. Each entry carries `entry_hash` (SHA-256 of its canonical
payload) and `prev_hash` (the previous entry's hash), so any retroactive
modification, deletion, or reordering breaks verification. Results are stored
elsewhere; the ledger holds only immutable pre-match predictions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from goalsignal.utils.hashing import sha256_text
from goalsignal.utils.paths import resolve

GENESIS = "0" * 64
DEFAULT_PATH = "artifacts/predictions/ledger.jsonl"


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_predictions(
    payloads: list[dict], ledger_path: str | Path = DEFAULT_PATH
) -> list[dict]:
    """Append prediction payloads; returns the stored entries with hashes.

    A payload must not contain hash/bookkeeping fields; they are added here.
    Duplicate fixture_id + model_version pairs are rejected to prevent silent
    overwrites — corrections must be new entries with a new model_version.
    """
    path = resolve(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_entries(path)
    seen = {(e["payload"].get("fixture_id"), e["payload"].get("model_version"))
            for e in existing}
    prev_hash = existing[-1]["entry_hash"] if existing else GENESIS

    stored = []
    with open(path, "a", encoding="utf-8") as f:
        for payload in payloads:
            key = (payload.get("fixture_id"), payload.get("model_version"))
            if key in seen:
                raise ValueError(
                    f"ledger already has a prediction for fixture={key[0]} "
                    f"model={key[1]}; refusing to overwrite"
                )
            seen.add(key)
            payload = dict(payload)
            payload.setdefault(
                "prediction_timestamp", datetime.now(UTC).isoformat(timespec="seconds")
            )
            entry = {
                "payload": payload,
                "prev_hash": prev_hash,
                "entry_hash": sha256_text(prev_hash + _canonical(payload)),
            }
            f.write(_canonical(entry) + "\n")
            prev_hash = entry["entry_hash"]
            stored.append(entry)
    return stored


def verify_ledger(ledger_path: str | Path = DEFAULT_PATH) -> list[str]:
    """Return a list of integrity violations (empty = ledger intact)."""
    entries = _read_entries(resolve(ledger_path))
    problems = []
    prev_hash = GENESIS
    for i, entry in enumerate(entries):
        if entry.get("prev_hash") != prev_hash:
            problems.append(f"entry {i}: chain break (prev_hash mismatch)")
        expected = sha256_text(entry.get("prev_hash", "") + _canonical(entry["payload"]))
        if entry.get("entry_hash") != expected:
            problems.append(f"entry {i}: payload hash mismatch (entry modified?)")
        prev_hash = entry.get("entry_hash", "")
    return problems


def list_entries(ledger_path: str | Path = DEFAULT_PATH) -> list[dict]:
    return _read_entries(resolve(ledger_path))
