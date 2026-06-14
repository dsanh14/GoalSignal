"""Unit tests for the append-only prediction ledger."""

from __future__ import annotations

import json

import pytest

from goalsignal.ledger.storage import append_predictions, list_entries, verify_ledger


def _payload(fid: str) -> dict:
    return {
        "fixture_id": fid,
        "home_team": "Atlantis",
        "away_team": "Ruritania",
        "kickoff_timestamp": "2030-01-01",
        "model_version": "test-v1",
        "home_win_probability": 0.5,
        "draw_probability": 0.3,
        "away_win_probability": 0.2,
    }


def test_append_and_verify(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_predictions([_payload("f1"), _payload("f2")], path)
    assert verify_ledger(path) == []
    assert len(list_entries(path)) == 2


def test_duplicate_prediction_rejected(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_predictions([_payload("f1")], path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        append_predictions([_payload("f1")], path)
    assert len(list_entries(path)) == 1


def test_tampered_entry_fails_verification(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_predictions([_payload("f1"), _payload("f2")], path)
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    entries[0]["payload"]["home_win_probability"] = 0.99  # retroactive edit
    path.write_text(
        "\n".join(json.dumps(e, sort_keys=True, separators=(",", ":")) for e in entries)
        + "\n"
    )
    problems = verify_ledger(path)
    assert any("hash mismatch" in p for p in problems)


def test_deleted_entry_breaks_chain(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_predictions([_payload("f1"), _payload("f2"), _payload("f3")], path)
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")  # drop the middle
    problems = verify_ledger(path)
    assert any("chain break" in p for p in problems)


def test_appending_continues_chain(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_predictions([_payload("f1")], path)
    append_predictions([_payload("f2")], path)
    assert verify_ledger(path) == []
    entries = list_entries(path)
    assert entries[1]["prev_hash"] == entries[0]["entry_hash"]
