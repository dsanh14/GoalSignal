"""Unit tests for hashing utilities."""

from __future__ import annotations

from goalsignal.utils.hashing import sha256_file, sha256_json, sha256_text


def test_sha256_text_is_deterministic():
    assert sha256_text("goalsignal") == sha256_text("goalsignal")
    assert sha256_text("a") != sha256_text("b")


def test_sha256_json_is_order_invariant():
    assert sha256_json({"a": 1, "b": [2, 3]}) == sha256_json({"b": [2, 3], "a": 1})
    assert sha256_json({"a": 1}) != sha256_json({"a": 2})


def test_sha256_file_matches_content(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    q = tmp_path / "g.txt"
    q.write_text("hello", encoding="utf-8")
    assert sha256_file(p) == sha256_file(q)
