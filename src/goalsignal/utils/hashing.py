"""Deterministic hashing utilities for dataset versioning and match identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed to bound memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """SHA-256 of canonical (sorted-key, compact) JSON serialization."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(canonical)


def canonical_match_id(
    date: str,
    home_team: str,
    away_team: str,
    tournament: str,
    city: str,
    country: str,
) -> str:
    """Deterministic match identifier from normalized identity fields.

    Scores are deliberately excluded: they may be corrected upstream without
    changing which match the row refers to.
    """
    key = "|".join(
        part.strip().casefold()
        for part in (date, home_team, away_team, tournament, city, country)
    )
    return sha256_text(key)
