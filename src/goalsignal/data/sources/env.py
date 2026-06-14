"""Minimal .env loader (no third-party dependency).

Reads KEY=VALUE lines from a git-ignored `.env` into the process environment
without overriding variables already set in the real environment. Values are
never logged. This keeps credentials out of source while needing no
python-dotenv dependency.
"""

from __future__ import annotations

import os

from goalsignal.utils.paths import resolve


def load_env_file(path: str = ".env") -> list[str]:
    """Load `.env` into os.environ; return the names (not values) that were set."""
    p = resolve(path)
    if not p.exists():
        return []
    loaded: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def has_env(name: str) -> bool:
    """True when `name` is set and non-empty (value never exposed)."""
    return bool(os.environ.get(name))
