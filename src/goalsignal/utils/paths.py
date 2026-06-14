"""Repository path resolution.

All relative paths in configuration are resolved against the repository root,
so commands behave identically regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Locate the repository root by walking up from this file to pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not locate repository root (no pyproject.toml found)")


def resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p
