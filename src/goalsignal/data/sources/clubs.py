"""Canonical club identity resolution (contract + pure normalizer)."""

from __future__ import annotations

import unicodedata

from goalsignal.data.sources.base import MilestoneNotImplementedError
from goalsignal.data.sources.config import PlayerFeaturesConfig


def normalize_club_name(name: str) -> str:
    """Casefold/strip accents and common suffixes for alias comparison."""
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    tokens = [t for t in stripped.casefold().split() if t not in {"fc", "cf", "sc", "ac"}]
    return " ".join(tokens)


class ClubResolver:
    name = "clubs"
    role = "club_identity"

    def __init__(self, config: PlayerFeaturesConfig | None = None):
        self.config = config or PlayerFeaturesConfig()

    def resolve_clubs(self, aliases):
        raise MilestoneNotImplementedError(
            "Club resolution + alias-store I/O land in Milestone C."
        )
