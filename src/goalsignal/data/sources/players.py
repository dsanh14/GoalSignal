"""Canonical player identity resolution (contract + pure matcher).

Players are never merged on name alone. Resolution uses a source ID when
available; otherwise it requires the name plus at least one corroborating
identifier (date of birth, or nationality+club). Anything weaker is reported
as ambiguous or unmatched with a review status — never silently merged.
"""

from __future__ import annotations

import unicodedata

from goalsignal.data.sources.base import MilestoneNotImplementedError
from goalsignal.data.sources.config import PlayerFeaturesConfig


def normalize_player_name(name: str) -> str:
    """Casefold and strip accents for comparison (display name kept elsewhere)."""
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.casefold().split())


def resolve_player(query: dict, candidates: list[dict], source: str) -> dict:
    """Resolve `query` against canonical `candidates`.

    Returns {"status", "canonical_player_id" | None, "reason"} where status is
    one of: matched, ambiguous, unmatched. Matching precedence:

    1. Exact source-ID match (per `source`) — unambiguous.
    2. Normalized name + date_of_birth agreement.
    3. Normalized name + nationality + club agreement.

    Name-only agreement is treated as ambiguous, never a match.
    """
    # 1. Source ID.
    sid = query.get("source_player_ids", {}).get(source)
    if sid:
        hits = [c for c in candidates if c.get("source_player_ids", {}).get(source) == sid]
        if len(hits) == 1:
            return {"status": "matched", "canonical_player_id": hits[0]["canonical_player_id"],
                    "reason": "source_id"}

    qname = normalize_player_name(query.get("full_name", ""))
    if not qname:
        return {"status": "unmatched", "canonical_player_id": None, "reason": "no_name"}
    name_hits = [c for c in candidates if normalize_player_name(c.get("full_name", "")) == qname]
    if not name_hits:
        return {"status": "unmatched", "canonical_player_id": None, "reason": "no_name_match"}

    # 2. Name + DOB.
    if query.get("date_of_birth"):
        dob_hits = [c for c in name_hits if c.get("date_of_birth") == query["date_of_birth"]]
        if len(dob_hits) == 1:
            return {"status": "matched", "canonical_player_id": dob_hits[0]["canonical_player_id"],
                    "reason": "name_dob"}

    # 3. Name + nationality + club.
    if query.get("nationality") and query.get("club"):
        nc_hits = [
            c for c in name_hits
            if c.get("nationality") == query["nationality"] and c.get("club") == query["club"]
        ]
        if len(nc_hits) == 1:
            return {"status": "matched",
                    "canonical_player_id": nc_hits[0]["canonical_player_id"],
                    "reason": "name_nationality_club"}

    return {
        "status": "ambiguous",
        "canonical_player_id": None,
        "reason": f"name matched {len(name_hits)} candidate(s) without a corroborating "
        "identifier; requires review",
    }


class PlayerResolver:
    name = "players"
    role = "player_identity"

    def __init__(self, config: PlayerFeaturesConfig | None = None):
        self.config = config or PlayerFeaturesConfig()

    def resolve_players(self, aliases):
        raise MilestoneNotImplementedError(
            "Batch player resolution + alias-store I/O land in Milestone C. Use the "
            "pure `resolve_player(query, candidates, source)` function now."
        )
