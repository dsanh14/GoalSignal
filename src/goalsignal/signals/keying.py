"""Resolve manual signal entries by match id or normalized team pair.

Knockout pairings are generated dynamically during simulation, so a manual
signal keyed only by ``match_id`` never attaches to them. This module adds a
robust, minimal fallback: a signal row may carry ``team_a``/``team_b`` and is
then resolvable by the (orientation-aware) team pair.

**Precedence (highest first):**

1. exact ``match_id`` match,
2. forward team-pair match — the fixture is ``(team_a, team_b)`` in the same
   order the row lists them (orientation ``+1``),
3. reverse team-pair match — the fixture lists the teams swapped
   (orientation ``-1``); the caller flips the directional probabilities.

Team names are normalized (trimmed, casefolded, whitespace collapsed) so minor
formatting differences still match. Orientation is returned alongside the
payload so directional signals (market odds, expert W/D/L, venue A-advantage)
can be flipped when matched in reverse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WS = re.compile(r"\s+")


def normalize_team(name: str | None) -> str:
    """Canonicalize a team name for pair matching."""
    if name is None:
        return ""
    return _WS.sub(" ", str(name).strip().casefold())


def pair_key(team_a: str | None, team_b: str | None) -> str:
    """Directional, normalized key for an ordered (team_a, team_b) pair."""
    return f"{normalize_team(team_a)}\x1f{normalize_team(team_b)}"


@dataclass
class PairIndex[T]:
    """Index of payloads resolvable by match id and/or team pair.

    ``orientation`` is ``+1`` for a match-id or forward-pair hit and ``-1`` for a
    reverse-pair hit. ``0`` means "not found" (payload ``None``).
    """

    by_match: dict[str, T]
    by_pair: dict[str, tuple[T, int]]

    @classmethod
    def build(cls, entries: list[tuple[str | None, str | None, str | None, T]]) -> PairIndex[T]:
        """Build from ``(match_id, team_a, team_b, payload)`` tuples.

        A forward pair always wins over a reverse pair, and the first entry for a
        given key wins (so explicit rows are stable). Rows missing both a
        ``match_id`` and a team pair are ignored.
        """
        by_match: dict[str, T] = {}
        by_pair: dict[str, tuple[T, int]] = {}
        for match_id, team_a, team_b, payload in entries:
            if match_id:
                by_match.setdefault(str(match_id).strip(), payload)
            if team_a and team_b:
                fwd = pair_key(team_a, team_b)
                rev = pair_key(team_b, team_a)
                # Forward orientation takes priority; don't let a reverse entry
                # shadow an existing forward one.
                prev = by_pair.get(fwd)
                if prev is None or prev[1] == -1:
                    by_pair[fwd] = (payload, 1)
                by_pair.setdefault(rev, (payload, -1))
        return cls(by_match=by_match, by_pair=by_pair)

    def resolve(
        self, match_id: str | None, team_a: str | None, team_b: str | None
    ) -> tuple[T | None, int]:
        """Return ``(payload, orientation)`` using the documented precedence."""
        if match_id:
            hit = self.by_match.get(str(match_id).strip())
            if hit is not None:
                return hit, 1
        if team_a and team_b:
            hit = self.by_pair.get(pair_key(team_a, team_b))
            if hit is not None:
                return hit  # (payload, orientation)
        return None, 0

    def __len__(self) -> int:
        return len(self.by_match) + len(self.by_pair)
