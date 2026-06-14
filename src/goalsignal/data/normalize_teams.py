"""Date-aware team-name normalization using former_names.csv.

A former name maps to its current name only for matches dated within the
period the former name was in use. Applying mappings without the date check
would, for example, relabel historically distinct teams or rewrite eras in
which a country genuinely competed under the old name.

Conflicting mappings (the same former name active for two different current
names on overlapping dates) are surfaced for review, never resolved silently.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class NameMappingIssue:
    kind: str
    detail: str


@dataclass
class TeamNormalizer:
    """Resolves team names as of a given match date."""

    # former name -> list of (start, end, current)
    _periods: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = field(
        default_factory=dict
    )
    issues: list[NameMappingIssue] = field(default_factory=list)

    @classmethod
    def from_former_names(cls, former_names: pd.DataFrame) -> TeamNormalizer:
        norm = cls()
        for row in former_names.itertuples(index=False):
            current = str(row.current).strip()
            former = str(row.former).strip()
            if not current or not former:
                norm.issues.append(
                    NameMappingIssue("empty_name", f"row {row.source_row}: empty name field")
                )
                continue
            start = pd.to_datetime(str(row.start_date), errors="coerce")
            end = pd.to_datetime(str(row.end_date), errors="coerce")
            if pd.isna(start) or pd.isna(end):
                norm.issues.append(
                    NameMappingIssue(
                        "invalid_dates",
                        f"row {row.source_row}: unparseable dates for {former!r}",
                    )
                )
                continue
            if start > end:
                norm.issues.append(
                    NameMappingIssue(
                        "inverted_period",
                        f"row {row.source_row}: start {start.date()} after end "
                        f"{end.date()} for {former!r}",
                    )
                )
                continue
            norm._periods.setdefault(former, []).append((start, end, current))

        # Detect overlapping periods for the same former name with different targets.
        for former, periods in norm._periods.items():
            periods.sort()
            for (s1, e1, c1), (s2, e2, c2) in itertools.pairwise(periods):
                if s2 <= e1 and c1 != c2:
                    norm.issues.append(
                        NameMappingIssue(
                            "overlapping_mapping",
                            f"{former!r} maps to both {c1!r} ({s1.date()} to {e1.date()}) and "
                            f"{c2!r} ({s2.date()} to {e2.date()}) on overlapping dates",
                        )
                    )
        # Detect chains (a current name that is itself someone's former name).
        currents = {c for ps in norm._periods.values() for _, _, c in ps}
        for former in norm._periods:
            if former in currents:
                norm.issues.append(
                    NameMappingIssue(
                        "chained_mapping",
                        f"{former!r} is both a former name and a current name; "
                        "mappings are applied one step only",
                    )
                )
        return norm

    def canonical(self, name: str, match_date: pd.Timestamp) -> str:
        """Return the canonical (current) name for a team as of match_date.

        Names with no active mapping on the match date are returned unchanged.
        Mappings are applied a single step; chained renames are flagged at
        construction time rather than resolved transitively.
        """
        name = name.strip()
        for start, end, current in self._periods.get(name, ()):
            if start <= match_date <= end:
                return current
        return name

    def known_former_names(self) -> set[str]:
        return set(self._periods)
