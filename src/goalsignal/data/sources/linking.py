"""Deterministic fixture linking between supplementary sources and the
canonical GoalSignal match table.

Links are classified, never silently accepted. Ambiguous and unmatched records
are reported for human/Milestone-C review. Orientation correction (home/away
reversed) is detected and labelled `reversed` but treated as a separate class,
not folded into `exact`.
"""

from __future__ import annotations

import unicodedata

import pandas as pd


def normalize_team(name: str) -> str:
    if not isinstance(name, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(ch for ch in stripped.casefold().split())


def _date(value) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else ts.date().isoformat()


def link_fixtures(
    source_fixtures: pd.DataFrame,
    canonical_matches: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    """Classify each source fixture against canonical matches.

    `source_fixtures` needs: source_fixture_id, match_date, home_team, away_team.
    `canonical_matches` is the GoalSignal table (canonical_match_id, date,
    home_team, away_team). Returns a links frame with a `link_type` column:
    exact | reversed | ambiguous | unmatched.
    """
    index: dict[tuple, list[str]] = {}
    for row in canonical_matches.itertuples(index=False):
        key = (_date(row.date), normalize_team(row.home_team), normalize_team(row.away_team))
        index.setdefault(key, []).append(row.canonical_match_id)

    out = []
    for f in source_fixtures.itertuples(index=False):
        d = _date(f.match_date)
        h, a = normalize_team(f.home_team), normalize_team(f.away_team)
        forward = index.get((d, h, a), [])
        reverse = index.get((d, a, h), [])
        if len(forward) == 1:
            link_type, cid = "exact", forward[0]
        elif len(forward) > 1:
            link_type, cid = "ambiguous", None
        elif len(reverse) == 1:
            link_type, cid = "reversed", reverse[0]
        elif len(reverse) > 1:
            link_type, cid = "ambiguous", None
        else:
            link_type, cid = "unmatched", None
        out.append(
            {
                "source": source_name,
                "source_fixture_id": getattr(f, "source_fixture_id", None),
                "match_date": d,
                "home_team": f.home_team,
                "away_team": f.away_team,
                "link_type": link_type,
                "canonical_match_id": cid,
                "candidate_count": len(forward) + len(reverse),
            }
        )
    return pd.DataFrame(
        out,
        columns=["source", "source_fixture_id", "match_date", "home_team", "away_team",
                 "link_type", "canonical_match_id", "candidate_count"],
    )


def link_summary(links: pd.DataFrame) -> dict:
    counts = links["link_type"].value_counts().to_dict() if len(links) else {}
    total = len(links)
    accepted = counts.get("exact", 0) + counts.get("reversed", 0)
    return {
        "total": total,
        "exact": counts.get("exact", 0),
        "reversed": counts.get("reversed", 0),
        "ambiguous": counts.get("ambiguous", 0),
        "unmatched": counts.get("unmatched", 0),
        "link_rate": round(accepted / total, 4) if total else None,
    }
