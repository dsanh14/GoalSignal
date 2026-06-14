"""2026 FIFA World Cup fixtures derived from the user-provided dataset.

The dataset contains the 72 group-stage fixtures (some already played). Group
*membership* is derived from the fixture graph: each group of four plays six
internal matches, so groups are exactly the connected components of the
team-vs-team graph. Official group letters are not present in the data, so
groups get synthetic labels (G01..G12, ordered by first team name) — labeled
as synthetic, never invented.

The official Round-of-32 bracket mapping (which group winners meet which
thirds) is NOT in the dataset and is deliberately not fabricated; knockout
simulation beyond R32 qualification requires the user to supply
config/tournament_2026.yaml with the official bracket.
"""

from __future__ import annotations

import pandas as pd

from goalsignal.tournament.simulator import GroupFixture


class FixtureDerivationError(ValueError):
    pass


def derive_2026_group_stage(
    matches: pd.DataFrame,
) -> tuple[dict[str, list[str]], list[GroupFixture]]:
    wc = matches[
        (matches["tournament"] == "FIFA World Cup")
        & (matches["date"] >= pd.Timestamp("2026-01-01"))
    ]
    if len(wc) == 0:
        raise FixtureDerivationError("no 2026 FIFA World Cup fixtures found in dataset")

    # Union-find over teams to recover groups from the fixture graph.
    parent: dict[str, str] = {}

    def find(t: str) -> str:
        parent.setdefault(t, t)
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    for row in wc.itertuples(index=False):
        ra, rb = find(row.home_team), find(row.away_team)
        if ra != rb:
            parent[ra] = rb

    components: dict[str, list[str]] = {}
    for team in parent:
        components.setdefault(find(team), []).append(team)

    if len(components) != 12 or any(len(c) != 4 for c in components.values()):
        sizes = sorted(len(c) for c in components.values())
        raise FixtureDerivationError(
            f"expected 12 groups of 4 from the fixture graph, got components of sizes "
            f"{sizes}. The dataset may contain knockout fixtures or be incomplete; "
            "refusing to guess."
        )

    ordered = sorted(components.values(), key=lambda c: min(c))
    groups = {f"G{i + 1:02d}": sorted(c) for i, c in enumerate(ordered)}
    team_to_group = {t: g for g, ts in groups.items() for t in ts}

    fixtures = []
    for row in wc.sort_values(["date", "source_row"]).itertuples(index=False):
        played = row.status == "played"
        fixtures.append(
            GroupFixture(
                group=team_to_group[row.home_team],
                home=row.home_team,
                away=row.away_team,
                fixture_id=row.canonical_match_id,
                neutral=bool(row.neutral) if row.neutral is not None else True,
                played=played,
                home_goals=int(row.home_score_recorded) if played else None,
                away_goals=int(row.away_score_recorded) if played else None,
            )
        )
    expected = 6 * 12
    if len(fixtures) != expected:
        raise FixtureDerivationError(
            f"expected {expected} group fixtures, found {len(fixtures)}"
        )
    return groups, fixtures
