"""Group standings and tiebreakers.

Implements the FIFA group procedure: rank by points, then goal difference,
then goals scored; any remaining tie is re-ranked by the same criteria
restricted to matches among the tied teams (head-to-head mini-table), and a
final unresolved tie is decided by drawing of lots (an explicit RNG draw,
recorded as such). Every ranking decision is explainable via the returned
tiebreak trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POINTS = {"win": 3, "draw": 1, "loss": 0}


@dataclass
class TeamRecord:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def points(self) -> int:
        return 3 * self.wins + self.draws

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    def key(self) -> tuple:
        return (self.points, self.goal_difference, self.goals_for)


@dataclass
class GroupResult:
    ranking: list[str]
    records: dict[str, TeamRecord]
    tiebreaks: list[str] = field(default_factory=list)


def build_records(
    teams: list[str], results: list[tuple[str, str, int, int]]
) -> dict[str, TeamRecord]:
    """results: (home, away, home_goals, away_goals) among the given teams."""
    records = {t: TeamRecord(t) for t in teams}
    for home, away, hg, ag in results:
        if home not in records or away not in records:
            continue
        rh, ra = records[home], records[away]
        rh.played += 1
        ra.played += 1
        rh.goals_for += hg
        rh.goals_against += ag
        ra.goals_for += ag
        ra.goals_against += hg
        if hg > ag:
            rh.wins += 1
            ra.losses += 1
        elif hg < ag:
            ra.wins += 1
            rh.losses += 1
        else:
            rh.draws += 1
            ra.draws += 1
    return records


def _rank(
    teams: list[str],
    all_results: list[tuple[str, str, int, int]],
    records: dict[str, TeamRecord],
    rng: np.random.Generator,
    tiebreaks: list[str],
    depth: int = 0,
) -> list[str]:
    ordered = sorted(teams, key=lambda t: records[t].key(), reverse=True)
    final: list[str] = []
    i = 0
    while i < len(ordered):
        tied = [ordered[i]]
        while (
            i + len(tied) < len(ordered)
            and records[ordered[i + len(tied)]].key() == records[tied[0]].key()
        ):
            tied.append(ordered[i + len(tied)])
        if len(tied) == 1:
            final.append(tied[0])
        elif depth >= 1 or len(tied) == len(teams):
            # Head-to-head already applied (or is the full group, hence
            # identical): drawing of lots.
            order = list(rng.permutation(tied))
            tiebreaks.append(f"lots:{'/'.join(sorted(tied))}")
            final.extend(order)
        else:
            h2h = [r for r in all_results if r[0] in tied and r[1] in tied]
            sub_records = build_records(tied, h2h)
            tiebreaks.append(f"head_to_head:{'/'.join(sorted(tied))}")
            final.extend(_rank(tied, h2h, sub_records, rng, tiebreaks, depth + 1))
        i += len(tied)
    return final


def group_standings(
    teams: list[str],
    results: list[tuple[str, str, int, int]],
    rng: np.random.Generator,
) -> GroupResult:
    records = build_records(teams, results)
    tiebreaks: list[str] = []
    ranking = _rank(teams, results, records, rng, tiebreaks)
    return GroupResult(ranking=ranking, records=records, tiebreaks=tiebreaks)


def rank_third_placed(
    thirds: list[TeamRecord], rng: np.random.Generator
) -> list[str]:
    """Rank third-placed teams across groups: points, GD, GF, then lots."""
    keys = {r.team: (r.points, r.goal_difference, r.goals_for) for r in thirds}
    jitter = {r.team: rng.random() for r in thirds}
    return sorted(
        (r.team for r in thirds),
        key=lambda t: (*keys[t], jitter[t]),
        reverse=True,
    )
