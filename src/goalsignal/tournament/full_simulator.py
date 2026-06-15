"""Full 2026 World Cup simulation from groups through the champion."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.stats import skellam

from goalsignal.tournament.bracket_2026 import GROUPS, OfficialBracket
from goalsignal.tournament.rules import group_standings
from goalsignal.tournament.simulator import GroupFixture, _presample_scores

STAGES = ("round_of_32", "round_of_16", "quarterfinal", "semifinal", "final", "champion")


@dataclass
class FullSimulationResult:
    n_sims: int
    seed: int
    teams: list[str]
    groups: dict[str, list[str]]
    position_probs: dict[str, list[float]]
    expected_points: dict[str, float]
    best_third_probs: dict[str, float]
    advancement_probs: dict[str, dict[str, float]]
    third_place_probs: dict[str, float]
    fourth_place_probs: dict[str, float]
    matchup_counts: dict[int, Counter]
    winner_counts: dict[int, Counter]
    resolution_counts: Counter

    def mc_standard_error(self, p: float) -> float:
        return float(np.sqrt(p * (1.0 - p) / self.n_sims))


def apply_official_group_letters(
    groups: dict[str, list[str]],
    fixtures: list[GroupFixture],
    official_groups: dict[str, list[str]],
) -> tuple[dict[str, list[str]], list[GroupFixture]]:
    if set(official_groups) != GROUPS:
        raise ValueError("official groups must be exactly A-L")
    official_team_group = {
        team: group for group, teams in official_groups.items() for team in teams
    }
    if set(official_team_group) != {team for teams in groups.values() for team in teams}:
        raise ValueError("official group teams do not match the fixture graph")
    relabeled = []
    for fixture in fixtures:
        group = official_team_group[fixture.home]
        if official_team_group[fixture.away] != group:
            raise ValueError(f"fixture crosses official groups: {fixture.home} v {fixture.away}")
        relabeled.append(
            GroupFixture(
                group=group,
                home=fixture.home,
                away=fixture.away,
                fixture_id=fixture.fixture_id,
                neutral=fixture.neutral,
                played=fixture.played,
                home_goals=fixture.home_goals,
                away_goals=fixture.away_goals,
            )
        )
    return {g: list(official_groups[g]) for g in sorted(GROUPS)}, relabeled


def _simulate_group_orders(groups, fixtures, model, n_sims, rng):
    sampled = _presample_scores(fixtures, model, n_sims, rng)
    teams = [team for group in sorted(groups) for team in groups[group]]
    global_index = {team: i for i, team in enumerate(teams)}
    group_order = np.zeros((12, n_sims, 4), dtype=np.int16)
    third_keys = np.zeros((12, n_sims, 3), dtype=np.int16)
    pos_counts = {team: np.zeros(4, dtype=np.int64) for team in teams}
    points_sum = dict.fromkeys(teams, 0.0)
    fixture_idx = {g: [] for g in groups}
    for i, fixture in enumerate(fixtures):
        fixture_idx[fixture.group].append(i)

    for gi, group in enumerate(sorted(groups)):
        gteams = groups[group]
        local = {team: i for i, team in enumerate(gteams)}
        pts = np.zeros((n_sims, 4), dtype=np.int16)
        gf = np.zeros((n_sims, 4), dtype=np.int16)
        ga = np.zeros((n_sims, 4), dtype=np.int16)
        for i in fixture_idx[group]:
            fixture = fixtures[i]
            hi, ai = local[fixture.home], local[fixture.away]
            if fixture.played:
                hs = np.full(n_sims, fixture.home_goals, dtype=np.int16)
                away_scores = np.full(n_sims, fixture.away_goals, dtype=np.int16)
            else:
                hs, away_scores = sampled[i]
            gf[:, hi] += hs
            ga[:, hi] += away_scores
            gf[:, ai] += away_scores
            ga[:, ai] += hs
            pts[:, hi] += np.where(hs > away_scores, 3, np.where(hs == away_scores, 1, 0))
            pts[:, ai] += np.where(away_scores > hs, 3, np.where(hs == away_scores, 1, 0))
        gd = gf - ga
        key = ((pts.astype(np.int64) * 512) + (gd + 256)) * 512 + gf
        order = np.argsort(-key, axis=1, kind="stable")
        sorted_key = np.take_along_axis(key, order, axis=1)
        tied = np.flatnonzero((np.diff(sorted_key, axis=1) == 0).any(axis=1))
        for sim in tied:
            results = []
            for i in fixture_idx[group]:
                fixture = fixtures[i]
                if fixture.played:
                    score = (fixture.home_goals, fixture.away_goals)
                else:
                    score = (int(sampled[i][0][sim]), int(sampled[i][1][sim]))
                results.append((fixture.home, fixture.away, *score))
            standing = group_standings(gteams, results, rng)
            order[sim] = [local[team] for team in standing.ranking]
        global_order = np.take(
            np.array([global_index[team] for team in gteams], dtype=np.int16), order
        )
        group_order[gi] = global_order
        third_local = order[:, 2]
        rows = np.arange(n_sims)
        third_keys[gi, :, 0] = pts[rows, third_local]
        third_keys[gi, :, 1] = gd[rows, third_local]
        third_keys[gi, :, 2] = gf[rows, third_local]
        for pos in range(4):
            for team_i, team in enumerate(gteams):
                pos_counts[team][pos] += int((order[:, pos] == team_i).sum())
        for team_i, team in enumerate(gteams):
            points_sum[team] += float(pts[:, team_i].sum())

    jitter = rng.random((12, n_sims))
    rank_key = (
        ((third_keys[:, :, 0].astype(np.int64) * 512) + (third_keys[:, :, 1] + 256))
        * 512
        + third_keys[:, :, 2]
    ).astype(float) + jitter
    best_third_group_indices = np.argsort(-rank_key, axis=0)[:8]
    third_counts = dict.fromkeys(teams, 0)
    for rank in range(8):
        group_indices = best_third_group_indices[rank]
        occupants = group_order[group_indices, np.arange(n_sims), 2]
        for team_i in np.unique(occupants):
            third_counts[teams[int(team_i)]] += int((occupants == team_i).sum())
    return (
        teams,
        group_order,
        best_third_group_indices,
        pos_counts,
        points_sum,
        third_counts,
    )


def _pair_resolution_probabilities(home: str, away: str, model) -> np.ndarray:
    lam_home, lam_away = model.expected_goals(home, away, True)
    matrix = model.score_matrix(lam_home, lam_away)
    matrix = matrix / matrix.sum()
    reg_home = float(np.tril(matrix, -1).sum())
    reg_away = float(np.triu(matrix, 1).sum())
    draw = float(np.trace(matrix))
    et_home = float(1.0 - skellam.cdf(0, lam_home / 3, lam_away / 3))
    et_away = float(skellam.cdf(-1, lam_home / 3, lam_away / 3))
    et_draw = float(skellam.pmf(0, lam_home / 3, lam_away / 3))
    return np.array(
        [
            reg_home,
            reg_away,
            draw * et_home,
            draw * et_away,
            draw * et_draw * 0.5,
            draw * et_draw * 0.5,
        ]
    )


def simulate_full_tournament(
    groups: dict[str, list[str]],
    fixtures: list[GroupFixture],
    model,
    bracket: OfficialBracket,
    n_sims: int = 100_000,
    seed: int = 20260612,
) -> FullSimulationResult:
    rng = np.random.default_rng(seed)
    (
        teams,
        group_orders,
        best_third_indices,
        pos_counts,
        points_sum,
        third_counts,
    ) = _simulate_group_orders(groups, fixtures, model, n_sims, rng)
    group_letters = sorted(groups)
    advancement = {team: dict.fromkeys(STAGES, 0) for team in teams}
    third_place = dict.fromkeys(teams, 0)
    fourth_place = dict.fromkeys(teams, 0)
    matchup_counts = defaultdict(Counter)
    winner_counts = defaultdict(Counter)
    resolution_counts = Counter()
    pair_cache = {}

    def play(number, home, away):
        key = (home, away)
        if key not in pair_cache:
            pair_cache[key] = np.cumsum(_pair_resolution_probabilities(home, away, model))
        bucket = min(
            int(np.searchsorted(pair_cache[key], rng.random(), side="right")), 5
        )
        winner = home if bucket in (0, 2, 4) else away
        loser = away if winner == home else home
        stage = ("regulation", "regulation", "extra_time", "extra_time", "penalties",
                 "penalties")[bucket]
        matchup_counts[number][(home, away)] += 1
        winner_counts[number][(home, away, winner)] += 1
        resolution_counts[stage] += 1
        return winner, loser

    for sim in range(n_sims):
        standings = {
            group: [teams[int(i)] for i in group_orders[gi, sim]]
            for gi, group in enumerate(group_letters)
        }
        best_groups = [group_letters[int(i)] for i in best_third_indices[:, sim]]
        resolved = bracket.resolve_round_of_32(standings, best_groups)
        outcomes = {}
        for pair in resolved.values():
            for team in pair:
                advancement[team]["round_of_32"] += 1
        for number in range(73, 105):
            if number <= 88:
                home, away = resolved[number]
            else:
                entrants = []
                for slot in bracket.matches[number].entrants:
                    winner, loser = outcomes[int(slot[1:])]
                    entrants.append(winner if slot[0] == "W" else loser)
                home, away = entrants
            outcomes[number] = play(number, home, away)
            winner, loser = outcomes[number]
            if number <= 88:
                advancement[winner]["round_of_16"] += 1
            elif number <= 96:
                advancement[winner]["quarterfinal"] += 1
            elif number <= 100:
                advancement[winner]["semifinal"] += 1
            elif number <= 102:
                advancement[winner]["final"] += 1
            elif number == 103:
                third_place[winner] += 1
                fourth_place[loser] += 1
            else:
                advancement[winner]["champion"] += 1

    return FullSimulationResult(
        n_sims=n_sims,
        seed=seed,
        teams=teams,
        groups=groups,
        position_probs={team: list(pos_counts[team] / n_sims) for team in teams},
        expected_points={team: points_sum[team] / n_sims for team in teams},
        best_third_probs={team: third_counts[team] / n_sims for team in teams},
        advancement_probs={
            team: {stage: advancement[team][stage] / n_sims for stage in STAGES}
            for team in teams
        },
        third_place_probs={team: third_place[team] / n_sims for team in teams},
        fourth_place_probs={team: fourth_place[team] / n_sims for team in teams},
        matchup_counts=dict(matchup_counts),
        winner_counts=dict(winner_counts),
        resolution_counts=resolution_counts,
    )


def check_full_invariants(result: FullSimulationResult, tol: float = 1e-9) -> list[str]:
    problems = []
    expected = {
        "round_of_32": 32,
        "round_of_16": 16,
        "quarterfinal": 8,
        "semifinal": 4,
        "final": 2,
        "champion": 1,
    }
    for stage, count in expected.items():
        total = sum(result.advancement_probs[team][stage] for team in result.teams)
        if abs(total - count) > tol:
            problems.append(f"{stage} probabilities sum to {total}, expected {count}")
    for team in result.teams:
        values = [result.advancement_probs[team][stage] for stage in STAGES]
        if any(right > left + tol for left, right in pairwise(values)):
            problems.append(f"{team}: advancement probabilities are not monotonic")
    for number, counter in result.matchup_counts.items():
        if sum(counter.values()) != result.n_sims:
            problems.append(f"M{number} matchup probabilities do not sum to one")
    return problems
