"""Monte Carlo tournament simulation.

`simulate_groups` is the reference implementation: exact per-simulation
standings using the full tiebreaker procedure from rules.py. Score sampling is
vectorized (one categorical draw of n_sims scores per remaining fixture), the
standings loop is pure Python and deliberately transparent.

`simulate_groups_fast` vectorizes the standings computation with integer sort
keys and falls back to the exact procedure only for simulations containing an
unresolved (points, GD, GF) tie, so it is exact-by-construction wherever ties
need head-to-head or lots. Agreement between the two is tested.

Monte Carlo standard error for an estimated probability p̂ over N simulations
is sqrt(p̂ (1 - p̂) / N), reported alongside results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from goalsignal.tournament.rules import (
    TeamRecord,
    group_standings,
    rank_third_placed,
)


@dataclass
class GroupFixture:
    group: str
    home: str
    away: str
    neutral: bool = True
    played: bool = False
    home_goals: int | None = None
    away_goals: int | None = None


@dataclass
class SimulationResult:
    n_sims: int
    seed: int
    teams: list[str]
    position_probs: dict[str, list[float]]  # team -> [P(1st), P(2nd), P(3rd), P(4th)]
    expected_points: dict[str, float]
    advance_probs: dict[str, float]  # P(reach round of 32)
    best_third_probs: dict[str, float]
    groups: dict[str, list[str]]

    def mc_standard_error(self, p: float) -> float:
        return float(np.sqrt(p * (1.0 - p) / self.n_sims))


def _presample_scores(fixtures, model, n_sims: int, rng: np.random.Generator):
    """For each unplayed fixture, draw n_sims (home, away) scores at once."""
    sampled = {}
    for i, f in enumerate(fixtures):
        if f.played:
            continue
        lam_h, lam_a = model.expected_goals(f.home, f.away, f.neutral)
        matrix = model.score_matrix(lam_h, lam_a)
        flat = rng.choice(matrix.size, size=n_sims, p=matrix.ravel() / matrix.sum())
        sampled[i] = (flat // matrix.shape[1], flat % matrix.shape[1])
    return sampled


def simulate_groups(
    groups: dict[str, list[str]],
    fixtures: list[GroupFixture],
    model,
    n_sims: int = 10_000,
    seed: int = 20260612,
    n_best_thirds: int = 8,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    sampled = _presample_scores(fixtures, model, n_sims, rng)

    teams = [t for g in groups.values() for t in g]
    pos_counts = {t: np.zeros(4) for t in teams}
    points_sum = dict.fromkeys(teams, 0.0)
    advance_counts = dict.fromkeys(teams, 0)
    third_counts = dict.fromkeys(teams, 0)

    fixture_idx_by_group: dict[str, list[int]] = {g: [] for g in groups}
    for i, f in enumerate(fixtures):
        fixture_idx_by_group[f.group].append(i)

    for s in range(n_sims):
        thirds: list[TeamRecord] = []
        for gname, gteams in groups.items():
            results = []
            for i in fixture_idx_by_group[gname]:
                f = fixtures[i]
                if f.played:
                    results.append((f.home, f.away, f.home_goals, f.away_goals))
                else:
                    hs, as_ = sampled[i]
                    results.append((f.home, f.away, int(hs[s]), int(as_[s])))
            standing = group_standings(gteams, results, rng)
            for pos, team in enumerate(standing.ranking):
                pos_counts[team][pos] += 1
                points_sum[team] += standing.records[team].points
                if pos < 2:
                    advance_counts[team] += 1
                if pos == 2:
                    thirds.append(standing.records[team])
        for team in rank_third_placed(thirds, rng)[:n_best_thirds]:
            advance_counts[team] += 1
            third_counts[team] += 1

    return SimulationResult(
        n_sims=n_sims,
        seed=seed,
        teams=teams,
        position_probs={t: list(pos_counts[t] / n_sims) for t in teams},
        expected_points={t: points_sum[t] / n_sims for t in teams},
        advance_probs={t: advance_counts[t] / n_sims for t in teams},
        best_third_probs={t: third_counts[t] / n_sims for t in teams},
        groups=groups,
    )


def simulate_groups_fast(
    groups: dict[str, list[str]],
    fixtures: list[GroupFixture],
    model,
    n_sims: int = 100_000,
    seed: int = 20260612,
    n_best_thirds: int = 8,
) -> SimulationResult:
    """Vectorized standings with exact fallback for tied simulations."""
    rng = np.random.default_rng(seed)
    sampled = _presample_scores(fixtures, model, n_sims, rng)

    teams = [t for g in groups.values() for t in g]
    pos_counts = {t: np.zeros(4) for t in teams}
    points_sum = dict.fromkeys(teams, 0.0)
    advance_counts = {t: np.zeros(n_sims, dtype=bool) for t in teams}
    # Third-place record per group per sim: (points, gd, gf, team_index)
    thirds_key = np.zeros((len(groups), n_sims, 3), dtype=np.int64)
    thirds_team = np.zeros((len(groups), n_sims), dtype=np.int64)
    team_index = {t: i for i, t in enumerate(teams)}

    fixture_idx_by_group: dict[str, list[int]] = {g: [] for g in groups}
    for i, f in enumerate(fixtures):
        fixture_idx_by_group[f.group].append(i)

    for g_i, (gname, gteams) in enumerate(groups.items()):
        nt = len(gteams)
        t_idx = {t: k for k, t in enumerate(gteams)}
        pts = np.zeros((n_sims, nt), dtype=np.int64)
        gf = np.zeros((n_sims, nt), dtype=np.int64)
        ga = np.zeros((n_sims, nt), dtype=np.int64)
        for i in fixture_idx_by_group[gname]:
            f = fixtures[i]
            hi, ai = t_idx[f.home], t_idx[f.away]
            if f.played:
                hs = np.full(n_sims, f.home_goals)
                as_ = np.full(n_sims, f.away_goals)
            else:
                hs, as_ = sampled[i]
            gf[:, hi] += hs
            ga[:, hi] += as_
            gf[:, ai] += as_
            ga[:, ai] += hs
            pts[:, hi] += np.where(hs > as_, 3, np.where(hs == as_, 1, 0))
            pts[:, ai] += np.where(as_ > hs, 3, np.where(hs == as_, 1, 0))
        gd = gf - ga
        # Integer sort key: points dominate, then GD, then GF (all bounded).
        key = ((pts * 512) + (gd + 256)) * 512 + gf
        order = np.argsort(-key, axis=1, kind="stable")
        sorted_key = np.take_along_axis(key, order, axis=1)
        # Sims with any adjacent exact tie need the full tiebreak procedure.
        tied_sims = np.flatnonzero((np.diff(sorted_key, axis=1) == 0).any(axis=1))
        for s in tied_sims:
            results = []
            for i in fixture_idx_by_group[gname]:
                f = fixtures[i]
                if f.played:
                    results.append((f.home, f.away, f.home_goals, f.away_goals))
                else:
                    hs, as_ = sampled[i]
                    results.append((f.home, f.away, int(hs[s]), int(as_[s])))
            standing = group_standings(gteams, results, rng)
            order[s] = [t_idx[t] for t in standing.ranking]

        for pos in range(nt):
            occupant = order[:, pos]
            for k, team in enumerate(gteams):
                mask = occupant == k
                pos_counts[team][pos] += int(mask.sum())
                if pos < 2:
                    advance_counts[team] |= mask
                if pos == 2:
                    thirds_key[g_i, mask, 0] = pts[mask, k]
                    thirds_key[g_i, mask, 1] = gd[mask, k]
                    thirds_key[g_i, mask, 2] = gf[mask, k]
                    thirds_team[g_i, mask] = team_index[team]
        for k, team in enumerate(gteams):
            points_sum[team] += float(pts[:, k].sum())

    # Rank third-placed teams across groups per sim (points, gd, gf, jitter).
    third_counts = dict.fromkeys(teams, 0)
    jitter = rng.random((len(groups), n_sims))
    rank_key = (
        ((thirds_key[:, :, 0] * 512) + (thirds_key[:, :, 1] + 256)) * 512
        + thirds_key[:, :, 2]
    ).astype(np.float64) + jitter
    third_order = np.argsort(-rank_key, axis=0)[:n_best_thirds]  # (8, n_sims)
    for g_rank in range(n_best_thirds):
        winners = thirds_team[third_order[g_rank], np.arange(n_sims)]
        for ti in np.unique(winners):
            team = teams[int(ti)]
            mask = winners == ti
            third_counts[team] += int(mask.sum())
            advance_counts[team] |= mask

    return SimulationResult(
        n_sims=n_sims,
        seed=seed,
        teams=teams,
        position_probs={t: list(pos_counts[t] / n_sims) for t in teams},
        expected_points={t: points_sum[t] / n_sims for t in teams},
        advance_probs={t: float(advance_counts[t].mean()) for t in teams},
        best_third_probs={t: third_counts[t] / n_sims for t in teams},
        groups=groups,
    )


def check_invariants(result: SimulationResult, tol: float = 1e-9) -> list[str]:
    """Return a list of violated invariants (empty = all good)."""
    problems = []
    for gname, gteams in result.groups.items():
        for pos in range(4):
            total = sum(result.position_probs[t][pos] for t in gteams)
            if abs(total - 1.0) > tol:
                problems.append(f"group {gname} position {pos + 1} sums to {total}")
    n_advancers = 2 * len(result.groups) + 8
    total_adv = sum(result.advance_probs.values())
    if abs(total_adv - n_advancers) > 1e-6:
        problems.append(f"advance probabilities sum to {total_adv}, expected {n_advancers}")
    for t in result.teams:
        p_top2 = result.position_probs[t][0] + result.position_probs[t][1]
        if result.advance_probs[t] < p_top2 - tol:
            problems.append(f"{t}: P(advance) < P(top two)")
    return problems
