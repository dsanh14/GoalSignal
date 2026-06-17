"""Official bracket configuration and full-tournament regression tests."""

from __future__ import annotations

import copy
import math

import pytest

from goalsignal.tournament.bracket_2026 import GROUPS, THIRD_MATCHES, OfficialBracket
from goalsignal.tournament.full_simulator import (
    check_full_invariants,
    simulate_full_tournament,
)
from goalsignal.tournament.simulator import GroupFixture


class FixedModel:
    def expected_goals(self, home, away, neutral):
        return 1.25, 1.0

    def score_matrix(self, lam_home, lam_away):
        import numpy as np
        from scipy.stats import poisson

        goals = np.arange(9)
        matrix = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
        return matrix / matrix.sum()


def _groups_and_fixtures():
    groups = {group: [f"{group}{i}" for i in range(1, 5)] for group in sorted(GROUPS)}
    fixtures = []
    for group, teams in groups.items():
        for i in range(4):
            for j in range(i + 1, 4):
                fixtures.append(
                    GroupFixture(group=group, home=teams[i], away=teams[j])
                )
    return groups, fixtures


def test_official_bracket_sources_and_graph_validate():
    bracket = OfficialBracket.load()
    assert bracket.validate() == []
    assert len(bracket.matches) == 32
    assert len(bracket.source_manifest["sources"]) == 2
    assert all(len(source["sha256"]) == 64 for source in bracket.source_manifest["sources"])
    assert bracket.matches[104].entrants == ("W101", "W102")


def test_all_495_third_place_combinations_are_permutations():
    bracket = OfficialBracket.load()
    assert len(bracket.third_assignments) == math.comb(12, 8)
    for key, assignment in bracket.third_assignments.items():
        groups = key.split("-")
        assert len(groups) == len(set(groups)) == 8
        assert set(assignment) == THIRD_MATCHES
        assert sorted(value[1:] for value in assignment.values()) == sorted(groups)


def test_combination_lookup_is_order_independent_and_missing_fails():
    bracket = OfficialBracket.load()
    standings = {
        group: [f"{group}{i}" for i in range(1, 5)] for group in sorted(GROUPS)
    }
    groups = list("ABCDEFGHI")
    first = bracket.resolve_round_of_32(standings, groups[:8])
    second = bracket.resolve_round_of_32(standings, list(reversed(groups[:8])))
    assert first == second
    broken = copy.deepcopy(bracket)
    broken.third_assignments.pop("-".join(groups[:8]))
    with pytest.raises(KeyError, match="combination missing"):
        broken.resolve_round_of_32(standings, groups[:8])


def test_round_of_32_has_32_unique_qualified_entrants():
    bracket = OfficialBracket.load()
    standings = {
        group: [f"{group}{i}" for i in range(1, 5)] for group in sorted(GROUPS)
    }
    resolved = bracket.resolve_round_of_32(standings, list("ABCDEFGH"))
    entrants = [team for pair in resolved.values() for team in pair]
    assert len(resolved) == 16
    assert len(entrants) == len(set(entrants)) == 32
    assert {f"{group}1" for group in GROUPS}.issubset(entrants)
    assert {f"{group}2" for group in GROUPS}.issubset(entrants)


def test_seeded_full_simulation_is_reproducible_and_valid():
    groups, fixtures = _groups_and_fixtures()
    bracket = OfficialBracket.load()
    first = simulate_full_tournament(
        groups, fixtures, FixedModel(), bracket, n_sims=40, seed=7
    )
    second = simulate_full_tournament(
        groups, fixtures, FixedModel(), bracket, n_sims=40, seed=7
    )
    assert check_full_invariants(first) == []
    assert first.advancement_probs == second.advancement_probs
    assert first.matchup_counts == second.matchup_counts
    assert sum(p["champion"] for p in first.advancement_probs.values()) == pytest.approx(1)
    assert sum(p["final"] for p in first.advancement_probs.values()) == pytest.approx(2)


def test_target_trace_records_conditional_paths_and_strength():
    groups, fixtures = _groups_and_fixtures()
    bracket = OfficialBracket.load()
    target = "A1"
    ratings = {team: 1500.0 for teams in groups.values() for team in teams}
    strengths = {team: 0.0 for teams in groups.values() for team in teams}

    result = simulate_full_tournament(
        groups,
        fixtures,
        FixedModel(),
        bracket,
        n_sims=40,
        seed=9,
        target_team=target,
        opponent_elo=ratings,
        opponent_squad_strength=strengths,
        top_teams={"B1"},
    )
    trace = result.target_trace

    assert trace is not None
    assert sum(trace["finish_counts"].values()) == 40
    assert sum(trace["conditional_totals"].values()) == 40
    assert 0 <= trace["qualifying_third_probability"] <= 1
    assert 0 <= trace["top_team_before_quarterfinal_probability"] <= 1
    assert set(trace["expected_opponent_elo"]).issubset(trace["opponent_counts"])
    assert trace["conditional_advancement"] == simulate_full_tournament(
        groups,
        fixtures,
        FixedModel(),
        bracket,
        n_sims=40,
        seed=9,
        target_team=target,
        opponent_elo=ratings,
        opponent_squad_strength=strengths,
        top_teams={"B1"},
    ).target_trace["conditional_advancement"]
