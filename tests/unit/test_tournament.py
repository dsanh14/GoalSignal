"""Unit tests for tournament rules, knockout resolution, and simulators."""

from __future__ import annotations

import numpy as np
import pytest

from goalsignal.tournament.knockout import resolve_knockout
from goalsignal.tournament.rules import (
    build_records,
    group_standings,
    rank_third_placed,
)
from goalsignal.tournament.simulator import (
    GroupFixture,
    check_invariants,
    simulate_groups,
    simulate_groups_fast,
    validate_completed_overlay,
)

TEAMS = ["Atlantis", "Freedonia", "Ruritania", "Sylvania"]


def test_points_then_gd_then_gf():
    results = [
        ("Atlantis", "Freedonia", 3, 0),
        ("Ruritania", "Sylvania", 1, 0),
        ("Atlantis", "Ruritania", 2, 2),
        ("Freedonia", "Sylvania", 0, 0),
        ("Atlantis", "Sylvania", 1, 0),
        ("Freedonia", "Ruritania", 0, 1),
    ]
    standing = group_standings(TEAMS, results, np.random.default_rng(0))
    # Atlantis 7pts; Ruritania 7pts but worse GD (+4 vs +2)... check explicitly.
    rec = standing.records
    assert rec["Atlantis"].points == 7 and rec["Ruritania"].points == 7
    assert rec["Atlantis"].goal_difference > rec["Ruritania"].goal_difference
    assert standing.ranking[:2] == ["Atlantis", "Ruritania"]


def test_head_to_head_breaks_full_tie():
    # Atlantis and Freedonia both finish 6 pts, GD +1, GF 4; Atlantis won
    # their head-to-head meeting and must rank above.
    results = [
        ("Atlantis", "Freedonia", 2, 1),
        ("Atlantis", "Ruritania", 2, 1),
        ("Sylvania", "Atlantis", 1, 0),
        ("Freedonia", "Ruritania", 2, 1),
        ("Freedonia", "Sylvania", 1, 0),
        ("Ruritania", "Sylvania", 1, 1),
    ]
    standing = group_standings(TEAMS, results, np.random.default_rng(0))
    a, f = standing.records["Atlantis"], standing.records["Freedonia"]
    assert (a.points, a.goal_difference, a.goals_for) == (
        f.points, f.goal_difference, f.goals_for,
    )
    assert standing.ranking.index("Atlantis") < standing.ranking.index("Freedonia")
    assert any(t.startswith("head_to_head") for t in standing.tiebreaks)


def test_unresolvable_tie_uses_lots_and_is_seed_dependent():
    results = [  # all draws: everything identical
        ("Atlantis", "Freedonia", 0, 0),
        ("Ruritania", "Sylvania", 0, 0),
        ("Atlantis", "Ruritania", 0, 0),
        ("Freedonia", "Sylvania", 0, 0),
        ("Atlantis", "Sylvania", 0, 0),
        ("Freedonia", "Ruritania", 0, 0),
    ]
    s1 = group_standings(TEAMS, results, np.random.default_rng(1))
    assert any(t.startswith("lots") for t in s1.tiebreaks)
    rankings = {
        tuple(group_standings(TEAMS, results, np.random.default_rng(s)).ranking)
        for s in range(20)
    }
    assert len(rankings) > 1  # lots genuinely random across seeds


def test_three_way_tie_head_to_head_subtable():
    # Sylvania loses all three; the other three beat Sylvania and draw among
    # themselves -> 5 points each, broken only inside the H2H subtable (all
    # drawn there too, so lots), Sylvania last.
    results = [
        ("Atlantis", "Freedonia", 1, 1),
        ("Atlantis", "Ruritania", 0, 0),
        ("Freedonia", "Ruritania", 2, 2),
        ("Atlantis", "Sylvania", 2, 0),
        ("Freedonia", "Sylvania", 2, 0),
        ("Ruritania", "Sylvania", 2, 0),
    ]
    standing = group_standings(TEAMS, results, np.random.default_rng(0))
    assert standing.ranking[3] == "Sylvania"


def test_rank_third_placed():
    recs_a = build_records(["A3", "X"], [("A3", "X", 0, 0)])  # A3: draw -> 1 pt
    recs_b = build_records(["B3", "Y"], [("B3", "Y", 2, 0)])  # B3: win -> 3 pts
    ranked = rank_third_placed(
        [recs_a["A3"], recs_b["B3"]], np.random.default_rng(0)
    )
    assert ranked[0] == "B3"


class FixedModel:
    """Deterministic toy model: home always slightly stronger."""

    def __init__(self, lam_home=1.4, lam_away=1.0):
        self.lams = (lam_home, lam_away)

    def expected_goals(self, home, away, neutral):
        return self.lams

    def score_matrix(self, lh, la):
        from scipy.stats import poisson

        g = np.arange(9)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        return m / m.sum()


def _twelve_groups():
    groups = {f"G{i:02d}": [f"T{i}{c}" for c in "abcd"] for i in range(1, 13)}
    fixtures = []
    for g, ts in groups.items():
        for i in range(4):
            for j in range(i + 1, 4):
                fixtures.append(GroupFixture(group=g, home=ts[i], away=ts[j]))
    return groups, fixtures


def test_simulators_agree_and_satisfy_invariants():
    groups, fixtures = _twelve_groups()
    model = FixedModel()
    ref = simulate_groups(groups, fixtures, model, n_sims=2000, seed=11)
    fast = simulate_groups_fast(groups, fixtures, model, n_sims=2000, seed=11)
    assert check_invariants(ref) == []
    assert check_invariants(fast) == []
    # Same seed, same presampling -> headline probabilities agree closely
    # (exact tie resolution differs only in RNG consumption order for lots).
    for t in ref.teams:
        assert ref.advance_probs[t] == pytest.approx(fast.advance_probs[t], abs=0.05)
        assert ref.expected_points[t] == pytest.approx(fast.expected_points[t], abs=0.15)


def test_played_fixtures_are_respected():
    groups, fixtures = _twelve_groups()
    # Fix one match as a huge played win for T1d (weakest seed position).
    fixtures[0] = GroupFixture(
        group="G01", home="T1a", away="T1b", played=True, home_goals=0, away_goals=9
    )
    res = simulate_groups(groups, fixtures, FixedModel(), n_sims=500, seed=3)
    assert res.expected_points["T1b"] > res.expected_points["T1a"]


def test_completed_overlay_is_exactly_once():
    _groups, fixtures = _twelve_groups()
    fixtures[0].fixture_id = "done"
    fixtures[0].played = True
    fixtures[0].home_goals = 2
    fixtures[0].away_goals = 0
    active = {"done": {"regulation_home_goals": 2, "regulation_away_goals": 0}}
    validate_completed_overlay(fixtures, active)
    fixtures[1].fixture_id = "done"
    fixtures[1].played = True
    fixtures[1].home_goals = 2
    fixtures[1].away_goals = 0
    with pytest.raises(ValueError, match="overlay mismatch"):
        validate_completed_overlay(fixtures, active)


def test_knockout_resolution_stages():
    rng = np.random.default_rng(0)
    model = FixedModel()
    m = model.score_matrix(1.4, 1.0)
    saw_reg, saw_et, saw_pens = False, False, False
    for _ in range(500):
        out = resolve_knockout("H", "A", m, 1.4, 1.0, rng)
        assert out.winner in ("H", "A")
        if not out.extra_time:
            assert out.reg_home_goals != out.reg_away_goals
            saw_reg = True
        elif not out.shootout:
            saw_et = True
        else:
            assert out.reg_home_goals == out.reg_away_goals
            saw_pens = True
    assert saw_reg and saw_et and saw_pens


def test_shootout_baseline_is_even():
    rng = np.random.default_rng(42)
    model = FixedModel(1.0, 1.0)
    m = model.score_matrix(1.0, 1.0)
    wins = 0
    n_shootouts = 0
    for _ in range(4000):
        out = resolve_knockout("H", "A", m, 1.0, 1.0, rng)
        if out.shootout:
            n_shootouts += 1
            wins += out.winner == "H"
    assert n_shootouts > 100
    assert wins / n_shootouts == pytest.approx(0.5, abs=0.1)
