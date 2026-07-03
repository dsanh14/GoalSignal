"""Tests for the winner-only human adjustment layer (synthetic fixtures)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from goalsignal.tournament.bracket_2026 import MatchSlot
from goalsignal.tournament.human_adjustments import (
    HumanAdjustmentsConfig,
    adjust_bracket,
    baseline_probability,
    bracket_frame,
    load_simulation_baseline,
    render_markdown,
    write_human_adjusted,
)
from goalsignal.tournament.knockout_results import KnockoutResult
from goalsignal.tournament.performance_tags import PerformanceTag

# --------------------------------------------------------------------------- #
# Synthetic fixtures (fictional teams only).
# --------------------------------------------------------------------------- #

TEAMS = ("Astoria", "Borduria", "Cascadia", "Drachenland")


def _bracket_matches() -> dict[int, MatchSlot]:
    """A tiny synthetic knockout graph: two semifinals feeding a final."""
    return {
        101: MatchSlot(101, "semifinal", ("1A", "2B"), "2026-07-14", "15:00", "Alpha City"),
        102: MatchSlot(102, "semifinal", ("1B", "2A"), "2026-07-15", "15:00", "Beta City"),
        104: MatchSlot(104, "final", ("W101", "W102"), "2026-07-19", "15:00", "Alpha City"),
    }


def _write_sim(tmp_path, *, final_rows=None):
    sim = tmp_path / "sim"
    sim.mkdir()
    semis = pd.DataFrame([
        {
            "match_number": 101, "round": "semifinal", "date": "2026-07-14",
            "time_et": "15:00", "host_city": "Alpha City",
            "slot_1_team": "Astoria", "slot_2_team": "Borduria",
            "matchup_probability": 1.0,
            "conditional_slot_1_win_probability": 0.55,
        },
        {
            "match_number": 102, "round": "semifinal", "date": "2026-07-15",
            "time_et": "15:00", "host_city": "Beta City",
            "slot_1_team": "Cascadia", "slot_2_team": "Drachenland",
            "matchup_probability": 1.0,
            "conditional_slot_1_win_probability": 0.70,
        },
    ])
    semis.to_csv(sim / "wc2026_semifinal_matchups.csv", index=False)
    if final_rows is None:
        final_rows = [
            {
                "match_number": 104, "round": "final", "date": "2026-07-19",
                "time_et": "15:00", "host_city": "Alpha City",
                "slot_1_team": "Astoria", "slot_2_team": "Cascadia",
                "matchup_probability": 0.5,
                "conditional_slot_1_win_probability": 0.60,
            },
            {
                "match_number": 104, "round": "final", "date": "2026-07-19",
                "time_et": "15:00", "host_city": "Alpha City",
                "slot_1_team": "Cascadia", "slot_2_team": "Borduria",
                "matchup_probability": 0.3,
                "conditional_slot_1_win_probability": 0.65,
            },
        ]
    pd.DataFrame(final_rows).to_csv(sim / "wc2026_final_matchups.csv", index=False)
    bracket = {
        "label": "synthetic",
        "matches": [
            {
                "match_number": 101, "round": "semifinal",
                "modal_matchup": ["Astoria", "Borduria"],
                "modal_conditional_winner": "Astoria",
            },
            {
                "match_number": 102, "round": "semifinal",
                "modal_matchup": ["Cascadia", "Drachenland"],
                "modal_conditional_winner": "Cascadia",
            },
            {
                "match_number": 104, "round": "final",
                "modal_matchup": ["Astoria", "Cascadia"],
                "modal_conditional_winner": "Astoria",
            },
        ],
    }
    (sim / "wc2026_bracket.json").write_text(json.dumps(bracket), encoding="utf-8")
    (sim / "wc2026_tournament_meta.json").write_text(
        json.dumps({"n_sims": 1000, "seed": 7, "model_version": "test-v0",
                    "dataset_version": "abc"}),
        encoding="utf-8",
    )
    return sim


def _write_config(tmp_path, matches, **global_overrides):
    cfg = {
        "global": {
            "max_total_adjustment_pct": 15,
            "max_single_adjustment_pct": 10,
            "min_probability": 0.05,
            "max_probability": 0.95,
            **global_overrides,
        },
        "matches": matches,
    }
    path = tmp_path / "adjustments.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _adjustment(team, points, category="venue", **extra):
    return {"team": team, "category": category, "points": points,
            "reason": "synthetic reason", **extra}


# --------------------------------------------------------------------------- #
# Config validation.
# --------------------------------------------------------------------------- #


def test_config_loads_and_resolves_modifier_as_category(tmp_path):
    path = _write_config(tmp_path, {
        101: {"label": "Astoria vs Borduria", "adjustments": [
            _adjustment("Astoria", 5, category="altitude_boost"),
            _adjustment("Borduria", -3, category="tournament_form",
                        modifier="late_collapse_penalty", confidence="High"),
        ]},
    })
    config = HumanAdjustmentsConfig.load(path)
    first, second = config.matches[101].adjustments
    assert first.category == "venue"
    assert first.modifier == "altitude_boost"
    assert second.confidence == "high"
    assert config.config_hash
    assert config.configured_teams() == {"Astoria", "Borduria"}


@pytest.mark.parametrize("bad", [
    {101: {"adjustments": [_adjustment("Astoria", 5, category="vibes")]}},
    {101: {"adjustments": [_adjustment("Astoria", 12)]}},          # over max_single
    {101: {"adjustments": [{"team": "Astoria", "category": "venue", "points": 5}]}},
    {101: {"adjustments": [_adjustment("", 5)]}},
    {101: {"adjustments": [_adjustment("Astoria", 5, confidence="certain")]}},
    {101: {"adjustments": [_adjustment("Astoria", 5, category="venue",
                                       modifier="manual_nudge")]}},
    {42: {"adjustments": [_adjustment("Astoria", 5)]}},            # not knockout
    {"nope": {"adjustments": [_adjustment("Astoria", 5)]}},
])
def test_config_rejects_invalid_entries(tmp_path, bad):
    path = _write_config(tmp_path, bad)
    with pytest.raises(ValueError, match="invalid human adjustments config"):
        HumanAdjustmentsConfig.load(path)


def test_config_rejects_bad_global_bounds(tmp_path):
    path = _write_config(tmp_path, {}, min_probability=0.9, max_probability=0.1)
    with pytest.raises(ValueError, match="min_probability"):
        HumanAdjustmentsConfig.load(path)


# --------------------------------------------------------------------------- #
# Baseline loading + lookup.
# --------------------------------------------------------------------------- #


def test_load_simulation_baseline_and_orientation(tmp_path):
    sim = _write_sim(tmp_path)
    baseline = load_simulation_baseline(sim)
    assert set(TEAMS) == baseline.teams
    p, source = baseline_probability(baseline, 101, "Astoria", "Borduria")
    assert (p, source) == (0.55, "simulated_matchup")
    p_rev, _ = baseline_probability(baseline, 101, "Borduria", "Astoria")
    assert p_rev == pytest.approx(0.45)
    p_missing, source = baseline_probability(baseline, 104, "Borduria", "Drachenland")
    assert (p_missing, source) == (0.5, "neutral_fallback")


def test_load_simulation_baseline_requires_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_simulation_baseline(tmp_path)


# --------------------------------------------------------------------------- #
# Bracket walk.
# --------------------------------------------------------------------------- #


def test_no_adjustments_reproduces_baseline_winners(tmp_path):
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches()
    )
    by_number = {m.match_number: m for m in result.matches}
    assert by_number[101].predicted_winner == "Astoria"
    assert by_number[102].predicted_winner == "Cascadia"
    # Final uses the propagated pairing's simulated probability (0.60).
    final = by_number[104]
    assert (final.team_1, final.team_2) == ("Astoria", "Cascadia")
    assert final.baseline_p_team_1 == 0.60
    assert final.adjusted_p_team_1 == 0.60
    assert not any(m.winner_changed for m in result.matches)
    assert result.champion == "Astoria"
    assert result.warnings == []


def test_adjustment_flips_winner_and_propagates(tmp_path):
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {
        101: {"label": "Astoria vs Borduria", "adjustments": [
            _adjustment("Borduria", 6, category="style_matchup"),
        ]},
    }))
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches()
    )
    by_number = {m.match_number: m for m in result.matches}
    semi = by_number[101]
    assert semi.adjusted_p_team_1 == pytest.approx(0.49)
    assert semi.predicted_winner == "Borduria"
    assert semi.winner_changed
    # The final now pairs Cascadia vs Borduria (reverse orientation in the CSV).
    final = by_number[104]
    assert {final.team_1, final.team_2} == {"Borduria", "Cascadia"}
    assert final.baseline_source == "simulated_matchup"
    p_cascadia = (
        final.adjusted_p_team_1
        if final.team_1 == "Cascadia"
        else final.adjusted_p_team_2
    )
    assert p_cascadia == pytest.approx(0.65)
    assert result.champion == "Cascadia"
    assert any("differs from the modal" in note for note in final.notes)


def test_caps_and_probability_clipping(tmp_path):
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {
        102: {"adjustments": [
            _adjustment("Cascadia", 10),
            _adjustment("Cascadia", 10, category="tournament_form"),
            _adjustment("Drachenland", -10, category="injuries"),
        ]},
    }))
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches()
    )
    match = {m.match_number: m for m in result.matches}[102]
    # Cascadia sum 20 -> capped to 15; Drachenland -10; delta 25 -> capped 15.
    assert match.net_points_team_1 == 15
    assert match.net_points_team_2 == -10
    assert match.applied_delta_pct == pytest.approx(15.0)
    # 0.70 + 0.15 = 0.85 < 0.95: no clipping; now force clipping via config.
    assert match.adjusted_p_team_1 == pytest.approx(0.85)

    tight = HumanAdjustmentsConfig.load(_write_config(
        tmp_path, {
            102: {"adjustments": [_adjustment("Cascadia", 10)]},
        }, max_probability=0.75,
    ))
    clipped = adjust_bracket(load_simulation_baseline(sim), tight, _bracket_matches())
    match = {m.match_number: m for m in clipped.matches}[102]
    assert match.adjusted_p_team_1 == pytest.approx(0.75)


def test_unknown_team_raises_and_wrong_pairing_skips(tmp_path):
    sim = _write_sim(tmp_path)
    baseline = load_simulation_baseline(sim)
    unknown = HumanAdjustmentsConfig.load(_write_config(tmp_path, {
        101: {"adjustments": [_adjustment("Atlantis", 5)]},
    }))
    with pytest.raises(ValueError, match="Atlantis"):
        adjust_bracket(baseline, unknown, _bracket_matches())

    wrong_match = HumanAdjustmentsConfig.load(_write_config(tmp_path, {
        101: {"label": "Cascadia vs Drachenland", "adjustments": [
            _adjustment("Cascadia", 5),
        ]},
    }))
    result = adjust_bracket(baseline, wrong_match, _bracket_matches())
    match = {m.match_number: m for m in result.matches}[101]
    assert match.applied == ()
    assert len(match.skipped) == 1
    assert match.adjusted_p_team_1 == match.baseline_p_team_1
    assert any("skipped" in w for w in result.warnings)
    assert any("label" in w for w in result.warnings)


def test_neutral_fallback_pairing_is_flagged(tmp_path):
    sim = _write_sim(tmp_path, final_rows=[{
        "match_number": 104, "round": "final", "date": "2026-07-19",
        "time_et": "15:00", "host_city": "Alpha City",
        "slot_1_team": "Borduria", "slot_2_team": "Drachenland",
        "matchup_probability": 0.1,
        "conditional_slot_1_win_probability": 0.5,
    }])
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches()
    )
    final = {m.match_number: m for m in result.matches}[104]
    assert final.baseline_source == "neutral_fallback"
    assert final.baseline_p_team_1 == 0.5
    assert any("never observed" in note for note in final.notes)
    # Deterministic tie-break: slot-1 team advances.
    assert final.predicted_winner == final.team_1


# --------------------------------------------------------------------------- #
# Confirmed results overlay.
# --------------------------------------------------------------------------- #


def _result(number, round_name, team_a, team_b, winner, *, aet=False,
            penalties=False, score_a=None, score_b=None, notes=""):
    return KnockoutResult(
        match_number=number, round=round_name, team_a=team_a, team_b=team_b,
        score_a=score_a, score_b=score_b, aet=aet, penalties=penalties,
        winner=winner, notes=notes,
    )


def test_confirmed_winner_overrides_modal_predicted_winner(tmp_path):
    """M101's modal winner is Astoria (p=0.55); the confirmed result says
    Borduria won, so Borduria must advance in both walks."""
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    confirmed = {
        101: _result(101, "semifinal", "Astoria", "Borduria", "Borduria",
                     score_a=0, score_b=1),
    }
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(),
        confirmed_results=confirmed,
    )
    by_number = {m.match_number: m for m in result.matches}
    semi = by_number[101]
    assert semi.confirmed_result
    assert semi.decided_by == "regulation"
    assert semi.baseline_source == "confirmed_result"
    assert semi.predicted_winner == "Borduria"
    assert semi.adjusted_p_team_1 == 0.0  # Astoria (slot 1) lost
    # A fact is not an opinion flip.
    assert not semi.winner_changed
    assert semi.unadjusted_winner == "Borduria"
    assert any("modal simulated winner" in note for note in semi.notes)
    # The final now pairs Borduria with the other semifinal winner.
    final = by_number[104]
    assert {final.team_1, final.team_2} == {"Borduria", "Cascadia"}
    assert {final.unadjusted_team_1, final.unadjusted_team_2} == (
        {"Borduria", "Cascadia"}
    )


def test_penalty_win_propagates_confirmed_winner(tmp_path):
    """A shootout result (level after ET) propagates the shootout winner."""
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    confirmed = {
        102: _result(102, "semifinal", "Cascadia", "Drachenland", "Drachenland",
                     aet=True, penalties=True, score_a=1, score_b=1,
                     notes="Drachenland won 4-2 on penalties"),
    }
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(),
        confirmed_results=confirmed,
    )
    by_number = {m.match_number: m for m in result.matches}
    semi = by_number[102]
    assert semi.confirmed_result
    assert semi.decided_by == "penalties"
    assert semi.predicted_winner == "Drachenland"
    assert any("penalties" in note for note in semi.notes)
    # Drachenland (not the modal Cascadia) feeds the final.
    final = by_number[104]
    assert "Drachenland" in (final.team_1, final.team_2)
    assert result.champion in {"Astoria", "Drachenland"}


def test_downstream_pairings_update_after_confirmed_results(tmp_path):
    """Both semifinals confirmed against the modal picks re-pair the final —
    the synthetic analog of R16 pairings updating after confirmed R32 results."""
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    confirmed = {
        101: _result(101, "semifinal", "Astoria", "Borduria", "Borduria",
                     aet=True, penalties=True, score_a=1, score_b=1),
        102: _result(102, "semifinal", "Cascadia", "Drachenland", "Drachenland",
                     aet=True, penalties=True, score_a=0, score_b=0),
    }
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(),
        confirmed_results=confirmed,
    )
    final = {m.match_number: m for m in result.matches}[104]
    # Modal final was Astoria vs Cascadia; confirmed winners replace both.
    assert {final.team_1, final.team_2} == {"Borduria", "Drachenland"}
    assert not final.confirmed_result  # the final itself is still predicted
    assert final.baseline_source == "neutral_fallback"


def test_confirmed_pairing_overrides_propagated_pairing(tmp_path):
    """A confirmed pairing that contradicts the modal entrants wins."""
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    confirmed = {
        101: _result(101, "semifinal", "Astoria", "Drachenland", "Drachenland",
                     score_a=0, score_b=2),
    }
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(),
        confirmed_results=confirmed,
    )
    semi = {m.match_number: m for m in result.matches}[101]
    assert (semi.team_1, semi.team_2) == ("Astoria", "Drachenland")
    assert any("modal simulated matchup" in n for n in semi.notes)


def test_confirmed_match_ignores_configured_adjustments(tmp_path):
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {
        101: {"adjustments": [_adjustment("Astoria", 9)]},
    }))
    confirmed = {
        101: _result(101, "semifinal", "Astoria", "Borduria", "Borduria",
                     score_a=0, score_b=1),
    }
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(),
        confirmed_results=confirmed,
    )
    semi = {m.match_number: m for m in result.matches}[101]
    assert semi.applied == ()
    assert len(semi.skipped) == 1
    assert semi.predicted_winner == "Borduria"
    assert any("already confirmed" in w for w in result.warnings)


def test_confirmed_overlay_validation(tmp_path):
    sim = _write_sim(tmp_path)
    baseline = load_simulation_baseline(sim)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    wrong_round = {101: _result(101, "final", "Astoria", "Borduria", "Borduria")}
    with pytest.raises(ValueError, match="round"):
        adjust_bracket(baseline, config, _bracket_matches(),
                       confirmed_results=wrong_round)
    unknown_team = {
        101: _result(101, "semifinal", "Astoria", "Atlantis", "Atlantis"),
    }
    with pytest.raises(ValueError, match="Atlantis"):
        adjust_bracket(baseline, config, _bracket_matches(),
                       confirmed_results=unknown_team)


# --------------------------------------------------------------------------- #
# Performance-tag nudges.
# --------------------------------------------------------------------------- #


def _tag(team, number, tag, points, reason="synthetic tag reason"):
    return PerformanceTag(
        team=team, match_number=number, tag=tag, points=points, reason=reason
    )


def test_tags_create_bounded_adjustments(tmp_path):
    """Late-comeback and fatigue tags earned earlier nudge later matches,
    with the net capped at the tag-nudge cap."""
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    tags = [
        _tag("Cascadia", 101, "late_comeback", 4),
        _tag("Cascadia", 101, "extra_time_fatigue", -2),
        _tag("Drachenland", 101, "extra_time_fatigue", -2),
    ]
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(), tags=tags
    )
    match = {m.match_number: m for m in result.matches}[102]
    assert match.tag_points_team_1 == pytest.approx(2.0)   # +4 - 2
    assert match.tag_points_team_2 == pytest.approx(-2.0)
    # 0.70 + (2 - (-2))/100 = 0.74
    assert match.adjusted_p_team_1 == pytest.approx(0.74)
    assert "late_comeback" in match.tag_reasons
    assert "synthetic tag reason" in match.tag_reasons

    # A stack of tags beyond the cap is clamped to ±6 by default.
    heavy = [
        _tag("Cascadia", 101, "dominant_win", 5),
        _tag("Cascadia", 101, "late_comeback", 5),
        _tag("Cascadia", 101, "battle_tested", 5),
    ]
    capped = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(), tags=heavy
    )
    match = {m.match_number: m for m in capped.matches}[102]
    assert match.tag_points_team_1 == pytest.approx(6.0)
    assert match.adjusted_p_team_1 == pytest.approx(0.76)
    assert any("capped" in note for note in match.notes)


def test_tags_do_not_apply_to_earlier_or_confirmed_matches(tmp_path):
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    tags = [_tag("Astoria", 102, "dominant_win", 5)]
    confirmed = {
        101: _result(101, "semifinal", "Astoria", "Borduria", "Astoria",
                     score_a=2, score_b=0),
    }
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(),
        confirmed_results=confirmed, tags=tags,
    )
    by_number = {m.match_number: m for m in result.matches}
    # M101 is confirmed: no tag nudge is recorded or applied.
    assert by_number[101].tag_points_team_1 == 0.0
    assert by_number[101].adjusted_p_team_1 == 1.0
    # M102: Astoria is not playing, so nothing changes there either.
    assert by_number[102].tag_points_team_1 == 0.0
    # M104 (Astoria vs Cascadia): the M102-earned tag applies (102 < 104).
    final = by_number[104]
    assert final.tag_points_team_1 == pytest.approx(5.0)
    assert final.adjusted_p_team_1 == pytest.approx(0.65)


def test_tags_and_config_points_combine_under_total_cap(tmp_path):
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {
        102: {"adjustments": [_adjustment("Cascadia", 10)]},
    }))
    tags = [
        _tag("Cascadia", 101, "dominant_win", 5),
        _tag("Cascadia", 101, "battle_tested", 5),
    ]
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(), tags=tags
    )
    match = {m.match_number: m for m in result.matches}[102]
    # Config +10, tag nudge capped at +6 -> combined capped at 15 total.
    assert match.net_points_team_1 == 10
    assert match.tag_points_team_1 == 6
    assert match.applied_delta_pct == pytest.approx(15.0)
    assert match.adjusted_p_team_1 == pytest.approx(0.85)


def test_unknown_tag_teams_warn_but_do_not_fail(tmp_path):
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(),
        tags=[_tag("Atlantis", 101, "dominant_win", 5)],
    )
    assert any("Atlantis" in w for w in result.warnings)
    assert all(m.tag_points_team_1 == 0 for m in result.matches)


def test_artifacts_include_overlay_columns(tmp_path):
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    confirmed = {
        101: _result(101, "semifinal", "Astoria", "Borduria", "Borduria",
                     aet=True, penalties=True, score_a=1, score_b=1,
                     notes="won on penalties"),
    }
    tags = [_tag("Drachenland", 101, "extra_time_fatigue", -3)]
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches(),
        confirmed_results=confirmed, tags=tags,
    )
    paths = write_human_adjusted(result, out_dir=tmp_path / "out")
    frame = pd.read_csv(paths["csv"])
    assert {"confirmed_result", "decided_by", "tag_points_team_1",
            "tag_points_team_2", "tag_reasons"} <= set(frame.columns)
    confirmed_row = frame[frame["match_number"] == 101].iloc[0]
    assert bool(confirmed_row["confirmed_result"])
    assert confirmed_row["decided_by"] == "penalties"
    md = paths["md"].read_text(encoding="utf-8")
    assert "Confirmed results (overlay)" in md
    assert "Performance-tag nudges (bounded)" in md
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["n_confirmed_results"] == 1
    assert meta["n_matches_tag_nudged"] == 1


# --------------------------------------------------------------------------- #
# Artifacts.
# --------------------------------------------------------------------------- #


def test_write_artifacts_and_force_semantics(tmp_path):
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {
        101: {"adjustments": [_adjustment("Borduria", 6, confidence="medium")]},
    }))
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches()
    )
    paths = write_human_adjusted(result)
    assert set(paths) == {"csv", "md", "meta"}
    frame = pd.read_csv(paths["csv"])
    assert list(frame["match_number"]) == [101, 102, 104]
    assert {"predicted_winner", "adjusted_p_team_1", "winner_changed",
            "baseline_source", "adjustment_reasons", "adjustment_confidences",
            "unadjusted_team_1", "unadjusted_team_2", "unadjusted_winner",
            "notes"} <= set(frame.columns)
    flipped = frame[frame["match_number"] == 101].iloc[0]
    assert "synthetic reason" in flipped["adjustment_reasons"]
    assert flipped["adjustment_confidences"] == "medium"
    # The unadjusted walk records the no-opinion final (Astoria vs Cascadia).
    final = frame[frame["match_number"] == 104].iloc[0]
    assert (final["team_1"], final["team_2"]) != (
        final["unadjusted_team_1"], final["unadjusted_team_2"]
    )
    assert final["unadjusted_winner"] == "Astoria"
    md = paths["md"].read_text(encoding="utf-8")
    assert "synthetic reason" in md
    assert "Predicted champion" in md
    assert "not a fitted model" in md
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    assert meta["n_matches_adjusted"] == 1
    assert meta["n_winners_changed"] == 1
    assert meta["config_hash"] == config.config_hash
    with pytest.raises(FileExistsError, match="--force"):
        write_human_adjusted(result)
    assert write_human_adjusted(result, force=True) == paths


def test_bracket_frame_and_markdown_without_adjustments(tmp_path):
    sim = _write_sim(tmp_path)
    config = HumanAdjustmentsConfig.load(_write_config(tmp_path, {}))
    result = adjust_bracket(
        load_simulation_baseline(sim), config, _bracket_matches()
    )
    frame = bracket_frame(result)
    assert (frame["n_adjustments_applied"] == 0).all()
    md = render_markdown(result)
    assert "## Adjustment detail" not in md
    assert "## Warnings" not in md
