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
