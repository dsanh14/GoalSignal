"""Tests for the human-context regenerator (synthetic fixtures + repo defaults)."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from goalsignal.tournament.bracket_2026 import MatchSlot, OfficialBracket
from goalsignal.tournament.human_adjustments import HumanAdjustmentsConfig
from goalsignal.tournament.human_context import (
    EXPERT_MAX_SHIFT_PCT,
    EXPERT_SOURCE_MODEL,
    FORM_DELTA_CAP,
    PriorityAdjustment,
    PriorityMatch,
    build_adjustment_blocks,
    build_expert_rows,
    build_recent_form_update,
    merge_adjustments_yaml,
    resolve_r16_pairings,
    update_human_context,
)
from goalsignal.tournament.knockout_results import KnockoutResult
from goalsignal.tournament.performance_tags import PerformanceTag

# --------------------------------------------------------------------------- #
# Synthetic fixtures (fictional teams only).
# --------------------------------------------------------------------------- #


def _slot(number, entrants):
    return MatchSlot(number, "round_of_16", entrants, "2026-07-04", "15:00", "Alpha")


def _r16_graph():
    return {
        89: _slot(89, ("W73", "W74")),
        90: _slot(90, ("W75", "W76")),
    }


def _result(number, team_a, team_b, winner, **kw):
    return KnockoutResult(
        match_number=number, round="round_of_32", team_a=team_a, team_b=team_b,
        score_a=kw.get("score_a"), score_b=kw.get("score_b"),
        aet=kw.get("aet", False), penalties=kw.get("penalties", False),
        winner=winner, notes=kw.get("notes", ""),
    )


def _tag(team, number, tag, points, reason="synthetic tag reason"):
    return PerformanceTag(
        team=team, match_number=number, tag=tag, points=points, reason=reason
    )


PRIORITY = {
    89: PriorityMatch(
        89, "Astoria", "Borduria", "Astoria favored.", 0.7,
        extras=(
            PriorityAdjustment(
                "Astoria", "venue", "altitude_boost", 5, "home altitude",
                confidence="high",
            ),
        ),
    ),
    90: PriorityMatch(
        90, "Cascadia", "Drachenland", "Near even.", 0.6,
        fallback_baseline_a=0.62,
    ),
}


# --------------------------------------------------------------------------- #
# Pairing resolution.
# --------------------------------------------------------------------------- #


def test_r16_pairings_update_after_confirmed_r32_results():
    results = {
        73: _result(73, "Astoria", "Xanadu", "Astoria", score_a=2, score_b=0),
        74: _result(74, "Borduria", "Yonderland", "Borduria",
                    aet=True, penalties=True, score_a=1, score_b=1),
    }
    pairings, warnings = resolve_r16_pairings(results, _r16_graph(), PRIORITY)
    assert pairings[89].team_a == "Astoria"
    assert pairings[89].team_b == "Borduria"  # penalty winner propagated
    assert pairings[89].provisional == ()
    # M90: no confirmed feeders -> priority fallback, flagged provisional.
    assert pairings[90].team_a == "Cascadia"
    assert pairings[90].provisional == ("Cascadia", "Drachenland")
    assert not any("M89" in w for w in warnings)


def test_confirmed_upset_takes_precedence_over_priority():
    results = {
        73: _result(73, "Astoria", "Xanadu", "Xanadu", score_a=0, score_b=1),
        74: _result(74, "Borduria", "Yonderland", "Borduria", score_a=1, score_b=0),
    }
    pairings, warnings = resolve_r16_pairings(results, _r16_graph(), PRIORITY)
    assert pairings[89].team_a == "Xanadu"
    assert any("Xanadu" in w and "precedence" in w for w in warnings)


def test_unresolved_pairing_without_fallback_is_skipped():
    pairings, warnings = resolve_r16_pairings({}, _r16_graph(), {})
    assert pairings == {}
    assert len(warnings) == 2


# --------------------------------------------------------------------------- #
# Adjustment + expert generation.
# --------------------------------------------------------------------------- #


def _pairings(results=None):
    results = results if results is not None else {
        73: _result(73, "Astoria", "Xanadu", "Astoria", score_a=2, score_b=0),
        74: _result(74, "Borduria", "Yonderland", "Borduria",
                    aet=True, penalties=True, score_a=1, score_b=1),
    }
    pairings, _ = resolve_r16_pairings(results, _r16_graph(), PRIORITY)
    return pairings


TAGS = [
    _tag("Astoria", 73, "dominant_win", 5, "won 2-0 comfortably"),
    _tag("Borduria", 74, "penalty_win", 3, "advanced on penalties"),
    _tag("Borduria", 74, "extra_time_fatigue", -2, "120 minutes"),
    _tag("Cascadia", 75, "late_comeback", 4, "late turnaround"),
]


def test_adjustment_blocks_carry_tags_extras_and_reasons():
    blocks = build_adjustment_blocks(_pairings(), TAGS, PRIORITY)
    m89 = blocks[89]
    assert m89["label"] == "Astoria vs Borduria"
    teams = [(e["team"], e["points"]) for e in m89["adjustments"]]
    assert ("Astoria", 5) in teams          # dominant_win tag
    assert ("Borduria", 3) in teams         # penalty_win tag
    assert ("Borduria", -2) in teams        # fatigue tag
    assert ("Astoria", 5) in teams          # venue extra
    assert all(e["reason"] for e in m89["adjustments"])
    tag_entry = next(e for e in m89["adjustments"] if "dominant_win" in e["reason"])
    assert tag_entry["category"] == "tournament_form"
    assert tag_entry["modifier"] == "dominant_win_boost"
    assert "[dominant_win M73]" in tag_entry["reason"]
    # Provisional pairing labels say so.
    assert "provisional" in blocks[90]["label"]


def test_expert_rows_are_bounded_and_transparent():
    baselines = {("Astoria", "Borduria"): 0.55, ("Borduria", "Astoria"): 0.45}
    frame, warnings = build_expert_rows(_pairings(), TAGS, baselines, PRIORITY)
    assert warnings == []
    m89 = frame[frame["match_id"] == "M89"].iloc[0]
    # Astoria +5 tag +5 extra = +10; Borduria +1 -> delta +9 -> 0.55 + 0.09.
    assert float(m89["team_a_advance_prob"]) == pytest.approx(0.64)
    assert float(m89["team_b_advance_prob"]) == pytest.approx(0.36)
    assert m89["source_model"] == EXPERT_SOURCE_MODEL
    assert "net context shift +9" in m89["reasoning"]
    m90 = frame[frame["match_id"] == "M90"].iloc[0]
    # Fallback baseline used and flagged; Cascadia +4 tag.
    assert float(m90["team_a_advance_prob"]) == pytest.approx(0.66)
    assert "priority-table prior" in m90["reasoning"]
    assert "Provisional" in m90["reasoning"]


def test_expert_shift_is_capped():
    tags = [
        _tag("Astoria", 73, "dominant_win", 10, "huge"),
        _tag("Astoria", 74, "battle_tested", 10, "huge"),
        _tag("Astoria", 75, "late_comeback", 10, "huge"),
    ]
    # Tag nudge caps at 6, extras add 5 -> 11; delta 11 within the 15 cap;
    # add an opposing stack to verify the explicit cap too.
    baselines = {("Astoria", "Borduria"): 0.5}
    frame, _ = build_expert_rows(_pairings(), tags, baselines, PRIORITY)
    m89 = frame[frame["match_id"] == "M89"].iloc[0]
    shift = (float(m89["team_a_advance_prob"]) - 0.5) * 100
    assert shift <= EXPERT_MAX_SHIFT_PCT + 1e-9


def test_recent_form_deltas_are_bounded_and_audited():
    base = pd.DataFrame({
        "team": ["Astoria", "Borduria"],
        "elo_adj_last5": [0.30, 0.10],
        "elo_adj_last10": [0.25, 0.05],
        "gf_adj": [0.4, 0.2],
        "ga_adj": [0.3, 0.4],
        "xg_diff": [0.20, 0.00],
    })
    heavy = [
        *TAGS,
        _tag("Astoria", 74, "battle_tested", 5),
        _tag("Astoria", 75, "finishing_boost", 5),
    ]
    updated, audit, warnings = build_recent_form_update(base, heavy)
    astoria = updated[updated["team"] == "Astoria"].iloc[0]
    # +5 +5 +5 = 15 raw, capped at FORM_DELTA_CAP -> +0.06.
    assert float(astoria["elo_adj_last5"]) == pytest.approx(0.30 + FORM_DELTA_CAP / 100)
    assert float(astoria["xg_diff"]) == pytest.approx(0.20 + FORM_DELTA_CAP / 100)
    # Untouched columns stay put.
    assert float(astoria["gf_adj"]) == 0.4
    borduria = updated[updated["team"] == "Borduria"].iloc[0]
    assert float(borduria["elo_adj_last5"]) == pytest.approx(0.11)  # +3 -2
    assert set(audit["team"]) == {"Astoria", "Borduria"}
    assert all(audit["reasons"].str.len() > 0)
    # Tagged teams without a base row are skipped, not invented.
    assert any("Cascadia" in w for w in warnings)


def test_merge_adjustments_yaml_preserves_other_matches(tmp_path):
    existing = tmp_path / "adj.yaml"
    existing.write_text(yaml.safe_dump({
        "global": {"max_total_adjustment_pct": 15,
                   "max_single_adjustment_pct": 10,
                   "min_probability": 0.05, "max_probability": 0.95},
        "matches": {
            97: {"label": "kept", "adjustments": [
                {"team": "Astoria", "category": "venue", "points": 2,
                 "reason": "kept entry"},
            ]},
            89: {"label": "replaced", "adjustments": []},
        },
    }), encoding="utf-8")
    blocks = build_adjustment_blocks(_pairings(), TAGS, PRIORITY)
    merged, added, replaced = merge_adjustments_yaml(existing, blocks)
    assert added == [90]
    assert replaced == [89]
    assert merged["matches"][97]["label"] == "kept"
    assert merged["matches"][89]["label"] == "Astoria vs Borduria"


# --------------------------------------------------------------------------- #
# End-to-end command behavior.
# --------------------------------------------------------------------------- #


def _write_inputs(tmp_path):
    results = tmp_path / "results.csv"
    results.write_text(
        "match_number,round,team_a,team_b,score_a,score_b,aet,penalties,winner,notes\n"
        "73,round_of_32,Astoria,Xanadu,2,0,false,false,Astoria,\n"
        "74,round_of_32,Borduria,Yonderland,1,1,true,true,Borduria,\"pens\"\n",
        encoding="utf-8",
    )
    tags = tmp_path / "tags.csv"
    tags.write_text(
        "team,match_number,tag,points,reason\n"
        "Astoria,73,dominant_win,5,\"won 2-0\"\n"
        "Borduria,74,penalty_win,3,\"pens win\"\n",
        encoding="utf-8",
    )
    matchups = tmp_path / "matchups.csv"
    matchups.write_text(
        "match_id,stage,team_a,team_b,historical_team_a_advances,"
        "historical_team_b_advances\n"
        "M89,knockout,Astoria,Borduria,0.55,0.45\n",
        encoding="utf-8",
    )
    form = tmp_path / "recent_form.csv"
    form.write_text(
        "team,elo_adj_last5,elo_adj_last10,gf_adj,ga_adj,xg_diff\n"
        "Astoria,0.30,0.25,0.4,0.3,0.20\n"
        "Borduria,0.10,0.05,0.2,0.4,0.00\n",
        encoding="utf-8",
    )
    expert = tmp_path / "expert.csv"
    expert.write_text(
        "match_id,team_a,team_b,source_model,team_a_win_prob,draw_prob,"
        "team_b_win_prob,team_a_advance_prob,team_b_advance_prob,confidence,"
        "reasoning\n"
        ",Astoria,Borduria,other-source,,,,0.5,0.5,0.5,\"kept row\"\n"
        f"M89,Astoria,Borduria,{EXPERT_SOURCE_MODEL},,,,0.9,0.1,0.9,\"stale\"\n",
        encoding="utf-8",
    )
    return {
        "results_path": results,
        "tags_path": tags,
        "matchups_path": matchups,
        "recent_form_path": form,
        "recent_form_base_path": tmp_path / "recent_form_base.csv",
        "recent_form_audit_path": tmp_path / "recent_form_audit.csv",
        "expert_path": expert,
        "adjustments_path": tmp_path / "adjustments.yaml",
        "bracket_matches": _r16_graph(),
        "priority": PRIORITY,
    }


def test_update_human_context_end_to_end(tmp_path):
    kw = _write_inputs(tmp_path)
    # Existing targets refuse to overwrite without force.
    with pytest.raises(FileExistsError, match="--force"):
        update_human_context(**kw)
    update = update_human_context(**kw, force=True)
    # The regenerated YAML is valid under the strict loader.
    config = HumanAdjustmentsConfig.load(kw["adjustments_path"])
    assert 89 in config.matches
    # M90 has no tag or priority entries in this fixture: no empty block.
    assert 90 not in config.matches
    assert all(
        abs(a.points) <= config.max_single_adjustment_pct
        for m in config.matches.values() for a in m.adjustments
    )
    # Expert file: prior generated row replaced, foreign row kept.
    expert = pd.read_csv(kw["expert_path"], dtype=str).fillna("")
    ours = expert[expert["source_model"] == EXPERT_SOURCE_MODEL]
    assert len(ours) == 2
    assert not (ours["team_a_advance_prob"] == "0.9").any()
    assert (expert["source_model"] == "other-source").sum() == 1
    # Recent form updated over a preserved base snapshot with an audit.
    base = pd.read_csv(kw["recent_form_base_path"])
    assert float(base[base["team"] == "Astoria"]["elo_adj_last5"].iloc[0]) == 0.30
    updated = pd.read_csv(kw["recent_form_path"])
    assert float(
        updated[updated["team"] == "Astoria"]["elo_adj_last5"].iloc[0]
    ) == pytest.approx(0.35)
    audit = pd.read_csv(kw["recent_form_audit_path"])
    assert {"team", "column", "base_value", "delta", "updated_value",
            "reasons"} <= set(audit.columns)
    assert update.changes


def test_update_human_context_is_idempotent(tmp_path):
    kw = _write_inputs(tmp_path)
    update_human_context(**kw, force=True)
    first_form = kw["recent_form_path"].read_text(encoding="utf-8")
    first_yaml = kw["adjustments_path"].read_text(encoding="utf-8")
    first_expert = kw["expert_path"].read_text(encoding="utf-8")
    update_human_context(**kw, force=True)
    # Deltas are re-derived from the base snapshot, never compounded.
    assert kw["recent_form_path"].read_text(encoding="utf-8") == first_form
    assert kw["adjustments_path"].read_text(encoding="utf-8") == first_yaml
    assert kw["expert_path"].read_text(encoding="utf-8") == first_expert


# --------------------------------------------------------------------------- #
# Repository defaults (real bracket graph + tracked manual files).
# --------------------------------------------------------------------------- #


def test_real_r16_pairings_from_tracked_results():
    """Confirmed R32 results must produce the live R16 bracket, including the
    penalty winners (Paraguay, Morocco, Egypt) replacing the modal picks."""
    from goalsignal.tournament.knockout_results import load_knockout_results

    results = load_knockout_results()
    bracket = OfficialBracket.load()
    pairings, _ = resolve_r16_pairings(results, bracket.matches)
    assert (pairings[89].team_a, pairings[89].team_b) == ("Paraguay", "France")
    assert (pairings[90].team_a, pairings[90].team_b) == ("Canada", "Morocco")
    assert (pairings[92].team_a, pairings[92].team_b) == ("Mexico", "England")
    assert (pairings[93].team_a, pairings[93].team_b) == ("Portugal", "Spain")
    assert (pairings[94].team_a, pairings[94].team_b) == (
        "United States", "Belgium"
    )
    assert (pairings[95].team_a, pairings[95].team_b) == ("Argentina", "Egypt")
    assert pairings[95].provisional == ()
    assert (pairings[96].team_a, pairings[96].team_b) == ("Switzerland", "Colombia")
    assert pairings[96].provisional == ()
