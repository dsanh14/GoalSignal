from __future__ import annotations

import pytest

from goalsignal.signals.base import AdvanceProbs, OutcomeProbs
from goalsignal.signals.match_context import (
    MatchContextParams,
    adjust_advance,
    adjust_outcome,
    load_match_context,
)
from goalsignal.signals.pipeline import MatchSpec, build_signals, load_manual_inputs


def _write(path, rows: str):
    path.write_text(rows)
    return path


HEADER = (
    "match_id,team_a,team_b,stage,available_at,kickoff_at,source,"
    "lineup_edge,availability_edge,goalkeeper_edge,fatigue_edge,"
    "match_quality_edge,tactical_edge,climate_edge,reason\n"
)


def test_loader_requires_pre_kickoff_timezone_aware_evidence(tmp_path):
    late = _write(
        tmp_path / "late.csv",
        HEADER + "M1,A,B,knockout,2026-07-11T20:00:00Z,2026-07-11T20:00:00Z,test,10,,,,,,,late\n",
    )
    with pytest.raises(ValueError, match="before kickoff"):
        load_match_context(late)

    naive = _write(
        tmp_path / "naive.csv",
        HEADER + "M1,A,B,knockout,2026-07-11T19:00:00,2026-07-11T20:00:00Z,test,10,,,,,,,naive\n",
    )
    with pytest.raises(ValueError, match="timezone"):
        load_match_context(naive)

    no_reason = _write(
        tmp_path / "no_reason.csv",
        HEADER + "M1,A,B,knockout,2026-07-11T19:00:00Z,2026-07-11T20:00:00Z,test,10,,,,,,,\n",
    )
    with pytest.raises(ValueError, match="reason is required"):
        load_match_context(no_reason)


def test_component_and_total_caps_are_applied(tmp_path):
    path = _write(
        tmp_path / "context.csv",
        HEADER + "M1,A,B,knockout,2026-07-11T19:00:00Z,"
        "2026-07-11T20:00:00Z,test,100,100,-100,,,,,evidence\n",
    )
    context = load_match_context(path)["M1"]
    params = MatchContextParams(component_cap=20, total_cap=30)
    assert context.raw_points(params) == pytest.approx(20.0)
    assert context.advantage_points(params) == pytest.approx(20.0)


def test_adjustments_are_anchored_bounded_and_preserve_draw():
    params = MatchContextParams(max_probability_shift=0.03)
    outcome = OutcomeProbs(0.60, 0.25, 0.15)
    moved = adjust_outcome(outcome, 200, params)
    assert moved.draw == pytest.approx(outcome.draw)
    assert moved.home_win == pytest.approx(0.63)

    advance = AdvanceProbs(0.70, 0.30)
    moved_advance = adjust_advance(advance, -200, params)
    assert moved_advance.team_a_advances == pytest.approx(0.67)


def test_pipeline_context_uses_base_and_flips_reverse_pair(tmp_path):
    _write(
        tmp_path / "match_context.csv",
        HEADER + ",A,B,knockout,2026-07-11T19:00:00Z,"
        "2026-07-11T20:00:00Z,test,20,10,,,,,,A stronger XI\n",
    )
    inputs = load_manual_inputs(tmp_path)
    forward = build_signals(
        MatchSpec("dynamic", "knockout", "A", "B", AdvanceProbs(0.60, 0.40)),
        inputs,
    )["match_context"]
    reverse = build_signals(
        MatchSpec("dynamic", "knockout", "B", "A", AdvanceProbs(0.40, 0.60)),
        inputs,
    )["match_context"]
    assert forward is not None and reverse is not None
    assert forward.team_a_advances > 0.60
    assert reverse.team_a_advances < 0.40
    assert forward.team_a_advances == pytest.approx(reverse.team_b_advances)


def test_no_context_evidence_produces_no_signal(tmp_path):
    _write(
        tmp_path / "match_context.csv",
        HEADER + "M1,A,B,knockout,2026-07-11T19:00:00Z,"
        "2026-07-11T20:00:00Z,test,,,,,,,,notes only\n",
    )
    inputs = load_manual_inputs(tmp_path)
    signals = build_signals(MatchSpec("M1", "knockout", "A", "B", AdvanceProbs(0.6, 0.4)), inputs)
    assert signals["match_context"] is None
