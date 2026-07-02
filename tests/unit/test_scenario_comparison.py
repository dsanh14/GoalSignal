"""Tests for the scenario comparison report (synthetic fixtures)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from goalsignal.tournament.bracket_2026 import MatchSlot
from goalsignal.tournament.human_adjustments import (
    HumanAdjustmentsConfig,
    adjust_bracket,
    load_simulation_baseline,
    write_human_adjusted,
)
from goalsignal.tournament.scenario_comparison import (
    biggest_movers_frame,
    comparison_frame,
    flips_frame,
    load_human_scenario,
    load_modal_scenario,
    render_markdown,
    trace_downstream_effects,
    write_scenario_comparison,
)

# --------------------------------------------------------------------------- #
# Synthetic fixtures (fictional teams only): two semifinals feed a final.
# --------------------------------------------------------------------------- #


def _bracket_matches() -> dict[int, MatchSlot]:
    return {
        101: MatchSlot(101, "semifinal", ("1A", "2B"), "2026-07-14", "15:00", "Alpha City"),
        102: MatchSlot(102, "semifinal", ("1B", "2A"), "2026-07-15", "15:00", "Beta City"),
        104: MatchSlot(104, "final", ("W101", "W102"), "2026-07-19", "15:00", "Alpha City"),
    }


def _modal_entry(number, round_name, pair, winner, p):
    return {
        "match_number": number, "round": round_name, "modal_matchup": list(pair),
        "matchup_probability": 1.0, "modal_conditional_winner": winner,
        "conditional_win_probability": p,
    }


def _write_run(root, name, entries, meta=None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "wc2026_bracket.json").write_text(
        json.dumps({"label": "synthetic", "matches": entries}), encoding="utf-8"
    )
    (d / "wc2026_tournament_meta.json").write_text(
        json.dumps(meta or {"n_sims": 1000, "model_version": "test-v0"}),
        encoding="utf-8",
    )
    return d


def _write_primary_run(tmp_path):
    """The primary run: bracket + matchup CSVs so human-adjust can run on it."""
    entries = [
        _modal_entry(101, "semifinal", ("Astoria", "Borduria"), "Astoria", 0.55),
        _modal_entry(102, "semifinal", ("Cascadia", "Drachenland"), "Cascadia", 0.70),
        _modal_entry(104, "final", ("Astoria", "Cascadia"), "Astoria", 0.60),
    ]
    d = _write_run(tmp_path, "primary", entries)
    pd.DataFrame([
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
    ]).to_csv(d / "wc2026_semifinal_matchups.csv", index=False)
    pd.DataFrame([
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
    ]).to_csv(d / "wc2026_final_matchups.csv", index=False)
    return d


def _write_model_only_run(tmp_path):
    entries = [
        _modal_entry(101, "semifinal", ("Astoria", "Borduria"), "Astoria", 0.58),
        _modal_entry(102, "semifinal", ("Cascadia", "Drachenland"), "Cascadia", 0.72),
        _modal_entry(104, "final", ("Astoria", "Cascadia"), "Cascadia", 0.52),
    ]
    return _write_run(tmp_path, "model_only", entries)


def _write_ko_run(tmp_path):
    entries = [
        _modal_entry(101, "semifinal", ("Astoria", "Borduria"), "Astoria", 0.53),
        _modal_entry(102, "semifinal", ("Cascadia", "Drachenland"), "Cascadia", 0.66),
        _modal_entry(104, "final", ("Astoria", "Cascadia"), "Astoria", 0.51),
    ]
    return _write_run(tmp_path, "ko_survival", entries)


def _apply_human_overlay(tmp_path, primary, points=6):
    """Run the real human-adjust pipeline on the primary run (Borduria +points)."""
    cfg = {
        "global": {
            "max_total_adjustment_pct": 15,
            "max_single_adjustment_pct": 10,
            "min_probability": 0.05,
            "max_probability": 0.95,
        },
        "matches": {
            101: {"label": "Astoria vs Borduria", "adjustments": [{
                "team": "Borduria", "category": "style_matchup",
                "points": points, "confidence": "medium",
                "reason": "Borduria transition threat.",
            }]},
        },
    }
    path = tmp_path / "adjustments.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    config = HumanAdjustmentsConfig.load(path)
    result = adjust_bracket(
        load_simulation_baseline(primary), config, _bracket_matches()
    )
    write_human_adjusted(result, force=True)
    return result


def _snapshot(directory):
    return {
        p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()
    }


def _build_all(tmp_path, *, with_human=True, with_ko=True):
    primary = _write_primary_run(tmp_path)
    model_only = load_modal_scenario("model_only", _write_model_only_run(tmp_path))
    ko = load_modal_scenario(
        "knockout_survival", _write_ko_run(tmp_path) if with_ko else None
    )
    if with_human:
        _apply_human_overlay(tmp_path, primary)
    human = load_human_scenario(primary)
    comparison = comparison_frame(model_only, ko, human)
    movers = biggest_movers_frame(comparison, model_only, ko)
    reference = load_modal_scenario("reference", primary)
    traces = trace_downstream_effects(human, reference, _bracket_matches())
    return primary, model_only, ko, human, comparison, movers, traces


# --------------------------------------------------------------------------- #
# Report content.
# --------------------------------------------------------------------------- #


def test_writes_all_expected_files_and_content(tmp_path):
    primary, model_only, ko, human, comparison, movers, traces = _build_all(tmp_path)
    paths = write_scenario_comparison(
        comparison, movers, traces, model_only, ko, human, primary, force=True
    )
    assert set(paths) == {"md", "csv", "movers", "flips"}
    for path in paths.values():
        assert path.exists()
    frame = pd.read_csv(paths["csv"])
    assert list(frame["match_number"]) == [101, 102, 104]
    row = frame[frame["match_number"] == 101].iloc[0]
    assert row["model_only_winner"] == "Astoria"
    assert row["knockout_survival_winner"] == "Astoria"
    assert row["human_adjusted_winner"] == "Borduria"
    assert bool(row["flipped_by_opinion"])
    assert "Borduria transition threat." in row["reason"]
    assert row["confidence"] == "medium"


def test_flipped_match_appears_in_flips_csv_with_downstream(tmp_path):
    primary, model_only, ko, human, comparison, movers, traces = _build_all(tmp_path)
    paths = write_scenario_comparison(
        comparison, movers, traces, model_only, ko, human, primary, force=True
    )
    flips = pd.read_csv(paths["flips"])
    assert list(flips["match_number"]) == [101]
    assert flips.iloc[0]["human_adjusted_winner"] == "Borduria"
    assert "M104" in flips.iloc[0]["downstream_effects"]


def test_biggest_movers_columns_and_ordering(tmp_path):
    _, _, _, _, _, movers, _ = _build_all(tmp_path)
    assert {"comparison", "match_number", "stage", "subject",
            "from_prob", "to_prob", "delta"} <= set(movers.columns)
    assert not movers.empty
    # The +6 point overlay (0.55 -> 0.49) is the biggest single move.
    top = movers.iloc[0]
    assert top["comparison"] == "human_adjusted vs baseline"
    assert top["match_number"] == 101
    assert abs(movers["delta"]).is_monotonic_decreasing
    # Ensemble-vs-model rows exist for shared modal pairings.
    assert (movers["comparison"] == "knockout_survival vs model_only").any()


def test_downstream_tracing_catches_propagated_pairing_change(tmp_path):
    _, _, _, _, _, _, traces = _build_all(tmp_path)
    assert len(traces) == 1
    trace = traces[0]
    assert trace.match_number == 101
    assert (trace.flipped_from, trace.flipped_to) == ("Astoria", "Borduria")
    assert any(
        "M104" in effect and "was Astoria vs Cascadia" in effect
        for effect in trace.effects
    )
    # Astoria was the reference modal champion; Cascadia wins the scenario final.
    assert trace.champion_changed


def test_downstream_tracing_falls_back_to_modal_for_old_csv(tmp_path):
    """A pre-unadjusted-walk CSV still traces, flagged as modal-based."""
    primary, _, _, human, _, _, _ = _build_all(tmp_path)
    old_frame = human.frame.drop(
        columns=["unadjusted_team_1", "unadjusted_team_2", "unadjusted_winner"]
    )
    old_frame.to_csv(primary / "human_adjusted_bracket.csv", index=False)
    human = load_human_scenario(primary)
    traces = trace_downstream_effects(
        human, load_modal_scenario("reference", primary), _bracket_matches()
    )
    assert len(traces) == 1
    assert any("modal bracket" in effect for effect in traces[0].effects)


def test_markdown_table_rows_have_consistent_columns(tmp_path):
    """Reason text must not leak raw pipes into Markdown table rows."""
    _, model_only, ko, human, comparison, movers, traces = _build_all(tmp_path)
    md = render_markdown(comparison, movers, traces, model_only, ko, human)
    section = md.split("## Per-match comparison")[1].split("##")[0]
    table_lines = [
        line for line in section.splitlines() if line.startswith("|")
    ]
    pipe_counts = {line.count("|") for line in table_lines}
    assert len(pipe_counts) == 1


def test_scenario_language_present_in_markdown(tmp_path):
    _, model_only, ko, human, comparison, movers, traces = _build_all(tmp_path)
    md = render_markdown(comparison, movers, traces, model_only, ko, human)
    assert "scenario analysis, not calibrated forecasts" in md
    assert "opinion overlay" in md
    assert "scenario analysis layer" in md
    assert "human-adjusted scenario" in md.lower()
    assert "human-adjusted forecast" not in md.lower()
    assert "ledger" in md


# --------------------------------------------------------------------------- #
# Missing-artifact behavior.
# --------------------------------------------------------------------------- #


def test_report_runs_without_human_adjusted_csv(tmp_path):
    primary, model_only, ko, human, comparison, movers, traces = _build_all(
        tmp_path, with_human=False
    )
    assert not human.available
    assert traces == []
    assert len(comparison) == 3
    assert (comparison["human_adjusted_winner"] == "").all()
    assert not comparison["flipped_by_opinion"].any()
    md = render_markdown(comparison, movers, traces, model_only, ko, human)
    assert "unavailable" in md
    assert "goalsignal tournament human-adjust" in md
    paths = write_scenario_comparison(
        comparison, movers, traces, model_only, ko, human, primary, force=True
    )
    assert pd.read_csv(paths["csv"]).shape[0] == 3


def test_report_runs_without_knockout_survival_run(tmp_path):
    _, model_only, ko, human, comparison, movers, traces = _build_all(
        tmp_path, with_ko=False
    )
    assert not ko.available
    assert (comparison["knockout_survival_winner"] == "").all()
    # Human overlay still compared against model-only.
    assert comparison[comparison["match_number"] == 101].iloc[0][
        "human_adjusted_winner"
    ] == "Borduria"
    md = render_markdown(comparison, movers, traces, model_only, ko, human)
    assert "**unavailable**" in md
    assert not (movers["comparison"] == "knockout_survival vs model_only").any()


def test_original_simulation_artifacts_unmodified(tmp_path):
    primary = _write_primary_run(tmp_path)
    model_dir = _write_model_only_run(tmp_path)
    ko_dir = _write_ko_run(tmp_path)
    _apply_human_overlay(tmp_path, primary)
    before = {
        "model": _snapshot(model_dir),
        "ko": _snapshot(ko_dir),
        "primary": _snapshot(primary),
    }
    model_only = load_modal_scenario("model_only", model_dir)
    ko = load_modal_scenario("knockout_survival", ko_dir)
    human = load_human_scenario(primary)
    comparison = comparison_frame(model_only, ko, human)
    movers = biggest_movers_frame(comparison, model_only, ko)
    traces = trace_downstream_effects(
        human, load_modal_scenario("reference", primary), _bracket_matches()
    )
    out = tmp_path / "out"
    write_scenario_comparison(
        comparison, movers, traces, model_only, ko, human, out
    )
    assert _snapshot(model_dir) == before["model"]
    assert _snapshot(ko_dir) == before["ko"]
    assert _snapshot(primary) == before["primary"]


def test_refuses_overwrite_without_force(tmp_path):
    _, model_only, ko, human, comparison, movers, traces = _build_all(tmp_path)
    out = tmp_path / "out"
    paths = write_scenario_comparison(
        comparison, movers, traces, model_only, ko, human, out
    )
    with pytest.raises(FileExistsError, match="--force"):
        write_scenario_comparison(
            comparison, movers, traces, model_only, ko, human, out
        )
    assert write_scenario_comparison(
        comparison, movers, traces, model_only, ko, human, out, force=True
    ) == paths


def test_flips_frame_empty_when_no_flips(tmp_path):
    primary = _write_primary_run(tmp_path)
    _apply_human_overlay(tmp_path, primary, points=2)  # 0.55 -> 0.53: no flip
    human = load_human_scenario(primary)
    model_only = load_modal_scenario("model_only", _write_model_only_run(tmp_path))
    ko = load_modal_scenario("knockout_survival", None)
    comparison = comparison_frame(model_only, ko, human)
    assert flips_frame(comparison).empty
    traces = trace_downstream_effects(
        human, load_modal_scenario("reference", primary), _bracket_matches()
    )
    assert traces == []
