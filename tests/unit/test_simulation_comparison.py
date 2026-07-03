"""Tests for the simulation-comparison reporting layer (synthetic fixtures)."""

from __future__ import annotations

import json

import pandas as pd

from goalsignal.evaluation.simulation_comparison import (
    biggest_movers,
    classify_meta,
    discover_sim_runs,
    load_sim_run,
    matchup_diagnostics,
    render_markdown,
    team_comparison,
    write_comparison_artifacts,
)
from goalsignal.signals.base import AdvanceProbs
from goalsignal.signals.knockout_upset import (
    PenaltyTable,
    TeamStyle,
    TeamStyleTable,
)
from goalsignal.signals.meta_ensemble import MetaEnsemble, load_ensemble_config
from goalsignal.signals.pipeline import ManualInputs, MatchSpec

# --------------------------------------------------------------------------- #
# Synthetic simulation artifacts.
# --------------------------------------------------------------------------- #


def _write_run(root, name, *, source, version=None, upset=False, champ):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    teams = ["Alpha", "Beta", "Gamma", "Delta"]
    rows = []
    for i, t in enumerate(teams):
        c = champ.get(t, 0.1)
        rows.append({
            "group": "A",
            "team": t,
            "p_round_of_32": 1.0,
            "p_round_of_16": 0.6,
            "p_quarterfinal": 0.4,
            "p_semifinal": 0.3 - 0.02 * i,
            "p_final": 0.2 - 0.02 * i,
            "p_champion": c,
        })
    pd.DataFrame(rows).to_csv(d / "wc2026_team_advancement.csv", index=False)
    meta = {
        "n_sims": 1000,
        "prediction_source": source,
        "ensemble_version": version,
        "include_knockout_upset": upset,
        "ensemble_provenance": {
            "flagged_matchups": [["Alpha", "Beta", 0.21]],
            "missing_signal_counts": {"market": 5, "expert": 5},
        } if source == "ensemble" else None,
    }
    (d / "wc2026_tournament_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def _three_runs(root):
    return {
        "baseline": _write_run(
            root, "base", source="historical",
            champ={"Alpha": 0.4, "Beta": 0.3, "Gamma": 0.2, "Delta": 0.1},
        ),
        "final_ensemble": _write_run(
            root, "fe", source="ensemble", version="final_ensemble",
            champ={"Alpha": 0.35, "Beta": 0.30, "Gamma": 0.22, "Delta": 0.13},
        ),
        "knockout_survival": _write_run(
            root, "ks", source="ensemble", version="knockout_survival", upset=True,
            champ={"Alpha": 0.30, "Beta": 0.30, "Gamma": 0.25, "Delta": 0.15},
        ),
    }


# --------------------------------------------------------------------------- #
# Discovery + classification.
# --------------------------------------------------------------------------- #


def test_classify_meta():
    assert classify_meta({"prediction_source": "historical"}) == "baseline"
    assert classify_meta({
        "prediction_source": "ensemble", "ensemble_version": "final_ensemble",
        "include_knockout_upset": False,
    }) == "final_ensemble"
    assert classify_meta({
        "prediction_source": "ensemble", "ensemble_version": "knockout_survival",
        "include_knockout_upset": True,
    }) == "knockout_survival"
    assert classify_meta({"prediction_source": "ensemble",
                          "ensemble_version": "market_only"}) is None


def test_discover_sim_runs(tmp_path):
    _three_runs(tmp_path)
    found = discover_sim_runs(tmp_path)
    assert set(found) == {"baseline", "final_ensemble", "knockout_survival"}


# --------------------------------------------------------------------------- #
# Team comparison + movers.
# --------------------------------------------------------------------------- #


def test_team_comparison_and_movers(tmp_path):
    paths = _three_runs(tmp_path)
    runs = {lbl: load_sim_run(lbl, p) for lbl, p in paths.items()}
    comp = team_comparison(runs)
    assert not comp.empty
    # Delta columns exist for the three comparisons across headline stages.
    assert "delta__knockout_survival_vs_final_ensemble__p_champion" in comp.columns
    assert "delta__final_ensemble_vs_baseline__p_champion" in comp.columns
    movers = biggest_movers(comp, top=10)
    assert set(movers.columns) >= {
        "team", "stage", "comparison", "from_prob", "to_prob", "delta", "abs_delta"
    }
    # Sorted by absolute delta, descending.
    assert movers["abs_delta"].is_monotonic_decreasing
    # Alpha lost the most champion share between baseline and knockout_survival.
    top = movers.iloc[0]
    assert top["abs_delta"] > 0


def test_missing_run_does_not_crash(tmp_path):
    paths = _three_runs(tmp_path)
    del paths["knockout_survival"]  # only two runs present
    runs = {
        "baseline": load_sim_run("baseline", paths["baseline"]),
        "final_ensemble": load_sim_run("final_ensemble", paths["final_ensemble"]),
        "knockout_survival": load_sim_run("knockout_survival", None),  # missing
    }
    assert runs["knockout_survival"].available is False
    comp = team_comparison(runs)
    assert not comp.empty
    # Only baseline/final deltas exist; knockout_survival comparisons are absent.
    assert "delta__final_ensemble_vs_baseline__p_champion" in comp.columns
    assert "delta__knockout_survival_vs_baseline__p_champion" not in comp.columns
    movers = biggest_movers(comp)
    md = render_markdown(runs, comp, movers, pd.DataFrame(), diagnostics_illustrative=True)
    assert "missing" in md.lower()


def test_no_runs_present_is_empty(tmp_path):
    runs = {lbl: load_sim_run(lbl, None) for lbl in
            ("baseline", "final_ensemble", "knockout_survival")}
    comp = team_comparison(runs)
    assert comp.empty
    movers = biggest_movers(comp)
    assert movers.empty


# --------------------------------------------------------------------------- #
# Matchup diagnostics.
# --------------------------------------------------------------------------- #


def _inputs(*, include, styles=None, penalties=None) -> ManualInputs:
    from goalsignal.signals.keying import PairIndex
    from goalsignal.signals.recent_form import RecentFormTable
    from goalsignal.signals.squad_strength import SquadStrengthTable

    cfg = load_ensemble_config()
    return ManualInputs(
        config=cfg, market={}, squad=SquadStrengthTable(teams={}),
        form=RecentFormTable(teams={}), venue={}, expert={}, load_errors={},
        market_index=PairIndex.build([]), expert_index=PairIndex.build([]),
        venue_index=PairIndex.build([]),
        styles=styles or TeamStyleTable({}),
        penalties=penalties or PenaltyTable({}),
        include_knockout_upset=include,
    )


def _compact_styles() -> TeamStyleTable:
    return TeamStyleTable({
        "Fav": TeamStyle("Fav", possession_heavy=90, sterile_possession_risk=85,
                         struggles_vs_low_block=80),
        "Und": TeamStyle("Und", low_block_defense=90, defensive_compactness=88,
                         transition_threat=70),
    })


def test_matchup_diagnostics_rows_and_tags():
    styles = _compact_styles()
    specs = [MatchSpec("K1", "knockout", "Fav", "Und",
                       historical=AdvanceProbs(0.70, 0.30)),
             MatchSpec("G1", "group", "Fav", "Und")]  # group row is ignored
    diag = matchup_diagnostics(
        specs,
        _inputs(include=False, styles=styles),
        _inputs(include=True, styles=styles),
        MetaEnsemble(load_ensemble_config()),
    )
    assert len(diag) == 1  # only the knockout match
    row = diag.iloc[0]
    required = {
        "team_a", "team_b", "stage", "baseline_team_a_advances",
        "final_ensemble_team_a_advances", "knockout_survival_team_a_advances",
        "delta_from_final", "knockout_upset_internal_shift", "net_move_from_upset",
        "draw_prob", "expected_goals_total", "penalty_path_contribution",
        "style_tags", "penalty_tags", "provenance_tags",
    }
    assert required <= set(diag.columns)
    # Provenance tags are populated for a compact-underdog matchup.
    assert "low_block_survival_path" in row["provenance_tags"]
    assert row["style_tags"] != ""
    # The survival version moves the underdog (team B) up vs the favourite.
    assert row["knockout_survival_team_a_advances"] < row["baseline_team_a_advances"]
    # The net move attributable to the upset signal is non-zero and toward B.
    assert row["net_move_from_upset"] < 0.0


def test_penalty_path_contribution_isolated():
    """Penalty contribution is the part of the shift from penalty data."""
    styles = TeamStyleTable({
        "Fav": TeamStyle("Fav", low_block_defense=70),
        "Und": TeamStyle("Und", low_block_defense=92, defensive_compactness=92),
    })
    from goalsignal.signals.knockout_upset import PenaltyProfile

    pens = PenaltyTable({"Und": PenaltyProfile("Und", keeper_penalty_strength=90,
                                               penalty_strength=88)})
    specs = [MatchSpec("K1", "knockout", "Fav", "Und",
                       historical=AdvanceProbs(0.5, 0.5))]
    diag = matchup_diagnostics(
        specs,
        _inputs(include=False, styles=styles, penalties=pens),
        _inputs(include=True, styles=styles, penalties=pens),
        MetaEnsemble(load_ensemble_config()),
    )
    row = diag.iloc[0]
    assert abs(row["penalty_path_contribution"]) > 0.0
    assert "penalty_path_boost" in row["provenance_tags"]


# --------------------------------------------------------------------------- #
# Artifact writing + report content.
# --------------------------------------------------------------------------- #


def test_write_artifacts_and_report_language(tmp_path):
    paths = _three_runs(tmp_path)
    runs = {lbl: load_sim_run(lbl, p) for lbl, p in paths.items()}
    comp = team_comparison(runs)
    movers = biggest_movers(comp)
    diag = matchup_diagnostics(
        [MatchSpec("K1", "knockout", "Fav", "Und", historical=AdvanceProbs(0.7, 0.3))],
        _inputs(include=False, styles=_compact_styles()),
        _inputs(include=True, styles=_compact_styles()),
        MetaEnsemble(load_ensemble_config()),
    )
    out = write_comparison_artifacts(runs, comp, movers, diag, out_dir=tmp_path / "art")
    for key in ("comparison_csv", "movers_csv", "explanations_csv", "report_md"):
        assert out[key].exists()
    # The CSVs have the right shape.
    assert "abs_delta" in pd.read_csv(out["movers_csv"]).columns
    assert "provenance_tags" in pd.read_csv(out["explanations_csv"]).columns
    md = out["report_md"].read_text(encoding="utf-8")
    # Honest, non-overclaiming language is present.
    for phrase in [
        "Experimental", "not fitted", "no chronological knockout backtest",
        "does **not** demonstrate an accuracy improvement",
        "shrunk toward 50/50", "never a deterministic rule",
        "Production-grade",
    ]:
        assert phrase in md
