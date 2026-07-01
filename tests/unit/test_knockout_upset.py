"""Unit tests for the knockout "survive and advance" signal (synthetic teams)."""

from __future__ import annotations

import numpy as np
import pytest

from goalsignal.signals.base import AdvanceProbs, OutcomeProbs, advance_from_outcome
from goalsignal.signals.knockout_upset import (
    KnockoutUpsetParams,
    PenaltyProfile,
    PenaltyTable,
    TeamStyle,
    TeamStyleTable,
    knockout_upset_detail,
    knockout_upset_signal,
    load_penalties,
    load_team_styles,
    shootout_favorite_prob,
    shrunk_winrate,
    staged_favorite_advance,
)
from goalsignal.signals.meta_ensemble import MetaEnsemble, load_ensemble_config
from goalsignal.signals.pipeline import (
    ManualInputs,
    MatchSpec,
    build_signals,
    load_manual_inputs,
)

PARAMS = KnockoutUpsetParams()


# --------------------------------------------------------------------------- #
# Helpers to build small synthetic tables.
# --------------------------------------------------------------------------- #


def _styles(**by_team: dict) -> TeamStyleTable:
    return TeamStyleTable({t: TeamStyle(team=t, **vals) for t, vals in by_team.items()})


def _pens(**by_team: dict) -> PenaltyTable:
    return PenaltyTable({t: PenaltyProfile(team=t, **vals) for t, vals in by_team.items()})


def _compact_underdog() -> TeamStyleTable:
    """A possession-heavy favourite vs a compact low-block underdog."""
    return _styles(
        Fav={"possession_heavy": 90, "sterile_possession_risk": 80,
             "struggles_vs_low_block": 80, "chance_creation": 55},
        Und={"low_block_defense": 90, "defensive_compactness": 88,
             "transition_threat": 70, "set_piece_threat": 70, "aerial_threat": 72},
    )


# --------------------------------------------------------------------------- #
# Stage gating.
# --------------------------------------------------------------------------- #


def test_group_stage_returns_no_signal():
    """build_signals never emits knockout_upset for a group match."""
    inputs = _manual_inputs(include=True, styles=_compact_underdog())
    spec = MatchSpec("G1", "group", "Fav", "Und",
                     historical=OutcomeProbs(0.5, 0.25, 0.25))
    signals = build_signals(spec, inputs)
    assert signals.get("knockout_upset") is None


def test_signal_inactive_without_flag():
    """No opt-in => no knockout_upset signal even for a knockout match."""
    inputs = _manual_inputs(include=False, styles=_compact_underdog())
    spec = MatchSpec("K1", "knockout", "Fav", "Und",
                     historical=AdvanceProbs(0.65, 0.35))
    assert "knockout_upset" not in build_signals(spec, inputs)


def test_no_evidence_returns_none():
    """No style or penalty data for either side => the signal abstains."""
    out = knockout_upset_detail(
        "X", "Y", base_advance=AdvanceProbs(0.7, 0.3),
        styles=TeamStyleTable({}), penalties=PenaltyTable({}), params=PARAMS,
    )
    assert out.advance is None
    assert out.shift == 0.0
    assert knockout_upset_signal(
        "X", "Y", base_advance=AdvanceProbs(0.7, 0.3),
        styles=TeamStyleTable({}), penalties=PenaltyTable({}),
    ) is None


# --------------------------------------------------------------------------- #
# Survival mechanics.
# --------------------------------------------------------------------------- #


def test_staged_advance_lower_eg_helps_underdog():
    """Lower expected goals raises draw mass and lowers favourite advance."""
    high = staged_favorite_advance(3.2, 0.5, 0.5)
    low = staged_favorite_advance(1.6, 0.5, 0.5)
    assert high > low  # fewer goals => more draws => favourite advances less


def test_low_eg_high_draw_increases_underdog_shift():
    """For a real favourite, a low-event matchup shifts more to the underdog.

    Both style sets move *only* expected goals (not the skill gap): the underdog
    low block suppresses goals; the favourite's chance creation raises them. The
    lower-event tie routes more of the favourite's edge through the coin-flip
    path, so the underdog gains more.
    """
    base = AdvanceProbs(0.62, 0.38)
    low_event = _styles(Und={"low_block_defense": 95, "defensive_compactness": 95})
    high_event = _styles(Fav={"chance_creation": 95})
    low = knockout_upset_detail("Fav", "Und", base_advance=base,
                                styles=low_event, penalties=PenaltyTable({}), params=PARAMS)
    high = knockout_upset_detail("Fav", "Und", base_advance=base,
                                 styles=high_event, penalties=PenaltyTable({}), params=PARAMS)
    assert low.detail["expected_goals_total"] < high.detail["expected_goals_total"]
    assert low.detail["skill_adjusted"] == pytest.approx(low.detail["skill_base"])
    assert low.shift > high.shift


def test_possession_heavy_vs_low_block_increases_upset_risk():
    base = AdvanceProbs(0.70, 0.30)
    out = knockout_upset_detail("Fav", "Und", base_advance=base,
                                styles=_compact_underdog(), penalties=PenaltyTable({}),
                                params=PARAMS)
    assert out.favorite == "Fav" and out.underdog == "Und"
    assert out.shift > 0.0  # underdog gains
    assert out.advance.team_b_advances > base.team_b_advances
    assert "low_block_survival_path" in out.paths


def test_favorite_sterile_possession_downgrade():
    """A sterile, block-struggling favourite is downgraded vs a compact underdog."""
    base = AdvanceProbs(0.70, 0.30)
    sterile = knockout_upset_detail(
        "Fav", "Und", base_advance=base,
        styles=_styles(
            Fav={"sterile_possession_risk": 90, "struggles_vs_low_block": 85},
            Und={"low_block_defense": 90, "defensive_compactness": 85},
        ),
        penalties=PenaltyTable({}), params=PARAMS,
    )
    plain = knockout_upset_detail(
        "Fav", "Und", base_advance=base,
        styles=_styles(Und={"low_block_defense": 90}),
        penalties=PenaltyTable({}), params=PARAMS,
    )
    assert sterile.advance.team_a_advances < plain.advance.team_a_advances
    assert "favorite_sterile_possession_risk" in sterile.paths


def test_adjustment_is_capped_and_modest():
    """Even an extreme matchup is bounded by max_advance_shift."""
    base = AdvanceProbs(0.85, 0.15)
    out = knockout_upset_detail("Fav", "Und", base_advance=base,
                                styles=_compact_underdog(), penalties=PenaltyTable({}),
                                params=PARAMS)
    assert abs(out.shift) <= PARAMS.max_advance_shift + 1e-9


# --------------------------------------------------------------------------- #
# Penalty / shootout history.
# --------------------------------------------------------------------------- #


def test_shrunk_winrate_pulls_toward_half():
    assert shrunk_winrate(None, None) is None
    assert shrunk_winrate(0, 0) == pytest.approx(0.5)  # no record => prior 50/50
    # A perfect tiny record is pulled well below 1.0.
    assert shrunk_winrate(4, 0, prior_strength=6.0) == pytest.approx(0.7, abs=1e-9)
    # A single win barely moves off 0.5.
    assert abs(shrunk_winrate(1, 0) - 0.5) < 0.1
    # A large sample is shrunk only slightly.
    assert shrunk_winrate(20, 0) > 0.85


def test_shootout_prob_is_modest_and_capped():
    """Strong shootout indicators give a modest edge, never deterministic."""
    strong = PenaltyProfile(
        "Strong", keeper_penalty_strength=95, penalty_strength=95,
        world_cup_shootout_wins=6, world_cup_shootout_losses=0,
    )
    weak = PenaltyProfile("Weak", keeper_penalty_strength=20, penalty_strength=20,
                          world_cup_shootout_losses=4)
    p = shootout_favorite_prob(strong, weak, PARAMS)
    assert 0.5 < p <= 0.5 + PARAMS.shootout_cap + 1e-9
    assert p < 0.65  # modest, not 70/30
    # No data on either side => exactly a coin flip.
    assert shootout_favorite_prob(None, None, PARAMS) == 0.5


def test_croatia_style_history_is_a_modest_boost():
    """A strong shootout pedigree nudges, it does not dominate."""
    base = AdvanceProbs(0.5, 0.5)
    # Make the tie genuinely likely to reach penalties (compact, low event).
    styles = _styles(
        Fav={"low_block_defense": 80, "defensive_compactness": 80},
        Und={"low_block_defense": 88, "defensive_compactness": 90},
    )
    pens = _pens(Und={"keeper_penalty_strength": 80, "penalty_strength": 78,
                      "world_cup_shootout_wins": 4, "world_cup_shootout_losses": 0,
                      "tournament_experience": 85})
    out = knockout_upset_detail("Fav", "Und", base_advance=base,
                                styles=styles, penalties=pens, params=PARAMS)
    und_adv = out.advance.team_b_advances
    assert 0.50 < und_adv < 0.62  # 50/50 -> a few points, not a landslide


def test_penalty_strength_matters_more_when_draw_prob_high():
    base = AdvanceProbs(0.5, 0.5)
    und_pen = _pens(Und={"keeper_penalty_strength": 90, "penalty_strength": 88})
    low_event = _styles(Und={"low_block_defense": 95, "defensive_compactness": 95})
    high_event = _styles(Und={"transition_threat": 95, "attacking_directness": 95})

    def shift(styles, pens):
        return knockout_upset_detail("Fav", "Und", base_advance=base,
                                     styles=styles, penalties=pens, params=PARAMS).shift

    low_contrib = shift(low_event, und_pen) - shift(low_event, PenaltyTable({}))
    high_contrib = shift(high_event, und_pen) - shift(high_event, PenaltyTable({}))
    assert low_contrib > high_contrib > 0.0


# --------------------------------------------------------------------------- #
# Loaders are robust to missing files / fields.
# --------------------------------------------------------------------------- #


def test_missing_files_do_not_crash():
    styles = load_team_styles("does/not/exist.csv")
    pens = load_penalties("does/not/exist.csv")
    assert styles.teams == {} and pens.teams == {}
    # And the signal simply abstains.
    assert knockout_upset_signal("A", "B", base_advance=AdvanceProbs(0.6, 0.4),
                                 styles=styles, penalties=pens) is None


def test_loaders_read_example_csvs(tmp_path):
    style_csv = tmp_path / "team_styles.csv"
    style_csv.write_text(
        "team,low_block_defense,notes\nUnd,90,compact\nFav,30,open\n"
    )
    pen_csv = tmp_path / "penalties.csv"
    pen_csv.write_text(
        "team,keeper_penalty_strength,shootout_wins,shootout_losses\nUnd,85,3,1\n"
    )
    styles = load_team_styles(style_csv)
    pens = load_penalties(pen_csv)
    assert styles.get("Und").n("low_block_defense") == pytest.approx(0.8)
    assert pens.get("Und").keeper_penalty_strength == 85.0
    assert pens.has("Und") and not pens.has("Fav")


# --------------------------------------------------------------------------- #
# Pipeline + ensemble integration.
# --------------------------------------------------------------------------- #


def _manual_inputs(*, include: bool, styles=None, penalties=None) -> ManualInputs:
    """A minimal ManualInputs with empty match-keyed signals and given tables."""
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


def test_build_signals_dynamic_pair_without_match_id():
    """A dynamically generated pairing (synthetic id) still gets the signal."""
    inputs = _manual_inputs(include=True, styles=_compact_underdog())
    spec = MatchSpec("Fav|Und", "knockout", "Fav", "Und",
                     historical=AdvanceProbs(0.70, 0.30))
    signals = build_signals(spec, inputs)
    assert isinstance(signals["knockout_upset"], AdvanceProbs)
    # Underdog gains relative to the anchoring historical advance.
    assert signals["knockout_upset"].team_b_advances > 0.30


def test_match_id_precedence_over_team_pair():
    """A match-id-keyed market quote wins over a pair-keyed one (keying contract)."""
    from goalsignal.signals.keying import PairIndex

    idx: PairIndex = PairIndex.build([
        ("K1", None, None, "by_id"),       # match-id keyed only
        (None, "Fav", "Und", "by_pair"),   # team-pair keyed only
    ])
    # Same query resolves to the match-id payload when an id is supplied...
    assert idx.resolve("K1", "Fav", "Und") == ("by_id", 1)
    # ...and falls back to the team pair when no id is available.
    assert idx.resolve(None, "Fav", "Und") == ("by_pair", 1)
    # Reverse orientation resolves with a flip flag.
    assert idx.resolve(None, "Und", "Fav") == ("by_pair", -1)


def test_ensemble_renormalizes_when_knockout_upset_missing():
    """Absent knockout_upset weight is renormalized away (no crash, sums to 1)."""
    ensemble = MetaEnsemble(load_ensemble_config())
    signals = {
        "historical": AdvanceProbs(0.6, 0.4),
        "market": AdvanceProbs(0.55, 0.45),
        # knockout_upset deliberately absent
    }
    result = ensemble.blend_advance(signals, version="final_ensemble")
    assert "knockout_upset" in result.missing
    np.testing.assert_allclose(result.probs.as_array().sum(), 1.0)


def test_knockout_upset_moves_blended_advance_modestly():
    """With the flag on, the blended knockout advance shifts toward the underdog."""
    ensemble = MetaEnsemble(load_ensemble_config())
    spec = MatchSpec("K1", "knockout", "Fav", "Und",
                     historical=AdvanceProbs(0.70, 0.30))
    off = build_signals(spec, _manual_inputs(include=False, styles=_compact_underdog()))
    on = build_signals(spec, _manual_inputs(include=True, styles=_compact_underdog()))
    r_off = ensemble.blend_advance(off, version="final_ensemble")
    r_on = ensemble.blend_advance(on, version="final_ensemble")
    # Underdog advance is higher with the signal, but the move is small (weight 0.05).
    delta = r_on.probs.team_b_advances - r_off.probs.team_b_advances
    assert 0.0 < delta < 0.05


def test_config_is_valid_and_lists_knockout_upset():
    cfg = load_ensemble_config()
    assert "knockout_upset" in cfg.model_versions["final_ensemble"]
    assert "knockout_upset" in cfg.model_versions["knockout_survival"]
    # Params parse and are bounded.
    params = KnockoutUpsetParams.from_mapping(cfg.signal_params.get("knockout_upset"))
    assert 0.0 < params.shootout_cap <= 0.25
    assert 0.0 < params.max_advance_shift <= 0.25


def test_example_files_load_through_pipeline():
    """The bundled example CSVs load and produce an active signal end-to-end."""
    inputs = load_manual_inputs("data/manual", include_knockout_upset=True)
    spec = MatchSpec("K1", "knockout", "Spain", "Morocco",
                     historical=AdvanceProbs(0.70, 0.30))
    sig = build_signals(spec, inputs)["knockout_upset"]
    assert isinstance(sig, AdvanceProbs)
    assert sig.team_b_advances > 0.30  # Morocco gains a survival path


# --------------------------------------------------------------------------- #
# Full-simulator integration.
# --------------------------------------------------------------------------- #


def test_simulator_runs_with_knockout_upset_enabled():
    """The real full simulator runs when the ensemble adapter uses knockout_upset."""
    import pandas as pd

    from goalsignal.tournament.bracket_2026 import GROUPS, OfficialBracket
    from goalsignal.tournament.ensemble_adapter import EnsembleGoalAdapter
    from goalsignal.tournament.full_simulator import (
        check_full_invariants,
        simulate_full_tournament,
    )
    from goalsignal.tournament.model_adapter import RatingsGoalAdapter
    from goalsignal.tournament.simulator import GroupFixture

    class _StubGoalModel:
        def predict_expected_goals(self, frame: pd.DataFrame) -> np.ndarray:
            d = frame["elo_diff"].to_numpy(dtype=float) / 400.0
            return np.column_stack([np.exp(0.2 + 0.3 * d), np.exp(0.2 - 0.3 * d)])

        def score_matrix(self, lam_home: float, lam_away: float) -> np.ndarray:
            from scipy.stats import poisson

            h = poisson.pmf(np.arange(8), lam_home)
            a = poisson.pmf(np.arange(8), lam_away)
            m = np.outer(h, a)
            return m / m.sum()

    groups = {g: [f"{g}{i}" for i in range(1, 5)] for g in sorted(GROUPS)}
    fixtures = [
        GroupFixture(group=g, home=teams[i], away=teams[j],
                     fixture_id=f"{g}-{i}{j}", neutral=True, played=False)
        for g, teams in groups.items()
        for i in range(4) for j in range(i + 1, 4)
    ]
    teams = [t for ts in groups.values() for t in ts]
    ratings = {t: 1500.0 + 6.0 * (hash(t) % 21 - 10) for t in teams}
    base = RatingsGoalAdapter(ratings, _StubGoalModel())
    styles = _compact_underdog()
    pens = PenaltyTable({})

    from goalsignal.signals.base import davidson_outcome

    def blend_fn(home, away, neutral, knockout):
        outcome = davidson_outcome(ratings[home] - ratings[away])
        if not knockout:
            return outcome, {"home": home, "away": away}
        adv = advance_from_outcome(outcome)
        upset = knockout_upset_signal(home, away, base_advance=adv,
                                      styles=styles, penalties=pens, params=PARAMS)
        if upset is not None:
            blended = AdvanceProbs(
                0.9 * adv.team_a_advances + 0.1 * upset.team_a_advances,
                0.9 * adv.team_b_advances + 0.1 * upset.team_b_advances,
            )
            adv = blended
        return adv, {"home": home, "away": away}

    adapter = EnsembleGoalAdapter(base, blend_fn)
    result = simulate_full_tournament(
        groups, fixtures, adapter, OfficialBracket.load(), n_sims=120, seed=11
    )
    assert check_full_invariants(result) == []
    assert len(adapter.provenance) > 0
