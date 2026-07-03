"""Knockout-only "survive and advance" signal.

Group-stage forecasting asks *who is better over 90 minutes*. Knockout
forecasting must also ask *who can survive and advance* — a worse team over a
single match can still keep the game low-event, ride a compact low block to a
draw, and win the coin-flip of extra time and penalties. This module produces a
controlled, uncertainty-aware **advance** adjustment for knockout ties only.

It is deliberately *anchored*: starting from a base advance estimate (the
historical model, the market, or whichever advance probability the ensemble
already has), it re-derives the advance probability through an explicit
regulation / extra-time / penalty staged model and reports only the *difference*
caused by style and shootout evidence. With no style or penalty evidence the
adjustment is exactly zero, so it never randomly boosts underdogs. Its blend
weight in ``config/ensemble.yaml`` is small (0.05), so even a large per-match
shift moves the final ensemble only modestly.

Conceptual advance model (for the favourite ``F`` versus the underdog ``U``)::

    P(F advances) = P(F wins in regulation)
                  + P(regulation draw) * [ P(F wins ET)
                                          + P(ET draw) * P(F wins shootout) ]

Lower expected goals (a compact, low-block matchup) raise the regulation- and
ET-draw mass, routing more of the favourite's edge through the near-coin-flip
penalty path — which is exactly where a survival-minded underdog gains.

Inputs are *file-first* and every field is optional:

* ``data/manual/team_styles.csv`` — 0-100 style indicators per team.
* ``data/manual/penalties.csv`` — penalty/shootout indicators and (shrunk)
  shootout history per team.

Nothing here is fitted to match results, so the signal carries no leakage risk.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skellam

from goalsignal.signals.base import AdvanceProbs

# --------------------------------------------------------------------------- #
# Tunable parameters (all overridable from config/ensemble.yaml).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KnockoutUpsetParams:
    """Coefficients for the knockout survival adjustment.

    All values are modest and configurable. ``base_expected_goals`` is a typical
    two-team international knockout total; the style coefficients nudge it and
    the regulation goal-margin within tight, explainable bounds.
    """

    base_expected_goals: float = 2.6
    min_expected_goals: float = 1.4
    max_expected_goals: float = 3.6
    # Convert a base advance gap into a dimensionless skill half-difference.
    edge_per_advance: float = 1.6
    # Expected-goals (draw-mass) modifiers — larger => more draws => more pens.
    low_block_eg: float = 0.7        # underdog low block suppresses goals
    compactness_eg: float = 0.5      # underdog compactness suppresses goals
    sterile_eg: float = 0.4          # favourite sterile possession suppresses goals
    transition_eg: float = 0.3       # underdog transition threat adds goals
    creation_eg: float = 0.3         # favourite chance creation adds goals
    # Regulation skill modifiers (in skill units) — each shrinks the favourite's edge.
    struggles_block_edge: float = 0.45   # favourite struggles vs an actual low block
    sterile_edge: float = 0.30           # favourite sterile possession
    transition_edge: float = 0.30        # underdog transition threat
    set_piece_edge: float = 0.30         # underdog set-piece / aerial threat
    # Shootout head-to-head: deviation from 0.5 per unit rating gap, hard-capped.
    shootout_beta: float = 0.10
    shootout_cap: float = 0.12
    # Penalty-history shrinkage strength (pseudo-matches pulling toward 50/50).
    shootout_prior_strength: float = 6.0
    current_pen_weight: float = 0.7   # current keeper/taker weighted over history
    history_pen_weight: float = 0.3
    # Final per-match advance shift is hard-capped for safety.
    max_advance_shift: float = 0.15
    # Threshold (in normalized [-1, 1] units) above which a style path is tagged.
    path_tag_threshold: float = 0.2

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> KnockoutUpsetParams:
        """Build from a config sub-mapping, ignoring unknown keys."""
        if not mapping:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {k: float(v) for k, v in mapping.items() if k in known}
        return cls(**kwargs)


# --------------------------------------------------------------------------- #
# Style profiles.
# --------------------------------------------------------------------------- #

_STYLE_COLUMNS: tuple[str, ...] = (
    "possession_heavy",
    "low_block_defense",
    "transition_threat",
    "set_piece_threat",
    "pressing_intensity",
    "chance_creation",
    "sterile_possession_risk",
    "struggles_vs_low_block",
    "defensive_compactness",
    "attacking_directness",
    "aerial_threat",
)


def _norm_0_100(value: float | None) -> float:
    """Map a 0-100 indicator to [-1, 1] centred at 50; missing => 0 (neutral)."""
    if value is None:
        return 0.0
    return float(np.clip((value - 50.0) / 50.0, -1.0, 1.0))


@dataclass(frozen=True)
class TeamStyle:
    """Optional 0-100 style indicators for one team (any field may be ``None``)."""

    team: str
    possession_heavy: float | None = None
    low_block_defense: float | None = None
    transition_threat: float | None = None
    set_piece_threat: float | None = None
    pressing_intensity: float | None = None
    chance_creation: float | None = None
    sterile_possession_risk: float | None = None
    struggles_vs_low_block: float | None = None
    defensive_compactness: float | None = None
    attacking_directness: float | None = None
    aerial_threat: float | None = None

    def has_any(self) -> bool:
        return any(getattr(self, c) is not None for c in _STYLE_COLUMNS)

    def n(self, column: str) -> float:
        """Normalized [-1, 1] value of one style column (0 if absent)."""
        return _norm_0_100(getattr(self, column))


_EMPTY_STYLE = TeamStyle(team="")


@dataclass
class TeamStyleTable:
    """Loaded per-team style table."""

    teams: dict[str, TeamStyle]

    def get(self, team: str) -> TeamStyle:
        """Return the team's style, or an all-neutral profile if absent."""
        return self.teams.get(team, _EMPTY_STYLE)

    def has(self, team: str) -> bool:
        s = self.teams.get(team)
        return s is not None and s.has_any()


def load_team_styles(path: str | Path, *, require: bool = False) -> TeamStyleTable:
    """Load a team-style CSV. A missing file yields an empty table."""
    p = Path(path)
    if not p.exists():
        if require:
            raise FileNotFoundError(f"team styles file not found: {p}")
        return TeamStyleTable(teams={})
    df = pd.read_csv(p)
    if "team" not in df.columns:
        raise ValueError("team styles CSV must have a 'team' column")
    teams: dict[str, TeamStyle] = {}
    for _, row in df.iterrows():
        team = str(row["team"]).strip()
        if not team:
            continue
        kwargs: dict[str, float] = {}
        for col in _STYLE_COLUMNS:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != "":
                kwargs[col] = float(row[col])
        teams[team] = TeamStyle(team=team, **kwargs)
    return TeamStyleTable(teams=teams)


# --------------------------------------------------------------------------- #
# Penalty / shootout profiles (shrunk toward 50/50).
# --------------------------------------------------------------------------- #

_PEN_CURRENT: tuple[tuple[str, float], ...] = (
    ("penalty_strength", 1.0),
    ("keeper_penalty_strength", 1.0),
    ("penalty_taker_depth", 0.7),
    ("tournament_experience", 0.4),
    ("manager_continuity", 0.2),
)


@dataclass(frozen=True)
class PenaltyProfile:
    """Optional penalty/shootout indicators for one team (any may be ``None``).

    Current ratings (``*_strength``, ``*_depth``, experience, continuity) are
    0-100. Shootout records are raw win/loss counts and are shrunk toward 50/50
    before use because the samples are tiny.
    """

    team: str
    penalty_strength: float | None = None
    keeper_penalty_strength: float | None = None
    penalty_taker_depth: float | None = None
    shootout_wins: float | None = None
    shootout_losses: float | None = None
    world_cup_shootout_wins: float | None = None
    world_cup_shootout_losses: float | None = None
    continental_shootout_wins: float | None = None
    continental_shootout_losses: float | None = None
    tournament_experience: float | None = None
    manager_continuity: float | None = None

    def has_any(self) -> bool:
        return any(
            getattr(self, f.name) is not None
            for f in fields(self)
            if f.name != "team"
        )


def shrunk_winrate(
    wins: float | None, losses: float | None, *, prior_strength: float = 6.0
) -> float | None:
    """Beta-shrunk shootout win rate, pulled toward 0.5 for small samples.

    ``(wins + 0.5 * k) / (wins + losses + k)`` with ``k = prior_strength``: a
    team with 4-0 and ``k=6`` reads as 0.70, not 1.0, and a 1-0 record barely
    moves off 0.5. Returns ``None`` if neither count is present.
    """
    if wins is None and losses is None:
        return None
    w = float(wins or 0.0)
    losses_n = float(losses or 0.0)
    n = w + losses_n
    k = float(prior_strength)
    return (w + 0.5 * k) / (n + k)


@dataclass
class PenaltyTable:
    """Loaded per-team penalty/shootout table."""

    teams: dict[str, PenaltyProfile]

    def get(self, team: str) -> PenaltyProfile | None:
        return self.teams.get(team)

    def has(self, team: str) -> bool:
        p = self.teams.get(team)
        return p is not None and p.has_any()


def load_penalties(path: str | Path, *, require: bool = False) -> PenaltyTable:
    """Load a penalties CSV. A missing file yields an empty table."""
    p = Path(path)
    if not p.exists():
        if require:
            raise FileNotFoundError(f"penalties file not found: {p}")
        return PenaltyTable(teams={})
    df = pd.read_csv(p)
    if "team" not in df.columns:
        raise ValueError("penalties CSV must have a 'team' column")
    known = {f.name for f in fields(PenaltyProfile)} - {"team"}
    teams: dict[str, PenaltyProfile] = {}
    for _, row in df.iterrows():
        team = str(row["team"]).strip()
        if not team:
            continue
        kwargs: dict[str, float] = {}
        for col in known:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != "":
                kwargs[col] = float(row[col])
        teams[team] = PenaltyProfile(team=team, **kwargs)
    return PenaltyTable(teams=teams)


def shootout_rating(
    profile: PenaltyProfile | None, params: KnockoutUpsetParams
) -> float | None:
    """Blend current keeper/taker strength and shrunk history into [-1, 1].

    Current indicators (weighted higher) dominate old country shootout history,
    which is shrunk toward 50/50 first. Returns ``None`` when a team has neither
    current ratings nor any shootout record.
    """
    if profile is None or not profile.has_any():
        return None
    # Current component: weighted mean of present 0-100 ratings, mapped to [-1, 1].
    cur_num = 0.0
    cur_w = 0.0
    for col, weight in _PEN_CURRENT:
        val = getattr(profile, col)
        if val is not None:
            cur_num += weight * _norm_0_100(val)
            cur_w += weight
    current = cur_num / cur_w if cur_w > 0.0 else None
    # History component: shrunk win rates, WC weighted most, mapped to [-1, 1].
    hist_specs = (
        (profile.shootout_wins, profile.shootout_losses, 0.5),
        (profile.world_cup_shootout_wins, profile.world_cup_shootout_losses, 1.0),
        (profile.continental_shootout_wins, profile.continental_shootout_losses, 0.7),
    )
    hist_num = 0.0
    hist_w = 0.0
    for wins, losses, weight in hist_specs:
        rate = shrunk_winrate(wins, losses, prior_strength=params.shootout_prior_strength)
        if rate is not None:
            hist_num += weight * (2.0 * rate - 1.0)
            hist_w += weight
    history = hist_num / hist_w if hist_w > 0.0 else None
    num = 0.0
    wsum = 0.0
    if current is not None:
        num += params.current_pen_weight * current
        wsum += params.current_pen_weight
    if history is not None:
        num += params.history_pen_weight * history
        wsum += params.history_pen_weight
    if wsum <= 0.0:
        return None
    return num / wsum


def shootout_favorite_prob(
    fav: PenaltyProfile | None,
    und: PenaltyProfile | None,
    params: KnockoutUpsetParams,
) -> float:
    """P(favourite wins the shootout), shrunk and hard-capped around 0.5.

    With no penalty data on either side this is exactly 0.5. The deviation from
    0.5 is ``beta * (rating_fav - rating_und)`` clipped to ``+/- shootout_cap``,
    so even a lopsided edge yields a modest price (e.g. 0.56/0.44), never a
    deterministic "this country always wins penalties".
    """
    r_fav = shootout_rating(fav, params)
    r_und = shootout_rating(und, params)
    if r_fav is None and r_und is None:
        return 0.5
    gap = (r_fav or 0.0) - (r_und or 0.0)
    dev = float(np.clip(params.shootout_beta * gap, -params.shootout_cap, params.shootout_cap))
    return 0.5 + dev


# --------------------------------------------------------------------------- #
# Staged regulation / extra-time / penalty advance model.
# --------------------------------------------------------------------------- #


def _wdl(lam_fav: float, lam_und: float) -> tuple[float, float, float]:
    """Skellam (favourite_win, draw, underdog_win) for one period."""
    fav = float(1.0 - skellam.cdf(0, lam_fav, lam_und))
    draw = float(skellam.pmf(0, lam_fav, lam_und))
    und = float(skellam.cdf(-1, lam_fav, lam_und))
    return fav, draw, und


def _lambdas(eg_total: float, skill: float) -> tuple[float, float]:
    """Split a goal total into favourite/underdog Poisson means *multiplicatively*.

    ``skill`` is a dimensionless half-difference: ``lam_fav = eg/2 * e^{skill}``,
    ``lam_und = eg/2 * e^{-skill}``. The split is ratio-preserving, so lowering
    ``eg_total`` (a low-event, compact matchup) scales both means down while
    keeping their *ratio* fixed — which raises the draw mass and routes more of
    the favourite's edge through the near-coin-flip penalty path. (An additive
    margin would instead inflate the favourite's relative edge as goals fall.)
    """
    half = max(eg_total, 0.1) / 2.0
    lam_fav = max(half * float(np.exp(skill)), 0.05)
    lam_und = max(half * float(np.exp(-skill)), 0.05)
    return lam_fav, lam_und


def staged_favorite_advance(
    eg_total: float, skill: float, shootout_fav_prob: float
) -> float:
    """P(favourite advances) from the regulation/ET/penalty staged model.

    ``eg_total`` is split multiplicatively by ``skill`` (see :func:`_lambdas`);
    extra time runs at one third intensity (30 of 90 minutes).
    """
    lam_fav, lam_und = _lambdas(eg_total, skill)
    reg_fav, reg_draw, _ = _wdl(lam_fav, lam_und)
    et_fav, et_draw, _ = _wdl(lam_fav / 3.0, lam_und / 3.0)
    return reg_fav + reg_draw * (et_fav + et_draw * shootout_fav_prob)


# --------------------------------------------------------------------------- #
# The signal.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KnockoutUpset:
    """Result of the knockout survival adjustment, with provenance.

    ``advance`` is ``None`` when the signal does not apply (group stage, or no
    style/penalty evidence for either side). ``shift`` is the signed advance
    probability moved toward the underdog; ``paths`` are human-readable
    provenance tags; ``detail`` carries the intermediate quantities.
    """

    advance: AdvanceProbs | None
    favorite: str | None
    underdog: str | None
    shift: float
    paths: list[str]
    detail: dict


def knockout_upset_detail(
    team_a: str,
    team_b: str,
    *,
    base_advance: AdvanceProbs,
    styles: TeamStyleTable,
    penalties: PenaltyTable,
    params: KnockoutUpsetParams | None = None,
) -> KnockoutUpset:
    """Compute the survival-aware advance adjustment for a knockout tie.

    The favourite/underdog roles come from ``base_advance``. Returns a
    :class:`KnockoutUpset` whose ``advance`` is ``None`` when there is no style
    or penalty evidence for either team (so the ensemble simply renormalizes the
    signal away).
    """
    p = params or KnockoutUpsetParams()
    have_evidence = (
        styles.has(team_a) or styles.has(team_b)
        or penalties.has(team_a) or penalties.has(team_b)
    )
    if not have_evidence:
        return KnockoutUpset(None, None, None, 0.0, [], {"reason": "no_evidence"})

    a_is_fav = base_advance.team_a_advances >= base_advance.team_b_advances
    fav_name, und_name = (team_a, team_b) if a_is_fav else (team_b, team_a)
    p_fav_base = max(base_advance.team_a_advances, base_advance.team_b_advances)

    fav_style = styles.get(fav_name)
    und_style = styles.get(und_name)

    # Base (dimensionless) skill half-difference implied by the advance edge.
    skill_base = p.edge_per_advance * (p_fav_base - 0.5) * 2.0

    # --- expected-goals (draw mass) -------------------------------------------
    eg = p.base_expected_goals
    eg -= p.low_block_eg * max(und_style.n("low_block_defense"), 0.0)
    eg -= p.compactness_eg * max(und_style.n("defensive_compactness"), 0.0)
    eg -= p.sterile_eg * max(fav_style.n("sterile_possession_risk"), 0.0)
    eg += p.transition_eg * max(und_style.n("transition_threat"), 0.0)
    eg += p.creation_eg * max(fav_style.n("chance_creation"), 0.0)
    eg = float(np.clip(eg, p.min_expected_goals, p.max_expected_goals))

    # --- regulation skill shrink (upset shapes) -------------------------------
    low_block_survival = max(fav_style.n("struggles_vs_low_block"), 0.0) * max(
        und_style.n("low_block_defense"), 0.0
    )
    sterile = max(fav_style.n("sterile_possession_risk"), 0.0)
    transition = max(und_style.n("transition_threat"), 0.0)
    set_piece = max(
        (und_style.n("set_piece_threat") + und_style.n("aerial_threat")) / 2.0, 0.0
    )
    skill = skill_base
    skill -= p.struggles_block_edge * low_block_survival
    skill -= p.sterile_edge * sterile
    skill -= p.transition_edge * transition
    skill -= p.set_piece_edge * set_piece
    skill = max(skill, 0.0)  # never flip the favourite/underdog roles outright

    # --- shootout (only meaningful when draw mass is high) --------------------
    fav_pen = penalties.get(fav_name)
    und_pen = penalties.get(und_name)
    shoot_fav = shootout_favorite_prob(fav_pen, und_pen, p)

    # --- staged advance: adjusted vs neutral, anchored to the base ------------
    adjusted_fav = staged_favorite_advance(eg, skill, shoot_fav)
    neutral_fav = staged_favorite_advance(p.base_expected_goals, skill_base, 0.5)
    # Underdog gains exactly the style/shootout-driven difference.
    raw_shift = neutral_fav - adjusted_fav
    shift = float(np.clip(raw_shift, -p.max_advance_shift, p.max_advance_shift))

    p_fav_final = float(np.clip(p_fav_base - shift, 0.02, 0.98))
    if a_is_fav:
        advance = AdvanceProbs(p_fav_final, 1.0 - p_fav_final)
    else:
        advance = AdvanceProbs(1.0 - p_fav_final, p_fav_final)

    lam_fav_adj, lam_und_adj = _lambdas(eg, skill)
    reg_draw = float(skellam.pmf(0, lam_fav_adj, lam_und_adj))
    paths = _provenance(
        low_block_survival=low_block_survival,
        sterile=sterile,
        transition=transition,
        set_piece=set_piece,
        shoot_fav=shoot_fav,
        reg_draw=reg_draw,
        threshold=p.path_tag_threshold,
    )
    detail = {
        "favorite": fav_name,
        "underdog": und_name,
        "p_fav_base": round(p_fav_base, 4),
        "expected_goals_total": round(eg, 3),
        "regulation_draw_prob": round(reg_draw, 4),
        "skill_base": round(skill_base, 3),
        "skill_adjusted": round(skill, 3),
        "shootout_fav_prob": round(shoot_fav, 4),
        "shift_to_underdog": round(shift, 4),
    }
    return KnockoutUpset(advance, fav_name, und_name, shift, paths, detail)


# Provenance tags grouped by what drives them, so diagnostics can report the
# style-shape contribution separately from the penalty/shootout contribution.
STYLE_PROVENANCE_TAGS: tuple[str, ...] = (
    "low_block_survival_path",
    "favorite_sterile_possession_risk",
    "transition_threat",
    "set_piece_underdog_path",
)
PENALTY_PROVENANCE_TAGS: tuple[str, ...] = ("penalty_path_boost",)


def _provenance(
    *,
    low_block_survival: float,
    sterile: float,
    transition: float,
    set_piece: float,
    shoot_fav: float,
    reg_draw: float,
    threshold: float,
) -> list[str]:
    """Human-readable tags for whichever upset paths are materially active."""
    paths: list[str] = []
    if low_block_survival >= threshold * threshold:
        paths.append("low_block_survival_path")
    if sterile >= threshold:
        paths.append("favorite_sterile_possession_risk")
    if transition >= threshold:
        paths.append("transition_threat")
    if set_piece >= threshold:
        paths.append("set_piece_underdog_path")
    # Penalties only matter when the match is genuinely likely to reach them.
    if abs(shoot_fav - 0.5) >= 0.01 and reg_draw >= 0.20:
        paths.append("penalty_path_boost")
    return paths


def knockout_upset_signal(
    team_a: str,
    team_b: str,
    *,
    base_advance: AdvanceProbs,
    styles: TeamStyleTable,
    penalties: PenaltyTable,
    params: KnockoutUpsetParams | None = None,
) -> AdvanceProbs | None:
    """Advance-probability signal for a knockout tie, or ``None`` if inactive.

    Thin wrapper over :func:`knockout_upset_detail` returning just the
    :class:`~goalsignal.signals.base.AdvanceProbs` the meta-ensemble blends.
    """
    return knockout_upset_detail(
        team_a, team_b,
        base_advance=base_advance, styles=styles, penalties=penalties, params=params,
    ).advance
