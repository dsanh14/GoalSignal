"""Glue: load the manual signal files and assemble per-match signal sets.

Keeps the CLI thin and the blend logic testable. Given a directory of manual
inputs and a list of match specs, it builds the ``{signal_name: probs|None}``
dict each match needs and runs the meta-ensemble. Adjustment signals
(squad/form/venue) are computed as group-stage distributions and reduced to
advance probabilities for knockout matches via the closed-form tiebreak.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from goalsignal.signals.base import (
    AdvanceProbs,
    OutcomeProbs,
    advance_from_outcome,
    davidson_outcome,
)
from goalsignal.signals.expert import expert_consensus, load_expert_predictions
from goalsignal.signals.keying import PairIndex
from goalsignal.signals.knockout_upset import (
    KnockoutUpsetParams,
    PenaltyTable,
    TeamStyleTable,
    knockout_upset_signal,
    load_penalties,
    load_team_styles,
)
from goalsignal.signals.market import load_market_odds
from goalsignal.signals.meta_ensemble import (
    BlendResult,
    EnsembleConfig,
    MetaEnsemble,
    load_ensemble_config,
)
from goalsignal.signals.recent_form import form_signal, load_recent_form
from goalsignal.signals.squad_strength import load_squad_strength, squad_signal
from goalsignal.signals.venue_context import (
    VenueCoefficients,
    load_venue_context,
)

SIGNAL_NAMES = (
    "historical",
    "market",
    "squad_strength",
    "recent_form",
    "expert",
    "venue_context",
    "knockout_upset",
)

# File name (without extension) per signal under the manual directory. A real
# file (`market_odds.csv`) is preferred; the bundled `*.example.csv` is the
# fallback so the pipeline runs out of the box.
_FILES = {
    "market": "market_odds",
    "squad_strength": "squad_strength",
    "recent_form": "recent_form",
    "venue_context": "venue_context",
    "expert": "expert_predictions",
    "team_styles": "team_styles",
    "penalties": "penalties",
}


def _resolve_manual_file(directory: Path, stem: str) -> Path:
    real = directory / f"{stem}.csv"
    return real if real.exists() else directory / f"{stem}.example.csv"


def _is_pair_key(key: str) -> bool:
    return str(key).startswith("pair::")


def _market_index(market: dict) -> PairIndex:
    return PairIndex.build(
        [
            (None if _is_pair_key(k) else k, q.team_a, q.team_b, q)
            for k, q in market.items()
        ]
    )


def _expert_index(expert: dict) -> PairIndex:
    entries = []
    for k, preds in expert.items():
        first = preds[0]
        entries.append((None if _is_pair_key(k) else k, first.team_a, first.team_b, preds))
    return PairIndex.build(entries)


def _venue_index(venue: dict) -> PairIndex:
    return PairIndex.build(
        [
            (None if _is_pair_key(k) else k, c.team_a, c.team_b, c)
            for k, c in venue.items()
        ]
    )


@dataclass
class ManualInputs:
    """All loaded manual signal sources plus the resolved ensemble config.

    The ``*_index`` fields resolve match-keyed signals by ``match_id`` first and
    then by normalized team pair (so they attach to dynamic knockout pairings).
    """

    config: EnsembleConfig
    market: dict
    squad: object
    form: object
    venue: dict
    expert: dict
    load_errors: dict[str, list[str]]
    market_index: PairIndex
    expert_index: PairIndex
    venue_index: PairIndex
    styles: TeamStyleTable
    penalties: PenaltyTable
    include_knockout_upset: bool = False

    @property
    def venue_coeffs(self) -> VenueCoefficients:
        v = self.config.signal_params.get("venue", {})
        return VenueCoefficients(
            travel_per_1000km=float(v.get("travel_per_1000km", 8.0)),
            rest_per_day=float(v.get("rest_per_day", 6.0)),
            timezone_per_hour=float(v.get("timezone_per_hour", 3.0)),
        )

    @property
    def knockout_upset_params(self) -> KnockoutUpsetParams:
        return KnockoutUpsetParams.from_mapping(
            self.config.signal_params.get("knockout_upset")
        )


def load_manual_inputs(
    directory: str | Path = "data/manual",
    config: EnsembleConfig | None = None,
    *,
    include_knockout_upset: bool = False,
) -> ManualInputs:
    """Load every manual signal file from ``directory`` (graceful if absent).

    ``include_knockout_upset`` opts the knockout survival signal into the blend;
    when ``False`` (default) the signal is never produced and the ensemble is
    byte-for-byte unchanged. The style/penalty tables are always loaded (cheap)
    so they can be inspected regardless of the flag.
    """
    cfg = config or load_ensemble_config()
    d = Path(directory)
    errors: dict[str, list[str]] = {}
    market_err: list[str] = []
    expert_err: list[str] = []
    market = load_market_odds(_resolve_manual_file(d, _FILES["market"]), on_error=market_err)
    squad = load_squad_strength(_resolve_manual_file(d, _FILES["squad_strength"]))
    form = load_recent_form(_resolve_manual_file(d, _FILES["recent_form"]))
    venue = load_venue_context(_resolve_manual_file(d, _FILES["venue_context"]))
    expert = load_expert_predictions(
        _resolve_manual_file(d, _FILES["expert"]), on_error=expert_err
    )
    styles = load_team_styles(_resolve_manual_file(d, _FILES["team_styles"]))
    penalties = load_penalties(_resolve_manual_file(d, _FILES["penalties"]))
    if market_err:
        errors["market"] = market_err
    if expert_err:
        errors["expert"] = expert_err
    return ManualInputs(
        config=cfg,
        market=market,
        squad=squad,
        form=form,
        venue=venue,
        expert=expert,
        load_errors=errors,
        market_index=_market_index(market),
        expert_index=_expert_index(expert),
        venue_index=_venue_index(venue),
        styles=styles,
        penalties=penalties,
        include_knockout_upset=include_knockout_upset,
    )


@dataclass(frozen=True)
class MatchSpec:
    """A match to forecast, with optional historical-model probabilities."""

    match_id: str
    stage: str  # "group" or "knockout"
    team_a: str
    team_b: str
    historical: OutcomeProbs | AdvanceProbs | None = None
    neutral: bool = True  # venue neutrality (matters only for the live model)

    @property
    def knockout(self) -> bool:
        return self.stage.lower().startswith("knock")


def load_matches(path: str | Path) -> list[MatchSpec]:
    """Load match specs (incl. optional historical probs) from a CSV."""
    df = pd.read_csv(path)
    required = {"match_id", "stage", "team_a", "team_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"matches CSV missing columns: {sorted(missing)}")
    specs: list[MatchSpec] = []
    for _, row in df.iterrows():
        stage = str(row["stage"]).strip().lower()
        knockout = stage.startswith("knock")
        historical: OutcomeProbs | AdvanceProbs | None = None
        if knockout:
            cols = ("historical_team_a_advances", "historical_team_b_advances")
            if all(c in df.columns and pd.notna(row[c]) for c in cols):
                historical = AdvanceProbs(float(row[cols[0]]), float(row[cols[1]]))
        else:
            cols = ("historical_home_win", "historical_draw", "historical_away_win")
            if all(c in df.columns and pd.notna(row[c]) for c in cols):
                historical = OutcomeProbs(
                    float(row[cols[0]]), float(row[cols[1]]), float(row[cols[2]])
                )
        neutral = True
        if "neutral" in df.columns and pd.notna(row["neutral"]):
            neutral = bool(str(row["neutral"]).strip().lower() in {"1", "true", "yes"})
        specs.append(
            MatchSpec(
                match_id=str(row["match_id"]).strip(),
                stage=stage,
                team_a=str(row["team_a"]).strip(),
                team_b=str(row["team_b"]).strip(),
                historical=historical,
                neutral=neutral,
            )
        )
    return specs


def _market_probs(quote, ko: bool, method: str):
    """Market signal for one matchup; ``None`` if it cannot supply the mode."""
    if quote is None:
        return None
    if ko:
        return quote.advance(method)
    if quote.two_way:
        return None  # a knockout-only (2-way) price cannot give group W/D/L
    return quote.outcome(method)


def build_signals(
    spec: MatchSpec, inputs: ManualInputs
) -> dict[str, OutcomeProbs | AdvanceProbs | None]:
    """Assemble the ``{signal_name: probs|None}`` set for one match.

    Match-keyed signals (market, expert, venue) resolve by ``match_id`` first and
    then by normalized team pair, flipping perspective for a reverse-orientation
    match (see :mod:`goalsignal.signals.keying`).
    """
    params = inputs.config.signal_params
    scale = float(params.get("davidson_scale", 400.0))
    nu = float(params.get("davidson_nu", 1.0))
    method = inputs.config.market_overround_method
    a_tb = inputs.config.knockout_tiebreak_a_prob
    ko = spec.knockout

    def as_mode(outcome: OutcomeProbs | None):
        if outcome is None:
            return None
        return advance_from_outcome(outcome, a_tiebreak_prob=a_tb) if ko else outcome

    squad = squad_signal(
        inputs.squad, spec.team_a, spec.team_b,
        points_per_z=float(params.get("squad_points_per_z", 60.0)), scale=scale, nu=nu,
    )
    form = form_signal(
        inputs.form, spec.team_a, spec.team_b,
        points_per_z=float(params.get("form_points_per_z", 40.0)), scale=scale, nu=nu,
    )

    # Market (directional: flip W/D/L or A/B advance for a reverse match).
    quote, m_orient = inputs.market_index.resolve(spec.match_id, spec.team_a, spec.team_b)
    market = _market_probs(quote, ko, method)
    if market is not None and m_orient < 0:
        market = market.flip()

    # Expert consensus (directional).
    preds, e_orient = inputs.expert_index.resolve(spec.match_id, spec.team_a, spec.team_b)
    expert = expert_consensus(preds, knockout=ko) if preds else None
    if expert is not None and e_orient < 0:
        expert = expert.flip()

    # Venue advantage (team-A relative: a reverse match negates the advantage).
    ctx, v_orient = inputs.venue_index.resolve(spec.match_id, spec.team_a, spec.team_b)
    venue = None
    if ctx is not None and ctx.has_any():
        venue = as_mode(
            davidson_outcome(ctx.advantage_points(inputs.venue_coeffs) * v_orient,
                             scale=scale, nu=nu)
        )

    signals: dict[str, OutcomeProbs | AdvanceProbs | None] = {
        "historical": spec.historical,
        "market": market,
        "squad_strength": as_mode(squad),
        "recent_form": as_mode(form),
        "venue_context": venue,
        "expert": expert,
    }

    # Knockout-only survival adjustment (opt-in). Anchored to the best available
    # advance estimate so it never randomly boosts underdogs.
    if ko and inputs.include_knockout_upset:
        signals["knockout_upset"] = _knockout_upset(spec, signals, inputs)

    return signals


def _base_advance(
    spec: MatchSpec, signals: dict[str, OutcomeProbs | AdvanceProbs | None]
) -> AdvanceProbs:
    """Best available advance estimate to anchor the knockout survival signal.

    Precedence: the historical model, then market, then squad strength; falls
    back to a 50/50 prior when nothing is available.
    """
    for name in ("historical", "market", "squad_strength"):
        probs = signals.get(name)
        if isinstance(probs, AdvanceProbs):
            return probs
    return AdvanceProbs(0.5, 0.5)


def _knockout_upset(
    spec: MatchSpec,
    signals: dict[str, OutcomeProbs | AdvanceProbs | None],
    inputs: ManualInputs,
) -> AdvanceProbs | None:
    return knockout_upset_signal(
        spec.team_a,
        spec.team_b,
        base_advance=_base_advance(spec, signals),
        styles=inputs.styles,
        penalties=inputs.penalties,
        params=inputs.knockout_upset_params,
    )


def blend_match(
    spec: MatchSpec,
    inputs: ManualInputs,
    ensemble: MetaEnsemble,
    *,
    version: str | None = None,
) -> tuple[BlendResult, dict[str, OutcomeProbs | AdvanceProbs | None]]:
    """Build signals for ``spec`` and blend them; returns (result, signals)."""
    signals = build_signals(spec, inputs)
    if spec.knockout:
        result = ensemble.blend_advance(signals, version=version)
    else:
        result = ensemble.blend(signals, version=version)
    return result, signals
