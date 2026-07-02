"""Winner-only human adjustment challenger over full-tournament artifacts.

This layer is an opt-in challenger, not a model. It reads an *existing*
simulation directory (never re-running or overwriting the simulator), applies
transparent, YAML-configured percentage-point adjustments to knockout advance
probabilities, propagates a fixed predicted winner through the official
M73-M104 bracket graph, and writes a fully auditable report. The deployed
model, the prediction ledger, the result store, and every existing artifact
are untouched; outputs are new files inside the simulation directory.

Adjustment math (documented in the report as well):

1. The baseline probability for a pairing is the simulated conditional
   advance probability (``conditional_slot_1_win_probability``) for that
   pairing in the round matchup CSVs; a pairing never observed in simulation
   falls back to a flagged neutral 0.5.
2. Each configured adjustment adds ``points`` percentage points to one team.
   Per-team sums are capped at ``max_total_adjustment_pct``; the difference
   between the two teams' nets is capped again at the same limit and applied
   to the baseline, then clipped into [min_probability, max_probability].
3. The predicted winner is the team with the higher adjusted probability
   (ties break toward the higher baseline, then slot order), and winners /
   losers feed later rounds deterministically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from goalsignal.tournament.bracket_2026 import MatchSlot
from goalsignal.tournament.reporting import ROUND_FILES
from goalsignal.utils.hashing import sha256_json
from goalsignal.utils.paths import resolve

KNOCKOUT_MATCH_NUMBERS = range(73, 105)
CONFIDENCE_LEVELS = ("low", "medium", "high")

MODIFIER_CATEGORIES: dict[str, tuple[str, ...]] = {
    "venue": ("home_host_boost", "altitude_boost", "travel_rest_penalty"),
    "injuries": ("attack_downgrade", "defense_downgrade", "goalkeeper_downgrade"),
    "tournament_form": (
        "dominant_win_boost",
        "late_comeback_boost",
        "late_collapse_penalty",
        "extra_time_fatigue_penalty",
        "penalty_survival_boost",
    ),
    "opponent_quality": ("hard_group_boost", "easy_path_penalty", "battle_tested_boost"),
    "style_matchup": (
        "low_block_vs_possession_boost",
        "transition_threat_boost",
        "set_piece_boost",
        "sterile_possession_penalty",
    ),
    "expert_override": ("manual_nudge",),
}
MODIFIER_TO_CATEGORY = {
    modifier: category
    for category, modifiers in MODIFIER_CATEGORIES.items()
    for modifier in modifiers
}

CSV_NAME = "human_adjusted_bracket.csv"
MD_NAME = "human_adjusted_bracket.md"
META_NAME = "human_adjusted_meta.json"


# --------------------------------------------------------------------------- #
# Configuration.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Adjustment:
    team: str
    category: str
    modifier: str | None
    points: float
    reason: str
    confidence: str | None

    @property
    def display_category(self) -> str:
        return f"{self.category}/{self.modifier}" if self.modifier else self.category


@dataclass(frozen=True)
class MatchAdjustments:
    match_number: int
    label: str | None
    adjustments: tuple[Adjustment, ...]


@dataclass
class HumanAdjustmentsConfig:
    max_total_adjustment_pct: float
    max_single_adjustment_pct: float
    min_probability: float
    max_probability: float
    matches: dict[int, MatchAdjustments]
    config_hash: str
    source_path: str

    @classmethod
    def load(cls, path: str | Path = "config/human_adjustments_2026.yaml"):
        config_path = resolve(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        problems: list[str] = []
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: config must be a mapping")
        global_cfg = raw.get("global") or {}
        max_total = float(global_cfg.get("max_total_adjustment_pct", 15))
        max_single = float(global_cfg.get("max_single_adjustment_pct", max_total))
        min_p = float(global_cfg.get("min_probability", 0.05))
        max_p = float(global_cfg.get("max_probability", 0.95))
        if not 0.0 <= min_p < max_p <= 1.0:
            problems.append("global: require 0 <= min_probability < max_probability <= 1")
        if max_total <= 0 or max_single <= 0:
            problems.append("global: adjustment caps must be positive")
        matches: dict[int, MatchAdjustments] = {}
        for key, block in (raw.get("matches") or {}).items():
            try:
                number = int(key)
            except (TypeError, ValueError):
                problems.append(f"matches.{key!r}: match key must be an integer")
                continue
            if number not in KNOCKOUT_MATCH_NUMBERS:
                problems.append(f"M{number}: knockout match numbers are 73-104")
                continue
            block = block or {}
            entries = []
            for i, item in enumerate(block.get("adjustments") or []):
                prefix = f"M{number}.adjustments[{i}]"
                entries.append(
                    _parse_adjustment(item, prefix, max_single, problems)
                )
            matches[number] = MatchAdjustments(
                match_number=number,
                label=block.get("label"),
                adjustments=tuple(a for a in entries if a is not None),
            )
        if problems:
            raise ValueError(
                "invalid human adjustments config: " + "; ".join(problems)
            )
        return cls(
            max_total_adjustment_pct=max_total,
            max_single_adjustment_pct=max_single,
            min_probability=min_p,
            max_probability=max_p,
            matches=matches,
            config_hash=sha256_json(raw),
            source_path=str(path),
        )

    def configured_teams(self) -> set[str]:
        return {
            adj.team
            for match in self.matches.values()
            for adj in match.adjustments
        }


def _parse_adjustment(
    item: object, prefix: str, max_single: float, problems: list[str]
) -> Adjustment | None:
    if not isinstance(item, dict):
        problems.append(f"{prefix}: each adjustment must be a mapping")
        return None
    team = str(item.get("team") or "").strip()
    if not team:
        problems.append(f"{prefix}: 'team' is required")
    reason = str(item.get("reason") or "").strip()
    if not reason:
        problems.append(f"{prefix}: 'reason' is required")
    raw_category = str(item.get("category") or "").strip()
    modifier = item.get("modifier")
    modifier = str(modifier).strip() if modifier is not None else None
    category = raw_category
    if raw_category in MODIFIER_CATEGORIES:
        if modifier is not None and modifier not in MODIFIER_CATEGORIES[raw_category]:
            problems.append(
                f"{prefix}: modifier {modifier!r} is not in category {raw_category!r}"
            )
    elif raw_category in MODIFIER_TO_CATEGORY:
        # A known modifier name used directly as the category.
        category = MODIFIER_TO_CATEGORY[raw_category]
        if modifier is None:
            modifier = raw_category
        elif modifier != raw_category:
            problems.append(
                f"{prefix}: category {raw_category!r} conflicts with modifier {modifier!r}"
            )
    else:
        problems.append(f"{prefix}: unknown category {raw_category!r}")
    try:
        points = float(item.get("points"))
    except (TypeError, ValueError):
        problems.append(f"{prefix}: 'points' must be numeric")
        points = 0.0
    if abs(points) > max_single:
        problems.append(
            f"{prefix}: |points| {abs(points):g} exceeds "
            f"max_single_adjustment_pct {max_single:g}"
        )
    confidence = item.get("confidence")
    if confidence is not None:
        confidence = str(confidence).strip().lower()
        if confidence not in CONFIDENCE_LEVELS:
            problems.append(
                f"{prefix}: confidence must be one of {', '.join(CONFIDENCE_LEVELS)}"
            )
    return Adjustment(
        team=team,
        category=category,
        modifier=modifier,
        points=points,
        reason=reason,
        confidence=confidence,
    )


# --------------------------------------------------------------------------- #
# Simulation baseline loading (read-only).
# --------------------------------------------------------------------------- #


@dataclass
class SimulationBaseline:
    path: Path
    meta: dict
    modal: dict[int, dict]
    pair_probs: dict[tuple[int, str, str], float]
    teams: set[str]


def load_simulation_baseline(sim_dir: str | Path) -> SimulationBaseline:
    """Load the matchup CSVs and modal bracket of an existing simulation."""
    path = Path(sim_dir)
    bracket_path = path / "wc2026_bracket.json"
    if not bracket_path.exists():
        raise FileNotFoundError(f"no simulation bracket found at {bracket_path}")
    bracket = json.loads(bracket_path.read_text(encoding="utf-8"))
    modal = {int(m["match_number"]): m for m in bracket.get("matches", [])}
    meta_path = path / "wc2026_tournament_meta.json"
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    pair_probs: dict[tuple[int, str, str], float] = {}
    teams: set[str] = set()
    for filename in ROUND_FILES.values():
        round_path = path / filename
        if not round_path.exists():
            continue
        frame = pd.read_csv(round_path)
        for row in frame.itertuples(index=False):
            key = (int(row.match_number), str(row.slot_1_team), str(row.slot_2_team))
            pair_probs[key] = float(row.conditional_slot_1_win_probability)
            teams.update(key[1:])
    if not pair_probs:
        raise FileNotFoundError(f"no matchup CSVs found in {path}")
    return SimulationBaseline(
        path=path, meta=meta, modal=modal, pair_probs=pair_probs, teams=teams
    )


def baseline_probability(
    baseline: SimulationBaseline, number: int, team_1: str, team_2: str
) -> tuple[float, str]:
    """Simulated advance probability for team_1 in a pairing, either orientation."""
    forward = baseline.pair_probs.get((number, team_1, team_2))
    if forward is not None:
        return forward, "simulated_matchup"
    reverse = baseline.pair_probs.get((number, team_2, team_1))
    if reverse is not None:
        return 1.0 - reverse, "simulated_matchup"
    return 0.5, "neutral_fallback"


# --------------------------------------------------------------------------- #
# Winner-only bracket walk.
# --------------------------------------------------------------------------- #


@dataclass
class AdjustedMatch:
    match_number: int
    round: str
    date: str
    host_city: str
    team_1: str
    team_2: str
    baseline_p_team_1: float
    baseline_source: str
    net_points_team_1: float
    net_points_team_2: float
    applied_delta_pct: float
    adjusted_p_team_1: float
    predicted_winner: str
    baseline_winner: str
    winner_changed: bool
    applied: tuple[Adjustment, ...] = ()
    skipped: tuple[Adjustment, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def adjusted_p_team_2(self) -> float:
        return 1.0 - self.adjusted_p_team_1

    @property
    def predicted_loser(self) -> str:
        return self.team_2 if self.predicted_winner == self.team_1 else self.team_1


@dataclass
class HumanAdjustedBracket:
    matches: list[AdjustedMatch]
    warnings: list[str]
    config: HumanAdjustmentsConfig
    baseline: SimulationBaseline

    @property
    def champion(self) -> str | None:
        finals = [m for m in self.matches if m.round == "final"]
        return finals[-1].predicted_winner if finals else None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _resolve_entrant(
    slot: str,
    number: int,
    winners: dict[int, str],
    losers: dict[int, str],
    modal_pair: list[str] | None,
    index: int,
    notes: list[str],
) -> str | None:
    if slot and slot[0] in "WL" and slot[1:].isdigit():
        feeder = int(slot[1:])
        source = winners if slot[0] == "W" else losers
        team = source.get(feeder)
        if team is None:
            notes.append(f"M{number}: feeder match M{feeder} is unresolved")
        return team
    # Group-stage slot (e.g. "1A", "THIRD"): resolved matchups come from the
    # simulation's modal bracket, never fabricated here.
    if modal_pair is not None and len(modal_pair) == 2:
        return str(modal_pair[index])
    notes.append(f"M{number}: no modal matchup available for slot {slot!r}")
    return None


def adjust_bracket(
    baseline: SimulationBaseline,
    config: HumanAdjustmentsConfig,
    bracket_matches: dict[int, MatchSlot],
) -> HumanAdjustedBracket:
    """Walk the official knockout graph and fix one adjusted winner per match."""
    warnings: list[str] = []
    unknown = sorted(config.configured_teams() - baseline.teams)
    if unknown:
        raise ValueError(
            "adjustment teams not present in the simulation artifacts: "
            + ", ".join(unknown)
        )
    winners: dict[int, str] = {}
    losers: dict[int, str] = {}
    results: list[AdjustedMatch] = []
    for number in sorted(bracket_matches):
        slot = bracket_matches[number]
        modal = baseline.modal.get(number, {})
        notes: list[str] = []
        modal_pair = modal.get("modal_matchup")
        entrants = [
            _resolve_entrant(
                symbolic, number, winners, losers, modal_pair, index, notes
            )
            for index, symbolic in enumerate(slot.entrants)
        ]
        if entrants[0] is None or entrants[1] is None:
            warnings.append(
                f"M{number}: could not resolve both entrants; match skipped"
            )
            continue
        team_1, team_2 = entrants
        if modal_pair and set(modal_pair) != {team_1, team_2}:
            notes.append(
                "propagated pairing differs from the modal simulated matchup "
                f"({' vs '.join(modal_pair)})"
            )
        p_1, source = baseline_probability(baseline, number, team_1, team_2)
        if source == "neutral_fallback":
            notes.append(
                "pairing never observed in simulation; neutral 0.5 baseline used"
            )
        match_cfg = config.matches.get(number)
        applied: list[Adjustment] = []
        skipped: list[Adjustment] = []
        if match_cfg is not None:
            for adj in match_cfg.adjustments:
                if adj.team in (team_1, team_2):
                    applied.append(adj)
                else:
                    skipped.append(adj)
                    warnings.append(
                        f"M{number}: adjustment for {adj.team!r} skipped; "
                        f"actual pairing is {team_1} vs {team_2}"
                    )
            if match_cfg.label:
                label_teams = {t.strip() for t in match_cfg.label.split(" vs ")}
                if label_teams and not label_teams & {team_1, team_2}:
                    warnings.append(
                        f"M{number}: config label {match_cfg.label!r} does not match "
                        f"the propagated pairing {team_1} vs {team_2}"
                    )
        cap = config.max_total_adjustment_pct
        net_1 = _clamp(sum(a.points for a in applied if a.team == team_1), -cap, cap)
        net_2 = _clamp(sum(a.points for a in applied if a.team == team_2), -cap, cap)
        delta = _clamp((net_1 - net_2) / 100.0, -cap / 100.0, cap / 100.0)
        adjusted_1 = (
            _clamp(p_1 + delta, config.min_probability, config.max_probability)
            if applied
            else p_1
        )
        baseline_winner = team_1 if p_1 >= 0.5 else team_2
        if adjusted_1 > 0.5:
            predicted = team_1
        elif adjusted_1 < 0.5:
            predicted = team_2
        else:
            predicted = baseline_winner
        winners[number] = predicted
        losers[number] = team_2 if predicted == team_1 else team_1
        results.append(
            AdjustedMatch(
                match_number=number,
                round=slot.round,
                date=slot.date,
                host_city=slot.host_city,
                team_1=team_1,
                team_2=team_2,
                baseline_p_team_1=p_1,
                baseline_source=source,
                net_points_team_1=net_1,
                net_points_team_2=net_2,
                applied_delta_pct=delta * 100.0,
                adjusted_p_team_1=adjusted_1,
                predicted_winner=predicted,
                baseline_winner=baseline_winner,
                winner_changed=predicted != baseline_winner,
                applied=tuple(applied),
                skipped=tuple(skipped),
                notes=tuple(notes),
            )
        )
    return HumanAdjustedBracket(
        matches=results, warnings=warnings, config=config, baseline=baseline
    )


# --------------------------------------------------------------------------- #
# Artifacts.
# --------------------------------------------------------------------------- #


def bracket_frame(result: HumanAdjustedBracket) -> pd.DataFrame:
    rows = []
    for match in result.matches:
        rows.append({
            "match_number": match.match_number,
            "round": match.round,
            "date": match.date,
            "host_city": match.host_city,
            "team_1": match.team_1,
            "team_2": match.team_2,
            "baseline_p_team_1": match.baseline_p_team_1,
            "baseline_source": match.baseline_source,
            "n_adjustments_applied": len(match.applied),
            "net_points_team_1": match.net_points_team_1,
            "net_points_team_2": match.net_points_team_2,
            "applied_delta_pct": match.applied_delta_pct,
            "adjusted_p_team_1": match.adjusted_p_team_1,
            "adjusted_p_team_2": match.adjusted_p_team_2,
            "predicted_winner": match.predicted_winner,
            "baseline_winner": match.baseline_winner,
            "winner_changed": match.winner_changed,
            "notes": " | ".join(match.notes),
        })
    return pd.DataFrame(rows)


def render_markdown(result: HumanAdjustedBracket) -> str:
    config = result.config
    meta = result.baseline.meta
    lines = [
        "# Human-adjusted knockout bracket (winner-only challenger)",
        "",
        "Opinion layer over an existing simulation. Adjustments are configured in "
        f"`{config.source_path}`, applied transparently, and reported below. "
        "This is not a fitted model, makes no accuracy claim, and does not touch "
        "the deployed model, the prediction ledger, or existing artifacts.",
        "",
        "## Provenance",
        "",
        f"- Simulation directory: `{result.baseline.path}`",
        f"- Simulation model version: {meta.get('model_version', 'unknown')}",
        f"- Simulations: {meta.get('n_sims', 'unknown')}, "
        f"seed {meta.get('seed', 'unknown')}, "
        f"dataset {meta.get('dataset_version', 'unknown')}",
        f"- Adjustments config hash: `{config.config_hash}`",
        f"- Caps: max total ±{config.max_total_adjustment_pct:g} pct pts per match, "
        f"max single ±{config.max_single_adjustment_pct:g} pct pts, "
        f"probabilities clipped to [{config.min_probability:g}, "
        f"{config.max_probability:g}]",
        "",
        "## Method",
        "",
        "Baseline = simulated conditional advance probability for the propagated "
        "pairing (regulation + extra time + penalties, as simulated). Each "
        "adjustment adds signed percentage points to one team; per-team sums are "
        "capped, the two nets are differenced, capped again, added to the "
        "baseline, and clipped. Winners are fixed by the adjusted probability "
        "and feed later rounds, so the path below is a single deterministic "
        "bracket, not a re-simulation.",
        "",
        "## Predicted bracket",
        "",
        "| M | Round | Pairing | Baseline p(team 1) | Adjusted p(team 1) | "
        "Predicted winner | Changed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for match in result.matches:
        lines.append(
            f"| {match.match_number} | {match.round} | "
            f"{match.team_1} vs {match.team_2} | "
            f"{match.baseline_p_team_1:.3f} | {match.adjusted_p_team_1:.3f} | "
            f"**{match.predicted_winner}** | "
            f"{'YES' if match.winner_changed else ''} |"
        )
    champion = result.champion
    if champion:
        lines += ["", f"**Predicted champion: {champion}**"]
    adjusted = [m for m in result.matches if m.applied or m.skipped]
    if adjusted:
        lines += ["", "## Adjustment detail", ""]
        for match in adjusted:
            lines += [
                f"### M{match.match_number} — {match.team_1} vs {match.team_2} "
                f"({match.round})",
                "",
                f"- Baseline p({match.team_1}) = {match.baseline_p_team_1:.3f} "
                f"({match.baseline_source})",
            ]
            for adj in match.applied:
                confidence = f", confidence {adj.confidence}" if adj.confidence else ""
                lines.append(
                    f"- {adj.points:+g} pts {adj.team} "
                    f"[{adj.display_category}{confidence}]: {adj.reason}"
                )
            for adj in match.skipped:
                lines.append(
                    f"- SKIPPED {adj.points:+g} pts {adj.team} "
                    f"[{adj.display_category}]: not in this pairing"
                )
            lines += [
                f"- Net: {match.team_1} {match.net_points_team_1:+g} pts, "
                f"{match.team_2} {match.net_points_team_2:+g} pts → applied "
                f"{match.applied_delta_pct:+.1f} pct pts (caps enforced)",
                f"- Adjusted p({match.team_1}) = {match.adjusted_p_team_1:.3f}; "
                f"predicted winner **{match.predicted_winner}**"
                + (" (flipped from baseline)" if match.winner_changed else ""),
                "",
            ]
    if result.warnings:
        lines += ["## Warnings", ""]
        lines += [f"- {warning}" for warning in result.warnings]
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- Winner-only: probabilities here rank one fixed bracket path; they are "
        "not calibrated forecasts and are not written to the ledger.",
        "- Adjustment points are human opinions, not fitted coefficients.",
        "- Downstream pairings assume every earlier predicted winner is correct.",
        "",
    ]
    return "\n".join(lines)


def write_human_adjusted(
    result: HumanAdjustedBracket,
    out_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> dict[str, Path]:
    out = Path(out_dir) if out_dir is not None else result.baseline.path
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out / CSV_NAME,
        "md": out / MD_NAME,
        "meta": out / META_NAME,
    }
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not force:
        raise FileExistsError(
            "human-adjusted artifacts already exist (pass --force to overwrite): "
            + ", ".join(existing)
        )
    bracket_frame(result).to_csv(paths["csv"], index=False)
    paths["md"].write_text(render_markdown(result), encoding="utf-8")
    meta = {
        "layer": "human_adjustments",
        "config_path": result.config.source_path,
        "config_hash": result.config.config_hash,
        "simulation_dir": str(result.baseline.path),
        "simulation_model_version": result.baseline.meta.get("model_version"),
        "simulation_n_sims": result.baseline.meta.get("n_sims"),
        "simulation_dataset_version": result.baseline.meta.get("dataset_version"),
        "n_matches": len(result.matches),
        "n_matches_adjusted": sum(1 for m in result.matches if m.applied),
        "n_winners_changed": sum(1 for m in result.matches if m.winner_changed),
        "predicted_champion": result.champion,
        "warnings": result.warnings,
    }
    paths["meta"].write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return paths
