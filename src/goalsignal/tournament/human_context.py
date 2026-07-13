"""Regenerate human-context manual inputs from live knockout evidence.

``goalsignal tournament update-human-context`` reads the two manual overlay
files — confirmed knockout results and knockout performance tags — resolves
the real (or provisional) round-of-16 pairings through the official bracket
graph, and regenerates the three human-signal inputs:

- ``data/manual/recent_form.csv`` — bounded tournament-form deltas applied to
  a preserved base snapshot (``recent_form_base.csv``), with a full audit
  trail in ``recent_form_context_audit.csv``;
- ``data/manual/expert_predictions.csv`` — one structured advance-probability
  row per R16 matchup under a dedicated ``source_model``, derived from the
  matchup baseline plus bounded tag/priority nudges, with reasoning;
- ``config/human_adjustments_2026.yaml`` — per-match adjustment blocks for
  the R16 (existing blocks for other matches are preserved).

Everything is an opinion/overlay layer: transparent, bounded, reversible, and
idempotent (re-running with the same inputs produces the same outputs).
Nothing here touches ``Datasets/``, the ledger, the result store, or the
deployed model. Existing target files are never overwritten without
``force=True``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from goalsignal.tournament.bracket_2026 import MatchSlot, OfficialBracket
from goalsignal.tournament.human_adjustments import HumanAdjustmentsConfig
from goalsignal.tournament.knockout_results import (
    DEFAULT_RESULTS_PATH,
    KnockoutResult,
    load_knockout_results,
)
from goalsignal.tournament.performance_tags import (
    DEFAULT_TAG_NUDGE_CAP,
    DEFAULT_TAGS_PATH,
    PerformanceTag,
    load_performance_tags,
    tag_nudge,
)
from goalsignal.utils.paths import resolve

DEFAULT_RECENT_FORM_PATH = "data/manual/recent_form.csv"
DEFAULT_RECENT_FORM_BASE_PATH = "data/manual/recent_form_base.csv"
DEFAULT_RECENT_FORM_AUDIT_PATH = "data/manual/recent_form_context_audit.csv"
DEFAULT_EXPERT_PATH = "data/manual/expert_predictions.csv"
DEFAULT_MATCHUPS_PATH = "data/manual/knockout_matchups.csv"
DEFAULT_ADJUSTMENTS_PATH = "config/human_adjustments_2026.yaml"

EXPERT_SOURCE_MODEL = "knockout-context-2026"
R16_MATCH_NUMBERS = tuple(range(89, 97))

#: Cap (percentage points) on the net form delta applied per team.
FORM_DELTA_CAP = 6.0
FORM_COLUMNS_ADJUSTED = ("elo_adj_last5", "xg_diff")

#: Cap (percentage points) on the expert advance-probability shift.
EXPERT_MAX_SHIFT_PCT = 15.0
EXPERT_MIN_PROB = 0.05
EXPERT_MAX_PROB = 0.95

EXPERT_COLUMNS = (
    "match_id",
    "team_a",
    "team_b",
    "source_model",
    "team_a_win_prob",
    "draw_prob",
    "team_b_win_prob",
    "team_a_advance_prob",
    "team_b_advance_prob",
    "confidence",
    "reasoning",
)


@dataclass(frozen=True)
class PriorityAdjustment:
    """A structural (non-tag) opinion entry for one team in one R16 match."""

    team: str
    category: str
    modifier: str | None
    points: float
    reason: str
    confidence: str = "medium"


@dataclass(frozen=True)
class PriorityMatch:
    """Editable priority context for one R16 match (opinions, not facts)."""

    match_number: int
    team_a: str
    team_b: str
    summary: str
    expert_confidence: float
    extras: tuple[PriorityAdjustment, ...] = ()
    # Used only when the pairing is absent from the matchup baselines
    # (e.g. a stale forecast file); recorded in the reasoning when used.
    fallback_baseline_a: float | None = None


#: Priority R16 context (2026). Teams whose feeder match is unconfirmed are
#: provisional and flagged as such wherever they are used.
R16_PRIORITY: dict[int, PriorityMatch] = {
    89: PriorityMatch(
        89,
        "Paraguay",
        "France",
        "France's dominant form against Paraguay's penalty survival; "
        "France still clearly favored.",
        0.75,
    ),
    90: PriorityMatch(
        90,
        "Canada",
        "Morocco",
        "Morocco's penalty and low-block survival makes them favorites over "
        "a Canada side that advanced but looked less dangerous.",
        0.70,
    ),
    92: PriorityMatch(
        92,
        "Mexico",
        "England",
        "Mexico City altitude and home crowd against England's slow-start "
        "warning; Mexico slight favorite.",
        0.70,
        extras=(
            PriorityAdjustment(
                "Mexico",
                "venue",
                "altitude_boost",
                7,
                "Mexico City altitude and home crowd.",
                confidence="high",
            ),
            PriorityAdjustment(
                "Mexico",
                "venue",
                "home_host_boost",
                3,
                "Unbeaten (3-0) at Estadio Azteca this tournament; kickoff confirmed "
                "for its original 6pm local/8pm ET slot despite storm-related rumors "
                "of a move to noon.",
                confidence="medium",
            ),
        ),
    ),
    93: PriorityMatch(
        93,
        "Portugal",
        "Spain",
        "Spain remain strong after the 3-0 over Austria, but Portugal's "
        "late comeback and transition threat make this near 50/50.",
        0.65,
        extras=(
            PriorityAdjustment(
                "Portugal",
                "style_matchup",
                "transition_threat_boost",
                4,
                "Portugal's transition and finishing threat against Spain's "
                "possession keeps this near even.",
            ),
        ),
    ),
    94: PriorityMatch(
        94,
        "United States",
        "Belgium",
        "Belgium slight favorite only: comeback resilience tempered by a "
        "defensive warning, against a USA side that won cleanly at home.",
        0.65,
        extras=(
            PriorityAdjustment(
                "United States",
                "venue",
                "home_host_boost",
                3,
                "Home crowd and venue familiarity.",
            ),
        ),
    ),
    95: PriorityMatch(
        95,
        "Argentina",
        "Egypt",
        "Argentina favored over penalty-surviving Egypt, but a likely "
        "untested path reduces confidence.",
        0.55,
        fallback_baseline_a=0.72,
    ),
    96: PriorityMatch(
        96,
        "Switzerland",
        "Colombia",
        "Colombia favored if they advance: battle-tested, against "
        "Switzerland's clean but less demanding run.",
        0.60,
    ),
}


@dataclass(frozen=True)
class ResolvedPairing:
    match_number: int
    team_a: str
    team_b: str
    provisional: tuple[str, ...]  # teams whose feeder match is unconfirmed

    @property
    def label(self) -> str:
        return f"{self.team_a} vs {self.team_b}"


@dataclass
class ContextUpdate:
    """Everything the command computed, plus a transparent change log."""

    pairings: dict[int, ResolvedPairing]
    adjustments_yaml: dict
    expert_frame: pd.DataFrame
    form_frame: pd.DataFrame | None
    form_audit: pd.DataFrame | None
    changes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def resolve_r16_pairings(
    results: dict[int, KnockoutResult],
    bracket_matches: dict[int, MatchSlot],
    priority: dict[int, PriorityMatch] | None = None,
) -> tuple[dict[int, ResolvedPairing], list[str]]:
    """Resolve R16 pairings from confirmed R32 winners.

    A slot whose feeder match is confirmed uses the real winner. An
    unconfirmed slot falls back to the priority table's expected team
    (flagged provisional); with no fallback the pairing is skipped.
    """
    priority = R16_PRIORITY if priority is None else priority
    warnings: list[str] = []
    pairings: dict[int, ResolvedPairing] = {}
    for number, slot in sorted(bracket_matches.items()):
        if slot.round != "round_of_16":
            continue
        expected = priority.get(number)
        teams: list[str | None] = []
        provisional: list[str] = []
        for index, symbolic in enumerate(slot.entrants):
            if not (symbolic and symbolic[0] == "W" and symbolic[1:].isdigit()):
                warnings.append(
                    f"M{number}: unexpected R16 entrant slot {symbolic!r}; skipped"
                )
                teams.append(None)
                continue
            feeder = int(symbolic[1:])
            result = results.get(feeder)
            if result is not None:
                teams.append(result.winner)
            else:
                fallback = (
                    (expected.team_a, expected.team_b)[index] if expected else None
                )
                if fallback is not None:
                    provisional.append(fallback)
                teams.append(fallback)
        if teams[0] is None or teams[1] is None:
            warnings.append(
                f"M{number}: feeder result(s) unconfirmed and no priority "
                "fallback; pairing skipped"
            )
            continue
        if expected is not None:
            confirmed = {t for t in (teams[0], teams[1]) if t not in provisional}
            unexpected = confirmed - {expected.team_a, expected.team_b}
            if unexpected:
                warnings.append(
                    f"M{number}: confirmed team(s) {', '.join(sorted(unexpected))} "
                    f"differ from the priority context "
                    f"({expected.team_a} vs {expected.team_b}); "
                    "confirmed results take precedence"
                )
        pairings[number] = ResolvedPairing(
            match_number=number,
            team_a=str(teams[0]),
            team_b=str(teams[1]),
            provisional=tuple(provisional),
        )
    return pairings, warnings


def _tag_entries(
    pairing: ResolvedPairing, tags: list[PerformanceTag]
) -> list[dict]:
    entries: list[dict] = []
    for team in (pairing.team_a, pairing.team_b):
        for tag in tags:
            if tag.team != team or tag.match_number >= pairing.match_number:
                continue
            entry = {
                "team": team,
                "category": tag.category,
                "points": tag.points,
                "confidence": "medium",
                "reason": f"[{tag.tag} M{tag.match_number}] {tag.reason}",
            }
            if tag.modifier is not None:
                entry["modifier"] = tag.modifier
            entries.append(entry)
    return entries


def build_adjustment_blocks(
    pairings: dict[int, ResolvedPairing],
    tags: list[PerformanceTag],
    priority: dict[int, PriorityMatch] | None = None,
) -> dict[int, dict]:
    """One YAML match block per resolved R16 pairing (tags + priority extras)."""
    priority = R16_PRIORITY if priority is None else priority
    blocks: dict[int, dict] = {}
    for number, pairing in pairings.items():
        entries = _tag_entries(pairing, tags)
        expected = priority.get(number)
        if expected is not None:
            for extra in expected.extras:
                if extra.team not in (pairing.team_a, pairing.team_b):
                    continue
                entry = {
                    "team": extra.team,
                    "category": extra.category,
                    "points": extra.points,
                    "confidence": extra.confidence,
                    "reason": extra.reason,
                }
                if extra.modifier is not None:
                    entry["modifier"] = extra.modifier
                entries.append(entry)
        if not entries:
            continue
        label = pairing.label
        if pairing.provisional:
            label += (
                " (provisional: "
                + ", ".join(pairing.provisional)
                + " pending feeder result)"
            )
        blocks[number] = {"label": label, "adjustments": entries}
    return blocks


def _load_matchup_baselines(path: str | Path) -> dict[tuple[str, str], float]:
    p = resolve(path)
    if not p.exists():
        return {}
    frame = pd.read_csv(p)
    needed = {"team_a", "team_b", "historical_team_a_advances"}
    if not needed <= set(frame.columns):
        return {}
    baselines: dict[tuple[str, str], float] = {}
    for row in frame.itertuples(index=False):
        value = getattr(row, "historical_team_a_advances", None)
        if value is None or pd.isna(value):
            continue
        a, b = str(row.team_a), str(row.team_b)
        baselines[(a, b)] = float(value)
        baselines[(b, a)] = 1.0 - float(value)
    return baselines


def _team_shift(
    team: str,
    pairing: ResolvedPairing,
    tags: list[PerformanceTag],
    expected: PriorityMatch | None,
) -> float:
    shift = tag_nudge(tags, team, pairing.match_number, cap=DEFAULT_TAG_NUDGE_CAP).points
    if expected is not None:
        shift += sum(e.points for e in expected.extras if e.team == team)
    return shift


def build_expert_rows(
    pairings: dict[int, ResolvedPairing],
    tags: list[PerformanceTag],
    baselines: dict[tuple[str, str], float],
    priority: dict[int, PriorityMatch] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Advance-probability rows: matchup baseline + bounded context shift."""
    priority = R16_PRIORITY if priority is None else priority
    warnings: list[str] = []
    rows: list[dict] = []
    for number, pairing in pairings.items():
        expected = priority.get(number)
        key = (pairing.team_a, pairing.team_b)
        baseline = baselines.get(key)
        baseline_note = ""
        if baseline is None and expected is not None and expected.fallback_baseline_a:
            baseline = expected.fallback_baseline_a
            baseline_note = (
                " Baseline is a priority-table prior (pairing absent from "
                "matchup baselines)."
            )
        if baseline is None:
            baseline = 0.5
            baseline_note = " Baseline defaulted to 0.5 (no matchup baseline)."
            warnings.append(
                f"M{number}: no matchup baseline for {pairing.label}; used 0.5"
            )
        shift_a = _team_shift(pairing.team_a, pairing, tags, expected)
        shift_b = _team_shift(pairing.team_b, pairing, tags, expected)
        delta = max(
            -EXPERT_MAX_SHIFT_PCT, min(EXPERT_MAX_SHIFT_PCT, shift_a - shift_b)
        )
        p_a = max(EXPERT_MIN_PROB, min(EXPERT_MAX_PROB, baseline + delta / 100.0))
        reasoning = (
            (expected.summary if expected else "Derived from performance tags.")
            + f" Baseline {baseline:.2f} for {pairing.team_a}; net context shift "
            f"{delta:+g} pct pts (tags + priority context, bounded)."
            + baseline_note
        )
        if pairing.provisional:
            reasoning += (
                " Provisional: "
                + ", ".join(pairing.provisional)
                + " not yet confirmed by a feeder result."
            )
        rows.append({
            "match_id": f"M{number}",
            "team_a": pairing.team_a,
            "team_b": pairing.team_b,
            "source_model": EXPERT_SOURCE_MODEL,
            "team_a_win_prob": "",
            "draw_prob": "",
            "team_b_win_prob": "",
            "team_a_advance_prob": round(p_a, 3),
            "team_b_advance_prob": round(1.0 - p_a, 3),
            "confidence": expected.expert_confidence if expected else 0.5,
            "reasoning": reasoning,
        })
    return pd.DataFrame(rows, columns=list(EXPERT_COLUMNS)), warnings


def merge_expert_frame(
    existing_path: str | Path, generated: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Replace prior generated rows (same source_model), keep everything else."""
    p = resolve(existing_path)
    if p.exists():
        existing = pd.read_csv(p, dtype=str).fillna("")
        for column in EXPERT_COLUMNS:
            if column not in existing.columns:
                existing[column] = ""
        existing = existing[list(EXPERT_COLUMNS)]
        kept = existing[existing["source_model"] != EXPERT_SOURCE_MODEL]
        replaced = len(existing) - len(kept)
    else:
        kept = pd.DataFrame(columns=list(EXPERT_COLUMNS))
        replaced = 0
    merged = pd.concat([kept, generated.astype(str)], ignore_index=True)
    return merged, replaced


def build_recent_form_update(
    base_frame: pd.DataFrame, tags: list[PerformanceTag]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply bounded tag-derived form deltas to the base snapshot.

    Returns (updated form frame, audit frame, warnings). Teams with tags but
    no base row are skipped — a base value is never invented.
    """
    warnings: list[str] = []
    updated = base_frame.copy()
    audit_rows: list[dict] = []
    teams_in_file = set(updated["team"].astype(str))
    tagged_teams = sorted({t.team for t in tags})
    for team in tagged_teams:
        nudge = tag_nudge(tags, team, 105, cap=FORM_DELTA_CAP)
        if not nudge.tags or nudge.points == 0:
            continue
        if team not in teams_in_file:
            warnings.append(
                f"recent form: no base row for {team!r}; tag-derived delta "
                "skipped (add a base row to include it)"
            )
            continue
        mask = updated["team"].astype(str) == team
        for column in FORM_COLUMNS_ADJUSTED:
            if column not in updated.columns:
                continue
            base_value = float(updated.loc[mask, column].iloc[0])
            delta = nudge.points / 100.0
            new_value = round(base_value + delta, 4)
            updated.loc[mask, column] = new_value
            audit_rows.append({
                "team": team,
                "column": column,
                "base_value": base_value,
                "delta": round(delta, 4),
                "updated_value": new_value,
                "reasons": nudge.reasons(),
            })
    audit = pd.DataFrame(
        audit_rows,
        columns=["team", "column", "base_value", "delta", "updated_value", "reasons"],
    )
    return updated, audit, warnings


def merge_adjustments_yaml(
    existing_path: str | Path, blocks: dict[int, dict]
) -> tuple[dict, list[int], list[int]]:
    """Merge generated R16 blocks into the existing config mapping.

    Returns (merged raw mapping, added match numbers, replaced match numbers).
    Blocks for matches outside the generated set are preserved verbatim.
    """
    p = resolve(existing_path)
    raw = (
        yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None
    ) or {}
    raw.setdefault(
        "global",
        {
            "max_total_adjustment_pct": 15,
            "max_single_adjustment_pct": 10,
            "min_probability": 0.05,
            "max_probability": 0.95,
        },
    )
    matches = {int(k): v for k, v in (raw.get("matches") or {}).items()}
    added = sorted(n for n in blocks if n not in matches)
    replaced = sorted(n for n in blocks if n in matches)
    matches.update(blocks)
    raw["matches"] = {number: matches[number] for number in sorted(matches)}
    return raw, added, replaced


YAML_HEADER = """\
# Winner-only human adjustment layer for the 2026 knockout bracket.
#
# MANAGED FILE: the round-of-16 match blocks (M89-M96) are regenerated by
#   goalsignal tournament update-human-context
# from data/manual/knockout_results_2026.csv (confirmed results) and
# data/manual/knockout_performance_tags.csv (performance tags). Blocks for
# other matches are preserved verbatim; hand-written comments are not.
#
# Units: `points` are percentage points added to (positive) or removed from
# (negative) that team's advance probability for that match. Per-team sums
# and the two-team net difference are capped at `max_total_adjustment_pct`,
# and the adjusted probability is clipped into
# [min_probability, max_probability]. Every entry carries its reason;
# `confidence` (low|medium|high) is annotation only and never scales points.
#
# This file holds opinions, not model output. Nothing here touches the
# deployed model, the prediction ledger, or existing simulation artifacts.
"""


def render_adjustments_yaml(raw: dict) -> str:
    body = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=88)
    return YAML_HEADER + "\n" + body


def update_human_context(
    *,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    tags_path: str | Path = DEFAULT_TAGS_PATH,
    matchups_path: str | Path = DEFAULT_MATCHUPS_PATH,
    recent_form_path: str | Path = DEFAULT_RECENT_FORM_PATH,
    recent_form_base_path: str | Path = DEFAULT_RECENT_FORM_BASE_PATH,
    recent_form_audit_path: str | Path = DEFAULT_RECENT_FORM_AUDIT_PATH,
    expert_path: str | Path = DEFAULT_EXPERT_PATH,
    adjustments_path: str | Path = DEFAULT_ADJUSTMENTS_PATH,
    bracket_matches: dict[int, MatchSlot] | None = None,
    priority: dict[int, PriorityMatch] | None = None,
    force: bool = False,
) -> ContextUpdate:
    """Compute and write the regenerated human-context files.

    Refuses to overwrite existing targets unless ``force`` is set. The base
    form snapshot (``recent_form_base.csv``) is created once from the current
    form file and reused afterwards, so re-runs never compound deltas.
    """
    results = load_knockout_results(results_path)
    tags = load_performance_tags(tags_path)
    if bracket_matches is None:
        bracket_matches = OfficialBracket.load().matches
    update = _compute_update(
        results, tags, bracket_matches, matchups_path,
        recent_form_path, recent_form_base_path, expert_path,
        adjustments_path, priority,
    )
    if not results:
        update.warnings.append(
            f"no confirmed results found at {results_path}; pairings are "
            "provisional where a priority fallback exists"
        )
    _write_update(
        update,
        recent_form_path=recent_form_path,
        recent_form_base_path=recent_form_base_path,
        recent_form_audit_path=recent_form_audit_path,
        expert_path=expert_path,
        adjustments_path=adjustments_path,
        force=force,
    )
    return update


def _compute_update(
    results: dict[int, KnockoutResult],
    tags: list[PerformanceTag],
    bracket_matches: dict[int, MatchSlot],
    matchups_path: str | Path,
    recent_form_path: str | Path,
    recent_form_base_path: str | Path,
    expert_path: str | Path,
    adjustments_path: str | Path,
    priority: dict[int, PriorityMatch] | None,
) -> ContextUpdate:
    pairings, warnings = resolve_r16_pairings(results, bracket_matches, priority)
    blocks = build_adjustment_blocks(pairings, tags, priority)
    merged_yaml, added, replaced = merge_adjustments_yaml(adjustments_path, blocks)
    baselines = _load_matchup_baselines(matchups_path)
    expert_generated, expert_warnings = build_expert_rows(
        pairings, tags, baselines, priority
    )
    warnings.extend(expert_warnings)
    expert_frame, expert_replaced = merge_expert_frame(expert_path, expert_generated)

    base_path = resolve(recent_form_base_path)
    form_path = resolve(recent_form_path)
    form_frame = form_audit = None
    form_warnings: list[str] = []
    source = base_path if base_path.exists() else form_path
    if source.exists():
        base_frame = pd.read_csv(source)
        if "team" in base_frame.columns:
            form_frame, form_audit, form_warnings = build_recent_form_update(
                base_frame, tags
            )
        else:
            warnings.append(
                f"recent form file {source} has no 'team' column; skipped"
            )
    else:
        warnings.append(
            f"no recent form file found at {form_path}; form update skipped"
        )
    warnings.extend(form_warnings)

    changes = [
        "R16 pairings resolved: "
        + "; ".join(
            f"M{n} {p.label}"
            + (f" [provisional: {', '.join(p.provisional)}]" if p.provisional else "")
            for n, p in sorted(pairings.items())
        ),
        f"human adjustments: {len(blocks)} R16 match block(s) generated "
        f"({len(added)} added: {added}; {len(replaced)} replaced: {replaced})",
        f"expert predictions: {len(expert_generated)} row(s) generated under "
        f"source_model={EXPERT_SOURCE_MODEL!r} ({expert_replaced} prior "
        "generated row(s) replaced)",
    ]
    if form_audit is not None:
        adjusted_teams = sorted(set(form_audit["team"])) if len(form_audit) else []
        changes.append(
            f"recent form: {len(adjusted_teams)} team(s) adjusted "
            f"({', '.join(adjusted_teams) if adjusted_teams else 'none'}), "
            f"deltas capped at ±{FORM_DELTA_CAP:g} pct pts"
        )
    return ContextUpdate(
        pairings=pairings,
        adjustments_yaml=merged_yaml,
        expert_frame=expert_frame,
        form_frame=form_frame,
        form_audit=form_audit,
        changes=changes,
        warnings=warnings,
    )


def _write_update(
    update: ContextUpdate,
    *,
    recent_form_path: str | Path,
    recent_form_base_path: str | Path,
    recent_form_audit_path: str | Path,
    expert_path: str | Path,
    adjustments_path: str | Path,
    force: bool,
) -> None:
    targets: dict[str, Path] = {"adjustments": resolve(adjustments_path)}
    targets["expert"] = resolve(expert_path)
    if update.form_frame is not None:
        targets["recent_form"] = resolve(recent_form_path)
        targets["recent_form_audit"] = resolve(recent_form_audit_path)
    existing = [str(p) for p in targets.values() if p.exists()]
    if existing and not force:
        raise FileExistsError(
            "human-context targets already exist (pass --force to update): "
            + ", ".join(existing)
        )
    # Validate the merged YAML through the strict config loader before
    # replacing the real file.
    yaml_text = render_adjustments_yaml(update.adjustments_yaml)
    tmp = targets["adjustments"].with_suffix(".yaml.tmp")
    tmp.write_text(yaml_text, encoding="utf-8")
    try:
        HumanAdjustmentsConfig.load(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    base_path = resolve(recent_form_base_path)
    form_path = resolve(recent_form_path)
    if update.form_frame is not None and not base_path.exists() and form_path.exists():
        # Preserve the pre-update form file once so re-runs are idempotent.
        base_path.write_text(form_path.read_text(encoding="utf-8"), encoding="utf-8")
        update.changes.append(f"recent form base snapshot created: {base_path}")
    targets["adjustments"].write_text(yaml_text, encoding="utf-8")
    update.expert_frame.to_csv(targets["expert"], index=False)
    if update.form_frame is not None:
        update.form_frame.to_csv(targets["recent_form"], index=False)
        update.form_audit.to_csv(targets["recent_form_audit"], index=False)
    update.changes.append(
        "written: " + ", ".join(str(p) for p in targets.values())
    )
