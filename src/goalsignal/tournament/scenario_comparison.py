"""Scenario comparison report across three knockout bracket views.

Compares, match by match, up to three scenarios over the official 2026
knockout schedule:

1. **model-only** — the modal bracket of a historical (default goal model)
   simulation run;
2. **knockout_survival** — the modal bracket of the knockout-survival ensemble
   run, when one exists;
3. **human-adjusted scenario** — the opinion overlay written by
   ``goalsignal tournament human-adjust`` (``human_adjusted_bracket.csv``),
   when it exists.

Everything here is read-only presentation: no simulator is re-run, no model
probability is changed, and no original artifact is modified. The human layer
is a *scenario analysis layer* — an opinion overlay for stress-testing
tactical views — not a calibrated forecast, and every generated report says
so explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from goalsignal.tournament.bracket_2026 import MatchSlot
from goalsignal.tournament.human_adjustments import CSV_NAME as HUMAN_CSV_NAME
from goalsignal.tournament.human_adjustments import META_NAME as HUMAN_META_NAME

MD_NAME = "scenario_comparison.md"
CSV_NAME = "scenario_comparison.csv"
MOVERS_NAME = "scenario_biggest_movers.csv"
FLIPS_NAME = "scenario_flips.csv"

FINAL_MATCH_NUMBER = 104

SCENARIO_LANGUAGE = (
    "Human adjustments are **scenario analysis, not calibrated forecasts**. "
    "The model probabilities remain unchanged; the prediction ledger and the "
    "original simulation artifacts are untouched. Adjusted probabilities rank "
    "one fixed bracket path under stated opinions. The scenario analysis "
    "layer is useful for stress-testing tactical opinions, not for claiming "
    "improved accuracy."
)


# --------------------------------------------------------------------------- #
# Scenario loading (all read-only, all tolerant of missing artifacts).
# --------------------------------------------------------------------------- #


@dataclass
class ModalScenario:
    """The modal bracket of one simulation run, or a recorded absence."""

    label: str
    path: Path | None
    modal: dict[int, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.modal)

    def winner(self, number: int) -> str | None:
        entry = self.modal.get(number)
        return entry.get("modal_conditional_winner") if entry else None

    def pairing(self, number: int) -> tuple[str, str] | None:
        entry = self.modal.get(number)
        pair = entry.get("modal_matchup") if entry else None
        return (str(pair[0]), str(pair[1])) if pair and len(pair) == 2 else None

    def winner_probability(self, number: int) -> float | None:
        entry = self.modal.get(number)
        value = entry.get("conditional_win_probability") if entry else None
        return float(value) if value is not None else None

    @property
    def champion(self) -> str | None:
        return self.winner(FINAL_MATCH_NUMBER)


@dataclass
class HumanScenario:
    """The human-adjusted scenario CSV of one run, or a recorded absence."""

    path: Path | None
    frame: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.frame is not None and not self.frame.empty

    def row(self, number: int) -> pd.Series | None:
        if not self.available:
            return None
        rows = self.frame[self.frame["match_number"] == number]
        return rows.iloc[0] if len(rows) else None

    @property
    def champion(self) -> str | None:
        row = self.row(FINAL_MATCH_NUMBER)
        return str(row["predicted_winner"]) if row is not None else None


def load_modal_scenario(label: str, sim_dir: str | Path | None) -> ModalScenario:
    """Load a run's modal bracket; a missing/unreadable run stays unavailable."""
    if sim_dir is None:
        return ModalScenario(label=label, path=None)
    path = Path(sim_dir)
    scenario = ModalScenario(label=label, path=path)
    bracket_path = path / "wc2026_bracket.json"
    meta_path = path / "wc2026_tournament_meta.json"
    if bracket_path.exists():
        try:
            data = json.loads(bracket_path.read_text(encoding="utf-8"))
            scenario.modal = {
                int(m["match_number"]): m for m in data.get("matches", [])
            }
        except Exception:  # pragma: no cover - defensive
            scenario.modal = {}
    if meta_path.exists():
        try:
            scenario.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive
            scenario.meta = {}
    return scenario


def load_human_scenario(sim_dir: str | Path) -> HumanScenario:
    """Load ``human_adjusted_bracket.csv`` from a run, if it exists."""
    path = Path(sim_dir)
    csv_path = path / HUMAN_CSV_NAME
    scenario = HumanScenario(path=csv_path)
    if csv_path.exists():
        try:
            scenario.frame = pd.read_csv(csv_path)
        except Exception:  # pragma: no cover - defensive
            scenario.frame = None
    meta_path = path / HUMAN_META_NAME
    if meta_path.exists():
        try:
            scenario.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive
            scenario.meta = {}
    return scenario


def _optional(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row or pd.isna(row[column]):
        return ""
    return str(row[column])


# --------------------------------------------------------------------------- #
# Per-match comparison.
# --------------------------------------------------------------------------- #


def comparison_frame(
    model_only: ModalScenario,
    knockout_survival: ModalScenario,
    human: HumanScenario,
) -> pd.DataFrame:
    """One row per knockout match across every available scenario."""
    numbers: set[int] = set(model_only.modal) | set(knockout_survival.modal)
    if human.available:
        numbers |= {int(n) for n in human.frame["match_number"]}
    rows = []
    for number in sorted(numbers):
        human_row = human.row(number)
        pairing = None
        stage = ""
        if human_row is not None:
            pairing = (str(human_row["team_1"]), str(human_row["team_2"]))
            stage = str(human_row["round"])
        else:
            for scenario in (knockout_survival, model_only):
                if scenario.pairing(number) is not None:
                    pairing = scenario.pairing(number)
                    stage = str(scenario.modal[number].get("round", ""))
                    break
        if pairing is None:
            continue
        flipped = (
            bool(human_row["winner_changed"]) if human_row is not None else False
        )
        rows.append({
            "match_number": number,
            "stage": stage,
            "team_a": pairing[0],
            "team_b": pairing[1],
            "model_only_winner": model_only.winner(number) or "",
            "knockout_survival_winner": knockout_survival.winner(number) or "",
            "human_adjusted_winner": _optional(human_row, "predicted_winner"),
            "baseline_advance_probability": _optional(
                human_row, "baseline_p_team_1"
            ),
            "human_adjusted_probability": _optional(
                human_row, "adjusted_p_team_1"
            ),
            "net_adjustment_points": _optional(human_row, "applied_delta_pct"),
            "flipped_by_opinion": flipped,
            "reason": _optional(human_row, "adjustment_reasons"),
            "confidence": _optional(human_row, "adjustment_confidences"),
            "provenance": " | ".join(
                part
                for part in (
                    _optional(human_row, "baseline_source"),
                    _optional(human_row, "notes"),
                )
                if part
            ),
        })
    return pd.DataFrame(rows)


def flips_frame(comparison: pd.DataFrame) -> pd.DataFrame:
    """Matches whose predicted winner was flipped by the opinion overlay."""
    if comparison.empty:
        return comparison
    return comparison[comparison["flipped_by_opinion"]].reset_index(drop=True)


def biggest_movers_frame(
    comparison: pd.DataFrame,
    model_only: ModalScenario,
    knockout_survival: ModalScenario,
    top: int = 10,
) -> pd.DataFrame:
    """Largest per-match probability moves, tagged by what caused them.

    Two comparisons share one schema: the opinion overlay against its own
    simulated baseline (``human_adjusted vs baseline``), and the
    knockout-survival modal winner probability against the model-only run's
    for matches where both runs saw the same modal pairing
    (``knockout_survival vs model_only``).
    """
    rows = []
    if not comparison.empty:
        for row in comparison.itertuples(index=False):
            base = row.baseline_advance_probability
            adjusted = row.human_adjusted_probability
            if base == "" or adjusted == "":
                continue
            base_f, adjusted_f = float(base), float(adjusted)
            if base_f == adjusted_f:
                continue
            rows.append({
                "comparison": "human_adjusted vs baseline",
                "match_number": row.match_number,
                "stage": row.stage,
                "subject": f"{row.team_a} (vs {row.team_b})",
                "from_prob": base_f,
                "to_prob": adjusted_f,
                "delta": adjusted_f - base_f,
            })
    if model_only.available and knockout_survival.available:
        for number in sorted(set(model_only.modal) & set(knockout_survival.modal)):
            pair_a = model_only.pairing(number)
            pair_b = knockout_survival.pairing(number)
            if pair_a is None or pair_b is None or set(pair_a) != set(pair_b):
                continue
            winner = model_only.winner(number)
            p_model = model_only.winner_probability(number)
            p_ko = knockout_survival.winner_probability(number)
            if winner != knockout_survival.winner(number):
                p_ko = 1.0 - p_ko if p_ko is not None else None
            if p_model is None or p_ko is None or p_model == p_ko:
                continue
            rows.append({
                "comparison": "knockout_survival vs model_only",
                "match_number": number,
                "stage": str(model_only.modal[number].get("round", "")),
                "subject": f"{winner} (vs "
                f"{pair_a[1] if winner == pair_a[0] else pair_a[0]})",
                "from_prob": p_model,
                "to_prob": p_ko,
                "delta": p_ko - p_model,
            })
    frame = pd.DataFrame(
        rows,
        columns=[
            "comparison", "match_number", "stage", "subject",
            "from_prob", "to_prob", "delta",
        ],
    )
    if frame.empty:
        return frame
    frame["abs_delta"] = frame["delta"].abs()
    frame = frame.sort_values("abs_delta", ascending=False).head(top)
    return frame.drop(columns="abs_delta").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Downstream effect tracing.
# --------------------------------------------------------------------------- #


@dataclass
class DownstreamTrace:
    """The bracket consequences of one opinion-driven flip."""

    match_number: int
    flipped_to: str
    flipped_from: str
    effects: list[str]
    champion_changed: bool


def _children_graph(bracket_matches: dict[int, MatchSlot]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for number, slot in bracket_matches.items():
        for symbolic in slot.entrants:
            if symbolic and symbolic[0] in "WL" and symbolic[1:].isdigit():
                children.setdefault(int(symbolic[1:]), []).append(number)
    return children


def _unadjusted_pairing(row: pd.Series) -> tuple[str, str] | None:
    """The no-opinion walk pairing recorded in the human CSV, if present."""
    a = _optional(row, "unadjusted_team_1")
    b = _optional(row, "unadjusted_team_2")
    return (a, b) if a and b else None


def trace_downstream_effects(
    human: HumanScenario,
    reference: ModalScenario,
    bracket_matches: dict[int, MatchSlot],
) -> list[DownstreamTrace]:
    """Explain what each opinion-driven flip changed further down the bracket.

    For every flipped match, walks its descendants in the official graph and
    reports each descendant whose scenario pairing differs from what it would
    have been *without* the opinion overlay. The exact comparator is the
    unadjusted deterministic walk recorded in the human CSV; older CSVs
    without those columns fall back to the reference run's modal pairings
    (flagged, because modal chains can disagree independently of any flip).
    """
    if not human.available:
        return []
    children = _children_graph(bracket_matches)
    frame = human.frame.set_index("match_number")
    traces: list[DownstreamTrace] = []
    flipped = frame[frame["winner_changed"].astype(bool)]
    for number, row in flipped.iterrows():
        effects: list[str] = []
        stack = list(children.get(int(number), []))
        seen: set[int] = set()
        while stack:
            descendant = stack.pop(0)
            if descendant in seen:
                continue
            seen.add(descendant)
            stack.extend(children.get(descendant, []))
            if descendant not in frame.index:
                continue
            drow = frame.loc[descendant]
            scenario_pair = {str(drow["team_1"]), str(drow["team_2"])}
            unadjusted = _unadjusted_pairing(drow)
            if unadjusted is not None:
                if set(unadjusted) != scenario_pair:
                    effects.append(
                        f"M{descendant} ({drow['round']}) is now "
                        f"{drow['team_1']} vs {drow['team_2']} "
                        f"(was {unadjusted[0]} vs {unadjusted[1]} without the "
                        f"flip); scenario winner {drow['predicted_winner']}"
                    )
                continue
            reference_pair = reference.pairing(descendant)
            if reference_pair is not None and set(reference_pair) != scenario_pair:
                effects.append(
                    f"M{descendant} ({drow['round']}) is now "
                    f"{drow['team_1']} vs {drow['team_2']} "
                    f"(was {reference_pair[0]} vs {reference_pair[1]} in the "
                    f"run's modal bracket, which can differ independently of "
                    f"this flip); scenario winner {drow['predicted_winner']}"
                )
        champion_changed = _champion_changed(human, reference, frame)
        traces.append(DownstreamTrace(
            match_number=int(number),
            flipped_from=str(row["baseline_winner"]),
            flipped_to=str(row["predicted_winner"]),
            effects=effects,
            champion_changed=champion_changed,
        ))
    return traces


def _champion_changed(
    human: HumanScenario, reference: ModalScenario, frame: pd.DataFrame
) -> bool:
    """Did the overlay change the champion vs the no-opinion comparator?"""
    if not human.champion:
        return False
    if FINAL_MATCH_NUMBER in frame.index:
        unadjusted = _optional(frame.loc[FINAL_MATCH_NUMBER], "unadjusted_winner")
        if unadjusted:
            return human.champion != unadjusted
    reference_champion = reference.champion
    return bool(reference_champion and human.champion != reference_champion)


# --------------------------------------------------------------------------- #
# Report rendering + writing.
# --------------------------------------------------------------------------- #


def _fmt_prob(value: str | float) -> str:
    if value == "" or value is None:
        return "—"
    return f"{float(value):.3f}"


def render_markdown(
    comparison: pd.DataFrame,
    movers: pd.DataFrame,
    traces: list[DownstreamTrace],
    model_only: ModalScenario,
    knockout_survival: ModalScenario,
    human: HumanScenario,
) -> str:
    """A one-minute-readable Markdown comparison of the three scenarios."""
    flips = flips_frame(comparison)
    n_adjusted = 0
    if human.available and "n_adjustments_applied" in human.frame.columns:
        n_adjusted = int((human.frame["n_adjustments_applied"] > 0).sum())
    ensemble_disagreements = 0
    if not comparison.empty:
        both = comparison[
            (comparison["model_only_winner"] != "")
            & (comparison["knockout_survival_winner"] != "")
        ]
        ensemble_disagreements = int(
            (both["model_only_winner"] != both["knockout_survival_winner"]).sum()
        )
    champions = [
        ("Model-only (historical simulation)", model_only.champion),
        ("Knockout-survival ensemble", knockout_survival.champion),
        ("Human-adjusted scenario (opinion overlay)", human.champion),
    ]
    lines = [
        "# World Cup 2026 — scenario comparison",
        "",
        "Side-by-side view of the model-only bracket, the knockout-survival "
        "ensemble bracket, and the **human-adjusted scenario** (an opinion "
        "overlay on the simulated probabilities).",
        "",
        "## Headline",
        "",
        "- Champions: " + "; ".join(
            f"{label.split(' (')[0]} → **{champ or 'unavailable'}**"
            for label, champ in champions
        ),
        f"- Matches changed by the opinion overlay: {n_adjusted}",
        f"- Picks flipped by the opinion overlay: {len(flips)}",
        f"- Model-only vs knockout-survival modal-pick disagreements: "
        f"{ensemble_disagreements}",
    ]
    if not movers.empty:
        top = movers.iloc[0]
        lines.append(
            f"- Biggest probability mover: M{top.match_number} {top.subject} "
            f"{top.from_prob:.3f} → {top.to_prob:.3f} ({top.comparison})"
        )
    lines += ["", "## Scenario availability", ""]
    for label, scenario in (
        ("model-only", model_only),
        ("knockout_survival", knockout_survival),
    ):
        status = (
            f"available (`{scenario.path}`)"
            if scenario.available
            else "**unavailable** — this column is blank below"
        )
        lines.append(f"- {label}: {status}")
    lines.append(
        "- human-adjusted scenario: "
        + (
            f"available (`{human.path}`)"
            if human.available
            else "**unavailable** — run `goalsignal tournament human-adjust` first"
        )
    )
    if traces:
        lines += ["", "## Opinion-driven flips and downstream effects", ""]
        for trace in traces:
            lines.append(
                f"- **M{trace.match_number}: {trace.flipped_from} → "
                f"{trace.flipped_to}** (flipped by the opinion overlay)"
            )
            for effect in trace.effects:
                lines.append(f"  - {effect}")
            if not trace.effects:
                lines.append(
                    "  - no downstream pairing differs from the reference run"
                )
            lines.append(
                "  - champion "
                + (
                    f"changed under this scenario (now {human.champion})"
                    if trace.champion_changed
                    else "unchanged under this scenario"
                )
            )
    if not movers.empty:
        lines += [
            "",
            "## Biggest probability movers",
            "",
            "| M | Stage | Subject | From | To | Delta | Comparison |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in movers.itertuples(index=False):
            lines.append(
                f"| {row.match_number} | {row.stage} | {row.subject} | "
                f"{row.from_prob:.3f} | {row.to_prob:.3f} | {row.delta:+.3f} | "
                f"{row.comparison} |"
            )
    if not comparison.empty:
        lines += [
            "",
            "## Per-match comparison",
            "",
            "Winners per scenario; probabilities are for team A. "
            "Blank cells mean that scenario is unavailable for the match.",
            "",
            "| M | Stage | Pairing | Model-only | KO-survival | Human scenario | "
            "Base p(A) | Scenario p(A) | Net pts | Flip | Why |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in comparison.itertuples(index=False):
            net = (
                f"{float(row.net_adjustment_points):+.1f}"
                if row.net_adjustment_points != ""
                and float(row.net_adjustment_points) != 0.0
                else ""
            )
            # Reasons are "|"-joined in the CSV; a raw pipe would break the
            # Markdown table row, so re-join with semicolons for the cell.
            why = (
                row.reason.replace(" | ", "; ") if row.flipped_by_opinion else ""
            )
            lines.append(
                f"| {row.match_number} | {row.stage} | "
                f"{row.team_a} vs {row.team_b} | "
                f"{row.model_only_winner or '—'} | "
                f"{row.knockout_survival_winner or '—'} | "
                f"{row.human_adjusted_winner or '—'} | "
                f"{_fmt_prob(row.baseline_advance_probability)} | "
                f"{_fmt_prob(row.human_adjusted_probability)} | {net} | "
                f"{'YES' if row.flipped_by_opinion else ''} | {why} |"
            )
    lines += [
        "",
        "## Scenario, not forecast",
        "",
        SCENARIO_LANGUAGE,
        "",
        "## Caveats and limitations",
        "",
        "- Opinion points are human judgments configured in YAML "
        "(with hard caps and strict validation), not fitted coefficients.",
        "- Downstream pairings in the human-adjusted scenario assume every "
        "earlier scenario pick is correct.",
        "- Modal brackets summarize the single most likely path per match; "
        "they hide the full simulated distribution.",
        "- No accuracy improvement is claimed for the opinion overlay; use it "
        "to make tactical assumptions explicit and inspect their consequences.",
        "",
    ]
    return "\n".join(lines)


def write_scenario_comparison(
    comparison: pd.DataFrame,
    movers: pd.DataFrame,
    traces: list[DownstreamTrace],
    model_only: ModalScenario,
    knockout_survival: ModalScenario,
    human: HumanScenario,
    out_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Path]:
    """Write the Markdown report + three CSVs; refuse overwrite without force."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out / MD_NAME,
        "csv": out / CSV_NAME,
        "movers": out / MOVERS_NAME,
        "flips": out / FLIPS_NAME,
    }
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not force:
        raise FileExistsError(
            "scenario comparison artifacts already exist (pass --force to "
            "overwrite): " + ", ".join(existing)
        )
    comparison.to_csv(paths["csv"], index=False)
    movers.to_csv(paths["movers"], index=False)
    flips = flips_frame(comparison)
    if not flips.empty:
        flips = flips.copy()
        by_number = {t.match_number: t for t in traces}
        flips["downstream_effects"] = [
            " ; ".join(by_number[m].effects)
            if m in by_number and by_number[m].effects
            else ""
            for m in flips["match_number"]
        ]
    flips.to_csv(paths["flips"], index=False)
    paths["md"].write_text(
        render_markdown(
            comparison, movers, traces, model_only, knockout_survival, human
        ),
        encoding="utf-8",
    )
    return paths
