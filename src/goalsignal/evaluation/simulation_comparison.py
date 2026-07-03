"""Compare full-tournament simulations and explain the knockout survival layer.

Two read-only diagnostics, neither of which re-runs or overwrites the simulator:

1. **Team-level comparison** across three prediction sources — the historical
   baseline, the ``final_ensemble`` ensemble, and the ``knockout_survival``
   ensemble (with ``--include-knockout-upset``) — reading each run's
   ``wc2026_team_advancement.csv`` artifact and reporting how semifinal/final/
   champion probabilities moved.

2. **Matchup-level diagnostics** that re-derive each knockout tie's advance
   probability under every version and decompose *why* it changed: the
   ``knockout_upset`` internal shift, the net move it causes inside the blend,
   the regulation draw probability, expected goals, the penalty-path
   contribution, and the style/penalty provenance tags.

The report is intentionally honest: it explains *what changed*, never claims an
accuracy improvement, and surfaces which signals were missing or illustrative.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from goalsignal.signals.base import AdvanceProbs
from goalsignal.signals.knockout_upset import (
    PENALTY_PROVENANCE_TAGS,
    STYLE_PROVENANCE_TAGS,
    KnockoutUpsetParams,
    PenaltyTable,
    knockout_upset_detail,
)
from goalsignal.signals.meta_ensemble import MetaEnsemble
from goalsignal.signals.pipeline import (
    ManualInputs,
    MatchSpec,
    _base_advance,
    build_signals,
)
from goalsignal.utils.paths import resolve

ADVANCEMENT_FILE = "wc2026_team_advancement.csv"
META_FILE = "wc2026_tournament_meta.json"

STAGE_COLUMNS = (
    "p_round_of_32",
    "p_round_of_16",
    "p_quarterfinal",
    "p_semifinal",
    "p_final",
    "p_champion",
)
HEADLINE_STAGES = ("p_semifinal", "p_final", "p_champion")
LABELS = ("baseline", "final_ensemble", "knockout_survival")


# --------------------------------------------------------------------------- #
# Loading + discovery of simulation artifacts.
# --------------------------------------------------------------------------- #


@dataclass
class SimRun:
    """One simulation artifact directory, loaded defensively.

    ``available`` is ``False`` (and ``advancement`` is ``None``) when the
    directory is missing or unreadable, so a partial set of runs never crashes
    the comparison.
    """

    label: str
    path: Path | None
    advancement: pd.DataFrame | None = None
    meta: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.advancement is not None

    @property
    def n_sims(self) -> int | None:
        return self.meta.get("n_sims")


def load_sim_run(label: str, path: str | Path | None) -> SimRun:
    """Load a simulation directory; tolerant of a missing/partial artifact."""
    if path is None:
        return SimRun(label=label, path=None)
    p = Path(path)
    adv_path = p / ADVANCEMENT_FILE
    meta_path = p / META_FILE
    advancement = None
    meta: dict = {}
    if adv_path.exists():
        try:
            advancement = pd.read_csv(adv_path)
        except Exception:  # pragma: no cover - defensive
            advancement = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive
            meta = {}
    return SimRun(label=label, path=p, advancement=advancement, meta=meta)


def classify_meta(meta: dict) -> str | None:
    """Map a run's metadata to a comparison label, or ``None`` if it is neither."""
    source = meta.get("prediction_source")
    if source == "historical":
        return "baseline"
    if source == "ensemble":
        version = meta.get("ensemble_version")
        if version == "final_ensemble" and not meta.get("include_knockout_upset"):
            return "final_ensemble"
        if version == "knockout_survival":
            return "knockout_survival"
    return None


def discover_sim_runs(sim_root: str | Path = "artifacts/simulations") -> dict[str, Path]:
    """Find the newest artifact directory for each comparison label.

    Scans ``sim_root`` for tournament runs and classifies each by its metadata.
    When several match a label, the most recently modified wins.
    """
    root = resolve(sim_root)
    found: dict[str, tuple[float, Path]] = {}
    if not root.exists():
        return {}
    for meta_path in root.glob(f"*/{META_FILE}"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive
            continue
        label = classify_meta(meta)
        if label is None:
            continue
        mtime = meta_path.stat().st_mtime
        if label not in found or mtime > found[label][0]:
            found[label] = (mtime, meta_path.parent)
    return {label: path for label, (_, path) in found.items()}


# --------------------------------------------------------------------------- #
# Team-level comparison.
# --------------------------------------------------------------------------- #


def team_comparison(runs: dict[str, SimRun]) -> pd.DataFrame:
    """Per-team headline-stage probabilities across the available runs + deltas.

    Returns an empty frame when no run has advancement data. Deltas are computed
    only for label pairs that are both present.
    """
    present = [lbl for lbl in LABELS if lbl in runs and runs[lbl].available]
    if not present:
        return pd.DataFrame()
    merged: pd.DataFrame | None = None
    for label in present:
        df = runs[label].advancement
        cols = ["team"] + [c for c in STAGE_COLUMNS if c in df.columns]
        sub = df[cols].copy()
        if "group" in df.columns and merged is None:
            sub.insert(0, "group", df["group"])
        rename = {c: f"{label}__{c}" for c in STAGE_COLUMNS if c in df.columns}
        sub = sub.rename(columns=rename)
        merged = sub if merged is None else merged.merge(sub, on="team", how="outer")
    # Deltas for the headline stages across the meaningful comparisons.
    comparisons = [
        ("final_ensemble", "baseline"),
        ("knockout_survival", "baseline"),
        ("knockout_survival", "final_ensemble"),
    ]
    for new, old in comparisons:
        if new in present and old in present:
            for stage in HEADLINE_STAGES:
                a, b = f"{new}__{stage}", f"{old}__{stage}"
                if a in merged.columns and b in merged.columns:
                    merged[f"delta__{new}_vs_{old}__{stage}"] = merged[a] - merged[b]
    sort_key = next(
        (c for c in (
            "knockout_survival__p_champion",
            "final_ensemble__p_champion",
            "baseline__p_champion",
        ) if c in merged.columns),
        None,
    )
    if sort_key is not None:
        merged = merged.sort_values(sort_key, ascending=False)
    return merged.reset_index(drop=True)


def biggest_movers(comparison: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Teams whose headline probabilities moved most between versions.

    One row per (team, stage, comparison) with the absolute and signed delta,
    sorted by absolute move. Required columns: ``team, stage, comparison,
    delta, abs_delta``. Empty in, empty out.
    """
    if comparison.empty:
        return pd.DataFrame(
            columns=["team", "stage", "comparison", "from_prob", "to_prob", "delta", "abs_delta"]
        )
    rows = []
    delta_cols = [c for c in comparison.columns if c.startswith("delta__")]
    for _, r in comparison.iterrows():
        for col in delta_cols:
            # delta__<new>_vs_<old>__<stage>
            body = col[len("delta__"):]
            pair, stage = body.rsplit("__", 1)
            new, old = pair.split("_vs_")
            delta = r[col]
            if pd.isna(delta):
                continue
            rows.append({
                "team": r["team"],
                "stage": stage,
                "comparison": f"{new}_vs_{old}",
                "from_prob": r.get(f"{old}__{stage}"),
                "to_prob": r.get(f"{new}__{stage}"),
                "delta": float(delta),
                "abs_delta": abs(float(delta)),
            })
    movers = pd.DataFrame(rows)
    if movers.empty:
        return movers
    return movers.sort_values("abs_delta", ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Matchup-level diagnostics (before/after + provenance).
# --------------------------------------------------------------------------- #


def _historical_advance(
    spec: MatchSpec, historical
) -> tuple[MatchSpec, AdvanceProbs | None]:
    """Fill the spec's knockout historical advance from a live provider if given."""
    if historical is None:
        return spec, spec.historical if isinstance(spec.historical, AdvanceProbs) else None
    sig = historical.advance(spec.team_a, spec.team_b, spec.neutral)
    if sig.advance is not None:
        from dataclasses import replace

        return replace(spec, historical=sig.advance), sig.advance
    base = spec.historical if isinstance(spec.historical, AdvanceProbs) else None
    return spec, base


def matchup_diagnostics(
    specs: list[MatchSpec],
    inputs_plain: ManualInputs,
    inputs_upset: ManualInputs,
    ensemble: MetaEnsemble,
    *,
    historical=None,
    params: KnockoutUpsetParams | None = None,
) -> pd.DataFrame:
    """Before/after advance probabilities + a knockout_upset decomposition.

    ``inputs_plain`` has the survival signal off; ``inputs_upset`` has it on.
    Only knockout specs are diagnosed. ``historical`` (optional) is a
    :class:`~goalsignal.signals.historical_adapter.LiveModelHistorical` used to
    fill the baseline advance; without it the spec's own historical advance is
    used.
    """
    p = params or inputs_upset.knockout_upset_params
    rows = []
    for spec in specs:
        if not spec.knockout:
            continue
        spec2, base_hist = _historical_advance(spec, historical)
        signals_plain = build_signals(spec2, inputs_plain)
        signals_upset = build_signals(spec2, inputs_upset)
        final = ensemble.blend_advance(signals_plain, version="final_ensemble")
        ks_plain = ensemble.blend_advance(signals_plain, version="knockout_survival")
        ks_upset = ensemble.blend_advance(signals_upset, version="knockout_survival")
        anchor = _base_advance(spec2, signals_plain)
        detail = knockout_upset_detail(
            spec2.team_a, spec2.team_b, base_advance=anchor,
            styles=inputs_upset.styles, penalties=inputs_upset.penalties, params=p,
        )
        no_pen = knockout_upset_detail(
            spec2.team_a, spec2.team_b, base_advance=anchor,
            styles=inputs_upset.styles, penalties=PenaltyTable({}), params=p,
        )
        # Sign the internal shift as a team_a delta (shift is toward the underdog).
        und_is_a = detail.underdog == spec2.team_a
        internal_a = detail.shift if und_is_a else -detail.shift
        pen_contrib = detail.shift - no_pen.shift  # toward the underdog
        style_tags = [t for t in detail.paths if t in STYLE_PROVENANCE_TAGS]
        pen_tags = [t for t in detail.paths if t in PENALTY_PROVENANCE_TAGS]
        rows.append({
            "match_id": spec2.match_id,
            "team_a": spec2.team_a,
            "team_b": spec2.team_b,
            "stage": spec2.stage,
            "baseline_team_a_advances": _p(base_hist),
            "final_ensemble_team_a_advances": round(final.probs.team_a_advances, 4),
            "knockout_survival_team_a_advances": round(ks_upset.probs.team_a_advances, 4),
            "delta_from_final": round(
                ks_upset.probs.team_a_advances - final.probs.team_a_advances, 4
            ),
            "knockout_upset_internal_shift": round(internal_a, 4),
            "net_move_from_upset": round(
                ks_upset.probs.team_a_advances - ks_plain.probs.team_a_advances, 4
            ),
            "favorite": detail.favorite,
            "underdog": detail.underdog,
            "draw_prob": detail.detail.get("regulation_draw_prob"),
            "expected_goals_total": detail.detail.get("expected_goals_total"),
            "shootout_fav_prob": detail.detail.get("shootout_fav_prob"),
            "penalty_path_contribution": round(pen_contrib, 4),
            "style_tags": "|".join(style_tags),
            "penalty_tags": "|".join(pen_tags),
            "provenance_tags": "|".join(detail.paths),
            "flagged_high_disagreement": ensemble.is_flagged(ks_upset),
            "missing_signals": "|".join(ks_upset.missing),
        })
    return pd.DataFrame(rows)


def _p(probs: AdvanceProbs | None) -> float | None:
    return round(probs.team_a_advances, 4) if probs is not None else None


# --------------------------------------------------------------------------- #
# Report assembly.
# --------------------------------------------------------------------------- #


def _flagged_from_meta(runs: dict[str, SimRun]) -> list[tuple]:
    """Flagged high-disagreement matchups from the richest available run meta."""
    for label in ("knockout_survival", "final_ensemble"):
        run = runs.get(label)
        if run is None:
            continue
        prov = run.meta.get("ensemble_provenance") or {}
        flagged = prov.get("flagged_matchups")
        if flagged:
            return flagged
    return []


def _missing_from_meta(runs: dict[str, SimRun]) -> dict:
    for label in ("knockout_survival", "final_ensemble"):
        run = runs.get(label)
        if run is None:
            continue
        prov = run.meta.get("ensemble_provenance") or {}
        if prov.get("missing_signal_counts") is not None:
            return prov["missing_signal_counts"]
    return {}


def render_markdown(
    runs: dict[str, SimRun],
    comparison: pd.DataFrame,
    movers: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    diagnostics_illustrative: bool,
) -> str:
    """Render the honest, non-overclaiming comparison report."""
    lines: list[str] = ["# Simulation comparison: historical vs ensemble vs knockout survival", ""]

    # --- run inventory --------------------------------------------------------
    lines.append("## Runs compared")
    lines.append("")
    lines.append("| Label | Found | n_sims | Source |")
    lines.append("| --- | --- | --- | --- |")
    for label in LABELS:
        run = runs.get(label)
        if run is not None and run.available:
            src = run.meta.get("ensemble_version") or run.meta.get("prediction_source", "?")
            lines.append(f"| `{label}` | yes (`{run.path.name}`) | {run.n_sims} | {src} |")
        else:
            lines.append(f"| `{label}` | **missing** | — | — |")
    lines.append("")
    missing_labels = [lbl for lbl in LABELS if lbl not in runs or not runs[lbl].available]
    if missing_labels:
        lines.append(
            "> Some runs are missing, so their comparisons are omitted. Generate them with "
            "`goalsignal tournament simulate` (see the report footer)."
        )
        lines.append("")

    # --- headline movers ------------------------------------------------------
    lines.append("## Biggest probability movers")
    lines.append("")
    if movers.empty:
        lines.append("_No two runs were both present, so no deltas were computed._")
    else:
        lines.append("Largest absolute moves across all available version pairs:")
        lines.append("")
        lines.append("| Team | Stage | Comparison | From | To | Δ |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for _, r in movers.iterrows():
            lines.append(
                f"| {r['team']} | {r['stage'].replace('p_', '')} | {r['comparison']} | "
                f"{_fmt(r['from_prob'])} | {_fmt(r['to_prob'])} | {r['delta']:+.4f} |"
            )
    lines.append("")
    lines.append(_headline_callouts(comparison))

    # --- knockout matchup shifts caused by knockout_upset ---------------------
    lines.append("## Knockout matchup shifts (and what caused them)")
    lines.append("")
    if diagnostics_illustrative:
        lines.append(
            "> These matchup diagnostics use **illustrative example data** "
            "(`data/manual/*.example.csv`), not the live model — read the direction "
            "and mechanism, not the exact numbers."
        )
        lines.append("")
    if diagnostics.empty:
        lines.append("_No knockout matchups were supplied for diagnostics._")
    else:
        ranked = diagnostics.reindex(
            diagnostics["net_move_from_upset"].abs().sort_values(ascending=False).index
        )
        lines.append(
            "Net advance move attributable to `knockout_upset` (team A perspective), "
            "with the draw probability and provenance that drove it:"
        )
        lines.append("")
        lines.append(
            "| Matchup | base→final→survival (A adv) | net upset move | draw p | E[goals] | tags |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for _, r in ranked.iterrows():
            chain = (
                f"{_fmt(r['baseline_team_a_advances'])}→"
                f"{_fmt(r['final_ensemble_team_a_advances'])}→"
                f"{_fmt(r['knockout_survival_team_a_advances'])}"
            )
            # Render pipe-joined tags as commas so they don't break the table.
            tags = (r["provenance_tags"] or "—").replace("|", ", ")
            lines.append(
                f"| {r['team_a']} vs {r['team_b']} | {chain} | "
                f"{r['net_move_from_upset']:+.4f} | {_fmt(r['draw_prob'])} | "
                f"{_fmt(r['expected_goals_total'])} | {tags} |"
            )
    lines.append("")

    # --- flagged high-disagreement -------------------------------------------
    lines.append("## High-disagreement matchups (flagged for review)")
    lines.append("")
    flagged = _flagged_from_meta(runs)
    if flagged:
        lines.append("From the ensemble run provenance (max total-variation distance):")
        lines.append("")
        for item in flagged[:15]:
            home, away, tvd = item[0], item[1], item[2]
            lines.append(f"- {home} vs {away} — TVD {tvd}")
    else:
        diag_flagged = (
            diagnostics[diagnostics["flagged_high_disagreement"]]
            if not diagnostics.empty and "flagged_high_disagreement" in diagnostics
            else pd.DataFrame()
        )
        if not diag_flagged.empty:
            lines.append("From the matchup diagnostics:")
            lines.append("")
            for _, r in diag_flagged.iterrows():
                lines.append(f"- {r['team_a']} vs {r['team_b']} (flagged)")
        else:
            lines.append("_No matchups exceeded the disagreement threshold._")
    lines.append("")

    # --- missing / illustrative ----------------------------------------------
    lines.append("## Missing and illustrative signals")
    lines.append("")
    missing = _missing_from_meta(runs)
    if missing:
        lines.append("Signals that were absent for some matchups (renormalized away):")
        lines.append("")
        for sig, count in sorted(missing.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{sig}` — missing in {count} matchups")
    else:
        lines.append("_No missing-signal counts were recorded in the run metadata._")
    lines.append("")
    lines.append(
        "The bundled `data/manual/*.example.csv` files are **illustrative fixtures**, "
        "not real market/squad/form/expert coverage. Treat any ensemble numbers built "
        "on them as a smoke test, not evidence."
    )
    lines.append("")

    # --- honesty / limitations ------------------------------------------------
    lines.append(_limitations_block())
    lines.append("")
    lines.append(_commands_footer())
    return "\n".join(lines) + "\n"


def _headline_callouts(comparison: pd.DataFrame) -> str:
    """Short prose answering 'who changed most' for the headline stages."""
    if comparison.empty:
        return ""
    parts = ["**Headline movers by stage**", ""]
    for new, old in (("knockout_survival", "final_ensemble"), ("final_ensemble", "baseline")):
        any_stage = False
        for stage in HEADLINE_STAGES:
            col = f"delta__{new}_vs_{old}__{stage}"
            if col not in comparison.columns:
                continue
            sub = comparison[["team", col]].dropna()
            if sub.empty:
                continue
            top = sub.reindex(sub[col].abs().sort_values(ascending=False).index).iloc[0]
            if not any_stage:
                parts.append(f"- `{new}` vs `{old}`:")
                any_stage = True
            parts.append(
                f"    - {stage.replace('p_', '')}: {top['team']} {top[col]:+.4f}"
            )
    return "\n".join(parts) + "\n" if len(parts) > 2 else ""


def _limitations_block() -> str:
    return (
        "## What this does and does not show\n\n"
        "**Production-grade** (validated, stable): signal validation, typed "
        "probability objects (`OutcomeProbs`/`AdvanceProbs`), missing-signal "
        "renormalization, opt-in ensemble simulation, and the test + invariant "
        "suite. The historical baseline is the only fully-backtested forecaster "
        "(2010-2025, 15,499 matches).\n\n"
        "**Experimental** (this layer): the `knockout_upset` survival "
        "coefficients, the penalty/shootout priors, and the style-matchup "
        "coefficients are all bounded priors, **not fitted** to results. There is "
        "**no chronological knockout backtest** yet, and the manual/example data "
        "is illustrative coverage only.\n\n"
        "**Not claimed:**\n\n"
        "- This does **not** demonstrate an accuracy improvement — it shows *what "
        "moves and why*, not that the movement is correct.\n"
        "- Penalty/shootout history is **not** treated as highly predictive; it is "
        "shrunk toward 50/50 and capped, and only matters when a tie is genuinely "
        "likely to reach penalties.\n"
        "- No team (Croatia-style or otherwise) is assumed to win shootouts; the "
        "shootout edge is a small, capped nudge, never a deterministic rule."
    )


def _commands_footer() -> str:
    return (
        "## Reproduce\n\n"
        "On macOS prefer the `UV_NO_EDITABLE=1` prefix (see AGENTS.md).\n\n"
        "```bash\n"
        "# 1. Historical baseline (default; never overwritten by ensemble runs):\n"
        "UV_NO_EDITABLE=1 uv run goalsignal tournament simulate \\\n"
        "    --sims 100000 --seed 20260612\n\n"
        "# 2. Final ensemble:\n"
        "UV_NO_EDITABLE=1 uv run goalsignal tournament simulate \\\n"
        "    --prediction-source ensemble --ensemble-version final_ensemble \\\n"
        "    --sims 100000 --seed 20260612\n\n"
        "# 3. Knockout survival ensemble (opt-in):\n"
        "UV_NO_EDITABLE=1 uv run goalsignal tournament simulate \\\n"
        "    --prediction-source ensemble --ensemble-version knockout_survival \\\n"
        "    --include-knockout-upset --sims 100000 --seed 20260612\n\n"
        "# 4. This comparison report + CSVs:\n"
        "UV_NO_EDITABLE=1 uv run goalsignal evaluate simulation-comparison\n"
        "```"
    )


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "—"
    return f"{float(value):.3f}"


def write_comparison_artifacts(
    runs: dict[str, SimRun],
    comparison: pd.DataFrame,
    movers: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    out_dir: str | Path = "artifacts/ensemble",
    diagnostics_illustrative: bool = True,
) -> dict[str, Path]:
    """Write the four comparison artifacts and return their paths."""
    base = resolve(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "comparison_csv": base / "simulation_comparison.csv",
        "movers_csv": base / "biggest_movers.csv",
        "explanations_csv": base / "knockout_survival_explanations.csv",
        "report_md": base / "simulation_comparison.md",
    }
    comparison.to_csv(paths["comparison_csv"], index=False)
    movers.to_csv(paths["movers_csv"], index=False)
    diagnostics.to_csv(paths["explanations_csv"], index=False)
    paths["report_md"].write_text(
        render_markdown(
            runs, comparison, movers, diagnostics,
            diagnostics_illustrative=diagnostics_illustrative,
        ),
        encoding="utf-8",
    )
    return paths
