"""``goalsignal signals`` — inspect, blend, and compare external signals.

A thin Typer layer over :mod:`goalsignal.signals.pipeline`. Every command is
robust to missing manual files: absent signals are simply reported as
unavailable and the ensemble renormalizes over what remains.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from goalsignal.signals.base import AdvanceProbs, OutcomeProbs
from goalsignal.signals.market import load_market_odds
from goalsignal.signals.meta_ensemble import (
    MetaEnsemble,
    disagreement_vs_reference,
    load_ensemble_config,
)
from goalsignal.signals.pipeline import (
    SIGNAL_NAMES,
    blend_match,
    build_signals,
    load_manual_inputs,
    load_matches,
)

signals_app = typer.Typer(help="External signals: market, squad, form, venue, expert.")

DirOpt = Annotated[
    Path, typer.Option("--dir", help="Directory of manual signal CSVs.")
]
MatchesOpt = Annotated[
    Path, typer.Option("--matches", help="CSV of matches to forecast.")
]
VersionOpt = Annotated[
    str, typer.Option("--version", help="Ensemble model version (see config/ensemble.yaml).")
]


def _fmt_probs(probs: OutcomeProbs | AdvanceProbs) -> str:
    if isinstance(probs, OutcomeProbs):
        return f"H {probs.home_win:.3f} | D {probs.draw:.3f} | A {probs.away_win:.3f}"
    return f"A-adv {probs.team_a_advances:.3f} | B-adv {probs.team_b_advances:.3f}"


@signals_app.command("validate")
def signals_validate(directory: DirOpt = Path("data/manual")) -> None:
    """Load every manual signal file and report coverage and parse errors."""
    inputs = load_manual_inputs(directory)
    typer.echo(f"Manual signal directory: {directory}")
    typer.echo(f"  market quotes:        {len(inputs.market)}")
    typer.echo(f"  squad-strength teams: {len(inputs.squad.teams)}")
    typer.echo(f"  recent-form teams:    {len(inputs.form.teams)}")
    typer.echo(f"  venue-context rows:   {len(inputs.venue)}")
    typer.echo(f"  expert matches:       {len(inputs.expert)}")
    if inputs.load_errors:
        typer.echo("Parse warnings (rows skipped):")
        for source, errs in inputs.load_errors.items():
            for e in errs:
                typer.echo(f"  [{source}] {e}")
    else:
        typer.echo("No parse errors.")


@signals_app.command("market")
def signals_market(
    csv: Annotated[
        Path, typer.Option("--csv", help="Market odds CSV.")
    ] = Path("data/manual/market_odds.example.csv"),
    method: Annotated[
        str, typer.Option("--method", help="Overround removal: proportional|power.")
    ] = "proportional",
) -> None:
    """Show implied and vig-removed market probabilities per match."""
    quotes = load_market_odds(csv)
    if not quotes:
        typer.echo(f"No market quotes found at {csv}.")
        raise typer.Exit(0)
    for match_id, q in quotes.items():
        kind = "knockout" if q.two_way else "group"
        probs = q.advance(method) if q.two_way else q.outcome(method)
        typer.echo(
            f"{match_id} [{kind}] source={q.source} overround={q.overround():.3f}  "
            f"{_fmt_probs(probs)}"
        )


@signals_app.command("blend")
def signals_blend(
    matches: MatchesOpt = Path("data/manual/matches.example.csv"),
    directory: DirOpt = Path("data/manual"),
    version: VersionOpt = "final_ensemble",
    out: Annotated[
        Path | None, typer.Option("--out", help="Optional CSV path for blended output.")
    ] = None,
) -> None:
    """Blend all signals for each match under a named ensemble version."""
    config = load_ensemble_config()
    inputs = load_manual_inputs(directory, config)
    ensemble = MetaEnsemble(config)
    specs = load_matches(matches)

    rows = []
    for spec in specs:
        result, _ = blend_match(spec, inputs, ensemble, version=version)
        used = ",".join(f"{k}:{w:.2f}" for k, w in result.used_weights.items())
        flag = " FLAG" if ensemble.is_flagged(result) else ""
        typer.echo(
            f"{spec.match_id} [{spec.stage}] {spec.team_a} vs {spec.team_b}\n"
            f"    {_fmt_probs(result.probs)}\n"
            f"    weights: {used}\n"
            f"    missing: {result.missing or 'none'} | "
            f"max-disagreement: {result.max_pairwise_disagreement:.3f}{flag}"
        )
        row = {
            "match_id": spec.match_id,
            "stage": spec.stage,
            "team_a": spec.team_a,
            "team_b": spec.team_b,
            "missing_signals": "|".join(result.missing),
            "max_pairwise_disagreement": round(result.max_pairwise_disagreement, 4),
        }
        row.update({k: round(v, 4) for k, v in result.probs.to_dict().items()})
        rows.append(row)

    if out is not None:
        import pandas as pd

        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        typer.echo(f"Wrote {len(rows)} blended forecasts to {out}")


@signals_app.command("disagreement")
def signals_disagreement(
    matches: MatchesOpt = Path("data/manual/matches.example.csv"),
    directory: DirOpt = Path("data/manual"),
    reference: Annotated[
        str, typer.Option("--reference", help=f"Reference signal ({', '.join(SIGNAL_NAMES)}).")
    ] = "historical",
) -> None:
    """Report how far each signal sits from a reference signal (group matches)."""
    config = load_ensemble_config()
    inputs = load_manual_inputs(directory, config)
    threshold = config.disagreement_threshold
    for spec in load_matches(matches):
        if spec.knockout:
            continue
        signals = build_signals(spec, inputs)
        if signals.get(reference) is None:
            typer.echo(f"{spec.match_id}: reference '{reference}' unavailable; skipped")
            continue
        report = disagreement_vs_reference(signals, reference)
        parts = []
        for name, gap in sorted(report.gaps.items(), key=lambda kv: -kv[1]):
            mark = "*" if gap >= threshold else " "
            parts.append(f"{mark}{name} {gap:.3f}")
        typer.echo(f"{spec.match_id} vs {reference}: " + " | ".join(parts))
    typer.echo(f"(* = total-variation gap >= threshold {threshold:.2f})")


@signals_app.command("predict")
def signals_predict(
    matches: MatchesOpt = Path("data/manual/matches.example.csv"),
    directory: DirOpt = Path("data/manual"),
    version: VersionOpt = "final_ensemble",
    live: Annotated[
        bool,
        typer.Option(
            "--live/--no-live",
            help="Use the trained live model for the historical signal "
            "(default: historical probs from the matches CSV).",
        ),
    ] = False,
    out: Annotated[
        Path | None, typer.Option("--out", help="Optional CSV path for predictions.")
    ] = None,
) -> None:
    """Run ensemble match predictions through the public prediction API."""
    from goalsignal.signals.api import EnsemblePredictor

    config = load_ensemble_config()
    inputs = load_manual_inputs(directory, config)
    historical = None
    if live:
        from goalsignal.cli import _live_model
        from goalsignal.signals.historical_adapter import LiveModelHistorical

        _matches, live_model = _live_model(Path("config/data.yaml"), None)
        historical = LiveModelHistorical(live_model)
        typer.echo("Historical signal: live model")
    else:
        typer.echo("Historical signal: matches CSV (fixture data)")
    predictor = EnsemblePredictor(inputs, MetaEnsemble(config), historical)
    specs = load_matches(matches)
    preds = [predictor.predict(spec, version=version) for spec in specs]
    for p in preds:
        flag = " FLAG" if p.flagged else ""
        probs = " ".join(f"{k}={v:.3f}" for k, v in p.probs.to_dict().items())
        typer.echo(
            f"{p.match_id} [{p.stage}] {p.team_a} vs {p.team_b} ({p.historical_source})\n"
            f"    {probs}\n"
            f"    used: {p.used_weights} | missing: {p.missing or 'none'} | "
            f"disagreement: {p.disagreement:.3f}{flag}"
        )
    if out is not None:
        import pandas as pd

        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([p.to_row() for p in preds]).to_csv(out, index=False)
        typer.echo(f"Wrote {len(preds)} predictions to {out}")


@signals_app.command("tune-weights")
def signals_tune_weights(
    matches: Annotated[
        Path, typer.Option("--matches", help="Validation CSV (with a 'label' column).")
    ] = Path("data/manual/backtest_sample.example.csv"),
    predictions: Annotated[
        Path | None,
        typer.Option(
            "--predictions",
            help="Real validation predictions (overrides --matches). MUST be a "
            "validation split, never the test fold.",
        ),
    ] = None,
    directory: DirOpt = Path("data/manual"),
    objective: Annotated[
        str, typer.Option("--objective", help="Objective: log_loss (default) or brier.")
    ] = "log_loss",
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory for tuning artifacts.")
    ] = Path("artifacts/ensemble"),
) -> None:
    """Tune ensemble weights on validation data only (does not touch config)."""
    from goalsignal.evaluation.ensemble_backtest import load_backtest_table
    from goalsignal.signals.tuning import (
        tune_weights,
        write_tuned_weights,
        write_tuning_report,
    )

    config = load_ensemble_config()
    inputs = load_manual_inputs(directory, config)
    table = load_backtest_table(predictions if predictions is not None else matches)
    if table.smoke:
        typer.echo("NOTE: small/sample validation set — treat tuned weights as a smoke test.")
    result = tune_weights(
        table.specs, table.labels, inputs, objective=objective, config=config
    )
    if result.low_coverage:
        typer.echo(f"WARNING: {result.coverage_warning}")
    weights_path = write_tuned_weights(result, out_dir / "tuned_weights.yaml")
    report_path = write_tuning_report(result, out_dir / "tuning_report.md")
    typer.echo(f"Tuned ({objective}) over signals: {result.signals_present}")
    typer.echo(f"  weights: { {k: round(v, 3) for k, v in result.weights.items()} }")
    typer.echo(f"  tuned metrics:   {result.validation_metrics}")
    typer.echo(f"  default metrics: {result.baseline_metrics}")
    typer.echo(f"Wrote {weights_path} and {report_path} (config/ensemble.yaml unchanged).")
