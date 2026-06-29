"""GoalSignal command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from goalsignal.signals.cli import signals_app
from goalsignal.utils.paths import resolve

app = typer.Typer(help="GoalSignal: leakage-safe forecasting for international football.")
data_app = typer.Typer(help="Inspect, validate, and build the canonical dataset.")
ratings_app = typer.Typer(help="Build and inspect Elo ratings.")
evaluate_app = typer.Typer(help="Chronological backtests and evaluations.")
app.add_typer(data_app, name="data")
app.add_typer(ratings_app, name="ratings")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(signals_app, name="signals")

ConfigOpt = Annotated[
    Path, typer.Option("--config", help="Path to data configuration YAML.")
]
InputDirOpt = Annotated[
    Path | None,
    typer.Option("--input-dir", help="Directory containing the four source CSVs."),
]


def _load_config(config_path: Path, input_dir: Path | None):
    from goalsignal.data.schemas import DataConfig

    cfg = DataConfig.load(config_path)
    if input_dir is not None:
        cfg.input.directory = str(input_dir)
    return cfg


def _load_and_build(cfg):
    from goalsignal.data.build_dataset import build
    from goalsignal.data.loaders import load_all

    raw = load_all(cfg)
    return raw, build(raw, cfg)


@data_app.command("inspect")
def data_inspect(
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Show source-file schemas, row counts, and basic profile."""
    cfg = _load_config(config, input_dir)
    from goalsignal.data.loaders import load_all

    raw = load_all(cfg)
    for name, df in raw.items():
        cols = [c for c in df.columns if c not in ("source_row", "source_file")]
        typer.echo(f"{name}: {len(df)} rows | columns: {', '.join(cols)}")
        typer.echo(f"  path: {cfg.input_path(name)}")
    results = raw["results"]
    missing = (results["home_score"].isin(["NA", ""])) | (results["away_score"].isin(["NA", ""]))
    typer.echo(
        f"results: {int((~missing).sum())} with scores, {int(missing.sum())} without "
        f"(scheduled fixtures); dates {results['date'].min()} to {results['date'].max()}"
    )


@data_app.command("validate")
def data_validate(
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Run all data-quality checks and write audit reports (no dataset output)."""
    cfg = _load_config(config, input_dir)
    from goalsignal.data.validation import write_reports

    raw, result = _load_and_build(cfg)
    reports_dir, report = write_reports(raw, result, cfg)

    s = report.summary
    r = s["results"]
    typer.echo(f"Canonical matches: {r['canonical_matches']} "
               f"({r['played_matches']} played, {r['scheduled_matches']} scheduled)")
    typer.echo(f"Excluded rows: {r['excluded_rows']}")
    typer.echo(f"Suspicious-scope rows: {r['suspicious_scope_rows']}")
    typer.echo(f"Strict goal-model eligible: {r['strict_goal_model_eligible']}")
    typer.echo(f"Reports written to: {reports_dir}")

    hard_errors = (
        len(result.exclusions[result.exclusions["severity"] == "error"])
        if len(result.exclusions)
        else 0
    )
    if hard_errors:
        typer.echo(
            f"NOTE: {hard_errors} rows excluded with severity=error; "
            "review excluded_matches.csv"
        )


@data_app.command("build")
def data_build(
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing canonical dataset.")
    ] = False,
) -> None:
    """Build the canonical match dataset, audit reports, and dataset manifest."""
    cfg = _load_config(config, input_dir)
    from goalsignal.data.metadata import write_manifest
    from goalsignal.data.validation import write_reports

    out_dir = resolve(cfg.output.processed_dir)
    out_file = out_dir / "matches.csv"
    if out_file.exists() and not force:
        typer.echo(f"Refusing to overwrite existing {out_file}; pass --force to rebuild.")
        raise typer.Exit(code=1)

    raw, result = _load_and_build(cfg)
    reports_dir, _report = write_reports(raw, result, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    matches = result.matches.copy()
    matches["date"] = matches["date"].dt.strftime("%Y-%m-%d")
    matches.to_csv(out_file, index=False)
    manifest_path = write_manifest(cfg, result, out_file)

    typer.echo(f"Canonical dataset: {out_file} ({result.stats['canonical_matches']} matches)")
    typer.echo(f"Reports: {reports_dir}")
    typer.echo(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    app()


def _canonical_matches(cfg):
    """Rebuild the canonical match table in memory (deterministic from raw)."""
    _raw, result = _load_and_build(cfg)
    return result.matches


def _elo_inputs(config: Path, input_dir: Path | None):
    from goalsignal.ratings.elo import EloConfig, compute_elo

    cfg = _load_config(config, input_dir)
    matches = _canonical_matches(cfg)
    elo_cfg = EloConfig.load()
    return matches, compute_elo(matches, elo_cfg), elo_cfg


@ratings_app.command("build")
def ratings_build(
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Compute the full Elo timeline and persist it with final ratings."""
    _matches, result, _elo_cfg = _elo_inputs(config, input_dir)
    out = resolve("artifacts/ratings")
    out.mkdir(parents=True, exist_ok=True)
    timeline = result.timeline.copy()
    timeline["date"] = timeline["date"].dt.strftime("%Y-%m-%d")
    timeline.to_csv(out / "elo_timeline.csv", index=False)
    import pandas as pd

    finals = pd.DataFrame(
        sorted(result.final_ratings.items(), key=lambda kv: -kv[1]),
        columns=["team", "rating"],
    )
    finals.to_csv(out / "final_ratings.csv", index=False)
    typer.echo(f"Rated matches: {len(timeline)}")
    typer.echo(f"Timeline: {out / 'elo_timeline.csv'}")
    typer.echo(f"Final ratings: {out / 'final_ratings.csv'}")


@ratings_app.command("inspect")
def ratings_inspect(
    top: Annotated[int, typer.Option(help="Number of teams to show.")] = 20,
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Show the current top-rated teams."""
    _matches, result, _elo_cfg = _elo_inputs(config, input_dir)
    ranked = sorted(result.final_ratings.items(), key=lambda kv: -kv[1])[:top]
    for i, (team, rating) in enumerate(ranked, 1):
        typer.echo(f"{i:3d}. {team:30s} {rating:7.1f}")


@evaluate_app.command("rolling")
def evaluate_rolling(
    start_year: Annotated[int, typer.Option(help="First test year.")] = 2010,
    end_year: Annotated[int, typer.Option(help="Last test year.")] = 2025,
    val_years: Annotated[int, typer.Option(help="Validation years before each test year.")] = 3,
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Expanding-window yearly backtest of all models."""
    from goalsignal.evaluation.backtest import run_backtest
    from goalsignal.features.build_features import build_match_frame

    matches, elo_result, _elo_cfg = _elo_inputs(config, input_dir)
    frame = build_match_frame(matches, elo_result.timeline)
    summary = run_backtest(frame, start_year, end_year, val_years)
    typer.echo(f"{'model':28s} {'log_loss':>9s} {'brier':>8s} {'rps':>8s} {'acc':>7s}")
    for name, m in sorted(summary["pooled"].items(), key=lambda kv: kv[1]["log_loss"]):
        typer.echo(
            f"{name:28s} {m['log_loss']:9.4f} {m['brier']:8.4f} "
            f"{m['rps']:8.4f} {m['accuracy']:7.4f}"
        )
    typer.echo("Reports: artifacts/reports/backtest/")


@evaluate_app.command("ensemble-backtest")
def evaluate_ensemble_backtest(
    matches: Annotated[
        Path,
        typer.Option("--matches", help="Backtest CSV with historical probs + 'label'."),
    ] = Path("data/manual/backtest_sample.example.csv"),
    predictions: Annotated[
        Path | None,
        typer.Option(
            "--predictions",
            help="Real historical predictions (e.g. "
            "artifacts/reports/backtest/test_predictions.csv); overrides --matches.",
        ),
    ] = None,
    directory: Annotated[
        Path, typer.Option("--dir", help="Directory of manual signal CSVs.")
    ] = Path("data/manual"),
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory for backtest artifacts.")
    ] = Path("artifacts/ensemble"),
) -> None:
    """Compare ensemble versions on identical group-stage matches (fixed weights).

    Pass --predictions to run on real historical model outputs; otherwise the
    bundled sample runs as a clearly-labelled smoke test.
    """
    from goalsignal.evaluation.ensemble_backtest import (
        assess_ensemble,
        calibration_by_version,
        coverage_by_signal,
        load_backtest_table,
        run_ensemble_backtest,
        score_versions,
        write_reports,
    )
    from goalsignal.signals.meta_ensemble import MetaEnsemble, load_ensemble_config

    source = predictions if predictions is not None else matches
    config = load_ensemble_config()
    inputs = load_manual_inputs_for_cli(directory, config)
    table = load_backtest_table(source)
    if table.smoke:
        typer.echo(
            "SMOKE TEST: small/sample data — results are illustrative, not conclusive."
        )
    else:
        typer.echo(f"Real backtest on {len(table.specs)} matches from {source}.")
    ensemble = MetaEnsemble(config)
    df = run_ensemble_backtest(table, inputs, ensemble)
    if df.empty:
        typer.echo("No versions had any available signal to score.", err=True)
        raise typer.Exit(code=1)
    coverage = coverage_by_signal(table, inputs)
    scored = score_versions(table, inputs, ensemble)
    calibration = calibration_by_version(scored)
    assessment = assess_ensemble(df, coverage)
    typer.echo(df.to_string(index=False))
    typer.echo(f"\nVerdict: {assessment['verdict']}")
    paths = write_reports(df, coverage, calibration, assessment, table.smoke, out_dir)
    for label, p in paths.items():
        typer.echo(f"  {label}: {p}")


@evaluate_app.command("ensemble-ablation")
def evaluate_ensemble_ablation(
    matches: Annotated[
        Path, typer.Option("--matches", help="Backtest CSV with historical probs + 'label'.")
    ] = Path("data/manual/backtest_sample.example.csv"),
    predictions: Annotated[
        Path | None,
        typer.Option("--predictions", help="Real historical predictions; overrides --matches."),
    ] = None,
    directory: Annotated[
        Path, typer.Option("--dir", help="Directory of manual signal CSVs.")
    ] = Path("data/manual"),
    out_dir: Annotated[
        Path, typer.Option("--out-dir", help="Directory for ablation artifacts.")
    ] = Path("artifacts/ensemble"),
) -> None:
    """Ablation: historical only vs historical + each signal group vs full ensemble."""
    from goalsignal.evaluation.ensemble_backtest import (
        load_backtest_table,
        run_ablation,
        write_ablation,
    )
    from goalsignal.signals.meta_ensemble import MetaEnsemble, load_ensemble_config

    config = load_ensemble_config()
    inputs = load_manual_inputs_for_cli(directory, config)
    table = load_backtest_table(predictions if predictions is not None else matches)
    if table.smoke:
        typer.echo("SMOKE TEST: sample data — ablation deltas are illustrative only.")
    df = run_ablation(table, inputs, MetaEnsemble(config))
    if df.empty:
        typer.echo("No ablations scorable.", err=True)
        raise typer.Exit(code=1)
    typer.echo(df.to_string(index=False))
    paths = write_ablation(df, table.smoke, out_dir)
    for label, p in paths.items():
        typer.echo(f"  {label}: {p}")


def load_manual_inputs_for_cli(directory: Path, config):
    """Thin wrapper so commands share one manual-inputs loader."""
    from goalsignal.signals.pipeline import load_manual_inputs

    return load_manual_inputs(directory, config)


tournament_app = typer.Typer(help="Tournament simulation.")
predict_app = typer.Typer(help="Match and schedule predictions.")
ledger_app = typer.Typer(help="Append-only prediction ledger.")
app.add_typer(tournament_app, name="tournament")
app.add_typer(predict_app, name="predict")
app.add_typer(ledger_app, name="ledger")


def _live_model(config: Path, input_dir: Path | None, with_results: bool = True):
    """Train the deployment model; recorded results are overlaid by default.

    The overlay marks recorded fixtures as played in memory (never touching
    `Datasets/`), which advances the cutoff and the model version suffix
    (+rN), so refreshed predictions are new immutable entries.
    """
    from goalsignal.data.metadata import compute_dataset_version
    from goalsignal.live import MODEL_VERSION, train_live_model

    cfg = _load_config(config, input_dir)
    matches = _canonical_matches(cfg)
    version = compute_dataset_version(cfg)
    n_applied = 0
    if with_results:
        from goalsignal.feedback.results import active_results, apply_results_overlay

        matches, n_applied = apply_results_overlay(matches, active_results())
    live = train_live_model(matches, version)
    if n_applied:
        live.model_version = f"{MODEL_VERSION}+r{n_applied}"
        live.diagnostics["recorded_results_applied"] = n_applied
    return matches, live


def _current_fifa(config: Path, input_dir: Path | None):
    import os

    from goalsignal.data.sources.config import (
        FifaCurrentRankingsConfig,
        validate_source_path,
    )
    from goalsignal.data.sources.fifa_current import load_current_fifa

    _load_env()
    source_cfg = FifaCurrentRankingsConfig.load()
    raw = os.environ.get(source_cfg.path_env, "")
    if not raw:
        return None
    path = validate_source_path(raw, kind="file", extensions=(".csv",))
    cfg = _load_config(config, input_dir)
    matches = _canonical_matches(cfg)
    canonical = set(matches["home_team"]) | set(matches["away_team"])
    return load_current_fifa(path, canonical)


@tournament_app.command("simulate")
def tournament_simulate(
    sims: Annotated[int, typer.Option(help="Number of Monte Carlo simulations.")] = 100_000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 20260612,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite the same versioned simulation.")
    ] = False,
    prediction_source: Annotated[
        str,
        typer.Option(
            "--prediction-source",
            help="Probability source: 'historical' (default goal model) or "
            "'ensemble' (blended meta-ensemble).",
        ),
    ] = "historical",
    ensemble_version: Annotated[
        str,
        typer.Option(
            "--ensemble-version",
            help="Ensemble model version when --prediction-source ensemble.",
        ),
    ] = "final_ensemble",
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Simulate the full 2026 World Cup from groups through the champion."""
    import time

    if prediction_source not in {"historical", "ensemble"}:
        typer.echo(
            f"Invalid --prediction-source {prediction_source!r}; "
            "expected 'historical' or 'ensemble'.",
            err=True,
        )
        raise typer.Exit(code=1)

    from goalsignal.feedback.results import (
        active_results,
        result_store_hash,
        verify_results,
    )
    from goalsignal.tournament.bracket_2026 import OfficialBracket
    from goalsignal.tournament.fixtures_2026 import derive_2026_group_stage
    from goalsignal.tournament.full_simulator import (
        apply_official_group_letters,
        check_full_invariants,
        simulate_full_tournament,
    )
    from goalsignal.tournament.live_update import (
        simulation_version,
    )
    from goalsignal.tournament.model_adapter import RatingsGoalAdapter
    from goalsignal.tournament.reporting import (
        advancement_frame,
        write_full_simulation,
        write_ticket_advisory,
    )
    from goalsignal.tournament.simulator import validate_completed_overlay

    result_problems = verify_results()
    if result_problems:
        typer.echo(f"Result store verification failed: {result_problems}", err=True)
        raise typer.Exit(code=1)
    bracket = OfficialBracket.load()

    matches, live = _live_model(config, input_dir)
    groups, fixtures = derive_2026_group_stage(matches)
    active = active_results()
    validate_completed_overlay(fixtures, active)
    fifa = _current_fifa(config, input_dir)
    if fifa is None:
        typer.echo(
            "FIFA_CURRENT_RANKINGS_PATH is required to resolve official groups A-L.",
            err=True,
        )
        raise typer.Exit(code=1)
    fifa_df, _manifest, _quality = fifa
    snapshot_id = fifa_df["source_snapshot_id"].iloc[0]
    official_groups = {
        g: list(block["canonical_team"])
        for g, block in fifa_df.groupby("group", sort=True)
    }
    groups, fixtures = apply_official_group_letters(groups, fixtures, official_groups)
    provenance_summary = None
    if prediction_source == "ensemble":
        from goalsignal.tournament.ensemble_source import (
            build_ensemble_adapter,
            ensemble_provenance_summary,
            format_provenance_summary,
        )

        adapter, _predictor = build_ensemble_adapter(live, version=ensemble_version)
        typer.echo(
            f"Prediction source: ensemble (version {ensemble_version}); "
            "historical signal from the live model."
        )
    else:
        adapter = RatingsGoalAdapter(live.ratings, live.goal_model)
    started = time.perf_counter()
    result = simulate_full_tournament(
        groups, fixtures, adapter, bracket, n_sims=sims, seed=seed
    )
    runtime = time.perf_counter() - started
    if prediction_source == "ensemble":
        provenance_summary = ensemble_provenance_summary(adapter)
        typer.echo(format_provenance_summary(provenance_summary))
    problems = check_full_invariants(result)
    if problems:
        for p in problems:
            typer.echo(f"INVARIANT VIOLATION: {p}", err=True)
        raise typer.Exit(code=1)
    if adapter.unrated_teams:
        typer.echo(f"NOTE: unrated teams given default 1500: {sorted(adapter.unrated_teams)}")

    df = advancement_frame(result)
    result_hash = result_store_hash()
    version = simulation_version(
        result_hash, live.model_version, snapshot_id, bracket.config_hash
    )
    if prediction_source == "ensemble":
        # Keep ensemble runs in a distinct version so they never overwrite the
        # canonical historical simulation artifacts.
        version = f"{version}.ensemble-{ensemble_version}"
    out = resolve(Path("artifacts/simulations") / version)
    if (out / "wc2026_tournament_meta.json").exists() and not force:
        typer.echo(f"Refusing to overwrite {out}; pass --force.", err=True)
        raise typer.Exit(code=1)
    completed = [
        {
            "fixture_id": f.fixture_id,
            "home_team": f.home,
            "away_team": f.away,
            "home_goals": f.home_goals,
            "away_goals": f.away_goals,
        }
        for f in fixtures if f.played and f.fixture_id in active
    ]
    meta = {
        "n_sims": sims,
        "seed": seed,
        "data_cutoff": str(live.cutoff.date()),
        "dataset_version": live.dataset_version,
        "model_version": live.model_version,
        "result_store_hash": result_hash,
        "active_completed_result_count": len(active),
        "completed_fixtures": completed,
        "remaining_fixture_count": sum(not f.played for f in fixtures),
        "fifa_snapshot_id": snapshot_id,
        "simulation_version": version,
        "bracket_config_hash": bracket.config_hash,
        "third_place_table_hash": bracket.table_hash,
        "official_source_manifest": bracket.source_manifest,
        "runtime_seconds": runtime,
        "resolution_counts": dict(result.resolution_counts),
        "prediction_source": prediction_source,
        "ensemble_version": ensemble_version if prediction_source == "ensemble" else None,
        "ensemble_provenance": provenance_summary,
        "diagnostics": live.diagnostics,
        "groups_label_note": "official groups A-L from the validated FIFA snapshot",
        "modal_bracket_note": "probabilistic summary only; no matchup is confirmed",
    }
    out = write_full_simulation(result, bracket, meta, version)
    top_contenders = set(fifa_df.nsmallest(10, "fifa_rank")["canonical_team"])
    ticket_paths = write_ticket_advisory(result, bracket, top_contenders)

    typer.echo(
        f"Simulations: {sims} (seed {seed}); cutoff {live.cutoff.date()}; "
        f"runtime {runtime:.2f}s"
    )
    typer.echo(df[["team", "p_round_of_32", "p_quarterfinal", "p_final", "p_champion"]]
               .head(15).to_string(index=False))
    typer.echo(f"Full tournament artifacts: {out}")
    typer.echo(f"Ticket advisory: {ticket_paths[0]} and {ticket_paths[1]}")


@tournament_app.command("validate-bracket")
def tournament_validate_bracket() -> None:
    """Validate official source hashes, all 495 combinations, and M73-M104."""
    from goalsignal.tournament.bracket_2026 import OfficialBracket

    bracket = OfficialBracket.load()
    typer.echo(
        f"Bracket valid: {len(bracket.matches)} matches, "
        f"{len(bracket.third_assignments)} third-place combinations."
    )


@tournament_app.command("inspect-bracket")
def tournament_inspect_bracket() -> None:
    """Display the official symbolic bracket without projected team names."""
    from goalsignal.tournament.bracket_2026 import OfficialBracket

    bracket = OfficialBracket.load()
    for number in sorted(bracket.matches):
        match = bracket.matches[number]
        typer.echo(
            f"M{number}: {match.entrants[0]} v {match.entrants[1]} | "
            f"{match.date} {match.time_et} ET | {match.host_city}"
        )


def _latest_tournament_dir(version: str | None = None) -> Path:
    root = resolve("artifacts/simulations")
    if version:
        path = root / version
    else:
        candidates = list(root.glob("*/wc2026_tournament_meta.json"))
        if not candidates:
            raise FileNotFoundError("no full tournament simulation exists")
        path = max(candidates, key=lambda item: item.stat().st_mtime).parent
    if not (path / "wc2026_tournament_meta.json").exists():
        raise FileNotFoundError(f"no full tournament simulation in {path}")
    return path


@tournament_app.command("advancement")
def tournament_advancement(
    version: Annotated[str | None, typer.Option(help="Simulation version.")] = None,
) -> None:
    """Print advancement probabilities from R32 through champion."""
    import pandas as pd

    path = _latest_tournament_dir(version) / "wc2026_team_advancement.csv"
    frame = pd.read_csv(path)
    cols = ["team", "p_round_of_32", "p_round_of_16", "p_quarterfinal",
            "p_semifinal", "p_final", "p_champion"]
    typer.echo(frame[cols].to_string(index=False))
    typer.echo(f"Source: {path}")


@tournament_app.command("matchup-probabilities")
def tournament_matchup_probabilities(
    match_number: Annotated[int, typer.Option("--match-number", min=73, max=104)],
    version: Annotated[str | None, typer.Option(help="Simulation version.")] = None,
) -> None:
    """Print likely matchups for one official match number."""
    import pandas as pd

    from goalsignal.tournament.bracket_2026 import OfficialBracket
    from goalsignal.tournament.reporting import ROUND_FILES

    bracket = OfficialBracket.load()
    round_name = bracket.matches[match_number].round
    if round_name == "third_place":
        filename = "wc2026_third_place_matchups.csv"
    else:
        filename = ROUND_FILES[round_name]
    path = _latest_tournament_dir(version) / filename
    frame = pd.read_csv(path)
    typer.echo(
        frame[frame["match_number"] == match_number]
        .head(20)
        .to_string(index=False)
    )


@tournament_app.command("bracket")
def tournament_bracket(
    version: Annotated[str | None, typer.Option(help="Simulation version.")] = None,
) -> None:
    """Display the modal probabilistic bracket summary."""
    import json

    path = _latest_tournament_dir(version) / "wc2026_bracket.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    typer.echo(data["label"])
    for match in data["matches"]:
        typer.echo(
            f"M{match['match_number']}: {' v '.join(match['modal_matchup'])} "
            f"({match['matchup_probability']:.2%}); conditional pick "
            f"{match['modal_conditional_winner']} "
            f"({match['conditional_win_probability']:.2%})"
        )


@tournament_app.command("ticket-advisory")
def tournament_ticket_advisory() -> None:
    """Print the current late-round ticket-planning report."""
    path = resolve("artifacts/reports/wc2026_ticket_advisory.csv")
    if not path.exists():
        raise typer.BadParameter("run tournament simulate first")
    typer.echo(path.read_text(encoding="utf-8"))


@predict_app.command("remaining")
def predict_remaining(
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Predict all scheduled fixtures and append them to the prediction ledger."""
    from goalsignal.feedback.results import active_results, result_store_hash
    from goalsignal.ledger.storage import DEFAULT_PATH, append_predictions, list_entries
    from goalsignal.live import build_prediction_payload

    matches, live = _live_model(config, input_dir)
    scheduled = matches[matches["status"] == "scheduled"].sort_values("date")
    if len(scheduled) == 0:
        typer.echo("No scheduled fixtures in the dataset.")
        raise typer.Exit(code=0)

    fifa = _current_fifa(config, input_dir)
    snapshot_id = fifa[1]["snapshot_id"] if fifa else None
    active = active_results()
    revision = {
        "revision": live.model_version,
        "result_store_hash": result_store_hash(),
        "active_result_count": len(active),
        "elo_state_hash": __import__("goalsignal.utils.hashing", fromlist=["sha256_json"])
        .sha256_json(live.ratings),
        "feature_set_version": "elo-venue-v1",
        "source_snapshot_ids": {"fifa_current": snapshot_id},
    }
    payloads = [
        build_prediction_payload(live, row, revision)
        for row in scheduled.itertuples(index=False)
    ]
    seen = {
        (e["payload"].get("fixture_id"), e["payload"].get("model_version"))
        for e in list_entries()
    }
    payloads = [
        p for p in payloads if (p["fixture_id"], p["model_version"]) not in seen
    ]
    if not payloads:
        typer.echo(f"No new predictions: revision {live.model_version} is already complete.")
        return
    entries = append_predictions(payloads)
    typer.echo(f"Appended {len(entries)} predictions to {resolve(DEFAULT_PATH)}")
    typer.echo(f"Data cutoff: {live.cutoff.date()}; model {live.model_version}; "
               f"weights {live.diagnostics['ensemble_weights']}")


LedgerPathOpt = Annotated[
    Path, typer.Option("--path", help="Ledger file (default: the project ledger).")
]


@ledger_app.command("list")
def ledger_list(
    scores: Annotated[
        bool, typer.Option("--scores", help="Include expected goals and likely score.")
    ] = False,
    path: LedgerPathOpt = Path("artifacts/predictions/ledger.jsonl"),
) -> None:
    """List ledger entries (fixture, kickoff, probabilities)."""
    from goalsignal.ledger.display import format_table
    from goalsignal.ledger.storage import list_entries

    entries = list_entries(path)
    if not entries:
        typer.echo("Ledger is empty.")
        return
    if scores:
        typer.echo(format_table(entries))
    else:
        for e in entries:
            p = e["payload"]
            typer.echo(
                f"{p['kickoff_timestamp']}  {p['home_team']:25s} v {p['away_team']:25s} "
                f"H {p['home_win_probability']:.3f}  D {p['draw_probability']:.3f}  "
                f"A {p['away_win_probability']:.3f}  [{p['model_version']}]"
            )
    typer.echo(f"{len(entries)} entries.")


@ledger_app.command("show")
def ledger_show(
    prediction_id: Annotated[
        str,
        typer.Option(
            "--prediction-id",
            help="Entry-hash prefix or fixture-id prefix (must match exactly one entry).",
        ),
    ],
    top_scorelines: Annotated[
        int, typer.Option("--top-scorelines", help="Scorelines to display.")
    ] = 5,
    path: LedgerPathOpt = Path("artifacts/predictions/ledger.jsonl"),
) -> None:
    """Show one full prediction entry, including scorelines and markets."""
    import json

    from goalsignal.ledger.display import SCORE_MODEL_UNRECORDED, find_entry
    from goalsignal.ledger.storage import list_entries

    entry = find_entry(list_entries(path), prediction_id)
    if entry is None:
        typer.echo(f"No unique entry matches id prefix {prediction_id!r}.", err=True)
        raise typer.Exit(code=1)
    p = dict(entry["payload"])
    p["top_scorelines"] = (p.get("top_scorelines") or [])[:top_scorelines]
    typer.echo(json.dumps(p, indent=2, ensure_ascii=False))
    typer.echo(f"entry_hash: {entry['entry_hash']}")
    typer.echo(f"W/D/L model: {p.get('model_version')}")
    typer.echo(f"score model: {p.get('score_model_version', SCORE_MODEL_UNRECORDED)}")


predictions_app = typer.Typer(help="Read-only views over stored predictions.")
app.add_typer(predictions_app, name="predictions")


@predictions_app.command("scores")
def predictions_scores(
    top_scorelines: Annotated[
        int, typer.Option("--top-scorelines", help="Scorelines per match (table view).")
    ] = 1,
    team: Annotated[
        str | None, typer.Option("--team", help="Filter: substring match on either team.")
    ] = None,
    date: Annotated[
        str | None, typer.Option("--date", help="Filter: exact kickoff date (YYYY-MM-DD).")
    ] = None,
    output_format: Annotated[
        str, typer.Option("--format", help="table | csv | json")
    ] = "table",
    path: LedgerPathOpt = Path("artifacts/predictions/ledger.jsonl"),
    latest_only: Annotated[
        bool, typer.Option("--latest-only/--no-latest-only")
    ] = True,
    show_revisions: Annotated[
        bool, typer.Option("--show-revisions", help="Show every immutable revision.")
    ] = False,
    model_version: Annotated[
        str | None, typer.Option("--model-version", help="Filter exact model revision.")
    ] = None,
) -> None:
    """Expected goals and exact-score forecasts for stored predictions."""
    from goalsignal.ledger.display import (
        filter_entries,
        format_csv,
        format_json,
        format_table,
        latest_entries,
        model_version_entries,
    )
    from goalsignal.ledger.storage import list_entries

    entries = model_version_entries(list_entries(path), model_version)
    if latest_only and not show_revisions:
        entries = latest_entries(entries)
    entries = filter_entries(entries, team=team, date=date)
    if not entries:
        typer.echo("No predictions match the given filters.")
        raise typer.Exit(code=1)
    if output_format == "table":
        typer.echo(format_table(entries, top_scorelines=top_scorelines))
    elif output_format == "csv":
        typer.echo(format_csv(entries), nl=False)
    elif output_format == "json":
        typer.echo(format_json(entries, top_scorelines=max(top_scorelines, 5)))
    else:
        typer.echo(f"Unknown format {output_format!r}; use table, csv, or json.", err=True)
        raise typer.Exit(code=2)


@ledger_app.command("verify")
def ledger_verify() -> None:
    """Verify the ledger's hash chain; nonzero exit on any violation."""
    from goalsignal.ledger.storage import list_entries, verify_ledger

    problems = verify_ledger()
    n = len(list_entries())
    if problems:
        for p in problems:
            typer.echo(f"FAIL: {p}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Ledger intact: {n} entries verified.")


@app.command("benchmark")
def benchmark(
    sims: Annotated[int, typer.Option(help="Simulations per run.")] = 20_000,
    repeats: Annotated[int, typer.Option(help="Timing repetitions.")] = 3,
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Benchmark reference vs vectorized group-stage simulators (measured, not claimed)."""
    import json
    import platform
    import statistics
    import time

    from goalsignal.tournament.fixtures_2026 import derive_2026_group_stage
    from goalsignal.tournament.model_adapter import RatingsGoalAdapter
    from goalsignal.tournament.simulator import simulate_groups, simulate_groups_fast

    matches, live = _live_model(config, input_dir)
    groups, fixtures = derive_2026_group_stage(matches)
    adapter = RatingsGoalAdapter(live.ratings, live.goal_model)

    results = {}
    for name, fn in (("reference", simulate_groups), ("vectorized", simulate_groups_fast)):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            res = fn(groups, fixtures, adapter, n_sims=sims, seed=1)
            times.append(time.perf_counter() - t0)
        results[name] = {
            "median_seconds": statistics.median(times),
            "all_seconds": times,
            "sims_per_second": sims / statistics.median(times),
            "sample_advance_prob_Mexico": res.advance_probs.get("Mexico"),
        }
        typer.echo(
            f"{name:12s} median {statistics.median(times):7.3f}s "
            f"({sims / statistics.median(times):,.0f} sims/s over {repeats} runs)"
        )
    env = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "n_sims": sims,
        "repeats": repeats,
    }
    out = resolve("artifacts/benchmarks")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "simulator_benchmark.json", "w", encoding="utf-8") as f:
        json.dump({"environment": env, "results": results}, f, indent=2)
    typer.echo(f"Benchmark written: {out / 'simulator_benchmark.json'}")


result_app = typer.Typer(help="Record completed-match results (separate from predictions).")
results_app = typer.Typer(help="Verify the append-only result store.")
feedback_app = typer.Typer(help="Score frozen predictions against recorded results.")
app.add_typer(result_app, name="result")
app.add_typer(results_app, name="results")
app.add_typer(feedback_app, name="feedback")


def _resolve_fixture(matches, fixture_id: str):
    rows = matches[matches["canonical_match_id"].str.startswith(fixture_id)]
    if len(rows) != 1:
        typer.echo(
            f"fixture id prefix {fixture_id!r} matches {len(rows)} fixtures; "
            "need exactly one", err=True,
        )
        raise typer.Exit(code=1)
    return rows.iloc[0]


def _persist_elo_update(cfg, fixture_id: str, result_entry_hash: str) -> dict:
    """Recompute ratings with all recorded results and persist this match's
    pre/post Elo update (derived, regenerable artifact)."""
    import json

    from goalsignal.feedback.results import (
        active_results,
        apply_results_overlay,
        list_results,
        result_store_hash,
    )
    from goalsignal.ratings.elo import EloConfig, compute_elo
    from goalsignal.utils.hashing import sha256_file

    matches = _canonical_matches(cfg)
    overlaid, _ = apply_results_overlay(matches, active_results())
    timeline = compute_elo(overlaid, EloConfig.load()).timeline
    row = timeline[timeline["canonical_match_id"] == fixture_id]
    if len(row) != 1:
        typer.echo("WARNING: could not locate Elo update for the recorded result", err=True)
        return {}
    entry_by_fixture = {
        e["payload"]["fixture_id"]: e["entry_hash"] for e in list_results()
    }
    active_ids = set(active_results())
    updates = []
    for order, row in enumerate(
        timeline[timeline["canonical_match_id"].isin(active_ids)].itertuples(index=False)
    ):
        updates.append({
            "canonical_match_id": row.canonical_match_id,
            "chronological_order": order,
            "chronological_key": [str(row.date.date()), int(
                overlaid.loc[
                    overlaid["canonical_match_id"] == row.canonical_match_id, "source_row"
                ].iloc[0]
            )],
            "date": str(row.date.date()),
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_elo_pre": float(row.home_elo_pre),
            "home_elo_post": float(row.home_elo_post),
            "away_elo_pre": float(row.away_elo_pre),
            "away_elo_post": float(row.away_elo_post),
            "expected_home": float(row.expected_home),
            "actual_home": float(row.actual_home),
            "delta": float(row.delta),
            "elo_config_hash": sha256_file(resolve("config/elo.yaml")),
            "active_result_store_hash": result_store_hash(),
            "result_entry_hash": entry_by_fixture[row.canonical_match_id],
        })
    out = resolve("artifacts/ratings/online_updates.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(u, sort_keys=True) + "\n" for u in updates),
        encoding="utf-8",
    )
    return next(u for u in updates if u["canonical_match_id"] == fixture_id)


@result_app.command("record")
def result_record(
    fixture_id: Annotated[
        str, typer.Option("--fixture-id", help="Canonical fixture id (prefix ok).")
    ],
    home_goals: Annotated[int, typer.Option("--home-goals", min=0)],
    away_goals: Annotated[int, typer.Option("--away-goals", min=0)],
    completed_at: Annotated[
        str, typer.Option("--completed-at", help="UTC timestamp or date the match finished.")
    ],
    source: Annotated[str, typer.Option("--source", help="Result provenance.")],
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Record a completed regulation-time result in the append-only result store."""
    from goalsignal.feedback.results import active_results, record_result, verify_results
    from goalsignal.ledger.storage import list_entries, verify_ledger

    cfg = _load_config(config, input_dir)
    matches = _canonical_matches(cfg)
    fixture = _resolve_fixture(matches, fixture_id)
    fid = fixture["canonical_match_id"]
    if fixture["status"] == "played":
        typer.echo("Fixture is already played in the source dataset; refusing.", err=True)
        raise typer.Exit(code=1)
    existing = active_results().get(fid)
    if existing is not None:
        same = (
            existing["regulation_home_goals"] == home_goals
            and existing["regulation_away_goals"] == away_goals
        )
        if same:
            typer.echo(
                f"Already recorded: {fixture['home_team']} {home_goals}-{away_goals} "
                f"{fixture['away_team']}; no append performed."
            )
            return
        typer.echo("Conflicting active result exists; use `result correct`.", err=True)
        raise typer.Exit(code=1)

    has_prediction = any(
        e["payload"].get("fixture_id") == fid for e in list_entries()
    )
    if not has_prediction:
        typer.echo("NOTE: no frozen prediction exists for this fixture.")

    pre_problems = verify_ledger()
    if pre_problems:
        typer.echo("Prediction ledger FAILED verification before recording; aborting.", err=True)
        raise typer.Exit(code=1)

    entry = record_result(
        fixture_id=fid,
        home_goals=home_goals,
        away_goals=away_goals,
        completed_at=completed_at,
        source=source,
        kickoff_date=str(fixture["date"].date()),
        match_date=str(fixture["date"].date()),
        home_team=fixture["home_team"],
        away_team=fixture["away_team"],
    )
    post_problems = verify_ledger()
    res_problems = verify_results()
    typer.echo(
        f"Recorded {fixture['home_team']} {home_goals}-{away_goals} "
        f"{fixture['away_team']} ({entry['payload']['outcome']}) "
        f"[entry {entry['entry_hash'][:12]}]"
    )
    typer.echo(f"Prediction ledger after recording: "
               f"{'INTACT' if not post_problems else 'VIOLATIONS: ' + str(post_problems)}")
    typer.echo(f"Result store: {'INTACT' if not res_problems else str(res_problems)}")

    update = _persist_elo_update(cfg, fid, entry["entry_hash"])
    if update:
        typer.echo(
            f"Elo: {update['home_team']} {update['home_elo_pre']:.1f} -> "
            f"{update['home_elo_post']:.1f} (delta {update['delta']:+.2f}); "
            f"{update['away_team']} {update['away_elo_pre']:.1f} -> "
            f"{update['away_elo_post']:.1f}"
        )
        typer.echo("Online update persisted: artifacts/ratings/online_updates.jsonl")
    typer.echo(
        "Next: `goalsignal feedback match --fixture-id ...` to score the frozen "
        "forecast, `goalsignal predict remaining` to refresh future fixtures, "
        "`goalsignal tournament simulate` to refresh advancement probabilities."
    )


@result_app.command("correct")
def result_correct(
    fixture_id: Annotated[str, typer.Option("--fixture-id")],
    home_goals: Annotated[int, typer.Option("--home-goals", min=0)],
    away_goals: Annotated[int, typer.Option("--away-goals", min=0)],
    completed_at: Annotated[str, typer.Option("--completed-at")],
    source: Annotated[str, typer.Option("--source")],
    reason: Annotated[str, typer.Option("--reason", help="Why the prior entry is wrong.")],
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Audited correction: append a superseding result referencing the old entry."""
    from goalsignal.feedback.results import list_results, record_result, verify_results

    cfg = _load_config(config, input_dir)
    fixture = _resolve_fixture(_canonical_matches(cfg), fixture_id)
    fid = fixture["canonical_match_id"]
    prior = [e for e in list_results() if e["payload"]["fixture_id"] == fid]
    if not prior:
        typer.echo("No existing result to correct; use `result record`.", err=True)
        raise typer.Exit(code=1)
    entry = record_result(
        fixture_id=fid, home_goals=home_goals, away_goals=away_goals,
        completed_at=completed_at, source=source,
        kickoff_date=str(fixture["date"].date()),
        corrects=prior[-1]["entry_hash"], correction_reason=reason,
        match_date=str(fixture["date"].date()),
        home_team=fixture["home_team"], away_team=fixture["away_team"],
    )
    typer.echo(f"Correction recorded [entry {entry['entry_hash'][:12]}], "
               f"supersedes {prior[-1]['entry_hash'][:12]}.")
    if verify_results():
        typer.echo("Result store verification failed after correction.", err=True)
        raise typer.Exit(code=1)
    _persist_elo_update(cfg, fid, entry["entry_hash"])


@results_app.command("verify")
def results_verify() -> None:
    from goalsignal.feedback.results import list_results, result_store_hash, verify_results

    problems = verify_results()
    if problems:
        for problem in problems:
            typer.echo(f"FAIL: {problem}", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"Result store intact: {len(list_results())} entries; "
        f"sha256={result_store_hash()}"
    )


@feedback_app.command("match")
def feedback_match(
    fixture_id: Annotated[str, typer.Option("--fixture-id", help="Fixture id (prefix ok).")],
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Score the frozen forecast for one completed fixture."""
    import json

    from goalsignal.data.metadata import compute_dataset_version
    from goalsignal.feedback.results import active_results
    from goalsignal.feedback.scoring import (
        reconstruct_scoreline_probability,
        score_prediction,
    )
    from goalsignal.ledger.storage import list_entries

    cfg = _load_config(config, input_dir)
    entries = [
        e for e in list_entries()
        if e["payload"].get("fixture_id", "").startswith(fixture_id)
    ]
    if not entries:
        typer.echo("No prediction found for that fixture.", err=True)
        raise typer.Exit(code=1)
    payload = entries[0]["payload"]  # the original frozen forecast
    fid = payload["fixture_id"]
    result = active_results().get(fid)
    if result is None:
        typer.echo("No recorded result for that fixture yet.", err=True)
        raise typer.Exit(code=1)

    report = score_prediction(payload, result)
    if len(entries) > 1:
        report["note_multiple_predictions"] = (
            f"{len(entries)} predictions exist for this fixture; scored the original "
            f"({payload['model_version']})"
        )

    if report["actual_scoreline_probability"] is None:
        # Reconstruct the frozen goal model: only valid if the dataset is
        # byte-identical to the one used at prediction time.
        if compute_dataset_version(cfg) != payload["dataset_version"]:
            report["actual_scoreline_probability_source"] = (
                "unavailable: dataset has changed since prediction; "
                "exact reconstruction impossible"
            )
        else:
            typer.echo("Reconstructing frozen goal model (deterministic retrain)...")
            _matches, live = _live_model(config, input_dir, with_results=False)
            if str(live.cutoff.date()) != payload["data_cutoff"]:
                report["actual_scoreline_probability_source"] = (
                    "unavailable: reconstructed cutoff differs from prediction cutoff"
                )
            else:
                feats = live.feature_row(
                    payload["home_team"], payload["away_team"],
                    bool(_resolve_fixture(_matches, fid)["neutral"]),
                )
                p, why = reconstruct_scoreline_probability(
                    payload, live.goal_model, feats,
                    report["actual_home_goals"], report["actual_away_goals"],
                )
                report["actual_scoreline_probability"] = p
                report["actual_scoreline_probability_source"] = why

    out = resolve("artifacts/reports/feedback")
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"match_{fid[:12]}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    p = payload
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    typer.echo("Original forecast:")
    typer.echo(f"  {p['home_team']} win{'':<12} {pct(p['home_win_probability'])}")
    typer.echo(f"  Draw{'':<20} {pct(p['draw_probability'])}")
    typer.echo(f"  {p['away_team']} win{'':<12} {pct(p['away_win_probability'])}")
    typer.echo(f"  Expected goals          {p['home_expected_goals']:.2f}-"
               f"{p['away_expected_goals']:.2f}")
    typer.echo(f"  Top scoreline           {report['top_scoreline']}")
    typer.echo("")
    typer.echo("Actual:")
    typer.echo(f"  {p['home_team']} {report['actual_home_goals']}-"
               f"{report['actual_away_goals']} {p['away_team']}")
    typer.echo(f"  Outcome: {report['actual_outcome']}")
    typer.echo("")
    typer.echo("Evaluation:")
    typer.echo(f"  Predicted outcome correct: "
               f"{'yes' if report['predicted_outcome_correct'] else 'no'}")
    typer.echo(f"  Exact score correct: {'yes' if report['exact_score_correct'] else 'no'}")
    typer.echo(f"  Probability assigned to actual outcome: "
               f"{pct(report['probability_of_actual_outcome'])}")
    sp = report["actual_scoreline_probability"]
    typer.echo(
        f"  Probability assigned to "
        f"{report['actual_home_goals']}-{report['actual_away_goals']}: "
        + (pct(sp) if sp is not None else "unavailable")
        + f" [{report['actual_scoreline_probability_source']}]"
    )
    typer.echo(f"  Log loss: {report['log_loss']:.4f}")
    typer.echo(f"  Brier contribution: {report['brier']:.4f}")
    typer.echo(f"  RPS contribution: {report['rps']:.4f}")
    typer.echo("")
    typer.echo("Interpretation: this result penalizes the model's live log loss and")
    typer.echo("calibration record; a single match is NOT evidence to retrain or")
    typer.echo("replace the model.")
    typer.echo(f"Report: {report_path}")


@feedback_app.command("summary")
def feedback_summary() -> None:
    """Aggregate realized performance over all recorded results."""
    import numpy as np

    from goalsignal.feedback.results import active_results
    from goalsignal.feedback.scoring import score_prediction
    from goalsignal.ledger.storage import list_entries

    results = active_results()
    if not results:
        typer.echo("No recorded results yet.")
        raise typer.Exit(code=0)
    first_prediction: dict[str, dict] = {}
    for e in list_entries():
        fid = e["payload"].get("fixture_id")
        if fid in results and fid not in first_prediction:
            first_prediction[fid] = e["payload"]

    reports = [
        score_prediction(first_prediction[fid], r)
        for fid, r in results.items()
        if fid in first_prediction
    ]
    if not reports:
        typer.echo("No recorded results have matching predictions.")
        raise typer.Exit(code=0)
    typer.echo(f"Scored matches: {len(reports)}")
    typer.echo(f"Mean log loss:  {np.mean([r['log_loss'] for r in reports]):.4f}")
    typer.echo(f"Mean Brier:     {np.mean([r['brier'] for r in reports]):.4f}")
    typer.echo(f"Mean RPS:       {np.mean([r['rps'] for r in reports]):.4f}")
    typer.echo(f"Outcome correct: "
               f"{sum(r['predicted_outcome_correct'] for r in reports)}/{len(reports)}")
    typer.echo(f"Exact score hits: "
               f"{sum(r['exact_score_correct'] for r in reports)}/{len(reports)}")
    typer.echo("Backtest reference (2010-2025 ensemble): log loss 0.8924.")
    typer.echo("Small samples are noisy; judge drift over many matches, not one.")


# === Milestone B: enrichment source ingestion CLI ===========================
sources_app = typer.Typer(help="Inspect and ingest optional enrichment sources.")
apifootball_app = typer.Typer(help="API-Sports / API-Football v3 live ingestion (optional).")
statsbomb_app = typer.Typer(help="StatsBomb open-data offline ingestion (optional).")
fifa_app = typer.Typer(help="Historical FIFA rankings ingestion (optional).")
fifa_current_app = typer.Typer(help="Frozen June 11, 2026 FIFA World Cup snapshot.")
app.add_typer(sources_app, name="sources")
app.add_typer(apifootball_app, name="api-football")
app.add_typer(statsbomb_app, name="statsbomb")
app.add_typer(fifa_app, name="fifa-rankings")
app.add_typer(fifa_current_app, name="fifa-current")


def _load_env():
    from goalsignal.data.sources.env import load_env_file

    return load_env_file()


@sources_app.command("list")
def sources_list() -> None:
    """List configured enrichment sources, roles, licenses, and configured state."""
    _load_env()
    from goalsignal.data.sources.config import SourcesConfig

    cfg = SourcesConfig.load()
    for s in cfg.sources:
        configured = "configured" if s.is_configured() else "not-configured"
        typer.echo(f"{s.name:15s} role={s.role:16s} [{configured}] license={s.license}")
    typer.echo(f"{len(cfg.sources)} sources. Credentials/paths come from .env (never shown).")


@sources_app.command("validate")
def sources_validate() -> None:
    """Validate all source configs and report env presence (values never shown)."""
    _load_env()
    from goalsignal.data.sources.config import (
        ApiFootballConfig,
        EnrichmentConfig,
        FifaRankingsConfig,
        PlayerFeaturesConfig,
        SourcesConfig,
        StatsBombConfig,
    )
    from goalsignal.data.sources.env import has_env

    SourcesConfig.load()
    EnrichmentConfig.load()
    PlayerFeaturesConfig.load()
    af = ApiFootballConfig.load()
    sb = StatsBombConfig.load()
    fifa = FifaRankingsConfig.load()
    typer.echo("Config files: all load OK.")
    key_state = "present" if has_env(af.credential_env) else "absent"
    typer.echo(f"api-football ({af.vendor}): key {key_state}; "
               f"{af.daily_request_limit}/day, {af.max_requests_per_minute}/min; "
               f"host={af.base_url}")
    typer.echo(f"statsbomb: path {'set' if has_env(sb.data_path_env) else 'unset'}")
    typer.echo(f"fifa-rankings: path {'set' if has_env(fifa.path_env) else 'unset'}")


@sources_app.command("coverage")
def sources_coverage() -> None:
    """Compute and write real source-coverage reports."""
    _load_env()
    from goalsignal.data.sources.coverage import build_source_coverage

    summary = build_source_coverage()
    for s in summary["sources"]:
        typer.echo(f"{s['source']:15s} -> {s['state']}")
    typer.echo("Reports: artifacts/reports/source_coverage_summary.{json,md}, "
               "enrichment_coverage.csv")


@sources_app.command("manifests")
def sources_manifests() -> None:
    """List source-snapshot manifests."""
    d = resolve("artifacts/manifests/sources")
    if not d.exists():
        typer.echo("No source manifests yet.")
        return
    import json

    for p in sorted(d.glob("*.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        typer.echo(f"{m['source']:15s} {m['snapshot_id']}  rows={m['row_count']}  "
                   f"hash={m['content_hash'][:12]}")


def _write_normalized(df, source: str, name: str) -> str:
    out = resolve(f"data/external/{source}/normalized")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _api_football_client():
    from goalsignal.data.sources.api_football import ApiFootballClient
    from goalsignal.data.sources.config import ApiFootballConfig

    return ApiFootballClient(ApiFootballConfig.load())


@apifootball_app.command("probe")
def apifootball_probe() -> None:
    """Single safe live probe of /status (verifies auth, reads quota)."""
    import json
    from datetime import UTC, datetime

    _load_env()
    from goalsignal.data.sources.api_football import (
        ApiFootballError,
        MissingApiKeyError,
    )

    out = resolve("artifacts/reports/api_football_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        meta = _api_football_client().probe()
        out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        typer.echo(f"Probe OK (API-Sports): status 200, plan={meta['subscription_plan']}, "
                   f"requests {meta['requests_current']}/{meta['requests_limit_day']}/day")
        typer.echo(f"Quota headers: {meta['quota_headers']}")
        typer.echo(f"Cache: {meta['cache_path']} (snapshot {meta['snapshot_id']})")
    except MissingApiKeyError as e:
        typer.echo(f"No API key configured: {e}", err=True)
        raise typer.Exit(code=2) from None
    except ApiFootballError as e:
        meta = {"provider": "api-football", "vendor": "API-Sports",
                "auth_configured": True, "auth_verified": False,
                "error_category": type(e).__name__, "error": str(e),
                "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "note": "stopped after one request; not retrying an auth failure. "
                "Verify the key/header/host before concluding the token is invalid."}
        out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        typer.echo(f"Probe FAILED ({type(e).__name__}): {e}", err=True)
        typer.echo(f"Saved error metadata (no secrets): {out}", err=True)
        raise typer.Exit(code=1) from None


@apifootball_app.command("discover-world-cup")
def apifootball_discover_world_cup(
    season: Annotated[int, typer.Option(help="World Cup season year.")] = 2026,
    refresh: Annotated[bool, typer.Option("--refresh", help="Force a live call.")] = False,
) -> None:
    """Discover the API-Football league id for the FIFA World Cup (no guessing)."""
    import json

    _load_env()
    from goalsignal.data.sources.api_football import ApiFootballError

    try:
        result = _api_football_client().discover_world_cup(season, refresh=refresh)
    except ApiFootballError as e:
        typer.echo(f"Discovery failed ({type(e).__name__}): {e}", err=True)
        raise typer.Exit(code=1) from None
    out = resolve("artifacts/reports/api_football_world_cup.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    sel = result.get("selected")
    if sel:
        typer.echo(f"Selected: league_id={sel['league_id']} '{sel['name']}' "
                   f"({sel['country']}), has_{season}_season={sel['has_season']}")
    else:
        typer.echo("No World Cup league found in the response.")
    typer.echo(f"Report: {out}")


@apifootball_app.command("fixtures")
def apifootball_fixtures(
    league: Annotated[int, typer.Option(help="API-Football league id.")],
    season: Annotated[int, typer.Option(help="Season year.")] = 2026,
    refresh: Annotated[bool, typer.Option("--refresh", help="Force a live call.")] = False,
) -> None:
    """Fetch + normalize fixtures for a league/season (cache-first)."""
    import datetime as dt

    _load_env()
    from goalsignal.data.sources.api_football import ApiFootballError
    from goalsignal.data.sources.api_football_normalize import normalize_fixtures

    try:
        data, manifest = _api_football_client().fixtures(
            {"league": league, "season": season}, refresh=refresh)
    except ApiFootballError as e:
        typer.echo(f"Fetch failed ({type(e).__name__}): {e}", err=True)
        raise typer.Exit(code=1) from None
    df = normalize_fixtures(data, manifest["snapshot_id"],
                            dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    path = _write_normalized(df, "api_football", "fixtures")
    typer.echo(f"Fixtures: {len(df)} -> {path} (snapshot {manifest['snapshot_id']})")


@apifootball_app.command("standings")
def apifootball_standings(
    league: Annotated[int, typer.Option(help="API-Football league id.")],
    season: Annotated[int, typer.Option(help="Season year.")] = 2026,
    refresh: Annotated[bool, typer.Option("--refresh", help="Force a live call.")] = False,
) -> None:
    """Fetch + normalize standings (cache-first)."""
    import datetime as dt

    _load_env()
    from goalsignal.data.sources.api_football import ApiFootballError
    from goalsignal.data.sources.api_football_normalize import normalize_standings

    try:
        data, manifest = _api_football_client().standings(
            {"league": league, "season": season}, refresh=refresh)
    except ApiFootballError as e:
        typer.echo(f"Fetch failed ({type(e).__name__}): {e}", err=True)
        raise typer.Exit(code=1) from None
    df = normalize_standings(data, manifest["snapshot_id"],
                             dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    path = _write_normalized(df, "api_football", "standings")
    typer.echo(f"Standings rows: {len(df)} -> {path}")


@apifootball_app.command("lineups")
def apifootball_lineups(
    fixture: Annotated[int, typer.Option(help="Provider fixture id.")],
    refresh: Annotated[bool, typer.Option("--refresh", help="Force a live call.")] = False,
) -> None:
    """Fetch + normalize confirmed lineups for a fixture (cache-first)."""
    import datetime as dt

    _load_env()
    from goalsignal.data.sources.api_football import ApiFootballError
    from goalsignal.data.sources.api_football_normalize import normalize_lineups

    try:
        data, manifest = _api_football_client().lineups(fixture, refresh=refresh)
    except ApiFootballError as e:
        typer.echo(f"Fetch failed ({type(e).__name__}): {e}", err=True)
        raise typer.Exit(code=1) from None
    df = normalize_lineups(data, manifest["snapshot_id"],
                           dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    path = _write_normalized(df, "api_football", "lineups")
    typer.echo(f"Lineup rows: {len(df)} -> {path}"
               + ("" if len(df) else "  (empty: confirmed lineup not yet available)"))


@apifootball_app.command("injuries")
def apifootball_injuries(
    league: Annotated[int, typer.Option(help="API-Football league id.")],
    season: Annotated[int, typer.Option(help="Season year.")] = 2026,
    refresh: Annotated[bool, typer.Option("--refresh", help="Force a live call.")] = False,
) -> None:
    """Fetch + normalize injuries (endpoint exists; coverage measured, not assumed)."""
    import datetime as dt

    _load_env()
    from goalsignal.data.sources.api_football import ApiFootballError
    from goalsignal.data.sources.api_football_normalize import normalize_injuries

    try:
        data, manifest = _api_football_client().injuries(
            {"league": league, "season": season}, refresh=refresh)
    except ApiFootballError as e:
        typer.echo(f"Fetch failed ({type(e).__name__}): {e}", err=True)
        raise typer.Exit(code=1) from None
    df = normalize_injuries(data, manifest["snapshot_id"],
                            dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    path = _write_normalized(df, "api_football", "injuries")
    typer.echo(f"Injury rows: {len(df)} -> {path}"
               + ("" if len(df) else "  (empty: no injuries reported for this competition)"))


@apifootball_app.command("fixture-players")
def apifootball_fixture_players(
    fixture: Annotated[int, typer.Option(help="Provider fixture id.")],
    refresh: Annotated[bool, typer.Option("--refresh", help="Force a live call.")] = False,
) -> None:
    """Fetch + normalize per-player match statistics (cache-first)."""
    import datetime as dt

    _load_env()
    from goalsignal.data.sources.api_football import ApiFootballError
    from goalsignal.data.sources.api_football_normalize import normalize_fixture_players

    try:
        data, manifest = _api_football_client().fixture_players(fixture, refresh=refresh)
    except ApiFootballError as e:
        typer.echo(f"Fetch failed ({type(e).__name__}): {e}", err=True)
        raise typer.Exit(code=1) from None
    df = normalize_fixture_players(data, manifest["snapshot_id"],
                                   dt.datetime.now(dt.UTC).isoformat(timespec="seconds"))
    path = _write_normalized(df, "api_football", "fixture_players")
    typer.echo(f"Fixture-player rows: {len(df)} -> {path}")


@apifootball_app.command("inspect-cache")
def apifootball_inspect_cache() -> None:
    """List cached API-Football raw snapshots and today's request usage."""
    from goalsignal.data.sources.cache import list_snapshots
    from goalsignal.data.sources.config import ApiFootballConfig
    from goalsignal.data.sources.throttle import DailyUsageTracker

    cfg = ApiFootballConfig.load()
    snaps = list_snapshots(cfg.cache_dir)
    used = DailyUsageTracker(cfg.cache_dir).current()
    typer.echo(f"Today's live requests used: {used}/"
               f"{cfg.daily_request_limit - cfg.daily_request_reserve} usable "
               f"({cfg.daily_request_limit}/day minus {cfg.daily_request_reserve} reserve)")
    if not snaps:
        typer.echo("No cached API-Football snapshots.")
        return
    for m in snaps:
        typer.echo(f"{m['snapshot_id']}  {m['endpoint_or_url']}  rows={m['row_count']}  "
                   f"hash={m['content_hash'][:12]}")


@statsbomb_app.command("inspect")
def statsbomb_inspect() -> None:
    """Report whether StatsBomb data is configured (with setup instructions)."""
    _load_env()
    from goalsignal.data.sources.statsbomb import StatsBombDataUnavailable, resolve_statsbomb_path

    try:
        root = resolve_statsbomb_path()
    except StatsBombDataUnavailable as e:
        typer.echo(str(e))
        raise typer.Exit(code=0) from None
    typer.echo(f"StatsBomb data found at {root}")


@statsbomb_app.command("ingest")
def statsbomb_ingest() -> None:
    """Ingest StatsBomb competitions/matches into normalized CSVs (offline)."""
    _load_env()
    from datetime import UTC, datetime

    from goalsignal.data.sources.manifests import build_snapshot_manifest, write_manifest
    from goalsignal.data.sources.statsbomb import (
        SB_ATTRIBUTION,
        SB_LICENSE,
        StatsBombDataUnavailable,
        StatsBombLoader,
        resolve_statsbomb_path,
    )

    try:
        root = resolve_statsbomb_path()
    except StatsBombDataUnavailable as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=0) from None
    loader = StatsBombLoader(root)
    comps, chash = loader.load_competitions()
    path = _write_normalized(comps, "statsbomb", "competitions")
    manifest = build_snapshot_manifest(
        source="statsbomb", role="event_enrichment", endpoint_or_url="competitions.json",
        available_at_semantics="historical (match completion)", license=SB_LICENSE,
        attribution=SB_ATTRIBUTION, content_hash=chash, row_count=len(comps),
        schema_version=1, cache_path=str(root),
        retrieval_timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    write_manifest(manifest)
    typer.echo(f"StatsBomb competitions: {len(comps)} -> {path} "
               f"(snapshot {manifest.snapshot_id})")


@statsbomb_app.command("coverage")
def statsbomb_coverage() -> None:
    """Report StatsBomb coverage (offline)."""
    _load_env()
    import json

    from goalsignal.data.sources.statsbomb import (
        StatsBombDataUnavailable,
        StatsBombLoader,
        resolve_statsbomb_path,
    )

    try:
        root = resolve_statsbomb_path()
    except StatsBombDataUnavailable as e:
        typer.echo(str(e))
        raise typer.Exit(code=0) from None
    typer.echo(json.dumps(StatsBombLoader(root).coverage(), indent=2))


def _fifa_paths():
    """Resolve and validate both FIFA file paths from config + env."""
    import os

    from goalsignal.data.sources.config import FifaRankingsConfig, validate_source_path

    cfg = FifaRankingsConfig.load()
    rankings = validate_source_path(
        os.environ.get(cfg.path_env, ""), kind="file", extensions=(".csv",))
    wc = None
    wc_raw = os.environ.get(cfg.wc_teams_path_env, "")
    if wc_raw:
        wc = validate_source_path(wc_raw, kind="file", extensions=(".csv",))
    return rankings, wc


def _canonical_team_set():
    import pandas as pd

    proc = resolve("data/processed/matches.csv")
    if not proc.exists():
        return None
    m = pd.read_csv(proc, usecols=["home_team", "away_team"])
    return set(m["home_team"]) | set(m["away_team"])


def _require_current_fifa():
    import os

    from goalsignal.data.sources.config import (
        FifaCurrentRankingsConfig,
        validate_source_path,
    )

    _load_env()
    cfg = FifaCurrentRankingsConfig.load()
    return validate_source_path(
        os.environ.get(cfg.path_env, ""), kind="file", extensions=(".csv",)
    )


@fifa_current_app.command("inspect")
def fifa_current_inspect() -> None:
    from goalsignal.data.sources.config import SourcePathError

    try:
        path = _require_current_fifa()
    except SourcePathError as exc:
        typer.echo(f"Current FIFA snapshot not configured/invalid: {exc}")
        raise typer.Exit(code=0) from None
    typer.echo(f"Current FIFA snapshot: {path}")


@fifa_current_app.command("validate")
def fifa_current_validate(
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    from goalsignal.data.sources.config import SourcePathError
    from goalsignal.data.sources.fifa_current import load_current_fifa, write_reports

    try:
        path = _require_current_fifa()
        cfg = _load_config(config, input_dir)
        matches = _canonical_matches(cfg)
        teams = set(matches["home_team"]) | set(matches["away_team"])
        df, manifest, quality = load_current_fifa(path, teams)
    except (SourcePathError, ValueError) as exc:
        typer.echo(f"Validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    write_reports(df, manifest, quality)
    typer.echo(
        f"Current FIFA snapshot OK: {quality['rows']} rows, {quality['groups']} groups, "
        f"release {quality['release_date']}, sha256={quality['sha256']}"
    )


@fifa_current_app.command("compare-elo")
def fifa_current_compare_elo(
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    from goalsignal.data.sources.fifa_current import compare_with_elo, write_reports

    current = _current_fifa(config, input_dir)
    if current is None:
        typer.echo("FIFA_CURRENT_RANKINGS_PATH is not configured.", err=True)
        raise typer.Exit(code=1)
    df, manifest, quality = current
    _matches, live = _live_model(config, input_dir)
    comparison = compare_with_elo(df, live.ratings)
    write_reports(df, manifest, quality, comparison)
    summary = comparison[1]
    typer.echo(
        f"Spearman={summary['spearman_rank_correlation']:.4f}; "
        f"Pearson={summary['pearson_strength_correlation']:.4f}; "
        f"top-10 overlap={summary['top_10_overlap']}"
    )
    typer.echo("Reports: artifacts/reports/fifa_current_2026_vs_elo.{csv,json}")


@fifa_current_app.command("report")
def fifa_current_report() -> None:
    import json

    path = resolve("artifacts/reports/fifa_current_2026_vs_elo.json")
    if not path.exists():
        typer.echo("No comparison report; run `fifa-current compare-elo`.", err=True)
        raise typer.Exit(code=1)
    typer.echo(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))


@fifa_app.command("inspect")
def fifa_inspect() -> None:
    """Report whether the two FIFA files are configured (rankings + WC validation)."""
    _load_env()
    from goalsignal.data.sources.config import SourcePathError

    try:
        rankings, wc = _fifa_paths()
    except SourcePathError as e:
        typer.echo(f"FIFA rankings not configured/invalid: {e}")
        raise typer.Exit(code=0) from None
    typer.echo(f"FIFA rankings timeline: {rankings}")
    typer.echo(f"FIFA WC validation file: {wc if wc else 'not configured (FIFA_WC_TEAMS_PATH)'}")


@fifa_app.command("validate")
def fifa_validate() -> None:
    """Validate FIFA file schemas/paths without writing reports."""
    _load_env()
    from goalsignal.data.sources.config import SourcePathError
    from goalsignal.data.sources.fifa_ingest import load_fifa_historical, quality_report

    try:
        rankings, _wc = _fifa_paths()
    except SourcePathError as e:
        typer.echo(f"Validation failed: {e}", err=True)
        raise typer.Exit(code=1) from None
    df, _ = load_fifa_historical(rankings)
    q = quality_report(df)
    typer.echo(f"FIFA rankings OK: {q['rows']} rows, {q['release_count']} releases, "
               f"{q['teams']} teams, {q['date_min']}..{q['date_max']}")
    typer.echo(f"missing_points={q['missing_points']}, "
               f"duplicate_team_release={q['duplicate_team_release']}, "
               f"normalized_collisions={q['normalized_team_collisions']}")
    typer.echo(f"LIMITATION: {q['limitation']}")


@fifa_app.command("ingest")
def fifa_ingest() -> None:
    """Ingest the FIFA timeline (real schema), reconstruct rank, write 4 reports."""
    _load_env()
    from goalsignal.data.sources.config import SourcePathError
    from goalsignal.data.sources.fifa_ingest import load_fifa_historical, write_reports

    try:
        rankings, _wc = _fifa_paths()
    except SourcePathError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from None
    df, manifest = load_fifa_historical(rankings)
    canonical = _canonical_team_set()
    quality = write_reports(df, manifest, canonical_teams=canonical)
    typer.echo(f"FIFA rankings: {quality['rows']} rows, {quality['release_count']} releases, "
               f"{quality['teams']} teams, {quality['date_min']}..{quality['date_max']}; "
               f"link_rate={quality.get('canonical_team_link_rate')}")
    if canonical is not None:
        from goalsignal.data.sources.readiness import team_alias_audit

        alias = team_alias_audit(df, canonical)
        typer.echo(f"Team aliases: {alias['exact_match']} exact, "
                   f"{alias['alias_assisted_candidates']} alias candidates, "
                   f"{alias['unmatched']} unmatched (review team_source_alias_candidates.csv)")
    typer.echo("Reports: fifa_rankings_{coverage.csv,quality.json,unmatched_teams.csv}, "
               "fifa_release_summary.csv, team_source_alias_candidates.csv")


@fifa_app.command("world-cup-validate")
def fifa_world_cup_validate() -> None:
    """Compare reconstructed historical FIFA ranks to wc_teams.csv per World Cup."""
    _load_env()
    import json

    from goalsignal.data.sources.config import SourcePathError
    from goalsignal.data.sources.fifa_ingest import load_fifa_historical
    from goalsignal.data.sources.fifa_wc_validation import load_wc_teams, write_reports

    try:
        rankings, wc = _fifa_paths()
    except SourcePathError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1) from None
    if wc is None:
        typer.echo("FIFA_WC_TEAMS_PATH not configured; nothing to validate.")
        raise typer.Exit(code=0)
    fifa_df, _ = load_fifa_historical(rankings)
    wc_df, _ = load_wc_teams(wc)
    summary = write_reports(wc_df, fifa_df)
    typer.echo(f"World Cup rank validation ({summary['total']} team-years):")
    typer.echo(json.dumps(summary["by_classification"], indent=2))
    typer.echo("Reports: fifa_world_cup_rank_validation.csv, fifa_world_cup_rank_summary.json")


@fifa_app.command("coverage")
def fifa_coverage() -> None:
    """Print the FIFA ranking quality/coverage summary if ingested."""
    import json

    q = resolve("artifacts/reports/fifa_rankings_quality.json")
    if not q.exists():
        typer.echo("No FIFA ranking quality report yet; run `fifa-rankings ingest`.")
        raise typer.Exit(code=0)
    typer.echo(json.dumps(json.loads(q.read_text(encoding="utf-8")), indent=2))


# === Phase 14-15: source readiness + player-data audit CLI ==================
@sources_app.command("readiness")
def sources_readiness() -> None:
    """Classify each feature family's readiness from the real audit artifacts."""
    _load_env()
    from collections import Counter

    from goalsignal.data.sources.readiness import build_source_readiness

    summary = build_source_readiness()
    by_state = Counter(v["state"] for v in summary["families"].values())
    for fam, info in summary["families"].items():
        typer.echo(f"{fam:28s} {info['state']}")
    typer.echo(f"\nBy state: {dict(by_state)}")
    typer.echo("Reports: source_readiness.{json,md}, enrichment_coverage.csv")


playerdata_app = typer.Typer(help="Transfermarkt-derived player/club audit (read-only, optional).")
app.add_typer(playerdata_app, name="player-data")
squads_app = typer.Typer(help="Official 2026 squad ingestion and player-data readiness.")
app.add_typer(squads_app, name="squads")
squad_model_app = typer.Typer(
    help="Offline 2026 squad-strength scenario challenger (research only)."
)
app.add_typer(squad_model_app, name="squad-model")


def _player_source():
    from goalsignal.data.sources.player_data import PlayerDataSource

    return PlayerDataSource.resolve_from_env()


def _squad_source_inputs():
    from goalsignal.data.sources.config import SquadDataConfig
    from goalsignal.data.sources.squads import (
        load_reviewed_aliases,
        load_squads,
        resolve_optional_reference_path,
        resolve_squad_path,
    )

    _load_env()
    config = SquadDataConfig.load()
    path = resolve_squad_path(config)
    canonical = _canonical_team_set()
    squads, manifest, quality = load_squads(
        path, canonical_teams=canonical, config=config
    )
    alias_path = resolve_optional_reference_path(
        config.player_aliases_path_env, config.player_aliases_default_path
    )
    aliases = load_reviewed_aliases(alias_path)
    return config, path, squads, manifest, quality, aliases


def _squad_links():
    from goalsignal.data.sources.squads import (
        link_squad_players,
        load_seed_link_candidates,
        resolve_optional_reference_path,
        revalidate_reviewed_aliases,
        revalidate_seed_links,
        write_alias_revalidation_reports,
        write_seed_link_reports,
    )

    config, path, squads, manifest, quality, aliases = _squad_source_inputs()
    source = _player_source()
    players = source.read_table("players")
    alias_report, alias_summary = revalidate_reviewed_aliases(
        squads,
        aliases,
        players,
        expected_rows=config.expected_alias_rows,
    )
    write_alias_revalidation_reports(alias_report, alias_summary)
    candidate_path = resolve_optional_reference_path(
        config.link_candidates_path_env, config.link_candidates_default_path
    )
    if candidate_path is None:
        raise ValueError("squad seed-link candidate file is not configured")
    candidates = load_seed_link_candidates(candidate_path)
    seed_report, seed_summary = revalidate_seed_links(squads, candidates, players)
    write_seed_link_reports(seed_report, seed_summary)
    links = link_squad_players(squads, players, alias_report, seed_report)
    links.attrs["seed_summary"] = seed_summary
    links.attrs["alias_summary"] = alias_summary
    links.attrs["players"] = players
    return config, path, squads, manifest, quality, source, links


def _squad_reconciliation(config, squads):
    from goalsignal.data.sources.squads import (
        load_official_extract,
        reconcile_official_extract,
        resolve_optional_reference_path,
        write_reconciliation_reports,
    )

    extract_path = resolve_optional_reference_path(
        config.official_extract_path_env, config.official_extract_default_path
    )
    if extract_path is None:
        raise ValueError("official expanded squad extract is not configured")
    report, summary = reconcile_official_extract(
        squads, load_official_extract(extract_path)
    )
    write_reconciliation_reports(report, summary)
    return summary


@squads_app.command("inspect")
def squads_inspect() -> None:
    """Show configured squad-related paths without reading secrets."""
    import os

    from goalsignal.data.sources.config import SquadDataConfig

    _load_env()
    config = SquadDataConfig.load()
    for label, env_name in (
        ("official squads", config.squads_path_env),
        ("official extract", config.official_extract_path_env),
        ("seed candidates", config.link_candidates_path_env),
        ("availability", config.availability_path_env),
        ("reviewed aliases", config.player_aliases_path_env),
    ):
        raw = os.environ.get(env_name, "")
        default = {
            config.squads_path_env: config.squads_default_path,
            config.official_extract_path_env: config.official_extract_default_path,
            config.link_candidates_path_env: config.link_candidates_default_path,
            config.player_aliases_path_env: config.player_aliases_default_path,
        }.get(env_name, "")
        configured = raw or default or "not configured"
        typer.echo(f"{label:18s} {configured} ({env_name})")
    typer.echo("Template: data/reference/world_cup_2026_squads_template.csv")


@squads_app.command("validate")
def squads_validate(
    cutoff: Annotated[
        str | None,
        typer.Option(help="Optional prediction cutoff; rejects unpublished rows."),
    ] = None,
) -> None:
    """Validate the official squad CSV and its publication-time semantics."""
    from goalsignal.data.sources.squads import (
        SquadDataUnavailable,
        assert_squads_available_at,
    )

    try:
        config, path, squads, manifest, quality, _aliases = _squad_source_inputs()
        reconciliation = _squad_reconciliation(config, squads)
        if cutoff:
            assert_squads_available_at(squads, cutoff)
    except (SquadDataUnavailable, ValueError) as exc:
        typer.echo(f"Squad validation failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Squad source valid: rows={quality['rows']}, teams={quality['teams']}, "
        f"groups={quality['groups']}, snapshot={manifest['snapshot_id']}, path={path}"
    )
    typer.echo(
        f"Official extract: {reconciliation['matched_rows']}/"
        f"{reconciliation['primary_rows']} reconciled"
    )


@squads_app.command("ingest")
def squads_ingest(
    force: Annotated[
        bool, typer.Option("--force", help="Replace generated normalized output.")
    ] = False,
) -> None:
    """Normalize an official squad CSV and write quality/provenance reports."""
    from goalsignal.data.sources.squads import SquadDataUnavailable, write_squad_reports

    try:
        config, _path, squads, manifest, quality, _aliases = _squad_source_inputs()
        reconciliation = _squad_reconciliation(config, squads)
    except (SquadDataUnavailable, ValueError) as exc:
        typer.echo(f"Squad ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    out = resolve(f"artifacts/player_data/squads_{manifest['snapshot_id']}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        typer.echo(f"Refusing to overwrite {out}; pass --force.", err=True)
        raise typer.Exit(code=1)
    normalized = squads[
        SQUAD_NORMALIZED_COLUMNS
    ].copy()
    normalized.to_csv(out, index=False)
    paths = write_squad_reports(squads, manifest, quality)
    typer.echo(
        f"Ingested {len(squads)} squad rows for {quality['teams']} teams; "
        f"normalized={out}"
    )
    typer.echo("Reports: " + ", ".join(str(path) for path in paths))
    typer.echo(f"Official extract reconciliation: {reconciliation}")


SQUAD_NORMALIZED_COLUMNS = [
    "snapshot_date",
    "group",
    "canonical_team",
    "player_name",
    "date_of_birth_normalized",
    "position",
    "club_normalized",
    "shirt_number",
    "squad_status",
    "source_name",
    "source_url_or_reference",
    "source_publication_date",
    "source_player_id",
    "notes",
    "source_row",
    "normalized_player_name",
]


@squads_app.command("coverage")
def squads_coverage() -> None:
    """Write squad coverage, or an explicit missing-source report."""
    from goalsignal.data.sources.squads import (
        SquadDataUnavailable,
        write_missing_source_reports,
        write_squad_reports,
    )

    try:
        _config, _path, squads, manifest, quality, _aliases = _squad_source_inputs()
    except SquadDataUnavailable as exc:
        report = write_missing_source_reports(str(exc))
        typer.echo(f"Squad coverage: {report['state']}")
        typer.echo("Report: artifacts/reports/squad_2026_quality.json")
        return
    write_squad_reports(squads, manifest, quality)
    typer.echo(
        f"Squad coverage: {quality['rows']} players, {quality['teams']} teams, "
        f"{quality['groups']} groups, {quality['conflict_count']} conflicts"
    )


@squads_app.command("link-players")
def squads_link_players() -> None:
    """Link official squad players to Transfermarkt with deterministic evidence."""
    from goalsignal.data.sources.squads import (
        SquadDataUnavailable,
        write_link_reports,
    )

    try:
        _config, _path, _squads, _manifest, _quality, _source, links = _squad_links()
    except SquadDataUnavailable as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    summary = write_link_reports(links)
    typer.echo(
        f"Player links: {summary['linked']}/{summary['total']} "
        f"({summary['link_rate']:.1%}); by class={summary['by_class']}"
    )
    typer.echo("Reports: artifacts/reports/squad_player_*.csv|json")
    typer.echo(
        f"Accepted seed links: {links.attrs['seed_summary']['accepted_deterministic']}; "
        f"reviewed aliases={links.attrs['alias_summary']}"
    )


def _activity_outputs(cutoff: str, force: bool):
    import json

    from goalsignal.data.sources.squads import (
        build_historical_valuations,
        build_player_activity,
        write_link_reports,
    )
    from goalsignal.utils.hashing import sha256_json

    config, _path, _squads, manifest, _quality, source, links = _squad_links()
    write_link_reports(links)
    appearances = source.read_table(
        "appearances",
        columns=[
            "game_id", "player_id", "player_club_id", "date", "competition_id",
            "minutes_played", "goals", "assists", "yellow_cards", "red_cards",
        ],
    )
    lineups = source.read_table(
        "game_lineups",
        columns=["game_id", "player_id", "club_id", "date", "type", "position"],
    )
    activity = build_player_activity(
        links,
        appearances,
        lineups,
        cutoff=cutoff,
        windows=tuple(config.activity_windows_days),
    )
    valuations_raw = source.read_table(
        "player_valuations", columns=["player_id", "date", "market_value_in_eur"]
    )
    tm_snapshot = sha256_json(source.file_hashes())[:16]
    valuations = build_historical_valuations(
        links, valuations_raw, cutoff=cutoff, source_snapshot_id=tm_snapshot
    )
    snapshot = manifest["snapshot_id"]
    out = resolve("artifacts/player_data")
    reports = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    activity_path = out / f"player_activity_{snapshot}.csv"
    valuation_path = out / f"player_valuations_{snapshot}.csv"
    if not force and (activity_path.exists() or valuation_path.exists()):
        raise FileExistsError("player outputs exist; pass --force to replace generated files")
    activity.to_csv(activity_path, index=False)
    valuations.to_csv(valuation_path, index=False)
    coverage_rows = []
    for dimension in ("overall", "national_team", "group", "position", "match_class"):
        grouped = [("all", activity)] if dimension == "overall" else activity.groupby(dimension)
        for value, block in grouped:
            coverage_rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "players": len(block),
                    "identity_available": int(
                        block["canonical_player_id"].fillna("").ne("").sum()
                    ),
                    "local_snapshot_available": int(
                        block["local_snapshot_available"].fillna(False).sum()
                    ),
                    "minutes_30d_available": int(block["minutes_30d"].notna().sum()),
                    "minutes_90d_available": int(block["minutes_90d"].notna().sum()),
                    "minutes_180d_available": int(block["minutes_180d"].notna().sum()),
                    "minutes_365d_available": int(block["minutes_365d"].notna().sum()),
                    "starts_90d_available": int(block["starts_90d"].notna().sum()),
                    "cutoff": cutoff,
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(reports / "player_activity_coverage.csv", index=False)
    activity.isna().mean().rename("missing_rate").rename_axis("field").reset_index().to_csv(
        reports / "player_activity_missingness.csv", index=False
    )
    temporal = {
        "valid": True,
        "cutoff": cutoff,
        "rows": len(activity),
        "strictly_prior": True,
        "target_match_excluded_when_provided": True,
        "current_profile_fields_used": False,
        "club_and_national_team_activity_separate": True,
    }
    (reports / "player_activity_temporal_validation.json").write_text(
        json.dumps(temporal, indent=2), encoding="utf-8"
    )
    valuation_coverage = []
    for dimension in ("overall", "national_team", "position"):
        grouped = (
            [("all", valuations)]
            if dimension == "overall"
            else valuations.groupby(dimension)
        )
        for value, block in grouped:
            ages = pd.to_numeric(block["valuation_age_days"], errors="coerce")
            valuation_coverage.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "players": len(block),
                    "local_snapshot_available": int(
                        block["local_snapshot_available"].fillna(False).sum()
                    ),
                    "available": int(block["available"].sum()),
                    "coverage": float(block["available"].mean()),
                    "coverage_among_local": (
                        float(
                            block.loc[
                                block["local_snapshot_available"].fillna(False),
                                "available",
                            ].mean()
                        )
                        if block["local_snapshot_available"].fillna(False).any()
                        else None
                    ),
                    "median_valuation_age_days": ages.median(),
                    "stale_over_365_days": int(ages.gt(365).sum()),
                    "cutoff": cutoff,
                }
            )
    pd.DataFrame(valuation_coverage).to_csv(
        reports / "player_valuation_coverage.csv", index=False
    )
    valuations[
        [
            "national_team",
            "player_name",
            "identity_status",
            "local_snapshot_available",
            "valuation_date",
            "valuation_age_days",
            "available",
        ]
    ].to_csv(reports / "player_valuation_age.csv", index=False)
    return activity, valuations, links, activity_path, valuation_path


@squads_app.command("activity")
def squads_activity(
    cutoff: Annotated[str, typer.Option(help="Prediction cutoff timestamp.")] = "2026-06-15",
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Build cutoff-safe club activity and historical valuations for linked squads."""
    from goalsignal.data.sources.squads import build_squad_aggregates

    try:
        activity, valuations, links, activity_path, valuation_path = _activity_outputs(
            cutoff, force
        )
        config, _path, squads, _manifest, _quality, _aliases = _squad_source_inputs()
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Squad activity failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    aggregates = build_squad_aggregates(squads, links, activity, valuations, config)
    reports = resolve("artifacts/reports")
    aggregates["activity"].to_csv(
        reports / "squad_2026_activity_summary.csv", index=False
    )
    aggregates["position"].to_csv(
        reports / "squad_2026_position_summary.csv", index=False
    )
    aggregates["depth"].to_csv(
        reports / "squad_2026_depth_summary.csv", index=False
    )
    aggregates["missingness"].to_csv(
        reports / "squad_2026_missingness.csv", index=False
    )
    portugal = aggregates["player_level"][
        aggregates["player_level"]["national_team"].eq("Portugal")
    ].copy()
    portugal.to_csv(reports / "portugal_squad_activity.csv", index=False)
    linked = int(portugal["canonical_player_id"].fillna("").ne("").sum()) if len(portugal) else 0
    accepted_local = int(
        portugal.get("local_snapshot_available", pd.Series(dtype=bool))
        .fillna(False)
        .sum()
    )
    accepted_web_only = int(
        portugal.get("identity_status", pd.Series(dtype=str))
        .eq("accepted_web_only")
        .sum()
    )
    minutes_cov = float(portugal.get("minutes_90d", pd.Series(dtype=float)).notna().mean()) \
        if len(portugal) else None
    valuation_cov = float(
        portugal.get("historical_valuation", pd.Series(dtype=float)).notna().mean()
    ) if len(portugal) else None
    lineup_path = reports / "national_team_lineup_coverage.csv"
    portugal_lineup = None
    if lineup_path.exists():
        lineup = pd.read_csv(lineup_path)
        matched = lineup[lineup["national_team"].eq("Portugal")]
        portugal_lineup = matched.iloc[0].to_dict() if len(matched) else None
    position_lines = []
    for position, block in portugal.groupby("position_group"):
        position_lines.append(
            f"- {position} coverage: "
            f"{int(block['canonical_player_id'].fillna('').ne('').sum())}/{len(block)} "
            f"linked; 90-day minutes available for "
            f"{int(block.get('minutes_90d', pd.Series(index=block.index)).notna().sum())}"
        )
    depth = aggregates["depth"][
        aggregates["depth"]["national_team"].eq("Portugal")
    ]
    portugal_md = [
        "# Portugal Squad Data Audit",
        "",
        "Descriptive source audit only. No forecast or title probability was changed.",
        "",
        f"- Official squad rows: {len(portugal)}",
        f"- Confident Transfermarkt links: {linked}",
        f"- Accepted local identities: {accepted_local}",
        f"- Accepted web-only identities: {accepted_web_only}",
        f"- Recent club-minutes coverage: {minutes_cov}",
        f"- Historical-valuation coverage: {valuation_cov}",
        f"- Ambiguous/unmatched players: {len(portugal) - linked}",
        *[
            f"- {days}-day minutes coverage: "
            f"{float(portugal[f'minutes_{days}d'].notna().mean()) if len(portugal) else None}"
            for days in config.activity_windows_days
        ],
        f"- 90-day starts coverage: "
        f"{float(portugal.get('starts_90d', pd.Series(dtype=float)).notna().mean())}",
        *position_lines,
        (
            "- Top-11/top-15/top-23 activity minutes: "
            + (
                f"{depth.iloc[0]['top_11_minutes_90d']}/"
                f"{depth.iloc[0]['top_15_minutes_90d']}/"
                f"{depth.iloc[0]['top_23_minutes_90d']}"
                if len(depth)
                else "unavailable"
            )
        ),
        (
            "- National-team lineup history: "
            f"{portugal_lineup['readiness']} "
            f"({portugal_lineup['games_with_lineups']} dated lineups)"
            if portugal_lineup
            else "- National-team lineup history: not audited"
        ),
        "- Expected-XI modeling: not ready until international lineup history is adequate.",
        "- Confirmed-lineup modeling: blocked by the API-Football plan.",
    ]
    (reports / "portugal_squad_data_audit.md").write_text(
        "\n".join(portugal_md) + "\n", encoding="utf-8"
    )
    typer.echo(
        f"Activity rows={len(activity)}, valuation rows={len(valuations)}; "
        f"{activity_path}; {valuation_path}"
    )


@squads_app.command("readiness")
def squads_readiness() -> None:
    """Classify squad feature families without training or deployment."""
    import json
    import os

    from goalsignal.data.sources.squads import (
        SquadDataUnavailable,
        build_feature_readiness,
        write_missing_source_reports,
        write_readiness_reports,
    )

    _load_env()
    try:
        *_rest, links = _squad_links()
        rate = (
            float(links["canonical_player_id"].fillna("").ne("").mean())
            if len(links)
            else None
        )
        local_rate = (
            float(links["local_snapshot_available"].fillna(False).mean())
            if len(links)
            else None
        )
        squad_available = True
    except SquadDataUnavailable as exc:
        write_missing_source_reports(str(exc))
        rate = None
        local_rate = None
        squad_available = False
    statsbomb_available = bool(os.environ.get("STATSBOMB_DATA_PATH"))
    statsbomb_report = resolve(
        "artifacts/reports/statsbomb_lineup_continuity_readiness.json"
    )
    statsbomb_report.write_text(
        json.dumps(
            {
                "state": "configured_not_audited"
                if statsbomb_available
                else "not_configured",
                "coverage": None,
                "note": "StatsBomb is optional and is never downloaded automatically.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lineup_path = resolve("artifacts/reports/national_team_lineup_coverage.csv")
    international_ready = False
    if lineup_path.exists():
        lineup = pd.read_csv(lineup_path)
        international_ready = bool(
            len(lineup) and lineup["readiness"].isin(["strong", "partial"]).any()
        )
    readiness = build_feature_readiness(
        squad_available=squad_available,
        identity_link_rate=rate,
        local_snapshot_rate=local_rate,
        statsbomb_available=statsbomb_available,
        international_lineup_ready=international_ready,
    )
    paths = write_readiness_reports(readiness)
    typer.echo(
        f"Squad source={'available' if squad_available else 'missing'}; "
        f"identity_link_rate={rate}; local_snapshot_rate={local_rate}; "
        f"reports={paths[0]}, {paths[1]}"
    )


@squads_app.command("team")
def squads_team(
    team: Annotated[str, typer.Option("--team")],
) -> None:
    """Print one team's squad-data readiness and descriptive player rows."""
    path = resolve("artifacts/reports/squad_2026_activity_summary.csv")
    player_path = resolve("artifacts/reports/portugal_squad_activity.csv")
    if not path.exists():
        typer.echo("No squad activity report; run `squads activity`.", err=True)
        raise typer.Exit(code=1)
    summary = pd.read_csv(path)
    row = summary[summary["national_team"].str.casefold().eq(team.casefold())]
    if row.empty:
        typer.echo(f"Team not found: {team}", err=True)
        raise typer.Exit(code=1)
    typer.echo(row.to_string(index=False))
    if team.casefold() == "portugal" and player_path.exists():
        typer.echo(pd.read_csv(player_path).to_string(index=False))


@playerdata_app.command("activity")
def playerdata_activity(
    cutoff: Annotated[str, typer.Option(help="Prediction cutoff timestamp.")] = "2026-06-15",
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Build linked squad-player activity from dated Transfermarkt observations."""
    squads_activity(cutoff=cutoff, force=force)


@playerdata_app.command("valuations")
def playerdata_valuations(
    cutoff: Annotated[str, typer.Option(help="Prediction cutoff timestamp.")] = "2026-06-15",
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Build latest historical valuations strictly before a cutoff."""
    try:
        _activity, valuations, _links, _activity_path, valuation_path = _activity_outputs(
            cutoff, force
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Player valuation extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"Historical valuations: {len(valuations)} rows -> {valuation_path}")


@playerdata_app.command("lineup-coverage")
def playerdata_lineup_coverage() -> None:
    """Measure dated national-team lineup coverage separately from club lineups."""
    from goalsignal.data.sources.player_data import PlayerDataUnavailable
    from goalsignal.data.sources.squads import national_lineup_coverage

    _load_env()
    try:
        source = _player_source()
    except PlayerDataUnavailable as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=0) from None
    links_path = resolve("artifacts/reports/squad_player_links.csv")
    links = pd.read_csv(links_path) if links_path.exists() else None
    coverage = national_lineup_coverage(source, links)
    out = resolve("artifacts/reports/national_team_lineup_coverage.csv")
    coverage.to_csv(out, index=False)
    counts = coverage["readiness"].value_counts().to_dict() if len(coverage) else {}
    lines = [
        "# National-Team Lineup Readiness",
        "",
        "Transfermarkt is club-centric. Club lineups are never treated as national-team "
        "lineups.",
        "",
        f"Readiness counts: {counts}",
    ]
    md = resolve("artifacts/reports/national_team_lineup_readiness.md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"National-team lineup coverage: {counts}; reports={out}, {md}")


def _squad_challenger_features(force: bool = False):
    import json

    from goalsignal.data.sources.player_data import PlayerDataSource
    from goalsignal.tournament.squad_challenger import (
        SquadChallengerConfig,
        build_team_squad_features,
        feature_artifact_version,
        squad_source_hashes,
    )
    from goalsignal.utils.hashing import sha256_file, sha256_json

    _load_env()
    config = SquadChallengerConfig.load()
    player_dir = resolve("artifacts/player_data")
    activity_paths = sorted(player_dir.glob("player_activity_*.csv"))
    valuation_paths = sorted(player_dir.glob("player_valuations_*.csv"))
    if len(activity_paths) != 1 or len(valuation_paths) != 1:
        raise ValueError(
            "expected exactly one current squad activity and valuation artifact"
        )
    activity_path, valuation_path = activity_paths[0], valuation_paths[0]
    activity = pd.read_csv(activity_path)
    valuations = pd.read_csv(valuation_path)
    if set(activity["prediction_cutoff"].astype(str)) != {
        pd.Timestamp(config.prediction_cutoff).isoformat()
    }:
        raise ValueError("squad activity cutoff does not match challenger config")
    source = PlayerDataSource.resolve_from_env()
    source_hashes = {
        **squad_source_hashes(),
        "activity_artifact": sha256_file(activity_path),
        "valuation_artifact": sha256_file(valuation_path),
        "transfermarkt_snapshot": sha256_json(source.file_hashes()),
    }
    version = feature_artifact_version(config, source_hashes)
    out = resolve(Path("artifacts/features/squad_2026") / version)
    feature_path = out / "team_squad_features.csv"
    metadata_path = out / "metadata.json"
    if metadata_path.exists() and not force:
        return config, pd.read_csv(feature_path), version, feature_path, metadata_path
    frame = build_team_squad_features(
        activity, valuations, config, source_hashes=source_hashes
    )
    if len(frame) != 48:
        raise ValueError(f"expected 48 squad feature rows, got {len(frame)}")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(feature_path, index=False)
    metadata = {
        "research_status": "offline scenario analysis; not trained or deployed",
        "feature_version": config.feature_version,
        "config_hash": config.config_hash,
        "artifact_version": version,
        "prediction_cutoff": config.prediction_cutoff,
        "source_hashes": source_hashes,
        "coverage_thresholds": config.coverage,
        "eligible_teams": frame.loc[
            frame["coverage_eligible"], "national_team"
        ].tolist(),
        "fallback_teams": frame.loc[
            ~frame["coverage_eligible"], "national_team"
        ].tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    reports = resolve("artifacts/reports")
    reports.mkdir(parents=True, exist_ok=True)
    coverage_columns = [
        "national_team",
        "identity_coverage",
        "local_activity_coverage",
        "valuation_coverage",
        "goalkeeper_local_coverage",
        "minimum_position_local_coverage",
        "stale_valuation_proportion",
        "coverage_confidence",
        "coverage_eligible",
        "fallback_used",
        "log_goal_adjustment",
    ]
    frame[coverage_columns].to_csv(
        reports / "squad_adjustment_coverage.csv", index=False
    )
    frame.loc[~frame["coverage_eligible"], coverage_columns].to_csv(
        reports / "squad_adjustment_fallbacks.csv", index=False
    )
    return config, frame, version, feature_path, metadata_path


def _squad_tournament_context(features: pd.DataFrame):
    from goalsignal.feedback.results import active_results
    from goalsignal.tournament.bracket_2026 import OfficialBracket
    from goalsignal.tournament.fixtures_2026 import derive_2026_group_stage
    from goalsignal.tournament.full_simulator import apply_official_group_letters
    from goalsignal.tournament.model_adapter import RatingsGoalAdapter
    from goalsignal.tournament.simulator import validate_completed_overlay
    from goalsignal.tournament.squad_challenger import SquadScenarioAdapter

    matches, live = _live_model(Path("config/data.yaml"), None)
    groups, fixtures = derive_2026_group_stage(matches)
    active = active_results()
    validate_completed_overlay(fixtures, active)
    fifa = _current_fifa(Path("config/data.yaml"), None)
    if fifa is None:
        raise ValueError("FIFA_CURRENT_RANKINGS_PATH is required")
    fifa_frame, fifa_manifest, _quality = fifa
    official_groups = {
        group: list(block["canonical_team"])
        for group, block in fifa_frame.groupby("group", sort=True)
    }
    groups, fixtures = apply_official_group_letters(
        groups, fixtures, official_groups
    )
    base = RatingsGoalAdapter(live.ratings, live.goal_model)
    challenger = SquadScenarioAdapter(base, features)
    return (
        live,
        groups,
        fixtures,
        OfficialBracket.load(),
        base,
        challenger,
        fifa_frame,
        fifa_manifest,
    )


@squad_model_app.command("build-features")
def squad_model_build_features(
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Build versioned 48-team squad scenario features."""
    try:
        config, frame, version, feature_path, metadata_path = (
            _squad_challenger_features(force)
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Squad feature build failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"Research features {version}: {len(frame)} teams; "
        f"eligible={int(frame['coverage_eligible'].sum())}; "
        f"fallback={int((~frame['coverage_eligible']).sum())}"
    )
    typer.echo(f"Config hash: {config.config_hash}")
    typer.echo(f"Artifacts: {feature_path}, {metadata_path}")


@squad_model_app.command("inspect")
def squad_model_inspect() -> None:
    """Show squad ranks, confidence, and bounded adjustments."""
    _config, frame, version, feature_path, _metadata = _squad_challenger_features()
    columns = [
        "national_team",
        "score_s7_coverage_shrunk",
        "coverage_confidence",
        "log_goal_adjustment",
        "fallback_used",
    ]
    typer.echo(f"Squad scenario features: {version} ({feature_path})")
    typer.echo(
        frame.sort_values("score_s7_coverage_shrunk", ascending=False)[columns]
        .head(20)
        .to_string(index=False)
    )


@squad_model_app.command("predict")
def squad_model_predict(
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Write research-only squad-aware predictions for remaining group fixtures."""
    import json

    import numpy as np

    from goalsignal.feedback.results import result_store_hash
    from goalsignal.tournament.squad_challenger import (
        base_outcome_probabilities,
    )
    from goalsignal.utils.hashing import sha256_json

    config, features, feature_version, _path, _metadata = (
        _squad_challenger_features()
    )
    live, _groups, fixtures, bracket, base, challenger, _fifa, fifa_manifest = (
        _squad_tournament_context(features)
    )
    result_hash = result_store_hash()
    version = sha256_json(
        {
            "challenger": config.challenger_version,
            "feature_version": feature_version,
            "result_hash": result_hash,
            "bracket_hash": bracket.config_hash,
        }
    )[:16]
    out = resolve(Path("artifacts/research_predictions") / version)
    csv_path, metadata_path = out / "remaining_group_matches.csv", out / "metadata.json"
    if metadata_path.exists() and not force:
        typer.echo(f"Refusing to overwrite {out}; pass --force.", err=True)
        raise typer.Exit(code=1)
    indexed = features.set_index("national_team")
    rows = []
    for fixture in fixtures:
        if fixture.played:
            continue
        base_probs = base_outcome_probabilities(
            base, fixture.home, fixture.away, fixture.neutral
        )
        squad_probs = challenger.outcome_probabilities(
            fixture.home, fixture.away, fixture.neutral
        )
        base_lam = base.expected_goals(fixture.home, fixture.away, fixture.neutral)
        squad_lam = challenger.expected_goals(
            fixture.home, fixture.away, fixture.neutral
        )
        home = indexed.loc[fixture.home]
        away = indexed.loc[fixture.away]
        feature_differences = {
            "diff_activity_score": home["score_activity"] - away["score_activity"],
            "diff_recent_minutes": home["z_minutes_90d_total"]
            - away["z_minutes_90d_total"],
            "diff_recent_starts": home["score_starts"] - away["score_starts"],
            "diff_goalkeeper_activity": home["score_goalkeeper"]
            - away["score_goalkeeper"],
            "diff_defender_activity": home["z_defender_active_90d"]
            - away["z_defender_active_90d"],
            "diff_midfielder_activity": home["z_midfielder_active_90d"]
            - away["z_midfielder_active_90d"],
            "diff_forward_activity": home["z_forward_active_90d"]
            - away["z_forward_active_90d"],
            "diff_valuation": home["score_valuation"] - away["score_valuation"],
            "diff_top_11_valuation": home["z_valuation_top_11"]
            - away["z_valuation_top_11"],
            "diff_top_15_valuation": home["z_valuation_top_15"]
            - away["z_valuation_top_15"],
            "diff_top_23_valuation": home["z_valuation_top_23"]
            - away["z_valuation_top_23"],
            "diff_depth": home["score_depth"] - away["score_depth"],
            "diff_inactivity": home["z_inactive_90d"] - away["z_inactive_90d"],
            "diff_activity_coverage": home["z_local_activity_coverage"]
            - away["z_local_activity_coverage"],
            "diff_valuation_coverage": home["z_valuation_coverage"]
            - away["z_valuation_coverage"],
        }
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "group": fixture.group,
                "home_team": fixture.home,
                "away_team": fixture.away,
                "base_home_win": base_probs[0],
                "base_draw": base_probs[1],
                "base_away_win": base_probs[2],
                "squad_home_win": squad_probs[0],
                "squad_draw": squad_probs[1],
                "squad_away_win": squad_probs[2],
                "max_absolute_change": float(
                    np.max(np.abs(squad_probs - base_probs))
                ),
                "base_home_xg": base_lam[0],
                "base_away_xg": base_lam[1],
                "squad_home_xg": squad_lam[0],
                "squad_away_xg": squad_lam[1],
                "home_coverage_confidence": home["coverage_confidence"],
                "away_coverage_confidence": away["coverage_confidence"],
                "fallback_used": bool(
                    home["fallback_used"] or away["fallback_used"]
                ),
                "feature_version": feature_version,
                "challenger_version": config.challenger_version,
                "prediction_cutoff": config.prediction_cutoff,
                "result_store_hash": result_hash,
                **feature_differences,
            }
        )
    frame = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    metadata = {
        "research_status": "scenario analysis; not trained or deployed",
        "current_world_cup_results_used_as_labels": False,
        "completed_fixtures_excluded": len(fixtures) - len(frame),
        "remaining_fixture_count": len(frame),
        "feature_version": feature_version,
        "challenger_version": config.challenger_version,
        "config_hash": config.config_hash,
        "model_version": live.model_version,
        "result_store_hash": result_hash,
        "official_bracket_hash": bracket.config_hash,
        "official_bracket_manifest": bracket.source_manifest,
        "fifa_snapshot_id": fifa_manifest["snapshot_id"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Research predictions: {len(frame)} remaining fixtures -> {csv_path}")
    typer.echo(f"Metadata: {metadata_path}")


def _markdown_table(frame: pd.DataFrame, title: str, note: str = "") -> str:
    headers = list(frame.columns)
    lines = [
        f"# {title}",
        "",
        note,
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _current_base_simulation(result_hash: str) -> tuple[Path, pd.DataFrame, dict]:
    import json

    matches = []
    for meta_path in resolve("artifacts/simulations").glob(
        "*/wc2026_tournament_meta.json"
    ):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("result_store_hash") == result_hash
            and meta.get("n_sims") == 100_000
            and "simulation_version" in meta
            and "challenger_version" not in meta
        ):
            matches.append((meta_path.stat().st_mtime, meta_path, meta))
    if not matches:
        raise ValueError("no verified 100,000-simulation base artifact for result store")
    _mtime, meta_path, meta = max(matches)
    advancement = pd.read_csv(meta_path.parent / "wc2026_team_advancement.csv")
    return meta_path.parent, advancement, meta


def _write_squad_research_reports(
    result,
    base_advancement: pd.DataFrame,
    features: pd.DataFrame,
    config,
    predictions: pd.DataFrame,
    live,
) -> list[Path]:
    import numpy as np

    from goalsignal.tournament.reporting import advancement_frame

    reports = resolve("artifacts/reports")
    reports.mkdir(parents=True, exist_ok=True)
    challenger = advancement_frame(result)
    base = base_advancement.rename(
        columns={column: f"base_{column}" for column in base_advancement if column != "team"}
    )
    compare = challenger.merge(base, on="team", how="left")
    for stage in (
        "round_of_32",
        "round_of_16",
        "quarterfinal",
        "semifinal",
        "final",
        "champion",
    ):
        compare[f"delta_{stage}"] = (
            compare[f"p_{stage}"] - compare[f"base_p_{stage}"]
        )
    feature_lookup = features.set_index("national_team")
    compare["fallback_used"] = compare["team"].map(
        feature_lookup["fallback_used"]
    )
    compare["change_interpretation"] = (
        "scenario change; squad and bracket effects are not causally separated"
    )
    compare_path = reports / "base_vs_squad_advancement.csv"
    compare.to_csv(compare_path, index=False)
    largest = pd.concat(
        [
            compare.nlargest(8, "delta_champion"),
            compare.nsmallest(8, "delta_champion"),
        ]
    ).drop_duplicates("team")
    compare_md = reports / "base_vs_squad_advancement.md"
    compare_md.write_text(
        _markdown_table(
            largest[
                [
                    "team",
                    "base_p_champion",
                    "p_champion",
                    "delta_champion",
                    "fallback_used",
                ]
            ],
            "Base vs Squad Scenario Advancement",
            "Differences are scenario changes, not validated predictive improvements.",
        ),
        encoding="utf-8",
    )

    ratings = pd.Series(live.ratings, name="rating")
    rating_rank = ratings.rank(ascending=False, method="min")
    squad_rank = feature_lookup["score_s7_coverage_shrunk"].rank(
        ascending=False, method="min"
    )
    r32_expected_elo = {}
    for team in result.teams:
        weighted = total = 0.0
        for number in range(73, 89):
            for pair, count in result.matchup_counts[number].items():
                if team in pair:
                    opponent = pair[1] if pair[0] == team else pair[0]
                    weighted += count * live.ratings.get(opponent, 1500.0)
                    total += count
        r32_expected_elo[team] = weighted / total if total else np.nan
    path_rank = pd.Series(r32_expected_elo).rank(ascending=True, method="min")
    contender = compare[
        compare["team"].isin(config.contender_report_teams)
    ].copy()
    contender["relative_title_change"] = (
        contender["delta_champion"] / contender["base_p_champion"].replace(0, np.nan)
    )
    contender["squad_strength_rank"] = contender["team"].map(squad_rank)
    contender["base_team_strength_rank"] = contender["team"].map(rating_rank)
    contender["path_difficulty_rank"] = contender["team"].map(path_rank)
    contender["coverage_confidence"] = contender["team"].map(
        feature_lookup["coverage_confidence"]
    )
    contender = contender.rename(
        columns={
            "base_p_champion": "base_title_probability",
            "p_champion": "squad_title_probability",
            "delta_champion": "absolute_change",
        }
    )
    contender_columns = [
        "team",
        "base_title_probability",
        "squad_title_probability",
        "absolute_change",
        "relative_title_change",
        "squad_strength_rank",
        "base_team_strength_rank",
        "path_difficulty_rank",
        "coverage_confidence",
        "fallback_used",
    ]
    contender = contender[contender_columns].sort_values(
        "squad_title_probability", ascending=False
    )
    contender_csv = reports / "squad_contender_comparison.csv"
    contender_md = reports / "squad_contender_comparison.md"
    contender.to_csv(contender_csv, index=False)
    contender_md.write_text(
        _markdown_table(
            contender,
            "Squad Scenario Contender Comparison",
            "Path rank is expected Round-of-32 opponent Elo among qualified runs.",
        ),
        encoding="utf-8",
    )

    portugal_features = features[features["national_team"].eq("Portugal")].copy()
    portugal_predictions = predictions[
        predictions["home_team"].eq("Portugal")
        | predictions["away_team"].eq("Portugal")
    ].copy()
    portugal_csv = reports / "portugal_squad_challenger.csv"
    pd.concat(
        [
            portugal_features.assign(record_type="team_features"),
            portugal_predictions.assign(record_type="remaining_fixture"),
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(portugal_csv, index=False)
    portugal_md = reports / "portugal_squad_challenger.md"
    rank = int(squad_rank["Portugal"])
    portugal_md.write_text(
        "# Portugal Squad Scenario Challenger\n\n"
        "Offline sensitivity analysis only; no expected XI is inferred and no "
        "production probability is changed.\n\n"
        f"- Squad-strength rank: {rank}/48\n"
        f"- Coverage confidence: "
        f"{float(feature_lookup.loc['Portugal', 'coverage_confidence']):.3f}\n"
        f"- Bounded log-goal adjustment: "
        f"{float(feature_lookup.loc['Portugal', 'log_goal_adjustment']):.4f}\n"
        f"- Base Elo rank: {int(rating_rank['Portugal'])}\n\n"
        + _markdown_table(
            portugal_predictions[
                [
                    "home_team",
                    "away_team",
                    "base_home_win",
                    "base_draw",
                    "base_away_win",
                    "squad_home_win",
                    "squad_draw",
                    "squad_away_win",
                ]
            ],
            "Remaining Portugal Group Matches",
        ),
        encoding="utf-8",
    )
    return [compare_path, compare_md, contender_csv, contender_md, portugal_csv, portugal_md]


def _write_portugal_path(result, live, features: pd.DataFrame) -> tuple[Path, Path]:
    import numpy as np

    trace = result.target_trace
    if not trace or trace["team"] != "Portugal":
        raise ValueError("Portugal trace is missing")
    reports = resolve("artifacts/reports")
    rows = []
    expected_elo = trace["expected_opponent_elo"]
    expected_squad = trace["expected_opponent_squad_strength"]
    for stage in sorted(set(expected_elo) | set(expected_squad)):
        rows.append(
            {
                "record_type": "expected_opponent_strength",
                "condition": "unconditional_given_round_reached",
                "round": stage,
                "opponent": "",
                "probability": np.nan,
                "expected_opponent_elo": expected_elo.get(stage, np.nan),
                "opponent_squad_strength": expected_squad.get(stage, np.nan),
            }
        )
    for stage, opponents in trace["opponent_counts"].items():
        for opponent, count in sorted(
            opponents.items(), key=lambda item: item[1], reverse=True
        ):
            rows.append(
                {
                    "record_type": "opponent",
                    "condition": "unconditional",
                    "round": stage,
                    "opponent": opponent,
                    "probability": count / result.n_sims,
                    "expected_opponent_elo": live.ratings.get(opponent, 1500.0),
                    "opponent_squad_strength": float(
                        features.set_index("national_team").loc[
                            opponent, "score_s7_coverage_shrunk"
                        ]
                    ),
                }
            )
    for position, total in trace["conditional_totals"].items():
        stage_counts = trace["conditional_advancement"].get(str(position), {})
        for stage in (
            "round_of_32",
            "round_of_16",
            "quarterfinal",
            "semifinal",
            "final",
            "champion",
        ):
            rows.append(
                {
                    "record_type": "conditional_advancement",
                    "condition": f"finish_{position}",
                    "round": stage,
                    "opponent": "",
                    "probability": stage_counts.get(stage, 0) / total if total else np.nan,
                    "expected_opponent_elo": np.nan,
                    "opponent_squad_strength": np.nan,
                }
            )
    frame = pd.DataFrame(rows)
    csv_path = reports / "portugal_path_difficulty.csv"
    md_path = reports / "portugal_path_difficulty.md"
    frame.to_csv(csv_path, index=False)
    finish = trace["finish_counts"]
    r32 = trace["opponent_counts"].get("round_of_32", {})
    r16 = trace["opponent_counts"].get("round_of_16", {})
    croatia = r32.get("Croatia", 0) / result.n_sims
    spain = r16.get("Spain", 0) / result.n_sims
    portugal = result.advancement_probs["Portugal"]
    md_path.write_text(
        "# Portugal Path Difficulty\n\n"
        "All opponents are probabilistic; none is described as certain.\n\n"
        f"- Finish first in Group K: {finish.get(1, 0) / result.n_sims:.3%}\n"
        f"- Finish second: {finish.get(2, 0) / result.n_sims:.3%}\n"
        f"- Finish third: {finish.get(3, 0) / result.n_sims:.3%}\n"
        f"- Qualify as a best third-place team: "
        f"{trace['qualifying_third_probability']:.3%}\n"
        f"- Face Croatia in Round of 32: {croatia:.3%}\n"
        f"- Face Spain in Round of 16: {spain:.3%}\n"
        f"- Face a top-five Elo team before the quarterfinals: "
        f"{trace['top_team_before_quarterfinal_probability']:.3%}\n"
        f"- Scenario title probability: {portugal['champion']:.3%}\n\n"
        + _markdown_table(
            frame[frame["record_type"].eq("conditional_advancement")],
            "Conditional Advancement by Group Finish",
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


@tournament_app.command("simulate-squad")
def tournament_simulate_squad(
    sims: Annotated[int, typer.Option(help="Research simulation count.")] = 100_000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 20260612,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Run the full squad-aware research scenario through the champion."""
    import json
    import resource
    import time

    from goalsignal.feedback.results import result_store_hash
    from goalsignal.tournament.full_simulator import (
        check_full_invariants,
        simulate_full_tournament,
    )
    from goalsignal.tournament.reporting import write_full_simulation
    from goalsignal.utils.hashing import sha256_json

    config, features, feature_version, _feature_path, _feature_meta = (
        _squad_challenger_features()
    )
    live, groups, fixtures, bracket, _base, challenger, _fifa, fifa_manifest = (
        _squad_tournament_context(features)
    )
    result_hash = result_store_hash()
    base_dir, base_advancement, base_meta = _current_base_simulation(result_hash)
    version = "squad-" + sha256_json(
        {
            "challenger": config.challenger_version,
            "feature_version": feature_version,
            "result_hash": result_hash,
            "bracket_hash": bracket.config_hash,
            "sims": sims,
            "seed": seed,
        }
    )[:16]
    out = resolve(Path("artifacts/simulations") / version)
    if (out / "wc2026_tournament_meta.json").exists() and not force:
        typer.echo(f"Refusing to overwrite {out}; pass --force.", err=True)
        raise typer.Exit(code=1)
    indexed = features.set_index("national_team")
    top_teams = set(
        sorted(live.ratings, key=live.ratings.get, reverse=True)[:5]
    )
    started = time.perf_counter()
    result = simulate_full_tournament(
        groups,
        fixtures,
        challenger,
        bracket,
        n_sims=sims,
        seed=seed,
        target_team="Portugal",
        opponent_elo=live.ratings,
        opponent_squad_strength=indexed["score_s7_coverage_shrunk"].to_dict(),
        top_teams=top_teams,
    )
    runtime = time.perf_counter() - started
    problems = check_full_invariants(result)
    if problems:
        raise ValueError(f"simulation invariants failed: {problems}")
    metadata = {
        "research_status": "squad-aware scenario challenger; not trained or deployed",
        "trained": False,
        "n_sims": sims,
        "seed": seed,
        "runtime_seconds": runtime,
        "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "data_cutoff": str(live.cutoff.date()),
        "model_version": live.model_version,
        "challenger_version": config.challenger_version,
        "squad_feature_version": feature_version,
        "adjustment_strength": config.adjustment,
        "coverage_thresholds": config.coverage,
        "fallback_teams": features.loc[
            features["fallback_used"], "national_team"
        ].tolist(),
        "result_store_hash": result_hash,
        "official_bracket_hash": bracket.config_hash,
        "official_bracket_manifest": bracket.source_manifest,
        "fifa_snapshot_id": fifa_manifest["snapshot_id"],
        "base_simulation_directory": str(base_dir),
        "base_simulation_version": base_meta["simulation_version"],
        "completed_fixture_count": sum(f.played for f in fixtures),
        "remaining_fixture_count": sum(not f.played for f in fixtures),
        "target_trace": result.target_trace,
    }
    write_full_simulation(result, bracket, metadata, version)

    prediction_dirs = sorted(resolve("artifacts/research_predictions").glob("*/metadata.json"))
    matching_prediction = None
    for metadata_path in reversed(prediction_dirs):
        candidate = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            candidate.get("result_store_hash") == result_hash
            and candidate.get("feature_version") == feature_version
        ):
            matching_prediction = pd.read_csv(
                metadata_path.parent / "remaining_group_matches.csv"
            )
            break
    if matching_prediction is None:
        raise ValueError("run `goalsignal squad-model predict` before simulation")
    paths = _write_squad_research_reports(
        result,
        base_advancement,
        features,
        config,
        matching_prediction,
        live,
    )
    portugal_paths = _write_portugal_path(result, live, features)
    typer.echo(
        f"Squad research simulation: {sims} sims, seed={seed}, "
        f"runtime={runtime:.2f}s -> {out}"
    )
    typer.echo(f"Reports: {paths + list(portugal_paths)}")


@tournament_app.command("compare-squad")
def tournament_compare_squad() -> None:
    """Print the current base-versus-squad scenario comparison."""
    path = resolve("artifacts/reports/base_vs_squad_advancement.csv")
    if not path.exists():
        typer.echo("Run `tournament simulate-squad` first.", err=True)
        raise typer.Exit(code=1)
    frame = pd.read_csv(path)
    typer.echo(
        frame[
            ["team", "base_p_champion", "p_champion", "delta_champion"]
        ]
        .sort_values("delta_champion", ascending=False)
        .to_string(index=False)
    )


@tournament_app.command("portugal-path")
def tournament_portugal_path() -> None:
    """Print the current Portugal conditional path scenario."""
    path = resolve("artifacts/reports/portugal_path_difficulty.md")
    if not path.exists():
        typer.echo("Run `tournament simulate-squad` first.", err=True)
        raise typer.Exit(code=1)
    typer.echo(path.read_text(encoding="utf-8"))


@squad_model_app.command("compare")
def squad_model_compare() -> None:
    """Alias for the current base-versus-squad tournament comparison."""
    tournament_compare_squad()


@squad_model_app.command("portugal")
def squad_model_portugal() -> None:
    """Print the Portugal squad and path research reports."""
    squad_path = resolve("artifacts/reports/portugal_squad_challenger.md")
    path_path = resolve("artifacts/reports/portugal_path_difficulty.md")
    if not squad_path.exists() or not path_path.exists():
        typer.echo("Run squad prediction and simulation commands first.", err=True)
        raise typer.Exit(code=1)
    typer.echo(squad_path.read_text(encoding="utf-8"))
    typer.echo(path_path.read_text(encoding="utf-8"))


@playerdata_app.command("inspect")
def playerdata_inspect() -> None:
    """Report whether player data is configured and its kind (read-only)."""
    _load_env()
    from goalsignal.data.sources.player_data import PlayerDataUnavailable

    try:
        src = _player_source()
    except PlayerDataUnavailable as e:
        typer.echo(str(e))
        raise typer.Exit(code=0) from None
    kind = "DuckDB file" if src.is_duckdb else "CSV(.gz) directory"
    typer.echo(f"Player data: {kind} at {src.path}")
    typer.echo(f"Tables: {', '.join(src.table_names())}")


@playerdata_app.command("inventory")
def playerdata_inventory() -> None:
    """Inventory every table (rows, columns, dtypes, nulls) — read-only."""
    _load_env()
    import json

    from goalsignal.data.sources.player_data import PlayerDataUnavailable, build_inventory

    try:
        src = _player_source()
    except PlayerDataUnavailable as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=0) from None
    inv = build_inventory(src)
    out = resolve("artifacts/reports/transfermarkt_table_inventory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    for name, t in inv["tables"].items():
        typer.echo(f"{name:18s} rows={t['rows']:>9d} cols={len(t['columns'])}")
    typer.echo(f"Report: {out}")


@playerdata_app.command("temporal-audit")
def playerdata_temporal_audit() -> None:
    """Classify every field by temporal safety (static/dated/current-state)."""
    _load_env()
    from goalsignal.data.sources.player_data import PlayerDataUnavailable, temporal_audit

    try:
        src = _player_source()
    except PlayerDataUnavailable as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=0) from None
    audit = temporal_audit(src)
    typer.echo(f"Temporal classification summary: {audit['summary']}")
    typer.echo("⚠️  current_state_unsafe fields must NEVER be applied to a historical match.")
    typer.echo("Full report written by `player-data coverage` "
               "(transfermarkt_temporal_field_audit.md).")


@playerdata_app.command("coverage")
def playerdata_coverage() -> None:
    """Write the full Transfermarkt audit (inventory, temporal, quality, coverage)."""
    _load_env()
    from goalsignal.data.sources.player_data import PlayerDataUnavailable, write_audit_reports

    try:
        src = _player_source()
    except PlayerDataUnavailable as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=0) from None
    result = write_audit_reports(src)
    typer.echo(f"Tables: {result['inventory_tables']}; "
               f"temporal: {result['temporal_summary']}")
    cov = result["coverage"]
    for k in ("players", "appearances", "game_lineups", "player_valuations", "games"):
        if k in cov:
            slim = {kk: vv for kk, vv in cov[k].items() if kk != "note"}
            typer.echo(f"  {k}: {slim}")
    typer.echo("Reports: transfermarkt_*.{json,csv,md}")


@playerdata_app.command("identity-candidates")
def playerdata_identity_candidates() -> None:
    """Emit entity-linking candidate report scaffolding (no auto-matching)."""
    _load_env()
    from goalsignal.data.sources.readiness import write_player_identity_scaffolding

    write_player_identity_scaffolding()
    typer.echo("Wrote candidate scaffolding: player_identity_candidates.csv, "
               "player_identity_conflicts.csv, player_unmatched.csv, "
               "club_identity_candidates.csv")
    typer.echo("Matching hierarchy (deterministic; name-only never auto-accepted): "
               "1) reviewed mapping 2) source ID 3) name+DOB 4) name+nationality+club+position "
               "5) manual review.")


# === Milestone D1: feature engineering + ablation CLI =======================
features_app = typer.Typer(help="D1 leakage-safe feature engineering.")
app.add_typer(features_app, name="features")


def _d1_inputs():
    """Build the canonical matches, Elo timeline, FIFA timeline, and D1 config."""
    import os

    from goalsignal.data.sources.env import load_env_file
    from goalsignal.data.sources.fifa_ingest import load_fifa_historical
    from goalsignal.features.d1 import load_d1_config
    from goalsignal.ratings.elo import EloConfig, compute_elo

    load_env_file()
    cfg = _load_config(Path("config/data.yaml"), None)
    matches = _canonical_matches(cfg)
    elo = compute_elo(matches, EloConfig.load()).timeline
    fifa_path = os.environ.get("FIFA_RANKINGS_PATH", "")
    if not fifa_path:
        typer.echo("FIFA_RANKINGS_PATH not set; FIFA features will be unavailable.", err=True)
        raise typer.Exit(code=1)
    fifa, fifa_manifest = load_fifa_historical(fifa_path)
    return matches, elo, fifa, fifa_manifest, load_d1_config()


@features_app.command("build-d1")
def features_build_d1(
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing version.")] = False,
) -> None:
    """Build and persist the versioned, leakage-safe D1 feature table."""
    from goalsignal.features.d1 import build_d1_table, write_feature_table

    matches, elo, fifa, fifa_manifest, d1cfg = _d1_inputs()
    table = build_d1_table(matches, elo, fifa, d1cfg)
    try:
        meta = write_feature_table(
            table, d1cfg,
            {"fifa_snapshot": fifa_manifest["snapshot_id"],
             "fifa_hash": fifa_manifest["content_hash"]},
            force=force,
        )
    except FileExistsError as e:
        typer.echo(f"{e}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"D1 features: {meta['rows']} rows, {len(meta['columns'])} cols, "
               f"{meta['date_min']}..{meta['date_max']}, version {meta['feature_version']}, "
               f"config_hash {meta['config_hash']}")
    typer.echo(f"FIFA available: {meta['fifa_available_rows']} rows")
    typer.echo(f"Table: {meta['path']}")


@features_app.command("inspect-d1")
def features_inspect_d1() -> None:
    """Show the D1 feature schema/metadata for the current version."""
    import json

    from goalsignal.features.d1 import load_d1_config

    version = load_d1_config()["native"]["feature_version"]
    schema = resolve(f"artifacts/features/d1/{version}/feature_schema.json")
    if not schema.exists():
        typer.echo("No D1 feature table; run `features build-d1`.", err=True)
        raise typer.Exit(code=1)
    meta = json.loads(schema.read_text())
    typer.echo(f"version={meta['feature_version']} rows={meta['rows']} "
               f"cols={len(meta['columns'])} {meta['date_min']}..{meta['date_max']}")
    typer.echo(f"config_hash={meta['config_hash']} source={meta['source_manifests']}")
    typer.echo(f"indicators={meta['indicator_columns']}")


@features_app.command("validate-d1")
def features_validate_d1() -> None:
    """Validate leakage-safety invariants of the D1 feature table."""
    from goalsignal.features.d1 import load_d1_config, load_feature_table

    version = load_d1_config()["native"]["feature_version"]
    table = load_feature_table(version)
    import pandas as pd

    table["date"] = pd.to_datetime(table["date"])
    issues = []
    # 2026 fixtures must have NO FIFA values (coverage ends 2024)
    f2026 = table[table["date"] >= pd.Timestamp("2026-01-01")]
    if len(f2026) and f2026["fifa_available"].sum() > 0:
        issues.append("2026 fixtures have fifa_available=1 (forward-fill leak!)")
    if len(f2026) and f2026["home_fifa_points"].notna().sum() > 0:
        issues.append("2026 fixtures carry FIFA points (must be NaN)")
    # fifa_available rows must have non-null points
    avail = table[table["fifa_available"] == 1.0]
    if avail["home_fifa_points"].isna().any():
        issues.append("fifa_available=1 rows with null home_fifa_points")
    if issues:
        for i in issues:
            typer.echo(f"FAIL: {i}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"D1 feature table valid: {len(table)} rows; "
               f"2026 fixtures={len(f2026)} (all FIFA-unavailable); "
               f"fifa_available={int(table['fifa_available'].sum())}")


@features_app.command("coverage-d1")
def features_coverage_d1() -> None:
    """Write the D1 feature coverage report."""
    from goalsignal.evaluation.d1_ablation import feature_coverage
    from goalsignal.features.d1 import load_d1_config, load_feature_table

    version = load_d1_config()["native"]["feature_version"]
    df = feature_coverage(load_feature_table(version))
    typer.echo(f"Coverage for {len(df)} features -> "
               "artifacts/reports/d1_feature_coverage.csv")
    low = df[df["coverage"] < 0.6].sort_values("coverage")
    if len(low):
        typer.echo("Lowest-coverage features:")
        for r in low.head(8).itertuples(index=False):
            typer.echo(f"  {r.feature}: {r.coverage:.1%}")


@evaluate_app.command("d1-ablation")
def evaluate_d1_ablation() -> None:
    """Run all D1 ablations on identical folds with paired bootstrap CIs."""
    from goalsignal.evaluation.d1_ablation import run_ablation, write_reports
    from goalsignal.features.d1 import load_d1_config, load_feature_table

    d1cfg = load_d1_config()
    table = load_feature_table(d1cfg["native"]["feature_version"])
    ablation = run_ablation(table, d1cfg)
    write_reports(ablation, d1cfg, table=table)
    s = ablation["summary"].sort_values("log_loss")
    typer.echo(f"{'experiment':22s} {'log_loss':>9s} {'delta':>9s} {'verdict'}")
    for r in s.itertuples(index=False):
        typer.echo(f"{r.experiment:22s} {r.log_loss:9.4f} "
                   f"{r.delta_log_loss_vs_baseline:+9.4f} {r.verdict}")
    typer.echo("Reports: artifacts/reports/d1_ablation_results.{csv,md}, "
               "d1_fold_results.csv, d1_champion_challenger.json, d1_research_summary.md")


@evaluate_app.command("d1-regimes")
def evaluate_d1_regimes() -> None:
    """Exploratory regime analysis of D1-G vs D1-0 (paired subgroups)."""
    from goalsignal.evaluation.d1_ablation import regime_analysis, run_ablation
    from goalsignal.features.d1 import load_d1_config, load_feature_table

    d1cfg = load_d1_config()
    table = load_feature_table(d1cfg["native"]["feature_version"])
    df = regime_analysis(run_ablation(table, d1cfg))
    typer.echo(df.to_string(index=False))
    typer.echo("Report: artifacts/reports/d1_regime_analysis.csv "
               "(exploratory; multiple-comparison caution)")


@evaluate_app.command("d1-report")
def evaluate_d1_report(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="2026 fallback inspection only.")
    ] = False,
) -> None:
    """Generate all D1 reports (or a 2026 fallback dry-run)."""
    from goalsignal.evaluation.d1_ablation import (
        fallback_dry_run,
        regime_analysis,
        run_ablation,
        write_reports,
    )
    from goalsignal.features.d1 import load_d1_config, load_feature_table

    d1cfg = load_d1_config()
    table = load_feature_table(d1cfg["native"]["feature_version"])
    if dry_run:
        rep = fallback_dry_run(table, d1cfg)
        typer.echo(f"2026 fixtures inspected: {rep['n_2026_fixtures_inspected']}; "
                   f"all FIFA-unavailable: {rep['all_2026_fifa_unavailable']}; "
                   f"ledger untouched: {rep['ledger_untouched']}")
        typer.echo("Report: artifacts/reports/d1_fallback_2026.json")
        return
    ablation = run_ablation(table, d1cfg)
    write_reports(ablation, d1cfg, table=table)
    regime_analysis(ablation)
    fallback_dry_run(table, d1cfg)
    typer.echo("All D1 reports written to artifacts/reports/d1_*")
