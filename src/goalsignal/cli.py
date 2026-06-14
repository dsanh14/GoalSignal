"""GoalSignal command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from goalsignal.utils.paths import resolve

app = typer.Typer(help="GoalSignal: leakage-safe forecasting for international football.")
data_app = typer.Typer(help="Inspect, validate, and build the canonical dataset.")
ratings_app = typer.Typer(help="Build and inspect Elo ratings.")
evaluate_app = typer.Typer(help="Chronological backtests and evaluations.")
app.add_typer(data_app, name="data")
app.add_typer(ratings_app, name="ratings")
app.add_typer(evaluate_app, name="evaluate")

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


@tournament_app.command("simulate")
def tournament_simulate(
    sims: Annotated[int, typer.Option(help="Number of Monte Carlo simulations.")] = 100_000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 20260612,
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Simulate the 2026 World Cup group stage from the dataset's fixtures.

    Knockout simulation beyond Round-of-32 qualification requires the official
    bracket mapping, which is not in the dataset and is never fabricated.
    """
    import json

    from goalsignal.tournament.fixtures_2026 import derive_2026_group_stage
    from goalsignal.tournament.model_adapter import RatingsGoalAdapter
    from goalsignal.tournament.simulator import check_invariants, simulate_groups_fast

    matches, live = _live_model(config, input_dir)
    groups, fixtures = derive_2026_group_stage(matches)
    adapter = RatingsGoalAdapter(live.ratings, live.goal_model)
    result = simulate_groups_fast(groups, fixtures, adapter, n_sims=sims, seed=seed)

    problems = check_invariants(result)
    if problems:
        for p in problems:
            typer.echo(f"INVARIANT VIOLATION: {p}", err=True)
        raise typer.Exit(code=1)
    if adapter.unrated_teams:
        typer.echo(f"NOTE: unrated teams given default 1500: {sorted(adapter.unrated_teams)}")

    import pandas as pd

    rows = []
    for g, ts in result.groups.items():
        for t in sorted(ts, key=lambda t: -result.advance_probs[t]):
            pp = result.position_probs[t]
            rows.append(
                {
                    "group": g,
                    "team": t,
                    "expected_points": round(result.expected_points[t], 3),
                    "p_first": round(pp[0], 4),
                    "p_second": round(pp[1], 4),
                    "p_third": round(pp[2], 4),
                    "p_fourth": round(pp[3], 4),
                    "p_best_third_advance": round(result.best_third_probs[t], 4),
                    "p_reach_round_of_32": round(result.advance_probs[t], 4),
                    "mc_se_advance": round(
                        result.mc_standard_error(result.advance_probs[t]), 5
                    ),
                }
            )
    df = pd.DataFrame(rows)
    out = resolve("artifacts/simulations")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "wc2026_group_stage.csv", index=False)
    meta = {
        "n_sims": sims,
        "seed": seed,
        "data_cutoff": str(live.cutoff.date()),
        "dataset_version": live.dataset_version,
        "model_version": "ensemble-v1",
        "diagnostics": live.diagnostics,
        "groups_label_note": "group labels G01..G12 are synthetic (derived from "
        "fixture graph); official letters are not in the dataset",
        "knockout_note": "knockout bracket beyond R32 qualification requires the "
        "official bracket mapping; not fabricated",
    }
    with open(out / "wc2026_group_stage_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    typer.echo(f"Simulations: {sims} (seed {seed}); cutoff {live.cutoff.date()}")
    typer.echo(df.sort_values("p_reach_round_of_32", ascending=False)
               .head(15).to_string(index=False))
    typer.echo(f"Full table: {out / 'wc2026_group_stage.csv'}")


@predict_app.command("remaining")
def predict_remaining(
    config: ConfigOpt = Path("config/data.yaml"),
    input_dir: InputDirOpt = None,
) -> None:
    """Predict all scheduled fixtures and append them to the prediction ledger."""
    from goalsignal.ledger.storage import DEFAULT_PATH, append_predictions
    from goalsignal.live import build_prediction_payload

    matches, live = _live_model(config, input_dir)
    scheduled = matches[matches["status"] == "scheduled"].sort_values("date")
    if len(scheduled) == 0:
        typer.echo("No scheduled fixtures in the dataset.")
        raise typer.Exit(code=0)

    payloads = [
        build_prediction_payload(live, row)
        for row in scheduled.itertuples(index=False)
    ]
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
) -> None:
    """Expected goals and exact-score forecasts for stored predictions."""
    from goalsignal.ledger.display import (
        filter_entries,
        format_csv,
        format_json,
        format_table,
    )
    from goalsignal.ledger.storage import list_entries

    entries = filter_entries(list_entries(path), team=team, date=date)
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
feedback_app = typer.Typer(help="Score frozen predictions against recorded results.")
app.add_typer(result_app, name="result")
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

    from goalsignal.feedback.results import active_results, apply_results_overlay
    from goalsignal.ratings.elo import EloConfig, compute_elo

    matches = _canonical_matches(cfg)
    overlaid, _ = apply_results_overlay(matches, active_results())
    timeline = compute_elo(overlaid, EloConfig.load()).timeline
    row = timeline[timeline["canonical_match_id"] == fixture_id]
    if len(row) != 1:
        typer.echo("WARNING: could not locate Elo update for the recorded result", err=True)
        return {}
    r = row.iloc[0]
    update = {
        "canonical_match_id": fixture_id,
        "date": str(r["date"].date()),
        "home_team": r["home_team"],
        "away_team": r["away_team"],
        "home_elo_pre": float(r["home_elo_pre"]),
        "home_elo_post": float(r["home_elo_post"]),
        "away_elo_pre": float(r["away_elo_pre"]),
        "away_elo_post": float(r["away_elo_post"]),
        "delta": float(r["delta"]),
        "result_entry_hash": result_entry_hash,
    }
    out = resolve("artifacts/ratings/online_updates.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(update, sort_keys=True) + "\n")
    return update


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
    from goalsignal.feedback.results import record_result, verify_results
    from goalsignal.ledger.storage import list_entries, verify_ledger

    cfg = _load_config(config, input_dir)
    matches = _canonical_matches(cfg)
    fixture = _resolve_fixture(matches, fixture_id)
    fid = fixture["canonical_match_id"]
    if fixture["status"] == "played":
        typer.echo("Fixture is already played in the source dataset; refusing.", err=True)
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
    from goalsignal.feedback.results import list_results, record_result

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
    )
    typer.echo(f"Correction recorded [entry {entry['entry_hash'][:12]}], "
               f"supersedes {prior[-1]['entry_hash'][:12]}.")


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
app.add_typer(sources_app, name="sources")
app.add_typer(apifootball_app, name="api-football")
app.add_typer(statsbomb_app, name="statsbomb")
app.add_typer(fifa_app, name="fifa-rankings")


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


def _player_source():
    from goalsignal.data.sources.player_data import PlayerDataSource

    return PlayerDataSource.resolve_from_env()


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
