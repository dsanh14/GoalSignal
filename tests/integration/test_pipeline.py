"""Integration tests: full validate/build pipeline on synthetic data."""

from __future__ import annotations

import json

from goalsignal.data.build_dataset import build
from goalsignal.data.loaders import load_all
from goalsignal.data.metadata import build_manifest
from goalsignal.data.validation import write_reports
from goalsignal.utils.paths import resolve


def test_reports_are_written(synthetic_config):
    raw = load_all(synthetic_config)
    result = build(raw, synthetic_config)
    reports_dir, _report = write_reports(raw, result, synthetic_config)

    expected = [
        "data_quality.json",
        "data_quality.md",
        "excluded_matches.csv",
        "duplicate_matches.csv",
        "suspicious_scope_matches.csv",
        "shootout_reconciliation.csv",
        "goalscorer_coverage.csv",
        "former_name_conflicts.csv",
    ]
    for name in expected:
        assert (reports_dir / name).exists(), f"missing report {name}"

    with open(reports_dir / "data_quality.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["results"]["raw_rows"] == 10
    # Absence of scorer rows is reported as coverage, never as zero goals.
    assert "events_not_joined_to_played_match" in summary["goalscorers"]


def test_manifest_versions_are_deterministic(synthetic_config, tmp_path):
    raw = load_all(synthetic_config)
    result = build(raw, synthetic_config)
    out = tmp_path / "matches.csv"
    m = result.matches.copy()
    m["date"] = m["date"].dt.strftime("%Y-%m-%d")
    m.to_csv(out, index=False)

    m1 = build_manifest(synthetic_config, result, out)
    m2 = build_manifest(synthetic_config, result, out)
    assert m1["dataset_version"] == m2["dataset_version"]
    assert m1["output"]["sha256"] == m2["output"]["sha256"]
    assert m1["stats"]["canonical_matches"] == 7


def test_cli_build_refuses_overwrite_without_force(synthetic_config, monkeypatch, tmp_path):
    processed = resolve(synthetic_config.output.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    (processed / "matches.csv").write_text("sentinel", encoding="utf-8")

    import typer

    from goalsignal import cli

    monkeypatch.setattr(cli, "_load_config", lambda *_: synthetic_config)
    try:
        cli.data_build(config=None, input_dir=None, force=False)
        raise AssertionError("expected typer.Exit")
    except typer.Exit as e:
        assert e.exit_code == 1
    assert (processed / "matches.csv").read_text(encoding="utf-8") == "sentinel"
