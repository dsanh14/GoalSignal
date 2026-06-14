"""Unit tests for raw loaders (synthetic data)."""

from __future__ import annotations

import pytest

from goalsignal.data.loaders import (
    DatasetNotFoundError,
    SchemaError,
    load_all,
    load_results,
)


def test_load_all_reads_every_file(synthetic_config):
    raw = load_all(synthetic_config)
    assert set(raw) == {"results", "shootouts", "goalscorers", "former_names"}
    assert len(raw["results"]) == 10
    assert "source_row" in raw["results"].columns


def test_missing_file_raises_actionable_error(synthetic_config, tmp_path):
    synthetic_config.input.directory = str(tmp_path / "nowhere")
    with pytest.raises(DatasetNotFoundError, match="--input-dir"):
        load_results(synthetic_config)


def test_missing_column_raises_schema_error(synthetic_config, synthetic_dir):
    bad = synthetic_dir / "results.csv"
    bad.write_text("date,home_team,away_team\n2000-01-01,A,B\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="missing required columns"):
        load_results(synthetic_config)
