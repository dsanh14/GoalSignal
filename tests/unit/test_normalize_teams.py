"""Unit tests for date-aware team normalization (synthetic data)."""

from __future__ import annotations

import pandas as pd

from goalsignal.data.normalize_teams import TeamNormalizer


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["source_row"] = df.index + 2
    df["source_file"] = "former_names.csv"
    return df


def test_mapping_applies_only_within_period():
    norm = TeamNormalizer.from_former_names(
        _frame(
            [
                {
                    "current": "Ruritania",
                    "former": "Old Ruritania",
                    "start_date": "1990-01-01",
                    "end_date": "2002-12-31",
                }
            ]
        )
    )
    assert norm.canonical("Old Ruritania", pd.Timestamp("1995-06-01")) == "Ruritania"
    # Outside the period the historical name is preserved.
    assert norm.canonical("Old Ruritania", pd.Timestamp("2010-01-01")) == "Old Ruritania"
    assert norm.canonical("Atlantis", pd.Timestamp("1995-06-01")) == "Atlantis"


def test_inverted_period_is_flagged_not_applied():
    norm = TeamNormalizer.from_former_names(
        _frame(
            [
                {
                    "current": "Freedonia",
                    "former": "Fredonia",
                    "start_date": "2005-01-01",
                    "end_date": "2000-01-01",
                }
            ]
        )
    )
    assert any(i.kind == "inverted_period" for i in norm.issues)
    assert norm.canonical("Fredonia", pd.Timestamp("2002-01-01")) == "Fredonia"


def test_overlapping_conflicting_mappings_are_flagged():
    norm = TeamNormalizer.from_former_names(
        _frame(
            [
                {
                    "current": "Sylvania",
                    "former": "Borduria",
                    "start_date": "1990-01-01",
                    "end_date": "2000-12-31",
                },
                {
                    "current": "Zubrowka",
                    "former": "Borduria",
                    "start_date": "1995-01-01",
                    "end_date": "2005-12-31",
                },
            ]
        )
    )
    assert any(i.kind == "overlapping_mapping" for i in norm.issues)


def test_chained_mapping_is_flagged():
    norm = TeamNormalizer.from_former_names(
        _frame(
            [
                {
                    "current": "B",
                    "former": "A",
                    "start_date": "1990-01-01",
                    "end_date": "1999-12-31",
                },
                {
                    "current": "C",
                    "former": "B",
                    "start_date": "2000-01-01",
                    "end_date": "2010-12-31",
                },
            ]
        )
    )
    assert any(i.kind == "chained_mapping" for i in norm.issues)
    # Single-step application only.
    assert norm.canonical("A", pd.Timestamp("1995-01-01")) == "B"
