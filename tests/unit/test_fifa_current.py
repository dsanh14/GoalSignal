from __future__ import annotations

import pandas as pd
import pytest

from goalsignal.data.sources.fifa_current import (
    RELEASE_DATE,
    available_as_of,
    compare_with_elo,
    load_current_fifa,
)


def _snapshot(path, bom: bool = False):
    text = "group,team,fifa_rank\n"
    rows = []
    rank = 1
    for group in "ABCDEFGHIJKL":
        for i in range(4):
            rows.append(f"{group},Team {group}{i},{rank}")
            rank += 1
    path.write_text(("\ufeff" if bom else "") + text + "\n".join(rows) + "\n")
    return {f"Team {g}{i}" for g in "ABCDEFGHIJKL" for i in range(4)}


def test_current_snapshot_bom_validation_and_raw_integrity(tmp_path):
    path = tmp_path / "current.csv"
    canonical = _snapshot(path, bom=True)
    before = path.read_bytes()
    df, manifest, quality = load_current_fifa(path, canonical)
    assert len(df) == 48 and df["group"].nunique() == 12
    assert df.groupby("group").size().eq(4).all()
    assert df["team"].is_unique and (df["fifa_rank"] > 0).all()
    assert set(df["ranking_release_date"]) == {RELEASE_DATE}
    assert df["canonical_team"].notna().all()
    assert manifest["snapshot_id"] == df["source_snapshot_id"].iloc[0]
    assert quality["valid"]
    assert path.read_bytes() == before


def test_current_snapshot_rejects_bad_shape(tmp_path):
    path = tmp_path / "current.csv"
    canonical = _snapshot(path)
    data = pd.read_csv(path).iloc[:-1]
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="48 rows"):
        load_current_fifa(path, canonical)


def test_current_snapshot_release_availability():
    assert not available_as_of("2026-06-11")
    assert available_as_of("2026-06-12")


def test_fifa_elo_comparison(tmp_path):
    path = tmp_path / "current.csv"
    canonical = _snapshot(path)
    df, _, _ = load_current_fifa(path, canonical)
    ratings = {team: 2000 - i * 10 for i, team in enumerate(sorted(canonical))}
    table, summary = compare_with_elo(df, ratings)
    assert len(table) == 48
    assert table["goalsignal_elo_rank"].between(1, 48).all()
    assert -1 <= summary["spearman_rank_correlation"] <= 1
