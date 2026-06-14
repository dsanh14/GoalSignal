"""D1 feature-engineering and ablation tests (synthetic fixtures only).

Exercises the leakage-safety invariants: target exclusion, as-of FIFA join,
no 2024->2026 forward-fill, fold-local preprocessing, identical paired folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goalsignal.features.d1 import (
    build_d1_table,
    build_team_history,
    compute_fifa_features,
    compute_native_features,
    compute_venue_features,
    load_d1_config,
    write_feature_table,
)

CONFIG = load_d1_config()


def _matches():
    rows = [
        # id, date, home, away, tournament, country, neutral, hs, as, outcome
        ("m1", "2018-06-01", "Brazil", "Chile", "Friendly", "Brazil", False, 2, 0, "home_win"),
        ("m2", "2018-06-08", "Brazil", "Austria", "Friendly", "Austria", True, 3, 0, "home_win"),
        ("m3", "2018-06-17", "Brazil", "Switzerland", "FIFA World Cup", "Russia", True, 1, 1, "draw"),  # noqa: E501
        ("m4", "2018-06-22", "Brazil", "Costa Rica", "FIFA World Cup", "Russia", True, 2, 0, "home_win"),  # noqa: E501
        ("c1", "2018-05-20", "Chile", "Peru", "Friendly", "Chile", False, 1, 0, "home_win"),
    ]
    df = pd.DataFrame(rows, columns=[
        "canonical_match_id", "date", "home_team", "away_team", "tournament",
        "country", "neutral", "home_score_recorded", "away_score_recorded",
        "regulation_outcome"])
    df["status"] = "played"
    df["source_row"] = range(2, len(df) + 2)
    df["regulation_home_score"] = df["home_score_recorded"]
    df["regulation_away_score"] = df["away_score_recorded"]
    return df


def _elo(matches):
    rows = []
    for r in matches.itertuples(index=False):
        rows.append({"canonical_match_id": r.canonical_match_id, "date": r.date,
                     "home_team": r.home_team, "away_team": r.away_team,
                     "neutral": r.neutral, "home_elo_pre": 1900.0, "away_elo_pre": 1700.0,
                     "expected_home": 0.65})
    return pd.DataFrame(rows)


def _fifa():
    rows = [
        {"team": "Brazil", "fifa_points": 1430.0, "fifa_rank": 2,
         "ranking_release_date": "2018-06-07", "normalized_team": "brazil"},
        {"team": "Brazil", "fifa_points": 1400.0, "fifa_rank": 3,
         "ranking_release_date": "2018-05-01", "normalized_team": "brazil"},
        # A FUTURE release that must never be selected for a June match:
        {"team": "Brazil", "fifa_points": 9999.0, "fifa_rank": 1,
         "ranking_release_date": "2019-01-01", "normalized_team": "brazil"},
        {"team": "Switzerland", "fifa_points": 1199.0, "fifa_rank": 6,
         "ranking_release_date": "2018-06-07", "normalized_team": "switzerland"},
    ]
    return pd.DataFrame(rows)


# --- team history + native form ---------------------------------------------


def test_team_history_home_away_and_points():
    long = build_team_history(_matches(), _elo(_matches()))
    brazil = long[long["team"] == "Brazil"].sort_values(["date", "source_row"])
    assert len(brazil) == 4  # m1..m4 (home perspective)
    assert list(brazil["points"]) == [3, 3, 1, 3]  # win, win, draw, win


def test_native_features_exclude_target_and_windows():
    long = build_team_history(_matches(), _elo(_matches()))
    nat = compute_native_features(long, CONFIG)
    brazil_home = nat[nat["is_home"]].merge(
        long[["canonical_match_id", "team", "date"]].drop_duplicates(),
        on="canonical_match_id")
    brazil_home = brazil_home[brazil_home["team"] == "Brazil"].sort_values("date")
    # First match has no prior history.
    first = brazil_home.iloc[0]
    assert first["n_prior"] == 0 and pd.isna(first["ppm_last5"])
    # Third match (m3): prior = m1(3pts), m2(3pts) -> ppm = 3.0, target excluded.
    third = brazil_home.iloc[2]
    assert third["n_prior"] == 2
    assert third["ppm_last5"] == pytest.approx(3.0)


def test_native_rest_and_days_since_prev():
    long = build_team_history(_matches(), _elo(_matches()))
    nat = compute_native_features(long, CONFIG)
    brazil = nat[nat["is_home"]].merge(
        long[["canonical_match_id", "team", "date"]].drop_duplicates(),
        on="canonical_match_id")
    brazil = brazil[brazil["team"] == "Brazil"].sort_values("date")
    # m2 is 7 days after m1.
    assert brazil.iloc[1]["days_since_prev"] == pytest.approx(7.0)
    assert pd.isna(brazil.iloc[0]["days_since_prev"])  # no prior match


# --- FIFA as-of / no future / no forward-fill -------------------------------


def test_fifa_as_of_no_future_release():
    matches = _matches()
    frame = build_d1_table(matches, _elo(matches), _fifa(), CONFIG)
    m3 = frame[frame["canonical_match_id"] == "m3"].iloc[0]  # 2018-06-17 Brazil v Switzerland
    # Uses 2018-06-07 release (1430), NEVER the 2019-01-01 future release (9999).
    assert m3["home_fifa_points"] == pytest.approx(1430.0)
    assert m3["fifa_points_diff"] == pytest.approx(1430.0 - 1199.0)
    assert m3["fifa_available"] == 1.0


def test_fifa_missing_team_unavailable():
    matches = _matches()
    # Chile has no FIFA rows -> c1 (Chile home) should be FIFA-unavailable.
    frame = build_d1_table(matches, _elo(matches), _fifa(), CONFIG)
    c1 = frame[frame["canonical_match_id"] == "c1"].iloc[0]
    assert c1["fifa_available"] == 0.0
    assert pd.isna(c1["home_fifa_points"])  # never zero-filled


def test_fifa_no_2024_to_2026_forward_fill():
    # FIFA ends 2024-09-19; a 2026 match must be FIFA-unavailable (cap < 471d).
    fifa = pd.DataFrame([{"team": "Brazil", "fifa_points": 1800.0, "fifa_rank": 1,
                          "ranking_release_date": "2024-09-19", "normalized_team": "brazil"}])
    m = pd.DataFrame([{
        "canonical_match_id": "w1", "date": "2026-06-20", "home_team": "Brazil",
        "away_team": "Serbia", "tournament": "FIFA World Cup", "country": "United States",
        "neutral": True, "home_score_recorded": 2, "away_score_recorded": 0,
        "regulation_outcome": "home_win", "status": "played", "source_row": 2,
        "regulation_home_score": 2, "regulation_away_score": 0}])
    feats = compute_fifa_features(
        m.assign(home_elo_pre=1800.0, away_elo_pre=1600.0), fifa, CONFIG)
    row = feats.iloc[0]
    assert row["fifa_available"] == 0.0
    assert pd.isna(row["home_fifa_points"])  # no 2024 forward-fill into 2026


def test_fifa_staleness_flag():
    fifa = pd.DataFrame([{"team": "Brazil", "fifa_points": 1800.0, "fifa_rank": 1,
                          "ranking_release_date": "2024-09-19", "normalized_team": "brazil"},
                         {"team": "Serbia", "fifa_points": 1500.0, "fifa_rank": 25,
                          "ranking_release_date": "2024-09-19", "normalized_team": "serbia"}])
    # A 2025-08 match: ~320 days -> available; >400 would be stale. Use 2025-11 (~420d) -> stale.
    m = pd.DataFrame([{
        "canonical_match_id": "s1", "date": "2025-11-15", "home_team": "Brazil",
        "away_team": "Serbia", "tournament": "Friendly", "country": "Brazil",
        "neutral": False, "home_elo_pre": 1800.0, "away_elo_pre": 1500.0}])
    row = compute_fifa_features(m, fifa, CONFIG).iloc[0]
    assert row["fifa_available"] == 1.0 and row["fifa_stale"] == 1.0


# --- venue ------------------------------------------------------------------


def test_venue_indicators():
    matches = _matches()
    v = compute_venue_features(matches.assign(
        home_elo_pre=1900.0, away_elo_pre=1700.0)).set_index("canonical_match_id")
    # m1 in Brazil, not neutral, Brazil is home -> home_at_home_country.
    assert v.loc["m1", "home_at_home_country"] == 1.0
    assert v.loc["m1", "is_neutral"] == 0.0
    # m3 neutral (Russia).
    assert v.loc["m3", "is_neutral"] == 1.0
    assert v.loc["m3", "home_at_home_country"] == 0.0


# --- feature table + provenance ---------------------------------------------


def test_build_table_has_targets_separate_and_indicators():
    matches = _matches()
    frame = build_d1_table(matches, _elo(matches), _fifa(), CONFIG)
    assert "label" in frame.columns  # target present but clearly named
    assert "abs_elo_diff" in frame.columns
    for ind in ("fifa_available", "home_form_available", "is_neutral"):
        assert ind in frame.columns


def test_write_feature_table_deterministic_and_force(tmp_path):
    matches = _matches()
    frame = build_d1_table(matches, _elo(matches), _fifa(), CONFIG)
    meta1 = write_feature_table(frame, CONFIG, {"fifa": "abc"},
                                out_root=str(tmp_path), force=True)
    assert meta1["config_hash"] == write_feature_table(
        frame, CONFIG, {"fifa": "abc"}, out_root=str(tmp_path), force=True)["config_hash"]
    assert meta1["source_manifests"] == {"fifa": "abc"}
    with pytest.raises(FileExistsError):
        write_feature_table(frame, CONFIG, {"fifa": "abc"}, out_root=str(tmp_path), force=False)


# --- fold-local preprocessing + identical folds -----------------------------


def test_fold_preprocessor_fits_on_train_only():
    from goalsignal.evaluation.d1_ablation import _FoldPreprocessor

    train = pd.DataFrame({"elo_diff": [100.0, 200.0, 300.0], "fifa_available": [1, 0, 1]})
    test = pd.DataFrame({"elo_diff": [1000.0], "fifa_available": [1]})
    pre = _FoldPreprocessor(["elo_diff", "fifa_available"]).fit(train)
    # mean/std come from TRAIN only (mean 200), so test transforms with train stats.
    assert pre.mean_["elo_diff"] == pytest.approx(200.0)
    x = pre.transform(test)
    # standardized test value uses train mean/std, not its own.
    assert x[0, 1] == pytest.approx((1000.0 - 200.0) / train["elo_diff"].std())


def test_ablation_identical_paired_matches():
    from goalsignal.evaluation.d1_ablation import run_experiment

    # Build a small multi-year synthetic table.
    rng = np.random.default_rng(0)
    n = 800
    years = rng.integers(2015, 2021, n)
    dates = [f"{y}-06-01" for y in years]
    elo_diff = rng.normal(0, 200, n)
    label = np.where(elo_diff > 50, 0, np.where(elo_diff < -50, 2, 1))
    table = pd.DataFrame({
        "canonical_match_id": [f"x{i}" for i in range(n)], "date": dates,
        "label": label, "elo_diff": elo_diff, "abs_elo_diff": np.abs(elo_diff),
        "home_adv": 1.0, "tournament": "Friendly", "neutral": False,
        "fifa_available": 1.0, "extra_feat": rng.normal(0, 1, n)})
    bt = {"start_year": 2019, "end_year": 2020, "val_years": 2}
    a = run_experiment(table, ["elo_diff", "abs_elo_diff", "home_adv"], bt)
    b = run_experiment(table, ["elo_diff", "abs_elo_diff", "home_adv", "extra_feat"], bt)
    # Both experiments evaluate the SAME test matches (identical canonical ids).
    assert set(a["predictions"]["canonical_match_id"]) == set(b["predictions"]["canonical_match_id"])  # noqa: E501


# --- regression: ledgers untouched ------------------------------------------


def test_ledger_and_result_store_verify():
    from goalsignal.feedback.results import verify_results
    from goalsignal.ledger.storage import verify_ledger
    from goalsignal.utils.paths import resolve

    if resolve("artifacts/predictions/ledger.jsonl").exists():
        assert verify_ledger() == []
    if resolve("artifacts/results/results.jsonl").exists():
        assert verify_results() == []
