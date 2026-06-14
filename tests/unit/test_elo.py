"""Unit tests for the Elo rating engine (synthetic data)."""

from __future__ import annotations

import pandas as pd
import pytest

from goalsignal.ratings.elo import EloConfig, compute_elo, expected_home_score


def _matches(rows: list[dict]) -> pd.DataFrame:
    base = {
        "status": "played",
        "neutral": False,
        "tournament": "Friendly",
        "shootout_played": False,
        "shootout_winner": None,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    df["source_row"] = df.index + 2
    df["canonical_match_id"] = [f"m{i}" for i in range(len(df))]
    return df


def _row(date, home, away, hs, as_, **kw):
    outcome = "draw" if hs == as_ else ("home_win" if hs > as_ else "away_win")
    return {
        "date": date, "home_team": home, "away_team": away,
        "home_score_recorded": hs, "away_score_recorded": as_,
        "regulation_outcome": outcome, **kw,
    }


@pytest.fixture
def config():
    return EloConfig(k_factor=20.0, home_advantage=60.0, importance=[])


def test_expected_score_symmetry():
    assert expected_home_score(1500, 1500, 400) == 0.5
    e = expected_home_score(1600, 1400, 400)
    assert e + expected_home_score(1400, 1600, 400) == pytest.approx(1.0)
    assert e > 0.5


def test_zero_sum_updates(config):
    df = _matches([_row("2000-01-01", "Atlantis", "Ruritania", 2, 0)])
    result = compute_elo(df, config)
    row = result.timeline.iloc[0]
    assert row["home_elo_post"] - row["home_elo_pre"] == pytest.approx(
        -(row["away_elo_post"] - row["away_elo_pre"])
    )
    total = sum(result.final_ratings.values())
    assert total == pytest.approx(2 * config.initial_rating)


def test_pre_match_rating_excludes_target_match(config):
    df = _matches([
        _row("2000-01-01", "Atlantis", "Ruritania", 3, 0),
        _row("2000-02-01", "Atlantis", "Ruritania", 1, 0),
    ])
    result = compute_elo(df, config)
    first, second = result.timeline.iloc[0], result.timeline.iloc[1]
    # Match 2's pre-match rating equals match 1's post-match rating: the
    # target match never contributes to its own pre-match rating.
    assert second["home_elo_pre"] == pytest.approx(first["home_elo_post"])
    assert first["home_elo_pre"] == config.initial_rating


def test_future_match_does_not_change_history(config):
    past = [_row("2000-01-01", "Atlantis", "Ruritania", 2, 1)]
    future = [_row("2030-01-01", "Atlantis", "Ruritania", 0, 5)]
    t_without = compute_elo(_matches(past), config).timeline
    t_with = compute_elo(_matches(past + future), config).timeline
    pd.testing.assert_frame_equal(
        t_without, t_with.iloc[: len(t_without)].reset_index(drop=True)
    )


def test_neutral_venue_removes_home_advantage(config):
    home = _matches([_row("2000-01-01", "Atlantis", "Ruritania", 1, 1, neutral=False)])
    neutral = _matches([_row("2000-01-01", "Atlantis", "Ruritania", 1, 1, neutral=True)])
    e_home = compute_elo(home, config).timeline.iloc[0]["expected_home"]
    e_neutral = compute_elo(neutral, config).timeline.iloc[0]["expected_home"]
    assert e_home > e_neutral == 0.5


def test_goal_difference_multiplier(config):
    narrow = _matches([_row("2000-01-01", "Atlantis", "Ruritania", 1, 0)])
    blowout = _matches([_row("2000-01-01", "Atlantis", "Ruritania", 5, 0)])
    d_narrow = compute_elo(narrow, config).timeline.iloc[0]["delta"]
    d_blowout = compute_elo(blowout, config).timeline.iloc[0]["delta"]
    assert d_blowout == pytest.approx(d_narrow * (11 + 5) / 8)


def test_shootout_draw_policy(config):
    df = _matches([
        _row("2000-01-01", "Atlantis", "Ruritania", 1, 1,
             shootout_played=True, shootout_winner="Atlantis"),
    ])
    result = compute_elo(df, config)
    # Policy "draw": shootout winner gets no extra credit.
    assert result.timeline.iloc[0]["actual_home"] == 0.5


def test_shootout_winner_partial_policy():
    config = EloConfig(shootout_policy="winner_partial", shootout_credit=0.25,
                       importance=[])
    df = _matches([
        _row("2000-01-01", "Atlantis", "Ruritania", 1, 1,
             shootout_played=True, shootout_winner="Atlantis"),
    ])
    assert compute_elo(df, config).timeline.iloc[0]["actual_home"] == 0.75


def test_unknown_outcome_falls_back_to_recorded_score(config):
    df = _matches([
        _row("2000-01-01", "Atlantis", "Ruritania", 2, 0,
             shootout_played=True, shootout_winner="Ruritania"),
    ])
    df["regulation_outcome"] = "unknown"
    result = compute_elo(df, config)
    assert result.timeline.iloc[0]["actual_home"] == 1.0
