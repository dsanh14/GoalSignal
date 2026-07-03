"""Tests for knockout performance tags and bounded nudges (synthetic data)."""

from __future__ import annotations

import pytest

from goalsignal.tournament.performance_tags import (
    DEFAULT_TAG_NUDGE_CAP,
    TAG_DEFINITIONS,
    load_performance_tags,
    tag_nudge,
)

HEADER = "team,match_number,tag,points,reason"


def _write(tmp_path, *rows):
    path = tmp_path / "tags.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_loads_and_maps_tags_to_adjustment_taxonomy(tmp_path):
    path = _write(
        tmp_path,
        "Astoria,74,penalty_win,3,\"won pens\"",
        "Astoria,74,extra_time_fatigue,-2,\"120 minutes\"",
        "Borduria,75,late_comeback,4,\"late winner\"",
    )
    tags = load_performance_tags(path)
    assert len(tags) == 3
    assert tags[0].category == "tournament_form"
    assert tags[0].modifier == "penalty_survival_boost"
    assert tags[1].points == -2
    assert tags[2].tag == "late_comeback"


def test_missing_file_yields_empty_unless_required(tmp_path):
    assert load_performance_tags(tmp_path / "nope.csv") == []
    with pytest.raises(FileNotFoundError):
        load_performance_tags(tmp_path / "nope.csv", require=True)


@pytest.mark.parametrize(
    "row, message",
    [
        ("Astoria,74,vibes,3,reason", "unknown tag"),
        ("Astoria,74,penalty_win,-3,reason", "expects non-negative"),
        ("Astoria,74,extra_time_fatigue,2,reason", "expects non-positive"),
        ("Astoria,74,penalty_win,11,reason", "exceeds cap"),
        ("Astoria,74,penalty_win,3,", "'reason' is required"),
        (",74,penalty_win,3,reason", "'team' is required"),
        ("Astoria,42,penalty_win,3,reason", "73-104"),
        ("Astoria,soon,penalty_win,3,reason", "integer"),
        ("Astoria,74,penalty_win,lots,reason", "numeric"),
    ],
)
def test_invalid_tags_rejected(tmp_path, row, message):
    path = _write(tmp_path, row)
    with pytest.raises(ValueError, match=message):
        load_performance_tags(path)


def test_tag_nudge_applies_only_to_later_matches_and_is_bounded(tmp_path):
    path = _write(
        tmp_path,
        "Astoria,74,penalty_win,3,\"won pens\"",
        "Astoria,74,extra_time_fatigue,-2,\"120 minutes\"",
        "Astoria,90,dominant_win,5,\"later evidence\"",
        "Borduria,74,dominant_win,5,\"first\"",
        "Borduria,75,dominant_win,5,\"second\"",
    )
    tags = load_performance_tags(path)
    # Only tags strictly before the target match count.
    same_match = tag_nudge(tags, "Astoria", 74)
    assert same_match.points == 0 and same_match.tags == ()
    future = tag_nudge(tags, "Astoria", 89)
    assert future.points == pytest.approx(1.0)  # +3 - 2; M90 tag excluded
    assert len(future.tags) == 2
    assert "won pens" in future.reasons()
    # Net nudges are clamped at the cap.
    capped = tag_nudge(tags, "Borduria", 89)
    assert capped.raw_points == 10
    assert capped.points == DEFAULT_TAG_NUDGE_CAP
    assert capped.capped
    tighter = tag_nudge(tags, "Borduria", 89, cap=2.5)
    assert tighter.points == 2.5


def test_repository_default_tags_file_is_valid():
    """The tracked manual tags file must always load cleanly and stay bounded."""
    tags = load_performance_tags()
    assert tags, "tracked knockout_performance_tags.csv should not be empty"
    assert all(t.tag in TAG_DEFINITIONS for t in tags)
    assert all(abs(t.points) <= 10 for t in tags)
