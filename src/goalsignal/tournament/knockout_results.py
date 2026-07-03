"""Confirmed knockout results overlay (manual, user-maintained, read-only).

``data/manual/knockout_results_2026.csv`` holds *confirmed* knockout results
entered by hand as the real tournament progresses. The file is an overlay:
nothing here touches ``Datasets/``, the canonical dataset, the ledger, or the
result store. Confirmed winners take precedence over modal simulated winners
when walking the official bracket (see
:mod:`goalsignal.tournament.human_adjustments`), so real R32 outcomes
propagate into R16 pairings and beyond.

Schema (one row per confirmed match)::

    match_number, round, team_a, team_b, score_a, score_b, aet, penalties,
    winner, notes

Score semantics follow the repository rule: ``score_a``/``score_b`` include
extra time and exclude penalty shootouts. ``score_a``/``score_b`` may be left
blank when only the winner is confirmed (winner-only row); the ``winner``
column is always authoritative. A drawn score requires ``penalties=true``
(knockout matches cannot end level), and ``penalties=true`` requires
``aet=true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from goalsignal.tournament.bracket_2026 import ROUND_MATCHES
from goalsignal.utils.paths import resolve

DEFAULT_RESULTS_PATH = "data/manual/knockout_results_2026.csv"

MATCH_ROUNDS: dict[int, str] = {
    number: round_name
    for round_name, numbers in ROUND_MATCHES.items()
    for number in numbers
}

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f", ""}

REQUIRED_COLUMNS = (
    "match_number",
    "round",
    "team_a",
    "team_b",
    "score_a",
    "score_b",
    "aet",
    "penalties",
    "winner",
)


@dataclass(frozen=True)
class KnockoutResult:
    """One confirmed knockout result (scores include ET, exclude shootouts)."""

    match_number: int
    round: str
    team_a: str
    team_b: str
    score_a: int | None
    score_b: int | None
    aet: bool
    penalties: bool
    winner: str
    notes: str = ""

    @property
    def loser(self) -> str:
        return self.team_b if self.winner == self.team_a else self.team_a

    @property
    def decided_by(self) -> str:
        if self.penalties:
            return "penalties"
        if self.aet:
            return "extra_time"
        return "regulation"


def _parse_bool(value: object, field: str, prefix: str, problems: list[str]) -> bool:
    text = str(value).strip().lower() if value is not None else ""
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    problems.append(f"{prefix}: {field} must be true/false, got {value!r}")
    return False


def _parse_score(value: object, field: str, prefix: str, problems: list[str]) -> int | None:
    text = str(value).strip() if value is not None else ""
    if text == "" or text.lower() == "nan":
        return None
    try:
        score = int(float(text))
    except ValueError:
        problems.append(f"{prefix}: {field} must be an integer, got {value!r}")
        return None
    if score < 0:
        problems.append(f"{prefix}: {field} must be non-negative")
        return None
    return score


def _validate_row(row: KnockoutResult, prefix: str, problems: list[str]) -> None:
    expected_round = MATCH_ROUNDS.get(row.match_number)
    if expected_round is None:
        problems.append(f"{prefix}: knockout match numbers are 73-104")
    elif row.round != expected_round:
        problems.append(
            f"{prefix}: round {row.round!r} does not match the official "
            f"round {expected_round!r} for M{row.match_number}"
        )
    if not row.team_a or not row.team_b:
        problems.append(f"{prefix}: team_a and team_b are required")
    if row.team_a == row.team_b:
        problems.append(f"{prefix}: team_a and team_b must differ")
    if row.winner not in (row.team_a, row.team_b):
        problems.append(
            f"{prefix}: winner {row.winner!r} must be team_a or team_b"
        )
    if row.penalties and not row.aet:
        problems.append(f"{prefix}: penalties=true requires aet=true")
    scores = (row.score_a, row.score_b)
    if (scores[0] is None) != (scores[1] is None):
        problems.append(f"{prefix}: provide both scores or leave both blank")
    elif scores[0] is not None and scores[1] is not None:
        if row.penalties and scores[0] != scores[1]:
            problems.append(
                f"{prefix}: a shootout implies a drawn score after extra time "
                "(scores include ET, exclude the shootout)"
            )
        if not row.penalties:
            if scores[0] == scores[1]:
                problems.append(
                    f"{prefix}: knockout matches cannot end level without "
                    "penalties=true"
                )
            else:
                by_score = row.team_a if scores[0] > scores[1] else row.team_b
                if row.winner != by_score:
                    problems.append(
                        f"{prefix}: winner {row.winner!r} contradicts the "
                        f"score {scores[0]}-{scores[1]}"
                    )


def load_knockout_results(
    path: str | Path = DEFAULT_RESULTS_PATH, *, require: bool = False
) -> dict[int, KnockoutResult]:
    """Load confirmed knockout results keyed by match number.

    A missing file yields an empty overlay unless ``require`` is set. Every
    row is validated; any problem raises ``ValueError`` listing all issues.
    """
    p = resolve(path)
    if not p.exists():
        if require:
            raise FileNotFoundError(f"knockout results file not found: {p}")
        return {}
    frame = pd.read_csv(p, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    problems: list[str] = []
    results: dict[int, KnockoutResult] = {}
    for i, raw in enumerate(frame.to_dict("records")):
        prefix = f"{Path(path).name} row {i + 1}"
        try:
            number = int(str(raw["match_number"]).strip())
        except ValueError:
            problems.append(f"{prefix}: match_number must be an integer")
            continue
        prefix = f"{Path(path).name} M{number}"
        if number in results:
            problems.append(f"{prefix}: duplicate match_number")
            continue
        result = KnockoutResult(
            match_number=number,
            round=str(raw["round"]).strip(),
            team_a=str(raw["team_a"]).strip(),
            team_b=str(raw["team_b"]).strip(),
            score_a=_parse_score(raw["score_a"], "score_a", prefix, problems),
            score_b=_parse_score(raw["score_b"], "score_b", prefix, problems),
            aet=_parse_bool(raw["aet"], "aet", prefix, problems),
            penalties=_parse_bool(raw["penalties"], "penalties", prefix, problems),
            winner=str(raw["winner"]).strip(),
            notes=str(raw.get("notes", "")).strip(),
        )
        _validate_row(result, prefix, problems)
        results[number] = result
    if problems:
        raise ValueError("invalid knockout results: " + "; ".join(problems))
    return results
