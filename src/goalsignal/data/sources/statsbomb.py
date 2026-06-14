"""StatsBomb open-data adapter (contract + offline scaffolding).

StatsBomb open data is an optional historical enrichment source. It is NOT a
dependency of the base workflow. Access is offline only: the user clones the
open-data repository locally and points `STATSBOMB_DATA_PATH` at it (optionally
using `statsbombpy`). Nothing is downloaded here.

Coverage of senior men's international football in StatsBomb open data is
sparse and uneven; missing coverage must never be read as a negative team
signal. Actual coverage is enumerated from the local `competitions.json` in
Milestone B.
"""

from __future__ import annotations

import pandas as pd
from pydantic import ValidationError

from goalsignal.data.sources.base import (
    CoverageReport,
    MilestoneNotImplementedError,
    SourceValidationError,
)
from goalsignal.data.sources.config import StatsBombConfig
from goalsignal.data.sources.schemas import StatsBombMatchRef


class StatsBombAdapter:
    name = "statsbomb"
    role = "event_enrichment"

    def __init__(self, config: StatsBombConfig | None = None):
        self.config = config or StatsBombConfig()

    def validate(self, records: list[dict]) -> list[dict]:
        """Validate raw match references against the StatsBomb match schema."""
        out = []
        for i, rec in enumerate(records):
            try:
                out.append(StatsBombMatchRef.model_validate(rec).model_dump(mode="json"))
            except ValidationError as exc:
                raise SourceValidationError(
                    f"StatsBomb record {i} failed schema validation: {exc}"
                ) from exc
        return out

    def load(self) -> pd.DataFrame:
        raise MilestoneNotImplementedError(
            "StatsBomb offline loading lands in Milestone B (source ingestion). "
            f"It will read a local clone at ${self.config.data_path_env}."
        )

    def link_fixtures(self, canonical_matches: pd.DataFrame) -> pd.DataFrame:
        raise MilestoneNotImplementedError(
            "StatsBomb fixture linking lands in Milestone C (entity linking)."
        )

    def report_event_coverage(self) -> pd.DataFrame:
        raise MilestoneNotImplementedError(
            "StatsBomb coverage reporting lands in Milestone B."
        )

    def build_manifest(self):
        raise MilestoneNotImplementedError(
            "Manifest building requires ingested content (Milestone B); use "
            "`goalsignal.data.sources.manifests.build_snapshot_manifest` directly."
        )

    def report_coverage(self) -> CoverageReport:
        return CoverageReport(
            source=self.name,
            rows=0,
            notes=["no data ingested (Milestone A defines contracts only)"],
        )


# --- offline loader + aggregation (Milestone B) -----------------------------
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

from goalsignal.utils.hashing import sha256_file  # noqa: E402

SB_LICENSE = "StatsBomb Open Data License (non-commercial; attribution required)"
SB_ATTRIBUTION = "Data provided by StatsBomb — https://github.com/statsbomb/open-data"


class StatsBombDataUnavailable(MilestoneNotImplementedError):
    """No local StatsBomb data path configured (not an error to abort the run)."""


def resolve_statsbomb_path(config: StatsBombConfig | None = None) -> Path:
    """Resolve the local open-data root, or raise with setup instructions."""
    config = config or StatsBombConfig()
    raw = os.environ.get(config.data_path_env, "")
    if not raw:
        raise StatsBombDataUnavailable(
            f"StatsBomb data not configured. Clone https://github.com/statsbomb/"
            f"open-data and set ${config.data_path_env} to its path (the directory "
            "containing data/competitions.json). StatsBomb is optional; other "
            "sources continue without it."
        )
    path = Path(raw)
    if not (path / "data" / "competitions.json").exists():
        raise StatsBombDataUnavailable(
            f"${config.data_path_env}={raw} does not contain data/competitions.json; "
            "point it at a StatsBomb open-data clone root."
        )
    return path


class StatsBombLoader:
    """Reads the StatsBomb open-data file layout offline. Never downloads."""

    def __init__(self, root: Path, config: StatsBombConfig | None = None):
        self.root = Path(root)
        self.config = config or StatsBombConfig()
        self.data = self.root / "data"

    def _read_json(self, rel: str):
        path = self.data / rel
        if not path.exists():
            raise FileNotFoundError(f"StatsBomb file missing: {path}")
        content = json.loads(path.read_text(encoding="utf-8"))
        return content, sha256_file(path)

    def load_competitions(self) -> tuple[pd.DataFrame, str]:
        comps, h = self._read_json("competitions.json")
        rows = [
            {
                "competition_id": c.get("competition_id"),
                "season_id": c.get("season_id"),
                "competition_name": c.get("competition_name"),
                "season_name": c.get("season_name"),
                "country_name": c.get("country_name"),
                "gender": c.get("competition_gender"),
            }
            for c in comps
        ]
        return pd.DataFrame(rows), h

    def load_matches(self, competition_id: int, season_id: int) -> tuple[pd.DataFrame, str]:
        matches, h = self._read_json(f"matches/{competition_id}/{season_id}.json")
        rows = []
        for m in matches:
            rows.append(
                {
                    "statsbomb_match_id": m.get("match_id"),
                    "match_date": m.get("match_date"),
                    "competition": (m.get("competition") or {}).get("competition_name"),
                    "season": (m.get("season") or {}).get("season_name"),
                    "home_team": (m.get("home_team") or {}).get("home_team_name"),
                    "away_team": (m.get("away_team") or {}).get("away_team_name"),
                    "home_score": m.get("home_score"),
                    "away_score": m.get("away_score"),
                }
            )
        return pd.DataFrame(rows), h

    def load_lineup(self, match_id: int) -> tuple[pd.DataFrame, str]:
        teams, h = self._read_json(f"lineups/{match_id}.json")
        rows = []
        for team in teams:
            tname = team.get("team_name")
            for p in team.get("lineup", []):
                rows.append(
                    {
                        "statsbomb_match_id": match_id,
                        "team": tname,
                        "player_id": p.get("player_id"),
                        "player_name": p.get("player_name"),
                        "jersey_number": p.get("jersey_number"),
                        "country": (p.get("country") or {}).get("name")
                        if isinstance(p.get("country"), dict) else p.get("country"),
                    }
                )
        return pd.DataFrame(rows), h

    def load_events(self, match_id: int) -> tuple[list, str]:
        events, h = self._read_json(f"events/{match_id}.json")
        return events, h

    @staticmethod
    def aggregate_events(events: list) -> pd.DataFrame:
        """Per-team source-level aggregates from one match's events.

        No model features and no cross-match rolling windows here (that is
        Milestone D); just totals validated against the event stream.
        """
        agg: dict[str, dict] = {}

        def team_row(name: str) -> dict:
            return agg.setdefault(
                name,
                {"team": name, "xg": 0.0, "np_xg": 0.0, "shots": 0, "shots_on_target": 0,
                 "set_piece_xg": 0.0, "open_play_xg": 0.0, "goals": 0, "cards": 0,
                 "substitutions": 0},
            )

        for ev in events:
            etype = (ev.get("type") or {}).get("name")
            team = (ev.get("team") or {}).get("name")
            if team is None:
                continue
            row = team_row(team)
            if etype == "Shot":
                shot = ev.get("shot") or {}
                xg = float(shot.get("statsbomb_xg") or 0.0)
                shot_type = (shot.get("type") or {}).get("name")
                outcome = (shot.get("outcome") or {}).get("name")
                play_pattern = (ev.get("play_pattern") or {}).get("name", "")
                row["shots"] += 1
                row["xg"] += xg
                if outcome in ("Goal", "Saved", "Saved to Post"):
                    row["shots_on_target"] += 1
                if outcome == "Goal":
                    row["goals"] += 1
                is_penalty = shot_type == "Penalty"
                if not is_penalty:
                    row["np_xg"] += xg
                if is_penalty or shot_type in ("Free Kick", "Corner") or \
                        play_pattern in ("From Corner", "From Free Kick", "From Throw In"):
                    row["set_piece_xg"] += xg
                else:
                    row["open_play_xg"] += xg
            elif etype == "Substitution":
                row["substitutions"] += 1
            elif etype in ("Foul Committed", "Bad Behaviour"):
                fc = ev.get("foul_committed") or ev.get("bad_behaviour") or {}
                if (fc.get("card") or {}).get("name"):
                    row["cards"] += 1

        df = pd.DataFrame(list(agg.values()))
        for col in ("xg", "np_xg", "set_piece_xg", "open_play_xg"):
            if col in df.columns:
                df[col] = df[col].round(4)
        return df

    def coverage(self) -> dict:
        comps, _ = self.load_competitions()
        gender = comps["gender"].fillna("unknown") if "gender" in comps.columns else None
        intl = comps[comps["competition_name"].str.contains(
            "World Cup|Euro|Copa|Internationa|Nations", case=False, na=False)] \
            if "competition_name" in comps.columns else comps.iloc[0:0]
        return {
            "competitions": len(comps),
            "seasons": int(comps[["competition_id", "season_id"]].drop_duplicates().shape[0])
            if {"competition_id", "season_id"}.issubset(comps.columns) else 0,
            "international_competitions": len(intl),
            "men_competitions": int((gender == "male").sum()) if gender is not None else None,
            "note": "international coverage in StatsBomb open data is sparse; "
            "missing coverage is not a team signal",
        }
