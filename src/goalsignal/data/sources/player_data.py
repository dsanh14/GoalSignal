"""Transfermarkt-derived player/club history — READ-ONLY audit.

`PLAYER_DATA_PATH` may be a DuckDB file or a directory of gzipped CSV tables
(the transfermarkt-datasets export). This module opens the source strictly
read-only (pandas over `.csv.gz`, or DuckDB in read_only mode); it never
mutates, vacuums, or writes into the source.

Transfermarkt is overwhelmingly CLUB football. National-team match coverage is
sparse, and the `players`/`national_teams` tables carry many CURRENT-STATE
fields (current club, current caps, current market value) that are UNSAFE to
apply to historical matches. The temporal audit classifies every field.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

from goalsignal.data.sources.base import MilestoneNotImplementedError
from goalsignal.data.sources.config import PlayerDataConfig
from goalsignal.utils.hashing import sha256_file
from goalsignal.utils.paths import resolve

# Field-level temporal classification (documented; drives leakage safety).
# Each entry: table -> {field: classification}. Classifications:
#   static_identity | dated_observation | current_state_unsafe | derived |
#   unclear_temporal
TEMPORAL_CLASSIFICATION = {
    "players": {
        "player_id": "static_identity", "name": "static_identity",
        "first_name": "static_identity", "last_name": "static_identity",
        "date_of_birth": "static_identity", "country_of_birth": "static_identity",
        "country_of_citizenship": "static_identity", "position": "static_identity",
        "sub_position": "static_identity", "foot": "static_identity",
        "height_in_cm": "static_identity",
        "current_club_id": "current_state_unsafe", "current_club_name": "current_state_unsafe",
        "current_national_team_id": "current_state_unsafe",
        "international_caps": "current_state_unsafe", "international_goals": "current_state_unsafe",
        "market_value_in_eur": "current_state_unsafe",
        "highest_market_value_in_eur": "current_state_unsafe",
        "contract_expiration_date": "current_state_unsafe", "last_season": "current_state_unsafe",
        "agent_name": "current_state_unsafe",
    },
    "appearances": {
        "appearance_id": "static_identity", "game_id": "static_identity",
        "player_id": "static_identity", "date": "dated_observation",
        "minutes_played": "dated_observation", "goals": "dated_observation",
        "assists": "dated_observation", "yellow_cards": "dated_observation",
        "red_cards": "dated_observation", "competition_id": "dated_observation",
        "player_club_id": "dated_observation",
        "player_current_club_id": "current_state_unsafe",
    },
    "game_lineups": {
        "game_lineups_id": "static_identity", "game_id": "static_identity",
        "player_id": "static_identity", "date": "dated_observation",
        "type": "dated_observation", "position": "dated_observation",
        "number": "dated_observation", "team_captain": "dated_observation",
        "club_id": "dated_observation",
    },
    "player_valuations": {
        "player_id": "static_identity", "date": "dated_observation",
        "market_value_in_eur": "dated_observation",
        "current_club_id": "current_state_unsafe",
        "current_club_name": "current_state_unsafe",
    },
    "national_teams": {
        "national_team_id": "static_identity", "name": "static_identity",
        "fifa_ranking": "current_state_unsafe", "squad_size": "current_state_unsafe",
        "average_age": "current_state_unsafe", "total_market_value": "current_state_unsafe",
        "coach_name": "current_state_unsafe", "last_season": "current_state_unsafe",
    },
    "games": {
        "game_id": "static_identity", "date": "dated_observation",
        "competition_id": "static_identity", "competition_type": "static_identity",
        "home_club_id": "dated_observation", "away_club_id": "dated_observation",
        "home_club_goals": "dated_observation", "away_club_goals": "dated_observation",
    },
    "clubs": {
        "club_id": "static_identity", "name": "static_identity",
        "total_market_value": "current_state_unsafe", "squad_size": "current_state_unsafe",
        "coach_name": "current_state_unsafe", "last_season": "current_state_unsafe",
    },
}


class PlayerDataUnavailable(MilestoneNotImplementedError):
    pass


class PlayerDataSource:
    """Read-only accessor over a Transfermarkt CSV.gz directory or DuckDB file."""

    def __init__(self, path: str | Path, config: PlayerDataConfig | None = None):
        self.path = resolve(path)
        self.config = config or PlayerDataConfig()
        self.is_duckdb = self.path.is_file() and self.path.suffix.lower() in (".duckdb", ".db")
        if self.is_duckdb:
            self._tables = None  # discovered lazily via duckdb
        else:
            self._csvs = {p.name.replace(".csv.gz", "").replace(".csv", ""): p
                          for p in sorted(self.path.glob("*.csv*"))}

    @classmethod
    def resolve_from_env(cls, config: PlayerDataConfig | None = None) -> PlayerDataSource:
        import os

        config = config or PlayerDataConfig()
        raw = os.environ.get(config.path_env, "")
        if not raw:
            raise PlayerDataUnavailable(
                f"player data not configured. Set ${config.path_env} to a DuckDB "
                "file or a directory of transfermarkt-datasets CSV(.gz) tables."
            )
        p = resolve(raw)
        if not p.exists():
            raise PlayerDataUnavailable(f"${config.path_env}={raw} does not exist.")
        return cls(p, config)

    def table_names(self) -> list[str]:
        if self.is_duckdb:
            con = self._duckdb_ro()
            try:
                names = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            finally:
                con.close()
            return names
        return sorted(self._csvs)

    def _duckdb_ro(self):
        try:
            import duckdb
        except ImportError as exc:
            raise PlayerDataUnavailable(
                "DuckDB path requires the optional 'duckdb' dependency "
                "(`uv add duckdb`); the CSV.gz directory format needs no extra dep."
            ) from exc
        return duckdb.connect(str(self.path), read_only=True)

    def read_table(self, name: str, columns: list[str] | None = None,
                   nrows: int | None = None) -> pd.DataFrame:
        """Read a table (or selected columns) read-only."""
        if self.is_duckdb:
            con = self._duckdb_ro()
            try:
                cols = "*" if not columns else ", ".join(columns)
                q = f"SELECT {cols} FROM {name}"
                if nrows:
                    q += f" LIMIT {int(nrows)}"
                return con.execute(q).fetch_df()
            finally:
                con.close()
        p = self._csvs[name]
        comp = "gzip" if p.suffix == ".gz" else "infer"
        return pd.read_csv(p, compression=comp, usecols=columns, nrows=nrows, low_memory=False)

    def count_rows(self, name: str) -> int:
        if self.is_duckdb:
            con = self._duckdb_ro()
            try:
                return int(con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            finally:
                con.close()
        p = self._csvs[name]
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8", errors="replace") as f:
            return max(0, sum(1 for _ in f) - 1)

    def file_hashes(self) -> dict[str, str]:
        """SHA-256 of each source file (proves the source is unchanged)."""
        if self.is_duckdb:
            return {self.path.name: sha256_file(self.path)}
        return {name: sha256_file(p) for name, p in self._csvs.items()}


def build_inventory(src: PlayerDataSource) -> dict:
    """Per-table inventory: rows, columns, dtypes, null rates, ids, date fields."""
    inv = {"source_kind": "duckdb" if src.is_duckdb else "csv_gz_directory",
           "path": str(src.path), "tables": {}}
    for name in src.table_names():
        sample = src.read_table(name, nrows=20000)
        nulls = {c: round(float(sample[c].isna().mean()), 4) for c in sample.columns}
        date_fields = [c for c in sample.columns
                       if "date" in c.lower() or c.lower() in ("season",)]
        id_fields = [c for c in sample.columns if c.lower().endswith("_id") or c.lower() == "id"]
        inv["tables"][name] = {
            "rows": src.count_rows(name),
            "columns": list(sample.columns),
            "dtypes": {c: str(sample[c].dtype) for c in sample.columns},
            "null_rate_sampled": nulls,
            "candidate_id_fields": id_fields,
            "date_fields": date_fields,
        }
    return inv


def temporal_audit(src: PlayerDataSource) -> dict:
    """Classify every field of known tables by temporal safety."""
    out = {"classifications": {}, "summary": {}}
    counts: dict[str, int] = {}
    for name in src.table_names():
        sample = src.read_table(name, nrows=5)
        known = TEMPORAL_CLASSIFICATION.get(name, {})
        table_cls = {}
        for c in sample.columns:
            cls = known.get(c, "unclear_temporal")
            table_cls[c] = cls
            counts[cls] = counts.get(cls, 0) + 1
        out["classifications"][name] = table_cls
    out["summary"] = counts
    out["note"] = ("current_state_unsafe fields (current club, current caps, "
                   "current market value, national-team metadata) must NEVER be "
                   "applied to a historical match. dated_observation fields are "
                   "safe only with an explicit pre-match cutoff.")
    return out


def build_coverage(src: PlayerDataSource) -> dict:
    """Real coverage measures (players, national teams, competitions, lineups)."""
    cov: dict = {}
    names = set(src.table_names())

    if "players" in names:
        p = src.read_table("players")
        caps = pd.to_numeric(p.get("international_caps"), errors="coerce")

        def _nn(col):
            return int(p[col].notna().sum()) if col in p else 0

        cov["players"] = {
            "total": len(p),
            "with_date_of_birth": _nn("date_of_birth"),
            "with_position": _nn("position"),
            "with_citizenship": _nn("country_of_citizenship"),
            "with_current_international_caps_gt0": int((caps > 0).sum()),
            "note": "international_caps is CURRENT-STATE (unsafe historically)",
        }
    if "games" in names:
        g = src.read_table("games", columns=["competition_type", "date"])
        gd = pd.to_datetime(g["date"], errors="coerce")
        cov["games"] = {
            "total": len(g),
            "by_competition_type": {str(k): int(v) for k, v in
                                    g["competition_type"].value_counts().items()},
            "date_min": str(gd.min().date()) if gd.notna().any() else None,
            "date_max": str(gd.max().date()) if gd.notna().any() else None,
            "note": "overwhelmingly CLUB football; national_team_competition games "
            "are a small minority",
        }
    if "appearances" in names:
        a = src.read_table("appearances", columns=["date", "minutes_played", "player_id"])
        ad = pd.to_datetime(a["date"], errors="coerce")
        cov["appearances"] = {
            "total": len(a),
            "distinct_players": int(a["player_id"].nunique()),
            "with_minutes": int(pd.to_numeric(a["minutes_played"], errors="coerce").notna().sum()),
            "date_min": str(ad.min().date()) if ad.notna().any() else None,
            "date_max": str(ad.max().date()) if ad.notna().any() else None,
            "note": "dated club appearances — safe with an explicit pre-match cutoff",
        }
    if "game_lineups" in names:
        gl = src.read_table("game_lineups", columns=["type", "game_id", "player_id"])
        cov["game_lineups"] = {
            "total": len(gl),
            "distinct_games": int(gl["game_id"].nunique()),
            "distinct_players": int(gl["player_id"].nunique()),
            "by_type": {str(k): int(v) for k, v in gl["type"].value_counts().items()},
        }
    if "player_valuations" in names:
        v = src.read_table("player_valuations", columns=["date", "player_id"])
        vd = pd.to_datetime(v["date"], errors="coerce")
        cov["player_valuations"] = {
            "total": len(v),
            "distinct_players": int(v["player_id"].nunique()),
            "date_min": str(vd.min().date()) if vd.notna().any() else None,
            "date_max": str(vd.max().date()) if vd.notna().any() else None,
            "note": "dated valuations — safe with an explicit pre-match cutoff",
        }
    if "national_teams" in names:
        nt = src.read_table("national_teams")
        cov["national_teams"] = {
            "total": len(nt),
            "note": "table fields (fifa_ranking, squad_size, ...) are CURRENT-STATE "
            "metadata — unsafe to apply historically",
        }
    if "competitions" in names:
        c = src.read_table("competitions")

        def _vc(col):
            return ({str(k): int(v) for k, v in c[col].value_counts().items()}
                    if col in c else {})

        cov["competitions"] = {
            "total": len(c),
            "by_type": _vc("type"),
            "by_confederation": _vc("confederation"),
        }
    return cov


def write_audit_reports(src: PlayerDataSource, out_dir: str = "artifacts/reports") -> dict:
    """Write the Transfermarkt inventory, temporal audit, quality, and coverage reports."""
    import json

    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(src)
    (out / "transfermarkt_table_inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8")

    temporal = temporal_audit(src)
    md = ["# Transfermarkt Temporal-Field Audit", "",
          "Source is **club-centric**; national-team match coverage is sparse.",
          "Classification per field (load-bearing for leakage safety):", "",
          f"Summary: {temporal['summary']}", "", temporal["note"], ""]
    for table, fields in temporal["classifications"].items():
        md.append(f"## {table}")
        for f, cls in fields.items():
            flag = " ⚠️ UNSAFE" if cls == "current_state_unsafe" else ""
            md.append(f"- `{f}`: {cls}{flag}")
        md.append("")
    (out / "transfermarkt_temporal_field_audit.md").write_text("\n".join(md), encoding="utf-8")

    quality_rows = []
    for name, t in inventory["tables"].items():
        nulls = t["null_rate_sampled"]
        mean_null = round(sum(nulls.values()) / max(len(nulls), 1), 4)
        n_unsafe = sum(1 for c in t["columns"]
                       if TEMPORAL_CLASSIFICATION.get(name, {}).get(c) == "current_state_unsafe")
        quality_rows.append({"table": name, "rows": t["rows"], "columns": len(t["columns"]),
                             "mean_null_rate_sampled": mean_null,
                             "date_fields": ";".join(t["date_fields"]),
                             "current_state_unsafe_fields": n_unsafe})
    pd.DataFrame(quality_rows).to_csv(out / "transfermarkt_table_quality.csv", index=False)

    coverage = build_coverage(src)
    (out / "transfermarkt_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8")
    # Flat coverage CSVs
    if "players" in coverage:
        pd.DataFrame([coverage["players"]]).to_csv(
            out / "transfermarkt_player_coverage.csv", index=False)
    if "national_teams" in coverage:
        pd.DataFrame([coverage["national_teams"]]).to_csv(
            out / "transfermarkt_national_team_coverage.csv", index=False)
    if "competitions" in coverage:
        comp = coverage["competitions"]
        rows = [{"dimension": "type", "key": k, "count": v}
                for k, v in comp.get("by_type", {}).items()]
        rows += [{"dimension": "confederation", "key": k, "count": v}
                 for k, v in comp.get("by_confederation", {}).items()]
        pd.DataFrame(rows).to_csv(out / "transfermarkt_competition_coverage.csv", index=False)
    lineup_cov = {**coverage.get("game_lineups", {}), **{
        "appearances_" + k: v for k, v in coverage.get("appearances", {}).items()
        if k not in ("note",)}}
    pd.DataFrame([lineup_cov]).to_csv(out / "transfermarkt_lineup_coverage.csv", index=False)

    return {"inventory_tables": len(inventory["tables"]),
            "temporal_summary": temporal["summary"], "coverage": coverage}
