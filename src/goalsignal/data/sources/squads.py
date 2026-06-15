"""Official squad ingestion, player linkage, and cutoff-safe source aggregates."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from goalsignal.data.sources.base import ProvenanceEnvelope
from goalsignal.data.sources.config import SquadDataConfig
from goalsignal.data.sources.linking import normalize_team
from goalsignal.data.sources.manifests import build_snapshot_manifest
from goalsignal.data.sources.players import normalize_player_name
from goalsignal.data.sources.readiness import KNOWN_FIFA_ALIASES
from goalsignal.data.sources.schemas import SquadMembershipRecord
from goalsignal.utils.hashing import sha256_file
from goalsignal.utils.paths import resolve

SCHEMA_VERSION = 1
GROUPS = set("ABCDEFGHIJKL")
SQUAD_COLUMNS = [
    "snapshot_date",
    "group",
    "national_team",
    "player_name",
    "date_of_birth",
    "position",
    "club",
    "shirt_number",
    "squad_status",
    "source_name",
    "source_url_or_reference",
    "source_publication_date",
    "source_player_id",
    "notes",
]
LINK_COLUMNS = [
    "snapshot_date",
    "group",
    "national_team",
    "player_name",
    "position",
    "normalized_player_name",
    "canonical_player_id",
    "transfermarkt_player_id",
    "match_class",
    "match_method",
    "candidate_count",
    "review_status",
]
READINESS_STATES = {
    "ready",
    "ready with cutoff",
    "restricted subset",
    "blocked by missing squad source",
    "blocked by identity coverage",
    "blocked by sparse international lineups",
    "blocked by provider plan",
    "unsupported",
    "temporally unsafe",
}


class SquadDataUnavailable(ValueError):
    pass


def resolve_squad_path(config: SquadDataConfig | None = None) -> Path:
    config = config or SquadDataConfig()
    raw = os.environ.get(config.squads_path_env, "") or config.squads_default_path
    if not raw:
        raise SquadDataUnavailable(
            f"official squad data is not configured. Set ${config.squads_path_env} "
            "to a verified FIFA/federation CSV matching "
            "data/reference/world_cup_2026_squads_template.csv"
        )
    path = resolve(raw)
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise SquadDataUnavailable(f"official squad path must be a readable CSV: {path}")
    return path


def resolve_optional_reference_path(
    env_name: str, default_path: str
) -> Path | None:
    raw = os.environ.get(env_name, "") or default_path
    if not raw:
        return None
    path = resolve(raw)
    return path if path.is_file() else None


def _parse_source_date(value: str) -> pd.Timestamp:
    value = str(value or "").strip()
    if not value:
        return pd.NaT
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y"):
        parsed = pd.to_datetime(value, format=fmt, errors="coerce")
        if pd.notna(parsed):
            return parsed
    return pd.NaT


def _looks_numeric(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", str(value or "").strip()))


def normalize_official_person_fields(
    *, date_of_birth: str, club: str, alternate_date: str = ""
) -> dict:
    """Recover deterministic optional-field shifts from the FIFA PDF extract."""
    primary_date = _parse_source_date(date_of_birth)
    alternate = _parse_source_date(alternate_date)
    if pd.notna(primary_date):
        normalized_date = primary_date
        normalized_club = "" if _looks_numeric(club) else str(club).strip()
        classification = "as_labeled"
    elif pd.notna(alternate):
        normalized_date = alternate
        normalized_club = str(date_of_birth).strip()
        classification = "optional_field_shift"
    else:
        normalized_date = pd.NaT
        normalized_club = (
            str(date_of_birth).strip()
            if date_of_birth and not _looks_numeric(date_of_birth)
            else str(club).strip() if club and not _looks_numeric(club) else ""
        )
        classification = "missing_date"
    return {
        "normalized_date_of_birth": normalized_date,
        "normalized_club": normalized_club,
        "normalization_class": classification,
    }


def _canonical_team(name: str, canonical_teams: set[str] | None) -> str | None:
    normalized = normalize_team(name)
    if not normalized:
        return None
    if canonical_teams:
        by_norm = {normalize_team(team): team for team in canonical_teams}
        if normalized in by_norm:
            return by_norm[normalized]
        alias = KNOWN_FIFA_ALIASES.get(normalized)
        if alias and normalize_team(alias) in by_norm:
            return by_norm[normalize_team(alias)]
        return None
    return KNOWN_FIFA_ALIASES.get(normalized, name.strip())


def load_squads(
    path: str | Path,
    *,
    canonical_teams: set[str] | None = None,
    config: SquadDataConfig | None = None,
    retrieved_at: datetime | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """Load an official squad CSV without mutating or replacing raw fields."""
    config = config or SquadDataConfig()
    path = resolve(path)
    content_hash = sha256_file(path)
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    raw.columns = [str(column).lstrip("\ufeff").strip() for column in raw.columns]
    missing_columns = [column for column in SQUAD_COLUMNS if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"squad CSV missing columns: {missing_columns}")
    raw = raw[SQUAD_COLUMNS].copy()
    frame = raw.copy()
    frame["source_row"] = np.arange(2, len(frame) + 2)
    for column in SQUAD_COLUMNS:
        frame[f"raw_{column}"] = raw[column]
        frame[column] = frame[column].str.strip()
    frame["canonical_team"] = frame["national_team"].map(
        lambda value: _canonical_team(value, canonical_teams)
    )
    frame["normalized_player_name"] = frame["player_name"].map(normalize_player_name)
    frame["snapshot_date_parsed"] = pd.to_datetime(
        frame["snapshot_date"], format="%Y-%m-%d", errors="coerce"
    )
    frame["publication_date_parsed"] = pd.to_datetime(
        frame["source_publication_date"], format="%Y-%m-%d", errors="coerce"
    )
    normalized_person = frame.apply(
        lambda row: normalize_official_person_fields(
            date_of_birth=row["date_of_birth"],
            club=row["club"],
            alternate_date=(
                re.search(r"\b\d{2}/\d{2}/\d{4}\b", row["notes"]).group(0)
                if re.search(r"\b\d{2}/\d{2}/\d{4}\b", row["notes"])
                else ""
            ),
        ),
        axis=1,
        result_type="expand",
    )
    frame["date_of_birth_parsed"] = normalized_person["normalized_date_of_birth"]
    frame["date_of_birth_normalized"] = frame["date_of_birth_parsed"].map(
        lambda value: value.date().isoformat() if pd.notna(value) else ""
    )
    frame["club_normalized"] = normalized_person["normalized_club"]
    frame["person_field_normalization"] = normalized_person["normalization_class"]
    frame["shirt_number_parsed"] = pd.to_numeric(
        frame["shirt_number"].replace("", pd.NA), errors="coerce"
    )

    conflicts = []
    errors = []
    required = [
        "snapshot_date",
        "national_team",
        "player_name",
        "position",
        "squad_status",
        "source_name",
        "source_url_or_reference",
        "source_publication_date",
    ]
    for column in required:
        rows = frame.loc[frame[column].eq(""), "source_row"].tolist()
        if rows:
            errors.append(f"{column} missing at source rows {rows[:10]}")
    if frame["snapshot_date_parsed"].isna().any():
        errors.append("invalid snapshot_date")
    if frame["publication_date_parsed"].isna().any():
        errors.append("invalid source_publication_date")
    invalid_groups = sorted(set(frame["group"]) - GROUPS)
    if invalid_groups:
        errors.append(f"invalid World Cup groups: {invalid_groups}")
    invalid_status = sorted(set(frame["squad_status"]) - set(config.allowed_statuses))
    if invalid_status:
        errors.append(f"invalid squad statuses: {invalid_status}")
    valid_positions = {"Goalkeeper", "Defender", "Midfielder", "Forward"}
    invalid_positions = sorted(set(frame["position"]) - valid_positions)
    if invalid_positions:
        errors.append(f"invalid squad positions: {invalid_positions}")
    if frame["canonical_team"].isna().any():
        teams = sorted(frame.loc[frame["canonical_team"].isna(), "national_team"].unique())
        errors.append(f"ambiguous/unmatched team identity: {teams}")
    if (
        frame["publication_date_parsed"] > frame["snapshot_date_parsed"]
    ).any():
        errors.append("source publication date is after snapshot date")
    if config.expected_rows is not None and len(frame) != config.expected_rows:
        errors.append(f"expected {config.expected_rows} rows, got {len(frame)}")
    if (
        config.expected_teams is not None
        and frame["canonical_team"].nunique() != config.expected_teams
    ):
        errors.append(
            f"expected {config.expected_teams} teams, got "
            f"{frame['canonical_team'].nunique()}"
        )
    team_sizes = frame.groupby("canonical_team").size()
    if (
        config.expected_players_per_team is not None
        and not team_sizes.eq(config.expected_players_per_team).all()
    ):
        errors.append(
            f"expected {config.expected_players_per_team} players per team: "
            f"{team_sizes[~team_sizes.eq(config.expected_players_per_team)].to_dict()}"
        )
    group_teams = frame[["group", "canonical_team"]].drop_duplicates()
    if config.expected_teams == 48 and (
        len(group_teams) != 48
        or set(group_teams["group"]) != GROUPS
        or not group_teams.groupby("group").size().eq(4).all()
    ):
        errors.append("expected exactly four teams in each group A-L")
    for columns, label in (
        (["snapshot_date", "canonical_team", "normalized_player_name"], "team/player"),
        (["snapshot_date", "canonical_team", "shirt_number"], "team/shirt_number"),
    ):
        if frame.duplicated(columns, keep=False).any():
            errors.append(f"duplicate {label} rows")

    duplicate_key = ["snapshot_date", "canonical_team", "normalized_player_name"]
    duplicates = frame.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        errors.append("duplicated canonical player within squad snapshot")
        for _, block in frame[duplicates].groupby(duplicate_key, dropna=False):
            conflicts.append(_conflict_row(block, "duplicate_player"))
    for _key, block in frame.groupby(duplicate_key, dropna=False):
        if len(block) < 2:
            continue
        for column in ("club", "position", "date_of_birth", "shirt_number"):
            values = {value for value in block[column] if value}
            if len(values) > 1:
                conflicts.append(_conflict_row(block, f"conflicting_{column}"))

    if errors:
        raise ValueError("; ".join(errors))
    retrieved_at = retrieved_at or datetime.now(UTC)
    for row in frame.itertuples(index=False):
        published = pd.Timestamp(row.publication_date_parsed).date()
        snapshot = pd.Timestamp(row.snapshot_date_parsed).date()
        available = datetime.combine(published, datetime.min.time(), tzinfo=UTC)
        SquadMembershipRecord.model_validate(
            {
                **{column: getattr(row, column) or None for column in SQUAD_COLUMNS},
                "snapshot_date": snapshot,
                "source_publication_date": published,
                "date_of_birth": (
                    pd.Timestamp(row.date_of_birth_parsed).date()
                    if pd.notna(row.date_of_birth_parsed)
                    else None
                ),
                "club": row.club_normalized or None,
                "shirt_number": (
                    int(row.shirt_number_parsed)
                    if pd.notna(row.shirt_number_parsed)
                    else None
                ),
                "provenance": ProvenanceEnvelope(
                    source=row.source_name,
                    source_record_id=str(row.source_player_id or row.source_row),
                    retrieved_at=retrieved_at,
                    available_at=available,
                    source_snapshot_hash=content_hash,
                    schema_version=SCHEMA_VERSION,
                ),
            }
        )
    manifest = build_snapshot_manifest(
        source="squad_2026",
        role="official_tournament_squad_membership",
        endpoint_or_url=path.name,
        available_at_semantics="each row available from source_publication_date",
        license=config.license,
        attribution=config.attribution,
        content_hash=content_hash,
        row_count=len(frame),
        schema_version=SCHEMA_VERSION,
        cache_path=str(path.resolve()),
        coverage_period_start=str(frame["snapshot_date_parsed"].min().date()),
        coverage_period_end=str(frame["snapshot_date_parsed"].max().date()),
        retrieval_timestamp=retrieved_at.isoformat(timespec="seconds"),
    ).model_dump()
    quality = {
        "valid": True,
        "rows": len(frame),
        "teams": int(frame["canonical_team"].nunique()),
        "groups": int(frame["group"].nunique()),
        "snapshot_id": manifest["snapshot_id"],
        "content_hash": content_hash,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }
    return frame, manifest, quality


def _conflict_row(block: pd.DataFrame, reason: str) -> dict:
    return {
        "snapshot_date": block.iloc[0]["snapshot_date"],
        "national_team": block.iloc[0]["national_team"],
        "player_name": block.iloc[0]["player_name"],
        "source_rows": ";".join(str(value) for value in block["source_row"]),
        "reason": reason,
    }


def assert_squads_available_at(frame: pd.DataFrame, cutoff) -> None:
    cutoff = pd.Timestamp(cutoff)
    future = frame[frame["publication_date_parsed"] > cutoff]
    if len(future):
        raise ValueError(
            f"{len(future)} squad rows were not published by cutoff {cutoff.isoformat()}"
        )


def write_squad_reports(frame: pd.DataFrame, manifest: dict, quality: dict) -> list[Path]:
    out = resolve("artifacts/reports")
    manifests = resolve("artifacts/manifests")
    out.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    quality_path = out / "squad_2026_quality.json"
    quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    team_counts = (
        frame.groupby(["snapshot_date", "group", "canonical_team", "squad_status"])
        .size()
        .rename("players")
        .reset_index()
    )
    team_counts.to_csv(out / "squad_2026_team_counts.csv", index=False)
    source_coverage = (
        frame.groupby(["source_name", "source_publication_date"])
        .agg(rows=("player_name", "size"), teams=("canonical_team", "nunique"))
        .reset_index()
    )
    source_coverage.to_csv(out / "squad_2026_source_coverage.csv", index=False)
    pd.DataFrame(
        quality["conflicts"],
        columns=["snapshot_date", "national_team", "player_name", "source_rows", "reason"],
    ).to_csv(out / "squad_2026_conflicts.csv", index=False)
    manifest_path = manifests / f"squad_2026_{manifest['snapshot_id']}.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable_existing = {k: v for k, v in existing.items() if k != "retrieval_timestamp"}
        comparable_new = {k: v for k, v in manifest.items() if k != "retrieval_timestamp"}
        if comparable_existing != comparable_new:
            raise FileExistsError(f"refusing to overwrite differing manifest {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return [
        quality_path,
        out / "squad_2026_team_counts.csv",
        out / "squad_2026_source_coverage.csv",
        out / "squad_2026_conflicts.csv",
        manifest_path,
    ]


def load_official_extract(path: str | Path) -> pd.DataFrame:
    path = resolve(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    required = [
        "group",
        "national_team",
        "shirt_number",
        "position",
        "fifa_player_name",
        "name_on_shirt",
        "date_of_birth",
        "club",
        "source_pdf_page",
        "source_url",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"official squad extract missing columns: {missing}")
    frame = frame.copy()
    frame["source_row"] = np.arange(2, len(frame) + 2)
    normalized = frame.apply(
        lambda row: normalize_official_person_fields(
            date_of_birth=row["date_of_birth"],
            club=row["club"],
            alternate_date=row["name_on_shirt"],
        ),
        axis=1,
        result_type="expand",
    )
    frame["date_of_birth_parsed"] = normalized["normalized_date_of_birth"]
    frame["club_normalized"] = normalized["normalized_club"]
    frame["person_field_normalization"] = normalized["normalization_class"]
    frame["normalized_player_name"] = frame["fifa_player_name"].map(
        normalize_player_name
    )
    return frame


def reconcile_official_extract(
    squads: pd.DataFrame, extract: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Reconcile primary membership to the expanded extract without mutation."""
    keys = ["group", "national_team", "shirt_number"]
    if extract.duplicated(keys, keep=False).any():
        raise ValueError("official extract has duplicate team/shirt-number rows")
    primary = squads.copy()
    primary["national_team"] = primary["national_team"].astype(str)
    joined = primary.merge(
        extract,
        on=keys,
        how="outer",
        suffixes=("_primary", "_extract"),
        indicator=True,
    ).rename(columns={"_merge": "merge_state"})
    rows = []
    for row in joined.to_dict("records"):
        merge_state = row["merge_state"]
        primary_name = row.get("player_name", "")
        extract_name = row.get("fifa_player_name", "")
        name_match = (
            normalize_player_name(primary_name) == normalize_player_name(extract_name)
            if merge_state == "both"
            else False
        )
        primary_dob = row.get("date_of_birth_parsed_primary", pd.NaT)
        extract_dob = row.get("date_of_birth_parsed_extract", pd.NaT)
        dob_match = (
            (pd.isna(primary_dob) and pd.isna(extract_dob))
            or (
                pd.notna(primary_dob)
                and pd.notna(extract_dob)
                and pd.Timestamp(primary_dob) == pd.Timestamp(extract_dob)
            )
        )
        primary_club = normalize_team(row.get("club_normalized_primary", ""))
        extract_club = normalize_team(row.get("club_normalized_extract", ""))
        club_match = primary_club == extract_club
        position_match = (
            normalize_team(row.get("position_primary", ""))
            == normalize_team(row.get("position_extract", ""))
        )
        page_present = bool(str(row.get("source_pdf_page", "")).strip())
        if merge_state != "both":
            classification = (
                "missing_from_extract"
                if merge_state == "left_only"
                else "missing_from_primary"
            )
        elif name_match and dob_match and club_match and position_match and page_present:
            shifts = {
                row.get("person_field_normalization_primary", ""),
                row.get("person_field_normalization_extract", ""),
            }
            classification = (
                "normalization_equivalent"
                if "optional_field_shift" in shifts
                else "exact"
            )
        else:
            classification = "substantive_discrepancy"
        rows.append(
            {
                "group": row.get("group", ""),
                "national_team": row.get("national_team", ""),
                "shirt_number": row.get("shirt_number", ""),
                "primary_player_name": primary_name,
                "extract_player_name": extract_name,
                "name_match": name_match,
                "date_of_birth_match": dob_match,
                "position_match": position_match,
                "club_match": club_match,
                "source_pdf_page_present": page_present,
                "primary_normalization": row.get(
                    "person_field_normalization_primary", ""
                ),
                "extract_normalization": row.get(
                    "person_field_normalization_extract", ""
                ),
                "classification": classification,
            }
        )
    report = pd.DataFrame(rows)
    counts = report["classification"].value_counts().to_dict()
    matched = int(
        report["classification"].isin(["exact", "normalization_equivalent"]).sum()
    )
    summary = {
        "primary_rows": len(squads),
        "extract_rows": len(extract),
        "matched_rows": matched,
        "reconciliation_rate": matched / len(squads) if len(squads) else None,
        "by_classification": {str(key): int(value) for key, value in counts.items()},
        "source_pages_missing": int((~report["source_pdf_page_present"]).sum()),
    }
    return report, summary


def write_reconciliation_reports(report: pd.DataFrame, summary: dict) -> list[Path]:
    out = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    full = out / "squad_extract_reconciliation.csv"
    discrepancies = out / "squad_extract_discrepancies.csv"
    summary_path = out / "squad_extract_summary.json"
    report.to_csv(full, index=False)
    report[
        ~report["classification"].isin(["exact", "normalization_equivalent"])
    ].to_csv(discrepancies, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return [full, discrepancies, summary_path]


def _name_token_key(name: str) -> tuple[str, ...]:
    return tuple(sorted(normalize_player_name(name).split()))


def load_seed_link_candidates(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        resolve(path), encoding="utf-8-sig", dtype=str, keep_default_na=False
    )
    required = [
        "national_team",
        "official_player_name",
        "date_of_birth",
        "transfermarkt_player_id",
        "transfermarkt_name",
        "link_status",
        "source_url",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"seed link CSV missing columns: {missing}")
    frame = frame.copy()
    frame["source_row"] = np.arange(2, len(frame) + 2)
    return frame


def revalidate_seed_links(
    squads: pd.DataFrame, candidates: pd.DataFrame, players: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    squad_lookup = {}
    for row in squads.itertuples(index=False):
        for team_name in {row.canonical_team, row.national_team}:
            squad_lookup[
                (normalize_team(team_name), normalize_player_name(row.player_name))
            ] = row
    players_by_id = {
        str(row.player_id): row for row in players.itertuples(index=False)
    }
    rows = []
    accepted_ids: dict[str, tuple[str, str]] = {}
    for candidate in candidates.itertuples(index=False):
        key = (
            normalize_team(candidate.national_team),
            normalize_player_name(candidate.official_player_name),
        )
        squad = squad_lookup.get(key)
        source_id = str(candidate.transfermarkt_player_id).strip()
        player = players_by_id.get(source_id)
        status = str(candidate.link_status).strip()
        if squad is None:
            classification = "conflicting"
            reason = "candidate does not map to one official squad player"
        elif status != "exact_dob_name":
            classification = "unmatched"
            reason = "source candidate status is unresolved"
        elif not source_id or player is None:
            classification = "rejected stale"
            reason = "Transfermarkt player ID is missing or absent"
        else:
            squad_dob = squad.date_of_birth_parsed
            candidate_dob = _parse_source_date(candidate.date_of_birth)
            player_dob = _parse_source_date(getattr(player, "date_of_birth", ""))
            dob_match = (
                pd.notna(squad_dob)
                and pd.notna(candidate_dob)
                and pd.notna(player_dob)
                and pd.Timestamp(squad_dob) == candidate_dob == player_dob
            )
            name_match = (
                _name_token_key(candidate.official_player_name)
                == _name_token_key(candidate.transfermarkt_name)
                == _name_token_key(getattr(player, "name", ""))
            )
            if not dob_match:
                classification = "conflicting"
                reason = "date of birth failed revalidation"
            elif not name_match:
                classification = "ambiguous"
                reason = "normalized name failed revalidation"
            elif source_id in accepted_ids and accepted_ids[source_id] != key:
                classification = "conflicting"
                reason = "Transfermarkt player maps to multiple squad players"
            else:
                classification = "accepted deterministic"
                reason = "exact date of birth and normalized name revalidated"
                accepted_ids[source_id] = key
        rows.append(
            {
                "source_row": candidate.source_row,
                "national_team": candidate.national_team,
                "official_player_name": candidate.official_player_name,
                "date_of_birth": candidate.date_of_birth,
                "transfermarkt_player_id": source_id,
                "transfermarkt_name": candidate.transfermarkt_name,
                "input_link_status": status,
                "classification": classification,
                "reason": reason,
            }
        )
    report = pd.DataFrame(rows)
    counts = report["classification"].value_counts().to_dict()
    summary = {
        "total": len(report),
        "accepted_deterministic": int(
            report["classification"].eq("accepted deterministic").sum()
        ),
        "by_classification": {str(key): int(value) for key, value in counts.items()},
    }
    return report, summary


def write_seed_link_reports(report: pd.DataFrame, summary: dict) -> list[Path]:
    out = resolve("artifacts/reports")
    full = out / "squad_seed_link_revalidation.csv"
    summary_path = out / "squad_seed_link_summary.json"
    report.to_csv(full, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return [full, summary_path]


def load_reviewed_aliases(path: str | Path | None) -> pd.DataFrame:
    columns = [
        "national_team",
        "squad_player_name",
        "transfermarkt_player_id",
        "review_status",
        "notes",
    ]
    if not path:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(resolve(path), encoding="utf-8-sig", dtype=str).fillna("")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"player alias CSV missing columns: {missing}")
    accepted = frame["review_status"].str.casefold().isin({"reviewed", "accepted"})
    return frame.loc[accepted, columns].copy()


def link_squad_players(
    squads: pd.DataFrame,
    players: pd.DataFrame,
    aliases: pd.DataFrame | None = None,
    seed_links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify each squad player; never accept a name-only match."""
    players = players.copy()
    players["normalized_name"] = players["name"].fillna("").map(normalize_player_name)
    players["name_token_key"] = players["name"].fillna("").map(_name_token_key)
    players["dob"] = pd.to_datetime(players.get("date_of_birth"), errors="coerce")
    aliases = aliases if aliases is not None else load_reviewed_aliases(None)
    alias_lookup = {
        (normalize_team(row.national_team), normalize_player_name(row.squad_player_name)):
        str(row.transfermarkt_player_id)
        for row in aliases.itertuples(index=False)
    }
    seed_links = seed_links if seed_links is not None else pd.DataFrame()
    accepted_seeds = seed_links[
        seed_links.get("classification", pd.Series(dtype=str)).eq(
            "accepted deterministic"
        )
    ]
    seed_lookup = {
        (
            normalize_team(row.national_team),
            normalize_player_name(row.official_player_name),
        ): str(row.transfermarkt_player_id)
        for row in accepted_seeds.itertuples(index=False)
    }
    rows = []
    for squad in squads.itertuples(index=False):
        name = squad.normalized_player_name
        team = normalize_team(squad.canonical_team)
        explicit = alias_lookup.get((team, name))
        if explicit:
            candidates = players[players["player_id"].astype(str).eq(explicit)]
            method = "reviewed_alias"
        elif seed_lookup.get((team, name)):
            seed_id = seed_lookup[(team, name)]
            candidates = players[players["player_id"].astype(str).eq(seed_id)]
            method = "seed_exact_dob_name"
        elif squad.source_player_id:
            candidates = players[players["player_id"].astype(str).eq(str(squad.source_player_id))]
            method = "exact_source_id"
        else:
            token_key = _name_token_key(squad.player_name)
            name_hits = players[
                players["name_token_key"].map(
                    lambda candidate, expected=token_key: candidate == expected
                )
            ]
            dob = squad.date_of_birth_parsed
            if pd.notna(dob):
                candidates = name_hits[name_hits["dob"].eq(pd.Timestamp(dob))]
                method = "name_date_of_birth"
            else:
                nationality = name_hits.get(
                    "country_of_citizenship", pd.Series("", index=name_hits.index)
                ).fillna("").map(normalize_team)
                club = name_hits.get(
                    "current_club_name", pd.Series("", index=name_hits.index)
                ).fillna("").map(normalize_team)
                position = name_hits.get(
                    "position", pd.Series("", index=name_hits.index)
                ).fillna("").map(normalize_team)
                corroborated = nationality.eq(team)
                if squad.club:
                    corroborated &= club.eq(normalize_team(squad.club))
                    method = "name_nationality_club"
                else:
                    corroborated &= position.eq(normalize_team(squad.position))
                    method = "name_nationality_position"
                candidates = name_hits[corroborated]
        count = len(candidates)
        if count == 1:
            candidate = candidates.iloc[0]
            match_class = (
                "exact"
                if method
                in {"reviewed_alias", "seed_exact_dob_name", "exact_source_id"}
                else "high-confidence deterministic"
            )
            canonical_id = f"tm:{candidate['player_id']}"
            source_id = str(candidate["player_id"])
            review = "accepted"
        elif count > 1:
            match_class, canonical_id, source_id, review = "ambiguous", "", "", "manual_review"
        else:
            name_count = int(
                players["name_token_key"]
                .map(
                    lambda candidate, expected=token_key: candidate == expected
                )
                .sum()
            )
            match_class = "ambiguous" if name_count else "unmatched"
            canonical_id = source_id = ""
            review = "manual_review"
            count = name_count
            if name_count:
                method = "name_only_rejected"
        rows.append(
            {
                "snapshot_date": squad.snapshot_date,
                "group": squad.group,
                "national_team": squad.canonical_team,
                "player_name": squad.player_name,
                "position": squad.position,
                "normalized_player_name": name,
                "canonical_player_id": canonical_id,
                "transfermarkt_player_id": source_id,
                "match_class": match_class,
                "match_method": method,
                "candidate_count": count,
                "review_status": review,
            }
        )
    links = pd.DataFrame(rows, columns=LINK_COLUMNS)
    accepted = links[links["canonical_player_id"].ne("")]
    duplicated = accepted.duplicated(
        ["snapshot_date", "canonical_player_id"], keep=False
    )
    if duplicated.any():
        links.loc[accepted.index[duplicated], ["match_class", "review_status"]] = [
            "conflicting",
            "manual_review",
        ]
        links.loc[
            accepted.index[duplicated],
            ["canonical_player_id", "transfermarkt_player_id"],
        ] = ""
    return links


def write_link_reports(links: pd.DataFrame) -> dict:
    out = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    links.to_csv(out / "squad_player_links.csv", index=False)
    links[links["match_class"].eq("unmatched")].to_csv(
        out / "squad_player_unmatched.csv", index=False
    )
    links[links["match_class"].eq("ambiguous")].to_csv(
        out / "squad_player_ambiguous.csv", index=False
    )
    links[links["match_class"].eq("conflicting")].to_csv(
        out / "squad_player_conflicts.csv", index=False
    )
    counts = links["match_class"].value_counts().to_dict()
    accepted = int(
        links["match_class"].isin({"exact", "high-confidence deterministic"}).sum()
    )
    summary = {
        "total": len(links),
        "linked": accepted,
        "link_rate": accepted / len(links) if len(links) else None,
        "by_class": {str(key): int(value) for key, value in counts.items()},
        "by_method": {
            str(key): int(value)
            for key, value in links["match_method"].value_counts().to_dict().items()
        },
        "link_rate_by_team": _link_rate_records(links, ["national_team"]),
        "link_rate_by_group": _link_rate_records(links, ["group"]),
        "link_rate_by_position": _link_rate_records(links, ["position"]),
    }
    (out / "squad_player_link_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _link_rate_records(links: pd.DataFrame, dimensions: list[str]) -> list[dict]:
    frame = links.copy()
    frame["linked"] = frame["canonical_player_id"].ne("")
    return (
        frame.groupby(dimensions, dropna=False)["linked"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(
            columns={
                "sum": "linked_players",
                "count": "squad_players",
                "mean": "link_rate",
            }
        )
        .to_dict("records")
    )


def write_alias_review_file(
    squads: pd.DataFrame, links: pd.DataFrame, players: pd.DataFrame
) -> Path:
    """Write unresolved rows for human review without accepting any candidate."""
    unresolved = links[links["canonical_player_id"].eq("")].copy()
    squad_fields = squads[
        [
            "canonical_team",
            "player_name",
            "date_of_birth_normalized",
            "club_normalized",
        ]
    ].rename(columns={"canonical_team": "national_team"})
    unresolved = unresolved.merge(
        squad_fields, on=["national_team", "player_name"], how="left"
    )
    player_names = players.assign(
        transfermarkt_player_id=players["player_id"].astype(str)
    )[["transfermarkt_player_id", "name"]].rename(
        columns={"name": "transfermarkt_name"}
    )
    unresolved = unresolved.merge(
        player_names, on="transfermarkt_player_id", how="left"
    )
    review = pd.DataFrame(
        {
            "national_team": unresolved["national_team"],
            "official_squad_player_name": unresolved["player_name"],
            "date_of_birth": unresolved["date_of_birth_normalized"],
            "official_club": unresolved["club_normalized"],
            "candidate_transfermarkt_player_id": unresolved[
                "transfermarkt_player_id"
            ],
            "transfermarkt_name": unresolved["transfermarkt_name"].fillna(""),
            "match_evidence": unresolved["match_method"],
            "review_status": "pending",
            "reviewer": "",
            "reviewed_at": "",
            "notes": "",
        }
    )
    path = resolve("data/reference/world_cup_2026_player_aliases.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(path, index=False)
    return path


def build_player_activity(
    links: pd.DataFrame,
    appearances: pd.DataFrame,
    lineups: pd.DataFrame,
    *,
    cutoff,
    windows: tuple[int, ...] = (30, 90, 180, 365),
    target_game_id: str | int | None = None,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    appearances = appearances.copy()
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    appearances = appearances[appearances["date"].lt(cutoff)]
    if target_game_id is not None:
        appearances = appearances[
            appearances["game_id"].astype(str).ne(str(target_game_id))
        ]
    lineups = lineups.copy()
    lineups["date"] = pd.to_datetime(lineups["date"], errors="coerce")
    lineups = lineups[lineups["date"].lt(cutoff)]
    if target_game_id is not None:
        lineups = lineups[lineups["game_id"].astype(str).ne(str(target_game_id))]
    appearances["_player_key"] = appearances["player_id"].astype(str)
    lineups["_player_key"] = lineups["player_id"].astype(str)
    appearances_by_player = {  # noqa: C416 - pandas GroupBy is not a mapping
        key: block
        for key, block in appearances.groupby("_player_key", sort=False)
    }
    starts_by_player = {
        key: block[
            block["type"].fillna("").str.casefold().str.contains("starting")
        ]
        for key, block in lineups.groupby("_player_key", sort=False)
    }
    rows = []
    linked = links[links["transfermarkt_player_id"].ne("")]
    for link in linked.itertuples(index=False):
        player_id = str(link.transfermarkt_player_id)
        apps = appearances_by_player.get(player_id, appearances.iloc[0:0])
        starts = starts_by_player.get(player_id, lineups.iloc[0:0])
        row = {
            "snapshot_date": link.snapshot_date,
            "group": getattr(link, "group", ""),
            "national_team": link.national_team,
            "player_name": link.player_name,
            "position": getattr(link, "position", ""),
            "canonical_player_id": link.canonical_player_id,
            "transfermarkt_player_id": player_id,
            "match_class": getattr(link, "match_class", ""),
            "prediction_cutoff": cutoff.isoformat(),
            "days_since_last_appearance": _days_since(apps["date"], cutoff),
            "days_since_last_start": _days_since(starts["date"], cutoff),
        }
        latest = apps.sort_values("date").iloc[-1] if len(apps) else None
        row["last_appearance_club_id"] = (
            str(latest.get("player_club_id", "")) if latest is not None else ""
        )
        row["last_appearance_competition_id"] = (
            str(latest.get("competition_id", "")) if latest is not None else ""
        )
        for days in windows:
            start = cutoff - pd.Timedelta(days=days)
            window_apps = apps[apps["date"].ge(start)]
            window_starts = starts[starts["date"].ge(start)]
            minutes = pd.to_numeric(window_apps.get("minutes_played"), errors="coerce")
            row[f"minutes_{days}d"] = minutes.sum(min_count=1)
            row[f"appearances_{days}d"] = len(window_apps)
            row[f"starts_{days}d"] = len(window_starts)
            if days >= 90:
                for stat in ("goals", "assists", "yellow_cards", "red_cards"):
                    values = pd.to_numeric(window_apps.get(stat), errors="coerce")
                    row[f"{stat}_{days}d"] = values.sum(min_count=1)
        rows.append(row)
    return pd.DataFrame(rows)


def _days_since(dates: pd.Series, cutoff: pd.Timestamp) -> int | None:
    dates = dates.dropna()
    return None if dates.empty else int((cutoff - dates.max()).days)


def build_historical_valuations(
    links: pd.DataFrame,
    valuations: pd.DataFrame,
    *,
    cutoff,
    source_snapshot_id: str,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    valuations = valuations.copy()
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations = valuations[valuations["date"].le(cutoff)]
    valuations["_player_key"] = valuations["player_id"].astype(str)
    values_by_player = {
        key: block.sort_values("date")
        for key, block in valuations.groupby("_player_key", sort=False)
    }
    rows = []
    for link in links[links["transfermarkt_player_id"].ne("")].itertuples(index=False):
        player = values_by_player.get(
            str(link.transfermarkt_player_id), valuations.iloc[0:0]
        )
        latest = player.iloc[-1] if len(player) else None
        rows.append(
            {
                "national_team": link.national_team,
                "group": getattr(link, "group", ""),
                "player_name": link.player_name,
                "position": getattr(link, "position", ""),
                "match_class": getattr(link, "match_class", ""),
                "canonical_player_id": link.canonical_player_id,
                "transfermarkt_player_id": link.transfermarkt_player_id,
                "prediction_cutoff": cutoff.isoformat(),
                "historical_valuation": (
                    float(latest["market_value_in_eur"])
                    if latest is not None and pd.notna(latest["market_value_in_eur"])
                    else np.nan
                ),
                "currency": "EUR",
                "valuation_date": (
                    latest["date"].date().isoformat() if latest is not None else ""
                ),
                "valuation_age_days": (
                    int((cutoff - latest["date"]).days) if latest is not None else np.nan
                ),
                "available": latest is not None,
                "source_snapshot_id": source_snapshot_id,
            }
        )
    return pd.DataFrame(rows)


def position_group(position: str, config: SquadDataConfig | None = None) -> str:
    config = config or SquadDataConfig.load()
    normalized = normalize_team(position)
    for group, terms in config.position_groups.items():
        if any(term in normalized for term in terms):
            return group
    return "unknown"


def build_squad_aggregates(
    squads: pd.DataFrame,
    links: pd.DataFrame,
    activity: pd.DataFrame,
    valuations: pd.DataFrame,
    config: SquadDataConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or SquadDataConfig.load()
    squad_base = squads.drop(columns=["national_team"]).rename(
        columns={"canonical_team": "national_team"}
    )
    base = squad_base.merge(
        links[
            ["snapshot_date", "national_team", "player_name", "canonical_player_id"]
        ],
        on=["snapshot_date", "national_team", "player_name"],
        how="left",
    )
    base["position_group"] = base["position"].map(lambda value: position_group(value, config))
    base = base.merge(
        activity, on=["national_team", "player_name", "canonical_player_id"], how="left"
    ).merge(
        valuations[
            [
                "national_team",
                "player_name",
                "canonical_player_id",
                "historical_valuation",
            ]
        ],
        on=["national_team", "player_name", "canonical_player_id"],
        how="left",
    )
    summaries = []
    depth = []
    position_rows = []
    missingness = []
    for team, block in base.groupby("national_team"):
        minutes = pd.to_numeric(block.get("minutes_90d"), errors="coerce")
        values = pd.to_numeric(block.get("historical_valuation"), errors="coerce")
        summaries.append(
            {
                "national_team": team,
                "squad_players": len(block),
                "linked_players": int(block["canonical_player_id"].fillna("").ne("").sum()),
                "total_recent_club_minutes_90d": minutes.sum(min_count=1),
                "median_recent_club_minutes_90d": minutes.median(),
                "total_recent_club_starts_90d": pd.to_numeric(
                    block.get("starts_90d"), errors="coerce"
                ).sum(min_count=1),
                "players_with_recent_club_minutes": int(minutes.notna().sum()),
                "players_recently_active_90d": int(minutes.fillna(0).gt(0).sum()),
                "players_inactive_90d": int(minutes.fillna(0).eq(0).sum()),
                "historical_valuation_coverage": float(values.notna().mean()),
                "minutes_weighted_historical_valuation": (
                    float((values * minutes).sum() / minutes[values.notna()].sum())
                    if values.notna().any()
                    and minutes[values.notna()].fillna(0).sum() > 0
                    else np.nan
                ),
                "descriptive_only": True,
            }
        )
        sorted_minutes = minutes.dropna().sort_values(ascending=False)
        depth.append(
            {
                "national_team": team,
                **{
                    f"top_{count}_minutes_90d": sorted_minutes.head(count).sum(min_count=1)
                    for count in (11, 15, 23)
                },
                "expected_xi_claim": False,
            }
        )
        for group, position_block in block.groupby("position_group"):
            position_rows.append(
                {
                    "national_team": team,
                    "position_group": group,
                    "players": len(position_block),
                    "linked": int(
                        position_block["canonical_player_id"].fillna("").ne("").sum()
                    ),
                    "minutes_90d_coverage": float(
                        pd.to_numeric(
                            position_block.get("minutes_90d"), errors="coerce"
                        ).notna().mean()
                    ),
                    "total_minutes_90d": pd.to_numeric(
                        position_block.get("minutes_90d"), errors="coerce"
                    ).sum(min_count=1),
                    "total_starts_90d": pd.to_numeric(
                        position_block.get("starts_90d"), errors="coerce"
                    ).sum(min_count=1),
                }
            )
        missingness.append(
            {
                "national_team": team,
                "identity_missing_rate": float(
                    block["canonical_player_id"].fillna("").eq("").mean()
                ),
                "minutes_90d_missing_rate": float(minutes.isna().mean()),
                "valuation_missing_rate": float(values.isna().mean()),
            }
        )
    return {
        "activity": pd.DataFrame(summaries),
        "position": pd.DataFrame(position_rows),
        "depth": pd.DataFrame(depth),
        "missingness": pd.DataFrame(missingness),
        "player_level": base,
    }


def national_lineup_coverage(
    source, squad_links: pd.DataFrame | None = None
) -> pd.DataFrame:
    names = set(source.table_names())
    if not {"games", "game_lineups"}.issubset(names):
        return pd.DataFrame(
            columns=["national_team", "games", "lineup_rows", "readiness"]
        )
    games = source.read_table(
        "games",
        columns=[
            "game_id", "date", "home_club_id", "away_club_id",
            "home_club_name", "away_club_name", "competition_type",
            "home_club_formation", "away_club_formation",
        ],
    )
    games = games[games["competition_type"].eq("national_team_competition")]
    lineups = source.read_table(
        "game_lineups", columns=["game_id", "player_id", "type", "position", "club_id"]
    )
    joined = lineups.merge(games, on="game_id", how="inner")
    teams = (
        sorted(squad_links["national_team"].unique())
        if squad_links is not None and len(squad_links)
        else sorted(set(games["home_club_name"].dropna()) | set(games["away_club_name"].dropna()))
    )
    rows = []
    for team in teams:
        team_games = games[
            games["home_club_name"].eq(team) | games["away_club_name"].eq(team)
        ].copy()
        expected_club = {
            str(row.game_id): str(
                row.home_club_id if row.home_club_name == team else row.away_club_id
            )
            for row in team_games.itertuples(index=False)
        }
        lineup_rows = joined[
            joined["game_id"].astype(str).isin(expected_club)
            & joined.apply(
                lambda row, clubs=expected_club: (
                    str(row["club_id"]) == clubs.get(str(row["game_id"]))
                ),
                axis=1,
            )
        ]
        game_count = int(team_games["game_id"].nunique())
        covered_games = int(lineup_rows["game_id"].nunique())
        dates = pd.to_datetime(team_games["date"], errors="coerce")
        recent_ids = set(
            team_games.loc[
                dates.ge(pd.Timestamp("2025-06-15")), "game_id"
            ].astype(str)
        )
        types = lineup_rows["type"].fillna("").str.casefold()
        formation_available = team_games.apply(
            lambda row, current_team=team: bool(
                str(
                    row["home_club_formation"]
                    if row["home_club_name"] == current_team
                    else row["away_club_formation"]
                ).strip()
            ),
            axis=1,
        )
        if covered_games >= 10:
            readiness = "strong"
        elif covered_games >= 4:
            readiness = "partial"
        elif covered_games:
            readiness = "sparse"
        else:
            readiness = "unavailable"
        rows.append(
            {
                "national_team": team,
                "games": game_count,
                "games_with_lineups": covered_games,
                "recent_games_with_lineups": int(
                    lineup_rows[
                        lineup_rows["game_id"].astype(str).isin(recent_ids)
                    ]["game_id"].nunique()
                ),
                "lineup_rows": len(lineup_rows),
                "starter_rows": int(types.str.contains("starting").sum()),
                "bench_rows": int(types.str.contains("substitute").sum()),
                "goalkeeper_rows": int(
                    lineup_rows["position"].fillna("").str.casefold().str.contains("goal").sum()
                ),
                "goalkeeper_continuity_available": bool(
                    lineup_rows["position"]
                    .fillna("")
                    .str.casefold()
                    .str.contains("goal")
                    .any()
                ),
                "formation_games": int(formation_available.sum()),
                "readiness": readiness,
            }
        )
    return pd.DataFrame(rows)


def write_missing_source_reports(reason: str) -> dict:
    out = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    quality = {
        "valid": False,
        "state": "blocked by missing squad source",
        "reason": reason,
        "required_env": "FIFA_2026_SQUADS_PATH",
        "template": "data/reference/world_cup_2026_squads_template.csv",
    }
    (out / "squad_2026_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    for name, columns in {
        "squad_2026_team_counts.csv": ["snapshot_date", "group", "national_team", "players"],
        "squad_2026_source_coverage.csv": ["source_name", "rows", "teams"],
        "squad_2026_conflicts.csv": ["national_team", "player_name", "reason"],
        "squad_player_links.csv": LINK_COLUMNS,
        "squad_player_unmatched.csv": LINK_COLUMNS,
        "squad_player_ambiguous.csv": LINK_COLUMNS,
        "squad_player_conflicts.csv": LINK_COLUMNS,
        "player_activity_coverage.csv": ["players", "cutoff", "state"],
        "player_activity_missingness.csv": ["field", "missing_rate", "state"],
        "player_valuation_coverage.csv": ["players", "available", "coverage", "state"],
        "player_valuation_age.csv": [
            "national_team", "player_name", "valuation_date", "valuation_age_days",
            "available",
        ],
        "squad_2026_activity_summary.csv": [
            "national_team", "squad_players", "linked_players",
            "total_recent_club_minutes_90d", "historical_valuation_coverage",
            "descriptive_only",
        ],
        "squad_2026_position_summary.csv": [
            "national_team", "position_group", "players", "linked",
            "minutes_90d_coverage",
        ],
        "squad_2026_depth_summary.csv": [
            "national_team", "top_11_minutes_90d", "top_15_minutes_90d",
            "top_23_minutes_90d", "expected_xi_claim",
        ],
        "squad_2026_missingness.csv": [
            "national_team", "identity_missing_rate", "minutes_90d_missing_rate",
            "valuation_missing_rate",
        ],
        "portugal_squad_activity.csv": [
            "national_team", "player_name", "canonical_player_id", "state",
        ],
    }.items():
        pd.DataFrame(columns=columns).to_csv(out / name, index=False)
    (out / "squad_player_link_summary.json").write_text(
        json.dumps({"total": 0, "linked": 0, "link_rate": None, "state": quality["state"]},
                   indent=2),
        encoding="utf-8",
    )
    (out / "player_activity_temporal_validation.json").write_text(
        json.dumps(
            {
                "valid": False,
                "state": quality["state"],
                "strictly_prior": None,
                "current_profile_fields_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "portugal_squad_data_audit.md").write_text(
        "# Portugal Squad Data Audit\n\n"
        f"State: **{quality['state']}**.\n\n"
        "No player was assumed selected and no forecast probability was changed.\n",
        encoding="utf-8",
    )
    return quality


def build_feature_readiness(
    *,
    squad_available: bool,
    identity_link_rate: float | None,
    statsbomb_available: bool,
    international_lineup_ready: bool,
) -> dict:
    squad_block = "blocked by missing squad source"
    identity_block = "blocked by identity coverage"
    base = (
        "ready with cutoff"
        if squad_available and identity_link_rate is not None and identity_link_rate >= 0.8
        else identity_block if squad_available else squad_block
    )
    families = {
        "recent club minutes": base,
        "recent club starts": base,
        "recent club goals/assists": base,
        "historical valuations": base,
        "player age": "ready" if squad_available else squad_block,
        "position depth": base,
        "goalkeeper activity": base,
        "expected-XI strength": "blocked by sparse international lineups",
        "bench strength": base,
        "lineup continuity": (
            "restricted subset"
            if international_lineup_ready
            else "blocked by sparse international lineups"
        ),
        "goalkeeper continuity": (
            "restricted subset"
            if international_lineup_ready
            else "blocked by sparse international lineups"
        ),
        "confirmed lineups": "blocked by provider plan",
        "injuries": "unsupported",
        "suspensions": "unsupported",
        "club strength": "restricted subset",
        "competition strength": "restricted subset",
        "path difficulty": "ready",
        "statsbomb international lineups": (
            "restricted subset"
            if statsbomb_available
            else "blocked by sparse international lineups"
        ),
    }
    if set(families.values()) - READINESS_STATES:
        raise ValueError("invalid readiness state")
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "squad_source_available": squad_available,
        "identity_link_rate": identity_link_rate,
        "families": families,
    }


def write_readiness_reports(readiness: dict) -> tuple[Path, Path]:
    out = resolve("artifacts/reports")
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "squad_feature_readiness.json"
    md_path = out / "squad_feature_readiness.md"
    json_path.write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    lines = [
        "# Squad Feature Readiness",
        "",
        "| Feature family | State |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| {family} | **{state}** |"
        for family, state in readiness["families"].items()
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
