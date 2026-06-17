"""Configuration loaders for enrichment sources.

Mirrors the existing `DataConfig.load()` pattern: each config is a pydantic
model loaded from a YAML file resolved against the repo root. All source
behavior (URLs, competition IDs, rate limits, retries, paths, feature
windows, strength weights) is config-driven — nothing is hard-coded.

Credentials are never stored in YAML; they are read from environment
variables named here and supplied via a git-ignored `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from goalsignal.utils.paths import resolve


def _load_yaml(path: str | Path) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class SourceRole(BaseModel):
    name: str
    role: str
    enabled: bool = False
    license: str
    attribution: str
    credential_env: str | None = None
    path_env: str | None = None
    notes: str | None = None

    def is_configured(self) -> bool:
        """True when the credential/path this source needs is present in env."""
        for var in (self.credential_env, self.path_env):
            if var and os.environ.get(var):
                return True
        # Sources needing neither a key nor a path are configured by default.
        return self.credential_env is None and self.path_env is None


class SourcesConfig(BaseModel):
    """Registry of enrichment sources and their roles."""

    sources: list[SourceRole] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path = "config/sources.yaml") -> SourcesConfig:
        return cls.model_validate(_load_yaml(path))


class RetryPolicy(BaseModel):
    max_retries: int = 3
    backoff_seconds: float = 1.0
    timeout_seconds: float = 20.0


class StatsBombConfig(BaseModel):
    enabled: bool = False
    data_path_env: str = "STATSBOMB_DATA_PATH"
    competitions: list[dict] = Field(default_factory=list)
    feature_windows_matches: list[int] = Field(default_factory=lambda: [3, 5, 10])
    feature_windows_days: list[int] = Field(default_factory=lambda: [180, 365])
    optional_dependency: str = "statsbombpy"
    license: str = "StatsBomb Open Data — non-commercial, attribution required"
    attribution: str = "Data provided by StatsBomb (https://github.com/statsbomb/open-data)"

    @classmethod
    def load(cls, path: str | Path = "config/statsbomb.yaml") -> StatsBombConfig:
        return cls.model_validate(_load_yaml(path))


class WorldCupMapping(BaseModel):
    """Discovered API-Football identifier for the 2026 World Cup (with provenance)."""

    league_id: int | None = None
    season: int | None = None
    name: str | None = None
    country: str | None = None
    discovered_at: str | None = None
    source_snapshot_id: str | None = None


class ApiFootballConfig(BaseModel):
    """API-Sports / API-Football v3 (direct access, x-apisports-key).

    Replaces the earlier (incorrect) football-data.org integration. The key is
    read from FOOTBALL_DATA_API_KEY but is an API-Sports key; it is only ever
    sent to `base_url`'s host.
    """

    enabled: bool = False
    provider: str = "api-football"
    vendor: str = "API-Sports"
    base_url: str = "https://v3.football.api-sports.io"
    auth_header: str = "x-apisports-key"
    credential_env: str = "FOOTBALL_DATA_API_KEY"
    # Free plan: 100 requests/day. Reserve some for manual debugging.
    daily_request_limit: int = 100
    daily_request_reserve: int = 10
    max_requests_per_minute: int = 8
    cache_first: bool = True
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    cache_dir: str = "data/external/api_football"
    world_cup: WorldCupMapping = Field(default_factory=WorldCupMapping)

    @classmethod
    def load(cls, path: str | Path = "config/api_football.yaml") -> ApiFootballConfig:
        return cls.model_validate(_load_yaml(path))


class SourcePathError(ValueError):
    """A configured source path is missing, wrong type, or unreadable."""


def validate_source_path(
    path_str: str, *, kind: str, extensions: tuple[str, ...] | None = None
) -> Path:
    """Validate a configured file/dir path with actionable errors.

    `kind` is "file" or "dir". Raises SourcePathError for: empty, missing,
    wrong type, unsupported extension, unreadable.
    """
    import os

    if not path_str:
        raise SourcePathError("path is not configured (env var empty)")
    p = resolve(path_str)
    if not p.exists():
        raise SourcePathError(f"path does not exist: {p}")
    if kind == "file" and not p.is_file():
        raise SourcePathError(f"expected a file but found a directory: {p}")
    if kind == "dir" and not p.is_dir():
        raise SourcePathError(f"expected a directory but found a file: {p}")
    if kind == "file" and extensions and p.suffix.lower() not in extensions:
        raise SourcePathError(
            f"unsupported extension {p.suffix!r} (expected one of {extensions}): {p}"
        )
    if not os.access(p, os.R_OK):
        raise SourcePathError(f"path is not readable: {p}")
    return p


class FifaRankingsConfig(BaseModel):
    """Historical FIFA ranking timeline (FIFA_RANKINGS_PATH) and the separate
    World Cup pre-tournament rank validation file (FIFA_WC_TEAMS_PATH).

    The two files have distinct roles and must not be overloaded onto one var.
    """

    enabled: bool = False
    path_env: str = "FIFA_RANKINGS_PATH"
    wc_teams_path_env: str = "FIFA_WC_TEAMS_PATH"
    license: str = "user-provided; verify FIFA terms before redistribution"
    attribution: str = "FIFA/Coca-Cola World Ranking"
    # Real schema of ranking_fifa_historical.csv (validated, not assumed).
    expected_columns: list[str] = Field(
        default_factory=lambda: ["team", "total_points", "date"]
    )
    wc_teams_expected_columns: list[str] = Field(
        default_factory=lambda: ["year", "team", "confederation", "rank"]
    )

    @classmethod
    def load(cls, path: str | Path = "config/fifa_rankings.yaml") -> FifaRankingsConfig:
        return cls.model_validate(_load_yaml(path))


class FifaCurrentRankingsConfig(BaseModel):
    """Frozen World Cup field snapshot, separate from historical rankings."""

    enabled: bool = False
    path_env: str = "FIFA_CURRENT_RANKINGS_PATH"
    release_date: str = "2026-06-11"
    expected_columns: list[str] = Field(
        default_factory=lambda: ["group", "team", "fifa_rank"]
    )
    license: str = "user-provided; verify FIFA terms before redistribution"
    attribution: str = "FIFA/Coca-Cola World Ranking"

    @classmethod
    def load(
        cls, path: str | Path = "config/fifa_current_rankings.yaml"
    ) -> FifaCurrentRankingsConfig:
        return cls.model_validate(_load_yaml(path))


class PlayerDataConfig(BaseModel):
    """Transfermarkt-derived player/club history.

    The source may be a DuckDB file or a directory of gzipped CSV tables
    (the transfermarkt-datasets export format). It is opened READ-ONLY; the
    source files are never mutated.
    """

    enabled: bool = False
    path_env: str = "PLAYER_DATA_PATH"
    license: str = (
        "Transfermarkt-derived; non-commercial research use. Verify Transfermarkt "
        "terms before redistribution. Do not scrape."
    )
    attribution: str = "Data derived from Transfermarkt (transfermarkt-datasets)"
    position_groups: list[str] = Field(
        default_factory=lambda: ["goalkeeper", "defence", "midfield", "attack", "bench"]
    )
    # Fields that are CURRENT-STATE on the players table and unsafe to apply to
    # historical matches (documented in the temporal audit).
    current_state_unsafe_fields: list[str] = Field(
        default_factory=lambda: [
            "current_club_id", "current_club_name", "current_national_team_id",
            "international_caps", "international_goals", "market_value_in_eur",
            "highest_market_value_in_eur", "last_season", "contract_expiration_date",
        ]
    )

    @classmethod
    def load(cls, path: str | Path = "config/player_data.yaml") -> PlayerDataConfig:
        return cls.model_validate(_load_yaml(path))


class SquadDataConfig(BaseModel):
    """Official 2026 squad membership and reviewed supplementary inputs."""

    enabled: bool = False
    squads_path_env: str = "FIFA_2026_SQUADS_PATH"
    squads_default_path: str = "Datasets/world_cup_2026_squads.csv"
    official_extract_path_env: str = "FIFA_2026_SQUAD_EXTRACT_PATH"
    official_extract_default_path: str = (
        "Datasets/world_cup_2026_official_squad_extract.csv"
    )
    link_candidates_path_env: str = "FIFA_2026_PLAYER_LINK_CANDIDATES_PATH"
    link_candidates_default_path: str = (
        "Datasets/world_cup_2026_player_link_candidates.csv"
    )
    availability_path_env: str = "FIFA_2026_AVAILABILITY_PATH"
    player_aliases_path_env: str = "FIFA_2026_PLAYER_ALIASES_PATH"
    player_aliases_default_path: str = (
        "data/reference/world_cup_2026_player_aliases.csv"
    )
    allowed_statuses: list[str] = Field(
        default_factory=lambda: ["selected", "reserve", "replacement", "withdrawn"]
    )
    position_groups: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "goalkeeper": ["goalkeeper", "keeper"],
            "defender": ["defender", "defence", "back"],
            "midfielder": ["midfielder", "midfield"],
            "forward": ["forward", "attack", "striker", "winger"],
        }
    )
    activity_windows_days: list[int] = Field(
        default_factory=lambda: [30, 90, 180, 365]
    )
    expected_rows: int | None = None
    expected_teams: int | None = None
    expected_players_per_team: int | None = None
    expected_alias_rows: int | None = None
    license: str = "Official FIFA/federation squad publications; per-source terms"
    attribution: str = "Official FIFA World Cup 2026 squad source"

    @classmethod
    def load(cls, path: str | Path = "config/squads.yaml") -> SquadDataConfig:
        return cls.model_validate(_load_yaml(path))


# Backwards-compatible alias (older code referenced PlayerFeaturesConfig).
class PlayerFeaturesConfig(BaseModel):
    enabled: bool = False
    path_env: str = "PLAYER_DATA_PATH"
    position_groups: list[str] = Field(
        default_factory=lambda: ["goalkeeper", "defence", "midfield", "attack", "bench"]
    )
    recent_minutes_window_days: int = 90
    strength_weights: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = "config/player_features.yaml") -> PlayerFeaturesConfig:
        return cls.model_validate(_load_yaml(path))


class EnrichmentConfig(BaseModel):
    """Top-level switch and feature-family registry for ablation studies."""

    enabled: bool = False
    feature_families: list[str] = Field(default_factory=list)
    missingness_indicators: list[str] = Field(default_factory=list)
    feature_set_version: int = 1

    @classmethod
    def load(cls, path: str | Path = "config/enrichment.yaml") -> EnrichmentConfig:
        return cls.model_validate(_load_yaml(path))
