"""Adapter protocols, provenance envelope, and temporal-safety helpers.

These are the contracts every enrichment source must satisfy. They encode the
non-negotiable rules at the type level: every external record carries full
provenance (`ProvenanceEnvelope`), and any value whose information became
available after the prediction timestamp is rejected
(`assert_available_before`) rather than silently used.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, Field


# --- exceptions -------------------------------------------------------------
class SourceValidationError(ValueError):
    """A source response or record failed schema validation."""


class FeatureAvailabilityError(ValueError):
    """A feature's information was not available before the prediction time."""


class MilestoneNotImplementedError(NotImplementedError):
    """Behavior intentionally deferred to a later enrichment milestone.

    Milestone A defines contracts only; ingestion (B), entity linking (C),
    feature engineering (D), and forecasting (E+) raise this until built, so
    no half-wired ingestion can run by accident.
    """


# --- enumerations -----------------------------------------------------------
class LineupStatus(enum.StrEnum):
    """Lifecycle of lineup knowledge for a fixture. Never collapse these."""

    UNAVAILABLE = "unavailable"
    PROJECTED = "projected"
    EXPECTED = "expected"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"


class AvailabilityStatus(enum.StrEnum):
    """Normalized player availability. Source wording is stored separately."""

    AVAILABLE = "available"
    DOUBTFUL = "doubtful"
    UNAVAILABLE = "unavailable"
    SUSPENDED = "suspended"
    UNKNOWN = "unknown"


class ForecastStage(enum.StrEnum):
    """Forecast mode. Early and final are separate immutable predictions."""

    EARLY = "early"
    FINAL = "final"


# --- provenance -------------------------------------------------------------
class ProvenanceEnvelope(BaseModel):
    """Mandatory provenance for every externally sourced field (rule 3).

    `available_at` is the moment the information became knowable to an
    observer (e.g. a ranking's release datetime, a lineup's announcement),
    NOT the moment we fetched it (`retrieved_at`). Leakage checks use
    `available_at`.
    """

    source: str
    source_record_id: str
    retrieved_at: datetime
    available_at: datetime
    source_snapshot_hash: str
    schema_version: int = 1

    model_config = {"frozen": True}


def assert_available_before(
    available_at: datetime, prediction_timestamp: datetime, *, label: str = "feature"
) -> None:
    """Reject information that post-dates the prediction (rules 4, 17-19).

    Raises FeatureAvailabilityError when `available_at` is strictly after
    `prediction_timestamp`. Equality is allowed (information available exactly
    at prediction time is admissible).
    """
    if available_at > prediction_timestamp:
        raise FeatureAvailabilityError(
            f"{label}: available_at {available_at.isoformat()} is after prediction "
            f"timestamp {prediction_timestamp.isoformat()}; refusing to use future "
            "information"
        )


def require_optional_dependency(module: str, extra: str) -> object:
    """Import an optional dependency or raise an actionable error.

    Keeps heavy/source-specific dependencies optional (rule 13): the base
    workflow never imports these, and enrichment paths fail with install
    guidance instead of a bare ImportError.
    """
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise MilestoneNotImplementedError(
            f"optional dependency '{module}' is required for this enrichment path; "
            f"install it with `uv sync --extra {extra}` (it is intentionally not "
            "part of the base install)"
        ) from exc


# --- coverage report --------------------------------------------------------
class CoverageReport(BaseModel):
    """Per-source coverage summary (never interpreted as a team signal)."""

    source: str
    rows: int
    coverage_period_start: str | None = None
    coverage_period_end: str | None = None
    notes: list[str] = Field(default_factory=list)


# --- adapter protocols ------------------------------------------------------
@runtime_checkable
class SourceAdapter(Protocol):
    """Common contract for every enrichment source.

    Implementations must use timeouts and bounded retries for any network
    access, validate response schemas, compute SHA-256 hashes, avoid silent
    overwrite, record retrieval metadata, fail with actionable errors, and
    support offline loading from cached files.
    """

    name: str
    role: str

    def load(self) -> pd.DataFrame:
        """Load normalized records from the local cache (offline-capable)."""
        ...

    def validate(self, records: list[dict]) -> list[dict]:
        """Validate raw records against this source's schema; return rows."""
        ...

    def build_manifest(self) -> SourceSnapshotManifestProto:
        """Build a deterministic snapshot manifest for the cached data."""
        ...

    def report_coverage(self) -> CoverageReport:
        """Summarize coverage of the cached data."""
        ...


@runtime_checkable
class FixtureSourceAdapter(SourceAdapter, Protocol):
    """Sources that supply fixtures, lineups, results, or standings."""

    def link_fixtures(self, canonical_matches: pd.DataFrame) -> pd.DataFrame:
        """Join source fixtures onto canonical GoalSignal fixtures."""
        ...


@runtime_checkable
class PlayerSourceAdapter(SourceAdapter, Protocol):
    """Sources that supply player, squad, or availability information."""

    def resolve_players(self, aliases: pd.DataFrame) -> pd.DataFrame:
        """Resolve source players to canonical player identities."""
        ...


@runtime_checkable
class RankingSourceAdapter(SourceAdapter, Protocol):
    """Sources that supply team rankings (e.g. FIFA)."""

    def as_of(self, team: str, match_date) -> dict | None:
        """Latest ranking strictly before `match_date` (no future leak)."""
        ...


@runtime_checkable
class EventSourceAdapter(SourceAdapter, Protocol):
    """Sources that supply event-level data (e.g. StatsBomb)."""

    def report_event_coverage(self) -> pd.DataFrame:
        """Per (year, tournament, team) event/lineup/xG coverage."""
        ...


class SourceSnapshotManifestProto(Protocol):
    """Structural type for a snapshot manifest (see manifests.py)."""

    snapshot_id: str
    source: str
    content_hash: str
