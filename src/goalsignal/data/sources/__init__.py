"""Enrichment source adapters (Milestone A: contracts and schemas only).

This package defines the contracts for optional enrichment sources (StatsBomb
open data, the football-data.org API, historical FIFA rankings, player/club
reference data). Milestone A ships protocols, provenance/manifest models,
normalized record schemas, and offline-testable adapter scaffolding. Network
fetching, real ingestion, feature engineering, and model training are
deferred to later milestones and raise `MilestoneNotImplementedError` until
then.

Nothing in this package is imported by the core forecasting pipeline, so the
base CPU workflow keeps running without any enrichment dependency installed.
"""

from goalsignal.data.sources.base import (
    AvailabilityStatus,
    EventSourceAdapter,
    FeatureAvailabilityError,
    FixtureSourceAdapter,
    ForecastStage,
    LineupStatus,
    MilestoneNotImplementedError,
    PlayerSourceAdapter,
    ProvenanceEnvelope,
    RankingSourceAdapter,
    SourceAdapter,
    SourceValidationError,
    assert_available_before,
    require_optional_dependency,
)
from goalsignal.data.sources.manifests import (
    SourceSnapshotManifest,
    build_snapshot_manifest,
    compute_snapshot_id,
)

__all__ = [
    "AvailabilityStatus",
    "EventSourceAdapter",
    "FeatureAvailabilityError",
    "FixtureSourceAdapter",
    "ForecastStage",
    "LineupStatus",
    "MilestoneNotImplementedError",
    "PlayerSourceAdapter",
    "ProvenanceEnvelope",
    "RankingSourceAdapter",
    "SourceAdapter",
    "SourceSnapshotManifest",
    "SourceValidationError",
    "assert_available_before",
    "build_snapshot_manifest",
    "compute_snapshot_id",
    "require_optional_dependency",
]
