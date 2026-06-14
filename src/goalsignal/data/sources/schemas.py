"""Normalized record schemas for enrichment sources.

These pydantic models define the *normalized* shape each adapter must produce.
They are validation contracts only (Milestone A): no source is ingested here.
Each record embeds a `ProvenanceEnvelope`, and lineup/availability records
keep expected and confirmed states strictly separate.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, model_validator

from goalsignal.data.sources.base import (
    AvailabilityStatus,
    LineupStatus,
    ProvenanceEnvelope,
)


class FifaRankingRecord(BaseModel):
    """One team's FIFA ranking snapshot at a release date."""

    team: str
    rank: int = Field(ge=1)
    points: float = Field(ge=0)
    ranking_release_date: date
    confederation: str | None = None
    provenance: ProvenanceEnvelope

    @model_validator(mode="after")
    def _release_not_after_available(self) -> FifaRankingRecord:
        # The ranking cannot be "available" before it is released.
        if self.provenance.available_at.date() < self.ranking_release_date:
            raise ValueError(
                "available_at precedes the ranking_release_date; a ranking is not "
                "knowable before it is published"
            )
        return self


class StatsBombMatchRef(BaseModel):
    """A StatsBomb match identity used for linking to canonical fixtures."""

    statsbomb_match_id: int
    match_date: date
    home_team: str
    away_team: str
    competition: str
    season: str | None = None
    provenance: ProvenanceEnvelope


class PlayerIdentity(BaseModel):
    """Canonical player entity. Never merged on name alone (rule: identity)."""

    canonical_player_id: str
    full_name: str
    normalized_name: str
    date_of_birth: date | None = None
    nationality: str | None = None
    position: str | None = None
    club: str | None = None
    effective_start: date | None = None
    effective_end: date | None = None
    source_player_ids: dict[str, str] = Field(default_factory=dict)
    review_status: str = "pending"


class ClubIdentity(BaseModel):
    """Canonical club entity with source aliases and effective dates."""

    canonical_club_id: str
    name: str
    normalized_name: str
    country: str | None = None
    effective_start: date | None = None
    effective_end: date | None = None
    source_club_ids: dict[str, str] = Field(default_factory=dict)
    review_status: str = "pending"


class LineupRecord(BaseModel):
    """A lineup snapshot. Expected and confirmed never share a record."""

    source_fixture_id: str
    lineup_status: LineupStatus
    formation: str | None = None
    goalkeeper: str | None = None
    captain: str | None = None
    starting_xi: list[str] = Field(default_factory=list)
    bench: list[str] = Field(default_factory=list)
    unavailable_players: list[str] = Field(default_factory=list)
    announced_at: datetime | None = None
    provenance: ProvenanceEnvelope

    @model_validator(mode="after")
    def _confirmed_requires_xi(self) -> LineupRecord:
        if self.lineup_status == LineupStatus.CONFIRMED and len(self.starting_xi) != 11:
            raise ValueError("a confirmed lineup must list exactly 11 starters")
        return self


class PlayerAvailabilityRecord(BaseModel):
    """Player availability with normalized category and raw source wording."""

    canonical_player_id: str | None = None
    source_player_id: str
    status: AvailabilityStatus
    raw_status_text: str
    return_date: date | None = None
    evidence_strength: str | None = None
    provenance: ProvenanceEnvelope


class FeatureRecord(BaseModel):
    """One timestamp-aware feature value for the feature store.

    `feature_available_at` is checked against the prediction timestamp; a
    feature with a later availability is rejected. `missing` distinguishes a
    genuinely absent value from a real zero — missing is never zero-filled.
    """

    fixture_id: str
    prediction_timestamp: datetime
    feature_name: str
    feature_value: float | None
    feature_source: str
    feature_available_at: datetime | None
    feature_version: int = 1
    source_snapshot_hash: str | None = None
    missing: bool = False

    @model_validator(mode="after")
    def _missing_has_no_value(self) -> FeatureRecord:
        if self.missing and self.feature_value is not None:
            raise ValueError("a missing feature must not carry a value")
        if not self.missing and self.feature_value is None:
            raise ValueError(
                "a non-missing feature must carry a value; set missing=True instead "
                "of zero-filling absent data"
            )
        return self
