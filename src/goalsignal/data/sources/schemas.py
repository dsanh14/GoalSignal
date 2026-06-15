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


class SquadMembershipRecord(BaseModel):
    snapshot_date: date
    group: str
    national_team: str
    player_name: str
    date_of_birth: date | None = None
    position: str
    club: str | None = None
    shirt_number: int | None = Field(default=None, ge=1)
    squad_status: str
    source_name: str
    source_url_or_reference: str | None = None
    source_publication_date: date
    source_player_id: str | None = None
    notes: str | None = None
    provenance: ProvenanceEnvelope

    @model_validator(mode="after")
    def _published_by_snapshot(self) -> SquadMembershipRecord:
        if self.source_publication_date > self.snapshot_date:
            raise ValueError("source publication date is after the squad snapshot date")
        if self.provenance.available_at.date() < self.source_publication_date:
            raise ValueError("provenance available_at precedes source publication")
        return self


class PlayerIdentityMappingRecord(BaseModel):
    snapshot_date: date
    national_team: str
    squad_player_name: str
    canonical_player_id: str
    source: str
    source_player_id: str
    match_method: str
    review_status: str
    effective_at: datetime
    published_at: datetime
    retrieved_at: datetime
    notes: str | None = None


class PlayerMatchActivityRecord(BaseModel):
    canonical_player_id: str
    source_player_id: str
    activity_type: str
    event_time: datetime
    published_at: datetime
    retrieved_at: datetime
    prediction_cutoff: datetime
    club_id: str | None = None
    competition_id: str | None = None
    fixture_id: str | None = None
    minutes: float | None = Field(default=None, ge=0)
    started: bool | None = None
    goals: float | None = Field(default=None, ge=0)
    assists: float | None = Field(default=None, ge=0)
    yellow_cards: float | None = Field(default=None, ge=0)
    red_cards: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _cutoff_safe(self) -> PlayerMatchActivityRecord:
        if self.event_time >= self.prediction_cutoff:
            raise ValueError("activity event must be strictly before prediction cutoff")
        if self.published_at > self.prediction_cutoff:
            raise ValueError("activity was not published by prediction cutoff")
        return self


class HistoricalValuationRecord(BaseModel):
    canonical_player_id: str
    source_player_id: str
    valuation: float | None = Field(default=None, ge=0)
    currency: str = "EUR"
    valuation_date: date | None = None
    valuation_age_days: int | None = Field(default=None, ge=0)
    prediction_cutoff: datetime
    available: bool
    source_snapshot_id: str

    @model_validator(mode="after")
    def _valuation_consistency(self) -> HistoricalValuationRecord:
        if self.available != (self.valuation is not None):
            raise ValueError("valuation availability and value disagree")
        if self.valuation_date and self.valuation_date >= self.prediction_cutoff.date():
            raise ValueError("valuation must be strictly before prediction cutoff")
        return self


class ExpectedLineupInputRecord(BaseModel):
    fixture_id: str
    prediction_cutoff: datetime
    national_team: str
    canonical_player_id: str
    position_group: str
    selected_in_squad: bool
    recent_national_team_start_rate: float | None = None
    recent_national_team_minutes: float | None = None
    recent_club_minutes: float | None = None
    recent_club_starts: float | None = None
    days_since_last_appearance: int | None = None
    historical_valuation: float | None = None
    goalkeeper_continuity: float | None = None
    source_availability_flags: dict[str, bool] = Field(default_factory=dict)
    candidate_starter_probability: float | None = None
    provenance: list[ProvenanceEnvelope] = Field(default_factory=list)

    @model_validator(mode="after")
    def _estimator_not_fitted(self) -> ExpectedLineupInputRecord:
        if self.candidate_starter_probability is not None:
            raise ValueError("starter probability must remain missing until an estimator is fit")
        return self


class PathDifficultyRecord(BaseModel):
    team: str
    simulated_group_finish: int = Field(ge=1, le=4)
    round: str
    opponent: str
    opponent_elo: float | None = None
    opponent_fifa_rank: int | None = Field(default=None, ge=1)
    opponent_squad_proxy_available: bool
    matchup_probability: float = Field(ge=0, le=1)
    conditional_advancement_probability: float = Field(ge=0, le=1)
    expected_opponent_strength: float | None = None
    probability_top_5_opponent: float | None = Field(default=None, ge=0, le=1)
    probability_top_10_opponent: float | None = Field(default=None, ge=0, le=1)
    simulation_version: str


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
