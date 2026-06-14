"""Schemas and configuration models for the data layer."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from goalsignal.utils.paths import resolve

# Expected source columns. `goalscorers.csv` ships an extra `minute` column
# relative to the project specification; it is accepted and preserved.
RESULTS_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]
SHOOTOUTS_COLUMNS = ["date", "home_team", "away_team", "winner", "first_shooter"]
GOALSCORERS_COLUMNS = ["date", "home_team", "away_team", "team", "scorer", "own_goal", "penalty"]
FORMER_NAMES_COLUMNS = ["current", "former", "start_date", "end_date"]

# recorded_score_scope values
SCOPE_REGULATION = "regulation"
SCOPE_AFTER_ET = "after_extra_time"
SCOPE_UNKNOWN = "regulation_or_extra_time_unknown"


class InputFiles(BaseModel):
    results: str = "results.csv"
    shootouts: str = "shootouts.csv"
    goalscorers: str = "goalscorers.csv"
    former_names: str = "former_names.csv"


class InputConfig(BaseModel):
    directory: str = "Datasets"
    files: InputFiles = Field(default_factory=InputFiles)


class OutputConfig(BaseModel):
    processed_dir: str = "data/processed"
    reports_dir: str = "artifacts/reports"
    manifests_dir: str = "artifacts/manifests"


class ValidationConfig(BaseModel):
    implausible_score_threshold: int = 32
    suspicious_tournament_terms: list[str] = Field(
        default_factory=lambda: ["olympic", "u-23", "u23", "under-23", "youth", "b team"]
    )


class ScoreScopePolicy(BaseModel):
    knockout_capable_tournament_patterns: list[str] = Field(default_factory=list)


class DataConfig(BaseModel):
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    schema_version: int = 1
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    score_scope_policy: ScoreScopePolicy = Field(default_factory=ScoreScopePolicy)

    @classmethod
    def load(cls, path: str | Path = "config/data.yaml") -> DataConfig:
        with open(resolve(path), encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)

    def input_path(self, name: str) -> Path:
        filename = getattr(self.input.files, name)
        return resolve(self.input.directory) / filename
