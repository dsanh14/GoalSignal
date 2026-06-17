"""Transparent 2026 squad-strength scenario challenger.

This module does not fit outcomes. It converts cutoff-safe squad source
aggregates into bounded, coverage-shrunk sensitivity adjustments around the
unchanged team-level champion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from goalsignal.models.poisson import outcome_probs
from goalsignal.utils.hashing import sha256_file, sha256_json
from goalsignal.utils.paths import resolve


@dataclass
class SquadChallengerConfig:
    raw: dict
    config_hash: str

    @classmethod
    def load(
        cls, path: str | Path = "config/squad_challenger_2026.yaml"
    ) -> SquadChallengerConfig:
        raw = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
        return cls(raw=raw, config_hash=sha256_json(raw))

    def __getattr__(self, name):
        if name in self.raw:
            return self.raw[name]
        raise AttributeError(name)


def _safe_sum(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.sum(min_count=1)) if numeric.notna().any() else np.nan


def _safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    return float(numeric.median()) if numeric.notna().any() else np.nan


def _top_sum(values: pd.Series, count: int) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna().nlargest(count)
    return float(numeric.sum()) if len(numeric) else np.nan


def _zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    median = values.median()
    filled = values.fillna(median)
    std = float(filled.std(ddof=0))
    return (filled - float(filled.mean())) / std if std > 0 else filled * 0.0


def build_team_squad_features(
    activity: pd.DataFrame,
    valuations: pd.DataFrame,
    config: SquadChallengerConfig,
    *,
    source_hashes: dict[str, str],
) -> pd.DataFrame:
    """Build one deterministic live feature row per official team."""
    rows = []
    stale_days = int(config.coverage["stale_valuation_days"])
    for team, block in activity.groupby("national_team", sort=True):
        values = valuations[valuations["national_team"].eq(team)].copy()
        positions = {}
        for position, pblock in block.groupby("position"):
            positions[position.casefold()] = {
                "local": float(pblock["local_snapshot_available"].fillna(False).mean()),
                "minutes_90d": _safe_sum(pblock["minutes_90d"]),
                "active_90d": float(pblock["minutes_90d"].notna().mean()),
            }
        local = block["local_snapshot_available"].fillna(False)
        identified = block["canonical_player_id"].fillna("").ne("")
        valued = values["historical_valuation"].notna()
        valuation_age = pd.to_numeric(values["valuation_age_days"], errors="coerce")
        minutes_90 = pd.to_numeric(block["minutes_90d"], errors="coerce")
        valuation_numeric = pd.to_numeric(
            values["historical_valuation"], errors="coerce"
        )
        weighted = block[
            ["player_name", "minutes_90d"]
        ].merge(
            values[["player_name", "historical_valuation"]],
            on="player_name",
            how="left",
        )
        weighted_minutes = pd.to_numeric(weighted["minutes_90d"], errors="coerce")
        weighted_values = pd.to_numeric(
            weighted["historical_valuation"], errors="coerce"
        )
        valid_weight = weighted_minutes.notna() & weighted_values.notna()
        minutes_weighted_value = (
            float(
                (weighted_minutes[valid_weight] * weighted_values[valid_weight]).sum()
                / weighted_minutes[valid_weight].sum()
            )
            if valid_weight.any() and weighted_minutes[valid_weight].sum() > 0
            else np.nan
        )
        sorted_minutes = minutes_90.dropna().sort_values(ascending=False)
        position_local = [
            info["local"] for info in positions.values()
        ]
        goalkeeper = positions.get("goalkeeper", {})
        row = {
            "national_team": team,
            "prediction_cutoff": config.prediction_cutoff,
            "feature_version": config.feature_version,
            "feature_config_hash": config.config_hash,
            "source_hashes": json.dumps(source_hashes, sort_keys=True),
            "identity_coverage": float(identified.mean()),
            "local_activity_coverage": float(local.mean()),
            "valuation_coverage": float(valued.mean()),
            "web_only_identity_count": int(
                block["identity_status"].eq("accepted_web_only").sum()
            ),
            "conflict_excluded_count": int(
                block["identity_status"].eq("conflict").sum()
            ),
            "goalkeeper_local_coverage": goalkeeper.get("local", 0.0),
            "minimum_position_local_coverage": min(position_local)
            if position_local
            else 0.0,
            "position_missingness": 1.0 - min(position_local)
            if position_local
            else 1.0,
            "valuation_median_age_days": _safe_median(valuation_age),
            "stale_valuation_proportion": float(
                valuation_age.gt(stale_days).mean()
            ),
            "valuation_total": _safe_sum(valuation_numeric),
            "valuation_median": _safe_median(valuation_numeric),
            "valuation_minutes_weighted": minutes_weighted_value,
            "valuation_top_11": _top_sum(valuation_numeric, 11),
            "valuation_top_15": _top_sum(valuation_numeric, 15),
            "valuation_top_23": _top_sum(valuation_numeric, 23),
            "days_since_last_appearance_median": _safe_median(
                block["days_since_last_appearance"]
            ),
            "days_since_last_appearance_max": pd.to_numeric(
                block["days_since_last_appearance"], errors="coerce"
            ).max(),
            "recently_active_90d": int(minutes_90.notna().sum()),
            "inactive_90d": int(minutes_90.isna().sum()),
            "top_11_minutes_90d": float(sorted_minutes.head(11).sum()),
            "top_15_minutes_90d": float(sorted_minutes.head(15).sum()),
            "top_23_minutes_90d": float(sorted_minutes.head(23).sum()),
            "next_4_minutes_90d": float(sorted_minutes.iloc[11:15].sum()),
            "remaining_after_15_minutes_90d": float(
                sorted_minutes.iloc[15:].sum()
            ),
            "goalkeeper_minutes_90d": goalkeeper.get("minutes_90d", np.nan),
            "goalkeeper_active_90d": goalkeeper.get("active_90d", 0.0),
        }
        for days in (30, 90, 180, 365):
            minutes = pd.to_numeric(block[f"minutes_{days}d"], errors="coerce")
            starts = pd.to_numeric(block[f"starts_{days}d"], errors="coerce")
            row[f"minutes_{days}d_coverage"] = float(minutes.notna().mean())
            row[f"minutes_{days}d_total"] = _safe_sum(minutes)
            row[f"minutes_{days}d_median"] = _safe_median(minutes)
            row[f"starts_{days}d_total"] = _safe_sum(starts)
        for label in ("defender", "midfielder", "forward"):
            info = positions.get(label, {})
            row[f"{label}_minutes_90d"] = info.get("minutes_90d", np.nan)
            row[f"{label}_active_90d"] = info.get("active_90d", 0.0)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("national_team").reset_index(drop=True)
    return score_team_features(frame, config)


def score_team_features(
    frame: pd.DataFrame, config: SquadChallengerConfig
) -> pd.DataFrame:
    """Add scenario scores, coverage confidence, eligibility, and S7 adjustment."""
    out = frame.copy()
    component_columns = {
        "activity": [
            "minutes_30d_coverage",
            "minutes_90d_coverage",
            "minutes_180d_coverage",
            "minutes_90d_total",
            "recently_active_90d",
        ],
        "starts": ["starts_30d_total", "starts_90d_total", "starts_180d_total"],
        "valuation": [
            "valuation_total",
            "valuation_median",
            "valuation_minutes_weighted",
            "valuation_top_15",
        ],
        "positional": [
            "defender_minutes_90d",
            "midfielder_minutes_90d",
            "forward_minutes_90d",
            "minimum_position_local_coverage",
        ],
        "goalkeeper": [
            "goalkeeper_minutes_90d",
            "goalkeeper_active_90d",
            "goalkeeper_local_coverage",
        ],
        "depth": [
            "top_11_minutes_90d",
            "top_15_minutes_90d",
            "top_23_minutes_90d",
            "next_4_minutes_90d",
        ],
    }
    for component, columns in component_columns.items():
        zcols = []
        for column in columns:
            zcolumn = f"z_{column}"
            out[zcolumn] = _zscore(out, column)
            zcols.append(zcolumn)
        out[f"score_{component}"] = out[zcols].mean(axis=1)
    for column in (
        "inactive_90d",
        "local_activity_coverage",
        "valuation_coverage",
        "valuation_top_11",
        "valuation_top_15",
        "valuation_top_23",
        "defender_active_90d",
        "midfielder_active_90d",
        "forward_active_90d",
    ):
        out[f"z_{column}"] = _zscore(out, column)
    out["score_s1_activity"] = out["score_activity"]
    out["score_s2_starts"] = out["score_starts"]
    out["score_s3_valuation"] = out["score_valuation"]
    out["score_s4_positional"] = out[
        ["score_positional", "score_goalkeeper"]
    ].mean(axis=1)
    out["score_s5_depth"] = out["score_depth"]
    weights = config.adjustment["weights"]
    out["score_s6_full"] = sum(
        out[f"score_{name}"] * float(weight)
        for name, weight in weights.items()
    ) / sum(float(value) for value in weights.values())
    thresholds = config.coverage
    out["maximum_missingness"] = 1.0 - out[
        [
            "identity_coverage",
            "local_activity_coverage",
            "valuation_coverage",
            "goalkeeper_local_coverage",
            "minimum_position_local_coverage",
        ]
    ].min(axis=1)
    out["coverage_eligible"] = (
        out["identity_coverage"].ge(thresholds["minimum_identity_coverage"])
        & out["local_activity_coverage"].ge(
            thresholds["minimum_local_activity_coverage"]
        )
        & out["valuation_coverage"].ge(
            thresholds["minimum_valuation_coverage"]
        )
        & out["goalkeeper_local_coverage"].ge(
            thresholds["minimum_goalkeeper_local_coverage"]
        )
        & out["minimum_position_local_coverage"].ge(
            thresholds["minimum_position_local_coverage"]
        )
        & out["maximum_missingness"].le(thresholds["maximum_missingness"])
    )
    ratios = pd.DataFrame(
        {
            "identity": out["identity_coverage"]
            / thresholds["minimum_identity_coverage"],
            "local": out["local_activity_coverage"]
            / thresholds["minimum_local_activity_coverage"],
            "valuation": out["valuation_coverage"]
            / thresholds["minimum_valuation_coverage"],
            "goalkeeper": out["goalkeeper_local_coverage"]
            / thresholds["minimum_goalkeeper_local_coverage"],
            "position": out["minimum_position_local_coverage"]
            / thresholds["minimum_position_local_coverage"],
            "freshness": 1.0 - out["stale_valuation_proportion"].clip(0, 1),
        }
    ).clip(0, 1)
    out["coverage_confidence"] = ratios.mean(axis=1)
    out.loc[~out["coverage_eligible"], "coverage_confidence"] = 0.0
    clip = float(config.adjustment["score_clip"])
    out["score_s7_coverage_shrunk"] = (
        out["score_s6_full"].clip(-clip, clip) * out["coverage_confidence"]
    )
    max_adjustment = float(config.adjustment["maximum_log_goal_adjustment"])
    strength = float(config.adjustment["final_strength"])
    out["log_goal_adjustment"] = (
        out["score_s7_coverage_shrunk"] * strength
    ).clip(-max_adjustment, max_adjustment)
    out["fallback_used"] = ~out["coverage_eligible"]
    return out


class SquadScenarioAdapter:
    """Bounded expected-goal wrapper around the unchanged champion adapter."""

    def __init__(self, base_adapter, features: pd.DataFrame):
        self.base_adapter = base_adapter
        indexed = features.set_index("national_team")
        self.adjustments = indexed["log_goal_adjustment"].to_dict()
        self.eligible = indexed["coverage_eligible"].to_dict()
        self.unrated_teams = base_adapter.unrated_teams

    def team_adjustment(self, team: str) -> float:
        return (
            float(self.adjustments.get(team, 0.0))
            if self.eligible.get(team, False)
            else 0.0
        )

    def expected_goals(self, home: str, away: str, neutral: bool):
        lam_home, lam_away = self.base_adapter.expected_goals(home, away, neutral)
        difference = self.team_adjustment(home) - self.team_adjustment(away)
        return (
            float(lam_home * np.exp(difference / 2.0)),
            float(lam_away * np.exp(-difference / 2.0)),
        )

    def score_matrix(self, lam_home: float, lam_away: float):
        return self.base_adapter.score_matrix(lam_home, lam_away)

    def outcome_probabilities(self, home: str, away: str, neutral: bool):
        lam_home, lam_away = self.expected_goals(home, away, neutral)
        return outcome_probs(self.score_matrix(lam_home, lam_away))


def base_outcome_probabilities(adapter, home: str, away: str, neutral: bool):
    lam_home, lam_away = adapter.expected_goals(home, away, neutral)
    return outcome_probs(adapter.score_matrix(lam_home, lam_away))


def feature_artifact_version(
    config: SquadChallengerConfig, source_hashes: dict[str, str]
) -> str:
    return sha256_json(
        {
            "feature_version": config.feature_version,
            "config_hash": config.config_hash,
            "source_hashes": source_hashes,
        }
    )[:16]


def squad_source_hashes() -> dict[str, str]:
    paths = {
        "squad": resolve("Datasets/world_cup_2026_squads.csv"),
        "aliases": resolve("data/reference/world_cup_2026_player_aliases.csv"),
    }
    return {name: sha256_file(path) for name, path in paths.items()}
