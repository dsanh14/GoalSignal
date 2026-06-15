# Feature Availability, Early vs Final Forecasts, Missingness

This is the temporal-safety contract for the enrichment layer. It extends
`docs/leakage_prevention.md` to externally sourced features.

## Availability timestamps

Every enrichment record carries a `ProvenanceEnvelope` with two distinct times:

- `retrieved_at` — when GoalSignal fetched the data.
- `available_at` — when the information became *knowable* to any observer (a
  ranking's release datetime, a lineup's announcement, a result's completion).

Leakage checks use **`available_at`, never `retrieved_at`**. Any feature whose
`available_at` is strictly after the prediction timestamp is rejected by
`assert_available_before(...)`. Equality is allowed.

The feature store (Milestone D) records per row: `fixture_id`,
`prediction_timestamp`, `feature_name`, `feature_value`, `feature_source`,
`feature_available_at`, `feature_version`, `source_snapshot_hash`, and a
`missing` flag. Rows with `feature_available_at > prediction_timestamp` are
quarantined.

## Early vs final forecasts

Two forecast modes, stored as **separate immutable predictions** — a final
forecast never overwrites an early one:

| | Early | Final |
| --- | --- | --- |
| Timing | Before confirmed lineups | After confirmed lineups |
| Lineup input | Expected lineup only | Confirmed lineup |
| `forecast_stage` | `early` | `final` |
| `lineup_status` | `projected`/`expected` | `confirmed` |

Both record `prediction_timestamp`, `player_data_cutoff`, `injury_data_cutoff`,
`lineup_retrieved_at`, source snapshot hashes, `feature_set_version`, and model
version. An early forecast **cannot** read confirmed lineups; a final forecast
**preserves** the earlier early forecast.

Expected and confirmed lineup features are kept in separate records
(`LineupStatus`), never collapsed into one field. Expected-lineup estimation
uses only information available before the early forecast time; its method is
documented where implemented (Milestone D).

## Player availability

Availability is normalized to `available` / `doubtful` / `unavailable` /
`suspended` / `unknown`, with the **raw source wording stored separately**. It
is never reduced to a single Boolean. An injury or suspension report published
after kickoff is never used.

## Missingness policy

Missing enrichment is flagged explicitly and **never** encoded as zero. The
schema enforces this: a `FeatureRecord` with `missing=True` must carry no value,
and a non-missing record must carry a value (no silent zero-fill). Coverage
indicators (`statsbomb_available`, `lineup_available`, `injury_feed_available`,
`fifa_ranking_available`, `player_strength_available`) let models degrade
gracefully, and imputers are fit on training folds only.

This matters for fairness: smaller national teams tend to have less enrichment
coverage, so missing data must not become an implicit penalty. The coverage-bias
analysis (Milestone F) checks exactly this.

## Status

Milestone A: enums (`LineupStatus`, `AvailabilityStatus`, `ForecastStage`),
`ProvenanceEnvelope`, `assert_available_before`, and the `FeatureRecord`/
`LineupRecord`/`PlayerAvailabilityRecord` schemas with their invariants. The
feature store and the early/final forecast paths (ledger v2) are Milestones D–E.

## D1 availability (Milestone D1)

Every D1 feature carries an explicit availability/missingness indicator
(`fifa_available`, `fifa_stale`, `home/away_form_available`, `*_long_inactivity`,
`venue_known`). Missing enrichment is never silently zeroed; continuous columns
are median-imputed fold-locally with the indicator preserved. FIFA coverage ends
2024-09-19 → 2026 fixtures are FIFA-unavailable and fall back to native features.
See `docs/enrichment_features.md`.

## Squad And Player Inputs

Squad membership is available from its official source publication timestamp,
not from the retrieval timestamp or tournament start. A squad snapshot
published after a prediction cutoff is rejected.

Player activity and valuations use rows strictly before the cutoff. Expected-XI
inputs retain explicit source-availability flags and leave starter probability
missing until an estimator has passed chronological evaluation. Availability
records remain separate from membership.
