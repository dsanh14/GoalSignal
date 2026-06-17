# Leakage Prevention

The June 11, 2026 current FIFA snapshot is unavailable before its release date
and remains outside the historical timeline. Completed results alter only
future Elo state and future prediction revisions. Frozen forecasts are scored
without modification, and simulations fix observed scores instead of sampling
completed fixtures.

For a prediction generated at time T, every input must have been available
before T. This document records the project-wide rules; each modeling
milestone must add its own tests against them.

## Rules

1. No target-match information (result, goals, post-match ratings) in any
   feature for that match.
2. No future matches in rolling statistics, normalizations, or scalers:
   transformers fit on training folds only.
3. Chronological evaluation only — expanding-window, rolling-origin, year and
   tournament holdouts. Random splits are never the primary evaluation.
4. Hyperparameters are never tuned on final test periods; calibrators fit on
   validation-period predictions only.
5. Team-name normalization is date-aware: a mapping applies only within the
   period the former name was in use (`normalize_teams.py`), so future
   renames never relabel historical matches.
6. `shootouts.csv` `first_shooter` is post-kickoff information and must never
   be a pre-match feature.
7. Scheduled fixtures (`status = scheduled`, NA scores) carry no outcome
   information; they exist for forecasting only and are excluded from all
   training (`strict_exclusion_reason = not_played`).
8. Goalscorer-derived features may use only events dated before the
   prediction cutoff.
9. Predictions are immutable and timestamped with their data cutoff and
   dataset version, so any retroactive contamination is detectable.

## Required tests (added per modeling milestone)

- Adding, modifying, or removing a future match leaves historical features
  and predictions byte-identical.
- The target match does not update its own pre-match rating.
- Calibrators trained on validation predictions only.
- Results affect only forecasts generated after the result timestamp.

## Status

- Data layer enforces rules 5–7 structurally (unit-tested).
- Elo: `test_pre_match_rating_excludes_target_match` and
  `test_future_match_does_not_change_history` verify the target match never
  updates its own pre-match rating and that future matches leave the
  historical timeline byte-identical.
- Backtest protocol (`evaluation/backtest.py`): components fit on train
  only; temperature calibrators and ensemble weights fit on validation
  predictions only; identical protocol used by the live pipeline
  (`live.py`).
- Ledger: predictions are hash-chained and append-only; retroactive edits
  fail `goalsignal ledger verify` (tamper tests in `tests/unit/test_ledger.py`).

## Squad And Player Data

- Official squad rows cannot be used before `source_publication_date`.
- Player activity windows require `event_date < prediction_cutoff`; the target
  game is explicitly excludable.
- Historical valuations use the latest dated observation strictly before the
  cutoff. Missing valuation is never zero.
- Current club, caps, current valuation, club total value, current FIFA rank,
  and season-final outcomes are prohibited historical inputs.
- Club and national-team lineup histories remain separate.
- Expected starter probabilities remain missing until a fitted estimator is
  evaluated chronologically.
- The 2026 squad scenario is not trained. It uses only activity and historical
  valuations before the active cutoff, never uses current World Cup outcomes
  as labels, and falls back to the champion below coverage thresholds.
- Completed World Cup fixtures are fixed from the result store and excluded
  from research prediction rows and Monte Carlo score sampling.

## D1 feature engineering (Milestone D1)

The D1 feature builder (`features/d1.py`) enforces leakage safety structurally:
- Rolling form/attack/defense/rest use only a team's matches **strictly earlier**
  in `(date, source_row)` order — the target match is never in its own features
  (tested: `test_native_features_exclude_target_and_windows`).
- FIFA features use the latest release **strictly before** the fixture date; a
  future release is never selected (tested: `test_fifa_as_of_no_future_release`).
- **No 2024→2026 forward-fill:** availability is capped (450 days) so every 2026
  fixture is FIFA-unavailable; `goalsignal features validate-d1` enforces this.
- Opponent-adjusted goal residuals use a FIXED Elo-based expectation (no fitting),
  so they introduce no fold dependence and no circularity with any challenger.
- Imputation/standardization are fit on the **training fold only**
  (`_FoldPreprocessor`; tested: `test_fold_preprocessor_fits_on_train_only`).
- Ablations use identical folds and identical paired test matches.
