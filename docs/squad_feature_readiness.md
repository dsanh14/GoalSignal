# Squad Feature Readiness

This milestone builds source contracts and audits only. It does not train a
squad-aware model, estimate an XI, or alter a forecast.

## Current Classification

- Recent club minutes, starts, goals/assists, historical valuations, positional
  depth, goalkeeper activity, and bench-depth proxies are currently
  **blocked by identity coverage** at a 75.0% deterministic link rate.
- Player age is **ready** where the official source supplies DOB.
- Club and competition strength: **restricted subset**. Dated appearances
  and competitions are usable; current total market values and season-final
  outcomes are not.
- Expected-XI strength, lineup continuity, and goalkeeper continuity:
  **blocked by sparse international lineups**.
- Confirmed World Cup lineups: **blocked by provider plan** on API-Football
  Free.
- Injuries and suspensions: **unsupported** for this competition/source setup.
- Path difficulty: **ready** from existing probabilistic matchup
  distributions, but is not deployed as a prediction feature.

The generated source of truth is
`artifacts/reports/squad_feature_readiness.{json,md}`.

The dated extraction produced 936 activity rows. At 90 days, 549 have
non-missing minute totals. Historical valuation coverage is 705/936 (75.3%),
with a median age of 188 days and 269 values older than 365 days.
National-team lineup coverage is partial for 22 teams, sparse for 4, and
unavailable for 22. Portugal has 0/26 deterministic links under the current
evidence, so no Portugal squad-strength or expected-XI claim is supported.

## Expected-Lineup Contract

`ExpectedLineupInputRecord` separates selected-in-squad, recent national-team
history, recent club activity, historical valuation, goalkeeper continuity,
and source-availability flags. `candidate_starter_probability` must remain
missing until a chronologically evaluated estimator is trained.

## Path-Difficulty Contract

`PathDifficultyRecord` supports opponent Elo, valid FIFA rank, squad-proxy
availability, matchup probability, conditional advancement, expected opponent
strength, and top-5/top-10 encounter probabilities. It supports future
conditional analysis for Portugal without hard-coding Croatia or Spain.
