# Squad Feature Readiness

This milestone builds source contracts and audits only. It does not train a
squad-aware model, estimate an XI, or alter a forecast.

## Current Classification

- Recent club minutes, starts, goals/assists, historical valuations, positional
  depth, goalkeeper activity, and bench-depth proxies are **ready with cutoff**
  at 98.8% identity coverage and 93.8% local-snapshot coverage.
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

The dated extraction covers all 1,248 squad rows while leaving unavailable
fields missing. There are 488 players with 30-day minutes, 647 with 90-day
minutes, and 1,170 with local start-count availability. Historical valuation
coverage is 838/1,248 (67.1%), or 71.6% among locally linkable players, with a
median age of 188 days and 324 values older than 365 days.
National-team lineup coverage is partial for 22 teams, sparse for 4, and
unavailable for 22. Portugal is 26/26 accepted-local, but its national-team
lineup history remains unavailable, so expected-XI modeling is still blocked.

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
