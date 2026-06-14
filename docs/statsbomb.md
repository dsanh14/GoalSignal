# StatsBomb Open Data

Optional historical event-level enrichment. **Not** a dependency of the base
workflow.

## Access (offline only)

GoalSignal does not download StatsBomb data. The user clones the open-data
repository locally and points `STATSBOMB_DATA_PATH` at it:

```bash
git clone https://github.com/statsbomb/open-data
export STATSBOMB_DATA_PATH=/path/to/open-data
```

The `statsbombpy` package is an optional convenience loader
(`uv sync --extra statsbomb`); the adapter can also read the JSON files
directly. Missing the dependency raises an actionable install message, never a
bare ImportError.

## License and attribution

StatsBomb Open Data is released under the StatsBomb Open Data License:
non-commercial use with **required attribution**. Any published output using
it must credit StatsBomb. See the license in the open-data repository and
respect its terms; GoalSignal stores the attribution string in every source
manifest.

## Coverage caveat (important)

StatsBomb open data covers specific competitions and seasons, and its coverage
of **senior men's international football is sparse and uneven**. The exact set
of available competitions/seasons is enumerated from the local
`competitions.json` in Milestone B — it is deliberately **not** listed here, to
avoid fabricating coverage.

Consequences, treated as first-class concerns:

- Missing StatsBomb coverage for a team is **never** a negative signal about
  that team. It is a data gap, flagged via a `statsbomb_available` missingness
  indicator.
- Because richer-data teams may receive systematically different forecasts, a
  coverage-bias analysis (Milestone F) compares performance for high- vs
  low-coverage teams before any StatsBomb feature is trusted.

## Match linking

StatsBomb matches join onto canonical GoalSignal fixtures by normalized date,
home team, away team, competition, and venue where available — never by row
order. Linking produces, in Milestone C:

- `artifacts/reports/statsbomb_match_links.csv`
- `artifacts/reports/statsbomb_unmatched.csv`
- `artifacts/reports/statsbomb_ambiguous.csv`
- `artifacts/reports/statsbomb_coverage.csv` (year, tournament, team,
  confederation, match/lineup/event/xG availability)

Ambiguous matches are reported, never silently resolved.

## Planned features (Milestone D, leakage-safe windows)

recent xG for/against, non-penalty xG, shots, shots on target, shot quality,
set-piece vs open-play xG, big chances, red-card rate, substitution profile,
starting-XI continuity, formation continuity, goalkeeper identity and a
shot-stopping proxy — each over windows of 3/5/10 matches and 180/365 days,
using only matches strictly before the prediction cutoff.

## Commands (Milestone B)

```bash
export STATSBOMB_DATA_PATH=/path/to/open-data
goalsignal statsbomb inspect    # confirm the data path (or print setup steps)
goalsignal statsbomb ingest     # normalize competitions -> data/external/statsbomb/normalized/
goalsignal statsbomb coverage   # real coverage counts from the local clone
```

`StatsBombLoader` reads the open-data file layout (`competitions.json`,
`matches/<comp>/<season>.json`, `lineups/<match>.json`, `events/<match>.json`),
hashes files for deterministic manifests, normalizes to CSV, and computes
per-match per-team aggregates (xG, non-penalty xG, shots, shots on target,
set-piece vs open-play xG, goals, cards, substitutions) — source-level totals
only, with **no cross-match rolling windows** (those are Milestone D).

## Status

Milestone B: offline loader, normalization, and aggregation implemented and
tested on synthetic data. Currently **`not_configured`** here (no
`STATSBOMB_DATA_PATH` set); set it to a local clone to ingest. Match linking to
canonical fixtures is prepared in `linking.py` and finalized in Milestone C.
