# Historical FIFA Rankings

This historical timeline is distinct from the frozen current World Cup
snapshot in `docs/current_fifa_snapshot.md`. Never append that 48-team snapshot
to the historical series.

Optional external strength baseline and Elo-disagreement signal.

## Two separate files

| Env var | Role | Real file |
| --- | --- | --- |
| `FIFA_RANKINGS_PATH` | historical ranking timeline | `ranking_fifa_historical.csv` |
| `FIFA_WC_TEAMS_PATH` | World Cup pre-tournament rank validation | `wc_teams.csv` |

Do not overload one variable for both. Verify FIFA's terms before redistribution.

## Real schema and rank reconstruction

`ranking_fifa_historical.csv` columns (validated, not assumed):
`team, total_points, date, id, id_num, team_short`. There is **no per-team rank
column** — `id`/`id_num` are release identifiers. Rank is **reconstructed**
within each release by sorting `total_points` descending, using **standard
competition ranking** (`rank = 1 + count of teams with strictly greater
points`; equal-points display order by team name for determinism). Missing
points are **never** replaced with zero, and rank is never inferred where
points are absent.

Real coverage (2026-06 snapshot): **67,894 rows, 335 releases, 235 teams,
1992-12-31 to 2024-09-19**, 11 missing points, 0 duplicate (team, release).

## Limitation: ends in 2024

The timeline ends **2024-09-19**. It therefore **cannot provide valid live 2026
FIFA values**. The as-of join returns the latest release before the match with
`days_since_release`; for a 2026 match that is ~620 days stale, and callers must
refuse or down-weight it rather than treat it as current.

`wc_teams.csv` columns: `year, team, confederation, rank`.

## Leakage-safe as-of join

The only correct way to attach a ranking to a match is a **backward** as-of
join: use the latest ranking released *strictly before* the match date. Never
the nearest future ranking.

This is implemented and tested now as a pure function:

```python
from goalsignal.data.sources.fifa_rankings import as_of_ranking
as_of_ranking(rankings_df, team="Brazil", match_date="2026-06-20")
# -> latest ranking released before 2026-06-20, or None
```

A ranking released exactly on the match date is excluded by default (it may be
published after kickoff). The record's `available_at` must not precede its
`ranking_release_date` (enforced by the schema).

## Derived fields (Milestone D)

home/away FIFA points and ranks, points/rank differences, ranking release date,
days since release, and an **Elo–FIFA disagreement** signal.

## Comparison design (Milestone F, no assumed winner)

Ablations will compare, on identical chronological folds:

- custom Elo only (current baseline),
- FIFA only,
- custom Elo + FIFA,
- Elo/FIFA disagreement.

FIFA rankings are **not assumed** to improve performance; the experiment decides,
and a negative result is reported as such.

## File setup and commands

```bash
export FIFA_RANKINGS_PATH=Datasets/ranking_fifa_historical.csv
export FIFA_WC_TEAMS_PATH=Datasets/wc_teams.csv
goalsignal fifa-rankings inspect             # confirm both files
goalsignal fifa-rankings validate            # schema + quality, no reports written
goalsignal fifa-rankings ingest              # reconstruct rank + 4 reports + alias audit
goalsignal fifa-rankings world-cup-validate  # compare reconstructed ranks to wc_teams.csv
goalsignal fifa-rankings coverage
```

World Cup validation (248 team-years, real run): **188 exact, 28 small, 6 large
discrepancy, 26 unmatched** (aliases like "South Korea"↔"Korea Republic"). The
six large discrepancies reflect FIFA's official methodology vs reconstructed
standard-competition ranking; neither dataset is modified.

`load_rankings` applies a documented column-mapping layer (e.g. `country_full`→
`team`, `rank_date`→`ranking_release_date`, `total_points`→`ranking_points`),
so common public schemas load without editing the file. Ingestion validates
dates, ranks, points, and duplicate (team, release-date) rows, and writes:

- `artifacts/reports/fifa_rankings_coverage.csv`
- `artifacts/reports/fifa_rankings_quality.json`
- `artifacts/reports/fifa_rankings_unmatched_teams.csv` (teams not linked to a
  canonical GoalSignal team — for Milestone C alias work)

## Status

Milestone B: loader, validation, the leakage-safe `as_of_ranking` join, and
reports implemented and tested on synthetic data. Currently **`not_configured`**
(no `FIFA_RANKINGS_PATH` set).
