# Transfermarkt Player/Club Data

Optional player/club history from a Transfermarkt-derived export. Configured by
`PLAYER_DATA_PATH`, opened **read-only**.

## Format (real, as provided)

The configured source is a **directory of gzipped CSV tables**
(`transfermarkt-datasets` export), not a DuckDB file. The loader auto-detects:
a `.duckdb`/`.db` file is opened with DuckDB in `read_only=True` (optional
`duckdb` dependency); a directory of `*.csv(.gz)` is read with pandas. Source
files are never mutated; a before/after SHA-256 over the source proves it.

Tables present (real row counts, 2026-06 snapshot):

| Table | Rows | Note |
| --- | --- | --- |
| players | 47,716 | identity + many CURRENT-STATE fields |
| appearances | 1,885,697 | dated club appearances (2012–2026) |
| game_lineups | 3,172,509 | dated club lineups (start/sub) |
| player_valuations | 507,815 | dated market values (2000–2026) |
| games | 88,808 | mostly club; only 670 national-team-competition |
| clubs | 796 | current-state value/squad |
| competitions | 67 | 5 national-team, 3 international-cup |
| national_teams | 118 | CURRENT-STATE only (fifa_ranking, squad…) |
| transfers, club_games, countries, game_events | … | club-centric |

## Critical finding: club-centric, not international

Transfermarkt is **overwhelmingly club football**. National-team match coverage
is sparse (670 national-team-competition games), and the international fields
(`players.international_caps`, `current_national_team_id`,
`national_teams.fifa_ranking`) are **current-state metadata**. So this source
can supply **club-form / market-value proxies** (dated, cutoff-safe) but does
**not** directly enrich GoalSignal's international fixtures with lineups or
appearances.

## Temporal safety

Every field is classified in
`artifacts/reports/transfermarkt_temporal_field_audit.md` and summarized in
[player_temporal_semantics.md](player_temporal_semantics.md). The audit found
**23 current-state-unsafe fields**; these must never be applied to a historical
match.

## Commands (read-only)

```bash
export PLAYER_DATA_PATH=Datasets/transfermarkt-datasets
goalsignal player-data inspect            # kind + table list
goalsignal player-data inventory          # rows/cols/dtypes/nulls per table
goalsignal player-data temporal-audit     # field-level temporal classification
goalsignal player-data coverage           # full audit + coverage reports
goalsignal player-data identity-candidates  # entity-linking scaffolding (no auto-match)
```

Reports: `transfermarkt_table_inventory.json`,
`transfermarkt_temporal_field_audit.md`, `transfermarkt_table_quality.csv`,
`transfermarkt_{player,national_team,competition,lineup}_coverage.csv`,
`transfermarkt_coverage.json`.

## Licensing

Transfermarkt-derived; non-commercial research use. **Verify Transfermarkt
terms before redistribution. Do not scrape.** Attribution recorded in config.
