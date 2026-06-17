# 2026 World Cup Squad Data

GoalSignal accepts squad membership only from official FIFA tournament lists,
official federation announcements, or a manually curated CSV that cites those
sources. It does not infer final squads from call-ups and does not scrape
Transfermarkt, Wikipedia, social media, betting sites, or fan databases.

## Configuration

All inputs are optional and configured in `.env`:

- `FIFA_2026_SQUADS_PATH`: verified squad-membership CSV.
- `FIFA_2026_SQUAD_EXTRACT_PATH`: expanded official-document audit extract.
- `FIFA_2026_PLAYER_LINK_CANDIDATES_PATH`: conservative identity seed candidates.
- `FIFA_2026_AVAILABILITY_PATH`: timestamped availability records.
- `FIFA_2026_PLAYER_ALIASES_PATH`: reviewed player identity mappings.

The membership template is
`data/reference/world_cup_2026_squads_template.csv`. Required fields are:
snapshot date, national team, player name, position, squad status, source name,
and source publication date. Optional values remain missing rather than being
guessed. Allowed statuses are configured in `config/squads.yaml`.

Availability such as injured, doubtful, or suspended is not squad membership
and belongs in the separate availability input.

## Validation And Provenance

The loader is UTF-8/BOM safe, preserves every raw field, normalizes teams
through reviewed aliases, validates groups A-L and configured statuses, rejects
duplicate players within a snapshot, and records conflicts. Every snapshot has
a content-derived manifest and each row is unavailable before its source
publication date.

```bash
goalsignal squads inspect
goalsignal squads validate --cutoff 2026-06-15
goalsignal squads ingest
goalsignal squads coverage
```

Without `FIFA_2026_SQUADS_PATH`, coverage and readiness commands write an
explicit `blocked by missing squad source` report. No player is assumed
selected.

Quality and source coverage live under `artifacts/reports/squad_2026_*`;
content manifests under `artifacts/manifests/squad_2026_*.json`; normalized
records under `artifacts/player_data/`. These are source audits and descriptive
aggregates, not deployed features.

## Real 2026 Snapshot

The repository defaults to three role-separated CSVs in `Datasets/`.
Validation found 1,248 selected players, 48 teams, 26 players per team, and
four teams in each group A-L. The expanded FIFA extract reconciles all 1,248
rows: 1,040 exact and 208 normalization-equivalent optional-field shifts, with
no substantive discrepancy or missing PDF page. Raw files remain read-only.

The completed 312-row alias review is revalidated against both the official
squad and local Transfermarkt snapshot. Results are 234 accepted-local, 63
accepted web-only, and 15 conflicts. The alias source is preserved byte-for-byte;
generated classifications live under `artifacts/reports/squad_alias_*`.
