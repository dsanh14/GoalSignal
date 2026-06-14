# Data Setup

## Source files

GoalSignal operates on four user-provided CSVs (it never downloads or
replaces them). They currently live in `Datasets/` at the repository root:

| File | Rows (2026-06-12 snapshot) | Contents |
| --- | --- | --- |
| `results.csv` | 49,477 | Senior men's full internationals, 1872–present; includes scheduled 2026 World Cup fixtures with `NA` scores |
| `shootouts.csv` | 678 | Penalty shootout winner and (where known) first shooter |
| `goalscorers.csv` | 47,606 | Goal events with scorer, minute, own-goal and penalty flags (extra `minute` column beyond the original spec) |
| `former_names.csv` | 36 | Date-bounded historical team-name mappings |

To use a different location:

```bash
uv run goalsignal data validate --input-dir path/to/dir
```

The directory must contain all four files with the standard names (names are
configurable in `config/data.yaml`).

## Build outputs

`uv run goalsignal data build` produces:

- `data/processed/matches.csv` — canonical match table (one row per match,
  played and scheduled), with normalized team names, canonical match IDs,
  score-scope fields, and shootout joins.
- `artifacts/reports/` — audit reports (see `docs/data_quality.md`).
- `artifacts/manifests/dataset_<version>.json` — content-hashed manifest.
  The `dataset_version` is a deterministic function of the source-file
  SHA-256 hashes, schema version, and scope/validation policy.

All outputs are reproducible and git-ignored. The build refuses to overwrite
an existing dataset without `--force`.
