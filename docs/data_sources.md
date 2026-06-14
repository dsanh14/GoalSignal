# Enrichment Data Sources

GoalSignal's base forecasting workflow (Elo, Poisson, Dixon-Coles, multinomial
logistic, calibration, ensemble) depends on **only** the user-provided
international-results dataset. Everything documented here is an **optional
enrichment layer** layered on top to test one research question:

> Do player availability, confirmed lineups, event-level performance, FIFA
> rankings, rest, travel, and venue context provide stable out-of-sample
> predictive value beyond the current team-level model?

Status: **Milestones A (contracts) + B (ingestion) complete.** A: protocols,
schemas, configuration, provenance/manifest models. B: a real football-data.org
client (cache, retries, redaction, normalization), StatsBomb offline loader +
aggregation, FIFA loader + leakage-safe as-of join + reports, fixture-linking
preparation, and real coverage reports — all exercised by tests. No enriched
model is trained and the deployed baseline is untouched (that is Milestones D+).

Provider note: the live source is **API-Sports / API-Football** (see
[api_football.md](api_football.md)). An earlier pass used the wrong provider
(football-data.org); that was removed — see
[api_football_migration.md](api_football_migration.md).

Current configured state (`goalsignal sources coverage`): api-football auth
**verified** (Free plan), but 2026 World Cup match data is **plan-locked** and
World Cup injuries are unsupported; StatsBomb `not_configured`, FIFA
`not_configured`.

## Source roles

| Source | Role | Access | License | Credential / path |
| --- | --- | --- | --- | --- |
| International results (`Datasets/`) | Canonical fixture backbone, results, Elo | Local, already present | User-owned | — |
| StatsBomb open data | Historical event enrichment (xG, shots, lineups, cards) | Offline local clone | Non-commercial, attribution required | `STATSBOMB_DATA_PATH` |
| API-Sports / API-Football v3 | Live fixtures, standings, teams, squads, confirmed lineups, events, player stats, injuries | HTTPS API (`v3.football.api-sports.io`, header `x-apisports-key`) | Per-plan ToS | `FOOTBALL_DATA_API_KEY` (an API-Sports key) |
| Historical FIFA rankings | External strength baseline + Elo-disagreement signal | User-provided CSV | Verify FIFA terms | `FIFA_RANKINGS_PATH` |
| Player/squad/availability | Player strength, lineups, injuries/suspensions | Provider-dependent | Per provider | `PLAYER_DATA_PATH` |

No single source contains everything. In particular, **injuries are not
covered for the World Cup competition by API-Football** (the competition's
`cov_injuries` flag is false; see [api_football.md](api_football.md)), so the
injuries/suspensions feature family stays unsupported for the World Cup and is
never fabricated.

## Credentials and configuration

- Credentials and local data paths are read from environment variables (see
  [`.env.example`](../.env.example)); copy it to `.env` (git-ignored) and fill
  in real values. **No secret is ever stored in YAML or committed.**
- All source behavior (URLs, competition IDs, rate limits, retries, paths,
  feature windows, strength weights) lives in `config/*.yaml`, never hard-coded.
- Every source is **disabled by default**. The base workflow runs with none set.
- Heavy/source-specific Python dependencies are optional extras:
  `uv sync --extra http` (httpx, for the API), `uv sync --extra statsbomb`
  (statsbombpy), or `uv sync --extra enrichment` (both).

## Cache and manifest layout

- Raw API responses and local source snapshots cache under `data/external/`
  (git-ignored, reproduced from source). football-data.org responses are
  namespaced by retrieval date: `data/external/football_data/<date>/`.
- Curated reference data (`data/reference/player_aliases.csv`,
  `club_aliases.csv`) is **tracked** — it is human-curated evidence.
- Every ingested snapshot gets a deterministic, content-derived snapshot ID and
  a manifest under `artifacts/manifests/sources/` recording source, endpoint/
  URL, retrieval timestamp, available-at semantics, license, attribution,
  content hash, row count, schema version, coverage period, and cache path. A
  source is never identified merely as "latest".

## Provenance and leakage rules (enforced by the contracts)

- Every externally sourced field carries a `ProvenanceEnvelope`: source,
  source record ID, `retrieved_at`, `available_at`, source snapshot hash,
  schema version.
- `available_at` is when information became *knowable* (ranking release,
  lineup announcement), not when we fetched it. Any feature with
  `available_at` after the prediction timestamp is **rejected**
  (`assert_available_before`).
- Missing enrichment is flagged explicitly and **never** encoded as zero.
- Expected and confirmed lineups are separate records and separate forecasts
  (see [feature_availability.md](feature_availability.md)).

## Related docs

- [statsbomb.md](statsbomb.md) — StatsBomb access, coverage caveats, features.
- [football_data_api.md](football_data_api.md) — verified endpoints, limits,
  supported vs unsupported fields.
- [fifa_rankings.md](fifa_rankings.md) — as-of join, comparison design.
- [player_identity.md](player_identity.md) — canonical player resolution rules.
- [feature_availability.md](feature_availability.md) — early vs final forecasts,
  availability timestamps, missingness policy.

Deferred to later milestones (not yet written, to avoid empty placeholders):
`player_data.md`, `enrichment_features.md`, `enrichment_experiments.md`,
`coverage_bias.md`, `model_card.md`.
