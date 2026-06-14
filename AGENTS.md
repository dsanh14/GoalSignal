# AGENTS.md — Working Agreement for GoalSignal

Read this file before modifying the repository. It records the rules every
agent session must follow and the verified current state of the project.

## Purpose

GoalSignal is a leakage-safe probabilistic forecasting and tournament
simulation system for international football. Application: forecast every 2026
FIFA World Cup match and each team's advancement probabilities. Research
question: how much stable out-of-sample predictive signal historical match
data contains, evaluated strictly chronologically.

## Project status (verified 2026-06-12)

**Complete:** the core statistical pipeline and group-stage simulation — data
foundation, Elo ratings, chronological backtesting with baselines, Poisson and
Dixon-Coles goal models, calibration, ensemble, Monte Carlo group-stage
simulation of the real 2026 fixtures, and a hash-chained prediction ledger
holding 70 immutable World Cup forecasts. 55 tests pass; ruff clean.

**Not complete:** the full original roadmap. See "Open work" below. Do not
describe the project as finished.

### Original roadmap → implemented milestone mapping

| Original workstream | Status | Where |
| --- | --- | --- |
| 1. Data validation / normalization / canonical dataset | DONE (M1) | `data/`, `goalsignal data *` |
| 2. Elo team-strength ratings | DONE (M2) | `ratings/elo.py`, `config/elo.yaml` |
| 3. Feature engineering | PARTIAL (M3) | Elo + venue only (`features/build_features.py`); form/travel/H2H/squad open |
| 4. Baselines 0–4 | DONE (M3) | `models/baselines.py` |
| 5. Goal models (Poisson, Dixon-Coles) | DONE (M4) | `models/poisson.py`, `models/dixon_coles.py`; bivariate Poisson / neg-binomial open |
| 6. Direct outcome model | DONE (M4) | `models/outcome_classifier.py` (softmax regression) |
| 7. Ensemble | DONE (M5) | `models/ensemble.py` |
| 8. Calibration | DONE (M5) | `models/calibration.py` (temperature scaling); isotonic open |
| 9. Chronological evaluation + bootstrap uncertainty | DONE (M3) | `evaluation/backtest.py`, `evaluation/metrics.py` |
| 10. Ablations + regime analysis (H1/H2/H4/H5/H6/H9/H10) | OPEN | — |
| 11. Tournament rules + Monte Carlo simulation | DONE for group stage (M6) | `tournament/`; knockout bracket config open |
| 12. Performance engineering | DONE for simulators (M6) | reference vs vectorized, `goalsignal benchmark`; parallel/C++ open |
| 13. Continuous learning (result record, drift, champion–challenger) | PARTIAL | `feedback/` + `goalsignal result record|correct`, `feedback match|summary` (append-only result store, post-match scoring, Elo online updates, future-only refresh under `ensemble-v1+rN`); drift + champion–challenger open |
| 14. Prediction ledger | DONE (M7) | `ledger/storage.py`, `goalsignal ledger *` |
| 15. API / dashboard / release audit | OPEN | research report done (M8: `docs/research_report.md`) |
| 16. Enrichment layer (players/lineups/StatsBomb/FIFA/rest/travel) | IN PROGRESS — **Milestones A (contracts) + B (ingestion) done** | A: `data/sources/` protocols/schemas/manifests/config. B: **API-Sports/API-Football** client (`http_client.py`, `api_football.py`, `throttle.py`), raw cache (`cache.py`), normalization (`api_football_normalize.py`), StatsBomb offline loader + aggregation, FIFA loader + as-of + reports, fixture linking (`linking.py`), coverage (`coverage.py`), `goalsignal sources|api-football|statsbomb|fifa-rankings` CLI. (Earlier wrong provider football-data.org was removed — `docs/api_football_migration.md`.) Linking finalization (C), features (D), early/final forecasts + ledger v2 (E), ablations (F), challenger (G) OPEN |

### Verified deployment snapshot (from manifests — do not edit by hand)

- Dataset version: `e762e61836662aed`
  (`artifacts/manifests/dataset_e762e61836662aed.json`; source SHA-256 hashes
  inside). 49,477 raw rows → 49,476 canonical (49,406 played + 70 scheduled).
- Data cutoff: 2026-06-12 (last played match 2026-06-11).
- Model: `ensemble-v1` — temperature-calibrated convex ensemble; weights
  dixon_coles 0.4777, multinomial_logistic 0.5223, elo_davidson 0.0;
  temperatures 1.040 / 1.048 / 0.840; Dixon-Coles rho −0.0313; trained on
  46,005 matches, calibrated on 3,364 (validation window 2023-06-12 →
  2026-06-12). Recorded in `artifacts/simulations/wc2026_group_stage_meta.json`.
- Tournament simulation: 100,000 sims, seed 20260612.
- Backtest (2010–2025, 15,499 matches): ensemble log loss 0.8924
  [0.8745, 0.9108]; details in `artifacts/reports/backtest/overall.json` and
  `docs/research_report.md`.

## Repository architecture

- `Datasets/` — **user-provided source data, read-only for GoalSignal.**
  results.csv, shootouts.csv, goalscorers.csv, former_names.csv. The code
  never writes here. Never scrape or download substitute data.
- `config/` — `data.yaml` (input paths, validation thresholds, score-scope
  policy), `elo.yaml` (K, scale, home advantage, importance multipliers,
  shootout policy).
- `src/goalsignal/`
  - `data/` — schemas, loaders, date-aware team normalization
    (`normalize_teams.py`), canonical build with score-scope semantics
    (`build_dataset.py`), audit reporting (`validation.py`), manifests
    (`metadata.py`).
  - `ratings/elo.py` — sequential leakage-free Elo timeline.
  - `features/build_features.py` — model input frame (pre-match Elo + venue).
  - `models/` — `baselines.py` (uniform, empirical, context, higher-rated,
    Elo-Davidson), `poisson.py` (IRLS GLM + score matrices/markets),
    `dixon_coles.py`, `outcome_classifier.py` (softmax regression),
    `calibration.py` (temperature scaling), `ensemble.py` (simplex weights).
  - `evaluation/` — `metrics.py` (log loss, Brier, RPS, ECE, reliability,
    block bootstrap), `backtest.py` (expanding-window protocol).
  - `tournament/` — `rules.py` (FIFA tiebreakers incl. head-to-head and
    lots), `knockout.py` (regulation/ET/shootout kept separate),
    `simulator.py` (reference + vectorized MC, invariants),
    `fixtures_2026.py` (groups derived from the fixture graph),
    `model_adapter.py`.
  - `ledger/storage.py` — hash-chained append-only prediction ledger.
  - `live.py` — deployment pipeline (mirrors the backtest protocol exactly).
  - `utils/` — repo-root path resolution, SHA-256 hashing.
  - `cli.py` — Typer app, entry point `goalsignal`.
- `tests/` — 55 tests (`unit/`, `integration/`), synthetic fixtures only
  (fictional teams); never depend on the real CSVs.
- `docs/` — data_setup, data_quality, leakage_prevention, research_report.
- `data/processed/`, `artifacts/` — generated outputs, git-ignored.

## Commands (all verified via `--help` and real runs)

All support `--config` (default `config/data.yaml`) and `--input-dir` to
point at a different dataset directory. On macOS prefer
`UV_NO_EDITABLE=1 uv run goalsignal ...` (see environment gotcha).

| Command | Purpose | Key flags | Writes |
| --- | --- | --- | --- |
| `data inspect` | Source schemas, row counts, date range | | — |
| `data validate` | Quality checks + audit reports | | `artifacts/reports/*.{json,md,csv}` |
| `data build` | Canonical dataset + reports + manifest | `--force` to overwrite | `data/processed/matches.csv`, `artifacts/manifests/dataset_<ver>.json` |
| `ratings build` | Full Elo timeline + final ratings | | `artifacts/ratings/elo_timeline.csv`, `final_ratings.csv` |
| `ratings inspect` | Top-rated teams now | `--top N` | — |
| `evaluate rolling` | Expanding-window backtest of all models | `--start-year --end-year --val-years` | `artifacts/reports/backtest/{overall.json,metrics_by_fold.csv,goal_metrics_by_fold.csv,test_predictions.csv}` |
| `tournament simulate` | 2026 group-stage Monte Carlo | `--sims --seed` | `artifacts/simulations/wc2026_group_stage.csv` + `_meta.json` |
| `predict remaining` | Ledger predictions for every scheduled fixture | | appends `artifacts/predictions/ledger.jsonl` |
| `ledger list` | Show stored predictions | | — |
| `ledger verify` | Verify hash chain (nonzero exit on tamper) | | — |
| `benchmark` | Measure reference vs vectorized simulator | `--sims --repeats` | `artifacts/benchmarks/simulator_benchmark.json` |
| `result record` | Append a completed result (separate, hash-chained store; duplicates rejected) | `--fixture-id --home-goals --away-goals --completed-at --source` | `artifacts/results/results.jsonl`, `artifacts/ratings/online_updates.jsonl` |
| `result correct` | Audited correction superseding a prior result | adds `--reason` | appends to result store |
| `feedback match` | Score a frozen forecast vs the recorded result (p(scoreline) read from payload or validated frozen-model reconstruction) | `--fixture-id` | `artifacts/reports/feedback/match_<id>.json` |
| `feedback summary` | Aggregate realized performance over recorded results | | — |

Recorded results are overlaid in memory by `_live_model` (never written to
`Datasets/`): the cutoff advances and refreshed predictions append under
`ensemble-v1+rN`, so frozen forecasts are never modified.

### End-to-end reproduction

```bash
uv sync
UV_NO_EDITABLE=1 uv run pytest && uv run ruff check .
UV_NO_EDITABLE=1 uv run goalsignal data build --force
UV_NO_EDITABLE=1 uv run goalsignal ratings build
UV_NO_EDITABLE=1 uv run goalsignal evaluate rolling --start-year 2010 --end-year 2025
UV_NO_EDITABLE=1 uv run goalsignal tournament simulate --sims 100000 --seed 20260612
UV_NO_EDITABLE=1 uv run goalsignal predict remaining   # refuses duplicates per model_version
UV_NO_EDITABLE=1 uv run goalsignal ledger verify
UV_NO_EDITABLE=1 uv run goalsignal benchmark
```

Everything except ledger appends is deterministic given the dataset and seeds;
`predict remaining` fails by design if predictions for the same fixtures and
model_version already exist (immutability).

## Artifact policy

- **Git-ignored, regenerate on demand** (large, deterministic): all of
  `artifacts/` and `data/processed/` are ignored via `.gitignore`. The big
  ones: `data/processed/matches.csv` (~12 MB), `artifacts/ratings/
  elo_timeline.csv` (~11 MB), `artifacts/reports/backtest/
  test_predictions.csv` (~13 MB).
- **Small evidence artifacts** worth publishing at release-audit time (a
  future decision, currently still ignored): the dataset manifest,
  `backtest/overall.json`, `metrics_by_fold.csv`,
  `simulator_benchmark.json`, `wc2026_group_stage.csv` + `_meta.json`, and
  `artifacts/predictions/ledger.jsonl` (the immutable forecast record —
  treat as append-only evidence; never delete or rewrite it to "clean up").
- Raw data in `Datasets/` is the user's; it is tracked in git history already
  but must never be modified by code.

## Incorporating new results (read-only `Datasets/` clarified)

GoalSignal never writes to `Datasets/`. New completed matches enter the
system when the **user** drops an updated snapshot of the source CSVs into
`Datasets/` (or another directory passed via `--input-dir`). The manifest
hashes then produce a **new dataset_version automatically**, the Elo cutoff
advances to the day after the newest played match, and rerunning
`data build --force`, `tournament simulate`, and `predict remaining`
regenerates forecasts under the new cutoff. Existing ledger entries are never
modified; new predictions for already-predicted fixtures require a new
`model_version`. The planned `result record` command (open work) will store
match results in a separate store — also never in `Datasets/`.

## Non-negotiable rules

1. **No fabrication.** Never invent data, row counts, metrics, benchmark
   results, test output, or claim a command passed without running it.
2. **No leakage.** A prediction at time T uses only information available
   before T. No random train/test splits as primary evaluation; no tuning on
   final test periods; calibrators and ensemble weights fit on validation
   predictions only.
3. **No silent data mutation.** Every exclusion or correction goes to the
   audit reports with source row, reason, severity, review status.
4. **Score semantics.** Recorded scores include extra time, exclude penalty
   shootouts. Extra time is provable only when a shootout exists; see
   `docs/data_quality.md`. Never add shootout goals to match goals; never
   treat a shootout winner as a regulation winner.
5. **Immutable predictions.** The ledger is append-only and hash-chained;
   `ledger verify` must always pass. Results are stored separately.
6. **Datasets are versioned** by content hash via manifests; never reference
   a dataset only as "latest".
7. **Baselines before complexity.** A complex model earns its place only by
   improving honest out-of-sample performance on identical chronological
   folds. Negative findings are reported, not hidden (see research report:
   H3/H7 only weakly supported).
8. **CPU-only base workflow.** Heavy dependencies stay optional. New
   dependencies require justification (current: numpy, pandas, scipy,
   pydantic, pyyaml, typer).
9. **Tournament rules live in configuration / dedicated modules**, separate
   from model logic. The official 2026 Round-of-32 bracket mapping and group
   letters are NOT in the dataset: group labels G01..G12 are synthetic, and
   knockout simulation beyond R32 qualification is intentionally absent.
   **Never fabricate official fixtures or bracket mappings.**

## Enrichment layer (optional, Milestone A = contracts only)

`src/goalsignal/data/sources/` defines the **optional** enrichment sources:
StatsBomb open data, the football-data.org API, historical FIFA rankings,
player/club identity. As of Milestone A there is **no ingestion and no network
call**: only protocols (`base.py`), normalized schemas (`schemas.py`),
deterministic source manifests (`manifests.py`), config loaders (`config.py`),
and offline-testable helpers (FIFA as-of join, player resolution, travel math,
rate limiter). All `fetch()`/`load()` paths raise `MilestoneNotImplementedError`.

Rules specific to this layer:
- **Provider: API-Sports / API-Football v3** (an earlier pass used the wrong
  provider, football-data.org — now removed; see
  `docs/api_football_migration.md`). Base `https://v3.football.api-sports.io`,
  header `x-apisports-key`, key from `$FOOTBALL_DATA_API_KEY` (an API-Sports
  key). Free plan: **100 requests/day**, ~10/min; the client is host-locked,
  cache-first, and tracks daily usage. **World Cup injuries are unsupported by
  the provider** (competition `cov_injuries=false`) — never invented. Provider
  `/predictions` are stored only as an external benchmark, never a feature.
- Credentials/paths come from env vars (`.env`, git-ignored;
  `.env.example` tracked): `FOOTBALL_DATA_API_KEY`, `STATSBOMB_DATA_PATH`,
  `FIFA_RANKINGS_PATH`, **`FIFA_WC_TEAMS_PATH`** (separate WC validation file),
  `PLAYER_DATA_PATH`. **Never hard-code keys; never commit `.env`.** Heavy deps
  are optional extras (`uv sync --extra http|statsbomb|enrichment`).
- Every external field carries a `ProvenanceEnvelope` (source, record id,
  retrieved_at, **available_at**, snapshot hash, schema version). Leakage checks
  use `available_at`; future-dated info is rejected (`assert_available_before`).
- Missing enrichment is flagged, **never zero-filled**. Expected vs confirmed
  lineups, and early vs final forecasts, stay separate. See
  `docs/data_sources.md`, `docs/football_data_api.md`, `docs/feature_availability.md`.
- The deployed `ensemble-v1` baseline is unchanged; enrichment is OFF by default
  and no challenger is auto-promoted.

### Milestone B ingestion (API-Sports / API-Football)

- **Secrets:** key read only from `$FOOTBALL_DATA_API_KEY` (git-ignored `.env`),
  sent as `x-apisports-key` **only to the host-locked API-Sports base URL**.
  Never logged/returned/cached/hashed. One redacting seam (`http_client.py`,
  masks both `x-apisports-key` and legacy `x-auth-token`); cache `request.json`
  omits headers.
- **Client:** `ApiFootballClient` (injectable `Transport`, host-lock, bounded
  retries, envelope-error parsing → typed errors incl. `PlanLimitationError`,
  per-minute + persistent **daily** throttle, cache-first with `--refresh`).
  Raw snapshots cache immutably under
  `data/external/api_football/raw/<snapshot_id>/`; usage counter under
  `.../usage/<date>.json`.
- **Verified live (2026-06-13):** `/status` → HTTP 200, **plan Free, 100/day**.
  World Cup discovered via `/leagues`: **league_id=1** ("World Cup", "World",
  seasons incl. 2026). Coverage audit
  (`artifacts/reports/api_football_coverage.json`): 2026 match data is
  **plan-locked on Free** ("Free plans do not have access to this season"), and
  World Cup **injuries are unsupported** (`cov_injuries=false`). So live 2026
  fixtures/lineups/player-stats need a paid plan; injuries never apply.
- **CLI:** `goalsignal api-football probe|discover-world-cup|fixtures|standings|
  lineups|injuries|fixture-players|inspect-cache`.
- **StatsBomb / FIFA:** offline only; both `not_configured`. Loaders implemented
  and tested on synthetic data.
- **Tests:** live API tests are marked `@pytest.mark.live_api` and excluded by
  default; run with `pytest -m live_api`. Normalized tables are CSV.

### Real-data audit (2026-06-13) — FIFA, Transfermarkt, StatsBomb

Local datasets in `Datasets/` were audited (read-only). Reports in
`artifacts/reports/`; readiness in `source_readiness.{json,md}`,
`enrichment_coverage.csv`. Honest findings:
- **FIFA rankings** (`ranking_fifa_historical.csv`): real schema `team,
  total_points, date, id, id_num, team_short` — **no rank column**, so rank is
  reconstructed (standard competition ranking) within each release. 67,894
  rows, 335 releases, 1992–**2024**. **Ends 2024 → cannot supply live 2026
  values** (as-of join exposes `days_since_release`; ~620 days stale for 2026).
  WC validation vs `wc_teams.csv` (separate `FIFA_WC_TEAMS_PATH`): 188 exact /
  28 small / 6 large / 26 unmatched (aliases). `goalsignal fifa-rankings
  inspect|validate|ingest|world-cup-validate|coverage`.
- **Transfermarkt** (`transfermarkt-datasets/`): **a directory of gzipped CSVs,
  not a DuckDB file** — opened read-only with pandas (source hash unchanged
  before/after). **Club-centric**: players 47,716, appearances 1.89M, lineups
  3.17M, valuations 508k, but only **670 national-team games**. 23
  **current-state-unsafe** fields (current club/caps/value) must never be used
  historically; dated appearance/lineup/valuation rows are cutoff-safe club
  proxies. `goalsignal player-data inspect|inventory|temporal-audit|coverage|
  identity-candidates`. See `docs/player_temporal_semantics.md`.
- **StatsBomb**: not present locally (`not_configured`); commands degrade
  gracefully with setup instructions.
- **Readiness** (`goalsignal sources readiness`): ready = FIFA points/rank,
  FIFA–Elo disagreement, historical valuations, rest, native form;
  restricted_subset = club minutes/starts/strength; blocked = StatsBomb,
  confirmed lineups (plan); unsupported = injuries/suspensions.
- No models trained, no predictions generated, ledger/result store unchanged.

### Milestone D1 — leakage-safe feature engineering + ablation (offline)

Built FIFA, FIFA-Elo disagreement, recent-form, attack/defense, rest, and basic
venue features and evaluated them via chronological ablation. **Deployed
`ensemble-v1` unchanged; nothing deployed; ledger/result store untouched.**
- **Features:** `src/goalsignal/features/d1.py` (leakage-safe: strictly-prior
  matches per `(date, source_row)`; FIFA as-of strictly before fixture;
  opponent-adjusted goal residuals use a FIXED Elo mapping — no fitting/
  circularity). Table `artifacts/features/d1/d1.1/` (49,406 rows, 92 cols,
  1872-2026), config in `config/features_{native,fifa}.yaml`,
  `config/experiments_d1.yaml`. CLI: `goalsignal features
  build-d1|inspect-d1|validate-d1|coverage-d1`.
- **FIFA coverage ends 2024-09-19** → availability capped at 450 days so **every
  2026 fixture is FIFA-unavailable** (no forward-fill; `validate-d1` enforces).
- **Ablation** (`evaluation/d1_ablation.py`, `goalsignal evaluate
  d1-ablation|d1-regimes|d1-report`): expanding-window 2010-2023, fold-local
  median-impute + standardize (train only), temperature calibration, identical
  paired test matches, paired year-block bootstrap.
- **Result (13,266 matches vs internal Elo-only baseline 0.8975 log loss):**
  attack/defense -0.0098 and recent form -0.0050 are the supported gains; all
  D1 -0.0127; **native-no-FIFA -0.0120 ≈ all-D1** (FIFA adds little beyond Elo);
  **disagreement and venue: no measurable difference**. Reports: `d1_*` in
  `artifacts/reports/`. Recommendation: advance native form + attack/defense to
  a deployment-grade eval **against the ensemble champion**; do not deploy here.

## Conventions

- Python 3.12, uv-managed. Ruff (line length 100); pytest; **97 tests** must pass
  and lint must be clean before finishing any change.
- Paths resolve against the repo root (`goalsignal.utils.paths.resolve`).
- CLI commands validate inputs, exit nonzero on failure, print artifact
  paths, and refuse to overwrite outputs unless `--force` is passed.
- Canonical match IDs: SHA-256 of normalized (date, home, away, tournament,
  city, country); scores excluded from identity.
- Experiments record hypothesis, periods, metrics, decision rule, and
  uncertainty before results are interpreted.

## Environment gotcha (macOS)

Something on this machine sets the UF_HIDDEN flag on `.pth` files, and
CPython 3.12+ skips hidden `.pth` files, breaking editable installs with
`ModuleNotFoundError: No module named 'goalsignal'`. Mitigations are in
pyproject (`link-mode = "copy"`, `cache-keys` so non-editable installs rebuild
on source changes); prefer `UV_NO_EDITABLE=1 uv run ...`. If the error
appears: `ls -lO .venv/lib/python3.12/site-packages/*.pth` and
`chflags nohidden` the flagged files.

## Open work (priority order)

1. **Official 2026 knockout bracket mapping** — user-supplied
   `config/tournament_2026.yaml` (schema + validation needed); unlocks
   R32→champion probabilities. Time-critical: group stage ends 2026-06-27.
2. **Form / venue / travel / rest / head-to-head features** — the next
   plausible accuracy gain (research report H8: new information, not new
   model classes).
3. **Ablation suite + regime analysis** — hypotheses H1, H2, H4, H5, H6, H9,
   H10 are recorded but untested.
4. **Result recording + live feedback** — `result record`, post-match scoring
   of frozen ledger entries, online state updates.
5. **Drift monitoring + champion–challenger** — `feedback *` and `model *`
   command families.
6. **Optional API / dashboard** — keep dependencies optional.
7. **Release audit** — decide which evidence artifacts to publish; final
   reproducibility pass.

## Milestone workflow

For each milestone: inspect existing code → present a short plan → implement
with tests → run lint + tests + the real-data pipeline → update this file's
status sections → report honestly, including failures and open questions.
