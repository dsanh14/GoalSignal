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

**Complete:** the core statistical pipeline and full tournament simulation — data
foundation, Elo ratings, chronological backtesting with baselines, Poisson and
Dixon-Coles goal models, calibration, ensemble, Monte Carlo group-stage
and official FIFA knockout simulation of the real 2026 fixtures, and a
hash-chained prediction ledger holding immutable World Cup forecasts.
The real 1,248-row squad source is ingested and reconciled. Reviewed aliases
raise identity coverage to 1,233/1,248 (98.8%): 1,170 locally linkable, 63
accepted web-only, and 15 material conflicts. No squad model is trained.
Ruff and the current test suite must pass before changes finish.

**Not complete:** the full original roadmap. The live-update milestone now
includes a separate current-FIFA snapshot, six active results, frozen forecast
feedback, online Elo audit, future-only revisions, and result-aware versioned
group simulation. See "Open work" below. Do not
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
| 11. Tournament rules + Monte Carlo simulation | DONE through champion | `tournament/`, `config/tournament_2026.yaml`; official FIFA M73-M104 graph and all 495 Annexe C combinations |
| 12. Performance engineering | DONE for simulators (M6) | reference vs vectorized, `goalsignal benchmark`; parallel/C++ open |
| 13. Continuous learning (result record, drift, champion–challenger) | PARTIAL | `feedback/` + `goalsignal result record|correct`, `feedback match|summary` (append-only result store, post-match scoring, Elo online updates, future-only refresh under `ensemble-v1+rN`); drift + champion–challenger open |
| 14. Prediction ledger | DONE (M7) | `ledger/storage.py`, `goalsignal ledger *` |
| 15. API / dashboard / release audit | OPEN | research report done (M8: `docs/research_report.md`) |
| 16. Enrichment layer (players/lineups/StatsBomb/FIFA/rest/travel) | IN PROGRESS — contracts, ingestion, D1, real squad ingestion/reviewed-link audit done | `data/sources/squads.py`, `config/squads.yaml`, `goalsignal squads *`; 1,233/1,248 identities, 1,170 locally linkable, no squad model trained |

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
    `simulator.py` (reference + vectorized group MC), `full_simulator.py`
    (official M73-M104 path to champion), `bracket_2026.py` (validated FIFA
    symbolic slots + 495 Annexe C combinations), `fixtures_2026.py`,
    `model_adapter.py`, `reporting.py`.
  - `ledger/storage.py` — hash-chained append-only prediction ledger.
  - `live.py` — deployment pipeline (mirrors the backtest protocol exactly).
  - `utils/` — repo-root path resolution, SHA-256 hashing.
  - `cli.py` — Typer app, entry point `goalsignal`.
- `tests/` — unit/integration tests use synthetic fixtures only
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
   from model logic. The dataset's group labels remain synthetic; official
   A-L labels come from the validated frozen FIFA snapshot. The official 2026
   M73-M104 mapping and all 495 Annexe C rows are preserved under
   `data/reference/` and loaded through `config/tournament_2026.yaml`.
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

### Squad-data foundation (2026-06-15)

- Optional official squad, availability, and reviewed-alias paths are defined
  in `config/squads.yaml` and `.env.example`.
- `data/sources/squads.py` implements BOM-safe official squad validation,
  content manifests, strict publication cutoffs, deterministic player linkage,
  dated activity/valuation extraction, lineup coverage, descriptive aggregates,
  expected-lineup/path contracts, Portugal audit, and readiness reports.
- Local Transfermarkt remains read-only and club-centric; StatsBomb remains
  not configured; API-Football 2026 lineups remain plan-locked.
- The repository-default snapshot validates at 1,248 rows, 48 teams, and 26
  players per team; the official extract reconciles 100%.
- Reviewed identity resolution links 1,233/1,248 players (98.8%): 1,170 local,
  63 web-only, and 15 conflicts. Portugal is 26/26 accepted-local.
- Dated activity has 488/1,248 with 30-day minutes and 647/1,248 with 90-day
  minutes. Historical valuations cover 838/1,248 players.
- No model is trained, and forecasts/ledgers are untouched by this milestone.

### 2026 squad scenario challenger (2026-06-15)

- Added a configuration-driven, offline S1-S7 squad sensitivity analysis.
  S7 combines cutoff-safe activity, starts, historical valuation, positional,
  goalkeeper, and depth proxies with coverage shrinkage and bounded
  expected-goal adjustments.
- This is **not a trained model and is not deployed**. The live team-level
  model remains champion; 20 teams pass coverage thresholds and 28 receive
  exact base fallback.
- The verified research run used 100,000 simulations and seed `20260612`.
  Portugal ranked fifth in squad strength and moved from 4.195% to 4.871%
  title probability. The modal final was Spain-Argentina; Spain remained the
  modal champion.
- Outputs are versioned under `artifacts/features/squad_2026/`,
  `artifacts/research_predictions/`, and `artifacts/simulations/squad-*`.
  Production predictions, result history, and the default tournament command
  remain unchanged.

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

### Outcome-first signal layer + ensemble wiring (opt-in)

A win/advance-probability product layer sits **on top of** the deployed model;
nothing here changes `ensemble-v1`, the ledger, or the default behaviour of any
existing command.

- **Signals** (`src/goalsignal/signals/`): standardized `OutcomeProbs`
  (group W/D/L) and `AdvanceProbs` (knockout) from six sources — `historical`,
  `market` (decimal odds → vig-removed), `squad_strength`, `recent_form`,
  `venue_context`, `expert` (LLM/analyst). Adjustment signals map a scalar edge
  through a fixed (config-driven, **unfitted**) Davidson model. Manual inputs in
  `data/manual/*.example.csv`; every file/column optional.
- **Meta-ensemble** (`signals/meta_ensemble.py`, `config/ensemble.yaml`):
  configurable weighted linear pool; **renormalizes over available signals**;
  named versions `baseline_historical` / `market_only` / `squad_form_challenger`
  / `llm_adjusted_challenger` / `final_ensemble`; records provenance + pairwise
  disagreement.
- **Historical adapter** (`signals/historical_adapter.py`): converts the trained
  `LiveModel` into signals (W/D/L from `predict_outcome`; advancement from the
  goal model's reg/ET/pen resolution). Provenance `live_model|fixture|unavailable`;
  missing returns gracefully.
- **Prediction API** (`signals/api.py`): `EnsemblePredictor.predict_match_ensemble`
  / `predict_knockout_ensemble` / `predict_batch_ensemble` — the main internal
  interface. CLI: `goalsignal signals predict|blend|market|disagreement|validate|
  tune-weights`.
- **Tournament** (`tournament/ensemble_adapter.py`): `EnsembleGoalAdapter`
  reweights the score matrix to ensemble W/D/L marginals (GD/GF tiebreakers
  preserved) and adds an `advance_probs` hook to `full_simulator.
  _pair_resolution_probabilities` (backward-compatible — historical path
  unchanged). Opt in with `goalsignal tournament simulate --prediction-source
  ensemble [--ensemble-version V]`; writes a distinct artifact version
  (`<ver>.ensemble-<V>`) and a provenance summary. Verified end-to-end on the
  real fixtures (300 sims): 307 matchups, all historical from the live model,
  invariants hold.
- **Backtest** (`evaluation/ensemble_backtest.py`): fixed-weight, leakage-safe
  version comparison → `artifacts/ensemble/backtest_comparison.csv` (outcome
  metrics + coverage + missing rate + high-disagreement buckets; sample input
  flagged a smoke test).
- **Weight tuning** (`signals/tuning.py`): **validation-only**; writes
  `artifacts/ensemble/tuned_weights.yaml` with justifying metrics; never mutates
  `config/ensemble.yaml`.
- **Eval** (`evaluation/outcome_eval.py`): per-class + binary (advance)
  calibration tables, `compare()` summary. Metrics: log loss / Brier /
  calibration primary; accuracy secondary.
- **Docs:** `README.md`, `docs/ensemble_signals.md`. **Status:** product layer
  implemented, tested (signal + wiring suites), ruff clean. Not yet deployed and
  no challenger auto-promoted.

#### Empirical evaluation + dynamic keying (2026-06-28)

- **Real backtest** (`evaluation/ensemble_backtest.py`, `goalsignal evaluate
  ensemble-backtest --predictions artifacts/reports/backtest/test_predictions.csv`):
  reuses the deployed model's leakage-safe out-of-sample `ensemble_*` columns as
  the `historical` signal — no retraining, no new leakage. Writes four artifacts
  to `artifacts/ensemble/`: `backtest_comparison.csv`, `backtest_summary.md`
  (the no-overclaim verdict), `calibration_by_version.csv`, `coverage_by_signal.csv`.
  **Verified on 15,499 matches:** baseline_historical log loss **0.8924** (matches
  the canonical figure); final_ensemble **identical** because non-historical
  manual coverage is ~0 → verdict **INSUFFICIENT DATA, keep opt-in** (honest, not
  overclaimed). Smoke path (sample) remains for CI.
- **Ablation** (`goalsignal evaluate ensemble-ablation`): historical-only vs
  historical + each signal group vs full; `ablation_comparison.csv` +
  `ablation_summary.md`. On real data all deltas ≈ 0 (coverage ~0).
- **Tuning** (`signals/tuning.py`): validation-only; now also writes
  `tuning_report.md` and emits/records a **low-coverage warning**; still never
  mutates `config/ensemble.yaml`.
- **Dynamic keying** (`signals/keying.py`): market/expert/venue rows may carry
  `team_a`/`team_b` (+ `stage` for venue) and resolve by normalized team pair —
  precedence match_id > forward pair > reverse pair (directional probs flipped
  via `OutcomeProbs.flip`/`AdvanceProbs.flip`; venue advantage negated). This
  closes the earlier gap: market/venue now attach to dynamic knockout pairings
  in the ensemble tournament (was 0 coverage before). Example files gained
  `team_a`/`team_b` columns (match_id still wins, so prior results unchanged).
- **Tests:** `tests/unit/test_keying.py`, `tests/unit/test_ensemble_reports.py`.
  Suite **279 passed**, ruff clean.
- **Open:** adjustment scalings still unfitted; **no real manual signal data at
  historical scale** — the only honest blocker to concluding the ensemble beats
  the baseline. Provide real market/squad/form/venue/expert coverage, then re-run
  the real backtest + ablation before considering promotion.

#### Knockout "survive and advance" layer (2026-06-29, opt-in, experimental)

- **Signal** (`signals/knockout_upset.py`): knockout-only advance adjustment.
  Models `P(advance) = P(win reg) + P(draw)·P(win ET/pens)` with a staged
  regulation/ET/penalty model (reuses the `tournament/knockout.py` staging;
  skellam from scipy). Expected goals split favourite/underdog Poisson means
  **multiplicatively** so a low-event/compact tie raises draw mass and routes the
  favourite's edge through the coin-flip path. **Anchored**: re-derives advance
  with vs without style/penalty evidence and applies only the difference, so no
  evidence ⇒ exactly no change (returns `None`). Per-match shift hard-capped
  (`max_advance_shift` 0.15); blend weight 0.05.
- **Inputs** (file-first, team-keyed, both optional): `data/manual/team_styles.csv`
  (0-100 style indicators) and `data/manual/penalties.csv` (current keeper/taker
  ratings + shootout records). Shootout history is **Beta-shrunk toward 50/50**
  (`shootout_prior_strength`), current ratings weighted above old country history,
  head-to-head shootout deviation capped (`shootout_cap` 0.12). Penalties only
  move advance meaningfully when draw/ET prob is high. Provenance tags:
  `low_block_survival_path`, `favorite_sterile_possession_risk`,
  `transition_threat`, `set_piece_underdog_path`, `penalty_path_boost`.
- **Config** (`config/ensemble.yaml`): `knockout_upset: 0.05` added to
  `final_ensemble`; new `knockout_survival` version (market/upset-leaning);
  `signal_params.knockout_upset` block of bounded, **unfitted** coefficients.
  Absent for group matches and for non-opted knockout runs ⇒ renormalized away ⇒
  default behavior byte-for-byte unchanged.
- **CLI**: `goalsignal signals predict|blend --include-knockout-upset` and
  `goalsignal tournament simulate --prediction-source ensemble
  --include-knockout-upset` (group stage + historical path untouched; ensemble
  upset runs write a distinct `*.ko-upset` artifact dir). Lookup precedence
  documented: match_id > pair+stage > pair > team-level features.
- **Tests:** `tests/unit/test_knockout_upset.py` (21). Suite **300 passed**,
  ruff clean.
- **Open / experimental:** coefficients are priors, not fitted; uses a calibrated
  eg fallback (no per-team xG); not yet validated on a chronological knockout
  backtest (shootout outcomes are rare). Do not promote as default.

#### Simulation comparison report (2026-06-29, read-only diagnostics)

- **Module** (`evaluation/simulation_comparison.py`) + CLI `goalsignal evaluate
  simulation-comparison`. **Read-only** over existing `artifacts/simulations/`
  runs — never re-runs or overwrites the simulator. Auto-discovers newest
  baseline (historical) / final_ensemble / knockout_survival runs by classifying
  their `wc2026_tournament_meta.json` (override with
  `--baseline/--final-ensemble/--knockout-survival`). Graceful on a missing run.
- **Writes 4 artifacts to `artifacts/ensemble/`:** `simulation_comparison.csv`
  (per-team semifinal/final/champion + pairwise deltas), `biggest_movers.csv`
  (`team,stage,comparison,from_prob,to_prob,delta,abs_delta`),
  `knockout_survival_explanations.csv` (per-matchup before/after advance +
  knockout_upset decomposition: internal shift, `net_move_from_upset`, draw prob,
  E[goals], penalty-path contribution, style/penalty/provenance tags),
  `simulation_comparison.md` (honest narrative). Matchup diagnostics use
  `--matches` (default `data/manual/knockout_matchups.example.csv`) or `--live`.
- **Honesty:** the MD report has explicit production-grade vs experimental
  sections and a "not claimed" block (no accuracy claim; penalty history not
  highly predictive; no guaranteed shootout winners). Separates the *version*
  effect from the *knockout_upset* effect (`net_move_from_upset`).
- **Tests:** `tests/unit/test_simulation_comparison.py` (8). Suite **308 passed**,
  ruff clean.

## Conventions

- Python 3.12, uv-managed. Ruff (line length 100); pytest; all tests must pass
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

1. **Resolve the 15 reviewed/local identity conflicts** — verify DOB/name
   disagreements against authoritative sources without weakening safeguards.
2. **Squad-feature chronological ablation** — evaluate cutoff-safe local
   activity/valuation proxies against the ensemble champion; do not deploy
   before an honest paired result.
3. **Form / venue / travel / rest / head-to-head features** — the next
   plausible accuracy gain (research report H8: new information, not new
   model classes).
4. **Ablation suite + regime analysis** — hypotheses H1, H2, H4, H5, H6, H9,
   H10 are recorded but untested.
5. **Result recording + live feedback** — `result record`, post-match scoring
   of frozen ledger entries, online state updates.
6. **Drift monitoring + champion–challenger** — `feedback *` and `model *`
   command families.
7. **Optional API / dashboard** — keep dependencies optional.
8. **Release audit** — decide which evidence artifacts to publish; final
   reproducibility pass.

## Milestone workflow

For each milestone: inspect existing code → present a short plan → implement
with tests → run lint + tests + the real-data pipeline → update this file's
status sections → report honestly, including failures and open questions.
