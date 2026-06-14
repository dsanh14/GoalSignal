# GoalSignal

A leakage-safe probabilistic forecasting and tournament simulation system for
international football. The flagship application is forecasting every match of
the 2026 FIFA World Cup and estimating each team's probability of advancing
through every stage; the underlying research question is how much stable,
out-of-sample predictive signal historical match data actually contains.

## Status

Core research pipeline complete: data foundation → Elo ratings →
chronological backtesting (2010–2025, 15,499 test matches) → Poisson /
Dixon-Coles goal models → calibrated convex ensemble → Monte Carlo group-stage
simulation of the real 2026 fixtures (100k sims) → hash-chained prediction
ledger with 70 immutable World Cup forecasts. Headline: ensemble log loss
0.8924 [0.8745, 0.9108] vs 1.0986 uniform; full findings (including the
honest negative/weak results) in [docs/research_report.md](docs/research_report.md).
Roadmap and open work: [AGENTS.md](AGENTS.md).

An **optional enrichment layer** (StatsBomb events, the API-Sports /
API-Football live API, historical FIFA rankings, player/lineup/rest/travel
features) is being added to test whether richer information beats the
team-level baseline. **Milestones A (contracts) + B (ingestion)** are done: a
host-locked API-Football v3 client with daily-quota throttling, cache-first
replay, immutable raw caching and secret redaction; a StatsBomb offline loader;
a FIFA rankings loader with a leakage-safe as-of join; fixture-linking
preparation; and real coverage reports (`goalsignal sources|api-football|
statsbomb|fifa-rankings`). Live auth is verified (Free plan), though 2026 World
Cup data is plan-locked. A **real-data audit** (2026-06-13) ingested the local
FIFA ranking timeline (1992–2024, rank reconstructed), validated it against the
World Cup `wc_teams.csv`, and audited the Transfermarkt export (read-only,
club-centric) and source readiness — no models trained, ledger untouched. The
baseline model is unchanged, enrichment is off by default, and credentials live
only in a git-ignored `.env`. See [docs/enrichment_coverage.md](docs/enrichment_coverage.md),
[docs/fifa_rankings.md](docs/fifa_rankings.md),
[docs/player_data.md](docs/player_data.md), and
[docs/api_football.md](docs/api_football.md).

## Data

The historical dataset is **provided by the user** and is never downloaded or
replaced by this project. The four CSVs live in [Datasets/](Datasets/):

- `results.csv` — ~49k international matches, 1872–present (scores include
  extra time, exclude penalty shootouts; future fixtures carry `NA` scores)
- `shootouts.csv` — penalty shootout outcomes
- `goalscorers.csv` — goal-level events (partial coverage; absence of a
  scorer row never implies absence of a goal)
- `former_names.csv` — date-bounded historical team-name mappings

A different location can be supplied with `--input-dir`.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Usage

```bash
uv run goalsignal data inspect        # source schemas, row counts, date ranges
uv run goalsignal data validate       # quality checks + audit reports
uv run goalsignal data build          # canonical dataset + reports + manifest
uv run goalsignal ratings build       # Elo timeline + final ratings
uv run goalsignal ratings inspect     # current top-rated teams
uv run goalsignal evaluate rolling    # 2010-2025 expanding-window backtest
uv run goalsignal tournament simulate # 100k-sim 2026 WC group stage
uv run goalsignal predict remaining   # ledger predictions for scheduled fixtures
uv run goalsignal ledger list         # show immutable predictions
uv run goalsignal ledger verify       # verify the hash chain
uv run goalsignal benchmark           # measured simulator performance
```

On macOS, prefer `UV_NO_EDITABLE=1 uv run ...` (see the environment note in
[AGENTS.md](AGENTS.md)).

Outputs:

- `data/processed/matches.csv` — canonical match table
- `artifacts/reports/` — data-quality audits and backtest reports
- `artifacts/ratings/`, `artifacts/simulations/`, `artifacts/benchmarks/`
- `artifacts/predictions/ledger.jsonl` — append-only, hash-chained forecasts
- `artifacts/manifests/` — content-hashed dataset manifests

## Development

```bash
uv run pytest          # unit + integration tests (synthetic fixtures only)
uv run ruff check .    # lint
```

Core principles: no future-data leakage, no silent data mutation, every
exclusion auditable, immutable pre-match predictions, honest negative results.
See [AGENTS.md](AGENTS.md) for the full working agreement.
