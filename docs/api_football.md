# API-Sports / API-Football (v3)

The live football data provider. **This replaces an earlier, incorrect
integration against football-data.org** (see "Provider correction" below).

## Identity and authentication

- Vendor: **API-Sports**; product: **API-Football v3** (direct access, not
  RapidAPI).
- Base URL: `https://v3.football.api-sports.io`
- Auth header: **`x-apisports-key`**, value from `$FOOTBALL_DATA_API_KEY` (an
  API-Sports key, despite the historical env-var name). Read only from the
  git-ignored `.env`; never logged, returned, cached, hashed, or written to a
  manifest. The client is **host-locked**: the key is only ever sent to the
  configured API-Sports host.

## Response envelope

Every response is a JSON envelope:
`{"get", "parameters", "errors", "results", "paging", "response"}`. Logical
failures (bad/missing key, exhausted quota, plan limits) arrive as **HTTP 200
with a non-empty `errors` field**, so `errors` is always inspected and mapped to
typed exceptions: `AuthError`, `RateLimitError`, `PlanLimitationError`,
`ProviderLogicError`.

## Free plan and rate limiting (measured)

The user's dashboard shows a **Free plan, 100 requests/day**. Verified live via
`/status` and response headers:

- `x-ratelimit-requests-limit: 100`, `x-ratelimit-requests-remaining` (daily)
- `x-ratelimit-limit: 10`, `x-ratelimit-remaining` (per minute)

Protection (config in `config/api_football.yaml`): `daily_request_limit: 100`,
`daily_request_reserve: 10` (so 90 usable), `max_requests_per_minute: 8`,
`cache_first: true`. A persistent per-UTC-day counter
(`data/external/api_football/usage/<date>.json`) stops requests before the
reserve; cache-first replays a prior identical response with **no live call and
no quota use**; `--refresh` forces a fresh call.

## Caching and offline replay

Raw responses cache immutably (content-addressed) under
`data/external/api_football/raw/<snapshot_id>/` (`request.json` without
headers, `response.json`, `manifest.json`), plus a `request_index.json`
mapping (endpoint, params) → snapshot for cache-first. List with
`goalsignal api-football inspect-cache`.

## Commands

```bash
goalsignal api-football probe                 # one /status call: verify auth, read quota
goalsignal api-football discover-world-cup    # find the WC league id via /leagues (no guessing)
goalsignal api-football fixtures   --league 1 --season 2026
goalsignal api-football standings  --league 1 --season 2026
goalsignal api-football lineups    --fixture <id>
goalsignal api-football injuries   --league 1 --season 2026
goalsignal api-football fixture-players --fixture <id>
goalsignal api-football inspect-cache
```

## World Cup discovery (no guessing)

`discover-world-cup` queries `/leagues?search=World Cup` and selects the FIFA
World Cup. Discovered and stored in config with provenance:

- **league_id = 1**, name "World Cup", country "World", type "Cup".
- Seasons listed in metadata: 2010, 2014, 2018, 2022, **2026**.

## Coverage audit (measured 2026-06-13, Free plan)

Auth and discovery succeed, but **2026-season match data is plan-locked** on the
Free plan (`"Free plans do not have access to this season"`). The World Cup
competition's 2026 coverage flags additionally show **no injuries coverage**.
Classification (in `artifacts/reports/api_football_coverage.json`):

| Field | State |
| --- | --- |
| status, leagues (search + metadata) | supported_and_populated |
| fixtures, standings, teams, squads, lineups, fixture events/statistics/players, predictions | supported_but_unavailable_under_free_plan |
| injuries | unsupported_for_competition (World Cup `cov_injuries=false`) |
| head-to-head | not_yet_tested |

**Consequences (honest):** the Free plan **cannot** supply live 2026 World Cup
fixtures, lineups, or player stats — a paid API-Sports plan would be required.
Injuries are unavailable for the World Cup regardless of plan and are **never
fabricated**.

## Provider predictions policy

API-Football's own `/predictions` are normalized **only as an external
benchmark** (`usage="external_benchmark"`) and are never mixed into GoalSignal
training features.

## Provider correction (the earlier mistake)

An earlier pass integrated **football-data.org** (`api.football-data.org/v4`,
header `X-Auth-Token`). That was the wrong provider: the key is an API-Sports
key. The earlier probe's HTTP 400 "invalid token" was a *wrong-provider*
artifact, **not** evidence of a bad API-Sports key — the API-Sports `/status`
probe with `x-apisports-key` succeeds. The old diagnostic is preserved as
`artifacts/reports/football_data_probe.DEPRECATED.json`; the old provider's
module, config, and CLI group were removed so the key can never be sent to the
wrong host. See `docs/api_football_migration.md`.

## Status

Migration complete: client, cache, normalization, CLI, tests (offline via fake
transport), and a verified live `/status` probe + World Cup discovery. Live
match ingestion is gated by the Free-plan season lock above. Live tests are
marked `@pytest.mark.live_api` and excluded from the default suite.
