# Provider Migration: football-data.org → API-Sports / API-Football

## What happened

The live football provider was initially implemented against **football-data.org**
based on the `FOOTBALL_DATA_API_KEY` env-var name. That was incorrect: the
user's key belongs to **API-Sports / API-Football**. The two providers have
different hosts, auth headers, response schemas, and endpoints, so the
integration was migrated wholesale.

## Wrong vs correct contract

| | Wrong (removed) | Correct (current) |
| --- | --- | --- |
| Vendor | football-data.org | **API-Sports** |
| Base URL | `https://api.football-data.org/v4` | `https://v3.football.api-sports.io` |
| Auth header | `X-Auth-Token` | `x-apisports-key` |
| Envelope | bare resource JSON | `{get, parameters, errors, results, paging, response}` |
| Errors | HTTP status codes | also HTTP 200 with `errors` field |
| Cache dir | `data/external/football_data/` | `data/external/api_football/` |
| CLI group | `goalsignal football-data` | `goalsignal api-football` |

The env-var name `FOOTBALL_DATA_API_KEY` was **kept** (the user's `.env` uses
it); only its meaning changed (it is an API-Sports key).

## Why the earlier "invalid token" was misleading

The football-data.org probe returned HTTP 400 "Your API token is invalid." That
was the wrong host rejecting an API-Sports key — **not** a bad key. After
migration, the API-Sports `/status` probe with `x-apisports-key` returned HTTP
200 (plan: Free, quota 100/day). So the token is valid; only the provider was
wrong.

## What was removed / deprecated

- Removed: `src/goalsignal/data/sources/football_data.py`,
  `football_data_normalize.py`, `config/football_data.yaml`,
  `tests/live/test_football_data_live.py`, the `football-data` CLI group, and
  `FootballDataConfig`.
- Deprecated (preserved as audit evidence):
  `artifacts/reports/football_data_probe.DEPRECATED.json` (marked
  `wrong_provider_diagnostic`), and `docs/football_data_api.md` (now a
  deprecation redirect).
- Security: the redaction set masks **both** `x-apisports-key` and the legacy
  `X-Auth-Token`, and the client is host-locked, so a stale config can never
  send the key to football-data.org.

## New verified facts

- Auth: OK (API-Sports `/status`, plan Free).
- Quota: 100/day (`x-ratelimit-requests-limit`), 10/min (`x-ratelimit-limit`).
- World Cup: discovered league_id **1** ("World Cup", "World", "Cup"),
  seasons include 2026.
- Coverage: 2026 match data plan-locked on Free; World Cup injuries unsupported.
  See `artifacts/reports/api_football_coverage.json` and
  [api_football.md](api_football.md).
