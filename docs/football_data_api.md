# football-data.org — DEPRECATED (wrong provider)

**This provider was a mistake and has been removed.** An earlier pass wired the
live source to football-data.org (`https://api.football-data.org/v4`, header
`X-Auth-Token`). The user's API key actually belongs to **API-Sports /
API-Football**, not football-data.org.

What this means:

- The football-data.org client, config (`config/football_data.yaml`), CLI group
  (`goalsignal football-data ...`), and normalization module were **removed** so
  the API-Sports key can never be sent to the wrong host.
- The earlier football-data.org probe returned HTTP 400 "Your API token is
  invalid." That was a **wrong-provider** result, **not** evidence of an invalid
  API-Sports token. The API-Sports `/status` probe with `x-apisports-key`
  succeeds. The old diagnostic is preserved (marked deprecated) at
  `artifacts/reports/football_data_probe.DEPRECATED.json`.

**Use [api_football.md](api_football.md) instead.** Migration details:
[api_football_migration.md](api_football_migration.md).
