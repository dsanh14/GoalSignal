# Player Data — Temporal Semantics (leakage hazards)

This is the load-bearing leakage document for Transfermarkt player data. Full
per-field classification:
`artifacts/reports/transfermarkt_temporal_field_audit.md`.

## Classifications

- **static_identity** — fixed attributes (player_id, name, date_of_birth,
  position, height). Safe at any time.
- **dated_observation** — carries its own date; safe **only with an explicit
  pre-match cutoff** (use rows strictly before the prediction time). Examples:
  `appearances.{date, minutes_played, goals, assists, cards}`,
  `game_lineups.{date, type, position}`, `player_valuations.{date,
  market_value_in_eur}`, `games.date`.
- **current_state_unsafe** — reflects "now", not the match date. **Never apply
  to a historical match.** Examples: `players.current_club_id`,
  `players.current_national_team_id`, `players.international_caps`,
  `players.market_value_in_eur`, `players.highest_market_value_in_eur`,
  `players.contract_expiration_date`, `clubs.total_market_value`,
  `national_teams.fifa_ranking`, `national_teams.squad_size`.
- **unclear_temporal** — semantics not yet established; treat as unsafe until
  confirmed.

## Concrete hazards (do NOT do these)

- Using `players.international_caps` (current total) for a 2018 match — it
  includes caps earned after 2018. Reconstruct caps as of the match date by
  counting dated international appearances instead (and note Transfermarkt's
  international coverage is sparse).
- Using `players.current_club_id` for a 2014 match — the player may have moved
  clubs since. Use dated `appearances.player_club_id` instead.
- Using `players.market_value_in_eur` (current) historically — use the latest
  **dated** `player_valuations` row strictly before the match.
- Using `national_teams.fifa_ranking` (current) — use the historical FIFA
  timeline with the leakage-safe as-of join (see `docs/fifa_rankings.md`).

## Safe pattern

For any player feature at match date T: build it only from
`*.date < T` rows of dated tables; flag missing coverage explicitly (never
zero-fill); never read a `current_*` field. Imputers, if any, fit on training
folds only.
