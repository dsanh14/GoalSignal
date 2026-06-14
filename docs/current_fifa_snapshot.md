# Current FIFA Snapshot

`FIFA_CURRENT_RANKINGS_PATH` points to the frozen June 11, 2026 World Cup field
snapshot (`group, team, fifa_rank`). It is separate from the historical FIFA
timeline and `wc_teams.csv`; it is never merged into chronological backtests.

The BOM-safe loader requires 48 unique teams in groups A-L with four teams per
group and validates canonical mappings. The snapshot is unavailable before its
release date. It is diagnostic only: results update GoalSignal Elo, never FIFA
rank or model weights.
