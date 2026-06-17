# 2026 Squad Features

The deterministic team table is stored at
`artifacts/features/squad_2026/<version>/team_squad_features.csv`. Each row
records the prediction cutoff, config and source hashes, feature version,
coverage status, and fallback eligibility.

Supported feature families:

- activity coverage, minutes, starts, active-player counts, and recency
- goalkeeper, defender, midfielder, and forward activity
- cutoff-safe historical valuation totals, medians, top-11/15/23 summaries,
  minutes-weighted value, age, and staleness
- top-11/15/23 activity and bench-depth proxies
- identity, local snapshot, valuation, position, and goalkeeper missingness

Fixture research artifacts contain standardized home-minus-away differences
for these families. Normalization uses only the current 48-team World Cup
field. Top-11 activity is a descriptive proxy, not an expected lineup.
Current undated profile values, future activity, future valuations, confirmed
lineups, and inferred starters are excluded.

Coverage eligibility requires identity 92%, local activity 75%, valuation
50%, goalkeeper 66%, each position group 70%, and maximum missingness 25%.
Valuations older than 365 days contribute to the freshness confidence term.

