# D1 Enrichment Features (definitions)

Leakage-safe features built from the canonical match table, the pre-match Elo
timeline, and the historical FIFA ranking timeline. Config:
`config/features_native.yaml`, `config/features_fifa.yaml`. Feature version
`d1.1`. Build with `goalsignal features build-d1`; table at
`artifacts/features/d1/<version>/features.csv` with `feature_schema.json`.

**Same-day policy:** a team's rolling features use only its matches strictly
earlier in `(date, source_row)` order; FIFA uses only releases strictly before
the fixture date.

## FIFA (Phase 2-3)

As-of join to the latest FIFA release strictly before the fixture. Columns:
`home/away_fifa_points`, `home/away_fifa_rank`, `fifa_points_diff`,
`fifa_rank_diff`, `home/away_days_since_fifa_release`, `fifa_release_age_days`,
`home/away_fifa_points_change_{1,3}`, `home/away_fifa_rank_change_{1,3}`
(vs the team's *previous* releases only), `fifa_available`, `fifa_stale`.

- **Staleness:** `fifa_stale=1` when the chosen release is older than
  `stale_after_days` (400). **Availability cap** `unavailable_after_days`
  (450): older than that → `fifa_available=0` and all FIFA columns NaN.
- **Coverage ends 2024-09-19.** The earliest 2026 match is ~471 days later, so
  **every 2026 fixture is FIFA-unavailable** — no 2024→2026 forward-fill. Verified
  by `features validate-d1` and the fallback dry-run.
- **FIFA-Elo disagreement:** `favorite_disagree` (1 when the Elo favorite ≠ the
  FIFA favorite, sign-based, no scaling) and `abs_elo_fifa_disagreement`
  (raw; standardization is fold-local in the model, never global).

## Recent form (Phase 4)

Per team, over windows `[3,5,10]` matches (and 365-day, recency-weighted):
`ppm_last{3,5,10}` (points/match), `winrate_last10`, `gf/ga/gd_per_match_last10`,
`clean_sheet_rate_last10`, `fts_rate_last10` (failure to score),
`gd_volatility_last10`, `recency_wpoints` (exp decay, half-life 365d),
`ppm_comp_last10` (competitive matches only — friendlies excluded; friendlies
are **not** assumed equal). `form_available` indicates ≥1 prior match.

## Attack / defense (Phase 5)

Opponent-adjusted residuals vs a **fixed** expected-goals mapping of the
pre-match Elo expected score `E` (no fitting, no fold dependence, no
circularity): `expected_for = base_total · w(E)`, `expected_against =
base_total · w(1-E)` with `w(E)=E^γ/(E^γ+(1-E)^γ)`, `base_total=2.6`, `γ=1`.

- `attack_resid_last10 = mean(goals_for - expected_for)` over prior matches.
- `defense_resid_last10 = mean(goals_against - expected_against)`.
- `opp_adj_points_last10 = mean(S - E)` where `S∈{1,0.5,0}` is the actual score
  and `E` the Elo expected score (over/under-performance vs Elo).

## Rest / congestion (Phase 6)

`home/away_days_since_prev`, `rest_days_diff`, `home/away_matches_prev_{7,14,30}d`,
`home/away_long_inactivity` (gap > 365d), `days_since_prev_competitive`,
`tournament_seq`. No previous match → NaN + indicators (never zero rest).
`days_since_prev` clipped at 1825 days.

## Venue (Phase 7, canonical fields only)

`is_neutral`, `home_at_home_country`, `away_at_home_country`, `host_indicator`,
`venue_known`. No travel/timezone/altitude/weather (deferred — no reliable
location data ingested).

## Missingness (Phase 9)

Indicator columns (`*_available`, `*_disagree`, `is_*`, `*_inactivity`,
`host_indicator`, …) are kept as 0/1 and never imputed. Continuous columns are
median-imputed and standardized **fold-locally** (train statistics only) in the
ablation runner. Missing enrichment is never silently zeroed before adding its
indicator.
