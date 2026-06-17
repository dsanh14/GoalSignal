# Squad Scenario Challenger

The 2026 squad challenger is an offline, scenario-based sensitivity analysis.
It is not trained on match outcomes and is not deployed. The production
`ensemble-v1+r10` model remains the champion and supplies the complete base
forecast.

The challenger standardizes supported squad activity, starts, historical
valuation, positional activity, goalkeeper activity, and depth proxies across
the current 48-team field. Variants S1-S6 expose individual feature families;
S7 combines them, shrinks the result by coverage confidence, clips the
adjustment, and applies it as a small opponent-relative expected-goal shift.

Teams that fail any configured coverage threshold receive exactly zero squad
adjustment. Missing data is not treated as zero and does not create a negative
penalty. This run adjusted 20 teams and used the base fallback for 28.

The configuration is
`config/squad_challenger_2026.yaml`; the implementation is
`src/goalsignal/tournament/squad_challenger.py`. Deployment would require
chronological historical reconstruction, comparison against the deployed
ensemble on identical folds, calibration, and a predeclared promotion rule.

