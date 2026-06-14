# Model Card

## Deployed champion — `ensemble-v1` (UNCHANGED)

Temperature-calibrated convex ensemble of Elo-Davidson, Dixon-Coles, and a
multinomial-logistic outcome model. Backtest (2010-2025, 15,499 matches):
log loss 0.8924 [0.8745, 0.9108]. Used for the 70 immutable 2026 World Cup
predictions in the ledger. **Not modified by Milestone D1.**

## D1 challengers (OFFLINE research only — NOT deployed)

Multinomial-logistic outcome model extended with leakage-safe D1 feature
families (FIFA, FIFA-Elo disagreement, recent form, attack/defense, rest,
venue). Evaluated by chronological ablation against an internal Elo-only
logistic baseline (D1-0, log loss 0.8975 on 13,266 test matches, 2010-2023).

| Challenger | Features added | Δ log loss vs baseline | Verdict |
| --- | --- | --- | --- |
| D1-D | attack/defense form | -0.0098 [-0.012, -0.007] | supported |
| D1-C | recent form | -0.0050 | supported |
| D1-A | FIFA rank/points | -0.0027 | supported (small) |
| D1-B | FIFA-Elo disagreement | -0.0006 | no measurable difference |
| D1-F | basic venue | -0.0001 | no measurable difference |
| D1-G | all D1 | -0.0127 | supported (best) |
| native-no-FIFA | form+attack/def+rest+venue | -0.0120 | supported |

- **Inputs:** pre-match Elo, FIFA timeline (≤2024), rolling form/attack/defense
  residuals, rest, venue. Each carries explicit availability indicators.
- **Calibration:** temperature scaling, fit on the validation window per fold.
- **Reproducibility:** feature version `d1.1`, config hash recorded in
  `d1_champion_challenger.json`; seeds in `config/experiments_d1.yaml`.
- **Periods:** train < (test_year - 3); validation = 3y window; calibration =
  validation; test = each year 2010-2023.

## Limitations & deployment status

- D1 gains are measured against the **internal logistic baseline**, not the
  deployed ensemble. **No challenger is deployed.** Recommendation: advance
  native form + attack/defense to a deployment-grade evaluation against the
  ensemble champion.
- FIFA data ends 2024-09-19; 2026 fixtures get no FIFA values (native fallback).
- Player/lineup/StatsBomb/live-API features are out of scope here (blocked or
  deferred per the source-readiness audit).
