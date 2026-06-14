# D1 Ablation Experiments

Research question: do FIFA, FIFA-Elo disagreement, recent form, attack/defense
form, rest/congestion, and basic venue context improve leakage-safe
chronological out-of-sample forecasting beyond the Elo baseline?

## Design

- **Challenger:** the multinomial-logistic outcome model (same family as the
  deployed one) extended with configurable feature columns. The deployed
  `ensemble-v1` champion is **not** modified.
- **Protocol:** expanding-window folds, test years 2010-2023 (kept inside FIFA
  coverage). Continuous features median-imputed + standardized using **train-fold
  statistics only**; temperature calibration on the validation window; one
  scored pass per test year.
- **Pairing:** all experiments use identical folds and evaluate on **identical
  test matches** (rows never dropped; missing values imputed with explicit
  availability indicators), so baseline-vs-challenger is paired.
- **Uncertainty:** paired year-block bootstrap (1000 resamples, 90% CI) on the
  delta log loss. Verdict: `supported_improvement` (CI < 0), `degradation`
  (CI > 0), or `no_measurable_difference` / `weak_evidence`.

Config: `config/experiments_d1.yaml`. Run: `goalsignal evaluate d1-ablation`.

## Results (13,266 test matches, 2010-2023; lower log loss better)

Baseline D1-0 (Elo-only logistic) log loss **0.8975**. Delta < 0 = improvement.

| Experiment | Δ log loss | 90% CI | Verdict |
| --- | --- | --- | --- |
| D1-G (all D1) | -0.0127 | [-0.0157, -0.0093] | supported |
| native-no-FIFA | -0.0120 | supported |
| form + attack/def | -0.0118 | supported |
| D1-D attack/defense | -0.0098 | [-0.012, -0.007] | supported |
| D1-C recent form | -0.0050 | supported |
| D1-A FIFA rank/points | -0.0027 | supported (small) |
| FIFA + disagreement | -0.0021 | supported (small) |
| D1-E rest | -0.0009 | supported (tiny) |
| rest + venue | -0.0009 | supported (tiny) |
| D1-B FIFA-Elo disagreement | -0.0006 | **no measurable difference** |
| D1-F venue | -0.0001 | **no measurable difference** |

(Exact numbers in `artifacts/reports/d1_ablation_results.{csv,md}`; per-fold in
`d1_fold_results.csv`.)

## Findings

1. **Attack/defense and recent form drive the gains** — the largest,
   fold-stable, uncertainty-supported improvements.
2. **FIFA rank/points add a small supported gain**, but most of it overlaps
   with Elo: native-no-FIFA (-0.0120) ≈ all-D1 (-0.0127).
3. **FIFA-Elo disagreement and venue add nothing measurable** (CIs cross zero).
4. Rest gives a tiny supported gain.
5. Gains are stable across folds.
6. **Fallback:** FIFA is unavailable after 2024, so 2026 fixtures carry no FIFA
   values; the native-no-FIFA challenger scores them (dry-run only,
   `d1_fallback_2026.json`).

## Regime analysis (exploratory)

`d1_regime_analysis.csv` (multiple-comparison caution): the challenger helps
most on neutral (-0.022), competitive (-0.015), and FIFA-absent (-0.026)
matches — native form fills in where FIFA is missing. Not a deployment claim.

## Recommendation

**Offline evidence only — do NOT deploy from this milestone.** The gain is
measured against the internal logistic baseline, not the deployed ensemble
champion. Advance the **native form + attack/defense** feature set to a
deployment-grade evaluation against `ensemble-v1`. FIFA, disagreement, and venue
are not worth carrying on their own evidence here.
