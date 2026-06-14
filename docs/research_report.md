# GoalSignal Research Report

Generated 2026-06-12 from real backtests on the user-provided dataset
(snapshot through 2026-06-11; dataset version in
`artifacts/manifests/`). All numbers below were produced by executed code
(`goalsignal evaluate rolling`, `goalsignal tournament simulate`,
`goalsignal benchmark`) and are reproducible from the committed configuration
and seeds.

## Protocol

Expanding-window chronological backtest, test years 2010–2025 (16 folds,
15,499 labeled test matches). Per fold: components fit on matches before the
3-year validation window; temperature calibration and convex ensemble weights
fit on validation predictions only; test predictions generated once. No
random splits, no tuning on test periods. Features are deliberately minimal
in v1: pre-match Elo (K=20, home advantage 60, importance-weighted), venue
neutrality. Uncertainty: year-block bootstrap, 1,000 resamples, 90% CIs.

## Headline results (pooled 2010–2025, log loss, 90% CI)

| Model | Log loss | Brier | ECE | Accuracy |
| --- | --- | --- | --- | --- |
| Ensemble (calibrated) | **0.8924** [0.8745, 0.9108] | 0.5252 | 0.008 | 0.590 |
| Multinomial logistic | 0.8933 [0.8753, 0.9121] | 0.5259 | 0.017 | 0.590 |
| Dixon-Coles | 0.8940 [0.8760, 0.9133] | 0.5263 | 0.012 | 0.589 |
| Independent Poisson | 0.8953 [0.8774, 0.9147] | 0.5269 | 0.016 | 0.589 |
| Elo-Davidson | 0.9036 [0.8874, 0.9209] | 0.5311 | 0.031 | 0.588 |
| Higher-rated heuristic | 0.9579 | 0.5673 | — | 0.577 |
| Context frequency | 1.0479 | 0.6313 | — | 0.478 |
| Uniform | 1.0986 | 0.6667 | 0.145 | 0.478 |

By competition (ensemble): qualification is most predictable (LL 0.812,
acc 0.647, n=6,108); friendlies noisier (0.954, n=4,865); World Cup finals
hardest (0.993, n=256 — small sample, wide uncertainty).

Goal models (fold-averaged, strict 90-minute-eligible matches): total-goal
MAE 1.52, exact-score top-1 12.5%, top-3 coverage 34%, scoreline NLL 2.978.

## Hypothesis outcomes

- **H3 (goal dependence / Dixon-Coles)** — *weakly supported.* Median fitted
  rho = −0.037 (negative low-score dependence, consistent with the
  literature). Outcome log loss improves 0.8953 → 0.8940 and scoreline NLL
  2.9786 → 2.9784 vs independent Poisson: directionally consistent across
  folds but small, and CIs overlap heavily.
- **H7 (signal combination)** — *weakly supported.* The calibrated ensemble
  beats its best component by ~0.001 log loss. Notably the optimizer
  assigns Elo-Davidson ~0 weight in recent folds: its signal is subsumed by
  the multinomial logistic built on the same Elo features. Combination
  helps only when components carry distinct information.
- **H8 (complexity vs calibration)** — *supported in v1's regime.* A
  4-parameter softmax regression on Elo features is within noise of the goal
  models for outcome prediction. Well-specified simple models are hard to
  beat once calibrated; the next material gain must come from new
  *information* (form, venue specifics, squads), not new model classes.
- **Calibration** — components were already near-calibrated (fitted
  temperatures 0.87–1.09); ensemble ECE 0.008. Temperature scaling is cheap
  insurance rather than a large win here.
- **H1 (recency), H2 (opponent-adjusted form), H4 (regimes), H5
  (importance), H6 (venue/travel), H9 (historical depth), H10 (parameter
  stability)** — *not yet tested.* These need the form/venue feature sets
  and the ablation harness; recorded as open work, not assumed.

## 2026 World Cup forecast (group stage)

100,000 Monte Carlo simulations (seed 20260612, data cutoff 2026-06-12,
i.e. after the two opening matches). Full table:
`artifacts/simulations/wc2026_group_stage.csv`; per-fixture probabilities in
the hash-chained ledger (`goalsignal ledger list`). Group labels are
synthetic (derived from the fixture graph); the official Round-of-32
bracket mapping is not in the dataset, so knockout-stage probabilities
beyond R32 qualification are intentionally not produced rather than
fabricated.

Top advance probabilities (P reach R32, MC s.e. ≤ 0.001): Mexico 0.995
(host, opener won), Spain 0.993, Argentina 0.981, England 0.967,
South Korea 0.967, Canada 0.963, France 0.959, Brazil 0.952.

## Performance engineering

Group-stage simulators (Apple Silicon, CPU only, measured by
`goalsignal benchmark`, median of 3 runs at 20,000 sims): reference
~11,100 sims/s; vectorized ~101,500 sims/s (**9.1× measured speedup**). The
vectorized path falls back to the exact tiebreak procedure only for
simulations containing unresolved (points, GD, GF) ties, so it is
correctness-equivalent by construction; agreement is tested.

## Limitations

1. Features are Elo + venue only; no form, travel, altitude, head-to-head,
   or squad information yet.
2. Scores in knockout matches decided in extra time are indistinguishable
   from 90-minute scores in the source (documented scope policy).
3. Shootout advancement uses a 50/50 baseline.
4. World Cup-specific evaluation rests on 256 matches; conclusions about
   tournament-specific skill are weak by construction.
5. Kickoff times are unknown (dates only); predictions are
   regulation-time probabilities.
