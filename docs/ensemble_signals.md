# Outcome-first signals and the meta-ensemble

GoalSignal's product output is calibrated **win/advance probabilities**. This
document describes the external-signal layer (`src/goalsignal/signals/`), the
configurable meta-ensemble that blends signals, the manual file schemas, and the
outcome-first evaluation utilities.

## Design

A **signal** is any information source wrapped to emit the same probability
object:

- `OutcomeProbs(home_win, draw, away_win)` for group-stage matches
  (label order `[home, draw, away] = [0, 1, 2]`, consistent with the rest of the
  codebase).
- `AdvanceProbs(team_a_advances, team_b_advances)` for knockout ties.

Both normalize on construction and validate non-negativity. `team_a` is the home
team for group matches; for knockout ties the labels are positional only.

There are two kinds of signal:

- **Distribution signals** (`historical`, `market`, `expert`) directly provide a
  probability vector.
- **Adjustment signals** (`squad_strength`, `recent_form`, `venue_context`)
  produce a scalar Elo-like *advantage* from whichever fields are present, then
  map it to a distribution via the **Davidson model**
  (`davidson_outcome`, the same family as the Elo-Davidson baseline). This lets
  adjustments participate in the same probability space as distributions.

Knockout reduction: an adjustment signal's `OutcomeProbs` is converted to
`AdvanceProbs` with `advance_from_outcome`, which assigns the draw mass to the
two teams by `knockout_tiebreak_a_prob` (0.5 ⇒ penalties are a coin flip). The
full staged regulation/ET/shootout model remains in `tournament/knockout.py`.

## Meta-ensemble

`MetaEnsemble` (`signals/meta_ensemble.py`) is a weighted average of signal
distributions (a linear opinion pool — output stays a valid probability vector,
order-invariant). Product rules enforced here:

1. **Configurable weights** from `config/ensemble.yaml` (`default_weights` and
   named `model_versions`). Nothing is hardcoded in the model layer.
2. **Renormalization on missing signals**: a signal with no value for a match is
   dropped and the remaining weights renormalize. If *every* weighted signal is
   missing, the blend raises (there is nothing to forecast from).

Each `BlendResult` carries full provenance: `probs`, `used_weights`
(renormalized), `missing`, `components`, and `max_pairwise_disagreement` (the
largest total-variation distance between any two available signals, used as a
review flag via `is_flagged`).

### Model versions

| Version | Intent |
| --- | --- |
| `baseline_historical` | historical signal only (the deployed champion) |
| `market_only` | pure market benchmark |
| `squad_form_challenger` | historical + squad + form |
| `llm_adjusted_challenger` | historical + expert |
| `final_ensemble` | all six signals at the default product weights |

## Manual file schemas

All files live in `data/manual/`, are optional, and tolerate missing columns.
A real `*.csv` overrides the bundled `*.example.csv`.

**market_odds.csv** (`market`): `match_id, source, team_a_odds, draw_odds,
team_b_odds, timestamp`. Blank `draw_odds` ⇒ two-way knockout market. Decimal
odds; the overround is removed (`proportional` default, or favourite-longshot-
correcting `power`).

**squad_strength.csv** (`squad_strength`, keyed by `team`): any subset of
`total_squad_value, starting_xi_value, top5_league_minutes,
champions_league_minutes, club_minutes_30d, club_minutes_90d, keeper_strength,
attacking_depth, defensive_depth, missing_stars, suspensions, avg_age`. Counts
(`missing_stars`, `suspensions`) are penalties; `avg_age` is informational.
Indicators are standardized across the loaded teams, then weighted.

**recent_form.csv** (`recent_form`, keyed by `team`): any subset of
`elo_adj_last5, elo_adj_last10, gf_adj, ga_adj, xg_diff`. These are expected to
be **opponent-adjusted** already; `ga_adj` is a penalty.

**venue_context.csv** (`venue_context`, keyed by `match_id`): any subset of
`host_boost, crowd_advantage, travel_km_a, travel_km_b, rest_days_a,
rest_days_b, heat_disadvantage_a, timezone_shift_a, timezone_shift_b`. All from
team A's perspective; coefficients are configurable in `config/ensemble.yaml`.

**expert_predictions.csv** (`expert`, keyed by `match_id`, multiple rows
allowed): `source_model, team_a_win_prob, draw_prob, team_b_win_prob,
team_a_advance_prob, team_b_advance_prob, confidence, reasoning`. A row may carry
the group triple, the knockout pair, or both; triples/pairs must sum to ~1
(validated, then renormalized). Sources are combined by confidence-weighted
consensus.

**matches.csv** (the forecast list): `match_id, stage, team_a, team_b` plus
optional historical columns (`historical_home_win/draw/away_win` for groups,
`historical_team_a_advances/team_b_advances` for knockouts). When the historical
columns are absent that signal is simply missing and the ensemble renormalizes.

## Evaluation

`evaluation/metrics.py` provides the canonical 3-way metrics (log loss, Brier,
RPS, ECE, reliability, block-bootstrap CI). `evaluation/outcome_eval.py` adds:

- `calibration_table` — per-outcome binned predicted-vs-empirical frequency.
- `binary_log_loss` / `binary_brier` / `binary_calibration_table` /
  `binary_summary` for advance probabilities.
- `compare` / `format_comparison` — a ranked summary table across models for
  backtesting baseline vs market-only vs challengers vs final ensemble.

Accuracy is reported only as a secondary metric.

## Wiring into prediction, simulation, and backtest

The signal layer is integrated into the real workflow as an **opt-in** path;
the deployed historical pipeline and ledger are untouched.

- **Historical adapter** (`signals/historical_adapter.py`): `LiveModelHistorical`
  converts the trained `LiveModel` into signal types — group W/D/L from the
  calibrated `predict_outcome`, knockout advancement from the goal model's
  regulation/extra-time/penalty resolution. No model logic is duplicated. Every
  value carries provenance (`live_model` / `fixture` / `unavailable`); a
  prediction that cannot be produced is returned missing, and the ensemble
  renormalizes. `UnavailableHistorical` is the null provider for sample-only runs.

- **Prediction API** (`signals/api.py`): `EnsemblePredictor` is the single
  internal interface — `predict_match_ensemble`, `predict_knockout_ensemble`,
  `predict_batch_ensemble`. Each `EnsemblePrediction` carries the final
  probabilities, version, components used, renormalized weights, missing signals,
  disagreement score, flagged status, and the historical source.

- **Tournament source** (`tournament/ensemble_adapter.py`): `EnsembleGoalAdapter`
  reweights the goal model's score matrix so its W/D/L marginals equal the
  ensemble's `OutcomeProbs` (keeping the within-region scoreline shape so GD/GF
  tiebreakers still work), and exposes `advance_probs` so knockout resolution
  matches the ensemble's `AdvanceProbs`. The existing vectorized simulator runs
  unchanged; the historical default path is byte-for-byte identical (a plain
  `RatingsGoalAdapter` has no `advance_probs`, so the resolution code falls back
  to the goal model). Select it with `--prediction-source ensemble`.

- **Ensemble backtest** (`evaluation/ensemble_backtest.py`): fixed-weight,
  leakage-safe comparison of versions on identical matches. Runs on the bundled
  sample (flagged a smoke test) **or** on real historical predictions via
  `--predictions artifacts/reports/backtest/test_predictions.csv` — those
  `ensemble_*` columns are the deployed model's out-of-sample test outputs, so
  reusing them adds no leakage. Writes four artifacts to `artifacts/ensemble/`:
  `backtest_comparison.csv`, `backtest_summary.md` (the no-overclaim
  "is-it-better?" verdict — see below), `calibration_by_version.csv`,
  `coverage_by_signal.csv`. The verdict is **INSUFFICIENT** whenever
  non-historical coverage is near zero (the honest outcome with sample-only
  manual data, where `final_ensemble ≈ baseline_historical`).

- **Ablation** (`evaluate ensemble-ablation`): historical-only vs historical +
  each signal group vs the full ensemble, each at its configured weight; writes
  `ablation_comparison.csv` + `ablation_summary.md` (log-loss delta vs
  historical-only per signal group).

- **Weight tuning** (`signals/tuning.py`): **validation-only** (never the test
  fold); writes `artifacts/ensemble/tuned_weights.yaml` and `tuning_report.md`
  with before/after validation metrics and a **low-coverage warning** when the
  data is too sparse to tune reliably. **Never** mutates `config/ensemble.yaml`.

## Dynamic signal keying (knockouts)

Knockout pairings are generated during simulation, so a manual signal keyed only
by `match_id` never attaches to them. `signals/keying.py` adds a minimal
team-pair fallback. A market/expert/venue row may carry `team_a`,`team_b` (names)
and is then resolvable by the normalized pair. **Precedence (highest first):**

1. exact `match_id`,
2. forward team-pair (fixture lists the teams in the row's order),
3. reverse team-pair (fixture lists them swapped) — the directional
   probabilities are flipped (`OutcomeProbs.flip` / `AdvanceProbs.flip`; venue
   advantage is negated).

Team names are normalized (trimmed, casefolded, whitespace-collapsed). Venue rows
may also carry a `stage` label. `squad_strength` and `recent_form` are already
team-keyed, so they attach to any matchup without extra columns.

## CLI

```bash
goalsignal signals validate        # coverage + parse warnings for manual files
goalsignal signals market          # implied + vig-removed market probabilities
goalsignal signals predict         # ensemble predictions via the public API
goalsignal signals blend           # blend signals per match under a version
goalsignal signals disagreement    # TVD of each signal vs a reference signal
goalsignal signals tune-weights    # validation-only weight tuning -> artifact + report
goalsignal evaluate ensemble-backtest          # compare versions (--predictions for real data)
goalsignal evaluate ensemble-ablation          # which signals actually help
goalsignal tournament simulate --prediction-source ensemble   # opt-in ensemble sim
```

## Assumptions and limitations

- Adjustment→distribution uses a fixed Davidson map with config-driven scaling;
  the `points_per_z` / coefficient defaults are reasonable priors, **not fitted**
  (deliberately, to avoid circularity and leakage). Fit them against the
  ensemble champion on chronological folds before treating any challenger as
  deployment-grade.
- The knockout reduction (`advance_from_outcome`) is a closed-form simplification
  of the staged simulator in `tournament/knockout.py`.
- The example CSVs are illustrative fixtures, not real data.
