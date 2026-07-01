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
| `final_ensemble` | all signals at the default product weights (incl. the opt-in `knockout_upset` for knockout ties) |
| `knockout_survival` | knockout-tuned profile that leans more on `market` and `knockout_upset` |

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

**team_styles.csv** (`knockout_upset`, keyed by `team`): any subset of 0-100
indicators `possession_heavy, low_block_defense, transition_threat,
set_piece_threat, pressing_intensity, chance_creation, sterile_possession_risk,
struggles_vs_low_block, defensive_compactness, attacking_directness,
aerial_threat` (+ free-text `notes`). Each is centred at 50 and mapped to
`[-1, 1]`; missing fields are neutral (no zero-fill of unknown style).

**penalties.csv** (`knockout_upset`, keyed by `team`): current 0-100 ratings
`penalty_strength, keeper_penalty_strength, penalty_taker_depth,
tournament_experience, manager_continuity` and raw shootout records
`shootout_wins/losses, world_cup_shootout_wins/losses,
continental_shootout_wins/losses` (+ `notes`). Records are **shrunk toward
50/50** before use; current keeper/taker ratings are weighted above old country
history. See "Why knockout prediction is different" below.

**matches.csv** (the forecast list): `match_id, stage, team_a, team_b` plus
optional historical columns (`historical_home_win/draw/away_win` for groups,
`historical_team_a_advances/team_b_advances` for knockouts). When the historical
columns are absent that signal is simply missing and the ensemble renormalizes.

## Why knockout prediction is different from group-stage prediction

Group-stage forecasting asks **who is better** over 90 minutes. For knockouts,
GoalSignal models not only who is better, but **who can survive and advance**.
Underdogs with compact defense, low expected-goals matchups, penalty strength,
and favorable style matchups receive a controlled upset-path adjustment.

The `knockout_upset` signal (`signals/knockout_upset.py`) is **knockout-only**
and **opt-in** (`--include-knockout-upset`). For group matches it is never
produced; for knockout matches without the flag it is also absent and the
ensemble renormalizes — so the default path is unchanged.

**Explicit advance model.** A knockout is resolved in stages, so winning in 90
minutes and advancing are not the same thing:

```
P(F advances) = P(F wins in regulation)
              + P(regulation draw) * [ P(F wins ET)
                                      + P(ET draw) * P(F wins shootout) ]
```

Expected goals split the favourite/underdog Poisson means **multiplicatively**
(ratio-preserving), so a low-event, compact matchup scales both means down,
**raises the draw mass**, and routes more of the favourite's edge through the
near-coin-flip extra-time/penalty path — exactly where a survival-minded
underdog gains. (An additive goal margin would wrongly inflate the favourite's
relative edge as goals fall.) Regulation, extra time (one-third intensity), and
the shootout reuse the same staging as `tournament/knockout.py`.

**Anchored, so it never randomly boosts underdogs.** The signal starts from a
base advance estimate (the historical model, else market, else squad, else
50/50), re-derives advance through the staged model **with** and **without** the
style/penalty evidence, and applies only the *difference*. With no style or
penalty data for either side it abstains (returns `None`); with normal,
high-event inputs the difference is tiny. The per-match shift is hard-capped
(`max_advance_shift`, default 0.15) and the blend weight is small (0.05), so the
net move on the final ensemble is modest by construction.

**Penalty/shootout history is a shrunk prior.** Shootout records are tiny
samples, so each is Beta-shrunk toward 50/50 before use:
`(wins + 0.5k) / (wins + losses + k)` with `k = shootout_prior_strength`
(default 6) — a 4–0 record reads as 0.70, a 1–0 record barely moves off 0.5.
Current keeper and penalty-taker strength are weighted **above** old country
history (`current_pen_weight` 0.7 vs `history_pen_weight` 0.3); World Cup
shootouts are weighted above continental and friendly records. The two teams'
ratings give a head-to-head shootout probability whose deviation from 0.5 is
hard-capped (`shootout_cap`, default 0.12) — a strong pedigree yields ~0.56/0.44,
never a deterministic "this country always wins penalties". Because the shootout
only resolves the penalty-path mass, **it only meaningfully moves the advance
probability when the draw/extra-time probability is high**: in a likely blowout
it barely registers.

**Style matchup features** detect upset-prone *shapes* rather than boosting all
underdogs. Each contributes a bounded, explainable adjustment with a provenance
tag:

| Shape | Effect | Provenance tag |
| --- | --- | --- |
| possession-heavy favourite that struggles to break a real low block | shrinks the favourite's regulation edge | `low_block_survival_path` |
| favourite with sterile possession / weak chance creation | suppresses goals and shrinks the edge | `favorite_sterile_possession_risk` |
| underdog with strong transition threat | shrinks the favourite's edge | `transition_threat` |
| underdog with set-piece / aerial threat | shrinks the favourite's edge | `set_piece_underdog_path` |
| underdog low block / compactness | suppresses expected goals → more penalties | (raises draw mass) |
| underdog with shootout/keeper/taker edge in a draw-likely tie | tilts the penalty path | `penalty_path_boost` |

All coefficients live under `signal_params.knockout_upset` in
`config/ensemble.yaml` and are modest, bounded priors — **not fitted** to match
results (no leakage). The matchup types this targets: a stronger favourite upset
on penalties, a compact side surviving a possession-heavy favourite, and ties
where extra time and penalties are a genuine path to advance.

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
may also carry a `stage` label. `squad_strength`, `recent_form`, `team_styles`,
and `penalties` are already team-keyed, so they attach to any matchup without
extra columns.

**Full lookup precedence** used for a (possibly dynamically generated) knockout
pairing, highest priority first:

1. `match_id`, when available;
2. normalized team pair **+ stage** (forward orientation), when a row carries one;
3. normalized team pair (forward, then reverse with a flip);
4. team-level features (`squad_strength`, `recent_form`, `team_styles`,
   `penalties`), which are keyed by team and always resolve.

`match_id` always wins over a team-pair hit. The `knockout_upset` signal consumes
team-level style/penalty tables (step 4) and is anchored on the best available
advance estimate (historical → market → squad → 50/50), so it attaches to
dynamic knockout pairings that have no `match_id`.

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

# Knockout "survive and advance" layer (opt-in; knockout matches only):
goalsignal signals predict --matches data/manual/matches.example.csv \
    --include-knockout-upset
goalsignal tournament simulate --prediction-source ensemble \
    --ensemble-version knockout_survival --include-knockout-upset \
    --sims 100000 --seed 20260612

# Comparison report across the three simulation sources:
goalsignal evaluate simulation-comparison           # reads existing artifacts
goalsignal evaluate simulation-comparison --live    # live-model matchup diagnostics
```

## Simulation comparison report

`evaluate simulation-comparison`
(`evaluation/simulation_comparison.py`) is **read-only** over existing
simulation artifacts — it never re-runs the simulator or overwrites a run. It
auto-discovers the newest `baseline` (historical), `final_ensemble`, and
`knockout_survival` runs under `artifacts/simulations/` (override with
`--baseline`/`--final-ensemble`/`--knockout-survival`) and writes four artifacts
to `artifacts/ensemble/`:

| Artifact | Contents |
| --- | --- |
| `simulation_comparison.csv` | per-team semifinal/final/champion probs for each run + pairwise deltas |
| `biggest_movers.csv` | the largest absolute moves (`team, stage, comparison, from_prob, to_prob, delta, abs_delta`) |
| `knockout_survival_explanations.csv` | per-matchup before/after advance with the `knockout_upset` decomposition |
| `simulation_comparison.md` | the honest narrative report |

The report answers: which teams' champion/semifinal/final probabilities moved
most; which knockout matchups shifted and **how much of that came from
`knockout_upset`** (the `net_move_from_upset` column, separate from the version
change); which ties were flagged high-disagreement; and which signals were
missing or illustrative. The matchup diagnostics use the matches CSV
(`--matches`, default `data/manual/knockout_matchups.example.csv`) for the
baseline advance, or the trained model with `--live`.

If a run is missing, its comparisons are omitted and the report says so — the
command never crashes on a partial set. It deliberately makes **no accuracy
claim**: it shows what moved and why, not that the movement is correct.

## Status: production-grade vs experimental

**Production-grade** (validated, stable, safe to rely on):

- signal validation and coverage reporting (`signals validate`);
- typed probability objects (`OutcomeProbs` / `AdvanceProbs`) that always
  normalize and never go negative;
- missing-signal renormalization in the meta-ensemble;
- opt-in ensemble simulation that leaves the historical default byte-for-byte
  unchanged and writes to distinct artifact directories;
- the test suite and full-simulator invariant checks.

**Experimental** (use as a controlled, explainable nudge — not evidence):

- the `knockout_upset` survival coefficients;
- the penalty/shootout priors;
- the style-matchup coefficients;
- the absence of a chronological knockout backtest;
- manual/example data coverage (illustrative fixtures, not real data).

**Do not claim:**

- that the knockout survival layer **improves accuracy** — it has not been
  backtested out-of-sample;
- that penalty/shootout history is **highly predictive** — it is shrunk toward
  50/50, capped, and only matters when a tie is likely to reach penalties;
- that Croatia-style (or any) teams are **guaranteed to win penalties** — the
  shootout edge is a small bounded nudge, never a deterministic rule.

`--include-knockout-upset` adds the signal to knockout ties only; group fixtures
and the default historical path are unaffected. Ensemble runs with the flag
write to a distinct `*.ensemble-<version>.ko-upset` artifact directory so they
never overwrite the canonical historical simulation.

## Assumptions and limitations

- Adjustment→distribution uses a fixed Davidson map with config-driven scaling;
  the `points_per_z` / coefficient defaults are reasonable priors, **not fitted**
  (deliberately, to avoid circularity and leakage). Fit them against the
  ensemble champion on chronological folds before treating any challenger as
  deployment-grade.
- The knockout reduction (`advance_from_outcome`) is a closed-form simplification
  of the staged simulator in `tournament/knockout.py`.
- **`knockout_upset` is experimental.** Its coefficients are bounded priors, not
  fitted to results; the staged advance model uses a calibrated expected-goals
  fallback rather than per-team xG; and it has not yet been validated on a
  chronological knockout backtest (knockout shootout outcomes are rare, so a
  clean out-of-sample evaluation needs care). Treat it as a controlled,
  explainable nudge, not a tuned model — it is intentionally weighted at 0.05.
- The example CSVs are illustrative fixtures, not real data.
