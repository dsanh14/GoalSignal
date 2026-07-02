# GoalSignal

GoalSignal is a World Cup 2026 forecasting engine focused on **win and
advancement probabilities, not exact scorelines**. It blends historical team
strength, market-implied probabilities, squad quality, recent form, venue
context, and LLM/expert consensus into calibrated match and bracket
predictions.

The product outputs are:

- **Group stage** — `home_win` / `draw` / `away_win` probabilities.
- **Knockout stage** — `team_a_advances` / `team_b_advances` probabilities.
- **Tournament** — each team's probability of reaching each round and lifting
  the trophy, via Monte Carlo simulation over those calibrated probabilities.

Everything is optimized and evaluated for **outcome probability quality** — log
loss, Brier score, and calibration — with accuracy reported only as a secondary
metric.

**New to the project?** Start with the
[5-minute demo walkthrough](docs/demo_walkthrough.md) — it runs the pipeline,
applies the human opinion overlay, and reproduces the "Mexico upset" scenario
end to end, with real example output.

## Why not exact scores?

Exact-score prediction is a harder problem than the question we actually care
about ("who advances?"), and optimizing for scoreline accuracy does not
optimize for calibrated outcome probabilities. GoalSignal keeps its
goal-scoring models (Poisson, Dixon–Coles) — they remain a useful **sub-signal
and baseline** — but they are no longer the product. The forecasting target is
the outcome distribution, scored by proper scoring rules.

## Model signals

Each information source is wrapped as a uniform **signal** that emits the same
small probability object (`OutcomeProbs` for groups, `AdvanceProbs` for
knockouts), so they can be blended, compared, and renormalized:

| Signal | Source | Module |
| --- | --- | --- |
| `historical` | the leakage-safe Elo / Poisson / Dixon–Coles ensemble | `models/`, `evaluation/` |
| `market` | bookmaker decimal odds, vig removed | `signals/market.py` |
| `squad_strength` | manual squad-quality file | `signals/squad_strength.py` |
| `recent_form` | opponent-adjusted recent form | `signals/recent_form.py` |
| `expert` | structured LLM/analyst predictions | `signals/expert.py` |
| `venue_context` | host/travel/rest/climate context | `signals/venue_context.py` |
| `knockout_upset` | knockout-only "survive and advance" adjustment (opt-in) | `signals/knockout_upset.py` |

The `historical` signal is GoalSignal's original, fully-validated statistical
pipeline: data foundation → Elo ratings → chronological backtesting
(2010–2025, 15,499 test matches) → calibrated convex ensemble, with a
hash-chained, immutable prediction ledger. Headline backtest: ensemble log loss
**0.8924 [0.8745, 0.9108]** vs 1.0986 uniform. Full findings (including honest
negative results) are in [docs/research_report.md](docs/research_report.md);
roadmap and working agreement in [AGENTS.md](AGENTS.md).

## Ensemble design

The meta-ensemble ([signals/meta_ensemble.py](src/goalsignal/signals/meta_ensemble.py))
is a configurable **linear opinion pool**: a weighted average of the signal
distributions. Two product rules are enforced here, never hardcoded in the
model layer:

1. **Configurable weights** — they live in
   [config/ensemble.yaml](config/ensemble.yaml), not in code. Default product
   weights:

   | Signal | Weight |
   | --- | --- |
   | historical | 35% |
   | market | 25% |
   | squad_strength | 15% |
   | recent_form | 10% |
   | expert | 10% |
   | venue_context | 5% |

2. **Renormalization on missing signals** — when a signal has no value for a
   match, its weight is dropped and the remaining weights are renormalized, so
   partial coverage still yields a proper distribution.

Several named **model versions** support champion/challenger backtesting:
`baseline_historical`, `market_only`, `squad_form_challenger`,
`llm_adjusted_challenger`, `final_ensemble`, and the knockout-tuned
`knockout_survival`. The blend records full provenance (which signals were used,
their renormalized weights, what was missing, and the maximum pairwise
disagreement) so every probability is reproducible from its parts.

## Why knockout prediction is different from group-stage prediction

Group-stage forecasting asks **who is better** over 90 minutes. For knockouts,
GoalSignal models not only who is better, but **who can survive and advance**.
Underdogs with compact defense, low expected-goals matchups, penalty strength,
and favorable style matchups receive a controlled upset-path adjustment.

The opt-in `knockout_upset` signal ([signals/knockout_upset.py](src/goalsignal/signals/knockout_upset.py))
models advancement explicitly — `P(advance) = P(win in regulation) + P(draw) ·
P(win in ET/penalties)` — because a team can be worse over 90 minutes yet still
have a real path through a draw and the shootout. It:

- **only applies to knockout matches** and only when you pass
  `--include-knockout-upset` (group fixtures and the default historical path are
  untouched);
- is **anchored** to a base advance estimate and abstains with no style/penalty
  evidence, so it never randomly boosts underdogs;
- treats a low expected-goals, compact matchup as raising the draw/penalty path
  (ratio-preserving goal split), which is where survival-minded underdogs gain;
- uses **penalty/shootout history as a shrunk prior** — small samples pulled
  toward 50/50, current keeper/taker weighted above old country history, and a
  hard cap so a strong pedigree nudges (≈54/46–56/44), never dictates;
- is **modest and bounded** — a hard-capped per-match shift at a 0.05 blend
  weight, with explainable provenance tags (`low_block_survival_path`,
  `favorite_sterile_possession_risk`, `penalty_path_boost`,
  `set_piece_underdog_path`, `transition_threat`).

```bash
# Knockout survival layer on the example data:
uv run goalsignal signals predict \
    --matches data/manual/matches.example.csv --include-knockout-upset

# The three comparable tournament simulations (historical / final / survival):
uv run goalsignal tournament simulate --sims 100000 --seed 20260612
uv run goalsignal tournament simulate --prediction-source ensemble \
    --ensemble-version final_ensemble --sims 100000 --seed 20260612
uv run goalsignal tournament simulate --prediction-source ensemble \
    --ensemble-version knockout_survival --include-knockout-upset \
    --sims 100000 --seed 20260612

# Compare them: writes a Markdown report + CSVs to artifacts/ensemble/
uv run goalsignal evaluate simulation-comparison        # reads existing artifacts
uv run goalsignal evaluate simulation-comparison --live # live-model matchup diagnostics
```

The comparison report answers *which teams' champion/semifinal/final
probabilities moved, which knockout matchups shifted, and how much came from
`knockout_upset`* — see `artifacts/ensemble/biggest_movers.csv` and
`knockout_survival_explanations.csv`. It is **read-only** over existing
simulation artifacts and makes **no accuracy claim**.

It is **experimental**: the coefficients are bounded priors (not fitted), it uses
a calibrated expected-goals fallback rather than per-team xG, and it has not yet
been validated on a chronological knockout backtest. Penalty/shootout history is
shrunk toward 50/50 and capped — no team is assumed to win shootouts. See
[docs/ensemble_signals.md](docs/ensemble_signals.md) for the file schemas, the
full lookup precedence, the comparison report, and the production-grade vs
experimental status.

## Market odds support

`signals/market.py` converts decimal odds to implied probabilities, removes the
bookmaker overround (`proportional` or favourite-longshot-correcting `power`),
and exposes
normalized market probabilities. A blank `draw_odds` marks a two-way knockout
market. The market layer is usable three ways: as a **standalone benchmark**, a
weighted **feature** in the ensemble, and a **disagreement detector** against
the internal model. Missing files and bad rows degrade gracefully.

## Squad / form / venue adjustments

These are file-first, manually maintainable signals, each robust to missing
fields (no zero-filling of unknowns):

- **Squad strength** — squad/XI value, top-5 & Champions League minutes, recent
  club minutes, keeper/attack/defense depth, missing stars, suspensions, age.
- **Recent form** — *opponent-adjusted* (not raw) Elo-weighted performance,
  opponent-adjusted goals for/against, xG differential, with competitive and
  major-tournament matches weighted above friendlies.
- **Venue context** — host-country boost (USA/Mexico/Canada), travel distance,
  rest days, heat/altitude proxy, time-zone change, crowd advantage.

Each converts an available-fields strength edge into a W/D/L distribution via a
Davidson map, so adjustments live in the same probability space as the other
signals. (An in-pipeline, leakage-safe form/venue feature builder also exists at
[features/d1.py](src/goalsignal/features/d1.py).)

## LLM / expert judgment layer

`signals/expert.py` accepts structured predictions (`match_id`, `source_model`,
group triple, knockout pair, `confidence`, `reasoning`). The LLM never silently
overwrites probabilities — it is one **bounded ensemble signal**, a
**disagreement detector**, and an **explanation** source. Probabilities are
validated to sum to ~1; multiple sources per match are combined into a
confidence-weighted consensus.

## Backtesting and calibration

Evaluation is outcome-first ([evaluation/metrics.py](src/goalsignal/evaluation/metrics.py),
[evaluation/outcome_eval.py](src/goalsignal/evaluation/outcome_eval.py)):

- multiclass **log loss** (primary), **Brier**, RPS;
- per-class and binary (advance) **calibration tables** and ECE;
- a `compare()` summary table to rank the baseline, market-only, challengers,
  and final ensemble on identical chronological folds;
- accuracy only as a secondary metric.

The historical signal is backtested with a strict expanding-window protocol
(`goalsignal evaluate rolling`); calibrators and ensemble weights are fit on
validation predictions only.

> **macOS:** prefix every command below with `UV_NO_EDITABLE=1` (e.g.
> `UV_NO_EDITABLE=1 uv run goalsignal ...`). A background process on this machine
> re-hides the editable-install `.pth` file, so the bare `uv run goalsignal`
> form intermittently fails with `ModuleNotFoundError: No module named
> 'goalsignal'`. The non-editable form is reliable. See [AGENTS.md](AGENTS.md).

## How to run predictions

```bash
uv sync

# Validate the manual input files and report coverage:
uv run goalsignal signals validate

# Ensemble match predictions through the public API.
#   --no-live (default): historical signal comes from the matches CSV (fixtures)
#   --live: historical signal comes from the trained live model (real data)
uv run goalsignal signals predict --matches data/manual/matches.example.csv
uv run goalsignal signals predict --live --out artifacts/signals/predictions.csv

# Blend signals under a named version (provenance + disagreement per match):
uv run goalsignal signals blend --version final_ensemble \
    --matches data/manual/matches.example.csv --out artifacts/signals/blended.csv
uv run goalsignal signals blend --version market_only
uv run goalsignal signals blend --version squad_form_challenger

# Inspect the market signal directly (implied + vig-removed):
uv run goalsignal signals market --csv data/manual/market_odds.example.csv

# Where does each signal disagree with the historical model?
uv run goalsignal signals disagreement --reference historical
```

The original statistical pipeline still runs end-to-end and produces the
`historical` signal and the immutable ledger:

```bash
uv run goalsignal data build --force
uv run goalsignal ratings build
uv run goalsignal evaluate rolling --start-year 2010 --end-year 2025
uv run goalsignal predict remaining
uv run goalsignal ledger verify
```

## How to run chronological backtesting

Compare ensemble versions on identical group-stage matches with fixed
(untuned) weights.

```bash
# Smoke test (bundled sample, clearly flagged as illustrative):
uv run goalsignal evaluate ensemble-backtest

# Full run on the real expanding-window predictions (uses ensemble_* as the
# historical signal; leakage-safe because those are out-of-sample test rows):
uv run goalsignal evaluate ensemble-backtest \
    --predictions artifacts/reports/backtest/test_predictions.csv
```

Writes four artifacts to `artifacts/ensemble/`:

- `backtest_comparison.csv` — log loss / Brier / RPS / ECE / accuracy plus signal
  coverage, missing-signal rate, and high-disagreement-bucket performance;
- `backtest_summary.md` — the **"is final_ensemble better?"** report (log loss,
  Brier, calibration, trusted vs experimental signals, verdict + recommendation;
  it says *insufficient* outright when coverage is too low to conclude);
- `calibration_by_version.csv` — per-class reliability bins per version;
- `coverage_by_signal.csv` — per-signal coverage and trust status.

### Ablation study

Which signals actually help? Compare historical-only vs historical + each signal
group vs the full ensemble (writes `ablation_comparison.csv` + `ablation_summary.md`):

```bash
uv run goalsignal evaluate ensemble-ablation \
    --predictions artifacts/reports/backtest/test_predictions.csv
```

## Tuning weights (validation only)

Tune ensemble weights on a **validation split only** (never the test fold) and
write them to a **separate** artifact — `config/ensemble.yaml` is never mutated.
A low-coverage warning is emitted (and recorded) when the data is too sparse to
tune reliably. Writes `tuned_weights.yaml` + `tuning_report.md`:

```bash
uv run goalsignal signals tune-weights --objective log_loss \
    --predictions <validation-split>.csv --out-dir artifacts/ensemble
```

## How to run tournament simulation

The Monte Carlo simulator forecasts the real 2026 fixtures through to champion
using the official FIFA bracket and tiebreakers (`tournament/`,
`config/tournament_2026.yaml`). Choose the probability source:

```bash
# Default: the historical goal model (unchanged behaviour).
uv run goalsignal tournament simulate --sims 100000 --seed 20260612

# Opt-in: drive the simulator from the blended meta-ensemble. The historical
# signal comes from the live model; squad/form signals apply per team; market/
# expert/venue apply where a manual match_id matches. Prints a provenance
# summary (coverage, missing signals, high-disagreement ties) and writes to a
# distinct artifact version so historical runs are never overwritten.
uv run goalsignal tournament simulate --prediction-source ensemble \
    --ensemble-version final_ensemble --sims 100000 --seed 20260612

uv run goalsignal tournament advancement     # per-team round-reach probabilities
uv run goalsignal tournament bracket
```

Both sources output round-of-32 / round-of-16 / quarterfinal / semifinal /
final / champion probabilities for every team.

## Human-adjusted scenario analysis

A **scenario analysis layer** for stress-testing tactical opinions on top of an
existing simulation — an **opinion overlay**, not a calibrated forecast.

- **Opinions live in YAML, not Python**
  ([config/human_adjustments_2026.yaml](config/human_adjustments_2026.yaml)):
  per-match, per-team percentage-point adjustments with a required `reason`
  and optional `confidence`, so views can change without touching code and
  every claim is auditable.
- **Caps and strict validation prevent uncontrolled edits**: per-adjustment
  and per-match point caps, probability clipping, a fixed category/modifier
  taxonomy, and hard errors on unknown teams, missing reasons, or
  out-of-range points.
- **Flips propagate through the official bracket graph**: each match's
  adjusted winner feeds the real M73–M104 advancement slots, so one flipped
  pick visibly reshapes the downstream pairings (and the report traces
  exactly which ones, against a recorded no-opinion walk).
- **Nothing else changes**: the layer is read-only over the simulation
  directory — model probabilities, the prediction ledger, and all original
  simulation artifacts are untouched; outputs are new files.

```bash
# Apply the opinion overlay to an existing run (writes
# human_adjusted_bracket.{csv,md} + meta into the simulation dir):
uv run goalsignal tournament human-adjust \
    --simulation-dir artifacts/simulations/<run-dir> \
    --config config/human_adjustments_2026.yaml

# Compare model-only vs knockout-survival vs the human-adjusted scenario
# (writes scenario_comparison.{md,csv}, scenario_biggest_movers.csv,
# scenario_flips.csv; missing scenarios are reported, not fatal):
uv run goalsignal tournament compare-scenarios \
    --simulation-dir artifacts/simulations/<run-dir>
```

**Interpreting the caveats.** Adjusted probabilities rank one fixed bracket
path under stated opinions; they are scenario analysis, not calibrated
forecasts, and no accuracy improvement is claimed. Use the reports to make
assumptions explicit and inspect their downstream consequences.

## How to add manual data files

Drop CSVs into [data/manual/](data/manual/) (see its README and the
`*.example.csv` schemas). Every file and every column is optional — strip the
`.example` suffix to activate a real file (e.g. `market_odds.csv`). See
[docs/ensemble_signals.md](docs/ensemble_signals.md) for the full schema and
design notes.

| File | Signal | Key |
| --- | --- | --- |
| `market_odds.csv` | market | `match_id` and/or `team_a`,`team_b` |
| `squad_strength.csv` | squad_strength | `team` |
| `recent_form.csv` | recent_form | `team` |
| `venue_context.csv` | venue_context | `match_id` and/or `team_a`,`team_b`,`stage` |
| `expert_predictions.csv` | expert | `match_id` and/or `team_a`,`team_b` |
| `matches.csv` | the match list to forecast | `match_id` |

**Dynamic knockout pairings.** Knockout opponents are decided during simulation,
so a row keyed only by `match_id` can't attach to them. Add `team_a`,`team_b`
columns (names) and the row also resolves by **normalized team pair**.
Precedence: exact `match_id` first, then a forward team-pair match, then a
reverse match (the directional probabilities are flipped automatically). So one
`market_odds.csv` row with `team_a=Spain,team_b=Germany` attaches to a
Spain–Germany tie whichever way the bracket presents it. Venue rows may also
carry a `stage` label.

**Reading the reports.** Backtest/ablation/tuning write Markdown + CSV to
`artifacts/ensemble/` — start with `backtest_summary.md` (the "is the ensemble
better?" verdict) and `ablation_summary.md` (which signals help).

## Data

The historical dataset is **provided by the user** and is never downloaded or
replaced by this project. The four CSVs live in [Datasets/](Datasets/):

- `results.csv` — ~49k international matches, 1872–present (scores include extra
  time, exclude penalty shootouts; future fixtures carry `NA` scores)
- `shootouts.csv` — penalty shootout outcomes
- `goalscorers.csv` — goal-level events (partial coverage)
- `former_names.csv` — date-bounded historical team-name mappings

A different location can be supplied with `--input-dir`. An **optional
enrichment layer** (StatsBomb, API-Football, FIFA rankings, Transfermarkt
player/lineup data) is off by default; see [AGENTS.md](AGENTS.md).

## Development

```bash
uv run pytest          # unit + integration tests (synthetic fixtures only)
uv run ruff check .    # lint
```

Core principles: no future-data leakage, no silent data mutation, every
exclusion auditable, immutable pre-match predictions, honest negative results,
and baselines before complexity. See [AGENTS.md](AGENTS.md) for the full working
agreement.
