# GoalSignal — 5-minute demo walkthrough

A guided tour for a first-time reader (recruiter, interviewer, reviewer): what
the project does, what to run, and one concrete end-to-end story — the
"Mexico upset" scenario. Every number below comes from committed code run on
the real dataset; nothing is mocked.

**What GoalSignal is in one sentence:** a leakage-safe probabilistic
forecasting and Monte Carlo tournament simulation system for the 2026 FIFA
World Cup, with an immutable hash-chained prediction ledger and an opt-in
**scenario analysis layer** for stress-testing human tactical opinions.

## 1. Setup (once)

```bash
uv sync
UV_NO_EDITABLE=1 uv run pytest && uv run ruff check .   # 338 tests, lint clean
```

macOS note: prefer `UV_NO_EDITABLE=1 uv run ...` (see AGENTS.md → environment
gotcha).

## 2. The baseline pipeline (already deployed)

The champion model is `ensemble-v1`: a temperature-calibrated convex ensemble
of Dixon-Coles and multinomial-logistic models, trained on ~46k international
matches, evaluated strictly chronologically (backtest log loss 0.8924 on
15,499 matches, 2010–2025). It drives a 100,000-run Monte Carlo simulation of
the real 2026 bracket:

```bash
UV_NO_EDITABLE=1 uv run goalsignal tournament simulate --sims 100000 --seed 20260612
UV_NO_EDITABLE=1 uv run goalsignal tournament advancement   # per-team round-reach probs
UV_NO_EDITABLE=1 uv run goalsignal tournament bracket       # modal knockout path
```

## 3. The scenario analysis layer (the demo)

The model knows historical team strength; it does not know that match 92 is in
Mexico City at altitude, or that England needed a late comeback against
DR Congo. Those opinions live in
[`config/human_adjustments_2026.yaml`](../config/human_adjustments_2026.yaml)
— YAML, not Python — as capped, validated percentage-point adjustments, each
with a required reason:

```yaml
92:
  label: "Mexico vs England"
  adjustments:
    - team: Mexico
      category: venue
      modifier: altitude_boost
      points: 7
      confidence: high
      reason: "Mexico City altitude and home crowd."
```

Apply the **opinion overlay** to an existing simulation, then compare
scenarios:

```bash
UV_NO_EDITABLE=1 uv run goalsignal tournament human-adjust \
    --simulation-dir artifacts/simulations/<run-dir>
UV_NO_EDITABLE=1 uv run goalsignal tournament compare-scenarios \
    --simulation-dir artifacts/simulations/<run-dir>
```

Both commands are read-only over the simulation directory: model
probabilities, the prediction ledger, and all original artifacts are
untouched — outputs are new files.

## 4. Example output (real excerpt)

From `scenario_comparison.md` generated on the real knockout-survival run
(`b1bfd6e3fb69c758.ensemble-knockout_survival.ko-upset`):

> **Headline**
>
> - Champions: Model-only → **Argentina**; Knockout-survival ensemble →
>   **Argentina**; Human-adjusted scenario → **Argentina**
> - Matches changed by the opinion overlay: 3
> - Picks flipped by the opinion overlay: 1
> - Biggest probability mover: M92 Mexico (vs England) 0.392 → 0.532

Per-match comparison (abridged — the report covers all 32 knockout matches):

| M | Stage | Pairing | Model-only | KO-survival | Human scenario | Base p(A) | Scenario p(A) | Net pts | Flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 92 | round_of_16 | Mexico vs England | England | England | **Mexico** | 0.392 | 0.532 | +14.0 | YES |
| 93 | round_of_16 | Portugal vs Spain | Spain | Spain | Spain | 0.348 | 0.468 | +12.0 | |
| 99 | quarterfinal | Brazil vs Mexico | England | England | **Brazil** | 0.590 | 0.590 | | |
| 100 | quarterfinal | Argentina vs Colombia | Argentina | Argentina | Argentina | 0.667 | 0.577 | −9.0 | |
| 104 | final | Spain vs Argentina | Argentina | Argentina | Argentina | 0.498 | 0.498 | | |

(Winner columns show each run's *own* modal pick, so they can name a team
outside the displayed human-scenario pairing — M99 is exactly that: both
simulation runs modally expect England there, while the scenario has
Brazil vs Mexico because the M92 flip propagated.)

Downstream effects are traced exactly (against a recorded no-opinion walk of
the same bracket, not against modal summaries):

> - **M92: England → Mexico** (flipped by the opinion overlay)
>   - M99 (quarterfinal) is now Brazil vs Mexico (was Brazil vs England without the flip); scenario winner Brazil
>   - M102 (semifinal) is now Brazil vs Argentina (was England vs Argentina without the flip); scenario winner Argentina
>   - M103 (third_place) is now France vs Brazil (was France vs England without the flip); scenario winner France
>   - champion unchanged under this scenario

## 5. How to reproduce the Mexico upset scenario

The exact steps behind the excerpt above (~1 minute on the committed
artifacts; regenerate the simulation first if `artifacts/simulations/` is
empty on your machine):

```bash
# 0. (Only if no simulation exists yet — ~a few minutes.)
UV_NO_EDITABLE=1 uv run goalsignal tournament simulate --sims 100000 --seed 20260612

# 1. Apply the opinion overlay from config/human_adjustments_2026.yaml.
#    M92 carries +7 Mexico (venue/altitude), +3 Mexico (form),
#    -4 England (form) => +14 net points, capped at ±15.
UV_NO_EDITABLE=1 uv run goalsignal tournament human-adjust \
    --simulation-dir artifacts/simulations/<run-dir> --force

# 2. Build the three-way comparison report.
UV_NO_EDITABLE=1 uv run goalsignal tournament compare-scenarios \
    --simulation-dir artifacts/simulations/<run-dir> --force
```

What you should see:

1. `human-adjust` prints `M92: Mexico vs England | p(Mexico) 0.392 -> 0.532 |
   winner Mexico (FLIPPED)` — the simulated advance probability plus the
   +14-point opinion delta crosses 0.5, so the scenario pick flips from
   England to Mexico.
2. `compare-scenarios` prints `M92: England -> Mexico; 3 downstream pairing
   change(s)` and writes `scenario_comparison.{md,csv}`,
   `scenario_biggest_movers.csv`, and `scenario_flips.csv` into the run
   directory.
3. Open `scenario_comparison.md`: the flip reshapes M99, M102, and M103
   (Mexico replaces England on that side of the bracket), while the predicted
   champion stays Argentina in all three scenarios.

To test a different opinion, edit `config/human_adjustments_2026.yaml`
(validation rejects unknown teams/categories, missing reasons, and points
beyond the caps) and re-run steps 1–2 with `--force`.

## 6. What this demo is — and is not

Human adjustments are **scenario analysis, not calibrated forecasts**. The
model probabilities remain unchanged; the ledger and original simulation
artifacts are untouched; adjusted probabilities rank one fixed bracket path.
The **scenario analysis layer** exists to make tactical assumptions explicit
and inspect their downstream consequences — not to claim improved accuracy.

## Where to go next

- [README](../README.md) — full command reference and design overview
- [docs/ensemble_signals.md](ensemble_signals.md) — signal layer, ensemble,
  and scenario-analysis design notes
- [docs/research_report.md](research_report.md) — the honest chronological
  evaluation (including negative findings)
- [AGENTS.md](../AGENTS.md) — canonical project status and working agreement
