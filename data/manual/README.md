# Manual signal inputs

Drop hand-maintained signal files here. Every file is **optional** and every
column within a file is optional — GoalSignal renormalizes the ensemble across
whichever signals are present, so a sparse file (or no file at all) still
produces a forecast.

The `*.example.csv` files document the schema and are safe to copy. Real inputs
should drop the `.example` suffix (e.g. `market_odds.csv`). Nothing in this
directory is required by the base statistical pipeline.

| File | Signal | Key | Notes |
| --- | --- | --- | --- |
| `market_odds.csv` | `market` | `match_id` | decimal odds; `draw_odds` blank ⇒ 2-way knockout market |
| `squad_strength.csv` | `squad_strength` | `team` | any subset of value/minutes/depth indicators |
| `recent_form.csv` | `recent_form` | `team` | **opponent-adjusted** form, not raw results |
| `venue_context.csv` | `venue_context` | `match_id` | host/travel/rest/climate, all per-match |
| `expert_predictions.csv` | `expert` | `match_id` | structured LLM/expert probabilities + reasoning |
| `team_styles.csv` | `knockout_upset` | `team` | 0-100 style indicators (low block, sterile possession, transition, set pieces…) |
| `penalties.csv` | `knockout_upset` | `team` | penalty/keeper ratings + shootout records (shrunk toward 50/50) |

`team_styles.csv` and `penalties.csv` feed the **knockout-only** "survive and
advance" signal — opt-in via `--include-knockout-upset`, applied to knockout
matches only. See [docs/ensemble_signals.md](../../docs/ensemble_signals.md)
("Why knockout prediction is different from group-stage prediction").

`knockout_matchups.example.csv` is not a signal file: it is a forecast list of
knockout ties (`match_id, stage, team_a, team_b` + optional
`historical_team_a_advances/team_b_advances`) used by
`goalsignal evaluate simulation-comparison` for before/after matchup diagnostics.

Validate coverage at any time with:

```bash
UV_NO_EDITABLE=1 uv run goalsignal signals validate
```
