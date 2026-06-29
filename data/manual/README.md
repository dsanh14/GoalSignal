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

Validate coverage at any time with:

```bash
UV_NO_EDITABLE=1 uv run goalsignal signals validate
```
