# Tournament Simulation

The simulator overlays every active World Cup result from the verified result
store. Observed scores are fixed exactly once and completed fixtures are
excluded from sampling.

Live runs are written under `artifacts/simulations/<version>/`. The version
includes the result-store hash, model revision, and current FIFA snapshot ID.
Metadata records completed fixture IDs/scores and remaining fixture count; a
mismatched result hash triggers the stale-artifact guard.
