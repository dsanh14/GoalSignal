# Tournament Simulation

The simulator overlays every active World Cup result from the verified result
store. Observed scores are fixed exactly once and completed fixtures are
excluded from sampling.

Live runs are written under `artifacts/simulations/<version>/`. The version
includes the result-store hash, model revision, and current FIFA snapshot ID.
Metadata records completed fixture IDs/scores and remaining fixture count; a
mismatched result hash triggers the stale-artifact guard.

## Squad research simulation

`goalsignal tournament simulate-squad` runs the same official group and
M73-M104 bracket machinery with the S7 squad scenario for eligible future
fixtures and exact base fallback otherwise. It writes a separate versioned
research directory and never changes the production simulator, prediction
ledger, or result store.

The simulator can trace a target team's group finish, best-third
qualification, round opponents, expected opponent Elo and squad strength, and
conditional advancement. The Portugal report treats Croatia and Spain as
probabilistic opponents, not fixed bracket assignments.
