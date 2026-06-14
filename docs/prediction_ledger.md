# Prediction Ledger

Forecasts are immutable and hash-chained. A live result state creates a new
revision such as `ensemble-v1+r6`; only scheduled fixtures receive it. Payloads
record result-store hash/count, Elo state hash, feature-set version, and source
snapshot IDs.

`predictions scores` shows the latest revision per fixture by default.
`--show-revisions` exposes all history and `--model-version` filters one exact
revision. Completed fixtures are never re-predicted.
