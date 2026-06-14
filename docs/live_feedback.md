# Live Feedback

Completed results are appended to `artifacts/results/results.jsonl`, a separate
hash chain. Corrections append a superseding entry; history is never rewritten.
Date-only results retain date precision and do not invent kickoff times.

The live update added Qatar 1-1 Switzerland, Brazil 1-1 Morocco, Haiti 0-1
Scotland, and Australia 2-0 Turkey. The last result was supplied in reverse
wording but resolves to the canonical Australia-v-Turkey fixture.

Feedback scores the original frozen forecast. W/D/L correctness, exact-score
correctness, top-k coverage, goal error, log loss, Brier score, and RPS remain
separate. Exact-score probability is reconstructed only after strict frozen
model validation.
