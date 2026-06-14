# Data Quality and Score-Scope Policy

Generated audit reports live in `artifacts/reports/` after
`goalsignal data validate` or `data build`. Nothing is ever silently dropped:
the canonical match count plus the exclusion ledger always accounts for every
raw row.

## Score-scope semantics

The source records full-time scores that **include extra time but exclude
penalty shootouts**. Consequences:

- A shootout row proves the match was level when play ended, so the
  **regulation outcome is a known draw**, but the exact 90-minute score is
  unknown (extra-time goals may be included, and some competitions go
  straight to penalties). These matches get
  `recorded_score_scope = regulation_or_extra_time_unknown` and
  `strict_goal_model_eligible = False`.
- Knockout matches **decided in extra time** (no shootout) are
  indistinguishable from 90-minute results in this source. By policy they
  keep `recorded_score_scope = regulation`, and decisive results in
  knockout-capable tournaments (configured in `config/data.yaml`) are flagged
  in `suspicious_scope_matches.csv` with reason
  `possible_extra_time_decisive_knockout_capable` as an **upper bound** on
  contamination. This is a documented limitation, not a claim of purity.
- Qualification competitions are excluded from the knockout-capable flag:
  their rare two-legged play-off extra-time cases cannot be separated from
  tens of thousands of ordinary group matches.

## Findings on the 2026-06-12 snapshot

- **49,477 raw rows → 49,476 canonical matches** (49,406 played + 70
  scheduled 2026 World Cup fixtures) + **1 exclusion**: an exact duplicate of
  1974-02-17 Tahiti vs New Caledonia (source row 9644).
- **Duplicate identity flag**: two 2026-06-06 Gibraltar vs Cayman Islands
  friendlies with different venue fields — kept, flagged for review.
- **Shootouts**: 640 reconcile cleanly to a tied result; **37 join to a
  decisive recorded score** (`matched_score_not_tied`) — predominantly
  two-legged ties where the shootout followed a decisive second leg with a
  tied aggregate; **1 unmatched** (2011 Saare County vs Åland Islands, no
  corresponding result row). All flagged, none dropped; the 37 inconsistent
  joins keep `shootout_played = True` but `regulation_outcome = unknown`.
- **Scope counts**: 48,729 regulation / 677 regulation-or-extra-time-unknown.
  Strict 90-minute goal-model eligible: 48,729.
- **Suspicious scope**: 4,259 flags, including 78 Olympic Games matches
  (present despite the dataset description claiming exclusion — retained,
  flagged with `tournament_term:olympic`) and possible-extra-time decisive
  knockout-capable matches.
- **Goalscorers**: all 47,606 events join to played matches; 84 duplicate
  events (same match, team, scorer, minute) flagged. Coverage by year is in
  `goalscorer_coverage.csv`; absence of scorer rows is reported as missing
  coverage, never as zero goals.
- **Former names**: 36 mappings, no inverted periods, no overlapping or
  chained conflicts. Mappings are applied date-aware and one step only.

## Reports

| Report | Contents |
| --- | --- |
| `data_quality.json` / `.md` | Summary statistics and policy notes |
| `excluded_matches.csv` | Every excluded row: source, reason, severity, review status |
| `duplicate_matches.csv` | Same (date, home, away) with different identity — kept and flagged |
| `suspicious_scope_matches.csv` | Scope-violation and possible-extra-time flags |
| `shootout_reconciliation.csv` | Join status for every shootout row |
| `goalscorer_coverage.csv` | Per-year scorer-event coverage vs recorded goals |
| `former_name_conflicts.csv` | Name-mapping issues |
