# Player and Club Identity

Canonical player and club entities, so the same person/club from different
sources resolves to one identity without ever merging distinct people.

## Identity rule: never merge on name alone

`resolve_player(query, candidates, source)` matches in strict precedence:

1. **Source ID** — an exact match on the source's own player ID is unambiguous.
2. **Name + date of birth** — normalized name plus matching DOB.
3. **Name + nationality + club** — normalized name plus both corroborators.
4. **Name + nationality + position** — deterministic only with supporting
   squad evidence.
5. **Manual review**.

Anything weaker (name-only) is returned as **ambiguous** with a review status,
never matched. A name with no candidate is **unmatched**. Both are reported,
never silently resolved:

- `artifacts/reports/player_identity_conflicts.csv`
- `artifacts/reports/player_unmatched.csv`
- `artifacts/reports/player_source_coverage.csv`

Name normalization casefolds, strips accents for comparison, handles Unicode
punctuation and `Last, First` ordering; the raw display name is preserved.
Reviewed aliases always take precedence. Name-only candidates are never
accepted.

## Canonical entities

`PlayerIdentity` carries: `canonical_player_id`, full and normalized name,
optional date of birth (only where legally available), nationality, position,
club, effective date range, per-source IDs, and a `review_status`. `ClubIdentity`
is analogous. Curated alias tables live in tracked, human-reviewed files:

- `data/reference/world_cup_2026_player_aliases.csv`
- `data/reference/club_aliases.csv`

Alias files are created only when reviewed mappings are needed. Unreviewed
fuzzy candidates are reports, never accepted aliases.

## Effective dates

Aliases and club affiliations carry effective start/end dates so a player's
club as of a given match is resolved correctly, and historical names are not
applied to the wrong period.

## Squad Linkage Outputs

`goalsignal squads link-players` classifies every official squad row as exact,
high-confidence deterministic, ambiguous, unmatched, or conflicting. Reports
are written to `artifacts/reports/squad_player_*`.

For the real 2026 snapshot, independent revalidation accepted 332 seed links.
The reviewed alias pass adds 234 local identities and 63 accepted web-only
identities. Final coverage is 1,233/1,248 (98.8%), with 1,170 locally linkable
players and 15 material conflicts (13 DOB, 2 local-name disagreements).
Web-only identities retain canonical IDs but never receive fabricated local
activity or valuations.
