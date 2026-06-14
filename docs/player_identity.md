# Player and Club Identity

Canonical player and club entities, so the same person/club from different
sources resolves to one identity without ever merging distinct people.

## Identity rule: never merge on name alone

`resolve_player(query, candidates, source)` matches in strict precedence:

1. **Source ID** — an exact match on the source's own player ID is unambiguous.
2. **Name + date of birth** — normalized name plus matching DOB.
3. **Name + nationality + club** — normalized name plus both corroborators.

Anything weaker (name-only) is returned as **ambiguous** with a review status,
never matched. A name with no candidate is **unmatched**. Both are reported,
never silently resolved:

- `artifacts/reports/player_identity_conflicts.csv`
- `artifacts/reports/player_unmatched.csv`
- `artifacts/reports/player_source_coverage.csv`

Name normalization casefolds and strips accents for comparison only; the
display name is preserved separately.

## Canonical entities

`PlayerIdentity` carries: `canonical_player_id`, full and normalized name,
optional date of birth (only where legally available), nationality, position,
club, effective date range, per-source IDs, and a `review_status`. `ClubIdentity`
is analogous. Curated alias tables live in tracked, human-reviewed files:

- `data/reference/player_aliases.csv`
- `data/reference/club_aliases.csv`

(created in Milestone C; tracked because they are curated evidence, not
generated output).

## Effective dates

Aliases and club affiliations carry effective start/end dates so a player's
club as of a given match is resolved correctly, and historical names are not
applied to the wrong period.

## Status

Milestone A: `PlayerIdentity`/`ClubIdentity` schemas, normalization helpers, and
the pure `resolve_player` matcher. Batch resolution and alias-store I/O are
Milestone C.
