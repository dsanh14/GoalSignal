# FIFA World Cup 2026 knockout mapping

GoalSignal's knockout bracket is configuration-driven and uses only official
FIFA documents. Projected team names are never stored in the bracket.

## Official sources

1. **Regulations for the FIFA World Cup 2026, May 2026.** Articles 12.6-12.11
   define matches 73-104 and their winner/loser edges. Annexe C contains all
   495 combinations for assigning the eight qualifying third-place teams.
2. **FIFA World Cup 2026 Match Schedule, published June 3, 2026.** Supplies the
   official dates, Eastern times, and host cities for matches 73-104.

The source URLs, local PDF paths, SHA-256 hashes, retrieval timestamp, normalized
output hashes, and extraction-review status are recorded in
`data/reference/fifa_2026_knockout_manifest.json`. The extraction command also
writes the requested generated copy at
`artifacts/manifests/fifa_2026_knockout_mapping.json`.

## Reproduction

```bash
python scripts/extract_fifa_2026_knockout.py
UV_NO_EDITABLE=1 uv run goalsignal tournament validate-bracket
```

The extractor uses `pdftotext -layout` to read Annexe C and refuses to write the
normalized table unless it finds exactly options 1 through 495. The compact
schedule table was manually cross-checked against the official one-page FIFA
schedule. Re-running the extractor preserves the raw PDFs and deterministically
regenerates:

- `data/reference/fifa_2026_third_place_combinations.csv`
- `data/reference/fifa_2026_knockout_mapping.csv`
- `data/reference/fifa_2026_knockout_manifest.json`

## Runtime validation

`OfficialBracket.load()` verifies the raw source hashes, all match numbers,
round sizes, symbolic group slots, advancement references, all 495 unique
combination keys, and every eight-team assignment permutation. Missing or
invalid mappings fail loudly; there is no heuristic fallback.

For each Monte Carlo draw, simulated group standings determine the eight best
third-place groups. Their canonical sorted key selects one exact Annexe C row.
The resulting 32 unique qualifiers populate matches 73-88, then official
winner/loser edges resolve matches 89-104.
