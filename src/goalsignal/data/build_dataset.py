"""Canonical dataset construction.

Transforms the raw user-provided CSVs into a single canonical match table with
explicit score-scope semantics, date-aware team normalization, deterministic
match identifiers, and a full audit trail. No row is ever silently dropped:
every exclusion lands in the exclusions ledger with a reason, and every
ambiguity lands in an audit frame.

Score-scope policy (documented limitation):
The source records full-time scores that include extra time but exclude
penalty shootouts. Extra time is provable only when a shootout row exists
(the match must have been level when it ended). Knockout matches decided in
extra time are indistinguishable from 90-minute results in this source, so
non-shootout matches are assigned scope "regulation" by policy and decisive
results in knockout-capable tournaments are flagged in the suspicious-scope
audit as an upper bound on contamination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from goalsignal.data.normalize_teams import TeamNormalizer
from goalsignal.data.schemas import SCOPE_REGULATION, SCOPE_UNKNOWN, DataConfig
from goalsignal.utils.hashing import canonical_match_id

OUTCOME_HOME = "home_win"
OUTCOME_DRAW = "draw"
OUTCOME_AWAY = "away_win"
OUTCOME_UNKNOWN = "unknown"


@dataclass
class BuildResult:
    matches: pd.DataFrame
    exclusions: pd.DataFrame
    duplicate_identities: pd.DataFrame
    shootout_reconciliation: pd.DataFrame
    suspicious_scope: pd.DataFrame
    name_mapping_issues: pd.DataFrame
    stats: dict = field(default_factory=dict)


def _parse_score(value: str) -> tuple[int | None, bool, bool]:
    """Parse a raw score string.

    Returns (score, is_missing, is_unparseable). Missing markers ("NA", empty)
    indicate a fixture without a recorded result, not a parse failure.
    Nonstandard but unambiguous integer strings (e.g. "00") parse normally.
    """
    s = value.strip()
    if s in ("", "NA", "NaN", "nan"):
        return None, True, False
    try:
        n = int(s)
    except ValueError:
        return None, False, True
    if n < 0:
        return None, False, True
    return n, False, False


def _parse_neutral(value: str) -> bool | None:
    s = value.strip().lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    return None


def _is_knockout_capable(tournament: str, patterns: list[str]) -> bool:
    # Qualification competitions are excluded even when a pattern matches as a
    # substring (e.g. "FIFA World Cup qualification" contains "FIFA World Cup");
    # see config/data.yaml for the rationale.
    if "qualif" in tournament.lower():
        return False
    return any(pat in tournament for pat in patterns)


def build(raw: dict[str, pd.DataFrame], config: DataConfig) -> BuildResult:
    normalizer = TeamNormalizer.from_former_names(raw["former_names"])
    exclusions: list[dict] = []
    records: list[dict] = []

    results = raw["results"]
    for row in results.itertuples(index=False):
        provenance = {"source_file": row.source_file, "source_row": int(row.source_row)}
        date = pd.to_datetime(str(row.date), errors="coerce")
        if pd.isna(date):
            exclusions.append(
                {
                    **provenance,
                    "canonical_match_id": None,
                    "reason": "unparseable_date",
                    "detail": f"date={row.date!r}",
                    "severity": "error",
                    "review_status": "pending",
                }
            )
            continue

        home_score, home_missing, home_bad = _parse_score(str(row.home_score))
        away_score, away_missing, away_bad = _parse_score(str(row.away_score))
        if home_bad or away_bad:
            exclusions.append(
                {
                    **provenance,
                    "canonical_match_id": None,
                    "reason": "unparseable_score",
                    "detail": f"home_score={row.home_score!r} away_score={row.away_score!r}",
                    "severity": "error",
                    "review_status": "pending",
                }
            )
            continue

        home_raw = str(row.home_team).strip()
        away_raw = str(row.away_team).strip()
        home = normalizer.canonical(home_raw, date)
        away = normalizer.canonical(away_raw, date)
        if home == away:
            exclusions.append(
                {
                    **provenance,
                    "canonical_match_id": None,
                    "reason": "identical_teams",
                    "detail": f"{home_raw} vs {away_raw} normalize to the same team",
                    "severity": "error",
                    "review_status": "pending",
                }
            )
            continue

        tournament = str(row.tournament).strip()
        city = str(row.city).strip()
        country = str(row.country).strip()
        match_id = canonical_match_id(
            date.strftime("%Y-%m-%d"), home, away, tournament, city, country
        )

        status = "scheduled" if (home_missing or away_missing) else "played"
        records.append(
            {
                "canonical_match_id": match_id,
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_team_raw": home_raw,
                "away_team_raw": away_raw,
                "tournament": tournament,
                "city": city,
                "country": country,
                "neutral": _parse_neutral(str(row.neutral)),
                "status": status,
                "home_score_recorded": home_score,
                "away_score_recorded": away_score,
                **provenance,
            }
        )

    matches = pd.DataFrame(records)

    # --- Duplicate handling -------------------------------------------------
    # Identical canonical identity: keep the first occurrence, exclude the rest.
    dup_mask = matches.duplicated(subset="canonical_match_id", keep="first")
    for row in matches[dup_mask].itertuples(index=False):
        exclusions.append(
            {
                "source_file": row.source_file,
                "source_row": int(row.source_row),
                "canonical_match_id": row.canonical_match_id,
                "reason": "duplicate_canonical_identity",
                "detail": f"{row.date.date()} {row.home_team} vs {row.away_team} "
                f"({row.tournament})",
                "severity": "error",
                "review_status": "pending",
            }
        )
    matches = matches[~dup_mask].reset_index(drop=True)

    # Same (date, home, away) but different tournament/city/country: keep both,
    # flag for review — they may be genuinely distinct matches or conflicting
    # records of one match.
    key_dups = matches[
        matches.duplicated(subset=["date", "home_team", "away_team"], keep=False)
    ].sort_values(["date", "home_team"])

    # --- Shootout reconciliation ---------------------------------------------
    match_index: dict[tuple, list[int]] = {}
    for idx, row in enumerate(matches.itertuples(index=False)):
        match_index.setdefault((row.date, row.home_team, row.away_team), []).append(idx)

    matches["shootout_played"] = False
    matches["shootout_winner"] = None
    matches["shootout_first_shooter"] = None

    recon_rows: list[dict] = []
    seen_shootout_keys: set[tuple] = set()
    for row in raw["shootouts"].itertuples(index=False):
        date = pd.to_datetime(str(row.date), errors="coerce")
        rec = {
            "source_row": int(row.source_row),
            "date": str(row.date),
            "home_team": str(row.home_team),
            "away_team": str(row.away_team),
            "winner": str(row.winner),
            "first_shooter": str(row.first_shooter),
        }
        if pd.isna(date):
            recon_rows.append({**rec, "status": "unparseable_date", "canonical_match_id": None})
            continue
        home = normalizer.canonical(str(row.home_team).strip(), date)
        away = normalizer.canonical(str(row.away_team).strip(), date)
        winner = normalizer.canonical(str(row.winner).strip(), date)
        first = str(row.first_shooter).strip()
        first = normalizer.canonical(first, date) if first else None

        key = (date, home, away)
        if key in seen_shootout_keys:
            recon_rows.append({**rec, "status": "duplicate_shootout", "canonical_match_id": None})
            continue
        seen_shootout_keys.add(key)

        candidates = match_index.get(key, [])
        if not candidates:
            recon_rows.append({**rec, "status": "unmatched", "canonical_match_id": None})
            continue
        if len(candidates) > 1:
            recon_rows.append(
                {**rec, "status": "ambiguous_join", "canonical_match_id": None}
            )
            continue
        idx = candidates[0]
        m = matches.iloc[idx]
        if winner not in (home, away):
            recon_rows.append(
                {
                    **rec,
                    "status": "winner_not_participant",
                    "canonical_match_id": m["canonical_match_id"],
                }
            )
            continue
        if first is not None and first not in (home, away):
            recon_rows.append(
                {
                    **rec,
                    "status": "first_shooter_not_participant",
                    "canonical_match_id": m["canonical_match_id"],
                }
            )
            continue

        matches.loc[idx, "shootout_played"] = True
        matches.loc[idx, "shootout_winner"] = winner
        matches.loc[idx, "shootout_first_shooter"] = first
        if m["status"] == "played" and m["home_score_recorded"] != m["away_score_recorded"]:
            status = "matched_score_not_tied"
        else:
            status = "matched"
        recon_rows.append({**rec, "status": status, "canonical_match_id": m["canonical_match_id"]})

    # --- Score scope ---------------------------------------------------------
    patterns = config.score_scope_policy.knockout_capable_tournament_patterns
    scope, reg_h, reg_a, reg_known, outcome = [], [], [], [], []
    et_played, et_known, strict_ok, strict_reason = [], [], [], []

    for row in matches.itertuples(index=False):
        if row.status != "played":
            scope.append(None)
            reg_h.append(None)
            reg_a.append(None)
            reg_known.append(False)
            outcome.append(OUTCOME_UNKNOWN)
            et_played.append(None)
            et_known.append(False)
            strict_ok.append(False)
            strict_reason.append("not_played")
            continue

        if row.shootout_played:
            # A shootout proves the match was level when play ended, hence the
            # regulation outcome was a draw. Whether extra time was played (and
            # therefore whether the recorded score equals the 90-minute score)
            # is unknowable from this source.
            tied = row.home_score_recorded == row.away_score_recorded
            scope.append(SCOPE_UNKNOWN)
            reg_h.append(None)
            reg_a.append(None)
            reg_known.append(False)
            outcome.append(OUTCOME_DRAW if tied else OUTCOME_UNKNOWN)
            et_played.append(None)
            et_known.append(False)
            strict_ok.append(False)
            strict_reason.append("shootout_score_scope_unknown")
        else:
            scope.append(SCOPE_REGULATION)
            reg_h.append(row.home_score_recorded)
            reg_a.append(row.away_score_recorded)
            reg_known.append(True)
            if row.home_score_recorded > row.away_score_recorded:
                outcome.append(OUTCOME_HOME)
            elif row.home_score_recorded < row.away_score_recorded:
                outcome.append(OUTCOME_AWAY)
            else:
                outcome.append(OUTCOME_DRAW)
            et_played.append(False)
            et_known.append(not _is_knockout_capable(row.tournament, patterns))
            strict_ok.append(True)
            strict_reason.append(None)

    matches["recorded_score_scope"] = scope
    matches["regulation_home_score"] = reg_h
    matches["regulation_away_score"] = reg_a
    matches["regulation_score_known"] = reg_known
    matches["regulation_outcome"] = outcome
    matches["extra_time_played"] = et_played
    matches["extra_time_status_known"] = et_known
    matches["strict_goal_model_eligible"] = strict_ok
    matches["strict_exclusion_reason"] = strict_reason

    # --- Suspicious-scope audit ------------------------------------------------
    terms = [t.lower() for t in config.validation.suspicious_tournament_terms]
    sus_rows: list[dict] = []
    for row in matches.itertuples(index=False):
        reasons = []
        tl = row.tournament.lower()
        for term in terms:
            if term in tl:
                reasons.append(f"tournament_term:{term}")
        if (
            row.status == "played"
            and not row.shootout_played
            and row.home_score_recorded != row.away_score_recorded
            and _is_knockout_capable(row.tournament, patterns)
        ):
            reasons.append("possible_extra_time_decisive_knockout_capable")
        if row.status == "played" and (
            row.home_score_recorded >= config.validation.implausible_score_threshold
            or row.away_score_recorded >= config.validation.implausible_score_threshold
        ):
            reasons.append("implausible_score")
        if reasons:
            sus_rows.append(
                {
                    "canonical_match_id": row.canonical_match_id,
                    "date": row.date.date().isoformat(),
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "tournament": row.tournament,
                    "home_score_recorded": row.home_score_recorded,
                    "away_score_recorded": row.away_score_recorded,
                    "reasons": ";".join(reasons),
                    "source_row": int(row.source_row),
                }
            )

    exclusions_df = pd.DataFrame(
        exclusions,
        columns=[
            "source_file",
            "source_row",
            "canonical_match_id",
            "reason",
            "detail",
            "severity",
            "review_status",
        ],
    )
    recon_df = pd.DataFrame(
        recon_rows,
        columns=[
            "source_row",
            "date",
            "home_team",
            "away_team",
            "winner",
            "first_shooter",
            "status",
            "canonical_match_id",
        ],
    )
    sus_df = pd.DataFrame(
        sus_rows,
        columns=[
            "canonical_match_id",
            "date",
            "home_team",
            "away_team",
            "tournament",
            "home_score_recorded",
            "away_score_recorded",
            "reasons",
            "source_row",
        ],
    )
    issues_df = pd.DataFrame(
        [{"kind": i.kind, "detail": i.detail} for i in normalizer.issues],
        columns=["kind", "detail"],
    )

    played = matches[matches["status"] == "played"]
    stats = {
        "raw_rows": len(results),
        "canonical_matches": len(matches),
        "played_matches": len(played),
        "scheduled_matches": int(len(matches) - len(played)),
        "excluded_rows": len(exclusions_df),
        "date_min": matches["date"].min().date().isoformat() if len(matches) else None,
        "date_max": matches["date"].max().date().isoformat() if len(matches) else None,
        "teams": int(
            pd.concat([matches["home_team"], matches["away_team"]]).nunique()
        ),
        "tournaments": int(matches["tournament"].nunique()),
        "shootouts_matched": int(matches["shootout_played"].sum()),
        "strict_goal_model_eligible": int(played["strict_goal_model_eligible"].sum()),
        "duplicate_identity_pairs": int(
            key_dups["canonical_match_id"].nunique() if len(key_dups) else 0
        ),
        "suspicious_scope_rows": len(sus_df),
    }

    return BuildResult(
        matches=matches,
        exclusions=exclusions_df,
        duplicate_identities=key_dups.reset_index(drop=True),
        shootout_reconciliation=recon_df,
        suspicious_scope=sus_df,
        name_mapping_issues=issues_df,
        stats=stats,
    )
