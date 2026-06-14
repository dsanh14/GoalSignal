"""D1 leakage-safe feature engineering.

Builds a timestamp-aware feature table from the canonical match table, the
leakage-free pre-match Elo timeline, and the normalized historical FIFA ranking
timeline. Every feature for a fixture at (date, source_row) uses ONLY that
team's matches strictly earlier in (date, source_row) order (the documented
same-day policy), and ONLY FIFA releases strictly before the fixture date.

Opponent-adjusted goal expectations use a FIXED, documented mapping of the
pre-match Elo expected score (no fitting, no fold dependence) so they introduce
no leakage and no circularity with any challenger model.

Missing data is flagged with explicit indicators and never silently zeroed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from goalsignal.data.sources.linking import normalize_team
from goalsignal.utils.paths import resolve

LABELS = {"home_win": 0, "draw": 1, "away_win": 2}


def load_d1_config(
    native: str = "config/features_native.yaml",
    fifa: str = "config/features_fifa.yaml",
    experiments: str = "config/experiments_d1.yaml",
) -> dict:
    def _load(p):
        with open(resolve(p), encoding="utf-8") as f:
            return yaml.safe_load(f)

    return {"native": _load(native), "fifa": _load(fifa), "experiments": _load(experiments)}


def _is_competitive(tournament: str, exclude_substrings: list[str]) -> bool:
    t = str(tournament).lower()
    return not any(s in t for s in exclude_substrings)


def build_team_history(matches: pd.DataFrame, elo_timeline: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per team per played match (home + away perspectives)."""
    played = matches[matches["status"] == "played"].copy()
    elo = elo_timeline[["canonical_match_id", "home_elo_pre", "away_elo_pre", "expected_home"]]
    df = played.merge(elo, on="canonical_match_id", how="inner")
    df["date"] = pd.to_datetime(df["date"])
    gf = pd.to_numeric(df["home_score_recorded"], errors="coerce")
    ga = pd.to_numeric(df["away_score_recorded"], errors="coerce")
    outcome = df["regulation_outcome"]

    home = pd.DataFrame({
        "canonical_match_id": df["canonical_match_id"], "date": df["date"],
        "source_row": df["source_row"], "team": df["home_team"], "opponent": df["away_team"],
        "is_home": True, "neutral": df["neutral"].fillna(False).astype(bool),
        "tournament": df["tournament"], "goals_for": gf, "goals_against": ga,
        "elo_own": df["home_elo_pre"], "elo_opp": df["away_elo_pre"],
        "e_expected": df["expected_home"],
        "points": outcome.map({"home_win": 3, "draw": 1, "away_win": 0}),
    })
    away = pd.DataFrame({
        "canonical_match_id": df["canonical_match_id"], "date": df["date"],
        "source_row": df["source_row"], "team": df["away_team"], "opponent": df["home_team"],
        "is_home": False, "neutral": df["neutral"].fillna(False).astype(bool),
        "tournament": df["tournament"], "goals_for": ga, "goals_against": gf,
        "elo_own": df["away_elo_pre"], "elo_opp": df["home_elo_pre"],
        "e_expected": 1.0 - df["expected_home"],
        "points": outcome.map({"home_win": 0, "draw": 1, "away_win": 3}),
    })
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date", "source_row"], kind="stable").reset_index(drop=True)
    return long


def _expected_goals(e: float, base_total: float, gamma: float) -> tuple[float, float]:
    """Split base total goals by relative strength (fixed, no fitting)."""
    eg = max(min(e, 1 - 1e-9), 1e-9)
    w = eg**gamma / (eg**gamma + (1 - eg) ** gamma)
    return base_total * w, base_total * (1 - w)


def compute_native_features(long: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Per (canonical_match_id, is_home) rolling features over strictly-prior matches."""
    nat = config["native"]["recent_form"]
    rest_cfg = config["native"]["rest"]
    ad = config["native"]["attack_defense"]
    excl = config["native"]["match_filters"]["competitive_tournaments_exclude_substrings"]
    match_windows = nat["match_windows"]
    half_life = nat["decay_half_life_days"]
    base_total, gamma = ad["base_total_goals"], ad["split_gamma"]
    congestion = rest_cfg["congestion_windows_days"]
    long_inact = rest_cfg["long_inactivity_days"]
    clip_days = rest_cfg["clip_days_since_prev"]

    out_rows = []
    for _team, g in long.groupby("team", sort=False):
        g = g.sort_values(["date", "source_row"], kind="stable")
        dates = g["date"].to_numpy()
        pts = g["points"].to_numpy(dtype=float)
        gf = g["goals_for"].to_numpy(dtype=float)
        ga = g["goals_against"].to_numpy(dtype=float)
        ev = g["e_expected"].to_numpy(dtype=float)
        comp = g["tournament"].map(lambda t: _is_competitive(t, excl)).to_numpy()
        cids = g["canonical_match_id"].to_numpy()
        is_home = g["is_home"].to_numpy()
        n = len(g)

        # precompute residuals per prior match
        s_resid = pts / 3.0  # normalized actual score in {1, 1/3, 0}... use S in {1,.5,0}
        s_actual = np.where(pts == 3, 1.0, np.where(pts == 1, 0.5, 0.0))
        result_resid = s_actual - ev
        gexp_for = np.empty(n)
        gexp_against = np.empty(n)
        for i in range(n):
            gf_e, ga_e = _expected_goals(ev[i], base_total, gamma)
            gexp_for[i], gexp_against[i] = gf_e, ga_e
        attack_resid = gf - gexp_for
        defense_resid = ga - gexp_against
        del s_resid

        for i in range(n):
            row = {"canonical_match_id": cids[i], "is_home": bool(is_home[i]), "n_prior": i}
            prior = slice(0, i)
            # count-based windows
            for w in match_windows:
                lo = max(0, i - w)
                sl = slice(lo, i)
                cnt = i - lo
                row[f"ppm_last{w}"] = pts[sl].mean() if cnt else np.nan
                if w == 10:
                    row["winrate_last10"] = (pts[sl] == 3).mean() if cnt else np.nan
                    row["gf_per_match_last10"] = gf[sl].mean() if cnt else np.nan
                    row["ga_per_match_last10"] = ga[sl].mean() if cnt else np.nan
                    row["gd_per_match_last10"] = (gf[sl] - ga[sl]).mean() if cnt else np.nan
                    row["clean_sheet_rate_last10"] = (ga[sl] == 0).mean() if cnt else np.nan
                    row["fts_rate_last10"] = (gf[sl] == 0).mean() if cnt else np.nan
                    row["gd_volatility_last10"] = (gf[sl] - ga[sl]).std() if cnt > 1 else np.nan
                    row["opp_adj_points_last10"] = result_resid[sl].mean() if cnt else np.nan
                    row["attack_resid_last10"] = attack_resid[sl].mean() if cnt else np.nan
                    row["defense_resid_last10"] = defense_resid[sl].mean() if cnt else np.nan
            # recency-weighted points (exp decay over all prior)
            if i > 0:
                age = (dates[i] - dates[prior]).astype("timedelta64[D]").astype(float)
                wts = 0.5 ** (age / half_life)
                row["recency_wpoints"] = float((pts[prior] * wts).sum() / wts.sum())
                # competitive-only ppm over last 10 competitive
                comp_idx = np.where(comp[prior])[0]
                if comp_idx.size:
                    take = comp_idx[-10:]
                    row["ppm_comp_last10"] = float(pts[take].mean())
                else:
                    row["ppm_comp_last10"] = np.nan
                # rest
                gap = (dates[i] - dates[i - 1]).astype("timedelta64[D]").astype(float)
                row["days_since_prev"] = min(gap, clip_days)
                row["long_inactivity"] = 1.0 if gap > long_inact else 0.0
                for cw in congestion:
                    row[f"matches_prev_{cw}d"] = int(
                        (age <= cw).sum()
                    )
                # days since previous competitive match
                if comp_idx.size:
                    row["days_since_prev_competitive"] = float(
                        (dates[i] - dates[prior][comp_idx[-1]]).astype("timedelta64[D]")
                        .astype(float)
                    )
                else:
                    row["days_since_prev_competitive"] = np.nan
                # tournament sequence: prior matches in same tournament within 60d + 1
                same_tourn = g["tournament"].to_numpy()
                seq = 1 + int(((same_tourn[prior] == same_tourn[i]) & (age <= 60)).sum())
                row["tournament_seq"] = seq
            else:
                row["recency_wpoints"] = np.nan
                row["ppm_comp_last10"] = np.nan
                row["days_since_prev"] = np.nan
                row["long_inactivity"] = np.nan
                for cw in congestion:
                    row[f"matches_prev_{cw}d"] = 0
                row["days_since_prev_competitive"] = np.nan
                row["tournament_seq"] = 1
            row["form_available"] = 1.0 if i >= 1 else 0.0
            out_rows.append(row)
    return pd.DataFrame(out_rows)


def compute_fifa_features(frame: pd.DataFrame, fifa_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Leakage-safe as-of FIFA features per fixture (home + away), with staleness.

    Returns a frame keyed by canonical_match_id with home_/away_ FIFA columns,
    availability/staleness flags, and momentum (vs previous releases only).
    """
    fcfg = config["fifa"]
    strictly_before = fcfg["as_of"]["strictly_before"]
    stale_after = fcfg["staleness"]["stale_after_days"]
    unavailable_after = fcfg["staleness"]["unavailable_after_days"]
    lags = fcfg["momentum"]["release_lags"]

    fifa = fifa_df.copy()
    fifa["rel"] = pd.to_datetime(fifa["ranking_release_date"])
    fifa["norm"] = fifa["team"].map(normalize_team)
    fifa["fifa_points"] = pd.to_numeric(fifa["fifa_points"], errors="coerce")
    fifa["fifa_rank"] = pd.to_numeric(fifa["fifa_rank"], errors="coerce")
    # per normalized team, sorted release history
    by_team: dict[str, pd.DataFrame] = {
        t: gg.sort_values("rel").reset_index(drop=True) for t, gg in fifa.groupby("norm")
    }

    def as_of(team_norm: str, match_date: pd.Timestamp) -> dict:
        gg = by_team.get(team_norm)
        if gg is None:
            return {"available": False}
        mask = (gg["rel"] < match_date) if strictly_before else (gg["rel"] <= match_date)
        sub = gg[mask]
        if sub.empty:
            return {"available": False}
        idx = len(sub) - 1
        row = sub.iloc[idx]
        age = (match_date - row["rel"]).days
        rec = {
            "available": age <= unavailable_after,
            "stale": age > stale_after,
            "points": float(row["fifa_points"]) if pd.notna(row["fifa_points"]) else None,
            "rank": int(row["fifa_rank"]) if pd.notna(row["fifa_rank"]) else None,
            "release_date": row["rel"].date().isoformat(),
            "age_days": int(age),
        }
        for lag in lags:
            j = idx - lag
            if j >= 0:
                prev = sub.iloc[j]
                rec[f"points_change_{lag}"] = (
                    rec["points"] - float(prev["fifa_points"])
                    if rec["points"] is not None and pd.notna(prev["fifa_points"]) else None
                )
                rec[f"rank_change_{lag}"] = (
                    rec["rank"] - int(prev["fifa_rank"])
                    if rec["rank"] is not None and pd.notna(prev["fifa_rank"]) else None
                )
            else:
                rec[f"points_change_{lag}"] = None
                rec[f"rank_change_{lag}"] = None
        return rec

    rows = []
    for r in frame.itertuples(index=False):
        md = pd.Timestamp(r.date)
        h = as_of(normalize_team(r.home_team), md)
        a = as_of(normalize_team(r.away_team), md)
        avail = bool(h.get("available") and a.get("available")
                     and h.get("points") is not None and a.get("points") is not None)
        rec = {
            "canonical_match_id": r.canonical_match_id,
            "fifa_available": 1.0 if avail else 0.0,
            "fifa_stale": 1.0 if (h.get("stale") or a.get("stale")) else 0.0,
            "home_fifa_points": h.get("points") if avail else np.nan,
            "away_fifa_points": a.get("points") if avail else np.nan,
            "home_fifa_rank": h.get("rank") if avail else np.nan,
            "away_fifa_rank": a.get("rank") if avail else np.nan,
            "home_days_since_fifa_release": h.get("age_days") if h.get("available") else np.nan,
            "away_days_since_fifa_release": a.get("age_days") if a.get("available") else np.nan,
            "home_fifa_release_date": h.get("release_date") if h.get("available") else None,
            "away_fifa_release_date": a.get("release_date") if a.get("available") else None,
        }
        rec["fifa_points_diff"] = (
            rec["home_fifa_points"] - rec["away_fifa_points"] if avail else np.nan)
        rec["fifa_rank_diff"] = (
            rec["home_fifa_rank"] - rec["away_fifa_rank"] if avail else np.nan)
        rec["fifa_release_age_days"] = (
            max(h.get("age_days", 0), a.get("age_days", 0)) if avail else np.nan)
        for lag in lags:
            hp = h.get(f"points_change_{lag}") if avail else np.nan
            ap = a.get(f"points_change_{lag}") if avail else np.nan
            rec[f"home_fifa_points_change_{lag}"] = hp
            rec[f"away_fifa_points_change_{lag}"] = ap
            rec[f"home_fifa_rank_change_{lag}"] = h.get(f"rank_change_{lag}") if avail else np.nan
            rec[f"away_fifa_rank_change_{lag}"] = a.get(f"rank_change_{lag}") if avail else np.nan
        # favorite disagreement: Elo favorite vs FIFA favorite (sign-based; no scaling)
        if avail:
            elo_fav = np.sign(r.home_elo_pre - r.away_elo_pre)
            fifa_fav = np.sign(rec["fifa_points_diff"])
            rec["favorite_disagree"] = 1.0 if (elo_fav != 0 and fifa_fav != 0
                                               and elo_fav != fifa_fav) else 0.0
        else:
            rec["favorite_disagree"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def compute_venue_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Venue context from canonical fields only (no travel/altitude/weather)."""
    rows = []
    for r in frame.itertuples(index=False):
        country = str(getattr(r, "country", "") or "")
        neutral = bool(r.neutral)
        home_at_home = 1.0 if (not neutral and country and
                               normalize_team(country) == normalize_team(r.home_team)) else 0.0
        away_at_home = 1.0 if (country and
                               normalize_team(country) == normalize_team(r.away_team)) else 0.0
        rows.append({
            "canonical_match_id": r.canonical_match_id,
            "is_neutral": 1.0 if neutral else 0.0,
            "home_at_home_country": home_at_home,
            "away_at_home_country": away_at_home,
            "host_indicator": 1.0 if (home_at_home or away_at_home) else 0.0,
            "venue_known": 1.0 if country else 0.0,
        })
    return pd.DataFrame(rows)


# Columns that are pure missingness/availability indicators (never zero-imputed
# away — they ARE the missingness signal).
INDICATOR_COLUMNS = [
    "fifa_available", "fifa_stale", "form_available", "venue_known",
    "home_long_inactivity", "away_long_inactivity", "favorite_disagree",
    "is_neutral", "home_at_home_country", "away_at_home_country", "host_indicator",
]


def build_d1_table(
    matches: pd.DataFrame, elo_timeline: pd.DataFrame, fifa_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Assemble the full leakage-safe D1 feature table for played matches."""
    base = elo_timeline[["canonical_match_id", "date", "home_team", "away_team",
                         "home_elo_pre", "away_elo_pre", "expected_home"]].copy()
    base["date"] = pd.to_datetime(base["date"])
    # carry canonical fields (tournament, country, neutral, label)
    played = matches[matches["status"] == "played"][
        ["canonical_match_id", "tournament", "country", "neutral", "regulation_outcome"]
    ]
    frame = base.merge(played, on="canonical_match_id", how="inner")
    frame["neutral"] = frame["neutral"].fillna(False).astype(bool)
    frame["elo_diff"] = frame["home_elo_pre"] - frame["away_elo_pre"]
    frame["abs_elo_diff"] = frame["elo_diff"].abs()
    frame["home_adv"] = (~frame["neutral"]).astype(float)
    frame["label"] = frame["regulation_outcome"].map(LABELS).fillna(-1).astype(int)

    # native rolling features (home + away perspectives -> wide)
    long = build_team_history(matches, elo_timeline)
    nat = compute_native_features(long, config)
    home_nat = nat[nat["is_home"]].drop(columns=["is_home"]).add_prefix("home_")
    away_nat = nat[~nat["is_home"]].drop(columns=["is_home"]).add_prefix("away_")
    home_nat = home_nat.rename(columns={"home_canonical_match_id": "canonical_match_id"})
    away_nat = away_nat.rename(columns={"away_canonical_match_id": "canonical_match_id"})
    frame = frame.merge(home_nat, on="canonical_match_id", how="left")
    frame = frame.merge(away_nat, on="canonical_match_id", how="left")
    frame["rest_days_diff"] = frame["home_days_since_prev"] - frame["away_days_since_prev"]

    # FIFA + venue
    frame = frame.merge(compute_fifa_features(frame, fifa_df, config),
                        on="canonical_match_id", how="left")
    # abs elo-fifa disagreement (RAW; standardization is fold-local in the model)
    frame["abs_elo_fifa_disagreement"] = (
        frame["fifa_points_diff"].abs() - frame["elo_diff"].abs()).abs()
    frame = frame.merge(compute_venue_features(frame), on="canonical_match_id", how="left")

    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    return frame


def write_feature_table(
    frame: pd.DataFrame, config: dict, source_manifests: dict,
    out_root: str = "artifacts/features/d1", force: bool = False,
) -> dict:
    """Persist the versioned feature table + schema/manifest. Returns metadata."""
    import json

    from goalsignal.utils.hashing import sha256_json

    version = config["native"]["feature_version"]
    config_hash = sha256_json(config)[:16]
    out = resolve(out_root) / version
    table_path = out / "features.csv"
    if table_path.exists() and not force:
        raise FileExistsError(f"{table_path} exists; pass force=True to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    saved = frame.copy()
    saved["date"] = pd.to_datetime(saved["date"]).dt.strftime("%Y-%m-%d")
    saved.to_csv(table_path, index=False)
    meta = {
        "feature_version": version,
        "config_hash": config_hash,
        "rows": len(frame),
        "date_min": str(pd.to_datetime(frame["date"]).min().date()),
        "date_max": str(pd.to_datetime(frame["date"]).max().date()),
        "columns": list(frame.columns),
        "indicator_columns": [c for c in INDICATOR_COLUMNS if c in frame.columns],
        "source_manifests": source_manifests,
        "fifa_available_rows": int(frame["fifa_available"].sum()),
    }
    (out / "feature_schema.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {**meta, "path": str(table_path)}


def load_feature_table(version: str, out_root: str = "artifacts/features/d1") -> pd.DataFrame:
    p = resolve(out_root) / version / "features.csv"
    if not p.exists():
        raise FileNotFoundError(f"no D1 feature table at {p}; run `features build-d1`")
    return pd.read_csv(p, low_memory=False)
