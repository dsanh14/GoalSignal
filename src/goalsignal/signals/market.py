"""Market / betting-odds signal and benchmark.

Decimal odds carry the market's implied probabilities plus a bookmaker margin
(the *overround* or *vig*). This module converts decimal odds to implied
probabilities, removes the overround, and exposes the result as an
:class:`~goalsignal.signals.base.OutcomeProbs` (3-way group market) or
:class:`~goalsignal.signals.base.AdvanceProbs` (2-way knockout market).

The market layer is usable three ways, all from the same parsed quote:

* a **standalone benchmark** to compare GoalSignal against,
* a weighted **feature** in the meta-ensemble,
* a **disagreement detector** (see :func:`goalsignal.signals.base.disagreement`).

Input format — a CSV with columns::

    match_id, source, team_a_odds, draw_odds, team_b_odds, timestamp

``draw_odds`` may be blank for a two-way (knockout) market. Missing files and
missing/garbled rows are handled gracefully: bad rows are skipped (optionally
reported) and a missing file yields an empty mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from goalsignal.signals.base import AdvanceProbs, OutcomeProbs

_REQUIRED_COLUMNS = {"match_id", "team_a_odds", "team_b_odds"}


def decimal_to_implied(odds) -> np.ndarray:
    """Convert decimal odds to raw implied probabilities (``1 / odds``).

    The result is *not* normalized — its sum exceeds 1 by the overround.
    """
    arr = np.asarray(odds, dtype=float)
    if np.any(arr <= 1.0):
        raise ValueError(f"decimal odds must be > 1.0, got {odds}")
    return 1.0 / arr


def remove_overround(implied, method: str = "proportional") -> np.ndarray:
    """Remove the bookmaker margin so probabilities sum to 1.

    Args:
        implied: raw implied probabilities (e.g. from :func:`decimal_to_implied`).
        method: ``"proportional"`` (divide by the booksum — fast, standard) or
            ``"power"`` (raise each implied prob to a common power ``k`` solved
            so the result sums to 1). With a positive margin ``k > 1``, which
            accentuates favourites relative to proportional scaling and so
            corrects part of the favourite-longshot bias.

    Returns:
        Normalized probabilities (sum 1).
    """
    p = np.asarray(implied, dtype=float)
    if np.any(p <= 0):
        raise ValueError("implied probabilities must be positive")
    if method == "proportional":
        return p / p.sum()
    if method == "power":
        # sum(p**k) is strictly decreasing in k for p in (0,1): it is the
        # outcome count at k=0 and tends to 0 as k grows, so a unique root with
        # sum == 1 exists. The booksum>1 case needs k>1, hence the wide bracket.
        lo, hi = 1e-6, 1000.0
        for _ in range(200):
            k = 0.5 * (lo + hi)
            if np.sum(p**k) > 1.0:
                lo = k
            else:
                hi = k
        out = p ** (0.5 * (lo + hi))
        return out / out.sum()
    raise ValueError(f"unknown overround-removal method: {method!r}")


@dataclass(frozen=True)
class MarketQuote:
    """One bookmaker quote for a single match."""

    match_id: str
    source: str
    team_a_odds: float
    team_b_odds: float
    draw_odds: float | None
    timestamp: str | None
    team_a: str | None = None  # team names enable dynamic team-pair matching
    team_b: str | None = None

    @property
    def two_way(self) -> bool:
        return self.draw_odds is None

    def implied(self) -> np.ndarray:
        """Raw (overround-inclusive) implied probabilities."""
        if self.two_way:
            return decimal_to_implied([self.team_a_odds, self.team_b_odds])
        return decimal_to_implied([self.team_a_odds, self.draw_odds, self.team_b_odds])

    def overround(self) -> float:
        """Bookmaker margin: ``booksum - 1`` (0 = a fair, margin-free book)."""
        return float(self.implied().sum() - 1.0)

    def outcome(self, method: str = "proportional") -> OutcomeProbs:
        """Normalized 3-way market probabilities. Requires a draw price."""
        if self.two_way:
            raise ValueError(f"match {self.match_id} is a two-way market; use .advance()")
        p = remove_overround(self.implied(), method)
        return OutcomeProbs(p[0], p[1], p[2])

    def advance(self, method: str = "proportional") -> AdvanceProbs:
        """Normalized knockout advance probabilities.

        For a three-way quote the draw mass is split evenly between the two
        teams (a regulation draw goes to extra time / penalties, ~50/50).
        """
        p = remove_overround(self.implied(), method)
        if self.two_way:
            return AdvanceProbs(p[0], p[1])
        return AdvanceProbs(p[0] + 0.5 * p[1], p[2] + 0.5 * p[1])


def load_market_odds(
    path: str | Path,
    *,
    require: bool = False,
    on_error: list[str] | None = None,
) -> dict[str, MarketQuote]:
    """Load a market-odds CSV into ``{match_id: MarketQuote}``.

    Keeps the most recent quote per ``match_id`` (by ``timestamp`` string order
    when present, else file order). Bad rows are skipped; when ``on_error`` is a
    list, a human-readable reason is appended for each skipped row.

    Args:
        path: CSV path.
        require: raise ``FileNotFoundError`` if the file is absent; otherwise an
            absent file yields ``{}`` (graceful degradation — the market signal
            is simply unavailable and the ensemble renormalizes).
        on_error: optional sink for per-row skip reasons.
    """
    p = Path(path)
    if not p.exists():
        if require:
            raise FileNotFoundError(f"market odds file not found: {p}")
        return {}

    df = pd.read_csv(p, dtype=str).fillna("")
    missing = {"team_a_odds", "team_b_odds"} - set(df.columns)
    if missing:
        raise ValueError(f"market odds CSV missing columns: {sorted(missing)}")
    if "match_id" not in df.columns and not {"team_a", "team_b"} <= set(df.columns):
        raise ValueError("market odds CSV needs a 'match_id' column or 'team_a'/'team_b' columns")

    quotes: dict[str, MarketQuote] = {}
    seen_ts: dict[str, str] = {}
    for i, row in df.iterrows():
        match_id = row.get("match_id", "").strip()
        name_a = row.get("team_a", "").strip() or None
        name_b = row.get("team_b", "").strip() or None
        # A row is usable if it has a match id or a full team pair.
        key = match_id or (f"pair::{name_a}|{name_b}" if name_a and name_b else "")
        if not key:
            continue
        try:
            team_a = float(row["team_a_odds"])
            team_b = float(row["team_b_odds"])
            draw_raw = row.get("draw_odds", "").strip()
            draw = float(draw_raw) if draw_raw else None
            quote = MarketQuote(
                match_id=match_id or key,
                source=row.get("source", "").strip() or "market",
                team_a_odds=team_a,
                team_b_odds=team_b,
                draw_odds=draw,
                timestamp=row.get("timestamp", "").strip() or None,
                team_a=name_a,
                team_b=name_b,
            )
            # validate odds eagerly so a bad price is skipped, not deferred.
            quote.implied()
        except (ValueError, TypeError) as exc:
            if on_error is not None:
                on_error.append(f"row {i} (match_id={match_id!r}): {exc}")
            continue
        ts = quote.timestamp or ""
        if key not in quotes or ts >= seen_ts.get(key, ""):
            quotes[key] = quote
            seen_ts[key] = ts
    return quotes


def market_signal(
    quotes: dict[str, MarketQuote],
    match_id: str,
    *,
    knockout: bool = False,
    method: str = "proportional",
) -> OutcomeProbs | AdvanceProbs | None:
    """Return the market signal for one match, or ``None`` if unavailable."""
    quote = quotes.get(match_id)
    if quote is None:
        return None
    return quote.advance(method) if knockout else quote.outcome(method)
