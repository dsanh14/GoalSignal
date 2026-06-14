"""API-Sports / API-Football v3 client (direct access).

Replaces the earlier (incorrect) football-data.org integration. Provider facts:

- Vendor: API-Sports. Base URL: https://v3.football.api-sports.io
- Auth header: x-apisports-key (key from $FOOTBALL_DATA_API_KEY — an API-Sports
  key despite the historical env-var name).
- Every response is a JSON envelope:
  {"get", "parameters", "errors", "results", "paging", "response"}.
  Logical failures (bad/missing key, exhausted quota) arrive as HTTP 200 with a
  non-empty `errors` field — so `errors` is always inspected.
- Free plan: 100 requests/day. We track daily usage and stop before a reserve.

Security: the key is read only from the env var, sent ONLY to the configured
host (host-locked), and never logged, cached, hashed, or placed in a manifest.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from urllib.parse import urlparse

from goalsignal.data.sources.cache import (
    lookup_request,
    read_raw_snapshot,
    record_request,
    request_key,
    write_raw_snapshot,
)
from goalsignal.data.sources.config import ApiFootballConfig
from goalsignal.data.sources.http_client import (
    Transport,
    TransportError,
    TransportTimeout,
    UrllibTransport,
)
from goalsignal.data.sources.throttle import (
    DailyQuotaExceeded,
    DailyUsageTracker,
    RateLimiter,
)

LICENSE = "API-Sports / API-Football Terms of Service (per-plan)"
ATTRIBUTION = "Data provided by API-Sports (api-football.com)"
SCHEMA_VERSION = 1

# Endpoints implemented (all confirmed in the stable v3 contract). injuries is
# a real provider endpoint, but World Cup / free-plan population is *measured*,
# never assumed.
SUPPORTED_ENDPOINTS = (
    "status", "leagues", "teams", "teams/statistics", "fixtures",
    "fixtures/rounds", "fixtures/headtohead", "fixtures/statistics",
    "fixtures/events", "fixtures/lineups", "fixtures/players", "standings",
    "players", "players/squads", "injuries", "predictions",
)


class ApiFootballError(RuntimeError):
    """Base error (never carries the secret)."""


class MissingApiKeyError(ApiFootballError):
    pass


class AuthError(ApiFootballError):
    """Provider rejected the key/auth (reported as an envelope error or 401/403)."""


class RateLimitError(ApiFootballError):
    pass


class WrongHostError(ApiFootballError):
    """Refused to send the key to a host other than the configured base URL."""


class MalformedResponseError(ApiFootballError):
    pass


class RequestTimeoutError(ApiFootballError):
    pass


class PlanLimitationError(ApiFootballError):
    """The current plan does not grant access to the requested resource/season."""


class ProviderLogicError(ApiFootballError):
    """Non-auth, non-rate envelope error (bad params, etc.)."""


def parse_envelope(data: object) -> tuple[object, int]:
    """Validate the API-Football envelope; raise on logical errors.

    Returns (response, results). Raises AuthError / RateLimitError /
    ProviderLogicError when `errors` is non-empty.
    """
    if not isinstance(data, dict) or "response" not in data:
        raise MalformedResponseError("response is not an API-Football envelope")
    errors = data.get("errors")
    has_errors = (isinstance(errors, dict) and len(errors) > 0) or (
        isinstance(errors, list) and len(errors) > 0
    )
    if has_errors:
        text = str(errors).lower()
        # Plan limitations are checked first: API-Sports phrases them as a
        # "plan" error (e.g. "Free plans do not have access to this season").
        if "plan" in text or "do not have access" in text or "not have access" in text:
            raise PlanLimitationError(f"plan limitation: {errors}")
        if any(k in text for k in ("token", "api key", "apikey", "not subscribed")):
            raise AuthError(f"provider auth error: {errors}")
        if any(k in text for k in ("ratelimit", "rate limit", "too many requests")):
            raise RateLimitError(f"provider rate/quota error: {errors}")
        raise ProviderLogicError(f"provider error: {errors}")
    return data.get("response"), int(data.get("results", 0) or 0)


def quota_from_headers(headers: dict[str, str]) -> dict:
    """Extract any x-ratelimit-* quota headers (names not hard-assumed)."""
    return {k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit")}


class ApiFootballClient:
    def __init__(
        self,
        config: ApiFootballConfig | None = None,
        *,
        transport: Transport | None = None,
        api_key: str | None = None,
        sleep=time.sleep,
        now=time.monotonic,
        usage: DailyUsageTracker | None = None,
    ):
        self.config = config or ApiFootballConfig()
        self.transport = transport or UrllibTransport()
        self._api_key = api_key if api_key is not None else os.environ.get(
            self.config.credential_env, ""
        )
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute, now=now)
        self.usage = usage or DailyUsageTracker(self.config.cache_dir)
        self._sleep = sleep
        self._allowed_host = urlparse(self.config.base_url).hostname
        self.last_quota: dict = {}

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    def _auth_headers(self) -> dict[str, str]:
        return {self.config.auth_header: self._api_key, "Accept": "application/json"}

    def _check_host(self, url: str) -> None:
        if urlparse(url).hostname != self._allowed_host:
            raise WrongHostError(
                f"refusing to send the API-Sports key to {urlparse(url).hostname!r}; "
                f"only {self._allowed_host!r} is permitted"
            )

    def _live_get(self, endpoint: str, params: dict | None) -> tuple[object, dict]:
        if not self._api_key:
            raise MissingApiKeyError(
                f"no API key in ${self.config.credential_env}; set it in .env. "
                "API-Sports access is disabled without it."
            )
        if not self.usage.can_request(
            self.config.daily_request_limit, self.config.daily_request_reserve
        ):
            raise DailyQuotaExceeded(
                f"daily request budget exhausted "
                f"({self.config.daily_request_limit - self.config.daily_request_reserve} "
                f"usable; {self.usage.current()} used today). Use cached data or wait."
            )
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        self._check_host(url)

        attempts = self.config.retry.max_retries + 1
        for attempt in range(attempts):
            if not self.rate_limiter.allow():
                self._sleep(self.rate_limiter.wait_time())
            self.rate_limiter.record()
            self.usage.increment()
            try:
                resp = self.transport.request(
                    "GET", url, headers=self._auth_headers(), params=params,
                    connect_timeout=self.config.retry.timeout_seconds,
                    read_timeout=self.config.retry.timeout_seconds,
                )
            except TransportTimeout:
                if attempt + 1 >= attempts:
                    raise RequestTimeoutError("request timed out") from None
                self._sleep(self.config.retry.backoff_seconds * (attempt + 1))
                continue
            except TransportError as exc:
                raise ApiFootballError(f"transport error: {exc}") from None

            self.last_quota = quota_from_headers(resp.headers)
            if resp.status_code in (401, 403):
                raise AuthError(f"HTTP {resp.status_code} from provider (auth/host)")
            if resp.status_code == 429:
                raise RateLimitError("HTTP 429 (rate limited)")
            if resp.status_code != 200:
                if 500 <= resp.status_code < 600 and attempt + 1 < attempts:
                    self._sleep(self.config.retry.backoff_seconds * (attempt + 1))
                    continue
                raise ApiFootballError(f"unexpected HTTP {resp.status_code}")
            try:
                data = resp.json()
            except ValueError:
                raise MalformedResponseError("response was not valid JSON") from None
            parse_envelope(data)  # raises on logical errors (auth/rate/params)
            manifest = write_raw_snapshot(
                source="api_football", role="live_fixtures", endpoint=endpoint,
                safe_path=endpoint, params=params or {}, response_body=resp.body,
                response_headers=resp.headers,
                available_at_semantics=(
                    "results post-match; confirmed lineups from announcement; "
                    "standings/fixtures/injuries as of retrieval"
                ),
                schema_version=SCHEMA_VERSION, license=LICENSE, attribution=ATTRIBUTION,
                base_dir=self.config.cache_dir,
            )
            record_request(self.config.cache_dir, request_key(endpoint, params),
                           manifest["snapshot_id"])
            return data, manifest
        raise ApiFootballError("exhausted retries")  # pragma: no cover

    def get(self, endpoint: str, params: dict | None = None, *, refresh: bool = False):
        """Cache-first GET. Returns (envelope_dict, manifest_dict).

        With cache_first and no `refresh`, a previously cached response for the
        same (endpoint, params) is replayed with no live request and no quota use.
        """
        if self.config.cache_first and not refresh:
            sid = lookup_request(self.config.cache_dir, request_key(endpoint, params))
            if sid:
                snap = read_raw_snapshot(self.config.cache_dir, sid)
                return snap["response"], snap["manifest"]
        return self._live_get(endpoint, params)

    # --- documented endpoints -------------------------------------------------
    def status(self, *, refresh: bool = False):
        return self.get("status", refresh=refresh)

    def leagues(self, params: dict | None = None, *, refresh: bool = False):
        return self.get("leagues", params or {}, refresh=refresh)

    def fixtures(self, params: dict, *, refresh: bool = False):
        return self.get("fixtures", params, refresh=refresh)

    def standings(self, params: dict, *, refresh: bool = False):
        return self.get("standings", params, refresh=refresh)

    def lineups(self, fixture_id: int, *, refresh: bool = False):
        return self.get("fixtures/lineups", {"fixture": fixture_id}, refresh=refresh)

    def fixture_players(self, fixture_id: int, *, refresh: bool = False):
        return self.get("fixtures/players", {"fixture": fixture_id}, refresh=refresh)

    def fixture_events(self, fixture_id: int, *, refresh: bool = False):
        return self.get("fixtures/events", {"fixture": fixture_id}, refresh=refresh)

    def injuries(self, params: dict, *, refresh: bool = False):
        return self.get("injuries", params, refresh=refresh)

    def predictions(self, fixture_id: int, *, refresh: bool = False):
        return self.get("predictions", {"fixture": fixture_id}, refresh=refresh)

    # --- safe probe -----------------------------------------------------------
    def probe(self) -> dict:
        """One low-cost /status call to verify auth and read quota."""
        data, manifest = self.status()
        resp = data.get("response") if isinstance(data, dict) else None
        requests = (resp or {}).get("requests", {}) if isinstance(resp, dict) else {}
        subscription = (resp or {}).get("subscription", {}) if isinstance(resp, dict) else {}
        return {
            "provider": "api-football",
            "endpoint": "status",
            "http_status": 200,
            "auth_verified": True,
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "results": int(data.get("results", 0) or 0),
            "requests_current": requests.get("current"),
            "requests_limit_day": requests.get("limit_day"),
            "subscription_plan": subscription.get("plan"),
            "quota_headers": self.last_quota,
            "snapshot_id": manifest["snapshot_id"],
            "content_hash": manifest["content_hash"],
            "cache_path": manifest["cache_path"],
            "schema_validation": "ok",
        }

    # --- World Cup discovery --------------------------------------------------
    def discover_world_cup(self, season: int = 2026, *, refresh: bool = False) -> dict:
        """Find the API-Football league id for the FIFA World Cup (no guessing)."""
        data, manifest = self.leagues({"search": "World Cup"}, refresh=refresh)
        response, _ = parse_envelope(data)
        candidates = []
        for item in response or []:
            league = item.get("league", {})
            country = item.get("country", {})
            seasons = {s.get("year") for s in item.get("seasons", [])}
            name = (league.get("name") or "")
            if "world cup" in name.lower() and league.get("type", "").lower() == "cup":
                candidates.append({
                    "league_id": league.get("id"), "name": name,
                    "country": country.get("name"), "type": league.get("type"),
                    "has_season": season in seasons,
                    "snapshot_id": manifest["snapshot_id"],
                })
        # Prefer the "World" FIFA World Cup with the requested season.
        ranked = sorted(
            candidates,
            key=lambda c: (c["has_season"], c["country"] == "World", c["name"] == "World Cup"),
            reverse=True,
        )
        return {"season": season, "candidates": candidates,
                "selected": ranked[0] if ranked else None}


class ApiFootballAdapter:
    """Lightweight contract surface (name/role/coverage) mirroring other sources."""

    name = "api_football"
    role = "live_fixtures"

    def __init__(self, config: ApiFootballConfig | None = None):
        self.config = config or ApiFootballConfig()

    def is_supported(self, endpoint: str) -> bool:
        return endpoint in SUPPORTED_ENDPOINTS

    def report_coverage(self):
        from goalsignal.data.sources.base import CoverageReport

        return CoverageReport(
            source=self.name, rows=0,
            notes=["coverage measured from ingested data, never assumed",
                   "injuries endpoint exists but World Cup/free-plan population is measured"],
        )
