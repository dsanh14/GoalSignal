"""Request throttling: per-minute sliding window + persistent daily usage.

API-Sports free plan allows only 100 requests/day, so we track usage across
process runs in a small JSON file and refuse to start a request that would
breach `daily_limit - reserve`. The per-minute limiter smooths bursts.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import UTC, datetime

from goalsignal.utils.paths import resolve


class RateLimiter:
    """Sliding 60s window limiter; offline-testable with an injected clock."""

    def __init__(self, per_minute: int, *, now=time.monotonic):
        if per_minute <= 0:
            raise ValueError("per_minute must be positive")
        self.per_minute = per_minute
        self._now = now
        self._calls: deque[float] = deque()

    def _evict(self, t: float) -> None:
        while self._calls and t - self._calls[0] >= 60.0:
            self._calls.popleft()

    def allow(self) -> bool:
        t = self._now()
        self._evict(t)
        return len(self._calls) < self.per_minute

    def wait_time(self) -> float:
        t = self._now()
        self._evict(t)
        if len(self._calls) < self.per_minute:
            return 0.0
        return max(0.0, 60.0 - (t - self._calls[0]))

    def record(self) -> None:
        self._calls.append(self._now())


class DailyQuotaExceeded(RuntimeError):
    """Refusing a request that would breach the reserved daily quota."""


class DailyUsageTracker:
    """Persists a per-UTC-day live-request counter under the cache dir.

    The counter reflects requests *we* made; the provider's own quota is the
    source of truth and is read from response headers / /status when available.
    """

    def __init__(self, cache_dir: str, *, today: str | None = None):
        self.dir = resolve(cache_dir) / "usage"
        self.today = today or datetime.now(UTC).date().isoformat()
        self._path = self.dir / f"{self.today}.json"

    def current(self) -> int:
        if not self._path.exists():
            return 0
        return int(json.loads(self._path.read_text(encoding="utf-8")).get("count", 0))

    def remaining(self, daily_limit: int, reserve: int) -> int:
        return max(0, (daily_limit - reserve) - self.current())

    def can_request(self, daily_limit: int, reserve: int) -> bool:
        return self.remaining(daily_limit, reserve) > 0

    def increment(self) -> int:
        self.dir.mkdir(parents=True, exist_ok=True)
        count = self.current() + 1
        self._path.write_text(
            json.dumps({"date": self.today, "count": count}), encoding="utf-8"
        )
        return count
