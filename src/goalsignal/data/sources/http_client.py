"""Minimal HTTP transport abstraction for source adapters.

A single seam for all outbound HTTP so secrets are redacted in one place and
tests can inject a fake transport with no network. The default transport uses
the Python standard library (`urllib`), so the base install needs no HTTP
dependency; an httpx-backed transport is available via the optional `http`
extra but is not required.

Authentication headers are never logged, echoed, or stored: `redact_headers`
masks them, and callers persist only redacted views.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

REDACTED = "***REDACTED***"
# Redact API-Sports' header and the legacy football-data.org header (so a key
# can never leak even if a stale config sends the wrong one), plus generics.
_SECRET_HEADERS = {
    "x-apisports-key", "x-auth-token", "authorization", "api-key", "x-api-key",
    "x-rapidapi-key",
}


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy with secret header values masked."""
    return {
        k: (REDACTED if k.lower() in _SECRET_HEADERS else v) for k, v in headers.items()
    }


class TransportTimeout(Exception):
    """The transport timed out (connect or read)."""


class TransportError(Exception):
    """A non-HTTP transport failure (DNS, connection refused, etc.)."""


@dataclass
class HttpResponse:
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> object:
        return json.loads(self.body.decode("utf-8"))


@runtime_checkable
class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, str] | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 20.0,
    ) -> HttpResponse: ...


class UrllibTransport:
    """Standard-library transport. No third-party dependency."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, str] | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 20.0,
    ) -> HttpResponse:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, method=method, headers=headers)
        # urllib uses a single timeout for the whole operation; use the larger.
        timeout = max(connect_timeout, read_timeout)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return HttpResponse(
                    status_code=resp.status,
                    headers=dict(resp.headers.items()),
                    body=resp.read(),
                )
        except urllib.error.HTTPError as exc:
            # HTTP errors (4xx/5xx) carry a real response we want to inspect.
            return HttpResponse(
                status_code=exc.code,
                headers=dict((exc.headers or {}).items()),
                body=exc.read() if hasattr(exc, "read") else b"",
            )
        except TimeoutError as exc:  # pragma: no cover - network dependent
            raise TransportTimeout(str(exc)) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network dependent
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise TransportTimeout(str(reason)) from exc
            raise TransportError(str(reason)) from exc


@dataclass
class FakeTransport:
    """Scripted transport for tests; records redacted request metadata.

    `responses` is consumed in order. Each item is either an `HttpResponse` or
    a callable `(method, url, headers, params) -> HttpResponse`. The recorded
    `calls` never contain secret header values.
    """

    responses: list = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: dict[str, str] | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 20.0,
    ) -> HttpResponse:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": dict(params or {}),
                "headers": redact_headers(headers),
            }
        )
        if not self.responses:
            raise AssertionError("FakeTransport ran out of scripted responses")
        nxt = self.responses.pop(0)
        return nxt(method, url, headers, params) if callable(nxt) else nxt
