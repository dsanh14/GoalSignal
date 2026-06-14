"""Live API-Sports / API-Football probe (opt-in only).

Skipped by default (pyproject sets `-m 'not live_api'`). Run explicitly with a
valid API-Sports key in .env:

    UV_NO_EDITABLE=1 uv run pytest -m live_api

Makes exactly one request (/status). Never prints the key. Sends it only to the
configured API-Sports host.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.live_api


def test_live_status_probe():
    from goalsignal.data.sources.api_football import ApiFootballClient
    from goalsignal.data.sources.config import ApiFootballConfig
    from goalsignal.data.sources.env import has_env, load_env_file

    load_env_file()
    if not has_env("FOOTBALL_DATA_API_KEY"):
        pytest.skip("FOOTBALL_DATA_API_KEY not set")
    meta = ApiFootballClient(ApiFootballConfig()).probe()
    assert meta["http_status"] == 200
    assert meta["auth_verified"] is True
    assert meta["schema_validation"] == "ok"
