"""Immutable raw-snapshot cache for source responses.

Layout (per snapshot):

    data/external/<source>/raw/<snapshot_id>/
        request.json    safe request metadata (NEVER headers or secrets)
        response.json    raw response bytes, preserved exactly
        manifest.json    deterministic manifest (see manifests.py)

Snapshot IDs are content-derived, so an identical response reuses its existing
snapshot rather than overwriting it. Raw files are never mutated; normalization
writes to a separate `normalized/` tree.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from goalsignal.data.sources.manifests import build_snapshot_manifest
from goalsignal.utils.hashing import sha256_text
from goalsignal.utils.paths import resolve

# Header names worth keeping for observability (rate-limit counters). Auth
# headers are never in this set and are never persisted.
_SAFE_RESPONSE_HEADERS = {
    "x-requests-available-minute",
    "x-requestcounter-reset",
    "content-type",
    "x-api-version",
}


def safe_response_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() in _SAFE_RESPONSE_HEADERS}


def write_raw_snapshot(
    *,
    source: str,
    role: str,
    endpoint: str,
    safe_path: str,
    params: dict,
    response_body: bytes,
    response_headers: dict[str, str],
    available_at_semantics: str,
    schema_version: int,
    license: str,
    attribution: str,
    base_dir: str,
    retrieved_at: str | None = None,
    row_count: int | None = None,
    coverage_period_start: str | None = None,
    coverage_period_end: str | None = None,
) -> dict:
    """Persist a raw snapshot immutably; reuse on identical content.

    Returns the manifest as a dict. `request.json` deliberately omits all
    headers so no secret can leak into the cache.
    """
    retrieved_at = retrieved_at or datetime.now(UTC).isoformat(timespec="seconds")
    content_hash = sha256_text(response_body.decode("utf-8", errors="replace"))

    manifest = build_snapshot_manifest(
        source=source,
        role=role,
        endpoint_or_url=f"{endpoint}:{safe_path}",
        available_at_semantics=available_at_semantics,
        license=license,
        attribution=attribution,
        content_hash=content_hash,
        row_count=row_count if row_count is not None else _infer_row_count(response_body),
        schema_version=schema_version,
        cache_path="",  # filled below once the snapshot dir is known
        request_parameters=dict(params),
        coverage_period_start=coverage_period_start,
        coverage_period_end=coverage_period_end,
        retrieval_timestamp=retrieved_at,
        notes=[f"safe_response_headers={safe_response_headers(response_headers)}"],
    )

    snap_dir = resolve(base_dir) / "raw" / manifest.snapshot_id
    manifest = manifest.model_copy(update={"cache_path": str(snap_dir)})

    if snap_dir.exists():
        # Content-derived ID already present: immutable, do not overwrite.
        existing = json.loads((snap_dir / "response.json").read_bytes().decode("utf-8")) \
            if (snap_dir / "response.json").exists() else None
        del existing
        return json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))

    snap_dir.mkdir(parents=True, exist_ok=True)
    request_meta = {
        "source": source,
        "endpoint": endpoint,
        "safe_path": safe_path,
        "params": dict(params),
        "retrieved_at": retrieved_at,
        # NOTE: headers intentionally excluded to keep secrets out of the cache.
    }
    (snap_dir / "request.json").write_text(
        json.dumps(request_meta, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    (snap_dir / "response.json").write_bytes(response_body)
    (snap_dir / "manifest.json").write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest.model_dump()


def _infer_row_count(body: bytes) -> int:
    try:
        obj = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return 0
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        for key in ("matches", "competitions", "teams", "standings", "areas", "persons"):
            if isinstance(obj.get(key), list):
                return len(obj[key])
        return 1
    return 0


def read_raw_snapshot(base_dir: str, snapshot_id: str) -> dict:
    """Replay a cached snapshot: returns {manifest, request, response}."""
    snap_dir = resolve(base_dir) / "raw" / snapshot_id
    if not snap_dir.exists():
        raise FileNotFoundError(f"no cached snapshot {snapshot_id} under {base_dir}")
    return {
        "manifest": json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8")),
        "request": json.loads((snap_dir / "request.json").read_text(encoding="utf-8")),
        "response": json.loads((snap_dir / "response.json").read_text(encoding="utf-8")),
    }


def request_key(endpoint: str, params: dict | None) -> str:
    """Deterministic key for an (endpoint, params) request (no secrets)."""
    from goalsignal.utils.hashing import sha256_json

    return sha256_json({"endpoint": endpoint, "params": params or {}})[:16]


def _index_path(base_dir: str):
    return resolve(base_dir) / "request_index.json"


def lookup_request(base_dir: str, key: str) -> str | None:
    """Return the snapshot_id previously cached for a request key, if any."""
    p = _index_path(base_dir)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(key)


def record_request(base_dir: str, key: str, snapshot_id: str) -> None:
    """Map a request key to its content-derived snapshot id (cache-first)."""
    p = _index_path(base_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    index[key] = snapshot_id
    p.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def list_snapshots(base_dir: str) -> list[dict]:
    raw = resolve(base_dir) / "raw"
    if not raw.exists():
        return []
    out = []
    for d in sorted(raw.iterdir()):
        mpath = d / "manifest.json"
        if mpath.exists():
            out.append(json.loads(mpath.read_text(encoding="utf-8")))
    return out
