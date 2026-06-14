"""Deterministic source-snapshot manifests.

Every ingested source snapshot is identified by a content-derived snapshot ID
(never "latest"), and records its license, terms, attribution, coverage
period, and cache path so downstream provenance is fully auditable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from goalsignal.utils.hashing import sha256_json
from goalsignal.utils.paths import resolve

MANIFESTS_DIR = "artifacts/manifests/sources"


class SourceSnapshotManifest(BaseModel):
    """Auditable record of one ingested source snapshot."""

    snapshot_id: str
    source: str
    role: str
    endpoint_or_url: str
    retrieval_timestamp: str
    available_at_semantics: str
    license: str
    terms_url: str | None = None
    attribution: str
    content_hash: str
    row_count: int
    schema_version: int
    coverage_period_start: str | None = None
    coverage_period_end: str | None = None
    cache_path: str
    request_parameters: dict = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def compute_snapshot_id(
    source: str,
    endpoint_or_url: str,
    content_hash: str,
    schema_version: int,
    request_parameters: dict | None = None,
) -> str:
    """Deterministic snapshot ID from identity + content (not a timestamp)."""
    basis = {
        "source": source,
        "endpoint_or_url": endpoint_or_url,
        "content_hash": content_hash,
        "schema_version": schema_version,
        "request_parameters": request_parameters or {},
    }
    return sha256_json(basis)[:16]


def build_snapshot_manifest(
    *,
    source: str,
    role: str,
    endpoint_or_url: str,
    available_at_semantics: str,
    license: str,
    attribution: str,
    content_hash: str,
    row_count: int,
    schema_version: int,
    cache_path: str,
    terms_url: str | None = None,
    coverage_period_start: str | None = None,
    coverage_period_end: str | None = None,
    request_parameters: dict | None = None,
    notes: list[str] | None = None,
    retrieval_timestamp: str | None = None,
) -> SourceSnapshotManifest:
    """Construct a manifest with a deterministic snapshot ID."""
    return SourceSnapshotManifest(
        snapshot_id=compute_snapshot_id(
            source, endpoint_or_url, content_hash, schema_version, request_parameters
        ),
        source=source,
        role=role,
        endpoint_or_url=endpoint_or_url,
        retrieval_timestamp=retrieval_timestamp
        or datetime.now(UTC).isoformat(timespec="seconds"),
        available_at_semantics=available_at_semantics,
        license=license,
        terms_url=terms_url,
        attribution=attribution,
        content_hash=content_hash,
        row_count=row_count,
        schema_version=schema_version,
        coverage_period_start=coverage_period_start,
        coverage_period_end=coverage_period_end,
        cache_path=cache_path,
        request_parameters=request_parameters or {},
        notes=notes or [],
    )


def write_manifest(manifest: SourceSnapshotManifest, directory: str = MANIFESTS_DIR) -> Path:
    """Write a manifest as JSON; refuses to overwrite a differing snapshot."""
    out_dir = resolve(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest.source}_{manifest.snapshot_id}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest.model_dump():
            raise FileExistsError(
                f"manifest {path} exists with different content; refusing silent "
                "overwrite (snapshot IDs are content-derived, so this indicates a "
                "hash collision or schema drift to investigate)"
            )
        return path
    path.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
