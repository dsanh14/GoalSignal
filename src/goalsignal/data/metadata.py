"""Dataset versioning and manifests.

A dataset version is a deterministic function of the source-file hashes, the
schema version, and the build-code state. Datasets are never identified only
as "latest"; downstream artifacts must reference the manifest's
dataset_version.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from goalsignal.data.build_dataset import BuildResult
from goalsignal.data.schemas import DataConfig
from goalsignal.utils.hashing import sha256_file, sha256_json
from goalsignal.utils.paths import repo_root, resolve


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(), capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root(), capture_output=True, text=True, check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": None}


def compute_dataset_version(config: DataConfig) -> str:
    """Deterministic dataset version from source hashes and build policy."""
    version_basis = {
        "source_sha256": {
            name: sha256_file(config.input_path(name))
            for name in ("results", "shootouts", "goalscorers", "former_names")
        },
        "schema_version": config.schema_version,
        "score_scope_policy": config.score_scope_policy.model_dump(),
        "validation": config.validation.model_dump(),
    }
    return sha256_json(version_basis)[:16]


def build_manifest(config: DataConfig, result: BuildResult, output_file: Path) -> dict:
    source_hashes = {
        name: {
            "path": str(config.input_path(name)),
            "sha256": sha256_file(config.input_path(name)),
        }
        for name in ("results", "shootouts", "goalscorers", "former_names")
    }
    manifest = {
        "dataset_version": compute_dataset_version(config),
        "built_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_version": config.schema_version,
        "sources": source_hashes,
        "output": {
            "path": str(output_file),
            "sha256": sha256_file(output_file),
            "rows": result.stats["canonical_matches"],
        },
        "stats": result.stats,
        **_git_state(),
    }
    return manifest


def write_manifest(config: DataConfig, result: BuildResult, output_file: Path) -> Path:
    manifest = build_manifest(config, result, output_file)
    out_dir = resolve(config.output.manifests_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dataset_{manifest['dataset_version']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path
