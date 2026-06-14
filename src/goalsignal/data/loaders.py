"""Raw CSV loaders.

Loaders read the user-provided files verbatim (all columns as strings) so that
parsing decisions are explicit, auditable steps downstream. Missing files or
unexpected schemas raise immediately with actionable messages.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from goalsignal.data.schemas import (
    FORMER_NAMES_COLUMNS,
    GOALSCORERS_COLUMNS,
    RESULTS_COLUMNS,
    SHOOTOUTS_COLUMNS,
    DataConfig,
)


class DatasetNotFoundError(FileNotFoundError):
    pass


class SchemaError(ValueError):
    pass


def _load_csv(path: Path, required_columns: list[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise DatasetNotFoundError(
            f"Required dataset file not found: {path}\n"
            f"Place the user-provided '{name}' file there, or pass --input-dir "
            f"pointing at the directory that contains it."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise SchemaError(
            f"{path} is missing required columns {missing}; found {list(df.columns)}"
        )
    # Track provenance: 1-based data row number in the source file (header = row 1).
    df["source_row"] = df.index + 2
    df["source_file"] = path.name
    return df


def load_results(config: DataConfig) -> pd.DataFrame:
    return _load_csv(config.input_path("results"), RESULTS_COLUMNS, "results")


def load_shootouts(config: DataConfig) -> pd.DataFrame:
    return _load_csv(config.input_path("shootouts"), SHOOTOUTS_COLUMNS, "shootouts")


def load_goalscorers(config: DataConfig) -> pd.DataFrame:
    return _load_csv(config.input_path("goalscorers"), GOALSCORERS_COLUMNS, "goalscorers")


def load_former_names(config: DataConfig) -> pd.DataFrame:
    return _load_csv(config.input_path("former_names"), FORMER_NAMES_COLUMNS, "former_names")


def load_all(config: DataConfig) -> dict[str, pd.DataFrame]:
    return {
        "results": load_results(config),
        "shootouts": load_shootouts(config),
        "goalscorers": load_goalscorers(config),
        "former_names": load_former_names(config),
    }
