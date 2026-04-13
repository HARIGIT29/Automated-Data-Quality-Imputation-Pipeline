"""Orchestrates profiling, cleaning, imputation, outliers, scaling, and encoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.categorical import encode_categoricals
from src.duplicates import drop_duplicate_rows
from src.impute_knn import impute_categorical_mode, impute_numeric_knn
from src.outliers import flag_outliers_isolation_forest
from src.profile import profile_dataframe, profile_to_html
from src.scale import scale_numeric
from src.types import coerce_types, split_numeric_categorical


@dataclass
class PipelineConfig:
    """User-configurable pipeline parameters."""

    knn_neighbors: int = 5
    contamination: float = 0.05
    scaler: str = "standard"  # "standard" | "minmax"
    encoding: str = "onehot"  # "onehot" | "ordinal" | "none"
    drop_duplicates: bool = True
    random_state: int = 42
    # Columns to pass through without scaling/encoding (e.g. target ID)
    exclude_from_features: list[str] = field(default_factory=list)


def run_pipeline(df: pd.DataFrame, config: PipelineConfig | None = None) -> dict[str, Any]:
    """
    Run full pipeline. Returns cleaned DataFrame, profiles, HTML, JSON report, log lines.

    Stages:
    1. Profile (before)
    2. Drop duplicates
    3. Coerce types
    4. k-NN impute numerics; mode impute categoricals
    5. Isolation Forest outlier flag on numerics
    6. Scale numeric feature columns (not is_outlier, not excluded)
    7. Encode categoricals
    """
    config = config or PipelineConfig()
    log: list[str] = []

    profile_before = profile_dataframe(df, "before")
    working = df.copy()

    if config.drop_duplicates:
        before_rows = len(working)
        working = drop_duplicate_rows(working)
        log.append(f"Dropped {before_rows - len(working)} duplicate rows.")

    working = coerce_types(working)
    log.append("Coerced object columns to numeric where possible.")

    _, _, num_cols, cat_cols = split_numeric_categorical(working)
    excluded = set(config.exclude_from_features)
    num_features = [c for c in num_cols if c not in excluded]
    cat_features = [c for c in cat_cols if c not in excluded]

    working = impute_numeric_knn(working, num_features, n_neighbors=config.knn_neighbors)
    log.append(f"k-NN imputation (k={config.knn_neighbors}) on {len(num_features)} numeric columns.")

    working = impute_categorical_mode(working, cat_features)
    log.append(f"Mode imputation on {len(cat_features)} categorical columns.")

    working = flag_outliers_isolation_forest(
        working,
        num_features,
        contamination=config.contamination,
        random_state=config.random_state,
    )
    n_out = int(working["is_outlier"].sum()) if "is_outlier" in working.columns else 0
    log.append(f"Isolation Forest: flagged {n_out} rows as outliers (contamination≈{config.contamination}).")

    scale_cols = [c for c in num_features if c in working.columns]
    working, fitted_scaler = scale_numeric(working, scale_cols, method=config.scaler)
    log.append(f"Scaled numerics with {config.scaler} scaler.")

    working = encode_categoricals(working, cat_features, method=config.encoding)
    log.append(f"Categorical encoding: {config.encoding}.")

    profile_after = profile_dataframe(working, "after")
    html = profile_to_html(profile_after)

    report: dict[str, Any] = {
        "profile_before": profile_before,
        "profile_after": profile_after,
        "config": {
            "knn_neighbors": config.knn_neighbors,
            "contamination": config.contamination,
            "scaler": config.scaler,
            "encoding": config.encoding,
            "drop_duplicates": config.drop_duplicates,
        },
        "pipeline_log": log,
        "n_rows_in": int(profile_before["n_rows"]),
        "n_rows_out": int(len(working)),
    }

    return {
        "cleaned_df": working,
        "profile_before": profile_before,
        "profile_after": profile_after,
        "report_json": report,
        "profile_html": html,
        "log": log,
        "scaler": fitted_scaler,
    }
