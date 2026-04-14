"""Data-quality and schema validation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_dataframe(
    df: pd.DataFrame,
    *,
    high_cardinality_threshold: int = 100,
) -> dict[str, Any]:
    """
    Run lightweight schema/data quality checks.

    Returns a dictionary with `errors`, `warnings`, and `summary`.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        errors.append("Dataset is empty (0 rows).")
    if df.shape[1] == 0:
        errors.append("Dataset has no columns.")

    cols = [str(c) for c in df.columns]
    if len(set(cols)) != len(cols):
        errors.append("Duplicate column names found.")
    if any(not str(c).strip() for c in df.columns):
        errors.append("One or more column names are blank/whitespace.")

    # Column-level checks
    null_only_cols: list[str] = []
    constant_cols: list[str] = []
    high_card_cols: list[str] = []
    mixed_type_cols: list[str] = []
    high_missing_cols: list[str] = []

    for c in df.columns:
        s = df[c]
        nunique = int(s.nunique(dropna=True))
        if int(s.isna().sum()) == len(s):
            null_only_cols.append(str(c))
        if nunique <= 1:
            constant_cols.append(str(c))
        if nunique > high_cardinality_threshold and s.dtype == object:
            high_card_cols.append(str(c))
        missing_ratio = float(s.isna().mean()) if len(s) else 0.0
        if missing_ratio > 0.5:
            high_missing_cols.append(str(c))

        # Mixed python object types in object/string columns.
        if s.dtype == object:
            types = s.dropna().map(type).nunique()
            if int(types) > 1:
                mixed_type_cols.append(str(c))

    if null_only_cols:
        warnings.append(f"All-null columns: {null_only_cols}.")
    if constant_cols:
        warnings.append(f"Constant/near-constant columns: {constant_cols}.")
    if high_card_cols:
        warnings.append(
            f"High-cardinality categorical columns (>{high_cardinality_threshold} unique): {high_card_cols}."
        )
    if mixed_type_cols:
        warnings.append(f"Mixed-type object columns: {mixed_type_cols}.")
    if high_missing_cols:
        warnings.append(f"Columns with >50% missing values: {high_missing_cols}.")

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_cells": int(df.isna().sum().sum()),
        "all_null_columns": null_only_cols,
        "constant_columns": constant_cols,
        "high_cardinality_columns": high_card_cols,
        "mixed_type_columns": mixed_type_cols,
        "high_missing_columns": high_missing_cols,
    }

    return {"errors": errors, "warnings": warnings, "summary": summary}
