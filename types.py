"""Data type inference and coercion."""

from __future__ import annotations

import pandas as pd


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to convert object columns to numeric where possible.
    Leaves non-parseable values as NaN for later imputation.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            coerced = pd.to_numeric(out[col], errors="coerce")
            # Only replace if we gained meaningful numeric content
            if coerced.notna().sum() >= out[col].notna().sum() * 0.5:
                out[col] = coerced
    return out


def split_numeric_categorical(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Split into numeric and categorical parts; infer if numeric_cols not provided."""
    if numeric_cols is None:
        num = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    else:
        num = [c for c in numeric_cols if c in df.columns]
    cat = [c for c in df.columns if c not in num]
    return df[num], df[cat], num, cat
