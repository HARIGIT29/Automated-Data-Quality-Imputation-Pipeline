"""Data type inference and coercion."""

from __future__ import annotations

import pandas as pd


def coerce_types(df: pd.DataFrame, skip_columns: set[str] | None = None) -> pd.DataFrame:
    """
    Attempt to convert object columns to numeric where possible.
    Leaves non-parseable values as NaN for later imputation.

    skip_columns: never coerce these (e.g. country, IDs) so they stay categorical/object.

    All-NaN object columns are left unchanged (avoids bogus numeric conversion).
    """
    skip = skip_columns or set()
    out = df.copy()
    for col in out.columns:
        if col in skip:
            continue
        if out[col].dtype == object:
            n_nonnull = int(out[col].notna().sum())
            if n_nonnull == 0:
                continue
            coerced = pd.to_numeric(out[col], errors="coerce")
            if coerced.notna().sum() >= n_nonnull * 0.5:
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


def apply_force_categorical(
    df: pd.DataFrame,
    num_features: list[str],
    cat_features: list[str],
    force_columns: set[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Treat named columns as categorical: remove from numeric list, add to categorical,
    stringify values (so int country codes still one-hot encode correctly).
    """
    out = df.copy()
    num = [c for c in num_features if c in out.columns]
    cat = [c for c in cat_features if c in out.columns]
    for c in force_columns:
        if c not in out.columns:
            continue
        if c in num:
            num.remove(c)
        if c not in cat:
            cat.append(c)
        out[c] = out[c].apply(lambda v: v if pd.isna(v) else str(v))
    return out, num, cat
