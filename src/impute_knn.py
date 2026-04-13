"""k-NN imputation for numeric columns; mode imputation for categoricals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def impute_numeric_knn(
    df: pd.DataFrame,
    numeric_cols: list[str],
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Impute missing numeric values using k-NN (Euclidean on scaled internal space)."""
    out = df.copy()
    cols = [c for c in numeric_cols if c in out.columns]
    if not cols:
        return out
    imputer = KNNImputer(n_neighbors=n_neighbors)
    out[cols] = imputer.fit_transform(out[cols])
    return out


def impute_categorical_mode(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Fill missing categoricals with column mode (or 'missing' if all NaN)."""
    out = df.copy()
    for col in categorical_cols:
        if col not in out.columns:
            continue
        s = out[col]
        if s.dtype == object or str(s.dtype) == "category":
            mode = s.mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "missing"
            out[col] = s.fillna(fill)
        else:
            mode = s.mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else 0
            out[col] = s.fillna(fill)
    return out
