"""Numeric and categorical imputation: k-NN or sklearn SimpleImputer."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


def impute_numeric(
    df: pd.DataFrame,
    numeric_cols: list[str],
    *,
    strategy: str = "knn",
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Impute missing numeric values.

    strategy: 'knn' | 'mean' | 'median' (SimpleImputer for mean/median).
    """
    out = df.copy()
    cols = [c for c in numeric_cols if c in out.columns]
    if not cols:
        return out

    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=n_neighbors)
        out[cols] = imputer.fit_transform(out[cols])
        return out

    if strategy not in ("mean", "median"):
        raise ValueError(f"Unknown numeric imputation strategy: {strategy}")

    imp = SimpleImputer(strategy=strategy)
    out[cols] = imp.fit_transform(out[cols])
    return out


def impute_categorical(
    df: pd.DataFrame,
    categorical_cols: list[str],
    *,
    strategy: str = "most_frequent",
    constant_fill: str = "missing",
) -> pd.DataFrame:
    """
    Impute missing categorical columns using SimpleImputer.

    strategy: 'most_frequent' | 'constant' (uses constant_fill).
    """
    out = df.copy()
    cols = [c for c in categorical_cols if c in out.columns]
    if not cols:
        return out

    # Ensure string-like for sklearn (handles int codes forced as categorical)
    for c in cols:
        out[c] = out[c].apply(lambda v: v if pd.isna(v) else str(v))

    if strategy == "constant":
        imp = SimpleImputer(strategy="constant", fill_value=constant_fill)
    elif strategy == "most_frequent":
        imp = SimpleImputer(strategy="most_frequent")
    else:
        raise ValueError(f"Unknown categorical imputation strategy: {strategy}")

    out[cols] = imp.fit_transform(out[cols])
    return out
