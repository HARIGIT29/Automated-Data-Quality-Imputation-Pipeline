"""Feature scaling for numeric columns."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_numeric(
    df: pd.DataFrame,
    numeric_cols: list[str],
    method: str = "standard",
) -> tuple[pd.DataFrame, StandardScaler | MinMaxScaler]:
    """
    Scale numeric columns in-place copy. Returns (dataframe, fitted scaler).
    method: 'standard' (zero mean, unit variance) or 'minmax' (0-1 range).
    """
    out = df.copy()
    cols = [c for c in numeric_cols if c in out.columns]
    if not cols:
        scaler: StandardScaler | MinMaxScaler = StandardScaler()
        return out, scaler

    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    out[cols] = scaler.fit_transform(out[cols].astype(float))
    return out, scaler
