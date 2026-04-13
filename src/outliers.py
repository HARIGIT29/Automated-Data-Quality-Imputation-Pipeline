"""Isolation Forest outlier detection on scaled numeric features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def flag_outliers_isolation_forest(
    df: pd.DataFrame,
    numeric_cols: list[str],
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add column `is_outlier` (1 = anomaly per Isolation Forest, 0 = inlier).
    Fits IF on StandardScaler-transformed numeric subset only.
    """
    out = df.copy()
    cols = [c for c in numeric_cols if c in out.columns]
    if not cols:
        out["is_outlier"] = 0
        return out

    X = out[cols].astype(float)
    X_scaled = StandardScaler().fit_transform(X)
    iso = IsolationForest(
        contamination=min(max(contamination, 0.001), 0.5),
        random_state=random_state,
    )
    pred = iso.fit_predict(X_scaled)
    # sklearn: -1 = outlier, 1 = inlier
    out["is_outlier"] = (pred == -1).astype(np.int8)
    return out
