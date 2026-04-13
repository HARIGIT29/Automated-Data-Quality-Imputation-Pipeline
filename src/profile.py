"""Dataset profiling for quality reports."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def profile_dataframe(df: pd.DataFrame, name: str = "dataset") -> dict[str, Any]:
    """Return structured profile: dtypes, missingness, duplicates, numeric summary."""
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing_pct = (missing / max(n_rows, 1) * 100).round(2)
    dup_count = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    per_column: dict[str, Any] = {}
    for col in df.columns:
        s = df[col]
        entry: dict[str, Any] = {
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(missing_pct[col]),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if col in numeric_cols and s.notna().any():
            entry["min"] = float(s.min())
            entry["max"] = float(s.max())
            entry["mean"] = float(s.mean())
            entry["std"] = float(s.std(ddof=0))
        per_column[col] = entry

    return {
        "name": name,
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "duplicate_rows": dup_count,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_cols,
        "columns": per_column,
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 4),
    }


def profile_to_html(profile: dict[str, Any]) -> str:
    """Minimal HTML summary for download/display."""
    rows = []
    rows.append("<h2>Dataset profile</h2>")
    rows.append(f"<p>Rows: {profile['n_rows']}, Columns: {profile['n_columns']}</p>")
    rows.append(f"<p>Duplicate rows: {profile['duplicate_rows']}</p>")
    rows.append("<table border='1'><tr><th>Column</th><th>dtype</th><th>Missing %</th></tr>")
    for col, info in profile["columns"].items():
        rows.append(
            f"<tr><td>{col}</td><td>{info['dtype']}</td><td>{info['missing_pct']}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)
