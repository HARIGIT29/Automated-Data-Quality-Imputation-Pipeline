"""Post-process cleaned frame: round floats for display/export."""

from __future__ import annotations

import pandas as pd


def round_float_columns(
    df: pd.DataFrame,
    decimals: int,
    *,
    skip_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Round float dtypes; keep integers, bools, strings unchanged."""
    skip = skip_columns or set()
    out = df.copy()
    for col in out.columns:
        if col in skip:
            continue
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out
