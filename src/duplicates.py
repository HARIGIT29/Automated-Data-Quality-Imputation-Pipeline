"""Duplicate row handling."""

from __future__ import annotations

import pandas as pd


def drop_duplicate_rows(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """Remove duplicate rows; keep first or last occurrence."""
    return df.drop_duplicates(keep=keep).reset_index(drop=True)
