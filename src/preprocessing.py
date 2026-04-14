"""String/date preprocessing utilities for cleaner model inputs."""

from __future__ import annotations

import re

import pandas as pd

MISSING_TEXT_TOKENS = {
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
    "-",
    "",
}


def normalize_text_value(value: object) -> object:
    """Normalize categorical strings while preserving missing values."""
    if pd.isna(value):
        return value
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    # Remove punctuation for category consistency (e.g. "United States.")
    s = re.sub(r"[^\w\s]", "", s)
    s = s.lower()
    if s in MISSING_TEXT_TOKENS:
        return pd.NA
    return s


def normalize_object_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply text normalization to selected object-like columns."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[col] = out[col].map(normalize_text_value)
    return out


def coerce_numeric_like_columns(
    df: pd.DataFrame,
    *,
    explicit_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Force known numeric-like text columns to numeric.
    Handles values like 'unknown', 'N/A' by coercing to NaN.
    """
    out = df.copy()
    candidates = explicit_columns or []
    applied: list[str] = []
    for col in candidates:
        if col not in out.columns:
            continue
        s = out[col]
        if s.dtype == object:
            cleaned = (
                s.astype(str)
                .str.strip()
                .str.lower()
                .replace({tok: pd.NA for tok in MISSING_TEXT_TOKENS})
            )
            out[col] = pd.to_numeric(cleaned, errors="coerce")
            applied.append(col)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            applied.append(col)
    return out, applied


def infer_date_columns(df: pd.DataFrame, candidates: list[str] | None = None) -> list[str]:
    """
    Infer date-like columns from candidate list or object columns by parse ratio.
    """
    cols = candidates if candidates is not None else list(df.columns)
    date_cols: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            date_cols.append(col)
            continue
        if s.dtype != object and "date" not in str(col).lower():
            continue
        parsed = pd.to_datetime(s, errors="coerce", utc=False)
        ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
        if "date" in str(col).lower() or ratio >= 0.6:
            date_cols.append(col)
    return date_cols


def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    """
    Parse mixed-format date strings robustly.
    Tries default parsing first, then day-first fallback for unresolved rows.
    """
    s = series.copy()
    parsed = pd.to_datetime(s, errors="coerce", utc=False)
    mask = parsed.isna() & s.notna()
    if mask.any():
        parsed2 = pd.to_datetime(s[mask], errors="coerce", dayfirst=True, utc=False)
        parsed.loc[mask] = parsed2
    return parsed


def expand_date_features(df: pd.DataFrame, date_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Replace each date column with useful numeric date parts.
    Returns updated dataframe and created feature names.
    """
    out = df.copy()
    created: list[str] = []
    for col in date_cols:
        if col not in out.columns:
            continue
        parsed = parse_mixed_datetime(out[col])
        out[f"{col}_year"] = parsed.dt.year
        out[f"{col}_month"] = parsed.dt.month
        out[f"{col}_day"] = parsed.dt.day
        out[f"{col}_quarter"] = parsed.dt.quarter
        out[f"{col}_dayofweek"] = parsed.dt.dayofweek
        created.extend(
            [
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_quarter",
                f"{col}_dayofweek",
            ]
        )
        out = out.drop(columns=[col])
    return out, created
