"""Load and save tabular datasets."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# Order: BOM UTF-8, strict UTF-8, common Windows / Excel exports, byte-safe fallback
_DEFAULT_CSV_ENCODINGS: tuple[str, ...] = (
    "utf-8-sig",
    "utf-8",
    "cp1252",
    "cp1250",
    "iso-8859-1",
    "latin-1",
)


def read_csv_bytes(
    data: bytes,
    *,
    encodings: tuple[str, ...] | None = None,
    **read_csv_kwargs: Any,
) -> tuple[pd.DataFrame, str]:
    """
    Read CSV from raw bytes, trying encodings until one succeeds.
    Excel-saved CSV on Windows often uses cp1252; 0xA0 is NBSP in that encoding.

    Returns (dataframe, encoding_used).
    """
    seq = encodings or _DEFAULT_CSV_ENCODINGS
    last_error: UnicodeDecodeError | None = None
    for enc in seq:
        try:
            buf = io.BytesIO(data)
            df = pd.read_csv(buf, encoding=enc, **read_csv_kwargs)
            return df, enc
        except UnicodeDecodeError as e:
            last_error = e
            continue
    if last_error:
        raise last_error
    raise ValueError("No encodings provided")


def load_table(path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **read_csv_kwargs)
    raw = path.read_bytes()
    df, _ = read_csv_bytes(raw, **read_csv_kwargs)
    return df


def save_table(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """Save DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as JSON (handles numpy/pandas types)."""

    def default(o: Any) -> Any:
        if hasattr(o, "item"):
            return o.item()
        raise TypeError

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=default)


def save_joblib(obj: Any, path: str | Path) -> None:
    """Persist Python objects for inference reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
