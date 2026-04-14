"""Optional encoding for categorical columns."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def _binary_bit_string(value: str, categories_sorted: list[str]) -> str:
    """
    One column per category: left char = first category alphabetically, etc.
    Example categories ['France','Germany','Spain']:
      France -> 100, Germany -> 010, Spain -> 001
    """
    n = len(categories_sorted)
    if n == 0:
        return ""
    try:
        idx = categories_sorted.index(value)
    except ValueError:
        return "0" * n
    return "".join("1" if j == idx else "0" for j in range(n))


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
    method: str = "onehot",
    *,
    log: list[str] | None = None,
) -> pd.DataFrame:
    """
    Encode categorical columns. Drops original cat columns and concatenates encoded.

    method:
      'onehot' — separate binary columns per category (sklearn).
      'binary_bits' — single string column per feature, e.g. Spain -> '001', France -> '010',
        Germany -> '100' when categories sort as France, Germany, Spain (left = first alpha).
      'ordinal', 'none'
    """
    if not categorical_cols or method == "none":
        return df.copy()

    cats = [c for c in categorical_cols if c in df.columns]
    if not cats:
        return df.copy()

    out = df.drop(columns=cats)
    sub = df[cats].astype(str)

    if method == "ordinal":
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        arr = enc.fit_transform(sub)
        for i, col in enumerate(cats):
            out[f"{col}_ordinal"] = arr[:, i]
        return out

    if method == "binary_bits":
        for col in cats:
            s = sub[col].fillna("missing").astype(str)
            s = s.replace({"nan": "missing", "<NA>": "missing"})
            uniques = sorted(s.unique().tolist())
            bit_col = s.map(lambda v, u=uniques: _binary_bit_string(str(v), u))
            out[f"{col}_binary"] = bit_col.values
            if log is not None:
                legend = ", ".join(f"{cat}={_binary_bit_string(cat, uniques)}" for cat in uniques)
                log.append(f"Binary bits for '{col}' (left→right = alphabetical): {legend}")
        return out

    # one-hot
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    oh = enc.fit_transform(sub)
    names = enc.get_feature_names_out(cats)
    oh_df = pd.DataFrame(oh, columns=names, index=df.index)
    return pd.concat([out.reset_index(drop=True), oh_df.reset_index(drop=True)], axis=1)
