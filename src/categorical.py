"""Optional encoding for categorical columns."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str],
    method: str = "onehot",
) -> pd.DataFrame:
    """
    Encode categorical columns. Drops original cat columns and concatenates encoded.
    method: 'onehot', 'ordinal', or 'none' (returns df unchanged for those cols).
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

    # one-hot
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    oh = enc.fit_transform(sub)
    names = enc.get_feature_names_out(cats)
    oh_df = pd.DataFrame(oh, columns=names, index=df.index)
    return pd.concat([out.reset_index(drop=True), oh_df.reset_index(drop=True)], axis=1)
