"""coerce_types edge cases."""

import numpy as np
import pandas as pd
import pytest

from src.types import coerce_types


def test_coerce_skips_all_nan_object_column():
    """Previously 0 >= 0 wrongly coerced all-NaN object columns to float."""
    b = pd.Series([np.nan, np.nan, np.nan], dtype=object)
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": b})
    out = coerce_types(df)
    assert out["b"].dtype == object


def test_coerce_skips_named_column():
    df = pd.DataFrame({"country": ["USA", "UK", "FR"]})
    out = coerce_types(df, skip_columns={"country"})
    # pandas may use string[pyarrow] / StringDtype instead of object
    assert not pd.api.types.is_numeric_dtype(out["country"])


@pytest.mark.parametrize(
    "values",
    [["USA", "UK", None], ["USA", "UK", np.nan]],
)
def test_country_string_column_onehot_without_force(values):
    """Object country names stay categorical and get one-hot columns."""
    from src.pipeline import PipelineConfig, run_pipeline

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "country": values})
    cfg = PipelineConfig(encoding="onehot", drop_duplicates=False, numeric_impute_strategy="median")
    out = run_pipeline(df, cfg)
    cleaned = out["cleaned_df"]
    assert "country" not in cleaned.columns
    assert any(str(c).startswith("country_") for c in cleaned.columns)
