"""Tests on synthetic dirty tabular data."""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import PipelineConfig, run_pipeline


@pytest.fixture
def dirty_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 80
    # Duplicate rows
    base = pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n),
            "num_b": rng.normal(10, 2, n),
            "cat": rng.choice(["X", "Y", "Z"], n),
        }
    )
    # Missing values
    base.loc[rng.choice(n, 15, replace=False), "num_a"] = np.nan
    base.loc[rng.choice(n, 10, replace=False), "cat"] = np.nan
    dup = base.iloc[:5].copy()
    return pd.concat([base, dup], ignore_index=True)


def test_pipeline_runs_and_reduces_duplicates(dirty_df: pd.DataFrame):
    cfg = PipelineConfig(drop_duplicates=True, contamination=0.1, encoding="onehot")
    out = run_pipeline(dirty_df, cfg)
    cleaned = out["cleaned_df"]
    assert len(cleaned) < len(dirty_df)
    assert cleaned.isna().sum().sum() == 0
    assert "is_outlier" in cleaned.columns


def test_no_duplicates_option_keeps_rows(dirty_df: pd.DataFrame):
    cfg = PipelineConfig(drop_duplicates=False, encoding="ordinal")
    out = run_pipeline(dirty_df, cfg)
    assert len(out["cleaned_df"]) == len(dirty_df)


def test_report_json_structure(dirty_df: pd.DataFrame):
    out = run_pipeline(dirty_df, PipelineConfig(encoding="none"))
    r = out["report_json"]
    assert "profile_before" in r and "profile_after" in r
    assert "pipeline_log" in r
    assert r["n_rows_out"] == len(out["cleaned_df"])


def test_exclude_column_name(dirty_df: pd.DataFrame):
    df = dirty_df.copy()
    df["id_col"] = range(len(df))
    cfg = PipelineConfig(exclude_from_features=["id_col"], encoding="none")
    out = run_pipeline(df, cfg)
    # id_col should remain if it was only excluded from feature lists — current pipeline
    # does not preserve excluded columns separately; they still exist if in dataframe
    assert "id_col" in out["cleaned_df"].columns
