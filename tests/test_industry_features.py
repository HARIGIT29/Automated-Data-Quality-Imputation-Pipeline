"""Industry-style behavior checks: dates, high-cardinality, outlier action."""

from __future__ import annotations

import pandas as pd

from src.pipeline import PipelineConfig, run_pipeline


def test_date_column_expands_not_onehot():
    df = pd.DataFrame(
        {
            "amount": [10, 20, 30, 40],
            "date": ["2022-01-01", "2022-01-02", "2022-02-01", "2022-03-01"],
            "country": ["US", "US", "UK", "FR"],
        }
    )
    out = run_pipeline(df, PipelineConfig(encoding="onehot", drop_duplicates=False))
    raw = out["cleaned_raw_df"]
    assert "date" not in raw.columns
    assert "date_year" in raw.columns
    assert "date_month" in raw.columns
    assert "date_dayofweek" in raw.columns


def test_mixed_date_formats_parse_without_collapse():
    df = pd.DataFrame(
        {
            "date": ["2022-01-01", "02/03/2022", "2023/04/05", "15/06/2024"],
            "amount": [1, 2, 3, 4],
            "country": ["us", "us", "uk", "fr"],
        }
    )
    out = run_pipeline(df, PipelineConfig(drop_duplicates=False, encoding="onehot"))
    raw = out["cleaned_raw_df"]
    assert "date_year" in raw.columns
    assert int(raw["date_year"].nunique(dropna=True)) >= 2


def test_high_cardinality_frequency_encoding_reduces_width():
    n = 120
    df = pd.DataFrame(
        {
            "amount": list(range(n)),
            "company": [f"Company_{i}" for i in range(n)],  # high-card
            "sector": ["A", "B", "A", "C"] * (n // 4),
        }
    )
    cfg = PipelineConfig(
        encoding="onehot",
        low_cardinality_threshold=10,
        high_cardinality_strategy="frequency",
        drop_duplicates=False,
    )
    out = run_pipeline(df, cfg)
    model = out["model_ready_df"]
    # high-card company should not create 120 one-hot columns
    assert "company_freq" in model.columns
    assert not any(c.startswith("company_Company_") for c in model.columns)


def test_outlier_remove_action_changes_row_count():
    df = pd.DataFrame(
        {
            "x": [1, 1, 1, 1, 1, 2000],
            "country": ["US", "US", "US", "US", "US", "US"],
        }
    )
    cfg = PipelineConfig(
        encoding="onehot",
        outlier_action="remove",
        contamination=0.2,
        drop_duplicates=False,
    )
    out = run_pipeline(df, cfg)
    raw = out["cleaned_raw_df"]
    assert len(raw) <= len(df)
    assert out["report_json"]["summary"]["outliers_removed"] >= 1
    assert out["report_json"]["summary"]["outliers_flagged"] >= 1


def test_text_normalization_merges_dirty_categories():
    df = pd.DataFrame(
        {
            "country": ["United States", "United States.", " united states ", "Germany."],
            "amount": [1, 2, 3, 4],
        }
    )
    out = run_pipeline(
        df,
        PipelineConfig(
            encoding="onehot",
            low_cardinality_threshold=10,
            drop_duplicates=False,
        ),
    )
    model = out["model_ready_df"]
    country_cols = [c for c in model.columns if c.startswith("country_")]
    # Should not create multiple United States variants
    assert len([c for c in country_cols if "united states" in c]) == 1


def test_funds_raised_millions_forced_numeric():
    df = pd.DataFrame(
        {
            "funds_raised_millions": ["100", "unknown", "N/A", "250"],
            "country": ["us", "us", "uk", "fr"],
        }
    )
    out = run_pipeline(
        df,
        PipelineConfig(
            encoding="onehot",
            numeric_impute_strategy="median",
            drop_duplicates=False,
        ),
    )
    raw = out["cleaned_raw_df"]
    model = out["model_ready_df"]
    assert "funds_raised_millions" in raw.columns
    assert pd.api.types.is_numeric_dtype(raw["funds_raised_millions"])
    assert "funds_raised_millions" in model.columns
    assert pd.api.types.is_numeric_dtype(model["funds_raised_millions"])


def test_summary_contains_imputation_and_encoding_counts():
    df = pd.DataFrame(
        {
            "country": ["us", "us", "uk", None],
            "amount": [1, None, 3, 4],
        }
    )
    out = run_pipeline(df, PipelineConfig(drop_duplicates=False, encoding="onehot"))
    summary = out["report_json"]["summary"]
    assert "numeric_imputed_cells" in summary
    assert "categorical_imputed_cells" in summary
    assert "encoded_features_created" in summary
