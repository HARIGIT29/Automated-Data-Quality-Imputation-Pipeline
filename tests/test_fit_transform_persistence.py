"""Fit/transform separation and artifact persistence."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.pipeline import ModelReadyPreprocessor, PipelineConfig


def test_fit_transform_separation_keeps_shape():
    train = pd.DataFrame(
        {
            "age": [22, 25, None, 40, 35],
            "salary": [30000, 35000, 40000, None, 50000],
            "country": ["France", "Spain", "Germany", "France", None],
        }
    )
    test = pd.DataFrame(
        {
            "age": [20, None],
            "salary": [25000, 45000],
            "country": ["Spain", "Italy"],  # unseen category
        }
    )
    cfg = PipelineConfig(encoding="onehot", numeric_impute_strategy="median", drop_duplicates=False)
    p = ModelReadyPreprocessor(cfg).fit(train)
    out_train = p.transform(train)
    out_test = p.transform(test)
    assert "is_outlier" in out_train.columns
    assert out_train.shape[1] == out_test.shape[1]
    assert out_test.isna().sum().sum() == 0


def test_preprocessor_save_and_load():
    df = pd.DataFrame({"x": [1.0, None, 3.0], "cat": ["a", "b", None]})
    cfg = PipelineConfig(encoding="onehot", numeric_impute_strategy="median", drop_duplicates=False)
    p = ModelReadyPreprocessor(cfg).fit(df)
    path = Path(__file__).resolve().parents[1] / "artifacts" / "test_preprocessor.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    p.save(str(path))
    loaded = ModelReadyPreprocessor.load(str(path))
    transformed = loaded.transform(df)
    assert transformed.isna().sum().sum() == 0
    assert path.exists()
