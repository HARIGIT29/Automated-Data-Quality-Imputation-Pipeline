"""Production-style preprocessing pipeline with fit/transform separation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.duplicates import drop_duplicate_rows
from src.format_output import round_float_columns
from src.profile import profile_dataframe, profile_to_html
from src.types import apply_force_categorical, coerce_types, split_numeric_categorical
from src.validation import validate_dataframe


@dataclass
class PipelineConfig:
    """User-configurable pipeline parameters."""

    knn_neighbors: int = 5
    contamination: float = 0.05
    scaler: str = "standard"  # "standard" | "minmax"
    encoding: str = "onehot"  # safer model-ready default
    drop_duplicates: bool = True
    random_state: int = 42
    exclude_from_features: list[str] = field(default_factory=list)
    numeric_impute_strategy: str = "median"  # "knn" | "mean" | "median"
    categorical_impute_strategy: str = "most_frequent"  # "most_frequent" | "constant"
    categorical_constant_fill: str = "missing"
    force_categorical_columns: list[str] = field(default_factory=list)
    round_decimals: int | None = 4
    target_column: str | None = None
    id_column: str | None = None
    high_cardinality_threshold: int = 100


def _binary_bit_string(value: str, categories_sorted: list[str]) -> str:
    n = len(categories_sorted)
    if n == 0:
        return ""
    try:
        idx = categories_sorted.index(value)
    except ValueError:
        return "0" * n
    return "".join("1" if j == idx else "0" for j in range(n))


class ModelReadyPreprocessor:
    """Fit/transform preprocessing object for train/test-safe workflows."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.log: list[str] = []
        self.validation: dict[str, Any] = {}

        self.numeric_imputer: KNNImputer | SimpleImputer | None = None
        self.categorical_imputer: SimpleImputer | None = None
        self.outlier_scaler: StandardScaler | None = None
        self.outlier_model: IsolationForest | None = None
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.encoder: OneHotEncoder | OrdinalEncoder | None = None

        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.passthrough_columns: list[str] = []
        self.force_cat: set[str] = set()
        self.binary_bits_mapping: dict[str, list[str]] = {}
        self.feature_columns_seen: list[str] = []
        self.fitted: bool = False

    def _validate(self, df: pd.DataFrame) -> None:
        v = validate_dataframe(df, high_cardinality_threshold=self.config.high_cardinality_threshold)
        self.validation = v
        for w in v["warnings"]:
            self.log.append(f"Warning: {w}")
        if v["errors"]:
            raise ValueError("Validation failed: " + "; ".join(v["errors"]))

    def _prepare_core(self, df: pd.DataFrame, *, for_fit: bool) -> pd.DataFrame:
        working = df.copy()
        if self.config.drop_duplicates and for_fit:
            before = len(working)
            working = drop_duplicate_rows(working)
            self.log.append(f"Dropped {before - len(working)} duplicate rows in fit data.")

        excluded = set(self.config.exclude_from_features)
        if self.config.target_column:
            excluded.add(self.config.target_column)
        if self.config.id_column:
            excluded.add(self.config.id_column)

        self.force_cat = set(self.config.force_categorical_columns) - excluded
        skip_coerce = self.force_cat | excluded
        working = coerce_types(working, skip_columns=skip_coerce)
        self.log.append("Coerced object columns to numeric where possible.")

        _, _, num_cols, cat_cols = split_numeric_categorical(working)
        num_features = [c for c in num_cols if c not in excluded]
        cat_features = [c for c in cat_cols if c not in excluded]
        working, num_features, cat_features = apply_force_categorical(
            working, num_features, cat_features, self.force_cat
        )
        if self.force_cat:
            self.log.append(f"Forced categorical columns: {sorted(self.force_cat)}.")

        for c in num_features:
            working[c] = pd.to_numeric(working[c], errors="coerce")

        if for_fit:
            self.numeric_features = num_features
            self.categorical_features = cat_features
            self.passthrough_columns = [c for c in working.columns if c not in num_features + cat_features]
            self.feature_columns_seen = list(working.columns)
        return working

    def _fit_imputers(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.numeric_features:
            if self.config.numeric_impute_strategy == "knn":
                self.numeric_imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            else:
                self.numeric_imputer = SimpleImputer(strategy=self.config.numeric_impute_strategy)
            out[self.numeric_features] = self.numeric_imputer.fit_transform(out[self.numeric_features])
        if self.categorical_features:
            if self.config.categorical_impute_strategy == "constant":
                self.categorical_imputer = SimpleImputer(
                    strategy="constant",
                    fill_value=self.config.categorical_constant_fill,
                )
            else:
                self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            cat_df = out[self.categorical_features].apply(lambda s: s.map(lambda v: v if pd.isna(v) else str(v)))
            out[self.categorical_features] = self.categorical_imputer.fit_transform(cat_df)
        return out

    def _transform_imputers(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.numeric_features and self.numeric_imputer is not None:
            out[self.numeric_features] = self.numeric_imputer.transform(out[self.numeric_features])
        if self.categorical_features and self.categorical_imputer is not None:
            cat_df = out[self.categorical_features].apply(lambda s: s.map(lambda v: v if pd.isna(v) else str(v)))
            out[self.categorical_features] = self.categorical_imputer.transform(cat_df)
        return out

    def _fit_outlier_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.numeric_features:
            out["is_outlier"] = 0
            return out
        self.outlier_scaler = StandardScaler()
        Xs = self.outlier_scaler.fit_transform(out[self.numeric_features].astype(float))
        self.outlier_model = IsolationForest(
            contamination=min(max(self.config.contamination, 0.001), 0.5),
            random_state=self.config.random_state,
        )
        pred = self.outlier_model.fit_predict(Xs)
        out["is_outlier"] = (pred == -1).astype(np.int8)
        return out

    def _transform_outlier(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.numeric_features or self.outlier_scaler is None or self.outlier_model is None:
            out["is_outlier"] = 0
            return out
        Xs = self.outlier_scaler.transform(out[self.numeric_features].astype(float))
        pred = self.outlier_model.predict(Xs)
        out["is_outlier"] = (pred == -1).astype(np.int8)
        return out

    def _fit_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.numeric_features:
            return out
        self.scaler = MinMaxScaler() if self.config.scaler == "minmax" else StandardScaler()
        out[self.numeric_features] = self.scaler.fit_transform(out[self.numeric_features].astype(float))
        return out

    def _transform_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.numeric_features or self.scaler is None:
            return out
        out[self.numeric_features] = self.scaler.transform(out[self.numeric_features].astype(float))
        return out

    def _fit_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cats = [c for c in self.categorical_features if c in out.columns]
        if not cats or self.config.encoding == "none":
            return out
        base = out.drop(columns=cats)
        sub = out[cats].astype(str)
        if self.config.encoding == "onehot":
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            arr = self.encoder.fit_transform(sub)
            names = self.encoder.get_feature_names_out(cats)
            return pd.concat([base.reset_index(drop=True), pd.DataFrame(arr, columns=names)], axis=1)
        if self.config.encoding == "ordinal":
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            arr = self.encoder.fit_transform(sub)
            for i, c in enumerate(cats):
                base[f"{c}_ordinal"] = arr[:, i]
            return base
        # binary_bits optional (advanced)
        for c in cats:
            vals = sub[c].fillna("missing").astype(str).replace({"nan": "missing", "<NA>": "missing"})
            uniques = sorted(vals.unique().tolist())
            self.binary_bits_mapping[c] = uniques
            base[f"{c}_binary"] = vals.map(lambda v, u=uniques: _binary_bit_string(v, u))
        return base

    def _transform_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        cats = [c for c in self.categorical_features if c in out.columns]
        if not cats or self.config.encoding == "none":
            return out
        base = out.drop(columns=cats)
        sub = out[cats].astype(str)
        if self.config.encoding == "onehot" and self.encoder is not None:
            arr = self.encoder.transform(sub)
            names = self.encoder.get_feature_names_out(cats)
            return pd.concat([base.reset_index(drop=True), pd.DataFrame(arr, columns=names)], axis=1)
        if self.config.encoding == "ordinal" and self.encoder is not None:
            arr = self.encoder.transform(sub)
            for i, c in enumerate(cats):
                base[f"{c}_ordinal"] = arr[:, i]
            return base
        for c in cats:
            uniques = self.binary_bits_mapping.get(c, [])
            vals = sub[c].fillna("missing").astype(str).replace({"nan": "missing", "<NA>": "missing"})
            base[f"{c}_binary"] = vals.map(lambda v, u=uniques: _binary_bit_string(v, u))
        return base

    def fit(self, train_df: pd.DataFrame) -> ModelReadyPreprocessor:
        self.log = []
        self._validate(train_df)
        work = self._prepare_core(train_df, for_fit=True)
        work = self._fit_imputers(work)
        self.log.append("Fitted imputers on train data.")
        work = self._fit_outlier_scaler(work)
        self.log.append("Fitted IsolationForest on train data.")
        work = self._fit_scale(work)
        self.log.append(f"Fitted {self.config.scaler} scaler on numeric train features.")
        _ = self._fit_encode(work)
        self.log.append(f"Fitted categorical encoder: {self.config.encoding}.")
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Preprocessor is not fitted. Call fit() first.")
        missing_cols = [c for c in self.feature_columns_seen if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for transform: {missing_cols}")
        work = self._prepare_core(df, for_fit=False)
        work = self._transform_imputers(work)
        work = self._transform_outlier(work)
        work = self._transform_scale(work)
        work = self._transform_encode(work)
        if self.config.round_decimals is not None:
            work = round_float_columns(work, self.config.round_decimals, skip_columns={"is_outlier"})
        return work

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.log = []
        self._validate(train_df)
        work = self._prepare_core(train_df, for_fit=True)
        work = self._fit_imputers(work)
        self.log.append("Fitted imputers on train data.")
        work = self._fit_outlier_scaler(work)
        self.log.append("Fitted IsolationForest on train data.")
        work = self._fit_scale(work)
        self.log.append(f"Fitted {self.config.scaler} scaler on numeric train features.")
        work = self._fit_encode(work)
        self.log.append(f"Fitted categorical encoder: {self.config.encoding}.")
        if self.config.round_decimals is not None:
            work = round_float_columns(work, self.config.round_decimals, skip_columns={"is_outlier"})
        self.fitted = True
        return work

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> ModelReadyPreprocessor:
        obj = joblib.load(path)
        if not isinstance(obj, ModelReadyPreprocessor):
            raise TypeError("Artifact is not a ModelReadyPreprocessor.")
        return obj


def run_pipeline(df: pd.DataFrame, config: PipelineConfig | None = None) -> dict[str, Any]:
    """Backward-compatible one-call pipeline for app/demo use."""
    cfg = config or PipelineConfig()
    profile_before = profile_dataframe(df, "before")
    processor = ModelReadyPreprocessor(cfg)
    cleaned = processor.fit_transform(df)
    profile_after = profile_dataframe(cleaned, "after")
    html = profile_to_html(profile_after)
    report: dict[str, Any] = {
        "profile_before": profile_before,
        "profile_after": profile_after,
        "config": asdict(cfg),
        "pipeline_log": processor.log,
        "validation": processor.validation,
        "n_rows_in": int(profile_before["n_rows"]),
        "n_rows_out": int(len(cleaned)),
        "summary": {
            "duplicates_removed": max(0, int(profile_before["n_rows"]) - int(len(df.drop_duplicates()))),
            "outliers_flagged": int(cleaned["is_outlier"].sum()) if "is_outlier" in cleaned.columns else 0,
            "columns_transformed": len(cleaned.columns),
        },
    }
    return {
        "cleaned_df": cleaned,
        "profile_before": profile_before,
        "profile_after": profile_after,
        "report_json": report,
        "profile_html": html,
        "log": processor.log,
        "processor": processor,
        "scaler": processor.scaler,
    }
