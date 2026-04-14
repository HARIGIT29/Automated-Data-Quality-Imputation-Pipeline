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
from src.preprocessing import coerce_numeric_like_columns, expand_date_features, infer_date_columns, normalize_object_columns
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
    low_cardinality_threshold: int = 25
    high_cardinality_strategy: str = "frequency"  # "frequency" | "ordinal" | "hash" | "exclude"
    hash_bins: int = 32
    outlier_action: str = "flag"  # "flag" | "remove" | "cap"
    ensure_model_ready_numeric: bool = True
    drop_invalid_rows: bool = True
    forced_numeric_columns: list[str] = field(
        default_factory=lambda: ["funds_raised_millions", "total_laid_off", "percentage_laid_off"]
    )


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
        self.high_card_encoder: OrdinalEncoder | None = None
        self.cap_bounds: dict[str, tuple[float, float]] = {}

        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.low_card_features: list[str] = []
        self.high_card_features: list[str] = []
        self.passthrough_columns: list[str] = []
        self.force_cat: set[str] = set()
        self.binary_bits_mapping: dict[str, list[str]] = {}
        self.frequency_maps: dict[str, dict[str, float]] = {}
        self.feature_columns_seen: list[str] = []
        self.created_date_features: list[str] = []
        self.fitted: bool = False
        self.duplicates_removed_count: int = 0
        self.invalid_rows_removed_count: int = 0
        self.outliers_removed_count: int = 0
        self.outliers_flagged_count: int = 0
        self.numeric_imputed_cells: int = 0
        self.categorical_imputed_cells: int = 0
        self.encoded_features_created: int = 0

    def _validate(self, df: pd.DataFrame) -> None:
        v = validate_dataframe(df, high_cardinality_threshold=self.config.high_cardinality_threshold)
        self.validation = v
        for w in v["warnings"]:
            self.log.append(f"Warning: {w}")
        if v["errors"]:
            raise ValueError("Validation failed: " + "; ".join(v["errors"]))

    def _prepare_core(self, df: pd.DataFrame, *, for_fit: bool) -> pd.DataFrame:
        working = df.copy()

        excluded = set(self.config.exclude_from_features)
        if self.config.target_column:
            excluded.add(self.config.target_column)
        if self.config.id_column:
            excluded.add(self.config.id_column)

        self.force_cat = set(self.config.force_categorical_columns) - excluded
        skip_coerce = self.force_cat | excluded
        object_cols = [c for c in working.columns if working[c].dtype == object]
        working = normalize_object_columns(working, object_cols)
        self.log.append("Normalized object columns (strip/regex cleanup/lowercase).")

        date_cols = infer_date_columns(working)
        working, created = expand_date_features(working, date_cols)
        if created:
            self.log.append(f"Converted date columns to numeric features: {created}.")
        if for_fit:
            self.created_date_features = created
        else:
            # Keep schema consistency for transform
            for col in self.created_date_features:
                if col not in working.columns:
                    working[col] = np.nan

        working, forced_numeric_applied = coerce_numeric_like_columns(
            working,
            explicit_columns=self.config.forced_numeric_columns,
        )
        if forced_numeric_applied:
            self.log.append(f"Forced numeric coercion on columns: {forced_numeric_applied}.")

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

        if self.config.drop_invalid_rows:
            feature_cols = [c for c in num_features + cat_features if c in working.columns]
            if feature_cols:
                before_invalid = len(working)
                working = working[working[feature_cols].notna().any(axis=1)].reset_index(drop=True)
                removed_invalid = before_invalid - len(working)
                if for_fit:
                    self.invalid_rows_removed_count = removed_invalid
                if removed_invalid:
                    self.log.append(f"Removed {removed_invalid} invalid rows (all feature values missing).")

        if for_fit:
            self.numeric_features = num_features
            self.categorical_features = cat_features
            low: list[str] = []
            high: list[str] = []
            for c in cat_features:
                n_unique = int(working[c].nunique(dropna=True))
                if n_unique <= self.config.low_cardinality_threshold:
                    low.append(c)
                else:
                    high.append(c)
            self.low_card_features = low
            self.high_card_features = high
            self.passthrough_columns = [c for c in working.columns if c not in num_features + cat_features]
            self.feature_columns_seen = list(working.columns)
            if self.high_card_features:
                self.log.append(
                    f"High-cardinality columns ({self.config.high_cardinality_strategy}): {self.high_card_features}."
                )
        return working

    def _fit_imputers(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.numeric_features:
            before_missing = int(out[self.numeric_features].isna().sum().sum())
            if self.config.numeric_impute_strategy == "knn":
                self.numeric_imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
            else:
                self.numeric_imputer = SimpleImputer(strategy=self.config.numeric_impute_strategy)
            out[self.numeric_features] = self.numeric_imputer.fit_transform(out[self.numeric_features])
            after_missing = int(out[self.numeric_features].isna().sum().sum())
            self.numeric_imputed_cells = max(0, before_missing - after_missing)
        if self.categorical_features:
            before_missing_cat = int(out[self.categorical_features].isna().sum().sum())
            if self.config.categorical_impute_strategy == "constant":
                self.categorical_imputer = SimpleImputer(
                    strategy="constant",
                    fill_value=self.config.categorical_constant_fill,
                )
            else:
                self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            cat_df = out[self.categorical_features].apply(lambda s: s.map(lambda v: v if pd.isna(v) else str(v)))
            out[self.categorical_features] = self.categorical_imputer.fit_transform(cat_df)
            after_missing_cat = int(out[self.categorical_features].isna().sum().sum())
            self.categorical_imputed_cells = max(0, before_missing_cat - after_missing_cat)
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
        self.outliers_flagged_count = int(out["is_outlier"].sum())
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

    def _apply_outlier_action(self, df: pd.DataFrame, *, for_fit: bool) -> pd.DataFrame:
        out = df.copy()
        if "is_outlier" not in out.columns:
            return out
        if self.config.outlier_action == "remove":
            before = len(out)
            out = out[out["is_outlier"] == 0].reset_index(drop=True)
            removed = before - len(out)
            if for_fit:
                self.outliers_removed_count = removed
            phase = "fit" if for_fit else "transform"
            self.log.append(f"Removed {removed} outlier rows during {phase} phase.")
        elif self.config.outlier_action == "cap":
            # Cap numeric features based on train quantiles.
            if for_fit:
                self.cap_bounds = {}
                for c in self.numeric_features:
                    if c in out.columns:
                        lo = float(out[c].quantile(0.01))
                        hi = float(out[c].quantile(0.99))
                        self.cap_bounds[c] = (lo, hi)
            capped_cells = 0
            for c, (lo, hi) in self.cap_bounds.items():
                if c not in out.columns:
                    continue
                before_vals = out[c].copy()
                out[c] = out[c].clip(lower=lo, upper=hi)
                capped_cells += int((before_vals != out[c]).sum())
            phase = "fit" if for_fit else "transform"
            self.log.append(f"Capped {capped_cells} numeric cells during {phase} phase.")
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
        low = [c for c in self.low_card_features if c in out.columns]
        high = [c for c in self.high_card_features if c in out.columns]
        cats = low + high
        if not cats or self.config.encoding == "none":
            return out
        base = out.drop(columns=cats)

        # Low-card strategy by selected encoding mode
        low_df = out[low].astype(str) if low else pd.DataFrame(index=out.index)
        if low:
            if self.config.encoding == "onehot":
                self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                arr = self.encoder.fit_transform(low_df)
                names = self.encoder.get_feature_names_out(low)
                base = pd.concat([base.reset_index(drop=True), pd.DataFrame(arr, columns=names)], axis=1)
            elif self.config.encoding == "ordinal":
                self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                arr = self.encoder.fit_transform(low_df)
                for i, c in enumerate(low):
                    base[f"{c}_ordinal"] = arr[:, i]
            elif self.config.encoding == "binary_bits":
                for c in low:
                    vals = low_df[c].fillna("missing").astype(str).replace({"nan": "missing", "<NA>": "missing"})
                    uniques = sorted(vals.unique().tolist())
                    self.binary_bits_mapping[c] = uniques
                    base[f"{c}_binary"] = vals.map(lambda v, u=uniques: _binary_bit_string(v, u))
            else:
                for c in low:
                    base[c] = out[c]

        # High-card strategy
        high_df = out[high].astype(str) if high else pd.DataFrame(index=out.index)
        if high:
            if self.config.high_cardinality_strategy == "frequency":
                for c in high:
                    freq = high_df[c].value_counts(normalize=True).to_dict()
                    self.frequency_maps[c] = {str(k): float(v) for k, v in freq.items()}
                    base[f"{c}_freq"] = high_df[c].map(self.frequency_maps[c]).fillna(0.0)
            elif self.config.high_cardinality_strategy == "ordinal":
                self.high_card_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                arr = self.high_card_encoder.fit_transform(high_df)
                for i, c in enumerate(high):
                    base[f"{c}_high_ordinal"] = arr[:, i]
            elif self.config.high_cardinality_strategy == "hash":
                bins = max(2, int(self.config.hash_bins))
                for c in high:
                    base[f"{c}_hash"] = high_df[c].map(lambda v, b=bins: hash(v) % b)
            elif self.config.high_cardinality_strategy == "exclude":
                pass
            else:
                raise ValueError(f"Unknown high_cardinality_strategy: {self.config.high_cardinality_strategy}")
        self.encoded_features_created = max(0, int(base.shape[1] - out.shape[1]))
        return base

    def _transform_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        low = [c for c in self.low_card_features if c in out.columns]
        high = [c for c in self.high_card_features if c in out.columns]
        cats = low + high
        if not cats or self.config.encoding == "none":
            return out
        base = out.drop(columns=cats)
        low_df = out[low].astype(str) if low else pd.DataFrame(index=out.index)
        high_df = out[high].astype(str) if high else pd.DataFrame(index=out.index)

        if low:
            if self.config.encoding == "onehot" and self.encoder is not None:
                arr = self.encoder.transform(low_df)
                names = self.encoder.get_feature_names_out(low)
                base = pd.concat([base.reset_index(drop=True), pd.DataFrame(arr, columns=names)], axis=1)
            elif self.config.encoding == "ordinal" and self.encoder is not None:
                arr = self.encoder.transform(low_df)
                for i, c in enumerate(low):
                    base[f"{c}_ordinal"] = arr[:, i]
            elif self.config.encoding == "binary_bits":
                for c in low:
                    uniques = self.binary_bits_mapping.get(c, [])
                    vals = low_df[c].fillna("missing").astype(str).replace({"nan": "missing", "<NA>": "missing"})
                    base[f"{c}_binary"] = vals.map(lambda v, u=uniques: _binary_bit_string(v, u))
            else:
                for c in low:
                    base[c] = out[c]

        if high:
            if self.config.high_cardinality_strategy == "frequency":
                for c in high:
                    fmap = self.frequency_maps.get(c, {})
                    base[f"{c}_freq"] = high_df[c].map(fmap).fillna(0.0)
            elif self.config.high_cardinality_strategy == "ordinal" and self.high_card_encoder is not None:
                arr = self.high_card_encoder.transform(high_df)
                for i, c in enumerate(high):
                    base[f"{c}_high_ordinal"] = arr[:, i]
            elif self.config.high_cardinality_strategy == "hash":
                bins = max(2, int(self.config.hash_bins))
                for c in high:
                    base[f"{c}_hash"] = high_df[c].map(lambda v, b=bins: hash(v) % b)
            elif self.config.high_cardinality_strategy == "exclude":
                pass
        return base

    def _ensure_numeric_model_ready(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee model-ready output is numeric-only when enabled."""
        if not self.config.ensure_model_ready_numeric:
            return df
        out = df.copy()
        non_numeric = [c for c in out.columns if not pd.api.types.is_numeric_dtype(out[c])]
        if non_numeric:
            self.log.append(f"Dropped non-numeric columns for model-ready output: {non_numeric}")
            out = out.drop(columns=non_numeric)
        return out

    def fit(self, train_df: pd.DataFrame) -> ModelReadyPreprocessor:
        self.log = []
        self._validate(train_df)
        work = self._prepare_core(train_df, for_fit=True)
        work = self._fit_imputers(work)
        self.log.append("Fitted imputers on train data.")
        if self.config.drop_duplicates:
            before = len(work)
            work = drop_duplicate_rows(work)
            self.duplicates_removed_count = before - len(work)
            self.log.append(f"Dropped {self.duplicates_removed_count} duplicate rows after imputation.")
        work = self._fit_outlier_scaler(work)
        self.log.append("Fitted IsolationForest on train data.")
        work = self._apply_outlier_action(work, for_fit=True)
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
        work = self._apply_outlier_action(work, for_fit=False)
        work = self._transform_scale(work)
        work = self._transform_encode(work)
        work = self._ensure_numeric_model_ready(work)
        if self.config.round_decimals is not None:
            work = round_float_columns(work, self.config.round_decimals, skip_columns={"is_outlier"})
        return work

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.log = []
        self._validate(train_df)
        work = self._prepare_core(train_df, for_fit=True)
        work = self._fit_imputers(work)
        self.log.append("Fitted imputers on train data.")
        if self.config.drop_duplicates:
            before = len(work)
            work = drop_duplicate_rows(work)
            self.duplicates_removed_count = before - len(work)
            self.log.append(f"Dropped {self.duplicates_removed_count} duplicate rows after imputation.")
        work = self._fit_outlier_scaler(work)
        self.log.append("Fitted IsolationForest on train data.")
        work = self._apply_outlier_action(work, for_fit=True)
        self.cleaned_raw_after_fit = work.copy()
        work = self._fit_scale(work)
        self.log.append(f"Fitted {self.config.scaler} scaler on numeric train features.")
        work = self._fit_encode(work)
        work = self._ensure_numeric_model_ready(work)
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
    model_ready = processor.fit_transform(df)
    cleaned_raw = getattr(processor, "cleaned_raw_after_fit", df.copy())
    profile_after = profile_dataframe(model_ready, "after")
    html = profile_to_html(profile_after)
    duplicates_removed = int(processor.duplicates_removed_count)
    outliers_flagged = int(processor.outliers_flagged_count)
    outliers_removed = int(processor.outliers_removed_count)
    invalid_removed = int(processor.invalid_rows_removed_count)
    rows_after_dups = int(profile_before["n_rows"]) - duplicates_removed
    report: dict[str, Any] = {
        "profile_before": profile_before,
        "profile_after": profile_after,
        "config": asdict(cfg),
        "pipeline_log": processor.log,
        "validation": processor.validation,
        "n_rows_in": int(profile_before["n_rows"]),
        "n_rows_out": int(len(model_ready)),
        "summary": {
            "duplicates_removed": duplicates_removed,
            "invalid_rows_removed": invalid_removed,
            "rows_after_duplicates": rows_after_dups,
            "missing_before": int(df.isna().sum().sum()),
            "missing_after_raw": int(cleaned_raw.isna().sum().sum()),
            "missing_after_model_ready": int(model_ready.isna().sum().sum()),
            "numeric_imputed_cells": int(processor.numeric_imputed_cells),
            "categorical_imputed_cells": int(processor.categorical_imputed_cells),
            "outliers_flagged": outliers_flagged,
            "outliers_removed": outliers_removed,
            "encoded_features_created": int(processor.encoded_features_created),
            "columns_created_model_ready": int(len(model_ready.columns) - len(df.columns)),
            "cleaned_raw_shape": [int(cleaned_raw.shape[0]), int(cleaned_raw.shape[1])],
            "model_ready_shape": [int(model_ready.shape[0]), int(model_ready.shape[1])],
            "high_cardinality_columns": processor.high_card_features,
            "low_cardinality_columns": processor.low_card_features,
        },
    }
    return {
        "cleaned_df": model_ready,  # backward compatibility alias
        "cleaned_raw_df": cleaned_raw,
        "model_ready_df": model_ready,
        "profile_before": profile_before,
        "profile_after": profile_after,
        "report_json": report,
        "profile_html": html,
        "log": processor.log,
        "processor": processor,
        "scaler": processor.scaler,
    }
