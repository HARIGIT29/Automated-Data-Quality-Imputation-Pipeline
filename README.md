# Automated Data Quality & Imputation Pipeline

Industry-style preprocessing system for tabular ML data with:
- schema/data-quality validation
- train/test-safe `fit()` / `transform()` preprocessing
- configurable imputation, outlier flags, scaling, and encoding
- low-cardinality vs high-cardinality categorical handling
- date feature engineering (`year/month/day/quarter/dayofweek`)
- strict text normalization (trim, punctuation cleanup, lowercase canonicalization)
- artifact persistence (`.joblib`) for inference reuse
- Streamlit dashboard with quality summaries and downloads

## Quick start

```bash
cd Automated-Data-Quality-Imputation-Pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Deterministic setup (for exact reproduction):

```bash
pip install -r requirements-pinned.txt
```

## Core architecture

| Path | Role |
|------|------|
| `src/pipeline.py` | `PipelineConfig`, `ModelReadyPreprocessor` (`fit/transform/save/load`), `run_pipeline` |
| `src/validation.py` | schema + data-quality checks (errors/warnings/summary) |
| `src/load_save.py` | robust CSV/Excel loading, JSON + joblib persistence |
| `src/profile.py` | before/after profiling + HTML report |
| `ARCHITECTURE.md` | architecture and data flow diagram |
| `app.py` | Streamlit workflow UI and dashboard |
| `tests/` | unit tests including fit/transform + persistence |
| `docs/VIVA_UPGRADE_SUMMARY.md` | viva talking points for production upgrades |

## Model-ready defaults

- Encoding default: **`onehot`** (numeric-safe output).
- Numeric imputation default: **`median`**.
- Optional advanced mode: `binary_bits` (string output) for presentation use cases.
- High-cardinality default: **frequency encoding** (prevents very wide sparse output).
- Outlier default: **flag** (`is_outlier`) with optional **remove** or **cap** mode.

## Train/test-safe workflow

`ModelReadyPreprocessor` supports:
- `fit(train_df)`
- `transform(test_df)`
- `fit_transform(train_df)`
- `save(path)` / `load(path)`

This avoids leakage from fitting imputers/scalers on the full dataset during model evaluation.

## Streamlit features

- Model-ready preset
- Optional train/test split demo mode
- target/id/exclude/force-categorical controls
- low-cardinality threshold + high-cardinality strategy controls
- outlier action control (`flag` / `remove` / `cap`)
- validation warnings (high cardinality, all-null columns, mixed object types, etc.)
- quality summary metrics and missingness visualization
- dual downloads: `cleaned_raw.csv` and `model_ready.csv`
- report downloads: `report.json`, `profile_after.html`, `preprocessor.joblib`
- model-ready output is numeric-only by default

## Testing

```bash
python -m pytest tests -v
```

## Deployment checklist

- Push repo to GitHub and connect Streamlit Cloud.
- Verify `app.py` is configured as main file.
- Use `requirements-pinned.txt` when deterministic environment is required.
- Keep `.joblib` artifacts out of git (already ignored).

## Notes

- CSV loader tries UTF-8, UTF-8 BOM, cp1252, cp1250, and latin variants.
- `is_outlier` is a flag (rows are not automatically dropped).
- `binary_bits` is optional and not recommended as default ML input.
- For production training, split data first and fit only on train data.
