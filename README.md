# Automated Data Quality & Imputation Pipeline

End-to-end **tabular** preprocessing: profiling, duplicate removal, type coercion, **k-NN** imputation for numerics, **mode** imputation for categoricals, **Isolation Forest** outlier flags, **StandardScaler** or **MinMaxScaler**, and optional **one-hot** or **ordinal** encoding. Delivered as a **Streamlit** app with JSON/HTML reports and CSV export.

## Setup

```bash
cd Automated-Data-Quality-Imputation-Pipeline
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Run Streamlit

```bash
streamlit run app.py
```

Open the URL shown in the terminal, upload a **CSV** or **Excel** file, set sidebar options, and click **Run pipeline**. Download **cleaned CSV**, **pipeline_report.json**, and **profile_after.html**.

## Run tests

```bash
python -m pytest tests -v
```

## Notebook demo

```bash
jupyter notebook notebooks/demo.ipynb
```

Ensure the notebook working directory is the project root (see first cell’s path hack), or set `PYTHONPATH` to this folder.

## Project layout

| Path | Role |
|------|------|
| `app.py` | Streamlit UI |
| `src/pipeline.py` | Config + `run_pipeline()` |
| `src/profile.py` | Before/after profiles, HTML |
| `src/impute_knn.py` | k-NN + mode imputation |
| `src/outliers.py` | Isolation Forest flags |
| `src/scale.py` | Standard / MinMax scaling |
| `src/categorical.py` | One-hot / ordinal encoding |
| `tests/` | Pytest |
| `notebooks/demo.ipynb` | Viva demo |
| `docs/ACADEMIC_REPORT_AND_SLIDES.md` | Report & PPT outline |

## Notes

- **CSV encoding**: Uploads try UTF-8 (with BOM), then Windows-1252 and other common encodings so Excel-exported CSVs do not raise `UnicodeDecodeError`.
- **Mixed data**: k-NN runs on numeric columns only; categoricals use mode imputation, then encoding.
- **Outliers**: `is_outlier` column (1 = flagged); rows are not dropped by default.
- **Exclude columns**: comma-separated names in the sidebar omit columns from the feature lists used for IF, scaling, and encoding (IDs/targets—use with care for missing values on those columns).

## Cursor

Open this folder in Cursor; run Streamlit or tests from the integrated terminal. Use `docs/ACADEMIC_REPORT_AND_SLIDES.md` to draft your college report and slides.
