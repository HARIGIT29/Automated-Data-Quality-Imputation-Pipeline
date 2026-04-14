"""Streamlit UI for industry-style preprocessing workflow."""

from __future__ import annotations

import io
import json

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src.load_save import read_csv_bytes
from src.pipeline import ModelReadyPreprocessor, PipelineConfig, run_pipeline

st.set_page_config(page_title="Data Quality Pipeline", layout="wide")
st.title("Automated Data Quality & Imputation Pipeline")
st.caption("Model-ready preprocessing with train/test-safe fit/transform and artifact export.")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

with st.sidebar:
    st.header("Workflow")
    model_ready_preset = st.checkbox("Model-ready preset", value=True)
    split_mode = st.checkbox("Train/test split workflow", value=False)
    test_size = st.slider("Test split ratio", 0.1, 0.5, 0.2, 0.05) if split_mode else 0.2

    st.header("Preprocessing")
    num_impute = st.selectbox("Numeric missing values", ["median", "mean", "knn"], index=0)
    knn_k = st.number_input("k-NN neighbors", min_value=1, max_value=50, value=5, step=1) if num_impute == "knn" else 5
    cat_impute = st.selectbox("Categorical missing values", ["most_frequent", "constant"], index=0)
    cat_constant_fill = st.text_input("Constant fill for categoricals", value="missing") if cat_impute == "constant" else "missing"
    contamination = st.slider("Isolation Forest contamination", 0.01, 0.5, 0.05, 0.01)
    scaler = st.selectbox("Scaler", ["standard", "minmax"], index=0)
    enc_default_idx = 0 if model_ready_preset else 2
    encoding = st.selectbox(
        "Categorical encoding",
        ["onehot", "ordinal", "binary_bits", "none"],
        index=enc_default_idx,
        help="Use onehot for model-ready defaults. binary_bits is advanced and returns strings.",
    )
    do_round = st.checkbox("Round numeric columns", value=True)
    round_decimals = st.number_input("Decimal places", min_value=0, max_value=10, value=4) if do_round else None
    drop_dup = st.checkbox("Drop duplicates in fit data", value=True)

    st.header("Column controls")
    target_col = st.text_input("Target column (optional)")
    id_col = st.text_input("ID column (optional)")
    force_cat_raw = st.text_input("Always categorical (comma-separated)")
    exclude_raw = st.text_input("Exclude columns (comma-separated)")
    high_card = st.number_input("High-cardinality warning threshold", min_value=10, max_value=10000, value=100, step=10)

run_btn = st.button("Run pipeline", type="primary")

if uploaded is None:
    st.info("Upload a dataset to begin.")
    st.stop()

try:
    suffix = uploaded.name.lower().split(".")[-1]
    raw = uploaded.getvalue()
    if suffix in {"xlsx", "xls"}:
        df_in = pd.read_excel(io.BytesIO(raw))
        csv_encoding = None
    elif suffix == "csv":
        df_in, csv_encoding = read_csv_bytes(raw)
    else:
        st.error("Unsupported file format. Upload CSV, XLSX, or XLS.")
        st.stop()
except Exception as exc:
    st.error(f"Failed to load file: {exc}")
    st.stop()

if df_in.empty:
    st.error("Uploaded file has no rows.")
    st.stop()

st.subheader("Raw preview")
if csv_encoding:
    hint = " — *Excel-on-Windows CSVs often need cp1252/latin-1.*" if csv_encoding not in ("utf-8", "utf-8-sig") else ""
    st.caption(f"CSV decoded with encoding: **{csv_encoding}**{hint}")
st.dataframe(df_in.head(20), use_container_width=True)

exclude_list = [c.strip() for c in exclude_raw.split(",") if c.strip()]
force_cat_list = [c.strip() for c in force_cat_raw.split(",") if c.strip()]

if run_btn:
    cfg = PipelineConfig(
        knn_neighbors=int(knn_k),
        contamination=float(contamination),
        scaler=scaler,
        encoding=encoding,
        drop_duplicates=drop_dup,
        exclude_from_features=exclude_list,
        numeric_impute_strategy=num_impute,
        categorical_impute_strategy=cat_impute,
        categorical_constant_fill=cat_constant_fill,
        force_categorical_columns=force_cat_list,
        round_decimals=(int(round_decimals) if do_round and round_decimals is not None else None),
        target_column=(target_col.strip() or None),
        id_column=(id_col.strip() or None),
        high_cardinality_threshold=int(high_card),
    )
    try:
        with st.spinner("Running pipeline..."):
            if split_mode:
                train_df, test_df = train_test_split(df_in, test_size=test_size, random_state=cfg.random_state)
                processor = ModelReadyPreprocessor(cfg).fit(train_df)
                cleaned_train = processor.transform(train_df)
                cleaned_test = processor.transform(test_df)
                result = run_pipeline(df_in, cfg)
                result["cleaned_train_df"] = cleaned_train
                result["cleaned_test_df"] = cleaned_test
                result["split_info"] = {"train_rows": len(train_df), "test_rows": len(test_df)}
                result["processor"] = processor
            else:
                result = run_pipeline(df_in, cfg)
        st.session_state["result"] = result
        st.session_state["cfg"] = cfg
    except Exception as exc:
        st.error(f"Pipeline failed safely: {exc}")
        st.stop()

if "result" in st.session_state:
    r = st.session_state["result"]
    cleaned = r["cleaned_df"]
    report = r["report_json"]
    processor = r.get("processor")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Quality summary", "Before / After", "Outliers", "Visuals", "Download"]
    )

    with tab1:
        st.markdown("### Quality summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows in", report["n_rows_in"])
        c2.metric("Rows out", report["n_rows_out"])
        c3.metric("Duplicates removed", report["summary"]["duplicates_removed"])
        c4.metric("Outliers flagged", report["summary"]["outliers_flagged"])
        st.json(report.get("validation", {}))
        st.markdown("### Pipeline log")
        for line in r["log"]:
            st.text(line)
        if "split_info" in r:
            st.info(f"Train/Test mode: {r['split_info']}")

    with tab2:
        st.markdown("#### Cleaned data sample")
        st.dataframe(cleaned.head(30), use_container_width=True)
        st.metric("Output columns", len(cleaned.columns))
        if "cleaned_train_df" in r and "cleaned_test_df" in r:
            st.markdown("#### Train sample")
            st.dataframe(r["cleaned_train_df"].head(10), use_container_width=True)
            st.markdown("#### Test sample")
            st.dataframe(r["cleaned_test_df"].head(10), use_container_width=True)

    with tab3:
        if "is_outlier" in cleaned.columns:
            n_out = int(cleaned["is_outlier"].sum())
            st.metric("Rows flagged as outliers", n_out)
            st.dataframe(cleaned[cleaned["is_outlier"] == 1].head(50), use_container_width=True)
        else:
            st.write("No outlier column (no numeric features).")

    with tab4:
        st.markdown("#### Missingness before vs after")
        missing_before = pd.Series(report["profile_before"]["columns"]).apply(lambda x: x["missing_count"])
        missing_after = pd.Series(report["profile_after"]["columns"]).apply(lambda x: x["missing_count"])
        mdf = pd.DataFrame({"before": missing_before, "after": missing_after}).fillna(0)
        st.bar_chart(mdf)
        st.markdown("#### Category frequency (top categorical-like column)")
        non_num = cleaned.select_dtypes(exclude=["number", "bool"])
        if not non_num.empty:
            col = non_num.columns[0]
            st.write(f"Column: `{col}`")
            st.bar_chart(non_num[col].value_counts().head(20))
        else:
            st.caption("No string categorical columns after encoding.")

    with tab5:
        csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button("Download cleaned CSV", data=csv_bytes, file_name="cleaned.csv", mime="text/csv")
        st.download_button(
            "Download report JSON",
            data=json.dumps(report, indent=2, default=str),
            file_name="pipeline_report.json",
            mime="application/json",
        )
        st.download_button("Download profile HTML", data=r["profile_html"], file_name="profile_after.html", mime="text/html")
        if processor is not None:
            artifact_buf = io.BytesIO()
            joblib.dump(processor, artifact_buf)
            st.download_button(
                "Download preprocessor artifact (.joblib)",
                data=artifact_buf.getvalue(),
                file_name="preprocessor.joblib",
                mime="application/octet-stream",
            )
else:
    st.caption("Click **Run pipeline** to generate cleaned data and reports.")
