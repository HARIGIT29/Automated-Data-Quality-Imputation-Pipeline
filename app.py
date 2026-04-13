"""
Streamlit UI: upload CSV/Excel, configure pipeline, view profiles, download artifacts.
Run: streamlit run app.py
"""

from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

from src.load_save import read_csv_bytes
from src.pipeline import PipelineConfig, run_pipeline

st.set_page_config(page_title="Data Quality Pipeline", layout="wide")
st.title("Automated Data Quality & Imputation Pipeline")
st.caption("Profile → dedupe → types → k-NN + mode impute → Isolation Forest → scale → encode")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

with st.sidebar:
    st.header("Parameters")
    knn_k = st.number_input("k-NN neighbors", min_value=1, max_value=50, value=5, step=1)
    contamination = st.slider("Isolation Forest contamination", 0.01, 0.5, 0.05, 0.01)
    scaler = st.selectbox("Scaler", ["standard", "minmax"])
    encoding = st.selectbox("Categorical encoding", ["onehot", "ordinal", "none"])
    drop_dup = st.checkbox("Drop duplicate rows", value=True)
    exclude_raw = st.text_input(
        "Exclude columns (comma-separated, optional)",
        help="IDs or targets left out of IF / scaling / encoding lists where applicable",
    )

run_btn = st.button("Run pipeline", type="primary")

if uploaded is None:
    st.info("Upload a dataset to begin.")
    st.stop()

suffix = uploaded.name.lower().split(".")[-1]
raw = uploaded.getvalue()
if suffix in {"xlsx", "xls"}:
    df_in = pd.read_excel(io.BytesIO(raw))
    csv_encoding = None
else:
    df_in, csv_encoding = read_csv_bytes(raw)

st.subheader("Raw preview")
if csv_encoding:
    hint = ""
    if csv_encoding not in ("utf-8", "utf-8-sig"):
        hint = " — *Excel-on-Windows CSVs often need cp1252 or latin-1.*"
    st.caption(f"CSV decoded with encoding: **{csv_encoding}**{hint}")
st.dataframe(df_in.head(20), use_container_width=True)

exclude_list = [c.strip() for c in exclude_raw.split(",") if c.strip()]

if run_btn:
    cfg = PipelineConfig(
        knn_neighbors=int(knn_k),
        contamination=float(contamination),
        scaler=scaler,
        encoding=encoding,
        drop_duplicates=drop_dup,
        exclude_from_features=exclude_list,
    )
    with st.spinner("Running pipeline..."):
        result = run_pipeline(df_in, cfg)

    st.session_state["result"] = result
    st.session_state["cfg"] = cfg

if "result" in st.session_state:
    r = st.session_state["result"]
    cleaned = r["cleaned_df"]
    report = r["report_json"]

    tab1, tab2, tab3, tab4 = st.tabs(["Profile & log", "Before / After", "Outliers", "Download"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Before")
            st.json(
                {
                    "n_rows": report["profile_before"]["n_rows"],
                    "n_columns": report["profile_before"]["n_columns"],
                    "duplicate_rows": report["profile_before"]["duplicate_rows"],
                }
            )
        with c2:
            st.markdown("### After")
            st.json(
                {
                    "n_rows": report["n_rows_out"],
                    "n_columns": len(cleaned.columns),
                }
            )
        st.markdown("### Pipeline log")
        for line in r["log"]:
            st.text(line)

    with tab2:
        st.markdown("#### Sample of cleaned data")
        st.dataframe(cleaned.head(30), use_container_width=True)
        st.metric("Output columns", len(cleaned.columns))

    with tab3:
        if "is_outlier" in cleaned.columns:
            n_out = int(cleaned["is_outlier"].sum())
            st.metric("Rows flagged as outliers", n_out)
            st.dataframe(cleaned[cleaned["is_outlier"] == 1].head(50), use_container_width=True)
        else:
            st.write("No outlier column (no numeric features).")

    with tab4:
        csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download cleaned CSV",
            data=csv_bytes,
            file_name="cleaned.csv",
            mime="text/csv",
        )
        profile_pack = {
            "profile_before": report["profile_before"],
            "profile_after": report["profile_after"],
            "config": report["config"],
            "pipeline_log": report["pipeline_log"],
        }
        st.download_button(
            "Download report JSON",
            data=json.dumps(profile_pack, indent=2, default=str),
            file_name="pipeline_report.json",
            mime="application/json",
        )
        st.download_button(
            "Download profile HTML",
            data=r["profile_html"],
            file_name="profile_after.html",
            mime="text/html",
        )
else:
    st.caption('Click **Run pipeline** to generate cleaned data and downloads.')
