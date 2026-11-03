"""PLL-Smart 2.1 Streamlit engineering analytics dashboard."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    analyze_locktime_values,
    append_history_entry,
    append_history_entries,
    compare_feature_distributions,
    compare_what_if_predictions,
    compute_global_feature_importance,
    create_history_entry,
    dataset_profile,
    explain_prediction,
    predict_batch,
    prediction_from_batch_row,
    read_csv_upload,
    safe_csv_download,
    summarize_drift,
    validate_dataset,
)
from data_utils import (
    FEATURE_COLUMNS,
    GLOBAL_SAFETY_INHIBIT_REASON,
    ILLUSTRATIVE_JITTER_USL_PS,
    ILLUSTRATIVE_SCREEN_THRESHOLD,
    JITTER_DATA_PATH,
    LOCKTIME_DATA_PATH,
    TARGET_COLUMN,
    analyze_locktime_dataset,
    decide_jitter_test,
    load_jitter_partitions,
)
from reporting import build_engineering_report_exports


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "jitter_model.joblib"
METRICS_PATH = ROOT / "outputs" / "jitter_metrics.json"
PREDICTIONS_PATH = ROOT / "outputs" / "test_predictions.csv"
LOCK_METRICS_PATH = ROOT / "outputs" / "locktime_metrics.json"
NAVY = "#081426"
PANEL = "#101f35"
CYAN = "#38d9f5"
BLUE = "#6b9cff"
MUTED = "#93a4bd"
SAFETY_INHIBIT = GLOBAL_SAFETY_INHIBIT_REASON
FEATURE_LABELS = {
    "vco_band_code": "VCO band code",
    "ring_osc_ghz": "Ring oscillator frequency (GHz)",
    "chargepump_ua": "Charge-pump current (µA)",
    "lock_time_us": "Lock time (µs)",
    "vdd_droop_norm": "Supply voltage droop (normalized)",
    "wafer_radius_norm": "Wafer radius (normalized)",
    "leak_screen_norm": "Leakage screening measurement (normalized)",
}

st.set_page_config(
    page_title="PLL-Smart 2.1 | Engineering Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: dark; }}
    .stApp {{ background: {NAVY}; color: #e7eef9; }}
    html, body, [class*="css"] {{ font-family: 'Segoe UI', sans-serif; }}
    h1, h2, h3 {{ font-family: 'Segoe UI', sans-serif; font-weight: 650; letter-spacing: -.025em; }}
    h1 {{ color: #f4f8ff; }}
    [data-testid="stSidebar"] {{ background: #0c192c; border-right: 1px solid #1b2e49; }}
    [data-testid="stMetric"] {{
        background: {PANEL}; border: 1px solid #1b3450; border-radius: 14px;
        padding: 15px 18px; min-height: 108px;
    }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}
    [data-testid="stMetricValue"] {{ color: #f1f7ff; font-weight: 650; }}
    .hero {{
        background: linear-gradient(112deg, #102641 0%, #0c1a2e 62%, #103044 100%);
        border: 1px solid #1d4058; border-radius: 18px; padding: 24px 28px;
        margin: 4px 0 18px 0;
    }}
    .eyebrow {{ color: {CYAN}; font-size: .73rem; font-weight: 700; letter-spacing: .15em; }}
    .hero p {{ color: #a9bad0; margin-bottom: 0; }}
    .small-note {{ color: {MUTED}; font-size: .84rem; }}
    .warning-box {{
        background: #332819; border: 1px solid #77552a; border-radius: 12px;
        padding: 13px 16px; color: #f9dfa5; margin: 9px 0 16px 0;
    }}
    .status-card {{ background: {PANEL}; border: 1px solid #1b3450; border-radius: 12px; padding: 15px; }}
    div[data-testid="stTabs"] button {{ color: #9fb3cb; }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {CYAN}; border-bottom-color: {CYAN};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}.")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_saved_metrics() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    paths = (METRICS_PATH, LOCK_METRICS_PATH, PREDICTIONS_PATH)
    if any(not path.exists() for path in paths):
        raise FileNotFoundError("Saved model evaluation outputs are missing; restore them before launch.")
    return (
        json.loads(METRICS_PATH.read_text(encoding="utf-8")),
        json.loads(LOCK_METRICS_PATH.read_text(encoding="utf-8")),
        pd.read_csv(PREDICTIONS_PATH),
    )


@st.cache_data
def load_reference_partitions() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    return load_jitter_partitions(JITTER_DATA_PATH)


@st.cache_data
def load_reference_locktime() -> tuple[pd.DataFrame, dict[str, Any]]:
    return analyze_locktime_dataset(LOCKTIME_DATA_PATH)


@st.cache_data
def load_uploaded_csv(contents: bytes) -> pd.DataFrame:
    return read_csv_upload(contents)


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#c8d6e8", "family": "Segoe UI"},
        margin={"l": 12, "r": 18, "t": 38, "b": 12},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        xaxis={"gridcolor": "#1c3049", "zerolinecolor": "#34506d"},
        yaxis={"gridcolor": "#1c3049", "zerolinecolor": "#34506d"},
    )
    return fig


def show_metrics(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, delta) in zip(columns, items):
        column.metric(label, value, delta, help=label)


def format_metric(value: Any, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.4f}{suffix}"


def filter_prediction_table(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    c1, c2, c3 = st.columns([1, 1, 1.4])
    statuses = ["All", *sorted(frame["validation_status"].dropna().unique())]
    chosen_status = c1.selectbox("Feature validation", statuses, key=f"{key}_status")
    recs = ["All", *sorted(frame["recommendation"].dropna().unique())]
    chosen_recommendation = c2.selectbox(
        "Safety recommendation", recs, key=f"{key}_rec"
    )
    sort_options = [
        column
        for column in (
            "predicted_jitter_ps",
            "upper_prediction_bound_ps",
            "absolute_error_ps",
            "die_id",
        )
        if column in frame.columns
    ]
    sort_by = c3.selectbox("Sort by", sort_options or list(frame.columns), key=f"{key}_sort")
    filtered = frame
    if chosen_status != "All":
        filtered = filtered.loc[filtered["validation_status"].eq(chosen_status)]
    if chosen_recommendation != "All":
        filtered = filtered.loc[filtered["recommendation"].eq(chosen_recommendation)]
    if sort_by in filtered.columns:
        filtered = filtered.sort_values(sort_by, ascending=False, na_position="last")
    return filtered


try:
    MODEL_ARTIFACT = load_model_artifact()
    MODEL_METRICS, LOCK_METRICS, ORIGINAL_TEST_PREDICTIONS = load_saved_metrics()
    REFERENCE_PARTITIONS, REFERENCE_DIAGNOSTICS = load_reference_partitions()
    REFERENCE_LOCK_ROWS, REFERENCE_LOCK_SUMMARY = load_reference_locktime()
except (FileNotFoundError, OSError, ValueError, KeyError) as error:
    st.error(f"Could not load a required saved PLL-Smart artifact: {error}")
    st.info("Restore the existing model, calibration, and evaluation files before starting the app.")
    st.stop()

REFERENCE_TRAIN = REFERENCE_PARTITIONS["train"]
if list(MODEL_ARTIFACT.get("feature_columns", [])) != list(FEATURE_COLUMNS):
    st.error("Saved model feature contract does not match the application.")
    st.stop()

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">INTELLIGENT PLL TESTING · PREDICTION · ENGINEERING ANALYTICS</div>
      <h1 style="margin:.4rem 0 .35rem 0;">PLL-Smart 2.1</h1>
      <p>Engineering analytics for adaptive PLL testing — with explicit full-test fallback.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="warning-box"><strong>Research prototype — synthetic data only.</strong> '
    "No production qualification or DPPM guarantee. The current illustrative held-out rule "
    "included <strong>52 measured failing dies among 96,121 candidates</strong>; "
    "<strong>keep the full jitter test enabled.</strong></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ◈ PLL-Smart")
    st.caption("Engineering analytics platform · v2.1")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload a jitter-measurement CSV",
        type=["csv"],
        help="Upload is analyzed in memory only. It never replaces the source dataset or retrains the model.",
        key="jitter_csv_upload",
    )
    uploaded_frame: pd.DataFrame | None = None
    upload_error: str | None = None
    uploaded_name = "Original synthetic dataset"
    if uploaded_file is not None:
        uploaded_name = uploaded_file.name
        try:
            uploaded_frame = load_uploaded_csv(uploaded_file.getvalue())
        except ValueError as error:
            upload_error = str(error)
            st.error(upload_error)
    st.caption("Saved model is loaded once per Streamlit process. Uploads do not retrain.")
    st.markdown("---")
    st.caption("Illustrative only: USL 2.2 ps · screen < 2.0 · lock +3 µs / 60 µs cap")

batch_frame: pd.DataFrame | None = None
batch_summary: dict[str, Any] | None = None
batch_comparison: dict[str, Any] | None = None
upload_validation_frame: pd.DataFrame | None = None
upload_validation: dict[str, Any] | None = None
upload_profile: dict[str, Any] | None = None
upload_drift: pd.DataFrame | None = None
upload_drift_summary: dict[str, Any] | None = None
upload_processing_error: str | None = None
if uploaded_frame is not None:
    content_key = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if st.session_state.get("upload_analysis_key") != content_key:
        try:
            with st.spinner("Validating, scoring, and comparing uploaded records…"):
                upload_validation_frame, upload_validation = validate_dataset(
                    uploaded_frame, MODEL_ARTIFACT["feature_ranges"]
                )
                upload_profile = dataset_profile(uploaded_frame)
                batch_frame, batch_summary, batch_comparison = predict_batch(
                    uploaded_frame,
                    MODEL_ARTIFACT["model"],
                    MODEL_ARTIFACT,
                    global_safety_inhibit=SAFETY_INHIBIT,
                )
                upload_drift = compare_feature_distributions(
                    uploaded_frame,
                    REFERENCE_TRAIN,
                    MODEL_ARTIFACT["feature_ranges"],
                )
                upload_drift_summary = summarize_drift(upload_drift)
                batch_history = []
                history_rows = batch_frame.loc[
                    batch_frame["predicted_jitter_ps"].notna()
                ].tail(5000)
                for _, row in history_rows.iterrows():
                    actual_value = (
                        float(row[TARGET_COLUMN])
                        if TARGET_COLUMN in row and pd.notna(row.get(TARGET_COLUMN))
                        else None
                    )
                    batch_history.append(
                        create_history_entry(
                            row,
                            prediction_from_batch_row(row),
                            actual_value,
                            source=f"batch:{uploaded_name}",
                        )
                    )
                st.session_state["prediction_history"] = append_history_entries(
                    st.session_state["prediction_history"], batch_history
                )
                st.session_state["upload_analysis"] = {
                    "batch_frame": batch_frame,
                    "batch_summary": batch_summary,
                    "batch_comparison": batch_comparison,
                    "upload_validation_frame": upload_validation_frame,
                    "upload_validation": upload_validation,
                    "upload_profile": upload_profile,
                    "upload_drift": upload_drift,
                    "upload_drift_summary": upload_drift_summary,
                }
                st.session_state["upload_analysis_key"] = content_key
        except (ValueError, KeyError, TypeError) as error:
            upload_processing_error = f"Uploaded dataset analysis failed: {error}"
            st.error(upload_processing_error)
    if st.session_state.get("upload_analysis_key") == content_key:
        analysis = st.session_state["upload_analysis"]
        batch_frame = analysis["batch_frame"]
        batch_summary = analysis["batch_summary"]
        batch_comparison = analysis["batch_comparison"]
        upload_validation_frame = analysis["upload_validation_frame"]
        upload_validation = analysis["upload_validation"]
        upload_profile = analysis["upload_profile"]
        upload_drift = analysis["upload_drift"]
        upload_drift_summary = analysis["upload_drift_summary"]

tabs = st.tabs(
    [
        "Executive Overview",
        "Dataset Explorer",
        "Batch Prediction",
        "Manual Prediction",
        "Actual vs Predicted",
        "Adaptive Lock-Time",
        "Model Performance",
        "Data Drift Monitoring",
        "Prediction History",
        "Reports & Recommendations",
        "Explainable AI",
        "Engineering What-if",
    ]
)

# 1. Executive Overview
with tabs[0]:
    st.subheader("Executive Overview")
    show_metrics(
        [
            ("Original test records", f"{MODEL_METRICS['test_rows']:,}", "supplied test split"),
            ("Original test MAE", f"{MODEL_METRICS['test_mae_ps']:.4f} ps", None),
            ("Original test R²", f"{MODEL_METRICS['test_r2']:.4f}", None),
            ("Lock-wait reduction", f"{LOCK_METRICS['savings_percent']:.2f}%", "synthetic example"),
        ]
    )
    st.markdown("")
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Platform workflow")
        st.markdown(
            "1. Inspect and validate uploaded PLL measurements.\n"
            "2. Score valid inputs with the saved HistGradientBoosting model.\n"
            "3. Add the calibration-only guard band and route uncertain cases to full test.\n"
            "4. Compare uploaded data with training distributions and inspect warnings.\n"
            "5. Explore adaptive wait time and export session reports."
        )
    with right:
        st.markdown("#### Original model holdout results")
        st.write(
            f"MAE **{MODEL_METRICS['test_mae_ps']:.5f} ps** · "
            f"RMSE **{MODEL_METRICS['test_rmse_ps']:.5f} ps** · "
            f"R² **{MODEL_METRICS['test_r2']:.5f}**"
        )
        st.write(
            f"Calibration guard band: **{MODEL_METRICS['guard_band_ps']:.5f} ps** "
            f"(illustrative alpha {MODEL_METRICS['conformal_alpha_illustrative']:.0%})."
        )
        st.write(
            f"Lock-time comparison: fixed **{LOCK_METRICS['mean_fixed_wait_us']:.2f} µs**, "
            f"adaptive **{LOCK_METRICS['mean_adaptive_wait_us']:.2f} µs**."
        )
    st.info(
        "Measured retrospective limitation: 52 known failing test dies passed the illustrative "
        "skip-candidate rule. All batch/manual recommendations retain full-test fallback; "
        "candidate status is not authorization."
    )
    if uploaded_frame is not None:
        st.success(f"Active upload: {uploaded_name} · {len(uploaded_frame):,} rows")
    elif upload_error is None:
        st.caption("Upload a CSV in the sidebar to explore and score a new dataset.")

# 2. Dataset Explorer
with tabs[1]:
    st.subheader("Dataset Explorer")
    source_choice = st.radio(
        "Dataset to inspect",
        ["Uploaded dataset", "Original training partition"],
        horizontal=True,
        key="explorer_dataset_source",
        disabled=uploaded_frame is None,
    )
    explore_frame = (
        uploaded_frame
        if uploaded_frame is not None and source_choice == "Uploaded dataset"
        else REFERENCE_TRAIN
    )
    if explore_frame is None:
        explore_frame = REFERENCE_TRAIN
    profile = (
        upload_profile
        if uploaded_frame is not None and source_choice == "Uploaded dataset"
        else dataset_profile(explore_frame)
    )
    show_metrics(
        [
            ("Rows", f"{profile['rows']:,}", None),
            ("Columns", f"{profile['columns']:,}", None),
            ("Missing cells", f"{int(explore_frame.isna().sum().sum()):,}", None),
            ("Duplicate rows", f"{int(explore_frame.duplicated(keep=False).sum()):,}", None),
        ]
    )
    details_a, details_b = st.columns(2)
    with details_a:
        with st.expander("Data types and missing values", expanded=True):
            schema = profile["dtypes"].join(profile["missing"], how="outer")
            st.dataframe(schema, width="stretch")
        with st.expander("Statistical summary"):
            st.dataframe(profile["statistics"], width="stretch")
    with details_b:
        with st.expander("Dataset preview", expanded=True):
            st.dataframe(profile["preview"], width="stretch", hide_index=True)
        if uploaded_frame is not None:
            st.markdown("#### Upload validation")
            st.json(
                {
                    "missing_required_columns": upload_validation["missing_columns"],
                    "invalid_feature_rows": upload_validation["invalid_feature_row_count"],
                    "duplicate_rows": upload_validation["duplicate_row_count"],
                    "out_of_training_range_rows": upload_validation[
                        "out_of_training_range_row_count"
                    ],
                    "screen_flagged_rows": upload_validation["screen_flagged_row_count"],
                    "invalid_numeric_values": upload_validation["invalid_type_counts"],
                },
                expanded=False,
            )
    selected_feature = st.selectbox(
        "Feature distribution",
        FEATURE_COLUMNS,
        format_func=lambda value: FEATURE_LABELS[value],
        key="explorer_feature",
    )
    distribution_rows: list[pd.DataFrame] = []
    current_values = pd.to_numeric(explore_frame.get(selected_feature), errors="coerce")
    if current_values is not None:
        distribution_rows.append(
            pd.DataFrame(
                {
                    "Value": current_values,
                    "Dataset": (
                        "Uploaded" if source_choice == "Uploaded dataset" else "Training"
                    ),
                }
            )
        )
    if uploaded_frame is not None and source_choice == "Uploaded dataset":
        training_values = REFERENCE_TRAIN[selected_feature].sample(
            n=min(8000, len(REFERENCE_TRAIN)), random_state=42
        )
        distribution_rows.append(
            pd.DataFrame({"Value": training_values, "Dataset": "Training reference"})
        )
    distribution = pd.concat(distribution_rows, ignore_index=True).dropna()
    if not distribution.empty:
        fig = px.histogram(
            distribution,
            x="Value",
            color="Dataset",
            barmode="overlay",
            histnorm="probability density",
            opacity=0.56,
            title=f"{FEATURE_LABELS[selected_feature]} distribution",
            color_discrete_sequence=[CYAN, BLUE],
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
    st.caption("The uploaded file is parsed in memory; its original bytes are not modified.")

# 3. Batch Prediction
with tabs[2]:
    st.subheader("Batch AI Prediction")
    if uploaded_frame is None:
        st.info("Upload a jitter CSV in the sidebar to run batch predictions.")
    elif upload_processing_error:
        st.error(upload_processing_error)
    elif batch_frame is not None and batch_summary is not None:
            st.warning(
                "Predictions outside training ranges are shown for analysis but cannot pass the "
                "safety rule. The global 52-failure inhibit keeps every test recommendation at "
                "RUN FULL JITTER TEST."
            )
            show_metrics(
                [
                    ("Total records", f"{batch_summary['total_records']:,}", None),
                    ("Successfully predicted", f"{batch_summary['successfully_predicted_records']:,}", None),
                    ("Invalid records", f"{batch_summary['invalid_records']:,}", None),
                    (
                        "Mean / maximum prediction",
                        (
                            f"{batch_summary['mean_predicted_jitter_ps']:.3f} / "
                            f"{batch_summary['maximum_predicted_jitter_ps']:.3f} ps"
                            if batch_summary["mean_predicted_jitter_ps"] is not None
                            else "—"
                        ),
                        None,
                    ),
                ]
            )
            show_metrics(
                [
                    (
                        "Predictions above USL",
                        f"{batch_summary['records_above_illustrative_limit']:,}",
                        "illustrative 2.2 ps",
                    ),
                    (
                        "Upper bounds above USL",
                        f"{batch_summary['upper_bounds_above_illustrative_limit']:,}",
                        "illustrative 2.2 ps",
                    ),
                    (
                        "Illustrative skip candidates",
                        f"{batch_summary['illustrative_candidate_skip_records']:,}",
                        "still routed to full test",
                    ),
                ]
            )
            filtered_batch = filter_prediction_table(batch_frame, "batch")
            display_columns = [
                column
                for column in (
                    "die_id_display",
                    "predicted_jitter_ps",
                    "upper_prediction_bound_ps",
                    "illustrative_usl_ps",
                    "validation_status",
                    "validation_reasons",
                    "recommendation",
                    "recommendation_reason",
                    TARGET_COLUMN,
                )
                if column in filtered_batch.columns
            ]
            st.dataframe(
                filtered_batch[display_columns],
                width="stretch",
                hide_index=True,
                height=420,
            )
            st.download_button(
                "Download scored batch CSV",
                data=safe_csv_download(batch_frame),
                file_name="pll_smart_batch_predictions.csv",
                mime="text/csv",
                key="download_batch_predictions",
            )
            if batch_comparison is not None:
                st.caption(
                    "Measured actual jitter was detected; comparison results are shown in "
                    "the Actual vs Predicted section."
                )

# 4. Manual Prediction
with tabs[3]:
    st.subheader("Manual Single-Die Prediction")
    st.caption(
        "All seven required feature values must be valid; the screen value is a normalized "
        "illustrative measurement. Out-of-range inputs are always full-test fallback."
    )
    medians = REFERENCE_TRAIN.loc[:, FEATURE_COLUMNS].median()
    with st.form("manual_prediction_form"):
        entered: dict[str, float] = {}
        feature_rows = [st.columns(2), st.columns(2), st.columns(2), st.columns(1)]
        for index, feature in enumerate(FEATURE_COLUMNS):
            col = feature_rows[index // 2][index % 2]
            lower, upper = MODEL_ARTIFACT["feature_ranges"][feature]
            entered[feature] = col.number_input(
                FEATURE_LABELS[feature],
                value=float(medians[feature]),
                help=f"Training range: {lower:.6g} to {upper:.6g}. Outside this range routes to full test.",
                format="%.6f",
                key=f"manual_{feature}",
            )
        actual_text = st.text_input(
            "Optional measured RMS jitter (ps)",
            value="",
            help="Enter an actual test result to calculate the signed error (prediction minus actual).",
        )
        submitted = st.form_submit_button("Predict RMS jitter")

    if submitted:
        decision = decide_jitter_test(
            entered,
            MODEL_ARTIFACT["model"],
            MODEL_ARTIFACT,
            global_safety_inhibit=SAFETY_INHIBIT,
        )
        actual_value: float | None = None
        actual_error: str | None = None
        if actual_text.strip():
            try:
                actual_value = float(actual_text)
                if not np.isfinite(actual_value):
                    raise ValueError("Actual jitter must be finite.")
            except ValueError as error:
                actual_error = f"Measured RMS jitter was not accepted: {error}"
        st.session_state["manual_last"] = {
            "features": entered,
            "decision": decision,
            "actual": actual_value,
            "actual_error": actual_error,
        }
        st.session_state["prediction_history"] = append_history_entry(
            st.session_state["prediction_history"],
            create_history_entry(entered, decision, actual_value),
        )

    if "manual_last" in st.session_state:
        last = st.session_state["manual_last"]
        decision = last["decision"]
        st.error(f"**{decision['recommendation']}** — {decision['reason']}")
        if last["actual_error"]:
            st.warning(last["actual_error"])
        show_metrics(
            [
                ("Predicted RMS jitter", format_metric(decision["predicted_jitter_ps"], " ps"), None),
                ("Calibration guard band", format_metric(decision["guard_band_ps"], " ps"), None),
                ("Upper prediction bound", format_metric(decision["upper_bound_ps"], " ps"), None),
                ("Illustrative USL", format_metric(decision["usl_ps"], " ps"), "not approved"),
            ]
        )
        st.write(f"Input validation: **{decision['validation_status']}**")
        if decision["out_of_training_range_features"]:
            st.warning(
                "Out of training range: "
                + ", ".join(decision["out_of_training_range_features"])
            )
        if last["actual"] is not None and decision["predicted_jitter_ps"] is not None:
            signed_error = decision["predicted_jitter_ps"] - last["actual"]
            absolute_error = abs(signed_error)
            percentage_error = (
                signed_error / abs(last["actual"]) * 100
                if abs(last["actual"]) > np.finfo(float).eps
                else None
            )
            show_metrics(
                [
                    ("Signed error (predicted − actual)", format_metric(signed_error, " ps"), None),
                    ("Absolute error", format_metric(absolute_error, " ps"), None),
                    ("Percentage error", format_metric(percentage_error, "%"), None),
                    (
                        "Direction",
                        "Overprediction" if signed_error > 0 else "Underprediction" if signed_error < 0 else "Exact",
                        None,
                    ),
                ]
            )

# 5. Actual vs Predicted
with tabs[4]:
    st.subheader("Actual vs Predicted Analysis")
    if uploaded_frame is None:
        st.info("Upload a dataset containing `rms_jitter_ps` to compare actual and predicted values.")
    else:
        if "split" in uploaded_frame.columns:
            split_values = set(
                uploaded_frame["split"].dropna().astype(str).str.lower()
            )
            if {"train", "calibration", "test"}.issubset(split_values):
                st.warning(
                    "This upload includes the original train, calibration, and test labels. "
                    "The displayed aggregate upload metrics include all valid labeled rows "
                    "and are not an independent holdout estimate. Original held-out metrics "
                    "remain separate."
                )
        if batch_frame is None or TARGET_COLUMN not in batch_frame.columns:
            st.info(
                "Uploaded data has no `rms_jitter_ps` column; "
                "no comparison metrics are calculated."
            )
        elif batch_comparison is None:
            st.info("No valid predictions are available for comparison.")
        elif not batch_comparison["sufficient"]:
            st.warning(batch_comparison["reason"])
        else:
            comparison = batch_comparison
            show_metrics(
                [
                    ("Actual/predicted pairs", f"{comparison['count']:,}", "new upload only"),
                    ("MAE", f"{comparison['mae_ps']:.5f} ps", None),
                    ("RMSE", f"{comparison['rmse_ps']:.5f} ps", None),
                    ("R²", f"{comparison['r2']:.5f}", None),
                ]
            )
            details_left, details_right = st.columns(2)
            with details_left:
                fig = px.scatter(
                    comparison["table"],
                    x=TARGET_COLUMN,
                    y="predicted_jitter_ps",
                    color="validation_status",
                    opacity=0.65,
                    title="Uploaded actual vs predicted",
                    labels={
                        TARGET_COLUMN: "Measured RMS jitter (ps)",
                        "predicted_jitter_ps": "Predicted RMS jitter (ps)",
                    },
                    color_discrete_sequence=[CYAN, "#f7a65a", BLUE],
                )
                values = comparison["table"][TARGET_COLUMN].to_numpy(dtype=float)
                low, high = float(np.min(values)), float(np.max(values))
                fig.add_shape(
                    type="line", x0=low, y0=low, x1=high, y1=high,
                    line={"color": "white", "dash": "dash"},
                )
                st.plotly_chart(chart_layout(fig), width="stretch")
            with details_right:
                residuals = comparison["table"].assign(
                    residual_ps=lambda df: df["predicted_jitter_ps"] - df[TARGET_COLUMN]
                )
                fig = px.histogram(
                    residuals,
                    x="residual_ps",
                    nbins=50,
                    title="New-upload residual distribution",
                    labels={"residual_ps": "Predicted − actual (ps)"},
                    color_discrete_sequence=[BLUE],
                )
                fig.add_vline(x=0, line_dash="dash", line_color=CYAN)
                st.plotly_chart(chart_layout(fig), width="stretch")
            st.markdown("#### Largest prediction errors")
            st.dataframe(
                comparison["largest_errors"],
                width="stretch",
                hide_index=True,
            )
            st.caption("These metrics are separate from the original held-out test results.")

# 6. Adaptive Lock-Time
with tabs[5]:
    st.subheader("Adaptive Lock-Time Optimization")
    st.caption(
        "Illustrative parameters only: 3.0 µs verification margin and 60.0 µs fixed-wait cap."
    )
    fixed_col, margin_col = st.columns(2)
    fixed_value = fixed_col.number_input(
        "Illustrative fixed-wait cap (µs)", value=60.0, min_value=0.001, step=1.0
    )
    margin_value = margin_col.number_input(
        "Illustrative verification margin (µs)", value=3.0, min_value=0.0, step=0.5
    )
    calc_col, batch_col = st.columns(2)
    with calc_col:
        lock_value = st.number_input(
            "Manual measured lock time (µs)", value=12.0, step=0.5, key="lock_manual"
        )
        if lock_value < 0:
            st.error("Negative lock time is invalid.")
        else:
            one_row, _ = analyze_locktime_values(
                [lock_value], fixed_wait_us=fixed_value, margin_us=margin_value
            )
            adaptive_value = float(one_row.iloc[0]["adaptive_wait_us"])
            saved_value = float(one_row.iloc[0]["time_saved_us"])
            st.metric("Adaptive wait", f"{adaptive_value:.3f} µs")
            st.metric("Time saved vs fixed", f"{saved_value:.3f} µs")
    with batch_col:
        lock_upload = st.file_uploader(
            "Optional lock-time CSV (`lock_time_us` column)",
            type=["csv"],
            key="locktime_csv_upload",
        )
        if lock_upload is not None:
            try:
                lock_input = load_uploaded_csv(lock_upload.getvalue())
                if "lock_time_us" not in lock_input:
                    st.error("Lock-time upload must include `lock_time_us`.")
                else:
                    batch_lock, batch_lock_summary = analyze_locktime_values(
                        lock_input["lock_time_us"],
                        fixed_wait_us=fixed_value,
                        margin_us=margin_value,
                    )
                    show_metrics(
                        [
                            ("Valid lock records", f"{batch_lock_summary['valid_records']:,}", None),
                            ("Invalid records", f"{batch_lock_summary['invalid_records']:,}", None),
                            ("Mean adaptive wait", format_metric(batch_lock_summary["mean_adaptive_wait_us"], " µs"), None),
                            ("Mean reduction", format_metric(batch_lock_summary["savings_percent"], "%"), "illustrative"),
                        ]
                    )
                    show_metrics(
                        [
                            ("Adaptive min / max", f"{batch_lock_summary['minimum_adaptive_wait_us']:.2f} / {batch_lock_summary['maximum_adaptive_wait_us']:.2f} µs", None),
                            ("Fixed min / max", f"{batch_lock_summary['minimum_fixed_wait_us']:.2f} / {batch_lock_summary['maximum_fixed_wait_us']:.2f} µs", None),
                            ("Time saved min / max", f"{batch_lock_summary['minimum_time_saved_us']:.2f} / {batch_lock_summary['maximum_time_saved_us']:.2f} µs", None),
                            ("Mean time saved", format_metric(batch_lock_summary["mean_time_saved_us"], " µs"), None),
                        ]
                    )
                    fig = go.Figure()
                    fig.add_trace(
                        go.Histogram(
                            x=batch_lock.loc[batch_lock["valid"], "fixed_wait_us"],
                            name="Fixed",
                            opacity=0.65,
                            marker_color="#f7a65a",
                        )
                    )
                    fig.add_trace(
                        go.Histogram(
                            x=batch_lock.loc[batch_lock["valid"], "adaptive_wait_us"],
                            name="Adaptive",
                            opacity=0.65,
                            marker_color=CYAN,
                        )
                    )
                    fig.update_layout(
                        barmode="overlay",
                        title="Uploaded lock-time waiting distributions",
                        xaxis_title="Wait (µs)",
                        yaxis_title="Record count",
                    )
                    st.plotly_chart(chart_layout(fig), width="stretch")
                    st.download_button(
                        "Download lock-time calculations",
                        safe_csv_download(batch_lock),
                        "pll_smart_locktime_analysis.csv",
                        "text/csv",
                        key="download_locktime",
                    )
            except (ValueError, KeyError, TypeError) as error:
                st.error(f"Lock-time analysis failed: {error}")

    st.markdown("#### Original supplied lock-time dataset")
    show_metrics(
        [
            ("Mean fixed", f"{LOCK_METRICS['mean_fixed_wait_us']:.3f} µs", None),
            ("Mean adaptive", f"{LOCK_METRICS['mean_adaptive_wait_us']:.3f} µs", None),
            ("Mean time saved", f"{LOCK_METRICS['mean_savings_us']:.3f} µs", None),
            ("Reduction", f"{LOCK_METRICS['savings_percent']:.2f}%", "synthetic dataset"),
        ]
    )
    original_lock_plot = px.histogram(
        REFERENCE_LOCK_ROWS,
        x=["fixed_wait_us", "calculated_adaptive_wait_us"],
        barmode="overlay",
        opacity=0.65,
        title="Original fixed versus adaptive wait",
        labels={"value": "Wait (µs)", "variable": "Strategy"},
        color_discrete_sequence=["#f7a65a", CYAN],
    )
    st.plotly_chart(chart_layout(original_lock_plot), width="stretch")
    st.caption("Calculated as min(lock_time_us + margin, fixed-wait cap); not production-qualified.")

# 7. Model Performance
with tabs[6]:
    st.subheader("Model Performance & Comparison")
    st.caption("Saved model/results are read-only here. No upload triggers model fitting.")
    show_metrics(
        [
            ("Held-out MAE", f"{MODEL_METRICS['test_mae_ps']:.5f} ps", "original test partition"),
            ("Held-out RMSE", f"{MODEL_METRICS['test_rmse_ps']:.5f} ps", None),
            ("Held-out R²", f"{MODEL_METRICS['test_r2']:.5f}", None),
            ("Calibration guard band", f"{MODEL_METRICS['guard_band_ps']:.5f} ps", "calibration partition"),
        ]
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model": "Train-mean baseline",
                    "MAE (ps)": MODEL_METRICS["baseline_test_mae_ps"],
                    "RMSE (ps)": MODEL_METRICS["baseline_test_rmse_ps"],
                    "R²": MODEL_METRICS["baseline_test_r2"],
                },
                {
                    "Model": MODEL_METRICS["model"],
                    "MAE (ps)": MODEL_METRICS["test_mae_ps"],
                    "RMSE (ps)": MODEL_METRICS["test_rmse_ps"],
                    "R²": MODEL_METRICS["test_r2"],
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    visual_left, visual_right = st.columns(2)
    with visual_left:
        sample_predictions = ORIGINAL_TEST_PREDICTIONS.sample(
            n=min(7000, len(ORIGINAL_TEST_PREDICTIONS)), random_state=42
        )
        fig = px.scatter(
            sample_predictions,
            x="actual_jitter_ps",
            y="predicted_jitter_ps",
            opacity=0.48,
            title="Original held-out actual vs predicted",
            labels={
                "actual_jitter_ps": "Measured RMS jitter (ps)",
                "predicted_jitter_ps": "Predicted RMS jitter (ps)",
            },
            color_discrete_sequence=[CYAN],
        )
        fig.add_hline(
            y=ILLUSTRATIVE_JITTER_USL_PS,
            line_dash="dot",
            line_color="#ff8b76",
            annotation_text="Illustrative USL",
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
    with visual_right:
        fig = px.histogram(
            ORIGINAL_TEST_PREDICTIONS,
            x="residual_ps",
            nbins=70,
            title="Original test residuals (actual − predicted)",
            color_discrete_sequence=[BLUE],
        )
        fig.add_vline(x=0, line_dash="dash", line_color=CYAN)
        st.plotly_chart(chart_layout(fig), width="stretch")

    st.markdown("#### Training feature correlation")
    correlation = REFERENCE_TRAIN.loc[:, FEATURE_COLUMNS].corr()
    heatmap = go.Figure(
        data=go.Heatmap(
            z=correlation.to_numpy(),
            x=[FEATURE_LABELS[name] for name in FEATURE_COLUMNS],
            y=[FEATURE_LABELS[name] for name in FEATURE_COLUMNS],
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            colorbar={"title": "Pearson r"},
        )
    )
    st.plotly_chart(chart_layout(heatmap, 440), width="stretch")

    st.markdown("#### Model-dependent feature importance")
    if "permutation_importance_v21" not in st.session_state:
        st.session_state["permutation_importance_v21"] = compute_global_feature_importance(
            MODEL_ARTIFACT["model"],
            REFERENCE_TRAIN,
            REFERENCE_TRAIN[TARGET_COLUMN],
            sample_size=1200,
            n_repeats=3,
            random_state=42,
        )
    importance_frame = st.session_state["permutation_importance_v21"]
    importance_plot = importance_frame.assign(
        feature_label=importance_frame["feature"].map(FEATURE_LABELS)
    )
    fig = px.bar(
        importance_plot,
        x="importance_mean",
        y="feature_label",
        error_x="importance_std",
        orientation="h",
        title="Permutation importance · reproducible training sample",
        color_discrete_sequence=[CYAN],
    )
    st.plotly_chart(chart_layout(fig), width="stretch")
    range_frame = pd.DataFrame(
        [
            {
                "Feature": FEATURE_LABELS[name],
                "Training minimum": MODEL_ARTIFACT["feature_ranges"][name][0],
                "Training maximum": MODEL_ARTIFACT["feature_ranges"][name][1],
            }
            for name in FEATURE_COLUMNS
        ]
    )
    st.markdown("#### Training feature ranges")
    st.dataframe(range_frame, width="stretch", hide_index=True)
    st.caption(
        "Permutation importance is model- and sample-dependent, not causal. "
        "No optional model was fit and the saved model was not replaced."
    )
    st.warning(
        "52 measured failures were among 96,121 original illustrative skip candidates. "
        "The model comparison must not be interpreted as authorization to reduce tests."
    )

# 8. Data Drift Monitoring
with tabs[7]:
    st.subheader("Data Drift Monitoring")
    if uploaded_frame is None:
        st.info("Upload a CSV with candidate feature columns to compare against training data.")
    else:
        drift_comparison = upload_drift
        drift_summary = upload_drift_summary
        drift_validation = upload_validation
        if drift_summary["warning_features"]:
            st.warning(
                f"Distribution shift warning on {drift_summary['warning_features']} of "
                f"{drift_summary['compared_features']} comparable features."
            )
        else:
            st.success(drift_summary["status"])
        show_metrics(
            [
                ("Compared features", str(drift_summary["compared_features"]), None),
                ("Features with shift warnings", str(drift_summary["warning_features"]), None),
                (
                    "Out-of-range rows",
                    f"{drift_validation['out_of_training_range_row_count']:,}",
                    "feature range validation",
                ),
            ]
        )
        st.dataframe(drift_comparison, width="stretch", hide_index=True)
        available = [feature for feature in FEATURE_COLUMNS if feature in uploaded_frame]
        if available:
            choice = st.selectbox(
                "Distribution comparison",
                available,
                format_func=lambda value: FEATURE_LABELS[value],
                key="drift_feature",
            )
            values = pd.to_numeric(uploaded_frame[choice], errors="coerce").dropna()
            training_sample = REFERENCE_TRAIN[choice].sample(
                n=min(10000, len(REFERENCE_TRAIN)), random_state=42
            )
            compare_plot = pd.concat(
                [
                    pd.DataFrame({"value": training_sample, "dataset": "Training"}),
                    pd.DataFrame({"value": values, "dataset": "Uploaded"}),
                ],
                ignore_index=True,
            )
            fig = px.histogram(
                compare_plot,
                x="value",
                color="dataset",
                barmode="overlay",
                histnorm="probability density",
                opacity=0.58,
                title=f"Training vs uploaded · {FEATURE_LABELS[choice]}",
                color_discrete_sequence=[BLUE, CYAN],
            )
            st.plotly_chart(chart_layout(fig), width="stretch")
        st.caption(
            "KS p-values and range-exceedance rates are monitoring signals only. "
            "A drift warning does not prove model failure; no warning does not establish safety."
        )

# 9. Prediction History
with tabs[8]:
    st.subheader("Prediction History")
    st.caption("History is stored in this Streamlit browser session only; it is not database-backed.")
    history = pd.DataFrame(st.session_state["prediction_history"])
    if history.empty:
        st.info("Manual and batch predictions in this session will appear here.")
    else:
        filter_options = ["All", *sorted(history["source"].dropna().unique())]
        source_filter = st.selectbox("Prediction source", filter_options, key="history_source")
        view_history = history
        if source_filter != "All":
            view_history = view_history.loc[view_history["source"].eq(source_filter)]
        validation_options = ["All", *sorted(view_history["validation_status"].dropna().unique())]
        validation_filter = st.selectbox(
            "Validation status", validation_options, key="history_validation"
        )
        if validation_filter != "All":
            view_history = view_history.loc[
                view_history["validation_status"].eq(validation_filter)
            ]
        st.dataframe(view_history, width="stretch", hide_index=True, height=440)
        st.download_button(
            "Export visible history CSV",
            safe_csv_download(view_history),
            "pll_smart_prediction_history.csv",
            "text/csv",
            key="download_history",
        )
        if st.button("Clear session history", key="clear_history"):
            st.session_state["prediction_history"] = []
            st.rerun()

# 10. Reports & Engineering Recommendations
with tabs[9]:
    st.subheader("Reports & Engineering Recommendations")
    st.error(
        "Safety inhibit remains active: 52 measured failing dies were among 96,121 "
        "original skip candidates. Keep the actual full jitter test enabled."
    )
    st.markdown(
        """
        **Engineering actions**
        1. Validate uploaded schema, types, missing values, duplicates, and training-range coverage.
        2. Review out-of-range and distribution-shift warnings; these are triage signals, not proof.
        3. Treat the 2.2 ps USL, leakage threshold of 2.0, 3 µs margin, and 60 µs cap as illustrative.
        4. Validate on representative held-out lots, wafers, testers, product corners, and defect populations.
        5. Run shadow mode, measure real test timing, and obtain quality/customer change approval.
        6. Preserve baseline full testing and define pooled audit, drift monitoring, and rollback procedures.
        """
    )
    st.info(
        "This app is not production-qualified. The synthetic held-out result is not a DPPM guarantee."
    )

    manual = st.session_state.get("manual_last")
    current_prediction = (
        manual["decision"] if manual else None
    )
    current_inputs = (
        manual["features"]
        if manual
        else (
            {
                "batch_upload": uploaded_name,
                "model_features": list(FEATURE_COLUMNS),
            }
            if uploaded_frame is not None
            else None
        )
    )
    comparison = batch_comparison
    lock_report = LOCK_METRICS
    report_dataset = {
        "name": uploaded_name,
        "rows": len(uploaded_frame) if uploaded_frame is not None else len(REFERENCE_TRAIN),
        "columns": uploaded_frame.shape[1] if uploaded_frame is not None else len(REFERENCE_TRAIN.columns),
        "invalid_rows": (
            batch_summary["invalid_records"]
            if uploaded_frame is not None and batch_summary is not None
            else REFERENCE_DIAGNOSTICS["invalid_row_count"]
            if uploaded_frame is None
            else "analysis unavailable"
        ),
        "source": "Uploaded in-memory CSV" if uploaded_frame is not None else "Original training CSV",
    }
    prediction_report = (
        {
            "predicted_jitter_ps": current_prediction.get("predicted_jitter_ps"),
            "guard_band_ps": current_prediction.get("guard_band_ps"),
            "upper_prediction_bound_ps": current_prediction.get("upper_bound_ps"),
            "illustrative_usl_ps": current_prediction.get("usl_ps"),
            "validation_status": current_prediction.get("validation_status"),
            "recommendation": current_prediction.get("recommendation"),
            "reason": current_prediction.get("reason"),
        }
        if current_prediction
        else (
            {
                "batch_summary": batch_summary,
                "recommendation": "RUN FULL JITTER TEST",
                "safety_inhibit": SAFETY_INHIBIT,
            }
            if uploaded_frame is not None
            else None
        )
    )
    comparison_report = (
        {
            "count": comparison.get("count"),
            "mae_ps": comparison.get("mae_ps"),
            "rmse_ps": comparison.get("rmse_ps"),
            "r2": comparison.get("r2"),
            "sufficient": comparison.get("sufficient"),
            "reason": comparison.get("reason"),
        }
        if comparison
        else None
    )
    report_warnings = [
        SAFETY_INHIBIT,
        "All limits and screening parameters are illustrative, not approved production limits.",
        "Synthetic-data evaluation is not production qualification or a DPPM guarantee.",
    ]
    report_recommendations = [
        "Keep full jitter testing enabled.",
        "Validate on representative lot-held-out production data before considering test changes.",
        "Review invalid inputs, out-of-range features, and drift warnings before interpreting predictions.",
    ]
    if uploaded_frame is not None:
        uploaded_validation = upload_validation
        if "split" in uploaded_frame.columns:
            split_values = set(
                uploaded_frame["split"].dropna().astype(str).str.lower()
            )
            if {"train", "calibration", "test"}.issubset(split_values):
                report_warnings.append(
                    "Uploaded aggregate comparisons include rows labelled train, "
                    "calibration, and test; they are not an independent holdout estimate."
                )
        if uploaded_validation["missing_columns"]:
            report_warnings.append(
                "Uploaded dataset is missing required features: "
                + ", ".join(uploaded_validation["missing_columns"])
                + "."
            )
        if uploaded_validation["invalid_feature_row_count"]:
            report_warnings.append(
                f"Uploaded dataset contains {uploaded_validation['invalid_feature_row_count']} "
                "invalid feature rows."
            )
            report_recommendations.append(
                "Resolve missing, invalid, duplicate, or negative lock-time records before analysis."
            )
        if uploaded_validation["out_of_training_range_row_count"]:
            report_warnings.append(
                f"{uploaded_validation['out_of_training_range_row_count']} uploaded rows "
                "contain features outside saved training ranges."
            )
            report_recommendations.append(
                "Investigate out-of-training-range features; route those records to full testing."
            )
        if uploaded_validation["screen_flagged_row_count"]:
            report_warnings.append(
                f"{uploaded_validation['screen_flagged_row_count']} records failed the "
                "illustrative leakage-screen condition."
            )
        drift_status = upload_drift_summary
        if drift_status["warning_features"]:
            report_warnings.append(
                f"Distribution shift warning on {drift_status['warning_features']} "
                "features; investigate before interpreting predictions."
            )
            report_recommendations.append(
                "Review feature-distribution differences and validate model behavior on representative data."
            )
    if comparison and comparison.get("sufficient"):
        report_recommendations.append(
            "Review the new-upload actual-versus-predicted error metrics and largest residuals; "
            "no production alert threshold is configured."
        )
    if current_prediction:
        report_recommendations.insert(0, current_prediction["reason"])

    report_payload = {
        "dataset_info": report_dataset,
        "input_measurements": current_inputs,
        "prediction_results": prediction_report,
        "actual_comparison": comparison_report,
        "model_performance": {
            "model": MODEL_METRICS["model"],
            "test_rows": MODEL_METRICS["test_rows"],
            "test_mae_ps": MODEL_METRICS["test_mae_ps"],
            "test_rmse_ps": MODEL_METRICS["test_rmse_ps"],
            "test_r2": MODEL_METRICS["test_r2"],
            "calibration_guard_band_ps": MODEL_METRICS["guard_band_ps"],
            "known_failures_among_candidates": MODEL_METRICS[
                "test_failures_among_skip_candidates"
            ],
        },
        "locktime_analysis": {
            "mean_fixed_wait_us": lock_report["mean_fixed_wait_us"],
            "mean_adaptive_wait_us": lock_report["mean_adaptive_wait_us"],
            "mean_time_saved_us": lock_report["mean_savings_us"],
            "savings_percent": lock_report["savings_percent"],
            "formula_mismatches": lock_report["formula_mismatches"],
        },
        "warnings": report_warnings,
        "recommendations": report_recommendations,
    }
    report_exports = build_engineering_report_exports(report_payload)
    st.download_button(
        "Download engineering report data (JSON)",
        report_exports["json"],
        "pll_smart_engineering_report.json",
        "application/json",
        key="download_report_json",
    )
    st.download_button(
        "Download engineering report (PDF)",
        report_exports["pdf"],
        "pll_smart_engineering_report.pdf",
        "application/pdf",
        key="download_report_pdf",
    )
    st.download_button(
        "Download engineering report (CSV)",
        report_exports["csv"],
        "pll_smart_engineering_report.csv",
        "text/csv",
        key="download_report_csv",
    )

# 11. Explainable AI
with tabs[10]:
    st.subheader("Explainable AI")
    st.markdown("#### Global model-dependent feature importance")
    if "permutation_importance_v21" not in st.session_state:
        st.session_state["permutation_importance_v21"] = compute_global_feature_importance(
            MODEL_ARTIFACT["model"],
            REFERENCE_TRAIN,
            REFERENCE_TRAIN[TARGET_COLUMN],
            sample_size=1200,
            n_repeats=3,
            random_state=42,
        )
    global_importance = st.session_state["permutation_importance_v21"].assign(
        feature_label=lambda frame: frame["feature"].map(FEATURE_LABELS)
    )
    fig = px.bar(
        global_importance,
        x="importance_mean",
        y="feature_label",
        error_x="importance_std",
        orientation="h",
        title="Permutation importance · training-data sample",
        labels={
            "importance_mean": "Increase in MAE when permuted (ps)",
            "feature_label": "Model feature",
            "importance_std": "Repeat standard deviation",
        },
        color_discrete_sequence=[CYAN],
    )
    st.plotly_chart(chart_layout(fig), width="stretch")
    st.caption(
        "Computed with a reproducible 1,200-row sample from the training partition. "
        "Importance measures model reliance under this method; it is not causal and "
        "does not describe an individual die."
    )
    st.markdown("#### Per-die explanation for the latest manual prediction")
    manual = st.session_state.get("manual_last")
    if manual is None:
        st.info(
            "Make a prediction in the Manual Prediction section to see a per-die explanation."
        )
    else:
        explanation = manual.get("explanation")
        if explanation is None:
            try:
                explanation = explain_prediction(
                    manual["features"],
                    MODEL_ARTIFACT["model"],
                    MODEL_ARTIFACT,
                    REFERENCE_TRAIN.loc[:, FEATURE_COLUMNS].median().to_dict(),
                )
                manual["explanation"] = explanation
            except (TypeError, ValueError) as error:
                st.error(f"Could not calculate the model-based explanation: {error}")
                explanation = None
        if explanation is not None and not explanation["valid"]:
            st.warning(
                f"No prediction explanation is available for this input: "
                f"{explanation['reason']} Full jitter test remains enabled."
            )
        elif explanation is not None:
            show_metrics(
                [
                    ("Observed-input prediction", format_metric(explanation["prediction_ps"], " ps"), None),
                    ("Training-median reference prediction", format_metric(explanation["reference_prediction_ps"], " ps"), None),
                    ("Input validation", explanation["validation_status"], None),
                ]
            )
            effects = explanation["effects"].assign(
                feature_label=lambda frame: frame["feature"].map(FEATURE_LABELS)
            )
            fig = px.bar(
                effects,
                x="effect_ps",
                y="feature_label",
                color="direction",
                orientation="h",
                title="Single-feature replacement impact",
                labels={
                    "effect_ps": "Observed prediction − feature-replaced prediction (ps)",
                    "feature_label": "Model feature",
                    "direction": "Direction relative to reference",
                },
                color_discrete_map={
                    "Raises prediction": "#f7a65a",
                    "Lowers prediction": CYAN,
                    "No measured change": MUTED,
                },
            )
            fig.add_vline(x=0, line_dash="dash", line_color="white")
            st.plotly_chart(chart_layout(fig), width="stretch")
            st.dataframe(
                effects[
                    ["feature_label", "effect_ps", "direction"]
                ].rename(
                    columns={
                        "feature_label": "Feature",
                        "effect_ps": "Replacement impact (ps)",
                        "direction": "Model response",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(explanation["method"])
            if explanation["out_of_training_range_features"]:
                st.warning(
                    "One or more inputs are outside saved training ranges: "
                    + ", ".join(explanation["out_of_training_range_features"])
                    + ". Treat this explanation as out-of-domain; use the full test."
                )
            st.info(
                "Only the seven approved model features are evaluated. Actual jitter, "
                "identifiers, split labels, and synthetic ground-truth annotations "
                "are not used as inference inputs."
            )

# 12. Engineering What-if Simulator
with tabs[11]:
    st.subheader("Engineering What-if Simulator")
    manual = st.session_state.get("manual_last")
    if manual is not None and all(
        name in manual["features"] for name in FEATURE_COLUMNS
    ):
        baseline_features = {
            name: manual["features"][name] for name in FEATURE_COLUMNS
        }
        baseline_source = "latest manual prediction"
    else:
        baseline_features = (
            REFERENCE_TRAIN.loc[:, FEATURE_COLUMNS].median().to_dict()
        )
        baseline_source = "training-partition feature medians"
    st.caption(
        f"Baseline uses the {baseline_source}. Modify one or more measurements, then "
        "submit to compare both scenarios using the saved model."
    )
    st.info(
        "This simulator does not modify the trained model, source datasets, calibration, "
        "or illustrative limits. Both scenarios retain the full-test safety inhibit."
    )
    with st.form("what_if_form"):
        what_if_columns = st.columns(2)
        modified_features: dict[str, float] = {}
        for index, feature in enumerate(FEATURE_COLUMNS):
            lower, upper = MODEL_ARTIFACT["feature_ranges"][feature]
            modified_features[feature] = what_if_columns[index % 2].number_input(
                f"Modified · {FEATURE_LABELS[feature]}",
                value=float(baseline_features[feature]),
                step=0.01,
                format="%.6f",
                help=(
                    f"Saved training range: {lower:.6g} to {upper:.6g}. "
                    "Values outside the range are scored for analysis when valid, "
                    "but force full-test fallback."
                ),
                key=f"what_if_{feature}",
            )
        run_what_if = st.form_submit_button("Compare baseline and modified scenario")

    if run_what_if:
        try:
            comparison = compare_what_if_predictions(
                baseline_features,
                modified_features,
                MODEL_ARTIFACT["model"],
                MODEL_ARTIFACT,
            )
            st.session_state["what_if_last"] = {
                "baseline_features": baseline_features,
                "modified_features": modified_features,
                "baseline_source": baseline_source,
                "comparison": comparison,
            }
        except (TypeError, ValueError) as error:
            st.error(
                f"What-if comparison failed; keep the full jitter test enabled: {error}"
            )

    what_if_last = st.session_state.get("what_if_last")
    if what_if_last is not None:
        comparison = what_if_last["comparison"]
        baseline_decision = comparison["baseline"]
        modified_decision = comparison["modified"]
        left, right = st.columns(2)
        with left:
            st.markdown("#### Baseline scenario")
            st.caption(what_if_last["baseline_source"])
            st.metric(
                "Predicted RMS jitter",
                format_metric(baseline_decision["predicted_jitter_ps"], " ps"),
            )
            st.metric(
                "Upper prediction bound",
                format_metric(baseline_decision["upper_bound_ps"], " ps"),
            )
            st.write(
                f"Validation: **{baseline_decision['validation_status']}**"
            )
            st.caption(baseline_decision["reason"])
        with right:
            st.markdown("#### Modified scenario")
            st.caption("Interactive feature values submitted above")
            st.metric(
                "Predicted RMS jitter",
                format_metric(modified_decision["predicted_jitter_ps"], " ps"),
            )
            st.metric(
                "Upper prediction bound",
                format_metric(modified_decision["upper_bound_ps"], " ps"),
            )
            st.write(
                f"Validation: **{modified_decision['validation_status']}**"
            )
            st.caption(modified_decision["reason"])
        show_metrics(
            [
                (
                    "Prediction change (modified − baseline)",
                    format_metric(comparison["prediction_delta_ps"], " ps"),
                    None,
                ),
                (
                    "Upper-bound change",
                    format_metric(comparison["upper_bound_delta_ps"], " ps"),
                    None,
                ),
                (
                    "Illustrative eligibility",
                    "Changed"
                    if comparison["illustrative_candidate_changed"]
                    else "Unchanged",
                    "analysis only",
                ),
            ]
        )
        st.warning(
            comparison["change_summary"]
            + " Final recommendation in both scenarios: RUN FULL JITTER TEST."
        )
        for scenario_name, decision in (
            ("Baseline", baseline_decision),
            ("Modified", modified_decision),
        ):
            if decision["out_of_training_range_features"]:
                st.warning(
                    f"{scenario_name} scenario is outside training ranges for: "
                    + ", ".join(decision["out_of_training_range_features"])
                    + "."
                )
