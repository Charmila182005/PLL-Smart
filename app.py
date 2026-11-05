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
    compare_regression_models,
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
NAVY = "#06151B"
PANEL = "#092832"
CYAN = "#19D5E8"
BLUE = "#00AFA3"
MUTED = "#91B2BC"
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
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background: #06151B;
        color: #EAF7FA;
    }
    html, body, [class*="css"] { font-family: Inter, "Segoe UI", sans-serif; }
    h1, h2, h3, h4 { color: #EAF7FA; letter-spacing: -.025em; }
    h1 { font-weight: 760; }
    h2, h3 { font-weight: 680; }
    p, li { color: #B7D0D6; }
    [data-testid="stHeader"] { background: rgba(6, 21, 27, .94); }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stMain"] > div { background: #06151B; }
    .block-container {
        width: 100%;
        max-width: none;
        padding: .85rem 1.5rem 1.1rem;
    }
    [data-testid="stVerticalBlock"] {
        gap: .68rem;
    }
    [data-testid="stSidebar"] {
        background: #071D25;
        border-right: 1px solid #17414B;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        width: 264px !important;
        min-width: 264px !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding: .7rem .65rem .85rem; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .38rem; }
    [data-testid="stMetric"] {
        background: #092832;
        border: 1px solid #174650;
        border-radius: 12px;
        padding: 11px 13px;
        min-height: 84px;
        box-shadow: 0 5px 14px rgba(0, 0, 0, .12);
    }
    [data-testid="stMetricLabel"] { color: #C5D5DF; font-size: .8rem; }
    [data-testid="stMetricValue"] { color: #EAF7FA; font-weight: 720; }
    [data-testid="stMetricDelta"] { font-size: .78rem; }
    .hero {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        overflow: hidden;
        min-height: 102px;
        background: #092832;
        border: 1px solid #17616D;
        border-radius: 11px;
        padding: 13px 20px;
        margin: 0 0 8px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, .12);
    }
    .hero-copy { position: relative; z-index: 1; }
    .hero [data-testid="stHeadingWithActionElements"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    .hero [data-testid="stHeaderActionElements"] { display: none !important; }
    .eyebrow {
        color: #19D5E8;
        font-size: .68rem;
        font-weight: 800;
        letter-spacing: .16em;
    }
    .hero h1 {
        margin: 0;
        padding: 0;
        line-height: 1.15;
        font-size: clamp(1.65rem, 2.3vw, 2.05rem);
    }
    .hero p { color: #C5D5DF; margin: 0; font-size: .86rem; }
    .hero-chip-art { flex: 0 0 144px; width: 144px; height: 88px; opacity: .9; }
    .kpi-card {
        display: flex;
        align-items: center;
        gap: 13px;
        min-height: 76px;
        box-sizing: border-box;
        padding: 10px 13px;
        background: #092832;
        border: 1px solid #174650;
        border-radius: 11px;
        box-shadow: inset 0 1px 0 rgba(25, 213, 232, .04), 0 7px 20px rgba(0,0,0,.13);
    }
    .kpi-icon {
        display: grid;
        place-items: center;
        flex: 0 0 38px;
        width: 38px;
        height: 38px;
        border: 1px solid #177080;
        border-radius: 9px;
        color: #19D5E8;
        background: rgba(25, 213, 232, .06);
    }
    .kpi-icon svg { width: 21px; height: 21px; stroke: currentColor; }
    .kpi-label { color: #C5D5DF; font-size: .75rem; margin-bottom: 3px; }
    .kpi-value { color: #EAF7FA; font-size: 1.18rem; font-weight: 740; line-height: 1.15; }
    .kpi-detail { color: #AFC4CE; font-size: .7rem; margin-top: 3px; }
    .warning-box {
        background: #241F16;
        border: 1px solid #8A642E;
        border-left: 4px solid #F2BD63;
        border-radius: 10px;
        padding: 9px 13px;
        color: #F8E3B7;
        margin: 8px 0 11px;
        font-size: .82rem;
        line-height: 1.4;
    }
    .small-note { color: #C5D5DF; font-size: .76rem; }
    .status-card {
        background: #092832;
        border: 1px solid #174650;
        border-radius: 11px;
        padding: 9px 11px;
    }
    .sidebar-brand {
        color: #EAF7FA;
        font-size: 1rem;
        font-weight: 780;
        letter-spacing: .04em;
        margin: 0;
    }
    .sidebar-brand-mark { color: #19D5E8; margin-right: 7px; }
    .sidebar-subtitle { color: #C5D5DF; font-size: .74rem; margin: 2px 0 5px; }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] {
        gap: 2px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label[data-baseweb="radio"] {
        min-height: 31px;
        padding: 3px 7px;
        margin: 0;
        border: 1px solid transparent;
        border-radius: 6px;
        color: #DCEAF0 !important;
        transition: background-color .12s ease, color .12s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label[data-baseweb="radio"] p {
        color: #DCEAF0 !important;
        font-size: .81rem;
        font-weight: 640;
        line-height: 1.15;
        white-space: nowrap;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label[data-baseweb="radio"] p::first-letter {
        color: #19D5E8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label[data-baseweb="radio"]:hover {
        background: #0B3440;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label[data-baseweb="radio"]:has(input[type="radio"]:checked) {
        background: #087F83;
        border: 1px solid #19D5E8;
        color: #FFFFFF !important;
        box-shadow: inset 3px 0 #19D5E8;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label[data-baseweb="radio"]:has(input[type="radio"]:checked) p {
        color: #FFFFFF !important;
        font-weight: 750;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
        accent-color: #19D5E8;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        background: transparent !important;
        border: 1px solid #19D5E8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child > div {
        background: #071D25 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input[type="radio"]:checked) > div:first-child {
        background: #19D5E8 !important;
        border-color: #19D5E8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input[type="radio"]:checked) > div:first-child > div {
        background: #19D5E8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #AFC5D0 !important;
        font-size: .72rem;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #AFC5D0 !important;
    }
    .st-key-overview-top-grid [data-testid="stHorizontalBlock"],
    .st-key-overview-bottom-grid [data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }
    .st-key-overview-top-grid { margin-top: .35rem; }
    .st-key-overview-bottom-grid { margin-top: .25rem; }
    .st-key-overview-top-grid [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-overview-bottom-grid [data-testid="stVerticalBlockBorderWrapper"] {
        height: 100%;
        box-sizing: border-box;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #092832;
        border-color: #174650;
        border-radius: 10px;
    }
    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] { min-width: 0; }
    .stButton > button, .stDownloadButton > button {
        border: 1px solid #1A6673;
        border-radius: 7px;
        font-weight: 650;
        background: #0B3440;
        color: #EAF7FA;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #19D5E8;
        color: #fff;
        background: #0C4652;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #174650;
        border-radius: 9px;
        overflow: hidden;
    }
    [data-testid="stExpander"] {
        border: 1px solid #174650;
        border-radius: 9px;
        background: #092832;
    }
    [data-testid="stFileUploader"] section {
        border: 1px dashed #1A6673;
        border-radius: 9px;
        background: #092832;
    }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stTextArea"] textarea {
        background: #071D25;
        border-color: #1A505B;
        color: #EAF7FA;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-testid="stMultiSelect"] [data-baseweb="select"] *,
    [data-testid="stCheckbox"] label,
    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] * {
        color: #EAF7FA;
    }
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stSubheader"] {
        color: #FFFFFF;
    }
    hr { border-color: #174650; }
    .arch-diagram { color: #EAF7FA; }
    .arch-panel {
        background: #092832;
        border: 1px solid #174650;
        border-radius: 12px;
        padding: 17px;
        margin: 12px 0;
    }
    .arch-panel-title { color: #19D5E8; font-size: .74rem; font-weight: 800; letter-spacing: .1em; margin: 0 0 12px; }
    .arch-flow { display:grid; grid-template-columns:minmax(0,1fr); justify-items:stretch; max-width:900px; margin:0 auto; }
    .arch-node {
        min-width:0;
        background:#0B3440;
        border:1px solid #1A505B;
        border-left:4px solid #19D5E8;
        border-radius:8px;
        padding:11px 14px;
        overflow-wrap:anywhere;
    }
    .arch-node.safety { background:#241F16; border-color:#8A642E; border-left-color:#F2BD63; }
    .arch-node.output { border-left-color:#00AFA3; }
    .arch-step { display:block; color:#91B2BC; font-size:.65rem; font-weight:800; letter-spacing:.1em; margin-bottom:3px; }
    .arch-node strong { display:block; color:#EAF7FA; font-size:.91rem; }
    .arch-node span:last-child { display:block; color:#B7D0D6; font-size:.79rem; line-height:1.4; margin-top:3px; }
    .arch-arrow { color:#19D5E8; font-size:1.2rem; font-weight:800; line-height:1; text-align:center; padding:5px 0; }
    .arch-branch-label { color:#91B2BC; font-size:.8rem; line-height:1.45; margin:0 0 12px; }
    .arch-analytics, .arch-endpoints { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
    .arch-analytics .arch-node { border-left-color:#00AFA3; height:100%; box-sizing:border-box; }
    [data-testid="stPlotlyChart"] { min-width: 0; }
    [data-testid="stPlotlyChart"] .xtick text,
    [data-testid="stPlotlyChart"] .ytick text,
    [data-testid="stPlotlyChart"] .legend text {
        fill: #D7E5EA !important;
        font-size: 12px !important;
    }
    .st-key-presentation-layout [data-testid="stCheckbox"] {
        padding: 3px 7px;
    }
    @media (max-width: 900px) {
        .hero-chip-art { flex-basis: 120px; width: 120px; }
        .kpi-card { padding: 12px; }
        .kpi-value { font-size: 1.08rem; }
    }
    @media (max-width: 640px) {
        .block-container { padding: .75rem .7rem 1.5rem; }
        .hero { min-height: 0; padding: 14px; }
        .hero-chip-art { display: none; }
        .kpi-card { min-height: 70px; padding: 8px; gap: 7px; }
        .kpi-icon { flex-basis: 32px; width: 32px; height: 32px; }
        .kpi-value { font-size: .92rem; }
        .kpi-label { font-size: .68rem; }
        .arch-panel { padding: 13px; }
        .arch-analytics, .arch-endpoints { grid-template-columns:minmax(0,1fr); }
        .arch-node { padding: 10px 12px; }
    }
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


@st.cache_data(show_spinner=False)
def compare_reference_models() -> pd.DataFrame:
    return compare_regression_models(
        REFERENCE_PARTITIONS,
        MODEL_ARTIFACT["model"],
        MODEL_METRICS,
    )


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#D7E5EA", "family": "Segoe UI", "size": 13},
        margin={"l": 10, "r": 10, "t": 30, "b": 32},
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 12},
            "orientation": "h",
            "y": 1.12,
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis={
            "gridcolor": "#174650",
            "zerolinecolor": "#28616B",
            "tickfont": {"size": 12},
            "title_font": {"size": 13},
            "automargin": True,
        },
        yaxis={
            "gridcolor": "#174650",
            "zerolinecolor": "#28616B",
            "tickfont": {"size": 12},
            "title_font": {"size": 13},
            "automargin": True,
        },
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


def render_kpi_card(
    column: Any,
    label: str,
    value: str,
    detail: str,
    icon_svg: str,
) -> None:
    column.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-icon">{icon_svg}</div>
          <div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-detail">{detail}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
      <div class="hero-copy">
        <div class="eyebrow">SEMICONDUCTOR TEST INTELLIGENCE / ENGINEERING WORKBENCH</div>
        <h1>PLL-Smart 2.1</h1>
        <p>PLL RMS Jitter Analysis &amp; Engineering Dashboard</p>
      </div>
      <svg class="hero-chip-art" viewBox="0 0 190 118" role="img" aria-label="Semiconductor wafer and chip illustration">
        <g fill="none" stroke="#19D5E8" stroke-width="1">
          <circle cx="78" cy="59" r="48" opacity=".42"/>
          <circle cx="78" cy="59" r="38" opacity=".25"/>
          <path d="M30 59h96M78 11v96M44 25l68 68M112 25 44 93" opacity=".22"/>
          <circle cx="78" cy="59" r="27" stroke-dasharray="2 5" opacity=".72"/>
          <rect x="126" y="34" width="42" height="42" rx="4" fill="#092832" opacity=".96"/>
          <rect x="135" y="43" width="24" height="24" rx="2" opacity=".86"/>
          <path d="M133 27v7m9-7v7m9-7v7m9-7v7m-27 42v7m9-7v7m9-7v7m9-7v7m-43-42h7m-7 9h7m-7 9h7m-7 9h7m42-27h7m-7 9h7m-7 9h7m-7 9h7" opacity=".8"/>
          <path d="M104 59h18m-44-35v8m-24 27h9m23 24v8" stroke="#00AFA3" opacity=".9"/>
        </g>
        <circle cx="53" cy="41" r="2" fill="#19D5E8" opacity=".75"/>
        <circle cx="93" cy="74" r="2" fill="#00AFA3" opacity=".9"/>
      </svg>
    </div>
    """,
    unsafe_allow_html=True,
)
feature_icon = (
    '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16v14H4z"/>'
    '<path d="M8 9h8M8 13h8M8 17h4"/></svg>'
)
kpi_columns = st.columns(4, gap="small")
render_kpi_card(
    kpi_columns[0],
    "Jitter Dataset",
    f"{sum(REFERENCE_DIAGNOSTICS['raw_partition_counts'].values()):,}",
    "records",
    feature_icon,
)
render_kpi_card(
    kpi_columns[1],
    "Lock-Time Dataset",
    f"{LOCK_METRICS['raw_rows']:,}",
    "records",
    '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/>'
    '<path d="M12 7v5l3 2"/></svg>',
)
render_kpi_card(
    kpi_columns[2],
    "Input Features",
    str(len(FEATURE_COLUMNS)),
    "approved model inputs",
    '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h6v6H4zM14 5h6v6h-6zM4 15h6v5H4zM14 15h6v5h-6z"/></svg>',
)
render_kpi_card(
    kpi_columns[3],
    "Target",
    "RMS Jitter",
    "picoseconds (ps)",
    '<svg viewBox="0 0 24 24" fill="none" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round"><path d="M3 13h4l3-7 4 12 3-7h4"/></svg>',
)
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><span class="sidebar-brand-mark">'
        '<svg viewBox="0 0 24 24" width="21" height="21" fill="none" '
        'stroke="currentColor" stroke-width="1.5" stroke-linecap="round">'
        '<rect x="6" y="6" width="12" height="12" rx="2"/>'
        '<path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/>'
        '</svg></span> PLL-Smart 2.1</div><div class="sidebar-subtitle">'
        'PLL Test Analytics</div>',
        unsafe_allow_html=True,
    )
    presentation_mode = st.checkbox(
        "Presentation layout",
        help="Hide secondary status details and use the full available dashboard width.",
        key="presentation-layout",
    )
    st.markdown(
        '<div class="status-card"><div class="eyebrow">SYSTEM STATUS</div>'
        '<div style="font-size:1.02rem;font-weight:750;color:#eaf7ff;margin-top:5px;">'
        '● &nbsp; MODEL ONLINE</div><div class="small-note">Research mode · synthetic data</div></div>',
        unsafe_allow_html=True,
    )
    navigation_items = [
        ("◈", "Executive Overview"),
        ("▤", "Dataset Explorer"),
        ("⌖", "Manual Prediction"),
        ("▦", "Batch Prediction"),
        ("◉", "Actual vs Predicted"),
        ("◷", "Adaptive Lock-Time"),
        ("∿", "Model Performance"),
        ("⇄", "Data Drift Monitoring"),
        ("◴", "Prediction History"),
        ("⊞", "Reports & Recommendations"),
        ("✳", "Explainable AI"),
        ("⌘", "Engineering What-if"),
        ("▣", "Dataset Management"),
    ]
    navigation_icons = {label: icon for icon, label in navigation_items}
    st.caption("ENGINEERING WORKSPACES")
    selected_section = st.radio(
        "Engineering workspaces",
        [label for _, label in navigation_items],
        format_func=lambda label: f"{navigation_icons[label]}  {label}",
        label_visibility="collapsed",
        key="selected_workspace",
    )
    st.markdown("---")
    st.caption("DATASET INPUT")
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

if presentation_mode:
    st.markdown(
        """
        <style>
        .block-container { max-width: none; padding: .6rem 1.15rem .8rem; }
        [data-testid="stSidebar"] .status-card { display: none; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            font-size: .69rem;
        }
        .hero { min-height: 92px; }
        .kpi-card { min-height: 70px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

# 1. Executive Overview
if selected_section == "Executive Overview":
    with st.container(key="overview-top-grid"):
        performance_column, comparison_column = st.columns([1, 1], gap="small")
        with performance_column:
            with st.container(border=True):
                st.markdown("#### Model Performance")
                st.caption(
                    f"Saved model · {MODEL_METRICS['test_rows']:,} held-out records"
                )
                metric_columns = st.columns(3, gap="small")
                metric_columns[0].metric("MAE", f"{MODEL_METRICS['test_mae_ps']:.4f} ps")
                metric_columns[1].metric("RMSE", f"{MODEL_METRICS['test_rmse_ps']:.4f} ps")
                metric_columns[2].metric("R²", f"{MODEL_METRICS['test_r2']:.4f}")
                st.caption("Synthetic holdout only · not production-qualified")
        with comparison_column:
            with st.container(border=True):
                st.markdown("#### Model Comparison")
                comparison_results = compare_reference_models()
                error_metrics = comparison_results.melt(
                    id_vars="Model",
                    value_vars=["MAE (ps)", "RMSE (ps)"],
                    var_name="Metric",
                    value_name="Error (ps)",
                )
                comparison_fig = px.bar(
                    error_metrics,
                    x="Model",
                    y="Error (ps)",
                    color="Metric",
                    barmode="group",
                    color_discrete_sequence=[CYAN, BLUE],
                    labels={"Model": "", "Error (ps)": "Error (ps)"},
                )
                comparison_fig.update_xaxes(tickangle=-16, tickfont={"size": 11})
                st.plotly_chart(                                                chart_layout(comparison_fig, 150), width="stretch")
                comparison_columns = [
                    column
                    for column in ("Model", "MAE (ps)", "RMSE (ps)", "R²")
                    if column in comparison_results.columns
                ]
                st.dataframe(
                    comparison_results.loc[:, comparison_columns].style.format(
                        {
                            column: "{:.4f}"
                            for column in comparison_columns
                            if column != "Model"
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                    height=100,
                )

    with st.container(key="overview-bottom-grid"):
        actual_column, insights_column = st.columns([1, 1], gap="small")
        with actual_column:
            with st.container(border=True):
                st.markdown("#### Actual vs Predicted RMS Jitter")
                prediction_sample = ORIGINAL_TEST_PREDICTIONS.sample(
                    n=min(2500, len(ORIGINAL_TEST_PREDICTIONS)),
                    random_state=42,
                )
                actual_predicted_fig = go.Figure()
                actual_predicted_fig.add_trace(
                    go.Scatter(
                        x=prediction_sample["actual_jitter_ps"],
                        y=prediction_sample["predicted_jitter_ps"],
                        mode="markers",
                        name="Synthetic test dies",
                        marker={"color": CYAN, "size": 5, "opacity": 0.48},
                        hovertemplate=(
                            "Measured: %{x:.3f} ps<br>Predicted: %{y:.3f} ps"
                            "<extra></extra>"
                        ),
                    )
                )
                range_min = float(
                    min(
                        prediction_sample["actual_jitter_ps"].min(),
                        prediction_sample["predicted_jitter_ps"].min(),
                    )
                )
                range_max = float(
                    max(
                        prediction_sample["actual_jitter_ps"].max(),
                        prediction_sample["predicted_jitter_ps"].max(),
                    )
                )
                actual_predicted_fig.add_trace(
                    go.Scatter(
                        x=[range_min, range_max],
                        y=[range_min, range_max],
                        mode="lines",
                        name="Ideal agreement",
                        line={"color": MUTED, "dash": "dash", "width": 1},
                        hoverinfo="skip",
                    )
                )
                actual_predicted_fig.update_layout(
                    xaxis_title="Measured RMS jitter (ps)",
                    yaxis_title="Predicted RMS jitter (ps)",
                    showlegend=False,
                )
                st.plotly_chart(
                    chart_layout(actual_predicted_fig, 210), width="stretch"
                )
        with insights_column:
            with st.container(border=True):
                st.markdown("#### Engineering Insights")
                st.markdown(
                    f"- **Held-out fit:** MAE {MODEL_METRICS['test_mae_ps']:.4f} ps · "
                    f"RMSE {MODEL_METRICS['test_rmse_ps']:.4f} ps · "
                    f"R² {MODEL_METRICS['test_r2']:.4f} "
                    f"({MODEL_METRICS['test_rows']:,} synthetic records)."
                    f"\n- **Adaptive wait:** {LOCK_METRICS['mean_adaptive_wait_us']:.2f} µs "
                    f"vs {LOCK_METRICS['mean_fixed_wait_us']:.2f} µs fixed "
                    f"({LOCK_METRICS['savings_percent']:.2f}% illustrative reduction)."
                )
                st.markdown(
                    '<div class="warning-box"><strong>RUN FULL JITTER TEST</strong><br>'
                    'Global safety inhibit remains active. The illustrative held-out rule '
                    'included <strong>52 measured failing dies among 96,121 candidates</strong>. '
                    'Never skip the full test.</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Synthetic-data research only; results do not establish production "
                    "qualification or a DPPM guarantee."
                )

    with st.expander("Workflow and system architecture"):
        workflow = (
            "Dataset",
            "Validation",
            "Seven approved features",
            "Saved-model prediction",
            "Calibration bound",
            "Full-test recommendation",
        )
        for row_start in (0, 3):
            workflow_columns = st.columns(3, gap="small")
            for index, (column, stage) in enumerate(
                zip(workflow_columns, workflow[row_start : row_start + 3]),
                start=row_start + 1,
            ):
                with column:
                    st.markdown(f"**{index:02d} · {stage}**")
        st.caption(
            "Uploads are validated and analyzed in memory. The saved model, source datasets, "
            "calibration, and report-generation logic are not changed by dashboard use. "
            "The app does not connect to PLL/ATE hardware and is not production-qualified."
        )

# 2. Dataset Explorer
if selected_section == "Dataset Explorer":
    st.subheader("Dataset Explorer")
    explorer_options = [
        "Original: Train partition",
        "Original: Calibration partition",
        "Original: Test partition",
    ]
    if uploaded_frame is not None:
        explorer_options.insert(0, "Uploaded jitter dataset")
    source_choice = st.selectbox(
        "Dataset to inspect",
        explorer_options,
        key="explorer_dataset_source",
    )
    is_uploaded_exploration = source_choice == "Uploaded jitter dataset"
    if is_uploaded_exploration:
        explore_frame = uploaded_frame
    else:
        partition_name = (
            source_choice.split(":", 1)[1]
            .replace("partition", "")
            .strip()
            .lower()
        )
        explore_frame = REFERENCE_PARTITIONS[partition_name]
    profile = (
        upload_profile
        if is_uploaded_exploration and upload_profile is not None
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
            if upload_validation is None:
                st.warning(
                    upload_processing_error
                    or "Upload validation is unavailable; no validation conclusions are shown."
                )
            else:
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
                    "Dataset": source_choice,
                }
            )
        )
    if is_uploaded_exploration:
        training_values = REFERENCE_TRAIN[selected_feature].sample(
            n=min(8000, len(REFERENCE_TRAIN)), random_state=42
        )
        distribution_rows.append(
            pd.DataFrame({"Value": training_values, "Dataset": "Original: Train reference"})
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
    if is_uploaded_exploration:
        st.caption("The uploaded file is parsed in memory; its original bytes are not modified.")
    else:
        st.caption(
            "Original split labels are preserved. Partitions are displayed independently "
            "and are not mixed."
        )

# 3. Batch Prediction
if selected_section == "Batch Prediction":
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
if selected_section == "Manual Prediction":
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
if selected_section == "Actual vs Predicted":
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
if selected_section == "Adaptive Lock-Time":
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
if selected_section == "Model Performance":
    st.subheader("Model Performance & Comparison")
    st.caption(
        "Synthetic/illustrative evaluation on the original held-out test partition. "
        "The saved model remains read-only; comparison estimators fit only on the "
        "original training partition."
    )
    show_metrics(
        [
            ("Held-out MAE", f"{MODEL_METRICS['test_mae_ps']:.5f} ps", "original test partition"),
            ("Held-out RMSE", f"{MODEL_METRICS['test_rmse_ps']:.5f} ps", None),
            ("Held-out R²", f"{MODEL_METRICS['test_r2']:.5f}", None),
            ("Calibration guard band", f"{MODEL_METRICS['guard_band_ps']:.5f} ps", "calibration partition"),
        ]
    )
    st.markdown("#### Baseline comparison on the same untouched test set")
    st.info(
        "DummyRegressor and LinearRegression are simple reference baselines. "
        "The comparison HistGradientBoostingRegressor is a fresh train-only fit; "
        "it is distinct from the existing saved model shown in the table. "
        "Previously reported saved-model holdout metrics are retained alongside "
        "the current read-only evaluation. No test rows are used for fitting or tuning."
    )
    comparison_results = compare_reference_models()
    display_columns = [
        "Model",
        "MAE (ps)",
        "RMSE (ps)",
        "R²",
        "Training rows",
        "Test rows",
        "Evaluation",
    ]
    previous_columns = [
        "Previously reported MAE (ps)",
        "Previously reported RMSE (ps)",
        "Previously reported R²",
    ]
    visible_columns = [
        column
        for column in (*display_columns, *previous_columns)
        if column in comparison_results.columns
    ]
    st.dataframe(
        comparison_results.loc[:, visible_columns].style.format(
            {
                column: "{:.5f}"
                for column in (
                    "MAE (ps)",
                    "RMSE (ps)",
                    "R²",
                    *previous_columns,
                )
                if column in comparison_results.columns
            }
        ),
        width="stretch",
        hide_index=True,
    )
    comparison_left, comparison_right = st.columns(2)
    with comparison_left:
        error_metrics = comparison_results.melt(
            id_vars="Model",
            value_vars=["MAE (ps)", "RMSE (ps)"],
            var_name="Metric",
            value_name="Error (ps)",
        )
        comparison_fig = px.bar(
            error_metrics,
            x="Model",
            y="Error (ps)",
            color="Metric",
            barmode="group",
            title="Held-out error metrics (lower is better)",
            color_discrete_sequence=[CYAN, BLUE],
        )
        comparison_fig.update_xaxes(tickangle=-20)
        st.plotly_chart(chart_layout(comparison_fig), width="stretch")
    with comparison_right:
        r2_fig = px.bar(
            comparison_results,
            x="Model",
            y="R²",
            title="Held-out coefficient of determination",
            color_discrete_sequence=[CYAN],
        )
        r2_fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
        r2_fig.update_xaxes(tickangle=-20)
        st.plotly_chart(chart_layout(r2_fig), width="stretch")
    st.caption(
        "Every row uses the same validated test partition and the seven approved "
        "features. These synthetic results are not production qualification; the "
        "safety layer continues to recommend RUN FULL JITTER TEST."
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
if selected_section == "Data Drift Monitoring":
    st.subheader("Data Drift Monitoring")
    st.caption(
        "Compares uploaded numeric features with the original training partition. "
        "Drift warnings are input-distribution monitoring signals, not measured "
        "prediction failures."
    )
    if uploaded_frame is None:
        st.info("Upload a CSV with candidate feature columns to compare against training data.")
    elif (
        upload_processing_error
        or upload_validation is None
        or upload_drift is None
        or upload_drift_summary is None
    ):
        st.error(
            "Drift analysis is unavailable because uploaded-data analysis did not "
            "complete. Do not interpret partial results; keep RUN FULL JITTER TEST. "
            + (upload_processing_error or "")
        )
    else:
        drift_comparison = upload_drift
        drift_summary = upload_drift_summary
        drift_validation = upload_validation
        missing_required = drift_validation["missing_columns"]
        missing_feature_values = sum(
            drift_validation["missing_value_counts"].get(feature, 0)
            for feature in FEATURE_COLUMNS
        )
        invalid_numeric_values = sum(
            drift_validation["invalid_type_counts"].get(feature, 0)
            for feature in FEATURE_COLUMNS
        )
        if missing_required:
            st.error(
                "Incompatible upload schema. Missing approved model feature(s): "
                + ", ".join(FEATURE_LABELS[feature] for feature in missing_required)
                + ". No predictions are available for missing inputs."
            )
        else:
            st.success(
                "Upload schema contains all seven approved model features. "
                "Row-level validation still determines which records can be scored."
            )
        if drift_summary["warning_features"]:
            st.warning(
                f"Distribution shift indicators were found for "
                f"{drift_summary['warning_features']} of "
                f"{drift_summary['compared_features']} comparable features. "
                "This is not, by itself, evidence of model failure."
            )
        else:
            st.info(drift_summary["status"])
        show_metrics(
            [
                ("Compared features", str(drift_summary["compared_features"]), None),
                (
                    "Insufficient comparison features",
                    str(drift_summary["insufficient_features"]),
                    "no finite upload values",
                ),
                ("Features with shift warnings", str(drift_summary["warning_features"]), None),
                (
                    "Invalid feature rows",
                    f"{drift_validation['invalid_feature_row_count']:,}",
                    "not scored",
                ),
                (
                    "Out-of-range rows",
                    f"{drift_validation['out_of_training_range_row_count']:,}",
                    "may overlap invalid rows",
                ),
                (
                    "Missing feature values",
                    f"{missing_feature_values:,}",
                    f"{len(missing_required)} missing feature columns",
                ),
            ]
        )
        validation_left, validation_right = st.columns(2)
        with validation_left:
            st.markdown("#### Upload schema and validation")
            st.write(
                f"**Columns:** {uploaded_frame.shape[1]} · "
                f"**Rows:** {len(uploaded_frame):,} · "
                f"**Raw missing cells:** {int(uploaded_frame.isna().sum().sum()):,}"
            )
            st.write(
                f"**Invalid numeric feature values:** {invalid_numeric_values:,} · "
                f"**Duplicate rows/IDs:** {drift_validation['duplicate_row_count']:,}"
            )
            if missing_required:
                st.write(
                    "**Missing required columns:** "
                    + ", ".join(FEATURE_LABELS[name] for name in missing_required)
                )
        with validation_right:
            st.markdown("#### Feature-level distribution comparison")
            st.caption(
                "The existing warning rule flags a feature when an uploaded value is "
                "outside its saved training range or the two-sided KS-test p-value is "
                "below 0.01. This monitoring threshold is not a production failure limit."
            )
        if drift_comparison.empty:
            st.info(
                "No approved features were available for a reference comparison. "
                "Check the upload schema and numeric values."
            )
        else:
            shifted_features: list[str] = []
            for _, feature_row in drift_comparison.iterrows():
                if feature_row["drift_warning"] != "SHIFT WARNING":
                    continue
                feature = feature_row["feature"]
                reasons: list[str] = []
                out_of_range_percent = feature_row.get(
                    "out_of_training_range_percent", np.nan
                )
                ks_p_value = feature_row.get("ks_p_value", np.nan)
                if pd.notna(out_of_range_percent) and out_of_range_percent > 0:
                    reasons.append(
                        f"{out_of_range_percent:.2f}% outside the training range"
                    )
                if pd.notna(ks_p_value) and ks_p_value < 0.01:
                    reasons.append(f"KS p-value {ks_p_value:.3g} < 0.01")
                shifted_features.append(
                    f"**{FEATURE_LABELS.get(feature, feature)}**: "
                    + ("; ".join(reasons) if reasons else "shift indicator")
                )
            if shifted_features:
                st.warning(
                    "Features with distribution-shift indicators:\n\n"
                    + "\n\n".join(f"- {item}" for item in shifted_features)
                )
            drift_display = drift_comparison.copy()
            drift_display["feature"] = drift_display["feature"].map(
                lambda feature: FEATURE_LABELS.get(feature, feature)
            )
            drift_display = drift_display.rename(
                columns={
                    "feature": "Feature",
                    "uploaded_count": "Uploaded numeric values",
                    "training_count": "Training numeric values",
                    "training_mean": "Training mean",
                    "uploaded_mean": "Uploaded mean",
                    "mean_change": "Mean change",
                    "mean_change_percent": "Mean change (%)",
                    "training_std": "Training standard deviation",
                    "uploaded_std": "Uploaded standard deviation",
                    "std_change": "Standard deviation change",
                    "ks_statistic": "KS statistic",
                    "ks_p_value": "KS p-value",
                    "out_of_training_range_percent": "Outside training range (%)",
                    "drift_warning": "Distribution indicator",
                }
            )
            st.dataframe(drift_display, width="stretch", hide_index=True)
        available = [feature for feature in FEATURE_COLUMNS if feature in uploaded_frame]
        if available:
            choice = st.selectbox(
                "Distribution comparison",
                available,
                format_func=lambda value: FEATURE_LABELS[value],
                key="drift_feature",
                help="Compare the uploaded feature distribution with the original training partition.",
            )
            values = (
                pd.to_numeric(uploaded_frame[choice], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
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
                labels={
                    "value": FEATURE_LABELS[choice],
                    "dataset": "Dataset",
                    "density": "Probability density",
                },
                color_discrete_sequence=[BLUE, CYAN],
            )
            fig.update_xaxes(title_text=FEATURE_LABELS[choice])
            fig.update_yaxes(title_text="Probability density")
            st.plotly_chart(chart_layout(fig), width="stretch")
        st.caption(
            "Missing/out-of-range/duplicate records are validation findings, not prediction "
            "errors. No uploaded actual measurements are used as model inputs. A shift warning "
            "does not prove model failure; absence of a warning does not establish safety."
        )

# 9. Prediction History
if selected_section == "Prediction History":
    st.subheader("Prediction History")
    st.caption(
        "History is session-based and is not permanent database storage. It is cleared "
        "when the session is cleared or expires."
    )
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
        history_columns = [
            "timestamp_utc",
            "source",
            "die_id",
            "predicted_jitter_ps",
            "guard_band_ps",
            "upper_prediction_bound_ps",
            "validation_status",
            "recommendation",
            "reason",
        ]
        visible_history_columns = [
            column for column in history_columns if column in view_history.columns
        ]
        history_display = view_history.loc[:, visible_history_columns].copy()
        if "timestamp_utc" in history_display.columns:
            history_display["timestamp_utc"] = pd.to_datetime(
                history_display["timestamp_utc"], errors="coerce", utc=True
            )
        if "die_id" in history_display.columns:
            history_display["die_id"] = history_display["die_id"].astype("string")
        st.dataframe(
            history_display,
            width="stretch",
            hide_index=True,
            height=440,
            column_config={
                "timestamp_utc": st.column_config.DatetimeColumn(
                    "Prediction timestamp (UTC)",
                    format="YYYY-MM-DD HH:mm:ss",
                ),
                "die_id": st.column_config.TextColumn(
                    "Die identifier",
                    help="Present when the input record supplied a die_id.",
                ),
                "predicted_jitter_ps": st.column_config.NumberColumn(
                    "Predicted jitter (ps)", format="%.5f"
                ),
                "upper_prediction_bound_ps": st.column_config.NumberColumn(
                    "Upper prediction bound (ps)", format="%.5f"
                ),
                "recommendation": st.column_config.TextColumn(
                    "Final engineering recommendation"
                ),
            },
        )
        st.caption(
            "The final recommendation is always RUN FULL JITTER TEST. The upper bound "
            "is the prediction plus the illustrative calibration guard band."
        )
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
if selected_section == "Reports & Recommendations":
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
    if uploaded_frame is not None and (
        upload_validation is None or upload_drift_summary is None
    ):
        report_warnings.append(
            "Uploaded jitter analysis did not complete; uploaded metrics are unavailable."
        )
        report_recommendations.append(
            "Do not act on incomplete upload analysis; RUN FULL JITTER TEST."
        )
    elif uploaded_frame is not None:
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
    st.caption(
        "JSON, PDF, and CSV are generated from the same timestamped report snapshot. "
        "They include the available dataset/model metrics and synthetic-data/full-test "
        "safety warnings; absent optional measurements are reported as unavailable."
    )
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
if selected_section == "Explainable AI":
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
if selected_section == "Engineering What-if":
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

# 13. Dataset Management
if selected_section == "Dataset Management":
    st.subheader("Dataset Management")
    st.info(
        "Reference datasets are loaded read-only. New uploads are parsed and analyzed "
        "separately in memory; uploading data never replaces a reference CSV or retrains "
        "the saved model."
    )

    st.markdown("### Jitter Measurement Dataset")
    jitter_source_loaded = JITTER_DATA_PATH.is_file() and bool(REFERENCE_PARTITIONS)
    if jitter_source_loaded:
        jitter_loaded_frame = pd.concat(
            [
                REFERENCE_PARTITIONS[name]
                for name in ("train", "calibration", "test")
            ],
            ignore_index=True,
        )
        jitter_source_columns = list(REFERENCE_PARTITIONS["train"].columns)
        jitter_schema = pd.DataFrame(
            {
                "Data type": REFERENCE_PARTITIONS["train"].dtypes.astype(str),
                "Missing values (usable rows)": jitter_loaded_frame.isna().sum(),
            }
        )
        jitter_describe = jitter_loaded_frame.describe(
            include="all"
        ).transpose()
        jitter_missing_usable = int(jitter_loaded_frame.isna().sum().sum())
    else:
        jitter_loaded_frame = pd.DataFrame()
        jitter_source_columns = []
        jitter_schema = pd.DataFrame()
        jitter_describe = pd.DataFrame()
        jitter_missing_usable = 0

    jitter_source_rows = int(
        sum(REFERENCE_DIAGNOSTICS["raw_partition_counts"].values())
    )
    jitter_source_invalid = int(REFERENCE_DIAGNOSTICS["invalid_row_count"])
    jitter_partition_rows = [
        {
            "Partition": name.title(),
            "Raw rows": REFERENCE_DIAGNOSTICS["raw_partition_counts"][name],
            "Usable rows": REFERENCE_DIAGNOSTICS["usable_partition_counts"][name],
        }
        for name in ("train", "calibration", "test")
    ]
    st.markdown(f"**Source:** `{JITTER_DATA_PATH.name}`")
    st.success(
        f"Reference dataset loaded successfully · {jitter_source_rows:,} source rows · "
        f"{len(jitter_source_columns)} columns"
        if jitter_source_loaded
        else "Reference jitter dataset did not load."
    )
    show_metrics(
        [
            ("Source rows", f"{jitter_source_rows:,}", None),
            ("Columns", f"{len(jitter_source_columns):,}", None),
            ("Missing values", f"{jitter_missing_usable:,}", "usable loaded rows"),
            ("Invalid records excluded", f"{jitter_source_invalid:,}", "loader validation"),
        ]
    )
    partition_left, partition_right = st.columns([1, 1.4])
    with partition_left:
        st.markdown("#### Preserved source partitions")
        st.dataframe(
            pd.DataFrame(jitter_partition_rows),
            width="stretch",
            hide_index=True,
        )
        if jitter_source_invalid:
            st.caption(
                "The raw partition counts are preserved. Invalid records are excluded "
                "from usable partition frames; the negative lock-time record is reported "
                "by the loader."
            )
    with partition_right:
        st.markdown("#### Source schema and data types")
        st.dataframe(jitter_schema, width="stretch")
    with st.expander("Jitter dataset descriptive statistics"):
        st.dataframe(jitter_describe, width="stretch")

    st.markdown("#### Optional jitter dataset upload")
    jitter_management_upload = st.file_uploader(
        "Upload a separate jitter-measurement CSV",
        type=["csv"],
        key="dataset_management_jitter_upload",
        help=(
            "Requires all seven approved model feature columns. Scoring uses the saved "
            "model and existing validation; this upload does not retrain."
        ),
    )
    if jitter_management_upload is not None:
        try:
            management_jitter_frame = load_uploaded_csv(
                jitter_management_upload.getvalue()
            )
            missing_model_features = [
                feature
                for feature in FEATURE_COLUMNS
                if feature not in management_jitter_frame.columns
            ]
            st.markdown(f"**Uploaded file:** `{jitter_management_upload.name}`")
            show_metrics(
                [
                    ("Upload rows", f"{len(management_jitter_frame):,}", None),
                    ("Upload columns", f"{len(management_jitter_frame.columns):,}", None),
                    (
                        "Missing values",
                        f"{int(management_jitter_frame.isna().sum().sum()):,}",
                        "raw upload",
                    ),
                    (
                        "Required features missing",
                        str(len(missing_model_features)),
                        None,
                    ),
                ]
            )
            if missing_model_features:
                st.error(
                    "Incompatible jitter schema. Missing model features: "
                    + ", ".join(missing_model_features)
                    + ". No predictions were generated."
                )
            else:
                managed_batch, managed_summary, _ = predict_batch(
                    management_jitter_frame,
                    MODEL_ARTIFACT["model"],
                    MODEL_ARTIFACT,
                    global_safety_inhibit=SAFETY_INHIBIT,
                )
                managed_invalid = managed_batch.loc[
                    ~managed_batch["feature_valid"]
                ]
                if managed_summary["invalid_records"]:
                    st.warning(
                        f"Schema compatible; {managed_summary['invalid_records']:,} "
                        "records failed feature validation and were not predicted. "
                        "Valid records remain full-test only."
                    )
                else:
                    st.success(
                        "Jitter schema compatible; all records passed feature validation. "
                        "Every recommendation remains RUN FULL JITTER TEST."
                    )
                show_metrics(
                    [
                        (
                            "Successfully predicted",
                            f"{managed_summary['successfully_predicted_records']:,}",
                            None,
                        ),
                        ("Invalid records", f"{managed_summary['invalid_records']:,}", None),
                        (
                            "Mean predicted jitter",
                            format_metric(
                                managed_summary["mean_predicted_jitter_ps"], " ps"
                            ),
                            None,
                        ),
                        (
                            "Illustrative candidate rows",
                            f"{managed_summary['illustrative_candidate_skip_records']:,}",
                            "not a skip authorization",
                        ),
                    ]
                )
                if not managed_invalid.empty:
                    st.dataframe(
                        managed_invalid[
                            [
                                "die_id_display",
                                "validation_status",
                                "validation_reasons",
                            ]
                        ].head(100),
                        width="stretch",
                        hide_index=True,
                    )
                st.dataframe(
                    managed_batch[
                        [
                            "die_id_display",
                            "predicted_jitter_ps",
                            "upper_prediction_bound_ps",
                            "validation_status",
                            "recommendation",
                        ]
                    ].head(100),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Showing at most 100 scored rows. The upload is independent of the "
                    "sidebar jitter upload and is not saved over project files."
                )
        except (ValueError, TypeError, KeyError) as error:
            st.error(
                f"Jitter upload rejected or could not be analyzed: {error}. "
                "The reference dataset and saved model were not changed."
            )

    st.divider()
    st.markdown("### Lock-Time Dataset")
    lock_source_loaded = LOCKTIME_DATA_PATH.is_file() and not REFERENCE_LOCK_ROWS.empty
    lock_source_columns = [
        "die_id",
        "lock_time_us",
        "fixed_wait_us",
        "adaptive_wait_us",
    ]
    lock_source_frame = REFERENCE_LOCK_ROWS.loc[
        :, [column for column in lock_source_columns if column in REFERENCE_LOCK_ROWS]
    ].copy()
    lock_source_schema = pd.DataFrame(
        {
            "Data type": lock_source_frame.dtypes.astype(str),
            "Missing values (usable rows)": lock_source_frame.isna().sum(),
        }
    )
    lock_source_missing = int(lock_source_frame.isna().sum().sum())
    st.markdown(f"**Source:** `{LOCKTIME_DATA_PATH.name}`")
    st.success(
        f"Reference dataset loaded successfully · "
        f"{LOCK_METRICS['raw_rows']:,} source rows · "
        f"{len(lock_source_columns)} columns"
        if lock_source_loaded
        else "Reference lock-time dataset did not load."
    )
    show_metrics(
        [
            ("Source rows", f"{LOCK_METRICS['raw_rows']:,}", None),
            ("Columns", f"{len(lock_source_columns):,}", None),
            ("Missing values", f"{lock_source_missing:,}", "usable loaded rows"),
            ("Invalid records excluded", f"{LOCK_METRICS['invalid_rows']:,}", "lock-time validation"),
        ]
    )
    lock_schema_left, lock_schema_right = st.columns([1, 1.4])
    with lock_schema_left:
        st.markdown("#### Lock-time validation")
        st.write(
            f"Formula mismatches against the recomputed adaptive wait: "
            f"**{LOCK_METRICS['formula_mismatches']:,}**."
        )
        st.caption(
            "Uploaded lock times are analyzed using the existing illustrative "
            "`min(lock_time_us + margin, fixed_wait)` calculation."
        )
    with lock_schema_right:
        st.markdown("#### Source schema and data types")
        st.dataframe(lock_source_schema, width="stretch")
    with st.expander("Lock-time dataset descriptive statistics"):
        st.dataframe(lock_source_frame.describe(include="all").transpose(), width="stretch")

    st.markdown("#### Optional lock-time dataset upload")
    lock_management_upload = st.file_uploader(
        "Upload a separate lock-time CSV",
        type=["csv"],
        key="dataset_management_locktime_upload",
        help=(
            "Requires a lock_time_us column. Jitter measurement datasets are not accepted "
            "as lock-time-only uploads."
        ),
    )
    if lock_management_upload is not None:
        try:
            management_lock_frame = load_uploaded_csv(
                lock_management_upload.getvalue()
            )
            jitter_signatures = {
                TARGET_COLUMN,
                "actual_jitter_ps",
                "split",
                "jitter_fail",
                "defect_injected",
            }
            if "lock_time_us" not in management_lock_frame.columns:
                st.error(
                    "Incompatible lock-time schema: required column `lock_time_us` "
                    "is missing. No lock-time analysis was run."
                )
            elif jitter_signatures.intersection(management_lock_frame.columns) or set(
                FEATURE_COLUMNS
            ).issubset(management_lock_frame.columns):
                st.error(
                    "This upload appears to be a jitter measurement dataset, not a "
                    "lock-time dataset. Use the jitter upload control instead."
                )
            else:
                managed_lock_rows, managed_lock_summary = analyze_locktime_values(
                    management_lock_frame["lock_time_us"]
                )
                st.markdown(f"**Uploaded file:** `{lock_management_upload.name}`")
                st.success(
                    "Lock-time schema compatible. Values were validated and processed "
                    "in memory; no files or model artifacts were changed."
                )
                show_metrics(
                    [
                        ("Upload rows", f"{managed_lock_summary['total_records']:,}", None),
                        ("Valid lock times", f"{managed_lock_summary['valid_records']:,}", None),
                        ("Invalid lock times", f"{managed_lock_summary['invalid_records']:,}", None),
                        (
                            "Mean adaptive wait",
                            format_metric(
                                managed_lock_summary["mean_adaptive_wait_us"], " µs"
                            ),
                            "illustrative",
                        ),
                    ]
                )
                st.dataframe(
                    managed_lock_rows.head(100),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Showing at most 100 calculated rows. Invalid values are retained "
                    "in the validation output but excluded from summary calculations."
                )
        except (ValueError, TypeError, KeyError) as error:
            st.error(
                f"Lock-time upload rejected or could not be analyzed: {error}. "
                "The reference dataset was not changed."
            )
