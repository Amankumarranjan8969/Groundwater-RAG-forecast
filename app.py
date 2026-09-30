"""NAYA-PANI — Groundwater Intelligence Dashboard (Rajasthan seasonal ML workflow).

Redesigned presentation layer only. Every analytical function, model, metric,
and pipeline call is preserved exactly as before; only the UI/navigation/CSS
around them has been rebuilt in an Apple-style product shell.
"""

from __future__ import annotations
from rag.assistant import answer as rag_answer
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
import streamlit as st

from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pipeline import (
    SEASON_ORDER,
    RunConfig,
    acf_pacf_table,
    build_aggregated_series,
    build_excel_export,
    load_seasonal_workbook,
    rainfall_cross_correlation,
    residual_long_table,
    run_season_pipeline,
    season_model_metrics,
    stationarity_summary,
    taylor_statistics,
    taylor_statistics_matrix,
    validate_data,
    wavelet_power_spectrum,
    MODEL_LIST,
    build_season_forecast_table,
    filter_models,
    fractional_year,
    recursive_forecast_constant,
    season_predictions_dict,
)

# =========================================================================== #
# GLOBAL PLOTLY LIGHT THEME — applied automatically to every figure.
# This fixes the "white text on light background" problem caused by
# Streamlit's default dark Plotly template. Only touches background, fonts,
# gridlines, and borders — scientific colorscales and trace colors are
# left completely untouched.
# =========================================================================== #
def apply_plotly_light_theme(fig):
    """Force a readable light theme on any Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            color="#1D1D1F",
            size=13,
        ),
        title=dict(font=dict(color="#1D1D1F", size=18)),
        legend=dict(
            font=dict(color="#1D1D1F", size=12),
            bgcolor="rgba(255,255,255,0)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=60, b=60),
    )

    fig.update_xaxes(
        title_font=dict(color="#1D1D1F", size=13),
        tickfont=dict(color="#6E6E73", size=11),
        linecolor="rgba(0,0,0,0.12)",
        gridcolor="rgba(0,0,0,0.07)",
        zerolinecolor="rgba(0,0,0,0.12)",
    )
    fig.update_yaxes(
        title_font=dict(color="#1D1D1F", size=13),
        tickfont=dict(color="#6E6E73", size=11),
        linecolor="rgba(0,0,0,0.12)",
        gridcolor="rgba(0,0,0,0.07)",
        zerolinecolor="rgba(0,0,0,0.12)",
    )

    # 3D scenes (surface / scatter_3d) use a different layout key.
    try:
        if fig.layout.scene is not None:
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title_font=dict(color="#1D1D1F"),
                        tickfont=dict(color="#6E6E73"),
                        gridcolor="rgba(0,0,0,0.10)",
                        backgroundcolor="rgba(0,0,0,0)",
                        linecolor="rgba(0,0,0,0.12)",
                    ),
                    yaxis=dict(
                        title_font=dict(color="#1D1D1F"),
                        tickfont=dict(color="#6E6E73"),
                        gridcolor="rgba(0,0,0,0.10)",
                        backgroundcolor="rgba(0,0,0,0)",
                        linecolor="rgba(0,0,0,0.12)",
                    ),
                    zaxis=dict(
                        title_font=dict(color="#1D1D1F"),
                        tickfont=dict(color="#6E6E73"),
                        gridcolor="rgba(0,0,0,0.10)",
                        backgroundcolor="rgba(0,0,0,0)",
                        linecolor="rgba(0,0,0,0.12)",
                    ),
                )
            )
    except Exception:
        pass

    # Secondary / sub-plot axes (e.g. make_subplots with secondary_y).
    for i in range(1, 10):
        xa = f"xaxis{i}"
        ya = f"yaxis{i}"
        try:
            if getattr(fig.layout, xa, None) is not None:
                fig.update_layout(**{xa: dict(
                    title_font=dict(color="#1D1D1F", size=13),
                    tickfont=dict(color="#6E6E73", size=11),
                    linecolor="rgba(0,0,0,0.12)",
                    gridcolor="rgba(0,0,0,0.07)",
                    zerolinecolor="rgba(0,0,0,0.12)",
                )})
            if getattr(fig.layout, ya, None) is not None:
                fig.update_layout(**{ya: dict(
                    title_font=dict(color="#1D1D1F", size=13),
                    tickfont=dict(color="#6E6E73", size=11),
                    linecolor="rgba(0,0,0,0.12)",
                    gridcolor="rgba(0,0,0,0.07)",
                    zerolinecolor="rgba(0,0,0,0.12)",
                )})
        except Exception:
            pass

    return fig


# --- Automatic application: wrap st.plotly_chart once, everywhere ----------
_original_plotly_chart = st.plotly_chart

def _themed_plotly_chart(fig, *args, **kwargs):
    try:
        fig = apply_plotly_light_theme(fig)
    except Exception:
        pass  # never break a chart because of theme application
    return _original_plotly_chart(fig, *args, **kwargs)

st.plotly_chart = _themed_plotly_chart
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Palette — kept for the matplotlib figures (unchanged)
# --------------------------------------------------------------------------- #
PALETTE = {
    "ink":   "#111827",
    "muted": "#5f6b73",
    "teal":  "#126782",
    "sea":   "#1d6f9a",
    "amber": "#d97706",
    "leaf":  "#15803d",
    "rose":  "#9f1239",
}

TAYLOR_MODEL_COLORS = {
    "CatBoost": "#2E86AB", "XGBoost": "#A23B72", "M5 Model Tree": "#F18F01",
    "Random Forest": "#C73E1D", "Cubist": "#6A994E", "AdaBoost + CART": "#8A5A9E",
    "Extra Trees": "#E07A5F",
}
TAYLOR_MODEL_MARKERS = {
    "CatBoost": "o", "XGBoost": "s", "M5 Model Tree": "^", "Random Forest": "D",
    "Cubist": "v", "AdaBoost + CART": "*", "Extra Trees": "P",
}
DEEP_MODEL_COLORS = {
    "PINN (Physics-Informed NN)": "#7C3AED", "Reservoir Computing (ESN)": "#0EA5E9",
    "SARIMA (SARIMAX)": "#DC2626", "LSTM (Deep Learning)": "#059669",
}
DEEP_MODEL_MARKERS = {
    "PINN (Physics-Informed NN)": "o", "Reservoir Computing (ESN)": "s",
    "SARIMA (SARIMAX)": "D", "LSTM (Deep Learning)": "^",
}

st.set_page_config(
    page_title="NAYA-PANI — Groundwater Intelligence",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Apple-style theme CSS
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
        /* ============================================================
           1) FORCE STREAMLIT'S OWN THEME VARIABLES TO LIGHT
           (BaseWeb reads these — overriding them stops dark inheritance)
           ============================================================ */
        :root, html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            --background-color: #F5F5F7 !important;
            --secondary-background-color: #FFFFFF !important;
            --text-color: #1D1D1F !important;
            --primary-color: #0071E3 !important;
            --primary-text-color: #1D1D1F !important;
            --secondary-text-color: #6E6E73 !important;
            --border-color: rgba(0,0,0,0.08) !important;
            --font: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                    "Segoe UI", Roboto, sans-serif;
        }

        /* Hide unnecessary Streamlit chrome */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Keep header visible because it contains the sidebar toggle */
header {
    visibility: visible !important;
}

/* Keep Streamlit header controls available */
[data-testid="stToolbar"] {
    display: flex !important;
    visibility: visible !important;
}

/* Hide only the decorative element */
[data-testid="stDecoration"] {
    display: none !important;
}

        /* ============================================================
           2) GLOBAL APP SURFACE
           ============================================================ */
        .stApp {
            background: #F5F5F7 !important;
            color: #1D1D1F !important;
        }
        .block-container {
            max-width: 1380px;
            padding-top: 1.25rem;
            padding-bottom: 5rem;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #1D1D1F !important;
            letter-spacing: -0.02em;
            font-weight: 700;
        }
        h1 { font-size: 2rem; }
        h2 { font-size: 1.5rem; }
        h3 { font-size: 1.15rem; }

        /* Scope body text colour to Streamlit markdown so it never bleeds
           into Plotly SVG (which controls its own text colour via
           fig.update_layout). */
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span,
        label {
            color: #1D1D1F;
        }

        /* Captions & help text */
        .stCaption,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        small {
            color: #6E6E73 !important;
        }

        /* ============================================================
           3) TOP NAV / BRAND
           ============================================================ */
        .top-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 26px;
            background: rgba(255,255,255,.9);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 22px;
            margin-bottom: 10px;
            box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 30px rgba(0,0,0,.045);
        }
        .top-nav .brand {
            display: flex; align-items: center; gap: 12px;
            font-size: 1.15rem; font-weight: 700;
            letter-spacing: -0.02em; color: #1D1D1F;
        }
        .top-nav .brand-icon { font-size: 1.55rem; }
        .top-nav .brand-subtitle {
            color: #6E6E73;
            font-size: .82rem;
            margin-top: 2px;
        }
        .top-nav .status-pill {
            background: rgba(52,199,89,.12);
            color: #1a8f3c;
            font-weight: 600;
            font-size: .8rem;
            padding: 6px 12px;
            border-radius: 999px;
            border: 1px solid rgba(52,199,89,.22);
            white-space: nowrap;
        }

        /* ============================================================
           4) PAGE HEADINGS
           ============================================================ */
        .page-heading { margin: 26px 0 18px 0; }
        .page-heading .eyebrow {
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .12em;
            color: #6E6E73;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .page-heading h1 {
            margin: 0 0 8px 0;
            font-size: 2.15rem;
            letter-spacing: -0.025em;
            color: #1D1D1F !important;
        }
        .page-heading p {
            color: #6E6E73;
            font-size: 1.02rem;
            max-width: 720px;
            line-height: 1.55;
            margin: 0;
        }

        /* ============================================================
           5) CARDS / METRICS
           ============================================================ */
        .metric-card,
        .feature-card {
            background: rgba(255,255,255,.9);
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 20px;
            padding: 22px;
            box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 30px rgba(0,0,0,.04);
            transition: transform .2s ease, box-shadow .2s ease;
        }
        .metric-card { min-height: 132px; }
        .metric-card:hover,
        .feature-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,.04), 0 14px 35px rgba(0,0,0,.07);
        }
        .metric-label {
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .08em;
            color: #6E6E73;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -.04em;
            color: #1D1D1F;
            margin-top: 8px;
            line-height: 1.1;
        }
        .metric-description {
            color: #6E6E73;
            font-size: .85rem;
            margin-top: 6px;
        }
        .feature-card .feature-icon { font-size: 1.75rem; margin-bottom: 12px; }
        .feature-card h3 { margin: 0 0 8px 0; font-size: 1.05rem; }
        .feature-card p { color: #6E6E73; font-size: .9rem; line-height: 1.5; margin: 0; }

        /* Streamlit metric widget */
        [data-testid="stMetric"] {
            background: #FFFFFF;
            padding: 18px 20px;
            border-radius: 20px;
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 30px rgba(0,0,0,.04);
        }
        [data-testid="stMetric"],
        [data-testid="stMetric"] *,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] *,
        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] * {
            color: #1D1D1F !important;
        }
        [data-testid="stMetricLabel"] { color: #6E6E73 !important; font-weight: 600; }
        [data-testid="stMetricValue"] { font-size: 1.65rem; font-weight: 700; }

        /* ============================================================
           6) SIDEBAR — WHITE PREMIUM PANEL
           ============================================================ */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"] {
            background: #FFFFFF !important;
            color: #1D1D1F !important;
            border-right: 1px solid rgba(0,0,0,0.06) !important;
            box-shadow: 2px 0 12px rgba(0,0,0,0.03) !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div,
        section[data-testid="stSidebar"] * {
            color: #1D1D1F !important;
        }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
        section[data-testid="stSidebar"] small {
            color: #6E6E73 !important;
        }

        /* ============================================================
           7) SELECTBOX / MULTISELECT — CLOSED CONTROL
           ============================================================ */
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] > div > div,
        [data-baseweb="select"] > div,
        [data-baseweb="select"] > div[role="combobox"],
        [data-baseweb="select"] > div[aria-expanded] {
            background-color: #FFFFFF !important;
            color: #1D1D1F !important;
            border: 1px solid rgba(0,0,0,0.08) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        [data-baseweb="select"] > div:hover {
            border-color: rgba(0,113,227,0.45) !important;
        }
        [data-baseweb="select"] > div[aria-expanded="true"],
        [data-baseweb="select"] > div:focus-within {
            border-color: #0071E3 !important;
            box-shadow: 0 0 0 3px rgba(0,113,227,0.12) !important;
            background-color: #FFFFFF !important;
        }

        /* Any inner element inside the closed control */
        [data-baseweb="select"] span,
        [data-baseweb="select"] p,
        [data-baseweb="select"] input,
        [data-baseweb="select"] [role="combobox"],
        [data-testid="stSelectbox"] span,
        [data-testid="stSelectbox"] p,
        [data-testid="stSelectbox"] input,
        [data-testid="stMultiSelect"] span,
        [data-testid="stMultiSelect"] p,
        [data-testid="stMultiSelect"] input {
            color: #1D1D1F !important;
            background-color: transparent !important;
        }

        /* The select arrow */
        [data-baseweb="select"] svg,
        [data-testid="stSelectbox"] svg,
        [data-testid="stMultiSelect"] svg {
            fill: #6E6E73 !important;
            color: #6E6E73 !important;
        }
        [data-baseweb="select"] > div:hover svg {
            fill: #0071E3 !important;
            color: #0071E3 !important;
        }

        /* Placeholder */
        [data-baseweb="select"] input::placeholder,
        [data-testid="stSelectbox"] input::placeholder,
        [data-testid="stMultiSelect"] input::placeholder {
            color: #6E6E73 !important;
        }

        /* ============================================================
           8) POPOVER / DROPDOWN MENU (the important one)
           ============================================================ */
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="popover"] div,
        [data-baseweb="menu"],
        [data-baseweb="menu"] > div,
        [role="listbox"],
        ul[role="listbox"],
        [data-baseweb="popover"] [role="listbox"] {
            background-color: #FFFFFF !important;
            color: #1D1D1F !important;
            border: 1px solid rgba(0,0,0,0.08) !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 30px rgba(0,0,0,0.10) !important;
        }

        [data-baseweb="popover"] *,
        [data-baseweb="menu"] *,
        [role="listbox"] *,
        [data-baseweb="popover"] span,
        [data-baseweb="popover"] p,
        [data-baseweb="menu"] span,
        [data-baseweb="menu"] p {
            color: #1D1D1F !important;
            background-color: transparent !important;
        }

        /* Individual options */
        [role="option"],
        [data-baseweb="menu"] [role="option"],
        [role="listbox"] [role="option"],
        [data-baseweb="popover"] [role="option"] {
            background-color: #FFFFFF !important;
            color: #1D1D1F !important;
            padding: 8px 14px !important;
        }
        [role="option"] *,
        [data-baseweb="menu"] [role="option"] *,
        [role="listbox"] [role="option"] * {
            color: #1D1D1F !important;
            background-color: transparent !important;
        }

        /* Hover state */
        [role="option"]:hover,
        [data-baseweb="menu"] [role="option"]:hover,
        [role="listbox"] [role="option"]:hover,
        [role="option"][aria-selected="false"]:hover {
            background-color: #F5F5F7 !important;
        }

        /* Selected state */
        [role="option"][aria-selected="true"],
        [data-baseweb="menu"] [role="option"][aria-selected="true"],
        [role="listbox"] [role="option"][aria-selected="true"],
        [aria-selected="true"] {
            background-color: #EAF3FF !important;
            color: #0071E3 !important;
        }
        [role="option"][aria-selected="true"] *,
        [aria-selected="true"] * {
            color: #0071E3 !important;
            background-color: transparent !important;
        }

        /* ============================================================
           9) MULTISELECT TAGS
           ============================================================ */
        [data-baseweb="tag"],
        [data-testid="stMultiSelect"] [data-baseweb="tag"] {
            background-color: #EAF3FF !important;
            color: #0071E3 !important;
            border: 1px solid rgba(0,113,227,0.15) !important;
            border-radius: 999px !important;
        }
        [data-baseweb="tag"] *,
        [data-baseweb="tag"] span,
        [data-baseweb="tag"] div {
            color: #0071E3 !important;
            background-color: transparent !important;
        }
        [data-baseweb="tag"] svg,
        [data-baseweb="tag"] [role="button"] svg {
            fill: #0071E3 !important;
            color: #0071E3 !important;
        }

        /* ============================================================
           10) TEXT / NUMBER / DATE INPUTS
           ============================================================ */
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input,
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input {
            background-color: #FFFFFF !important;
            color: #1D1D1F !important;
            border: 1px solid rgba(0,0,0,0.08) !important;
            border-radius: 12px !important;
        }
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stNumberInput"] input::placeholder,
        [data-testid="stDateInput"] input::placeholder {
            color: #6E6E73 !important;
        }
        [data-testid="stTextInput"] > div > div,
        [data-testid="stNumberInput"] > div > div,
        [data-testid="stDateInput"] > div > div,
        [data-baseweb="input"],
        [data-baseweb="base-input"] {
            background-color: #FFFFFF !important;
            color: #1D1D1F !important;
            border: 1px solid rgba(0,0,0,0.08) !important;
            border-radius: 12px !important;
        }
        /* number input stepper buttons */
        [data-testid="stNumberInput"] button,
        [data-testid="stNumberInput"] [role="button"] {
            background: #FFFFFF !important;
            color: #1D1D1F !important;
            border: 1px solid rgba(0,0,0,0.08) !important;
        }
        [data-testid="stNumberInput"] button:hover {
            background: #F5F5F7 !important;
        }
        [data-testid="stNumberInput"] button svg {
            fill: #1D1D1F !important;
            color: #1D1D1F !important;
        }

        /* ============================================================
           11) SLIDER
           ============================================================ */
        [data-testid="stSlider"] [role="slider"] {
            background-color: #0071E3 !important;
            border-color: #0071E3 !important;
        }
        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background: #FFFFFF !important;
            border: 2px solid #0071E3 !important;
            box-shadow: 0 2px 6px rgba(0,113,227,0.3) !important;
        }
        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"],
        [data-testid="stSlider"] [data-testid="stTickBar"] * {
            color: #6E6E73 !important;
        }
        [data-testid="stSlider"] [data-baseweb="slider"] [role="progressbar"],
        [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background: rgba(0,113,227,0.25) !important;
        }

        /* ============================================================
           12) RADIO / CHECKBOX
           ============================================================ */
        [data-testid="stRadio"] label { color: #1D1D1F !important; }
        [data-testid="stRadio"] label * { color: #1D1D1F !important; }
        [data-testid="stRadio"] [role="radio"] {
            background: #FFFFFF !important;
            border-color: rgba(0,0,0,0.2) !important;
        }
        [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
            background-color: #0071E3 !important;
            border-color: #0071E3 !important;
        }
        [data-testid="stRadio"] [role="radio"][aria-checked="true"] > div {
            background-color: #0071E3 !important;
        }

        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] label * {
            color: #1D1D1F !important;
        }
        [data-testid="stCheckbox"] [role="checkbox"] {
            background: #FFFFFF !important;
            border-color: rgba(0,0,0,0.2) !important;
        }
        [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"] {
            background-color: #0071E3 !important;
            border-color: #0071E3 !important;
        }

        /* Toggle */
        [data-testid="stToggle"] label,
        [data-testid="stToggle"] label * { color: #1D1D1F !important; }

        /* ============================================================
           13) EXPANDERS
           ============================================================ */
        [data-testid="stExpander"],
        [data-testid="stExpander"] > details,
        details {
            background: #FFFFFF !important;
            border: 1px solid rgba(0,0,0,0.06) !important;
            border-radius: 16px !important;
            overflow: hidden !important;
            box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 30px rgba(0,0,0,.04);
        }
        details summary,
        details summary *,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary *,
        .streamlit-expanderHeader,
        .streamlit-expanderHeader * {
            background: #FFFFFF !important;
            color: #1D1D1F !important;
            font-weight: 600 !important;
        }
        details summary:hover,
        [data-testid="stExpander"] summary:hover {
            background: #F5F5F7 !important;
        }
        details[open] summary {
            border-bottom: 1px solid rgba(0,0,0,0.06) !important;
        }
        details [data-testid="stMarkdownContainer"] *,
        details p, details li, details span {
            color: #1D1D1F !important;
        }
        details summary svg,
        [data-testid="stExpander"] summary svg {
            fill: #1D1D1F !important;
            color: #1D1D1F !important;
        }

        /* ============================================================
           14) TABS
           ============================================================ */
        .stTabs [data-baseweb="tab-list"] {
            background: #FFFFFF !important;
            border-radius: 999px !important;
            border: 1px solid rgba(0,0,0,0.06) !important;
            padding: 6px !important;
            gap: 4px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,.03);
            flex-wrap: wrap;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent !important;
            color: #1D1D1F !important;
            border-radius: 999px !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab"] * { color: #1D1D1F !important; }
        .stTabs [data-baseweb="tab"]:hover { background: #F5F5F7 !important; }
        .stTabs [aria-selected="true"] {
            background: #0071E3 !important;
            box-shadow: 0 3px 10px rgba(0,113,227,0.25) !important;
        }
        .stTabs [aria-selected="true"] * { color: #FFFFFF !important; }

        /* Radio-based horizontal nav used by the app shell */
        [data-testid="stRadio"] > div[role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            background: #FFFFFF !important;
            padding: 6px;
            border-radius: 999px;
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 30px rgba(0,0,0,.04);
            margin-bottom: 12px;
        }
        [data-testid="stRadio"] > div[role="radiogroup"] label {
            background: transparent !important;
            border-radius: 999px !important;
            padding: 8px 16px !important;
            transition: all .18s ease;
            cursor: pointer;
        }
        [data-testid="stRadio"] > div[role="radiogroup"] label:hover {
            background: #F5F5F7 !important;
        }
        [data-testid="stRadio"] > div[role="radiogroup"] label * {
            color: #1D1D1F !important;
            font-weight: 600 !important;
            font-size: .92rem !important;
        }
        [data-testid="stRadio"] > div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        [data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
            background: #0071E3 !important;
            box-shadow: 0 3px 10px rgba(0,113,227,.30);
        }
        [data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) * {
            color: #FFFFFF !important;
        }

        /* ============================================================
           15) BUTTONS
           ============================================================ */
        .stButton > button,
        .stDownloadButton > button,
        .stFormSubmitButton > button,
        button[kind="primary"],
        button[kind="secondary"] {
            background: #0071E3 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 10px 20px !important;
            box-shadow: 0 4px 14px rgba(0,113,227,0.22) !important;
            transition: background .18s ease, transform .18s ease;
        }
        .stButton > button *,
        .stDownloadButton > button *,
        .stFormSubmitButton > button * {
            color: #FFFFFF !important;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stFormSubmitButton > button:hover {
            background: #0077ED !important;
            transform: translateY(-1px);
        }
        .stButton > button:active {
            background: #0058b0 !important;
        }

        /* ============================================================
           16) DATAFRAMES / TABLES
           ============================================================ */
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: #FFFFFF !important;
            border-radius: 16px !important;
            border: 1px solid rgba(0,0,0,0.06) !important;
            padding: 6px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,.03);
        }
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"] * {
            color: #1D1D1F !important;
        }
        [data-testid="stDataFrame"] thead tr th,
        [data-testid="stTable"] thead tr th {
            background: #F5F5F7 !important;
            color: #1D1D1F !important;
            border-bottom: 1px solid rgba(0,0,0,0.06) !important;
        }
        [data-testid="stDataFrame"] tbody tr td,
        [data-testid="stTable"] tbody tr td {
            background: #FFFFFF !important;
            color: #1D1D1F !important;
            border-bottom: 1px solid rgba(0,0,0,0.04) !important;
        }

        /* ============================================================
           17) ALERTS / INFO / WARNING
           ============================================================ */
        [data-testid="stAlert"],
        [data-testid="stAlert"] > div {
            border-radius: 14px !important;
            border: 1px solid rgba(0,0,0,0.06) !important;
            color: #1D1D1F !important;
        }
        [data-testid="stAlert"] *,
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span {
            color: #1D1D1F !important;
        }

        /* ============================================================
           18) FILE UPLOADER
           ============================================================ */
        [data-testid="stFileUploader"] {
            background: #FFFFFF !important;
            border: 1px dashed rgba(0,0,0,0.14) !important;
            border-radius: 22px !important;
            padding: 1rem 1.1rem 1.1rem 1.1rem !important;
            box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 30px rgba(0,0,0,.045);
            transition: border-color .2s ease, box-shadow .2s ease;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #0071E3 !important;
            box-shadow: 0 4px 12px rgba(0,113,227,.12);
        }
        [data-testid="stFileUploader"] > label,
        [data-testid="stFileUploader"] > label * {
            color: #1D1D1F !important;
            font-weight: 600 !important;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(0,113,227,.04) !important;
            border: none !important;
            border-radius: 16px !important;
        }
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] div {
            color: #1D1D1F !important;
        }
        [data-testid="stFileUploaderDropzone"] svg {
            fill: #0071E3 !important;
            color: #0071E3 !important;
        }
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] button * {
            background: #0071E3 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
            padding: 0.4rem 1.1rem !important;
        }
        [data-testid="stFileUploaderFile"] {
            background: rgba(0,113,227,.06) !important;
            border-radius: 12px !important;
            padding: 8px 12px !important;
        }
        [data-testid="stFileUploaderFile"] * {
            color: #1D1D1F !important;
        }

        /* ============================================================
           19) CHAT (RAG) — LIGHT THEME
           ============================================================ */
        [data-testid="stChatMessage"] {
            background: #FFFFFF !important;
            border: 1px solid rgba(0,0,0,0.06) !important;
            border-radius: 16px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,.03);
            padding: 12px 16px !important;
        }
        [data-testid="stChatMessage"] *,
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span,
        [data-testid="stChatMessage"] div,
        [data-testid="stChatMessage"] strong,
        [data-testid="stChatMessage"] em {
            color: #1D1D1F !important;
            background-color: transparent !important;
        }
        [data-testid="stChatMessage"] a { color: #0071E3 !important; }
        [data-testid="stChatMessage"] code {
            background: #F5F5F7 !important;
            color: #1D1D1F !important;
            border-radius: 4px !important;
            padding: 1px 5px !important;
        }
        [data-testid="stChatMessage"] pre {
            background: #F5F5F7 !important;
            border-radius: 8px !important;
        }
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            background: #FFFFFF !important;
            border: 2px solid #0071E3 !important;
        }
        [data-testid="stChatMessageAvatarUser"] svg,
        [data-testid="stChatMessageAvatarAssistant"] svg {
            fill: #0071E3 !important;
            color: #0071E3 !important;
        }

        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div {
            background: #FFFFFF !important;
            border: 1px solid rgba(0,0,0,0.08) !important;
            border-radius: 16px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,.03);
        }
        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] input {
            background: #FFFFFF !important;
            color: #1D1D1F !important;
            border: none !important;
        }
        [data-testid="stChatInput"] textarea::placeholder,
        [data-testid="stChatInput"] input::placeholder {
            color: #6E6E73 !important;
        }
        [data-testid="stChatInput"] button svg {
            fill: #0071E3 !important;
            color: #0071E3 !important;
        }

        /* ============================================================
           20) PLOTLY CONTAINER — ISOLATE FROM GLOBAL TEXT RULES
           Do NOT force color on children; Plotly controls its own SVG
           text colours via apply_plotly_light_theme().
           ============================================================ */
        [data-testid="stPlotlyChart"],
        [data-testid="stPlotlyChart"] > div,
        [data-testid="stPlotlyChart"] .js-plotly-plot,
        [data-testid="stPlotlyChart"] .plot-container,
        [data-testid="stPlotlyChart"] svg {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
        }

        /* Plotly modebar — light, subtle, visible */
        [data-testid="stPlotlyChart"] .modebar {
            background: rgba(255,255,255,0.7) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(0,0,0,0.06) !important;
            backdrop-filter: blur(8px);
        }
        [data-testid="stPlotlyChart"] .modebar-btn svg,
        [data-testid="stPlotlyChart"] .modebar-btn path {
            fill: #6E6E73 !important;
        }
        [data-testid="stPlotlyChart"] .modebar-btn:hover svg,
        [data-testid="stPlotlyChart"] .modebar-btn:hover path {
            fill: #0071E3 !important;
        }

        /* ============================================================
           21) MISC CUSTOM BLOCKS
           ============================================================ */
        .stage-note {
            background: rgba(0,113,227,.06);
            border-left: 3px solid #0071E3;
            padding: .75rem 1rem;
            margin: .25rem 0 1rem 0;
            border-radius: 10px;
            color: #1D1D1F !important;
            font-size: .92rem;
            line-height: 1.5;
        }
        .stage-note * { color: #1D1D1F !important; }

        .notebook-caption {
            color: #6E6E73 !important;
            font-size: .88rem;
            font-style: italic;
        }

        .section-intro {
            background: #FFFFFF;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 20px;
            padding: 20px 24px;
            margin: .5rem 0 1.25rem 0;
            box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 30px rgba(0,0,0,.045);
        }
        .section-intro-title {
            color: #1D1D1F !important;
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: .45rem;
        }
        .section-intro-what {
            color: #1D1D1F !important;
            font-size: .95rem;
            line-height: 1.55;
            margin-bottom: .7rem;
        }
        .section-intro-why {
            color: #1D1D1F !important;
            font-size: .93rem;
            line-height: 1.55;
            background: rgba(0,113,227,.06);
            padding: .7rem .9rem;
            border-radius: 12px;
            border-left: 3px solid #0071E3;
        }
        .section-intro-why * { color: #1D1D1F !important; }

        .landing-hero {
            text-align: center;
            padding: 3rem 1rem 2rem 1rem;
            max-width: 780px;
            margin: 0 auto;
        }
        .landing-hero .eyebrow {
            font-size: .74rem;
            font-weight: 700;
            letter-spacing: .14em;
            color: #0071E3;
            text-transform: uppercase;
            margin-bottom: 14px;
        }
        .landing-hero h1 {
            font-size: 3rem;
            letter-spacing: -0.04em;
            line-height: 1.08;
            color: #1D1D1F !important;
            margin: 0 0 18px 0;
        }
        .landing-hero p {
            font-size: 1.15rem;
            color: #6E6E73;
            line-height: 1.55;
            margin: 0 auto 12px auto;
            max-width: 600px;
        }

        .status-strip {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            padding: 14px 20px;
            background: #FFFFFF;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 18px;
            margin-bottom: 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,.03);
        }
        .status-chip {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: .88rem;
            color: #1D1D1F;
            font-weight: 500;
        }
        .status-chip .dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #34C759;
            box-shadow: 0 0 0 3px rgba(52,199,89,.18);
        }

        .upload-hint {
            background: #FFFFFF;
            border: 1px dashed rgba(0,0,0,0.14);
            border-radius: 12px;
            padding: .65rem 1rem;
            margin: .35rem 0 1rem 0;
            color: #1D1D1F !important;
            font-size: .93rem;
        }
        .upload-hint * { color: #1D1D1F !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Small helpers (unchanged)
# --------------------------------------------------------------------------- #
def metric_text(value: Any, digits: int = 3) -> str:
    try:
        numeric = float(value)
        return "—" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def usable_seasons(data: pd.DataFrame) -> list[str]:
    present = set(data["Season"].unique())
    return [season for season in SEASON_ORDER if season in present]


def selected_result(results: dict[str, dict[str, Any]], key_suffix: str) -> dict[str, Any]:
    choices = list(results)
    season = st.selectbox("Season", choices, key=f"season_{key_suffix}")
    return results[season]


def summary_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    records = []
    for season, result in results.items():
        best = result["metrics"].iloc[0]
        records.append({
            "Season": season,
            "Best model": result["best_model"],
            "Bias-corrected test R²": best["Bias-corrected test R2"],
            "Bias-corrected test RMSE": best["Bias-corrected test RMSE"],
            "Modelling rows": result["lagged_rows"],
        })
    return pd.DataFrame(records)


def _matplotlib_to_streamlit(fig: plt.Figure) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def section_intro(title: str, what: str, why: str) -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-intro-title">{title}</div>
            <div class="section-intro-what">{what}</div>
            <div class="section-intro-why"><b>Why it matters:</b> {why}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Notebook-style matplotlib helpers (unchanged)
# --------------------------------------------------------------------------- #
def _compute_taylor_stats(obs: np.ndarray, pred: np.ndarray) -> dict[str, float] | None:
    obs = np.asarray(obs).flatten()
    pred = np.asarray(pred).flatten()
    min_len = min(len(obs), len(pred))
    if min_len < 2:
        return None
    obs = obs[:min_len]
    pred = pred[:min_len]
    std_obs = np.std(obs, ddof=1)
    if std_obs == 0:
        return None
    std_pred = np.std(pred, ddof=1)
    corr, _ = pearsonr(obs, pred)
    return {"std_ratio": std_pred / std_obs, "corr": corr}


def _plot_taylor_latex_style(ax: plt.Axes, obs: np.ndarray,
                             preds_dict: dict[str, np.ndarray],
                             title: str = "",
                             sigma_max: float | None = None,
                             extra_rmsd: float = 1.6,
                             extra_corr: float = 0.1,
                             model_colors: dict[str, str] | None = None,
                             model_markers: dict[str, str] | None = None) -> None:
    obs = np.asarray(obs).flatten()
    std_obs = np.std(obs, ddof=1)
    if std_obs == 0:
        ax.text(0.5, 0.5, "Observed std = 0", transform=ax.transAxes, ha="center")
        return
    model_stats: dict[str, dict[str, float]] = {}
    for name, pred in preds_dict.items():
        stats = _compute_taylor_stats(obs, pred)
        if stats:
            model_stats[name] = stats
    if not model_stats:
        ax.text(0.5, 0.5, "No valid models", transform=ax.transAxes, ha="center")
        return

    colors = model_colors or TAYLOR_MODEL_COLORS
    markers = model_markers or TAYLOR_MODEL_MARKERS

    ratios = [v["std_ratio"] for v in model_stats.values()]
    max_ratio = max(ratios) if ratios else 1.0
    if sigma_max is None:
        sigma_max = max(1.5, max_ratio * 1.15)

    ax.set_xlim(0, sigma_max)
    ax.set_ylim(0, sigma_max)
    ax.set_aspect("equal")

    theta_quarter = np.linspace(np.pi / 2, 0, 1000)
    ax.plot(sigma_max * np.cos(theta_quarter), sigma_max * np.sin(theta_quarter),
            "k", linewidth=1.2, zorder=1)
    ax.plot([0, sigma_max], [0, 0], "k", linewidth=1.2)
    ax.plot([0, 0], [0, sigma_max], "k", linewidth=1.2)

    sigma_levels = np.arange(0.2, sigma_max + 0.01, 0.2)
    for s in sigma_levels:
        if s > sigma_max or s <= 0:
            continue
        if abs(s - 1.0) < 1e-6:
            continue
        theta = np.linspace(np.pi / 2, 0, 1000)
        ax.plot(s * np.cos(theta), s * np.sin(theta), color="black",
                linestyle=":", linewidth=2.0, alpha=0.7, zorder=1)
    ax.plot(1.0 * np.cos(theta_quarter), 1.0 * np.sin(theta_quarter),
            color="darkred", linewidth=3.0, linestyle=":", label="Observed std", zorder=2)

    theta_rmsd = np.linspace(0, np.pi, 1000)
    x_rmsd = 1.0 + 1.0 * np.cos(theta_rmsd)
    y_rmsd = 1.0 * np.sin(theta_rmsd)
    mask = (x_rmsd >= 0) & (y_rmsd >= 0) & (x_rmsd ** 2 + y_rmsd ** 2 <= sigma_max ** 2)
    if np.any(mask):
        ax.plot(x_rmsd[mask], y_rmsd[mask], color="gray", linewidth=2.0,
                linestyle="-", label="RMSD = 1", zorder=2)

    ref_x = 1.0
    crmsd_levels = np.arange(0.2, sigma_max + 0.01, 0.2)
    cmap = plt.cm.tab10
    arc_colors = [cmap(i % 10) for i in range(len(crmsd_levels))]
    label_angle = np.radians(130)
    cos_a, sin_a = np.cos(label_angle), np.sin(label_angle)
    inward = 0.92
    for idx, r in enumerate(crmsd_levels):
        if r <= 0.1:
            continue
        color = arc_colors[idx]
        theta_circ = np.linspace(0, np.pi, 1000)
        x_circ = ref_x + r * np.cos(theta_circ)
        y_circ = r * np.sin(theta_circ)
        circ_mask = (x_circ >= 0) & (y_circ >= 0) & (x_circ ** 2 + y_circ ** 2 <= sigma_max ** 2)
        if np.any(circ_mask):
            ax.plot(x_circ[circ_mask], y_circ[circ_mask], color=color,
                    linestyle="-", linewidth=1.5, alpha=0.9)
        x_lab, y_lab = ref_x + r * cos_a, r * sin_a
        x_off = ref_x + inward * (x_lab - ref_x)
        y_off = inward * y_lab
        if 0 <= x_off <= sigma_max and 0 <= y_off <= sigma_max:
            ax.text(x_off, y_off, f"{r:.1f}", fontsize=9, fontweight="bold",
                    color=color, ha="center", va="center", rotation=0)

    if extra_rmsd is not None:
        theta_extra = np.linspace(0, np.pi, 1000)
        x_extra = ref_x + extra_rmsd * np.cos(theta_extra)
        y_extra = extra_rmsd * np.sin(theta_extra)
        extra_mask = (x_extra >= 0) & (y_extra >= 0) & (x_extra ** 2 + y_extra ** 2 <= sigma_max ** 2)
        if np.any(extra_mask):
            ax.plot(x_extra[extra_mask], y_extra[extra_mask], color="black", linewidth=2.0,
                    linestyle="-", alpha=0.9, label=f"RMSD = {extra_rmsd:.1f}", zorder=2)
            x_lab, y_lab = ref_x + extra_rmsd * cos_a, extra_rmsd * sin_a
            x_off = ref_x + inward * (x_lab - ref_x)
            y_off = inward * y_lab
            if 0 <= x_off <= sigma_max and 0 <= y_off <= sigma_max:
                ax.text(x_off, y_off, f"{extra_rmsd:.1f}", fontsize=9, fontweight="bold",
                        color="black", ha="center", va="center")

    corr_values = [0.20, 0.40, 0.60, 0.80, 0.90, 0.95, 0.99, 1.00]
    if extra_corr is not None and extra_corr not in corr_values:
        corr_values = sorted(corr_values + [extra_corr])
    for c in corr_values:
        theta = np.arccos(c)
        x_line, y_line = sigma_max * np.cos(theta), sigma_max * np.sin(theta)
        ax.plot([0, x_line], [0, y_line], color="gray", linestyle="--",
                linewidth=2.0, alpha=0.7, zorder=0.5)
        if c == 1.00:
            ax.text(sigma_max, -0.03, "1.00", ha="center", va="top", fontsize=9, color="black")
        else:
            offset = 1.03 if c >= 0.2 else 1.06
            ax.text(x_line * offset, y_line * offset, f"{c:.2f}", fontsize=9,
                    color="black", ha="center", va="center")

    mid_angle = np.pi / 4
    label_x = sigma_max * 1.05 * np.cos(mid_angle)
    label_y = sigma_max * 1.05 * np.sin(mid_angle)
    ax.text(label_x, label_y, "Correlation Coefficient", rotation=-45, fontsize=10,
            fontweight="bold", ha="center", va="center", color="black", clip_on=False)

    ax.plot(1.0, 0.0, "*", color="red", markersize=16, label="Observed",
            zorder=10, markeredgecolor="darkred")
    ax.annotate("Observed", xy=(1.0, 0.0), xytext=(1.0, 0.05), fontsize=9,
                ha="center", va="bottom", color="black")

    for name, stats in model_stats.items():
        std_ratio, corr = stats["std_ratio"], stats["corr"]
        theta = np.arccos(corr)
        x, y = std_ratio * np.cos(theta), std_ratio * np.sin(theta)
        if np.sqrt(x ** 2 + y ** 2) > sigma_max:
            continue
        color = colors.get(name, "black")
        marker = markers.get(name, "o")
        ax.scatter(x, y, s=100, color=color, marker=marker, edgecolor="white",
                   linewidth=0.6, label=name, zorder=5)

    ax.set_xlabel("Normalised Standard Deviation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalised Standard Deviation", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_observed_vs_predicted_on_ax(ax: plt.Axes, obs: np.ndarray,
                                      preds_dict: dict[str, np.ndarray],
                                      title: str = "",
                                      error_margin: float = 0.20,
                                      background_color: str = "lightyellow") -> None:
    obs = np.asarray(obs, dtype=float)
    lo = float(np.nanmin(obs))
    hi = float(np.nanmax(obs))
    span = max(hi - lo, 1e-6)
    x_pad = 0.08 * span
    x_min, x_max = lo - x_pad, hi + x_pad

    ax.set_facecolor(background_color)
    ax.plot([x_min, x_max], [x_min, x_max], "k--", lw=1.4, label="1:1 line")
    ax.fill_between(
        [x_min, x_max],
        [x_min * (1 - error_margin), x_max * (1 - error_margin)],
        [x_min * (1 + error_margin), x_max * (1 + error_margin)],
        color="#9ecae1", alpha=0.25, label=f"±{int(error_margin * 100)} % band",
    )
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(preds_dict), 1)))
    for colour, (name, pred) in zip(colours, preds_dict.items()):
        pred = np.asarray(pred, dtype=float)
        ax.scatter(obs, pred, s=22, alpha=0.75, edgecolor="k", linewidth=0.3,
                   color=colour, label=name)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel("Observed DTWL (mbgl)", fontsize=10)
    ax.set_ylabel("Predicted DTWL (mbgl)", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)


def _season_slice_from_result(result: dict[str, Any],
                              bias_corrected: bool = True) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    obs, preds_all = season_predictions_dict(result, bias_corrected=bias_corrected)
    return obs, filter_models(preds_all, MODEL_LIST)


# --------------------------------------------------------------------------- #
# Tab 1 — Data quality
# --------------------------------------------------------------------------- #
def render_data_quality(payload: dict[str, Any]) -> None:
    data = payload["clean_data"]
    st.subheader("Data quality and workbook check")
    section_intro(
        title="What this page does — Data quality & workbook check",
        what=(
            "Every time you upload the Excel workbook, this page reads each recognised "
            "season sheet (Monsoon, Pre_Monsoon, Post_Monsoon, Non_Monsoon), checks that "
            "the required columns exist, and records every correction or rejection the "
            "pipeline had to make — missing values filled, duplicate rows removed, "
            "out-of-range readings dropped, and so on."
        ),
        why=(
            "Machine-learning models are only as good as the data they see. Before any "
            "model runs, this page tells you exactly how many records survived, which "
            "villages are covered, what year range you are working with, and whether any "
            "season had to be 'rescued' because it was too small for a clean split."
        ),
    )
    st.markdown(
        '<div class="stage-note">The uploader checks each recognised season sheet, '
        'keeps the notebook’s validation rules, and records every correction.</div>',
        unsafe_allow_html=True,
    )
    first, second, third, fourth = st.columns(4)
    first.metric("Uploaded records", f"{len(data):,}")
    second.metric("Villages", f"{data['VILLAGE'].nunique():,}")
    third.metric("Year range", f"{data['YEAR'].min()}–{data['YEAR'].max()}")
    fourth.metric("Recognised seasons", f"{data['Season'].nunique():,}")
    st.markdown("#### Workbook mapping")
    st.dataframe(payload["workbook_report"], use_container_width=True, hide_index=True)
    st.markdown("#### Validation actions")
    st.dataframe(payload["validation_report"], use_container_width=True, hide_index=True)
    if payload["notices"]:
        for notice in payload["notices"]:
            st.info(notice)
    rescued_seasons = [
        season for season, result in (payload.get("results") or {}).items()
        if result.get("low_data_rescue_used")
    ]
    if rescued_seasons:
        st.warning(
            "Small dataset for **" + ", ".join(rescued_seasons) + "**: rows that would "
            "normally be dropped for having an incomplete lag/rolling value were kept and "
            "imputed instead (forward-fill within the village, then median) so there was "
            "enough data for the train/validation/test split. Raw measurements and the "
            "target itself were never altered — only engineered lag/rolling/trend columns."
        )
    with st.expander("Input format expected by this dashboard"):
        st.markdown(
            "Use an `.xlsx` workbook with sheets named **Monsoon**, **Pre_Monsoon**, "
            "**Post_Monsoon**, and/or **Non_Monsoon**. Each analysis-ready sheet needs "
            "`VILLAGE`, `YEAR`, `DTWL (mbgl)`, `Tmax`, `Tmin`, `SM_10cm`, `SM_40cm`, "
            "`SM_100cm`, `NDVI`, `NDMI`, `NDBI`, `ET`, and `Rainfall`. Extra numeric "
            "predictors are retained."
        )
    with st.expander("Preview cleaned records"):
        st.dataframe(data.head(50), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 2 — Features
# --------------------------------------------------------------------------- #
def render_features(results: dict[str, dict[str, Any]]) -> None:
    st.subheader("Feature engineering and lag preparation")
    section_intro(
        title="What this page does — Feature engineering & lag preparation",
        what=(
            "The raw workbook columns are transformed into the derived predictors the "
            "notebook uses. That includes hydrology features (Water_Balance = Rainfall "
            "− ET, GW_Stress = ET ÷ (Rainfall + 1)), vegetation features "
            "(Veg_Moisture = NDVI × NDMI), soil-moisture aggregates, cyclic season "
            "encoding, and lag/rolling/trend versions of every predictor so the model "
            "can see how this year depends on previous years."
        ),
        why=(
            "Groundwater responds to rainfall, evapotranspiration, and vegetation with "
            "a delay. Feeding the model lagged and rolling values lets it capture that "
            "memory, which is exactly what 'forecasting' needs to be believable rather "
            "than just fitting the noise."
        ),
    )
    st.markdown(
        '<div class="stage-note">This stage reproduces the notebook’s derived hydrology, '
        'vegetation, seasonality, lag, rolling-average, and trend variables.</div>',
        unsafe_allow_html=True,
    )
    result = selected_result(results, "features")
    columns = st.columns(4)
    columns[0].metric("Input records", result["raw_rows"])
    columns[1].metric("Modelling records", result["lagged_rows"])
    columns[2].metric("Removed for incomplete lags", result["rows_removed"])
    columns[3].metric("Model candidate fields", len(result["all_feature_names"]))

    definitions = pd.DataFrame([
        ("Temp_Range", "Tmax − Tmin"),
        ("Avg_SM", "Mean of 10 cm, 40 cm, and 100 cm soil moisture"),
        ("SM_Gradient", "100 cm soil moisture − 10 cm soil moisture"),
        ("Veg_Moisture", "NDVI × NDMI"),
        ("GW_Stress", "ET ÷ (Rainfall + 1)"),
        ("Water_Balance", "Rainfall − ET"),
        ("Veg_Health", "NDVI × (1 − NDBI)"),
        ("Season_Sin / Season_Cos", "Cyclic representation of the four seasons"),
    ], columns=["Derived feature", "Definition"])
    first, second = st.columns([1, 1.25])
    with first:
        st.markdown("#### Derived variables")
        st.dataframe(definitions, use_container_width=True, hide_index=True)
    with second:
        st.markdown("#### Lag-feature inventory")
        inventory = pd.DataFrame({
            "Feature group": ["Lag", "Rolling", "Trend"],
            "Count": [
                len(result["feature_groups"]["lag"]),
                len(result["feature_groups"]["rolling"]),
                len(result["feature_groups"]["trend"]),
            ],
            "Examples": [
                ", ".join(result["feature_groups"]["lag"][:4]),
                ", ".join(result["feature_groups"]["rolling"][:4]),
                ", ".join(result["feature_groups"]["trend"][:3]),
            ],
        })
        st.dataframe(inventory, use_container_width=True, hide_index=True)
    st.markdown("#### Lagged dataset preview")
    preview_columns = [
        column for column in
        ["VILLAGE", "YEAR", "Season", "DTWL (mbgl)"] + result["feature_groups"]["all"][:6]
        if column in result["lagged_data"].columns
    ]
    st.dataframe(result["lagged_data"][preview_columns].head(20),
                 use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 3 — Feature selection
# --------------------------------------------------------------------------- #
def render_selection(results: dict[str, dict[str, Any]]) -> None:
    st.subheader("Feature selection")
    section_intro(
        title="What this page does — Feature selection",
        what=(
            "From the whole pool of engineered features, the app picks the most "
            "informative ones for each season — using mRMR (minimum redundancy, maximum "
            "relevance) when available, otherwise mutual information. Only training-period "
            "rows are used, so the choice never peeks at the test years."
        ),
        why=(
            "Dozens of highly correlated predictors make a model unstable and easy to "
            "over-fit. Keeping only the strongest, least redundant features makes the "
            "model simpler, faster, and more honest about which drivers really matter."
        ),
    )
    st.markdown(
        '<div class="stage-note">The app selects the requested number of predictors '
        'from the training period only. It uses mRMR if available; otherwise falls '
        'back to mutual information.</div>',
        unsafe_allow_html=True,
    )
    result = selected_result(results, "selection")
    ranking = result["feature_ranking"].copy()
    st.caption(f"Selected features for {result['season']}: "
               f"{', '.join(result['selected_features'])}")
    chart_data = ranking.sort_values("Mutual information", ascending=True)
    figure = px.bar(
        chart_data, x="Mutual information", y="Feature", color="Selected",
        orientation="h",
        color_discrete_map={True: "#0071E3", False: "#c7c7cc"},
        labels={"Selected": "Included in model"},
        title=f"Training-period feature relevance — {result['season']}",
    )
    figure.update_layout(height=max(360, 28 * len(chart_data) + 120),
                         legend_title_text="Selected",
                         paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True)
    display_columns = ["Selection rank", "Feature", "Mutual information",
                       "Absolute correlation", "Selection method", "Selected"]
    st.dataframe(ranking[display_columns], use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 4 — Model training
# --------------------------------------------------------------------------- #
def render_models(results: dict[str, dict[str, Any]]) -> None:
    st.subheader("Model training and evaluation")
    section_intro(
        title="What this page does — Model training & evaluation",
        what=(
            "Each season's data is split into train → validation → test by year. Several "
            "models (Random Forest, Extra Trees, gradient-boosted trees, CatBoost, "
            "XGBoost, Cubist, M5, AdaBoost …) are trained on the training years, "
            "checked on the validation years, and finally scored on the held-out test "
            "years. R², RMSE, MAE, Pearson r, KGE, and NSE are reported both raw and "
            "after the notebook's bias-correction."
        ),
        why=(
            "Comparing many models on the same held-out years tells you which algorithm "
            "is genuinely best for this season — not just best at memorising the "
            "training set. The bias-corrected column shows what performance looks like "
            "once the model's systematic offset is removed."
        ),
    )
    st.markdown(
        '<div class="stage-note">All models are trained on earlier years, checked on '
        'the validation period, and finally evaluated on held-out years. Bias-corrected '
        'test metrics use the notebook’s training-mean adjustment.</div>',
        unsafe_allow_html=True,
    )
    result = selected_result(results, "models")
    metric_table = result["metrics"].copy()
    top = metric_table.iloc[0]
    cards = st.columns(4)
    cards[0].metric("Best model", str(top["Model"]))
    cards[1].metric("Bias-corrected test R²", metric_text(top["Bias-corrected test R2"]))
    cards[2].metric("Bias-corrected test RMSE", metric_text(top["Bias-corrected test RMSE"]))
    cards[3].metric("Bias-corrected test MAE", metric_text(top["Bias-corrected test MAE"]))
    figure = px.bar(
        metric_table.sort_values("Bias-corrected test R2"),
        x="Bias-corrected test R2", y="Model", orientation="h",
        color="Bias-corrected test RMSE", color_continuous_scale="Blues",
        title=f"Held-out performance by model — {result['season']}",
        labels={"Bias-corrected test R2": "Bias-corrected test R²"},
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#6E6E73")
    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True)
    st.markdown("#### Performance table")
    main_columns = [
        "Model", "Train R2", "Validation R2", "Test R2", "Bias-corrected test R2",
        "Test RMSE", "Bias-corrected test RMSE", "Test MAE",
        "Bias-corrected test MAE", "Bias-corrected test Pearson r",
        "Bias-corrected test KGE", "Bias-corrected test NSE", "Test inference (ms/row)",
    ]
    st.dataframe(metric_table[[c for c in main_columns if c in metric_table]],
                 use_container_width=True, hide_index=True)
    with st.expander("Model availability"):
        st.dataframe(result["availability"], use_container_width=True, hide_index=True)
        st.caption("CatBoost, XGBoost, LightGBM, Cubist, and M5 are included only when "
                   "their compatible packages are installed. The core scikit-learn "
                   "models always run.")
    with st.expander("Model parameters used in this run"):
        if result["parameters"].empty:
            st.info("No model exposes parameters in a dashboard-readable format.")
        else:
            st.dataframe(result["parameters"], use_container_width=True, hide_index=True)
    if result["model_errors"]:
        with st.expander("Model errors that were safely skipped"):
            for error in result["model_errors"]:
                st.warning(error)


# --------------------------------------------------------------------------- #
# Tab 5 — Diagnostics
# --------------------------------------------------------------------------- #
def render_diagnostics(results: dict[str, dict[str, Any]]) -> None:
    st.subheader("Prediction diagnostics")
    section_intro(
        title="What this page does — Prediction diagnostics",
        what=(
            "For one model in one season, this page puts observed vs. predicted DTWL "
            "side-by-side as a scatter plot with a 1:1 reference line, then shows the "
            "distribution of residuals (prediction − observation) and finally the full "
            "held-out time series per village."
        ),
        why=(
            "A single R² number hides where a model is wrong. Looking at the scatter, "
            "residual histogram, and timeline shows whether the model is biased high in "
            "dry years, missing peaks in wet years, or drifting over time — the things "
            "that actually matter for groundwater management."
        ),
    )
    st.markdown(
        '<div class="stage-note">Compare each model’s held-out predictions with '
        'observed DTWL, then inspect error distribution and the chronological test '
        'period.</div>',
        unsafe_allow_html=True,
    )
    result = selected_result(results, "diagnostics")
    predictions = result["predictions"].copy().sort_values(["Year", "Village"])
    model_choices = [
        column.removesuffix(" prediction") for column in predictions.columns
        if column.endswith(" prediction") and not column.endswith(" bias-corrected prediction")
    ]
    default_index = model_choices.index(result["best_model"]) if result["best_model"] in model_choices else 0
    chosen_model = st.selectbox("Model for diagnostic plots", model_choices,
                                index=default_index,
                                key=f"diagnostic_model_{result['season']}")
    estimate_column = f"{chosen_model} bias-corrected prediction"
    if estimate_column not in predictions:
        estimate_column = f"{chosen_model} prediction"
    observed = predictions["Actual DTWL (mbgl)"].to_numpy(dtype=float)
    predicted = predictions[estimate_column].to_numpy(dtype=float)
    low = float(np.nanmin(np.r_[observed, predicted]))
    high = float(np.nanmax(np.r_[observed, predicted]))
    padding = max((high - low) * 0.08, 0.5)

    left, right = st.columns(2)
    with left:
        scatter = go.Figure()
        scatter.add_trace(go.Scatter(
            x=observed, y=predicted, mode="markers", name="Held-out records",
            text=predictions["Year"],
            hovertemplate="Year %{text}<br>Observed %{x:.2f}<br>Predicted %{y:.2f}<extra></extra>",
        ))
        scatter.add_trace(go.Scatter(
            x=[low - padding, high + padding], y=[low - padding, high + padding],
            mode="lines", line={"dash": "dash", "color": "#6E6E73"},
            name="1:1 reference",
        ))
        scatter.update_layout(title="Observed vs predicted DTWL",
                              xaxis_title="Observed DTWL (mbgl)",
                              yaxis_title="Predicted DTWL (mbgl)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
        scatter.update_xaxes(range=[low - padding, high + padding])
        scatter.update_yaxes(range=[low - padding, high + padding],
                             scaleanchor="x", scaleratio=1)
        st.plotly_chart(scatter, use_container_width=True)
    with right:
        residuals = predicted - observed
        residual_figure = px.histogram(
            x=residuals, nbins=min(16, max(6, len(residuals) // 2)),
            marginal="box", title="Residual distribution",
            labels={"x": "Prediction − observation (mbgl)"},
        )
        residual_figure.add_vline(x=0, line_dash="dash", line_color="#6E6E73")
        residual_figure.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(residual_figure, use_container_width=True)

    line_data = predictions[["Year", "Village", "Actual DTWL (mbgl)", estimate_column]].melt(
        id_vars=["Year", "Village"], var_name="Series", value_name="DTWL (mbgl)"
    )
    line_data["Series"] = line_data["Series"].replace(
        {"Actual DTWL (mbgl)": "Observed", estimate_column: chosen_model}
    )
    timeline = px.line(
        line_data, x="Year", y="DTWL (mbgl)", color="Series",
        line_dash="Village", markers=True,
        title=f"Held-out time series — {result['season']}",
    )
    timeline.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(timeline, use_container_width=True)
    st.dataframe(predictions, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 6 — Sensitivity
# --------------------------------------------------------------------------- #
def render_sensitivity(results: dict[str, dict[str, Any]]) -> None:
    st.subheader("Leave-one-feature-out sensitivity")
    section_intro(
        title="What this page does — Leave-one-feature-out (LOFO) sensitivity",
        what=(
            "For the best model in the selected season, every selected feature is "
            "removed one at a time and the model is retrained. The chart reports how "
            "much the test-period correlation drops when each feature is taken away."
        ),
        why=(
            "This is a robustness check: it tells you which features are load-bearing "
            "and which the model could happily live without. A large positive drop "
            "means the feature genuinely supported the temporal pattern; a small or "
            "negative drop means it was noise."
        ),
    )
    st.markdown(
        '<div class="stage-note">For the best raw-test-R² model, each selected feature '
        'is removed in turn and the model is refit. A positive correlation drop means '
        'that feature supported the model’s temporal pattern.</div>',
        unsafe_allow_html=True,
    )
    result = selected_result(results, "sensitivity")
    if result["lofo"].empty:
        st.info("Feature sensitivity was turned off for this run. Enable it in the "
                "sidebar and run the analysis again.")
        return
    lofo = result["lofo"].copy().sort_values("Correlation drop")
    figure = px.bar(
        lofo, x="Correlation drop", y="Feature", orientation="h",
        title=f"Feature sensitivity — {result['season']}",
        color="Correlation drop", color_continuous_scale="RdBu_r",
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#6E6E73")
    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(lofo.sort_values("Correlation drop", ascending=False),
                 use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 8 — Cross-season analysis
# --------------------------------------------------------------------------- #
def render_multiseason_analysis(results: dict[str, dict[str, Any]]) -> None:
    st.subheader("Cross-season model analysis")
    section_intro(
        title="What this page does — Cross-season comparison & Taylor diagrams",
        what=(
            "This tab stacks every season side by side: a heatmap of metric × model × "
            "season, a bar chart of the train–test R² gap (an overfitting indicator), "
            "residual violins, and two 2×2 grids of journal-style matplotlib figures — "
            "Taylor diagrams (one per season) and observed-vs-predicted scatters."
        ),
        why=(
            "Comparing seasons in one view is how you spot a model that generalises "
            "well across the year versus one that only works for the monsoon. Taylor "
            "diagrams compress three things into one picture: correlation, variability "
            "ratio, and centred RMSD."
        ),
    )
    st.markdown(
        '<div class="stage-note">This consolidates the notebook’s season-wise metric '
        'tables, overfitting check, residual violin plots, and Taylor diagrams into '
        'interactive analysis views.</div>',
        unsafe_allow_html=True,
    )
    long_metrics = season_model_metrics(results)
    metric_options = {
        "Bias-corrected test R²": "Bias-corrected test R2",
        "Bias-corrected test RMSE": "Bias-corrected test RMSE",
        "Bias-corrected test NSE": "Bias-corrected test NSE",
        "Bias-corrected test KGE": "Bias-corrected test KGE",
        "Bias-corrected test Pearson r": "Bias-corrected test Pearson r",
    }
    metric_label = st.selectbox("Metric for season comparison", list(metric_options),
                                key="season_metric")
    metric_column = metric_options[metric_label]
    matrix = long_metrics.pivot(index="Model", columns="Season", values=metric_column)\
                         .reindex(columns=[s for s in SEASON_ORDER if s in results])
    if "RMSE" in metric_label:
        matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=True).index]
        scale = "RdYlGn_r"
    else:
        matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]
        scale = "RdYlGn"
    heatmap = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=matrix.columns, y=matrix.index, colorscale=scale,
        colorbar={"title": metric_label}, text=np.round(matrix.to_numpy(), 3),
        texttemplate="%{text}",
        hovertemplate="Model: %{y}<br>Season: %{x}<br>Value: %{z:.4f}<extra></extra>",
    ))
    heatmap.update_layout(title=f"{metric_label} by model and season",
                          height=max(380, 32 * len(matrix) + 130),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(heatmap, use_container_width=True)
    st.dataframe(long_metrics, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        gaps = long_metrics.sort_values("Train–test R2 gap", ascending=False)
        gap_chart = px.bar(
            gaps, x="Train–test R2 gap", y="Model", color="Season", orientation="h",
            title="Train–test R² gap (overfitting indicator)",
        )
        gap_chart.add_vline(x=0.1, line_dash="dash", line_color="#FF3B30",
                            annotation_text="0.10 reference")
        gap_chart.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gap_chart, use_container_width=True)
    with right:
        residuals = residual_long_table(results, bias_corrected=True)
        violin = px.violin(
            residuals, x="Model", y="Residual (prediction − observation)",
            color="Season", box=True, points="all",
            title="Bias-corrected held-out residuals",
            labels={"Residual (prediction − observation)": "Residual (mbgl)"},
        )
        violin.add_hline(y=0, line_dash="dash", line_color="#6E6E73")
        violin.update_xaxes(tickangle=-35)
        violin.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(violin, use_container_width=True)

    st.markdown("#### Taylor diagrams — all seasons (2×2, notebook layout)")
    st.markdown(
        '<div class="notebook-caption">Reference point = observed series. '
        'Angle = arccos(correlation); radial distance = σ(model)/σ(observed). '
        'Dashed arcs are centred RMSD contours. Six headline models only.</div>',
        unsafe_allow_html=True,
    )

    seasons_available = [s for s in SEASON_ORDER if s in results]
    if seasons_available:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        all_handles, all_labels = [], []
        for idx, season in enumerate(SEASON_ORDER):
            ax = axes[idx]
            if season not in results:
                ax.text(0.5, 0.5, f"No data for {season}",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_title(season, fontsize=12, fontweight="bold")
                continue
            obs, preds_filtered = _season_slice_from_result(results[season])
            if not preds_filtered:
                ax.text(0.5, 0.5, f"No selected models for {season}",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_title(season, fontsize=12, fontweight="bold")
                continue
            _plot_taylor_latex_style(ax, obs=obs, preds_dict=preds_filtered,
                                     title=season, sigma_max=None,
                                     extra_rmsd=1.6, extra_corr=0.1)
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in all_labels:
                    all_handles.append(hi)
                    all_labels.append(li)
        for j in range(len(SEASON_ORDER), len(axes)):
            axes[j].set_visible(False)
        fig.legend(all_handles, all_labels,
                   loc="center left", bbox_to_anchor=(0.89, 0.5),
                   frameon=True, edgecolor="black", fancybox=False,
                   fontsize=10, title="Model", title_fontsize=11)
        plt.tight_layout(rect=[0, 0, 0.94, 1])
        _matplotlib_to_streamlit(fig)

    st.markdown("#### Observed vs predicted — all seasons (2×2, notebook layout)")
    st.markdown(
        '<div class="notebook-caption">Each panel shows the six headline models '
        'against the observed values on the held-out test years. The pale band is '
        'the ±20 % envelope around the 1:1 line.</div>',
        unsafe_allow_html=True,
    )

    season_data: dict[str, dict[str, Any]] = {}
    for season in SEASON_ORDER:
        if season not in results:
            continue
        obs, preds_filtered = _season_slice_from_result(results[season])
        if not preds_filtered:
            continue
        season_data[season] = {"obs": obs, "preds": preds_filtered}
    if season_data:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        axes = axes.flatten()
        for idx, season in enumerate(SEASON_ORDER):
            ax = axes[idx]
            if season not in season_data:
                ax.text(0.5, 0.5, f"No data for {season}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=12)
                ax.set_title(season, fontsize=12, fontweight="bold")
                continue
            data = season_data[season]
            _plot_observed_vs_predicted_on_ax(
                ax, data["obs"], data["preds"],
                title=season, error_margin=0.20,
                background_color="lightyellow",
            )
            ax.legend(loc="lower right", fontsize=8, frameon=True)
        for j in range(len(SEASON_ORDER), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        _matplotlib_to_streamlit(fig)
    else:
        st.info("No held-out predictions available for the scatter grid.")


# --------------------------------------------------------------------------- #
# Tab 9 — Time series
# --------------------------------------------------------------------------- #
def render_timeseries_analysis(payload: dict[str, Any]) -> None:
    results = payload["results"]
    config = payload["config"]
    st.subheader("Time series, rainfall, and autocorrelation")
    section_intro(
        title="What this page does — Time series, rainfall, ACF/PACF and ADF",
        what=(
            "This page shows the full predicted timeline of DTWL alongside annual "
            "rainfall on a twin axis, the autocorrelation (ACF) and partial "
            "autocorrelation (PACF) of the selected series, cross-correlation between "
            "rainfall and DTWL at different lags, and an Augmented Dickey–Fuller (ADF) "
            "stationarity test."
        ),
        why=(
            "Groundwater is a slow, memory-heavy system. The ACF/PACF tell you how many "
            "years of memory the model needs, the cross-correlation shows whether "
            "rainfall leads DTWL by one year or several, and the ADF test tells you "
            "whether the series is stationary — a key assumption behind ARIMA-style "
            "models like the SARIMA in the deep-learning family."
        ),
    )
    st.markdown(
        '<div class="stage-note">This view adds the notebook’s rainfall–DTWL '
        'twin-axis plots, full season prediction timeline, ACF/PACF diagnostics, '
        'and rainfall cross-correlation.</div>',
        unsafe_allow_html=True,
    )
    result = selected_result(results, "timeseries")
    timeline = result["full_timeline"].copy()
    raw_models = [
        column.removesuffix(" prediction") for column in timeline.columns
        if column.endswith(" prediction") and not column.endswith(" bias-corrected prediction")
    ]
    default_index = raw_models.index(result["best_model"]) if result["best_model"] in raw_models else 0
    chosen_model = st.selectbox("Model for complete timeline", raw_models,
                                index=default_index,
                                key=f"timeline_model_{result['season']}")
    prediction_column = f"{chosen_model} bias-corrected prediction"
    if prediction_column not in timeline:
        prediction_column = f"{chosen_model} prediction"
    annual = timeline.groupby("Year", as_index=False).mean(numeric_only=True)\
                     .sort_values("Year")

    twin = make_subplots(specs=[[{"secondary_y": True}]])
    twin.add_trace(go.Bar(x=annual["Year"], y=annual["Rainfall"], name="Rainfall",
                          marker_color="#74a9cf", opacity=0.65), secondary_y=True)
    twin.add_trace(go.Scatter(x=annual["Year"], y=annual["Observed DTWL (mbgl)"],
                              mode="lines+markers", name="Observed DTWL",
                              line={"color": "#1D1D1F", "width": 3}),
                   secondary_y=False)
    twin.add_trace(go.Scatter(x=annual["Year"], y=annual[prediction_column],
                              mode="lines+markers", name=f"{chosen_model} prediction",
                              line={"dash": "dash", "width": 2}), secondary_y=False)
    twin.add_vline(x=config.train_end, line_dash="dot", line_color="#34C759",
                   annotation_text="Train ends")
    twin.add_vline(x=config.validation_end, line_dash="dot", line_color="#FF9F0A",
                   annotation_text="Validation ends")
    twin.update_layout(title=f"Rainfall and DTWL timeline — {result['season']}",
                       barmode="overlay",
                       legend={"orientation": "h", "y": 1.14},
                       paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    twin.update_xaxes(title_text="Year")
    twin.update_yaxes(title_text="DTWL (mbgl)", secondary_y=False)
    twin.update_yaxes(title_text="Rainfall", secondary_y=True)
    st.plotly_chart(twin, use_container_width=True)

    residuals = annual[prediction_column] - annual["Observed DTWL (mbgl)"]
    series_options = {
        "Observed DTWL": annual["Observed DTWL (mbgl)"],
        f"{chosen_model} residual": residuals,
    }
    series_name = st.selectbox("Series for autocorrelation", list(series_options),
                               key=f"acf_series_{result['season']}")
    acf_table = acf_pacf_table(series_options[series_name],
                               max_lag=min(10, max(2, len(annual) - 2)))
    acf_plot = go.Figure()
    acf_plot.add_trace(go.Bar(x=acf_table["Lag"], y=acf_table["ACF"], name="ACF"))
    acf_plot.add_trace(go.Scatter(x=acf_table["Lag"], y=acf_table["PACF"],
                                  mode="lines+markers", name="PACF",
                                  line={"color": "#FF3B30"}))
    acf_plot.add_hline(y=0, line_color="#6E6E73")
    acf_plot.update_layout(title=f"ACF and PACF — {series_name}",
                           xaxis_title="Lag (years)",
                           yaxis_title="Correlation / partial correlation",
                           paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)")
    correlation = rainfall_cross_correlation(
        annual["Rainfall"], annual["Observed DTWL (mbgl)"],
        max_lag=min(8, max(2, len(annual) - 2)),
    )
    cross_plot = px.bar(correlation,
                        x="Lag (years; + means rainfall leads DTWL)",
                        y="Correlation",
                        title="Rainfall–DTWL cross-correlation")
    cross_plot.add_hline(y=0, line_color="#6E6E73")
    cross_plot.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                             plot_bgcolor="rgba(0,0,0,0)")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(acf_plot, use_container_width=True)
    with right:
        st.plotly_chart(cross_plot, use_container_width=True)
    stationarity = stationarity_summary(series_options[series_name])
    st.markdown("#### Augmented Dickey–Fuller stationarity check")
    if stationarity.get("Status") == "Completed":
        adf_cards = st.columns(4)
        adf_cards[0].metric("ADF statistic", metric_text(stationarity["ADF statistic"]))
        adf_cards[1].metric("p-value", metric_text(stationarity["p-value"]))
        adf_cards[2].metric("Used lags", str(stationarity["Used lags"]))
        adf_cards[3].metric("5% critical value", metric_text(stationarity["5% critical value"]))
        st.caption(str(stationarity["Interpretation"]))
    else:
        st.info(str(stationarity.get("Reason", "The stationarity check could not be run.")))
    with st.expander("Full model timeline data"):
        st.dataframe(timeline, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 10 — 3D models
# --------------------------------------------------------------------------- #
def render_3d_analysis(payload: dict[str, Any]) -> None:
    results = payload["results"]
    st.subheader("Interactive 3D groundwater models")
    section_intro(
        title="What this page does — 3D groundwater & predictor explorer",
        what=(
            "The first 3D figure is a groundwater surface: year × season × DTWL, built "
            "from the seasonal averages. The second figure lets you pick any three "
            "numeric columns from the uploaded data (e.g. YEAR, Rainfall, DTWL) and "
            "explore their relationship as an interactive 3D scatter coloured by season."
        ),
        why=(
            "A 2D chart hides interactions. The 3D surface makes seasonal depletion and "
            "recharge patterns pop out at a glance, and the predictor explorer is a "
            "quick way to check for physical plausibility — for example that higher "
            "rainfall really does sit above shallower water tables."
        ),
    )
    st.markdown(
        '<div class="stage-note">The first 3D chart recreates the notebook’s '
        'groundwater surface: year × season × DTWL. The second lets you explore a 3D '
        'relationship between uploaded variables.</div>',
        unsafe_allow_html=True,
    )
    all_timelines = pd.concat([r["full_timeline"] for r in results.values()], ignore_index=True)
    prediction_columns = sorted(set.intersection(*[
        {c for c in r["full_timeline"].columns if c.endswith(" bias-corrected prediction")}
        for r in results.values()
    ])) if results else []
    measure_options = {"Observed DTWL (mbgl)": "Observed DTWL (mbgl)"}
    measure_options.update({c.removesuffix(" bias-corrected prediction"): c
                            for c in prediction_columns})
    measure_label = st.selectbox("Surface measure", list(measure_options),
                                 key="surface_measure")
    measure = measure_options[measure_label]
    surface_points = all_timelines.groupby(["Year", "Season"], as_index=False)[measure].mean()
    seasons = [s for s in SEASON_ORDER if s in surface_points["Season"].unique()]
    years = sorted(surface_points["Year"].unique())
    surface_matrix = surface_points.pivot(index="Season", columns="Year", values=measure)\
                                    .reindex(index=seasons, columns=years)
    season_positions = list(range(1, len(seasons) + 1))
    surface = go.Figure()
    surface.add_trace(go.Surface(z=surface_matrix.to_numpy(), x=years,
                                 y=season_positions, colorscale="Viridis",
                                 opacity=0.88, colorbar={"title": measure_label}))
    scatter_y = surface_points["Season"].map({s: i + 1 for i, s in enumerate(seasons)})
    surface.add_trace(go.Scatter3d(x=surface_points["Year"], y=scatter_y,
                                   z=surface_points[measure], mode="markers",
                                   marker={"size": 4, "color": "#1D1D1F"},
                                   name="Seasonal average observations"))
    surface.update_layout(
        title=f"3D groundwater surface — {measure_label}",
        scene={
            "xaxis": {"title": "Year"},
            "yaxis": {"title": "Season", "tickvals": season_positions, "ticktext": seasons},
            "zaxis": {"title": measure_label},
        },
        height=680, margin={"l": 0, "r": 0, "b": 0, "t": 50},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(surface, use_container_width=True)
    st.caption("The surface interpolates the annual seasonal averages; black markers "
               "retain the original aggregated positions.")

    st.markdown("#### 3D predictor relationship explorer")
    source = payload["clean_data"].copy()
    season_filter = st.multiselect("Seasons shown in predictor explorer",
                                   usable_seasons(source),
                                   default=usable_seasons(source),
                                   key="three_d_seasons")
    source = source.loc[source["Season"].isin(season_filter)]
    numeric_columns = source.select_dtypes(include=np.number).columns.tolist()
    x_axis = st.selectbox("X axis", numeric_columns,
                          index=numeric_columns.index("YEAR") if "YEAR" in numeric_columns else 0,
                          key="three_d_x")
    y_axis = st.selectbox("Y axis", numeric_columns,
                          index=numeric_columns.index("Rainfall") if "Rainfall" in numeric_columns
                          else min(1, len(numeric_columns) - 1),
                          key="three_d_y")
    z_axis = st.selectbox("Z axis", numeric_columns,
                          index=numeric_columns.index("DTWL (mbgl)") if "DTWL (mbgl)" in numeric_columns
                          else min(2, len(numeric_columns) - 1),
                          key="three_d_z")
    explorer = px.scatter_3d(source, x=x_axis, y=y_axis, z=z_axis,
                             color="Season", hover_name="VILLAGE",
                             title="3D uploaded-data relationship explorer", opacity=0.8)
    explorer.update_traces(marker={"size": 4})
    explorer.update_layout(height=680, margin={"l": 0, "r": 0, "b": 0, "t": 50},
                           paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(explorer, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 11 — Wavelet (CWT)
# --------------------------------------------------------------------------- #
def _plot_scalogram(cwt: dict[str, Any], title: str) -> plt.Figure:
    t = cwt["t"]
    period = cwt["period"]
    power_norm = cwt["power_norm"]
    coi = cwt["coi"]

    fig, ax = plt.subplots(figsize=(14, 6))
    log_power = np.log2(np.clip(power_norm, 1e-12, None))
    levels = np.linspace(-6, 6, 100)
    cf = ax.contourf(t, np.log2(period), log_power, levels=levels,
                     extend="both", cmap="jet")

    ax.fill_between(t, np.log2(coi), np.log2(period.max()),
                    where=np.log2(coi) < np.log2(period.max()),
                    color="white", alpha=0.45, hatch="//",
                    edgecolor="lightgray", linewidth=0)
    ax.plot(t, np.log2(coi), color="k", linewidth=0.8)

    try:
        ax.contour(t, np.log2(period), cwt["sig95"], [-99, 1],
                   colors="k", linewidths=0.9)
    except Exception:
        pass

    ax.set_ylim(np.log2(period.max()), np.log2(period.min()))
    yticks = [p for p in [0.5, 1, 2, 4, 8, 16, 32]
              if period.min() <= p <= period.max()]
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels([f"{v:g}" for v in yticks])
    ax.set_xlim(t.min(), t.max())
    year_ticks = np.arange(np.floor(t.min()), np.ceil(t.max()) + 1, 2)
    ax.set_xticks(year_ticks)
    ax.set_xticklabels([str(int(y)) for y in year_ticks], rotation=45, ha="right")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Period (years)", fontsize=12)
    ax.set_title(title, fontsize=14)

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar_ticks = np.arange(-6, 7)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"1/{2**-i}" if i < 0 else f"{2**i}" for i in cbar_ticks])
    cbar.set_label("Normalized power")
    plt.tight_layout()
    return fig


def _plot_global_spectrum(cwt: dict[str, Any], title: str) -> plt.Figure:
    period = cwt["period"]
    global_ws = cwt["global_ws"]
    signif = cwt.get("global_signif")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(global_ws, period, "b-", linewidth=1.5, label="Global spectrum")
    if signif is not None:
        ax.plot(signif, period, "r--", linewidth=1.2,
                label="95 % significance (red noise)")
    ax.set_xlabel("Power", fontsize=11)
    ax.set_ylabel("Period (years)", fontsize=11)
    ax.set_yscale("log", base=2)
    yticks = [p for p in [0.5, 1, 2, 4, 8, 16, 32]
              if period.min() <= p <= period.max()]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{v:g}" for v in yticks])
    ax.grid(alpha=0.35, which="both")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    return fig


def render_wavelet_analysis(payload: dict[str, Any]) -> None:
    st.subheader("Wavelet (CWT) analysis of aggregated DTWL")
    section_intro(
        title="What this page does — Continuous wavelet transform (CWT)",
        what=(
            "The seasonal DTWL series (median across all villages) is decomposed into "
            "time × period using a Morlet(6) mother wavelet. The resulting scalogram "
            "shows how much power sits at each cycle length (0.5 y, 1 y, 2 y, 4 y, …) "
            "at each point in time. A global wavelet spectrum condenses that into a "
            "single 'dominant periodicities' table."
        ),
        why=(
            "Groundwater often shows multi-year cycles tied to climate modes like ENSO "
            "(≈2–7 years) or the Indian Ocean Dipole (≈3–7 years). A CWT is the tool "
            "that reveals those cycles and tells you whether they are statistically "
            "significant against a red-noise background, or just noise."
        ),
    )
    st.markdown(
        '<div class="stage-note">Continuous wavelet transform of the seasonal '
        'aggregated DTWL series (median across villages). The scalogram uses the '
        'Morlet(6) mother wavelet, and the black lines mark 95 % significance '
        'against a red-noise background. The shaded hatched region is the cone of '
        'influence.</div>',
        unsafe_allow_html=True,
    )

    data = payload["clean_data"]
    aggregated = build_aggregated_series(data)
    if aggregated.empty or len(aggregated) < 16:
        st.info("At least 16 aggregated quarterly observations are needed for a "
                "meaningful CWT. Upload more years of seasonal data.")
        return

    options = st.columns(4)
    dt_choice = options[0].selectbox("Sampling step dt (years)", [0.25, 0.5, 1.0],
                                     index=0, key="cwt_dt")
    mother_choice = options[1].selectbox("Mother wavelet", ["Morlet", "Paul", "DOG"],
                                         index=0, key="cwt_mother")
    param = options[2].slider("Wavelet parameter", 4.0, 8.0, 6.0, 0.5,
                              key="cwt_param",
                              help="ω₀ for Morlet, order for Paul, order for DOG")
    detrend_flag = options[3].toggle("Linear detrend before CWT", value=True,
                                     key="cwt_detrend")

    try:
        cwt = wavelet_power_spectrum(
            aggregated["t"].to_numpy(),
            aggregated["DTWL (mbgl)"].to_numpy(),
            dt=float(dt_choice),
            mother_name=mother_choice,
            mother_param=float(param),
            detrend=bool(detrend_flag),
        )
    except Exception as exc:
        st.error(f"Wavelet analysis could not be completed: {exc}")
        st.caption("If pycwt is missing, install it with `pip install pycwt`.")
        st.dataframe(aggregated, use_container_width=True, hide_index=True)
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Aggregated points", len(aggregated))
    summary_cols[1].metric("Year span",
                           f"{aggregated['t'].min():.2f} – {aggregated['t'].max():.2f}")
    summary_cols[2].metric("Seasons aggregated",
                           aggregated["Season"].nunique())
    summary_cols[3].metric("Villages pooled",
                           data["VILLAGE"].nunique())

    with st.expander("Aggregated quarterly series (preview)"):
        st.dataframe(aggregated.head(24), use_container_width=True, hide_index=True)

    st.markdown("#### Time × period scalogram (normalised power, log₂)")
    fig = _plot_scalogram(cwt,
                          "CWT power spectrum — aggregated DTWL (median across villages)")
    _matplotlib_to_streamlit(fig)
    st.markdown(
        '<div class="notebook-caption">Vertical axis is log₂(period in years); '
        'contour lines outline the 95 % significance level. The hatched region is '
        'the cone of influence, where edge effects dominate.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Global wavelet spectrum")
        fig2 = _plot_global_spectrum(cwt, "Global wavelet spectrum")
        _matplotlib_to_streamlit(fig2)
    with right:
        st.markdown("#### Dominant periodicities")
        global_ws = cwt["global_ws"]
        period = cwt["period"]
        order = np.argsort(global_ws)[::-1][:8]
        peaks = pd.DataFrame({
            "Rank": np.arange(1, len(order) + 1),
            "Period (years)": period[order],
            "Power": global_ws[order],
        })
        peaks["Period (years)"] = peaks["Period (years)"].round(3)
        peaks["Power"] = peaks["Power"].round(4)
        st.dataframe(peaks, use_container_width=True, hide_index=True)
        st.caption("Higher power = stronger concentration of variance at that "
                   "period. Compare the top period with known climate cycles "
                   "(e.g. ENSO ≈ 2–7 years, IOD ≈ 3–7 years).")

    with st.expander("Aggregated series used for the CWT"):
        st.dataframe(aggregated, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Tab 7 — Forecast & export
# --------------------------------------------------------------------------- #
def render_forecast_and_export(payload: dict[str, Any]) -> None:
    results = payload["results"]
    st.subheader("Scenario forecast and export")
    section_intro(
        title="What this page does — Scenario forecasting",
        what=(
            "A forecast is a projection of what groundwater level (DTWL) is likely to "
            "be in future years. Because the uploaded workbook only contains historical "
            "exogenous drivers (rainfall, ET, NDVI, soil moisture), the app holds those "
            "at their last observed value and lets the model recursively update the "
            "target-history features (lags of DTWL) as it steps forward. The result is "
            "a 'business-as-usual' scenario, not a climate projection."
        ),
        why=(
            "Forecasts are the operational payoff of the whole pipeline. They are what "
            "planners use to decide where to intervene. Two grids are shown here: a "
            "single interactive line for the season you are viewing, and a 2×2 "
            "notebook-style figure covering all four seasons for the next 1–10 years, "
            "with blue arrows marking each forecast year."
        ),
    )
    st.markdown(
        '<div class="stage-note">Future exogenous fields are not available in the '
        'uploaded historical workbook. The forecast therefore holds the latest '
        'non-target predictors constant and recursively updates target-history '
        'variables.</div>',
        unsafe_allow_html=True,
    )
    result = selected_result(results, "forecast")
    forecast = result["forecast"]
    figure = px.line(forecast, x="Year", y="Scenario forecast DTWL (mbgl)",
                     markers=True,
                     title=f"{result['best_model']} scenario forecast — {result['season']}")
    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(forecast, use_container_width=True, hide_index=True)

    st.markdown("#### Notebook layout — next 3 years, all four seasons (2×2)")
    st.markdown(
        '<div class="notebook-caption">Each panel is one season. The historical '
        'annual mean (black) is extended from the last observed year using the six '
        'headline models trained on that season. Blue arrows mark each forecast '
        'year, and the inset zooms on the last decade plus the forecast.</div>',
        unsafe_allow_html=True,
    )

    n_steps = int(payload["config"].forecast_steps)
    seasons = [s for s in SEASON_ORDER if s in results]
    forecast_data: dict[str, dict[str, Any]] = {}

    for season in seasons:
        res = results[season]
        scaled_full = res.get("X_full_scaled")
        df_lag_full = res.get("df_lag_full")
        fitted_models = res.get("fitted_models", {})
        feature_cols = res.get("feature_cols", [])
        if scaled_full is None or df_lag_full is None or not fitted_models:
            continue

        wanted = filter_models(fitted_models, MODEL_LIST)
        if not wanted:
            continue

        last_row = scaled_full.iloc[-1:]
        last_meta = df_lag_full.iloc[-1]

        forecasts_dict: dict[str, np.ndarray] = {}
        for name, model in wanted.items():
            try:
                forecasts_dict[name] = recursive_forecast_constant(
                    model, last_row, feature_cols, n_steps=n_steps
                )
            except Exception as exc:
                st.warning(f"{season} — {name} could not forecast: {exc}")

        if not forecasts_dict:
            continue

        hist_years = df_lag_full["YEAR"].to_numpy()
        hist_target = df_lag_full["DTWL (mbgl)"]
        df_hist = pd.DataFrame({
            "Year": hist_years,
            "DTWL": np.asarray(hist_target).ravel(),
        })
        yearly_avg = df_hist.groupby("Year")["DTWL"].mean().reset_index()
        yearly_frac = yearly_avg["Year"].to_numpy(dtype=float)
        yearly_dtwl = yearly_avg["DTWL"].to_numpy(dtype=float)

        last_year = int(last_meta["YEAR"])
        last_season = str(last_meta["Season"])
        timeline_rows = []
        for step in range(n_steps):
            fyear = last_year + step + 1
            timeline_rows.append({
                "YEAR": fyear, "Season": last_season, "Step": step + 1,
                "Frac": fractional_year(fyear, last_season),
            })
        df_timeline = pd.DataFrame(timeline_rows)

        forecast_data[season] = {
            "yearly_frac": yearly_frac,
            "yearly_dtwl": yearly_dtwl,
            "forecast_frac": df_timeline["Frac"].to_numpy(),
            "forecast_timeline": df_timeline,
            "forecasts_dict": forecasts_dict,
            "n_steps": n_steps,
        }

    if forecast_data:
        fig, axes = plt.subplots(2, 2, figsize=(24, 15))
        axes = axes.flatten()
        all_handles, all_labels = [], []

        for idx, season in enumerate(SEASON_ORDER):
            ax = axes[idx]
            if season not in forecast_data:
                ax.text(0.5, 0.5, f"No forecast for {season}",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=14)
                ax.set_title(season, fontsize=14, fontweight="bold")
                continue

            data = forecast_data[season]
            yearly_frac = data["yearly_frac"]
            yearly_dtwl = data["yearly_dtwl"]
            forecast_frac = data["forecast_frac"]
            df_timeline = data["forecast_timeline"]
            forecasts_dict = data["forecasts_dict"]

            ax.plot(yearly_frac, yearly_dtwl, "k-", lw=2.5, alpha=0.85,
                    label="Historical (annual mean)")
            last_obs_x, last_obs_y = yearly_frac[-1], yearly_dtwl[-1]
            ax.plot(last_obs_x, last_obs_y, "ko", markersize=10,
                    label="Last observed year")

            colours = plt.cm.tab10(np.linspace(0, 1, len(forecasts_dict)))
            for colour, (name, preds) in zip(colours, forecasts_dict.items()):
                cont_x = np.concatenate([[last_obs_x], forecast_frac])
                cont_y = np.concatenate([[last_obs_y], preds])
                ax.plot(cont_x, cont_y, "--o", color=colour, lw=2.5,
                        markersize=9, label=name)

            sample_name = next(iter(forecasts_dict))
            for i, row in df_timeline.reset_index(drop=True).iterrows():
                frac = row["Frac"]
                y_val = float(forecasts_dict[sample_name][i])
                label = f"{int(row['YEAR'])}\n{row['Season']}"
                ax.annotate(label, xy=(frac, y_val),
                            xytext=(frac, y_val + 1.5),
                            arrowprops=dict(arrowstyle="->",
                                            color="darkblue", lw=1.4),
                            fontsize=9, fontweight="bold",
                            color="darkblue", ha="center", va="bottom")

            ax.set_xlim(yearly_frac[0] - 0.2, forecast_frac[-1] + 0.5)
            all_y = np.concatenate([yearly_dtwl] +
                                   [np.asarray(p) for p in forecasts_dict.values()])
            ax.set_ylim(all_y.min() - 1.0, all_y.max() + 1.0)
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel("DTWL (mbgl)", fontsize=12)
            ax.set_title(season, fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_locator(MultipleLocator(3))
            ax.tick_params(labelsize=11)

            hist_slice = slice(-10, None) if len(yearly_frac) > 10 else slice(0, None)
            zoom_x_hist = yearly_frac[hist_slice]
            zoom_y_hist = yearly_dtwl[hist_slice]
            all_y_zoom = np.concatenate([zoom_y_hist] +
                                        [np.asarray(p) for p in forecasts_dict.values()])
            axins = inset_axes(ax, width="100%", height="100%",
                               bbox_to_anchor=(0.02, 0.58, 0.40, 0.35),
                               bbox_transform=ax.transAxes, borderpad=0)
            axins.plot(zoom_x_hist, zoom_y_hist, "k-", lw=2)
            for colour, (name, preds) in zip(colours, forecasts_dict.items()):
                axins.plot(forecast_frac, preds, "--o", color=colour,
                           lw=2, markersize=6)
            axins.set_xlim(zoom_x_hist[0] - 0.2, forecast_frac[-1] + 0.3)
            axins.set_ylim(all_y_zoom.min() - 0.5, all_y_zoom.max() + 0.5)
            axins.grid(alpha=0.3)
            axins.tick_params(labelsize=8)
            for spine in axins.spines.values():
                spine.set_linewidth(1.4)
                spine.set_color("black")

            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in all_labels:
                    all_handles.append(hi)
                    all_labels.append(li)

        for j in range(len(SEASON_ORDER), len(axes)):
            axes[j].set_visible(False)

        fig.legend(all_handles, all_labels,
                   loc="center left", bbox_to_anchor=(0.89, 0.5),
                   frameon=True, edgecolor="black", fancybox=False,
                   fontsize=11, title="Model / Event", title_fontsize=12)
        fig.suptitle("Groundwater level forecasts — continuous extension "
                     "from last observed year",
                     fontsize=20, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 0.90, 0.96])
        _matplotlib_to_streamlit(fig)

        st.markdown("##### Per-season forecast tables")
        for season, data in forecast_data.items():
            df_out = data["forecast_timeline"].copy()
            for name, preds in data["forecasts_dict"].items():
                df_out[name] = preds
            with st.expander(f"{season} — next {data['n_steps']} years"):
                st.dataframe(df_out, use_container_width=True, hide_index=True)
                st.download_button(
                    f"Download {season} forecast CSV",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name=f"forecast_{season}.csv",
                    mime="text/csv",
                    key=f"dl_forecast_{season}",
                )

    st.markdown("#### Download the full analysis workbook")
    st.caption("The Excel file includes upload checks, data-quality actions, per-season "
               "feature rankings, parameter tables, model metrics, held-out "
               "predictions, full timelines, residuals, Taylor statistics, feature "
               "sensitivity, and forecasts.")
    export = build_excel_export(payload["workbook_report"],
                                payload["validation_report"], results)
    st.download_button("Download results as Excel", data=export,
                       file_name="groundwater_ml_dashboard_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")


# --------------------------------------------------------------------------- #
# Tab 12 — Deep learning & physics-informed models
# --------------------------------------------------------------------------- #
def render_deep_models(results: dict[str, dict[str, Any]]) -> None:
    st.subheader("Deep learning & physics-informed models")
    section_intro(
        title="What this page does — Deep learning & physics-informed models",
        what=(
            "A completely separate model family trained on the same split and the same "
            "selected features as the six headline models. It contains a "
            "physics-informed neural network (PINN), an echo-state network (reservoir "
            "computing), a classical SARIMAX statistical baseline, and (when PyTorch is "
            "installed) an LSTM."
        ),
        why=(
            "These models bring different inductive biases. The PINN adds a soft "
            "water-balance penalty so it fits physical behaviour, the ESN and LSTM "
            "capture long-range temporal memory of the lag history, and SARIMAX is a "
            "well-understood statistical yardstick. Comparing them against the tree "
            "models shows whether deeper architectures actually add value on this "
            "dataset or just complexity."
        ),
    )
    st.markdown(
        '<div class="stage-note">A second, separate model family trained on the same '
        'split and selected features as the six headline models above: a physics-informed '
        'neural network, a reservoir-computing (echo-state) network, a SARIMA/SARIMAX model, '
        'and — when PyTorch is installed — an LSTM. It lives in its own page so it never '
        'changes the six-model Taylor diagrams or cross-season views elsewhere.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("What each model is doing here"):
        st.markdown(
            "- **PINN (Physics-Informed NN)** — a small neural network whose training loss "
            "adds a soft *water-balance* penalty (`Rainfall − ET`) on top of the usual "
            "prediction error, nudging it toward physically sensible recharge/discharge "
            "behaviour instead of fitting the numbers alone.\n"
            "- **Reservoir Computing (ESN)** — an echo-state network: a large, fixed, random "
            "recurrent reservoir whose only *trained* part is the final linear read-out. It "
            "rides each row's own lag history (`…_lag4 → … → …_lag1 → present`) through the "
            "reservoir before reading out a prediction.\n"
            "- **SARIMA (SARIMAX)** — a classical autoregressive model with exogenous "
            "drivers, included as a statistical-time-series baseline alongside the "
            "machine-learning models above.\n"
            "- **LSTM (Deep Learning)** — a single-layer LSTM over the same lag history as "
            "the reservoir model, trained end-to-end with PyTorch. Shown only when the "
            "optional `torch` package is installed; otherwise it's listed as *Not installed* "
            "below, the same way CatBoost/XGBoost/LightGBM would be."
        )

    result = selected_result(results, "deep_models")
    availability = result.get("deep_availability")
    if availability is not None and not availability.empty:
        with st.expander("Model availability"):
            st.dataframe(availability, use_container_width=True, hide_index=True)
            st.caption("LSTM only appears as *Included* when the optional `torch` package "
                       "is installed. PINN, ESN, and SARIMA always run — they need only "
                       "NumPy and statsmodels, which this app already depends on.")

    metrics = result.get("deep_metrics")
    if metrics is None or metrics.empty:
        st.info("No deep-learning results for this season — either the sidebar toggle was "
                "off for this run, or every model in this family failed to fit (see below).")
        if result.get("deep_model_errors"):
            with st.expander("Model errors that were safely skipped"):
                for error in result["deep_model_errors"]:
                    st.warning(error)
        return

    top = metrics.iloc[0]
    cards = st.columns(4)
    cards[0].metric("Best deep model", str(top["Model"]))
    cards[1].metric("Bias-corrected test R²", metric_text(top["Bias-corrected test R2"]))
    cards[2].metric("Bias-corrected test RMSE", metric_text(top["Bias-corrected test RMSE"]))
    cards[3].metric("Bias-corrected test MAE", metric_text(top["Bias-corrected test MAE"]))

    figure = px.bar(
        metrics.sort_values("Bias-corrected test R2"),
        x="Bias-corrected test R2", y="Model", orientation="h",
        color="Bias-corrected test RMSE", color_continuous_scale="Purples",
        title=f"Deep-learning family — held-out performance — {result['season']}",
        labels={"Bias-corrected test R2": "Bias-corrected test R²"},
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#6E6E73")
    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True)

    st.markdown("#### Performance table")
    main_columns = [
        "Model", "Train R2", "Validation R2", "Test R2", "Bias-corrected test R2",
        "Test RMSE", "Bias-corrected test RMSE", "Test MAE", "Bias-corrected test MAE",
        "Bias-corrected test Pearson r", "Bias-corrected test KGE",
        "Bias-corrected test NSE", "Test inference (ms/row)",
    ]
    st.dataframe(metrics[[c for c in main_columns if c in metrics]],
                 use_container_width=True, hide_index=True)

    if result.get("deep_model_errors"):
        with st.expander("Model errors that were safely skipped"):
            for error in result["deep_model_errors"]:
                st.warning(error)

    predictions = result["deep_predictions"].copy().sort_values(["Year", "Village"])
    model_choices = [
        column.removesuffix(" prediction") for column in predictions.columns
        if column.endswith(" prediction") and not column.endswith(" bias-corrected prediction")
    ]
    if model_choices:
        default_index = model_choices.index(str(top["Model"])) if str(top["Model"]) in model_choices else 0
        chosen_model = st.selectbox("Model for diagnostic plot", model_choices,
                                    index=default_index,
                                    key=f"deep_diagnostic_model_{result['season']}")
        estimate_column = f"{chosen_model} bias-corrected prediction"
        observed = predictions["Actual DTWL (mbgl)"].to_numpy(dtype=float)
        predicted = predictions[estimate_column].to_numpy(dtype=float)
        low = float(np.nanmin(np.r_[observed, predicted]))
        high = float(np.nanmax(np.r_[observed, predicted]))
        padding = max((high - low) * 0.08, 0.5)
        scatter = go.Figure()
        scatter.add_trace(go.Scatter(
            x=observed, y=predicted, mode="markers", name="Held-out records",
            text=predictions["Year"],
            hovertemplate="Year %{text}<br>Observed %{x:.2f}<br>Predicted %{y:.2f}<extra></extra>",
        ))
        scatter.add_trace(go.Scatter(
            x=[low - padding, high + padding], y=[low - padding, high + padding],
            mode="lines", line={"dash": "dash", "color": "#6E6E73"},
            name="1:1 reference",
        ))
        scatter.update_layout(title=f"Observed vs predicted DTWL — {chosen_model}",
                              xaxis_title="Observed DTWL (mbgl)",
                              yaxis_title="Predicted DTWL (mbgl)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)")
        scatter.update_xaxes(range=[low - padding, high + padding])
        scatter.update_yaxes(range=[low - padding, high + padding], scaleanchor="x", scaleratio=1)
        st.plotly_chart(scatter, use_container_width=True)
        st.dataframe(predictions, use_container_width=True, hide_index=True)

    st.markdown("#### Taylor diagram — deep-learning family")
    st.markdown(
        '<div class="notebook-caption">Same journal-style Taylor diagram as the '
        'Cross-season page, scoped to the four deep-learning / physics-informed models '
        'for this season.</div>',
        unsafe_allow_html=True,
    )
    obs = predictions["Actual DTWL (mbgl)"].to_numpy(dtype=float) if not predictions.empty else None
    deep_preds = {
        name: predictions[f"{name} bias-corrected prediction"].to_numpy(dtype=float)
        for name in metrics["Model"] if f"{name} bias-corrected prediction" in predictions
    }
    if obs is not None and deep_preds:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        _plot_taylor_latex_style(ax, obs=obs, preds_dict=deep_preds, title=result["season"],
                                 sigma_max=None, extra_rmsd=1.6, extra_corr=0.1,
                                 model_colors=DEEP_MODEL_COLORS, model_markers=DEEP_MODEL_MARKERS)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(loc="upper right", fontsize=8, frameon=True)
        _matplotlib_to_streamlit(fig)


# --------------------------------------------------------------------------- #
# Chatbot — first-class page (middle column, light theme)
# --------------------------------------------------------------------------- #
def render_chat(payload: dict[str, Any] | None) -> None:
    """Grounded chat assistant — works offline or with ANTHROPIC_API_KEY."""
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">AI ASSISTANT</div>
        <h1>Ask the dashboard.</h1>
        <p>
            Plain-English questions about your results, metrics, models and
            forecasts — grounded in a curated knowledge base plus a live
            summary of the current run.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_intro(
        title="What this page does — Grounded Q&A assistant",
        what=(
            "A chat box that answers questions about the current run: which season "
            "performed best, what a metric means, how the Taylor diagram is read, "
            "what the wavelet peaks indicate, and so on."
        ),
        why=(
            "New users often do not know which metric to look at first. The assistant "
            "lets them ask in plain English, and it refuses to invent numbers — if the "
            "answer is not in the knowledge base or the current run, it says so."
        ),
    )

    if "rag_history" not in st.session_state:
        st.session_state["rag_history"] = []

    # Centre the chat at ~half the page width using a 1 : 2 : 1 column layout.
    left_pad, chat_col, right_pad = st.columns([1, 2, 1])
    with chat_col:
        for turn in st.session_state["rag_history"]:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("sources"):
                    st.caption("Sources: " + ", ".join(turn["sources"]))

        prompt = st.chat_input("Ask about seasons, metrics, models, Taylor, wavelet…")
        if prompt:
            st.session_state["rag_history"].append(
                {"role": "user", "content": prompt, "sources": []}
            )
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    result = rag_answer(prompt, payload)
                st.markdown(result["answer"])
                if result.get("sources"):
                    st.caption("Sources: " + ", ".join(result["sources"]))

            st.session_state["rag_history"].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
            })


# =========================================================================== #
# PRESENTATION LAYER
# =========================================================================== #
def render_brand_header(status_ready: bool = False) -> None:
    pill = (
        '<div class="status-pill">● Analysis Ready</div>'
        if status_ready
        else '<div class="status-pill" style="background:rgba(0,113,227,.10);'
             'color:#0058b0;border-color:rgba(0,113,227,.22);">● Awaiting upload</div>'
    )
    st.markdown(
        f"""
        <div class="top-nav">
            <div>
                <div class="brand">
                    <span class="brand-icon">💧</span>
                    <span>NAYA-PANI</span>
                </div>
                <div class="brand-subtitle">Groundwater Intelligence · Rajasthan</div>
            </div>
            {pill}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_landing_page() -> None:
    st.markdown("""
    <div class="landing-hero">
        <div class="eyebrow">SEASONAL GROUNDWATER FORECASTING</div>
        <h1>Understand what is happening.<br>Then decide what to do.</h1>
        <p>
            Upload your seasonal groundwater workbook and NAYA-PANI will run the
            full machine-learning workflow — validation, feature engineering,
            model training, forecasting, and explainable diagnostics — in one
            clean, product-grade dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <h3>Forecast</h3>
            <p>Projected groundwater levels for the next 1–10 years, per season,
            with recursive lag updating.</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <h3>Explain</h3>
            <p>See which environmental variables actually drive each seasonal
            forecast — feature selection and leave-one-out sensitivity.</p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Diagnose</h3>
            <p>Taylor diagrams, residual violins, ACF/PACF, ADF, wavelets and
            interactive 3D groundwater surfaces.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("Upload an `.xlsx` workbook above to begin. The file stays in your "
            "session and is never written back to your source workbook.")


def render_analysis_status(payload: dict[str, Any]) -> None:
    data = payload["clean_data"]
    results = payload["results"]
    seasons = ", ".join(results.keys())
    st.markdown(f"""
    <div class="status-strip">
        <div class="status-chip"><span class="dot"></span>
            <b>{len(data):,}</b>&nbsp;clean records</div>
        <div class="status-chip"><span class="dot"></span>
            <b>{data['VILLAGE'].nunique():,}</b>&nbsp;villages</div>
        <div class="status-chip"><span class="dot"></span>
            <b>{len(results)}</b>&nbsp;season{'' if len(results)==1 else 's'} modelled
            &nbsp;<span style="color:var(--secondary);">({seasons})</span></div>
        <div class="status-chip"><span class="dot"></span>
            Forecast horizon:&nbsp;<b>{payload['config'].forecast_steps}</b>&nbsp;years</div>
    </div>
    """, unsafe_allow_html=True)


def render_overview(payload: dict[str, Any]) -> None:
    results = payload["results"]
    data = payload["clean_data"]
    overview = summary_frame(results)

    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">OVERVIEW</div>
        <h1>Understand what is happening.</h1>
        <p>
            Seasonal groundwater forecasting, model performance and explainable
            insights in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    best_r2 = overview["Bias-corrected test R²"].max()
    seasons = len(overview)
    villages = data["VILLAGE"].nunique()
    records = len(data)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">OBSERVATIONS</div>
            <div class="metric-value">{records:,}</div>
            <div class="metric-description">Clean records</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">VILLAGES</div>
            <div class="metric-value">{villages:,}</div>
            <div class="metric-description">Monitoring locations</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SEASONS MODELLED</div>
            <div class="metric-value">{seasons}</div>
            <div class="metric-description">Seasonal models</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BEST TEST R²</div>
            <div class="metric-value">{best_r2:.2f}</div>
            <div class="metric-description">Held-out performance</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Model performance")
    figure = px.bar(
        overview, x="Season", y="Bias-corrected test R²",
        color="Best model", text="Bias-corrected test R²",
    )
    figure.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    figure.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Full season summary table"):
        st.dataframe(overview, use_container_width=True, hide_index=True)

    with st.expander("Data quality & workbook check"):
        render_data_quality(payload)

    with st.expander("Feature engineering & lag preparation"):
        render_features(results)


def render_forecast_page(payload: dict[str, Any]) -> None:
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">FORECAST</div>
        <h1>What happens next?</h1>
        <p>
            Projected groundwater levels for the next 1–10 years, per season.
            The forecast holds exogenous drivers at their last observed value
            and recursively updates the target-history features.
        </p>
    </div>
    """, unsafe_allow_html=True)
    render_forecast_and_export(payload)


def render_models_page(results: dict[str, dict[str, Any]]) -> None:
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">MODELS</div>
        <h1>How well does the system predict?</h1>
        <p>
            Compare classical tree-based models against the deep-learning and
            physics-informed family on the same held-out years.
        </p>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Model family",
        ["Classical ML", "Deep Learning"],
        horizontal=True,
        label_visibility="collapsed",
        key="models_family",
    )

    if mode == "Classical ML":
        render_models(results)
    else:
        render_deep_models(results)


def render_explainability(results: dict[str, dict[str, Any]]) -> None:
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">EXPLAINABLE AI</div>
        <h1>Why did the model predict this?</h1>
        <p>
            Understand which environmental variables influence groundwater
            forecasts — feature selection and leave-one-out sensitivity.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sub = st.radio(
        "Explainability view",
        ["Feature selection", "Feature sensitivity"],
        horizontal=True,
        label_visibility="collapsed",
        key="xai_view",
    )
    if sub == "Feature selection":
        render_selection(results)
    else:
        render_sensitivity(results)


def render_diagnostics_page(payload: dict[str, Any]) -> None:
    results = payload["results"]

    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">MODEL DIAGNOSTICS</div>
        <h1>Can we trust the prediction?</h1>
        <p>
            Scatter, residuals, timelines, cross-season comparison and the full
            rainfall / ACF / ADF picture.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sub = st.radio(
        "Diagnostics view",
        ["Prediction", "Cross-season", "Time series"],
        horizontal=True,
        label_visibility="collapsed",
        key="diag_view",
    )
    if sub == "Prediction":
        render_diagnostics(results)
    elif sub == "Cross-season":
        render_multiseason_analysis(results)
    else:
        render_timeseries_analysis(payload)


def render_hydrology(payload: dict[str, Any]) -> None:
    st.markdown("""
    <div class="page-heading">
        <div class="eyebrow">HYDROLOGY</div>
        <h1>Explore the groundwater system.</h1>
        <p>
            Temporal patterns, spatial relationships and hydrological signals —
            wavelet scalograms and interactive 3D surfaces.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sub = st.radio(
        "Hydrology view",
        ["Wavelet", "3D Analysis"],
        horizontal=True,
        label_visibility="collapsed",
        key="hydro_view",
    )
    if sub == "Wavelet":
        render_wavelet_analysis(payload)
    else:
        render_3d_analysis(payload)


# =========================================================================== #
# SPLASH / LOADING SCREEN — Apple-style "NAYA-PANI" intro
# =========================================================================== #
if "splash_shown" not in st.session_state:
    st.session_state["splash_shown"] = True
    st.markdown(
        """
        <style>
            /* ---- Splash overlay ---- */
            @keyframes splashFadeOut {
                0%   { opacity: 1; }
                85%  { opacity: 1; }
                100% { opacity: 0; pointer-events: none; }
            }
            @keyframes titleReveal {
                0%   { opacity: 0; letter-spacing: 0.45em; transform: scale(0.92); filter: blur(12px); }
                50%  { opacity: 1; letter-spacing: 0.22em; transform: scale(1.0);  filter: blur(0px); }
                100% { opacity: 1; letter-spacing: 0.18em; transform: scale(1.0);  filter: blur(0px); }
            }
            @keyframes subtitleSlideUp {
                0%   { opacity: 0; transform: translateY(28px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            @keyframes taglineFade {
                0%   { opacity: 0; }
                100% { opacity: 0.7; }
            }
            @keyframes ripple {
                0%   { transform: scale(0.7); opacity: 0.6; }
                100% { transform: scale(2.8); opacity: 0; }
            }
            @keyframes dotPulse {
                0%, 100% { opacity: 0.3; transform: scale(0.8); }
                50%      { opacity: 1;   transform: scale(1.2); }
            }

            .splash-overlay {
                position: fixed;
                inset: 0;
                z-index: 999999;
                background: linear-gradient(160deg, #0a0a0a 0%, #111827 40%, #0c1929 100%);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                animation: splashFadeOut 4.2s ease-in-out forwards;
                pointer-events: auto;
            }

            /* Water drop icon */
            .splash-icon {
                font-size: 3.5rem;
                margin-bottom: 2rem;
                animation: subtitleSlideUp 1s ease-out 0.2s both;
                position: relative;
            }
            .splash-icon::after {
                content: '';
                position: absolute;
                bottom: -8px;
                left: 50%;
                transform: translateX(-50%) scale(0.7);
                width: 50px;
                height: 50px;
                border-radius: 50%;
                border: 2px solid rgba(0, 180, 255, 0.25);
                animation: ripple 2s ease-out 0.8s infinite;
            }

            /* Title */
            .splash-title {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display',
                             'Segoe UI', Helvetica, Arial, sans-serif;
                font-size: 5.5rem;
                font-weight: 800;
                letter-spacing: 0.18em;
                color: #FFFFFF;
                margin: 0;
                line-height: 1;
                animation: titleReveal 1.6s cubic-bezier(0.25, 0.46, 0.45, 0.94) 0.3s both;
                background: linear-gradient(135deg, #FFFFFF 0%, #a3d8f4 50%, #5bcefa 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }

            /* Subtitle */
            .splash-subtitle {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text',
                             'Segoe UI', Helvetica, Arial, sans-serif;
                font-size: 1.35rem;
                font-weight: 500;
                color: rgba(255, 255, 255, 0.85);
                letter-spacing: 0.35em;
                text-transform: uppercase;
                margin-top: 1.2rem;
                animation: subtitleSlideUp 1s ease-out 1.1s both;
            }

            /* Tagline */
            .splash-tagline {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text',
                             'Segoe UI', Helvetica, Arial, sans-serif;
                font-size: 0.95rem;
                font-weight: 400;
                color: rgba(255, 255, 255, 0.5);
                letter-spacing: 0.08em;
                margin-top: 2.8rem;
                animation: taglineFade 1s ease-out 1.8s both;
            }

            /* Loading dots */
            .splash-dots {
                display: flex;
                gap: 10px;
                margin-top: 2.2rem;
                animation: taglineFade 0.8s ease-out 2.2s both;
            }
            .splash-dots span {
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background: rgba(91, 206, 250, 0.7);
                animation: dotPulse 1.4s ease-in-out infinite;
            }
            .splash-dots span:nth-child(2) { animation-delay: 0.2s; }
            .splash-dots span:nth-child(3) { animation-delay: 0.4s; }

            /* Gradient line */
            .splash-line {
                width: 120px;
                height: 2px;
                margin-top: 2rem;
                background: linear-gradient(90deg, transparent, rgba(91,206,250,0.5), transparent);
                border-radius: 2px;
                animation: taglineFade 1s ease-out 1.5s both;
            }
        </style>

        <div class="splash-overlay">
            <div class="splash-icon">💧</div>
            <h1 class="splash-title">NAYA-PANI</h1>
            <div class="splash-line"></div>
            <div class="splash-subtitle">Groundwater Intelligence</div>
            <div class="splash-tagline">Every drop counts. Every prediction matters.</div>
            <div class="splash-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================================== #
# PAGE ENTRY POINT
# =========================================================================== #
render_brand_header(status_ready=bool(st.session_state.get("groundwater_dashboard_payload")))

uploaded_file = st.file_uploader(
    "Upload groundwater workbook",
    type=["xlsx"],
    label_visibility="collapsed",
    help="The file stays in the app session and is not written back to your source workbook.",
)

if uploaded_file is None:
    render_landing_page()
    st.stop()

# ---------------------------------------------------------------- load + validate
try:
    upload_identity = (uploaded_file.name, uploaded_file.size)
    previous_identity = st.session_state.get("groundwater_dashboard_upload_identity")
    if previous_identity and previous_identity != upload_identity:
        st.session_state.pop("groundwater_dashboard_payload", None)
    st.session_state["groundwater_dashboard_upload_identity"] = upload_identity
    uploaded_file.seek(0)
    uploaded_data, upload_report, notices = load_seasonal_workbook(uploaded_file)
except Exception as exc:
    st.error(f"The workbook could not be prepared: {exc}")
    st.stop()

years = sorted(uploaded_data["YEAR"].unique().tolist())
available_seasons = usable_seasons(uploaded_data)
if len(years) < 8:
    st.error("At least eight distinct years are needed before the dashboard can create "
             "lags and temporal splits.")
    st.stop()

# ---------------------------------------------------------------- sidebar (settings-only)
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.caption("Configure the analysis, then click **Run complete analysis**.")

    with st.expander("📊 Dataset", expanded=True):
        selected_seasons = st.multiselect(
            "Seasons to analyse", available_seasons, default=available_seasons,
        )

    with st.expander("🎯 Analysis", expanded=True):
        default_train = 2012 if 2012 in years else years[max(0, int(len(years) * 0.60) - 1)]
        train_end = st.selectbox(
            "Training data through year", years[:-2],
            index=years[:-2].index(default_train) if default_train in years[:-2] else len(years[:-2]) - 2,
        )
        validation_years = [y for y in years if y > train_end and y < years[-1]]
        default_validation = 2014 if 2014 in validation_years else validation_years[max(0, len(validation_years) // 2 - 1)]
        validation_end = st.selectbox(
            "Validation data through year", validation_years,
            index=validation_years.index(default_validation),
        )
        forecast_steps = st.slider("Scenario forecast years", min_value=1, max_value=10, value=5)

    with st.expander("🔧 Advanced"):
        feature_count = st.slider("Selected features per season", min_value=3, max_value=15, value=6)
        tree_count = st.select_slider(
            "Tree estimators for ensemble models", options=[100, 250, 400], value=250,
        )
        include_optional = st.toggle("Include optional CatBoost/XGBoost/LightGBM", value=True)
        include_deep = st.toggle(
            "Include PINN, ESN, SARIMA & LSTM models", value=True,
            help="Trains a second, separate model family — shown on the Models page.",
        )
        run_lofo = st.toggle("Run leave-one-feature-out sensitivity", value=True)

    run_analysis = st.button(
        "Run complete analysis", type="primary",
        use_container_width=True, disabled=not selected_seasons,
    )

    st.markdown("---")
    st.caption("**About NAYA-PANI** — v1.0 · Groundwater Intelligence")

if run_analysis:
    try:
        clean_data, validation_report = validate_data(uploaded_data)
        config = RunConfig(
            train_end=int(train_end), validation_end=int(validation_end),
            selected_feature_count=int(feature_count), tree_count=int(tree_count),
            include_optional_models=include_optional, run_lofo=run_lofo,
            forecast_steps=int(forecast_steps), include_deep_models=include_deep,
        )
        progress = st.progress(0, text="Preparing the uploaded data")
        progress_text = st.empty()
        results: dict[str, dict[str, Any]] = {}
        stage_messages = 9
        total_steps = max(1, len(selected_seasons) * stage_messages)
        completed_steps = [0]
        for season in selected_seasons:
            def update(message: str, current_season: str = season) -> None:
                completed_steps[0] += 1
                progress.progress(min(completed_steps[0] / total_steps, 1.0),
                                  text=f"{current_season}: {message}")
                progress_text.caption(f"{current_season}: {message}")
            results[season] = run_season_pipeline(clean_data, season, config, progress=update)
        progress.progress(1.0, text="Analysis complete")
        progress_text.empty()
        st.session_state["groundwater_dashboard_payload"] = {
            "clean_data": clean_data, "workbook_report": upload_report,
            "validation_report": validation_report, "notices": notices,
            "results": results, "config": config,
        }
        st.success("Complete analysis finished. Explore the labelled outputs below.")
    except Exception as exc:
        st.error(f"The analysis stopped safely: {exc}")

payload = st.session_state.get("groundwater_dashboard_payload")

if not payload:
    st.info("Choose the analysis settings in the sidebar, then select **Run complete analysis**.")
    st.stop()

# ---------------------------------------------------------------- status + navigation
render_analysis_status(payload)
results = payload["results"]

navigation = st.radio(
    "Navigation",
    [
        "Overview",
        "Forecast",
        "Models",
        "Explainability",
        "Diagnostics",
        "Hydrology",
        "AI Assistant",
    ],
    horizontal=True,
    label_visibility="collapsed",
    key="main_nav",
)

st.markdown("<br>", unsafe_allow_html=True)

if navigation == "Overview":
    render_overview(payload)
elif navigation == "Forecast":
    render_forecast_page(payload)
elif navigation == "Models":
    render_models_page(results)
elif navigation == "Explainability":
    render_explainability(results)
elif navigation == "Diagnostics":
    render_diagnostics_page(payload)
elif navigation == "Hydrology":
    render_hydrology(payload)
elif navigation == "AI Assistant":
    render_chat(payload)