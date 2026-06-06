"""
XAI-Crop: District-Level Yield Explainability Dashboard
560 Indian districts · 1990-2015 · Multi-source fusion · XGBoost + TabNet + SHAP

Run with:
    streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent.parent
DATA   = BASE / "data/processed"
FIGS   = BASE / "outputs/figures"
MODELS = BASE / "outputs/models"

st.set_page_config(
    page_title="XAI-Crop — District Yield Explainability",
    layout="wide",
    page_icon="◈",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# STYLE INJECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:ital,wght@0,300;0,400;0,600;1,400&display=swap" rel="stylesheet">

<style>
/* ── Variables ──────────────────────────────────────────────────────────── */
:root {
  --ink:           #1c1c1c;
  --ink-mid:       #3a3a3a;
  --ink-muted:     #5c5c5c;
  --ink-faint:     #8c8c8c;
  --paper:         #f9f6f0;
  --paper-alt:     #f2ede4;
  --paper-dark:    #e6ddd2;
  --white:         #ffffff;
  --sb-bg:         #18271a;
  --sb-mid:        #22342a;
  --sb-fg:         #cfc5a8;
  --sb-muted:      #748c6a;
  --sb-faint:      #3d5440;
  --accent:        #c8922a;
  --accent-dim:    rgba(200,146,42,0.12);
  --accent-light:  #f5e6c8;
  --rule:          #c8b898;
  --rule-light:    #e0d8cc;
  --serif:         'Playfair Display', Georgia, 'Times New Roman', serif;
  --sans:          'Source Sans 3', 'Helvetica Neue', Arial, sans-serif;
  --radius:        2px;
}

/* ── Base ───────────────────────────────────────────────────────────────── */
html, body, .stApp {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: var(--sans) !important;
}
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 3rem !important;
}
* { box-sizing: border-box; }

/* ── Scrollbar ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--paper-alt); }
::-webkit-scrollbar-thumb { background: var(--rule); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--sb-bg) !important;
}
section[data-testid="stSidebar"] > div:first-child {
  background: var(--sb-bg) !important;
  padding-top: 0 !important;
}
/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
  color: var(--sb-fg) !important;
}
/* Sidebar selectbox label */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  color: var(--sb-muted) !important;
  font-family: var(--sans) !important;
  font-size: 0.62rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
}
/* Sidebar selectbox input */
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="select-control"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--sb-faint) !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  color: var(--sb-fg) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
  border-bottom-color: var(--accent) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg {
  fill: var(--sb-muted) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] [data-baseweb="select-placeholder"],
section[data-testid="stSidebar"] [data-baseweb="select"] div[class*="ValueContainer"] {
  color: var(--sb-fg) !important;
  font-family: var(--sans) !important;
  font-size: 0.85rem !important;
}
/* Sidebar divider */
section[data-testid="stSidebar"] hr {
  border: none !important;
  border-top: 1px solid var(--sb-faint) !important;
  margin: 1rem 0 !important;
  opacity: 0.6;
}
/* Sidebar caption */
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
  color: var(--sb-faint) !important;
  font-style: italic !important;
  font-size: 0.7rem !important;
}
/* Sidebar image */
section[data-testid="stSidebar"] img {
  filter: brightness(0.85) !important;
  opacity: 0.75 !important;
}

/* ── Dropdown popover (sidebar selects) ─────────────────────────────────── */
[data-baseweb="popover"] [data-baseweb="menu"] {
  background: #1e3020 !important;
  border: 1px solid var(--sb-faint) !important;
}
[data-baseweb="popover"] [role="option"] {
  background: transparent !important;
  color: var(--sb-fg) !important;
  font-family: var(--sans) !important;
  font-size: 0.82rem !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
  background: var(--sb-mid) !important;
  color: var(--accent) !important;
}

/* ── Titles & Headings ──────────────────────────────────────────────────── */
[data-testid="stHeadingWithActionElements"] h1,
.stMarkdown h1 {
  font-family: var(--serif) !important;
  font-weight: 400 !important;
  font-size: 1.9rem !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em !important;
  line-height: 1.25 !important;
}
[data-testid="stHeadingWithActionElements"] h2,
.stMarkdown h2 {
  font-family: var(--serif) !important;
  font-weight: 600 !important;
  font-size: 1.25rem !important;
  color: var(--ink) !important;
  letter-spacing: 0 !important;
}
[data-testid="stHeadingWithActionElements"] h3,
.stMarkdown h3 {
  font-family: var(--serif) !important;
  font-weight: 400 !important;
  font-size: 1.05rem !important;
  color: var(--ink-mid) !important;
}
[data-testid="stHeadingWithActionElements"] h4,
.stMarkdown h4 {
  font-family: var(--sans) !important;
  font-weight: 600 !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--ink-muted) !important;
  margin-bottom: 0.5rem !important;
}
.stCaption {
  font-family: var(--sans) !important;
  font-size: 0.75rem !important;
  font-variant: small-caps !important;
  letter-spacing: 0.1em !important;
  color: var(--ink-faint) !important;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--white) !important;
  border: none !important;
  border-top: 2px solid var(--accent) !important;
  border-radius: var(--radius) !important;
  padding: 14px 16px 12px !important;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--serif) !important;
  font-size: 1.55rem !important;
  font-weight: 400 !important;
  color: var(--ink) !important;
  line-height: 1.15 !important;
}
[data-testid="stMetricLabel"] p {
  font-family: var(--sans) !important;
  font-size: 0.62rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--ink-faint) !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--sans) !important;
  font-size: 0.75rem !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ── Horizontal rule ────────────────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--rule-light) !important;
  margin: 1.25rem 0 !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--rule-light) !important;
  gap: 0 !important;
  padding: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 24px !important;
  margin-bottom: -1px !important;
  font-family: var(--sans) !important;
  font-size: 0.78rem !important;
  font-weight: 400 !important;
  letter-spacing: 0.06em !important;
  color: var(--ink-muted) !important;
  transition: color 0.15s, border-color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--ink) !important;
  background: var(--paper-alt) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  font-family: var(--serif) !important;
  font-style: italic !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] {
  display: none !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding-top: 1.5rem !important;
  background: transparent !important;
}

/* ── Expanders ──────────────────────────────────────────────────────────── */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary {
  font-family: var(--sans) !important;
  font-size: 0.76rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--ink-muted) !important;
  background: var(--paper-alt) !important;
  border: none !important;
  border-top: 1px solid var(--rule-light) !important;
  border-radius: var(--radius) var(--radius) 0 0 !important;
  padding: 10px 14px !important;
}
.streamlit-expanderHeader:hover,
[data-testid="stExpander"] summary:hover {
  color: var(--accent) !important;
  background: var(--paper-dark) !important;
}
.streamlit-expanderContent,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  background: var(--paper-alt) !important;
  border: 1px solid var(--rule-light) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius) var(--radius) !important;
  padding: 14px 16px !important;
}
.streamlit-expanderContent p,
[data-testid="stExpander"] p {
  font-size: 0.84rem !important;
  line-height: 1.7 !important;
  color: var(--ink-mid) !important;
}

/* ── Alerts ─────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: var(--radius) !important;
  border: none !important;
  border-left: 3px solid var(--rule) !important;
  background: var(--paper-alt) !important;
  font-family: var(--sans) !important;
  font-size: 0.84rem !important;
  line-height: 1.65 !important;
  color: var(--ink-mid) !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
div[data-testid="stInfo"] {
  border-left-color: #5c7fa6 !important;
  background: #f0f3f7 !important;
}
div[data-testid="stSuccess"] {
  border-left-color: #4a7a50 !important;
  background: #f0f6f1 !important;
}
div[data-testid="stWarning"] {
  border-left-color: var(--accent) !important;
  background: var(--accent-light) !important;
}

/* ── Markdown body text ──────────────────────────────────────────────────── */
.stMarkdown p {
  font-family: var(--sans) !important;
  font-size: 0.87rem !important;
  color: var(--ink-mid) !important;
  line-height: 1.7 !important;
}
.stMarkdown strong { color: var(--ink) !important; font-weight: 600 !important; }
.stMarkdown code {
  background: var(--paper-dark) !important;
  color: var(--ink-mid) !important;
  font-size: 0.8rem !important;
  padding: 1px 5px !important;
  border-radius: 2px !important;
}
.stMarkdown ul li, .stMarkdown ol li {
  font-size: 0.87rem !important;
  color: var(--ink-mid) !important;
  line-height: 1.65 !important;
}

/* ── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  font-family: var(--sans) !important;
  font-size: 0.8rem !important;
  border: 1px solid var(--rule-light) !important;
}

/* ── Images ─────────────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
  box-shadow: 0 2px 16px rgba(0,0,0,0.09) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stImage"] [data-testid="caption"] {
  font-family: var(--sans) !important;
  font-size: 0.72rem !important;
  font-style: italic !important;
  color: var(--ink-faint) !important;
  text-align: center !important;
}

/* ── Landing page ────────────────────────────────────────────────────────── */
.landing-rule {
  border: none;
  border-top: 1px solid var(--rule);
  margin: 1.5rem 0;
  width: 60%;
}
.landing-title {
  font-family: var(--serif);
  font-size: 2.4rem;
  font-weight: 400;
  color: var(--ink);
  line-height: 1.2;
  letter-spacing: -0.02em;
  margin-bottom: 1rem;
}
.landing-sub {
  font-family: var(--sans);
  font-size: 0.9rem;
  color: var(--ink-muted);
  line-height: 1.75;
  max-width: 480px;
}
.landing-provenance {
  font-family: var(--sans);
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin-top: 1.5rem;
}
.begin-btn > button {
  background: none !important;
  border: none !important;
  border-bottom: 1px solid var(--accent) !important;
  color: var(--accent) !important;
  font-family: var(--serif) !important;
  font-size: 1rem !important;
  font-style: italic !important;
  padding: 0 0 3px !important;
  letter-spacing: 0.04em !important;
  cursor: pointer !important;
  box-shadow: none !important;
  transition: color 0.15s, border-color 0.15s !important;
  border-radius: 0 !important;
}
.begin-btn > button:hover {
  color: var(--ink) !important;
  border-bottom-color: var(--ink) !important;
}

/* ── Sidebar masthead ────────────────────────────────────────────────────── */
.sb-masthead {
  padding: 0 0 1rem 0;
  border-bottom: 1px solid var(--sb-faint);
  margin-bottom: 1.25rem;
}
.sb-overline {
  font-family: var(--sans);
  font-size: 0.58rem;
  letter-spacing: 0.28em;
  text-transform: uppercase;
  color: var(--accent) !important;
  margin-bottom: 4px;
}
.sb-title {
  font-family: var(--serif);
  font-size: 1.05rem;
  font-weight: 400;
  color: var(--sb-fg) !important;
  line-height: 1.2;
}
.sb-subtitle {
  font-family: var(--sans);
  font-size: 0.7rem;
  font-style: italic;
  color: var(--sb-muted) !important;
  margin-top: 3px;
}
.sb-top-rule {
  height: 3px;
  background: var(--accent);
  margin: 0 0 1.25rem 0;
}
.sb-metrics-table {
  width: 100%;
  border-collapse: collapse;
  font-family: var(--sans);
  font-size: 0.75rem;
  margin-bottom: 0.5rem;
}
.sb-metrics-table td {
  padding: 5px 0;
  border-bottom: 1px solid var(--sb-faint);
}
.sb-metrics-table tr:last-child td { border-bottom: none; }
.sb-metric-label { color: var(--sb-muted) !important; }
.sb-metric-val {
  text-align: right;
  font-family: var(--serif);
  font-size: 0.95rem;
  color: var(--sb-fg) !important;
}
.sb-metric-val.accent { color: var(--accent) !important; }
.sb-footer {
  font-family: var(--sans);
  font-size: 0.65rem;
  font-style: italic;
  color: var(--sb-faint) !important;
  border-top: 1px solid var(--sb-faint);
  padding-top: 0.75rem;
  line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df      = pd.read_csv(DATA / "master_clean.csv")
    pred    = pd.read_csv(DATA / "test_predictions.csv")
    df_long = pd.read_csv(DATA / "long_format.csv")
    shap_df = pd.read_csv(DATA / "shap_values.csv")
    with open(DATA / "meta.json") as f:
        meta = json.load(f)
    map_meta = (
        json.loads((DATA / "map_meta.json").read_text())
        if (DATA / "map_meta.json").exists() else {}
    )
    ctrl = (
        json.loads((DATA / "controllability.json").read_text())
        if (DATA / "controllability.json").exists() else {}
    )
    return df, pred, df_long, shap_df, meta, map_meta, ctrl


@st.cache_resource
def load_model():
    with open(MODELS / "xgboost_model.pkl", "rb") as f:
        return pickle.load(f)


df, pred_df, df_long, shap_df, meta, map_meta, ctrl = load_data()
model = load_model()

T    = meta["TARGET"]
DC   = meta["DISTRICT_COL"]
SC   = meta["STATE_COL"]
YC   = meta["YEAR_COL"]
FEAT = meta["feature_cols"]
RMSE = meta["model_rmse"]
MAE  = meta["model_mae"]
R2   = meta["model_r2"]

CONTROLLABLE = {
    "NITROGEN CONSUMPTION (tons)",
    "PHOSPHATE CONSUMPTION (tons)",
    "POTASH CONSUMPTION (tons)",
    "TOTAL FERTILISER CONSUMPTION (tons)",
    "GROSS IRRIGATED AREA (1000 ha)",
    "GROSS CROPPED AREA (1000 ha)",
    "RICE AREA (1000 ha)",
}
UNCONTROLLABLE_KEYWORDS = ["TEMPERATURE", "PERCIPITATION", "EVAPOTRANSPIRATION", "WINDSPEED"]


# ═══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB THEME
# ═══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":         "serif",
    "font.serif":          ["Palatino Linotype", "Palatino", "Georgia", "DejaVu Serif"],
    "axes.facecolor":      "#f9f6f0",
    "figure.facecolor":    "#f9f6f0",
    "savefig.facecolor":   "#f9f6f0",
    "axes.edgecolor":      "#c8b898",
    "axes.linewidth":      0.7,
    "axes.grid":           True,
    "grid.color":          "#e0d8cc",
    "grid.linewidth":      0.5,
    "grid.alpha":          0.8,
    "text.color":          "#1c1c1c",
    "axes.labelcolor":     "#3a3a3a",
    "xtick.color":         "#5c5c5c",
    "ytick.color":         "#5c5c5c",
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "axes.titlesize":      10,
    "axes.labelsize":      9,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "figure.dpi":          130,
    "legend.fontsize":     8,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#e0d8cc",
})

CHART_COLORS = {
    "primary":   "#18271a",
    "accent":    "#c8922a",
    "muted":     "#7a9470",
    "highlight": "#c8922a",
    "negative":  "#8b3a3a",
    "fill":      "#18271a",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    with st.sidebar:
        st.markdown('<div class="sb-top-rule"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sb-masthead">
          <div class="sb-overline">Yield Intelligence</div>
          <div class="sb-title">XAI · Crop</div>
          <div class="sb-subtitle">District-Level Observatory</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <table class="sb-metrics-table">
          <tr>
            <td class="sb-metric-label">XGBoost R²</td>
            <td class="sb-metric-val accent">{R2}</td>
          </tr>
          <tr>
            <td class="sb-metric-label">RMSE</td>
            <td class="sb-metric-val">{RMSE} Kg/ha</td>
          </tr>
          <tr>
            <td class="sb-metric-label">MAE</td>
            <td class="sb-metric-val">{MAE} Kg/ha</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="sb-footer">
          ICRISAT · IMD · data.gov.in<br>
          560 districts · 1990–2015
        </div>
        """, unsafe_allow_html=True)

    # Landing content
    col_left, col_right = st.columns([3, 2], gap="large")
    with col_left:
        st.markdown("""
        <div style="padding-top: 3rem;">
          <div style="font-family:'Source Sans 3',sans-serif; font-size:0.65rem;
                      letter-spacing:0.22em; text-transform:uppercase;
                      color:#8c8c8c; margin-bottom:1rem;">
            Agricultural Intelligence · India · 1990–2015
          </div>
          <div class="landing-title">
            District-Level Crop<br>Yield Intelligence
          </div>
          <div class="landing-sub">
            An explainability observatory for rice yield across 560 Indian districts.
            Integrates ICRISAT agronomy data, IMD rainfall records, and district-level
            fertilizer consumption to train a gradient-boosted model — then applies SHAP
            attribution to decompose each prediction into controllable and uncontrollable
            policy levers.
          </div>
          <hr class="landing-rule">
          <div class="landing-provenance">
            ICRISAT District-Level Data &nbsp;·&nbsp;
            IMD Rainfall &nbsp;·&nbsp;
            District Fertilizer Consumption &nbsp;·&nbsp;
            560 districts &nbsp;·&nbsp; 1990–2015
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="begin-btn">', unsafe_allow_html=True)
        if st.button("Begin Analysis  →"):
            st.session_state.started = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div style="padding-top:3rem; display:flex; align-items:center;
                    justify-content:center; height:100%;">
          <svg width="100%" viewBox="0 0 360 420"
               xmlns="http://www.w3.org/2000/svg" style="max-width:340px; opacity:0.85;">
            <defs>
              <pattern id="dp" x="0" y="0" width="22" height="22"
                        patternUnits="userSpaceOnUse">
                <circle cx="11" cy="11" r="1.2" fill="#c8922a" opacity="0.25"/>
              </pattern>
            </defs>
            <rect width="360" height="420" fill="url(#dp)"/>
            <!-- concentric rings -->
            <circle cx="180" cy="210" r="140" fill="none"
                    stroke="#18271a" stroke-width="0.6" opacity="0.12"/>
            <circle cx="180" cy="210" r="105" fill="none"
                    stroke="#18271a" stroke-width="0.6" opacity="0.18"/>
            <circle cx="180" cy="210" r="70"  fill="none"
                    stroke="#18271a" stroke-width="0.8" opacity="0.22"/>
            <circle cx="180" cy="210" r="36"  fill="none"
                    stroke="#c8922a" stroke-width="1.2" opacity="0.5"/>
            <circle cx="180" cy="210" r="5"   fill="#c8922a" opacity="0.7"/>
            <!-- cardinal cross hairs -->
            <line x1="180" y1="70"  x2="180" y2="350"
                  stroke="#18271a" stroke-width="0.5" opacity="0.12"
                  stroke-dasharray="4 6"/>
            <line x1="40"  y1="210" x2="320" y2="210"
                  stroke="#18271a" stroke-width="0.5" opacity="0.12"
                  stroke-dasharray="4 6"/>
            <!-- data points -->
            <circle cx="130" cy="150" r="4" fill="#18271a" opacity="0.35"/>
            <circle cx="220" cy="140" r="6" fill="#c8922a" opacity="0.5"/>
            <circle cx="245" cy="265" r="4" fill="#18271a" opacity="0.35"/>
            <circle cx="140" cy="280" r="5" fill="#c8922a" opacity="0.45"/>
            <circle cx="190" cy="155" r="3" fill="#18271a" opacity="0.3"/>
            <circle cx="155" cy="235" r="3.5" fill="#18271a" opacity="0.3"/>
            <!-- label -->
            <text x="180" y="388"
                  font-family="'Playfair Display',serif"
                  font-size="11" fill="#8c8c8c"
                  text-anchor="middle" font-style="italic">
              district yield observatory
            </text>
          </svg>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-top-rule"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-masthead">
      <div class="sb-overline">Yield Intelligence</div>
      <div class="sb-title">XAI · Crop</div>
      <div class="sb-subtitle">District-Level Observatory</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <table class="sb-metrics-table">
      <tr>
        <td class="sb-metric-label">XGBoost R²</td>
        <td class="sb-metric-val accent">{R2}</td>
      </tr>
      <tr>
        <td class="sb-metric-label">RMSE</td>
        <td class="sb-metric-val">{RMSE} Kg/ha</td>
      </tr>
      <tr>
        <td class="sb-metric-label">MAE</td>
        <td class="sb-metric-val">{MAE} Kg/ha</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")

    states    = sorted(df[SC].unique())
    sel_state = st.selectbox("State", states)
    dists     = sorted(df[df[SC] == sel_state][DC].unique())
    sel_dist  = st.selectbox("District", dists)
    crops_av  = [c.replace(" YIELD (Kg per ha)", "") for c in meta.get("yield_cols", [])]
    sel_crop  = st.selectbox("Crop", crops_av if crops_av else ["Rice"])

    st.markdown("---")
    st.markdown("""
    <div class="sb-footer">
      ICRISAT · IMD · data.gov.in · Kaggle<br>
      560 districts · 1990–2015
    </div>
    """, unsafe_allow_html=True)


# ─── Filter ───────────────────────────────────────────────────────────────────
sel_df   = df[(df[SC] == sel_state) & (df[DC] == sel_dist)]
sel_pred = (
    pred_df[(pred_df[SC] == sel_state) & (pred_df[DC] == sel_dist)]
    if SC in pred_df.columns else pd.DataFrame()
)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("XAI-Crop: District-Level Yield Explainability")
st.caption(
    "560 Indian districts  ·  1990–2015  ·  Multi-source fusion  ·  "
    "XGBoost + TabNet + SHAP  ·  Policy-actionable insights"
)

c1, c2, c3, c4, c5 = st.columns(5)
if not sel_pred.empty:
    latest = sel_pred.sort_values(YC, ascending=False).iloc[0]
    c1.metric("Actual Yield", f"{latest[T]:.0f} Kg/ha")
    c2.metric("Predicted",    f"{latest['predicted']:.0f} Kg/ha",
              delta=f"{latest['predicted'] - latest[T]:.0f}")
    c3.metric("Abs Error",    f"{latest['error']:.0f} Kg/ha")
elif not sel_df.empty:
    c1.metric("Avg Yield", f"{sel_df[T].mean():.0f} Kg/ha")
c4.metric("Years of data", len(sel_df))
c5.metric(
    "vs national avg",
    f"{sel_df[T].mean() - df[T].mean():+.0f} Kg/ha" if not sel_df.empty else "—",
)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "◈  Analysis",
    "▦  Maps",
    "⌕  SHAP",
    "≡  Benchmark",
    "⊙  Controllability",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"District Deep-Dive — {sel_dist}, {sel_state}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Rice Yield Distribution — {sel_dist}**")
        if not sel_df.empty and not sel_df[T].dropna().empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(sel_df[T].dropna(), bins=20,
                    color=CHART_COLORS["primary"], edgecolor="#f9f6f0", alpha=0.85)
            ax.set_xlabel("Rice Yield (Kg/ha)")
            ax.set_ylabel("Years")
            ax.set_title(f"Yield distribution · {sel_dist}", loc="left")
            fig.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.info("No rice yield data for this district.")

    with col2:
        st.markdown(f"**{sel_crop} Yield Trend — {sel_dist}**")
        crop_yield_col = f"{sel_crop} YIELD (Kg per ha)"
        if not sel_df.empty and crop_yield_col in sel_df.columns:
            trend = sel_df.groupby(YC)[crop_yield_col].mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(trend.index, trend.values, marker="o",
                    color=CHART_COLORS["primary"], linewidth=1.8,
                    markersize=4, markerfacecolor=CHART_COLORS["accent"])
            ax.fill_between(trend.index, trend.values,
                            alpha=0.1, color=CHART_COLORS["fill"])
            ax.set_xlabel("Year")
            ax.set_ylabel("Yield (Kg/ha)")
            ax.set_title(f"{sel_crop} yield trend · {sel_dist}", loc="left")
            fig.tight_layout()
            st.pyplot(fig); plt.close()
        else:
            st.info(f"No {sel_crop} yield data for {sel_dist}.")

    st.markdown("---")
    st.subheader("Pre-computed EDA Figures")

    eda_figures = [
        ("yield_distribution.png", "Yield Distribution",
         "Rice yield follows a right-skewed distribution, with most districts clustering "
         "between 800–2,500 Kg/ha. The long right tail reflects high-performing irrigated "
         "districts in Punjab and Haryana. The multi-crop overlay reveals Sugarcane's "
         "dominance in scale while pulses (Chickpea) sit at lower absolute yields."),
        ("yield_trend_yearly.png", "Yield Trend by Year",
         "All crops show a broadly upward trajectory 1990–2015, consistent with the Green "
         "Revolution legacy and improved inputs. Rice yield growth accelerated post-2000, "
         "likely driven by HYV seed adoption and expanded irrigation. Dips around "
         "2002–2003 and 2009–2010 align with documented drought years."),
        ("district_yield_top_bottom.png", "Top vs Bottom Districts",
         "Top-10 districts (primarily from West Bengal's Hooghly delta and Punjab) achieve "
         "yields 3–5× higher than the bottom-10, which are concentrated in rain-fed "
         "tribal belts of Jharkhand and Odisha. This gap underscores the critical role of "
         "irrigation access and soil quality in determining output."),
        ("correlation_heatmap.png", "Correlation Heatmap",
         "Rice Area (1000 ha) shows the strongest positive correlation with yield. "
         "Nitrogen and Gross Irrigated Area follow closely. Temperature variables show a "
         "weak negative correlation — high summer temperatures cause heat stress during "
         "grain filling. Wind speed shows near-zero correlation."),
        ("rainfall_vs_yield.png", "Rainfall vs Yield",
         "The relationship between Kharif (JUN–SEP) rainfall and rice yield is non-linear. "
         "Yields peak at moderate rainfall (~900–1,200 mm) and decline at extremes. "
         "State-level coloring reveals that irrigation infrastructure largely decouples "
         "Punjab and Haryana from rainfall dependency."),
        ("yield_by_season.png", "Yield by Season",
         "Kharif crops (including Rice) exhibit higher median yields but also greater "
         "variance, reflecting dependence on monsoon variability. Rabi crops (Chickpea) "
         "show a tighter distribution due to winter moisture being more predictable."),
        ("actual_vs_predicted.png", "Actual vs Predicted",
         f"XGBoost achieves R²={R2}, RMSE={RMSE} Kg/ha on the held-out 20% test set. "
         "Points cluster tightly along the diagonal, indicating well-calibrated predictions. "
         "Slight under-prediction at the high end (>4,000 Kg/ha) suggests rare high-yield "
         "outlier districts may benefit from additional features."),
        ("feature_importance.png", "Feature Importance",
         "XGBoost (gain-based importance) highlights Rice Area, State encoding, and Year "
         "as the top drivers. Nitrogen consumption ranks 5th, confirming fertilizer policy "
         "as the most actionable lever. Climate variables rank mid-table, providing context "
         "but not dominating when irrigation is available."),
    ]

    for fname, label, interpretation in eda_figures:
        fp = FIGS / fname
        if fp.exists():
            st.image(str(fp), caption=label, use_container_width=True)
            with st.expander(f"↳  How to read this — {label}"):
                st.markdown(interpretation)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MAPS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Choropleth Maps — India District Level")

    if map_meta:
        st.markdown("#### Summary Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Districts mapped",    map_meta.get("districts_mapped", "—"))
        mc2.metric("Highest avg yield",
                   f"{map_meta.get('highest_avg_yield', 0):.0f} Kg/ha",
                   help=map_meta.get("highest_district", ""))
        mc3.metric("Improving districts", map_meta.get("improving_districts", "—"))
        mc4.metric("Declining districts", map_meta.get("declining_districts", "—"))
        mc5, mc6 = st.columns(2)
        mc5.metric("Lowest avg yield",
                   f"{map_meta.get('lowest_avg_yield', 0):.0f} Kg/ha",
                   help=map_meta.get("lowest_district", ""))
        mc6.metric("Fastest declining",
                   map_meta.get("fastest_declining", "—"),
                   delta=f"{map_meta.get('fastest_declining_slope', 0):.1f} Kg/ha/yr",
                   delta_color="inverse")
        st.markdown("---")

    for label, fname, interpretation in [
        ("Average Rice Yield per District (1990–2015)", "choropleth_yield.html",
         "Districts in the Indo-Gangetic Plain (Punjab, Haryana, West Bengal) consistently "
         "show the highest average yields, driven by intensive irrigation and input use. "
         "The Northeast and Central India display mid-range yields, while rain-shadow "
         "regions show the lowest yields. This spatial pattern directly informs where "
         "targeted intervention can yield the greatest improvement."),
        ("Rice Yield Trend per District — OLS Slope, 1990–2015", "choropleth_trend.html",
         "Green districts are improving (positive slope Kg/ha/yr); red are declining. "
         "Most of India shows modest improvement, but pockets of decline appear in "
         "waterlogged eastern districts (soil degradation from over-irrigation) and "
         "drought-prone central districts."),
    ]:
        fp = FIGS / fname
        if fp.exists():
            st.markdown(f"**{label}**")
            with open(fp, encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=580, scrolling=True)
            with st.expander("↳  Map interpretation"):
                st.markdown(interpretation)
        else:
            st.warning(f"{fname} not found. Run analysis.py to generate it.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Comparison: XGBoost vs TabNet")
    tabnet_path = MODELS / "tabnet_results.json"
    if tabnet_path.exists():
        with open(tabnet_path) as f:
            tn = json.load(f)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**XGBoost**")
            st.metric("RMSE", f"{RMSE} Kg/ha")
            st.metric("MAE",  f"{meta['model_mae']} Kg/ha")
            st.metric("R²",   str(R2))
        with col_b:
            st.markdown("**TabNet**")
            st.metric("RMSE", f"{tn['rmse']} Kg/ha")
            st.metric("MAE",  f"{tn['mae']} Kg/ha")
            st.metric("R²",   str(tn['r2']))
        st.info(
            "XGBoost outperforms TabNet on this dataset for three reasons: "
            "(1) ~23 samples per district is too few for TabNet's attention heads to converge meaningfully; "
            "(2) agricultural features have threshold-style interactions that decision-tree splits capture natively; "
            "(3) the high feature-to-row ratio makes deep networks prone to overfitting on small-N tabular data."
        )
    else:
        st.warning("TabNet results not found — run `python tabnet_train.py` first.")
    st.markdown("---")

    st.subheader("Global SHAP Explainability")

    shap_figures = [
        ("shap/shap_beeswarm.png", "Global Feature Importance (Beeswarm)",
         "Each dot is one test sample. Features are ordered by mean |SHAP|. Pink = high "
         "feature value; blue = low. Rice Area shows the widest spread — large cultivated "
         "area consistently pushes yields higher. Nitrogen's band shows that high nitrogen "
         "use reliably increases predicted yield, making it the top policy lever."),
        ("shap/shap_waterfall_best.png", "Waterfall — Best Prediction",
         "Arrows show how each feature pushed the prediction above (red) or below (blue) "
         "the baseline. For the best-predicted sample, the model had strong, consistent "
         "signals across Rice Area, irrigation, and nitrogen — typical of a well-irrigated "
         "Punjab/Haryana district."),
        ("shap/shap_waterfall_worst.png", "Waterfall — Worst Prediction",
         "The worst prediction shows conflicting SHAP contributions — some features push "
         "strongly upward while others pull down, indicating an atypical district-year. "
         "This likely reflects a sudden shock (flood, fertilizer shortage) not captured "
         "by annual averages."),
    ]
    for fname, label, interp in shap_figures:
        fp = FIGS / fname
        if fp.exists():
            st.image(str(fp), caption=label, use_container_width=True)
            with st.expander(f"↳  How to read this — {label}"):
                st.markdown(interp)

    for feat in meta.get("top2_shap", []):
        safe = feat.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        fp = FIGS / f"shap/shap_dependence_{safe}.png"
        if fp.exists():
            st.image(str(fp), caption=f"SHAP Dependence — {feat}", use_container_width=True)
            with st.expander(f"↳  SHAP Dependence — {feat}"):
                st.markdown(
                    f"X-axis: actual values of **{feat}**. Y-axis: SHAP contribution. "
                    "Upward slope = positive marginal effect. Colour encodes a second "
                    "feature; alignment of colour gradient with SHAP gradient signals "
                    "a synergistic interaction."
                )

    st.markdown("---")
    st.subheader(f"● Live SHAP Waterfall — {sel_dist}")
    if not sel_pred.empty:
        row_feat = [c for c in FEAT if c in pred_df.columns]
        if row_feat:
            latest_row = sel_pred.sort_values(YC, ascending=False).iloc[0]
            X_row      = latest_row[row_feat].values.reshape(1, -1)
            explainer  = shap.TreeExplainer(model)
            sv         = explainer(pd.DataFrame(X_row, columns=row_feat))
            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(sv[0], show=False, max_display=15)
            plt.title(
                f"SHAP Waterfall  ·  {sel_dist}  ·  {int(latest_row[YC])}",
                fontsize=10, loc="left",
            )
            plt.tight_layout()
            st.pyplot(plt.gcf()); plt.close()
        else:
            st.info("Feature columns not available in prediction file.")
    else:
        st.info("No test-set predictions found for this district.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Benchmark — XGBoost vs TabNet")

    tabnet_path = MODELS / "tabnet_results.json"
    tn = None
    if tabnet_path.exists():
        with open(tabnet_path) as f:
            tn = json.load(f)
    elif meta.get("tabnet_rmse"):
        tn = {"rmse": meta["tabnet_rmse"], "mae": meta["tabnet_mae"], "r2": meta["tabnet_r2"]}

    if tn:
        b1, b2, b3 = st.columns(3)
        for col, metric, xgb_val, tn_val in [
            (b1, "RMSE (Kg/ha)", RMSE, tn["rmse"]),
            (b2, "MAE (Kg/ha)",  MAE,  tn["mae"]),
            (b3, "R²",           R2,   tn["r2"]),
        ]:
            delta = tn_val - xgb_val
            col.metric(
                label=metric,
                value=f"XGB: {xgb_val}  |  TabNet: {tn_val}",
                delta=f"TabNet {delta:+.4f}",
                delta_color="inverse" if metric != "R²" else "normal",
            )
        st.markdown("---")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (metric, xgb_v, tn_v) in zip(axes, [
            ("RMSE (Kg/ha)", RMSE, tn["rmse"]),
            ("MAE (Kg/ha)",  MAE,  tn["mae"]),
            ("R²",           R2,   tn["r2"]),
        ]):
            bars = ax.bar(
                ["XGBoost", "TabNet"], [xgb_v, tn_v],
                color=[CHART_COLORS["primary"], CHART_COLORS["accent"]],
                edgecolor="#f9f6f0", width=0.45,
            )
            ax.set_title(metric, loc="left")
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.015,
                    f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=8,
                )
            ax.set_ylim(0, max(xgb_v, tn_v) * 1.18)
        fig.tight_layout(pad=2)
        st.pyplot(fig); plt.close()

        with st.expander("↳  Why does XGBoost outperform TabNet on this dataset?"):
            st.markdown("""
**XGBoost is better suited to this dataset for several structural reasons:**

1. **Strong feature interactions on tabular data** — Gradient boosted trees exploit
   interactions (e.g. nitrogen × irrigated area → yield) directly through split
   hierarchies; TabNet must learn them via sequential attention steps.

2. **Moderate dataset size (~12,800 rows)** — TabNet's attention mechanism offers the
   most benefit on very large datasets (>100k rows). With ~10,000 training rows, the
   self-supervised pre-training benefit is limited and the model may under-generalise.

3. **Many sparse and imputed features** — ~40% of climate columns required median
   imputation. XGBoost natively handles missing values; TabNet's batch normalisation
   can be distorted by systematic imputation.

4. **Label-encoded geographic identifiers** — Tree splits handle integer-encoded
   categoricals efficiently. TabNet's MLP-style backbone treats them as continuous,
   introducing unintended ordinality.

5. **Interpretability alignment** — XGBoost + SHAP provides cleaner feature-level
   attribution, which is core to this dashboard's policy-recommendation objective.

**TabNet's potential strengths** (not fully activated here): semi-supervised
pre-training on unlabelled districts, or when a much larger external dataset
is incorporated (e.g. full India crop census post-2015).
            """)

        if "train_time_s" in tn:
            st.info(f"TabNet training time: **{tn['train_time_s']} s**")
    else:
        st.info(
            "TabNet results not found. "
            "Run `python tabnet_train.py` to generate `outputs/models/tabnet_results.json`."
        )

    if "enriched_rmse" in meta:
        st.markdown("---")
        st.subheader("Feature Enrichment Impact")
        tbl = pd.DataFrame({
            "Model variant": [
                "Baseline XGBoost",
                "XGBoost + IMD Rain",
                "XGBoost + IMD + Season",
            ],
            "RMSE": [meta["model_rmse"], 377.35, meta["enriched_rmse"]],
            "MAE":  [meta["model_mae"],  261.79, meta["enriched_mae"]],
            "R²":   [meta["model_r2"],   0.8710, meta["enriched_r2"]],
        })
        st.dataframe(tbl.set_index("Model variant"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CONTROLLABILITY
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("SHAP Controllability & Policy Framework")

    if ctrl:
        avg_ctrl   = ctrl.get("avg_controllable_shap",   0)
        avg_unctrl = ctrl.get("avg_uncontrollable_shap", 0)
        best_lever = ctrl.get("best_lever",      "Nitrogen Consumption")
        best_shap  = ctrl.get("best_lever_shap", 0)

        st.markdown("#### Policy Summary")
        ps1, ps2, ps3 = st.columns(3)
        ps1.metric("Controllable avg SHAP",   f"{avg_ctrl:+.1f} Kg/ha",
                   help="Average SHAP contribution of all controllable features")
        ps2.metric("Uncontrollable avg SHAP", f"{avg_unctrl:+.1f} Kg/ha",
                   help="Average SHAP contribution of climate/weather features")
        ps3.metric("Top policy lever", best_lever,
                   delta=f"avg |SHAP| = {best_shap:.1f} Kg/ha")

        st.markdown("---")
        st.markdown("#### Feature Controllability Breakdown")

        if "controllable" in ctrl:
            feat_importance: dict[str, float] = {}
            sv_path = DATA / "shap_values.csv"
            if sv_path.exists():
                sv_csv  = pd.read_csv(sv_path)
                sv_vals = sv_csv.drop(columns=["row_index"], errors="ignore")
                for col in sv_vals.columns:
                    feat_importance[col] = float(sv_vals[col].abs().mean())

            rows = []
            for feat in ctrl.get("controllable", []):
                rows.append({
                    "Feature":       feat,
                    "Category":      "Controllable",
                    "SHAP Impact":   round(feat_importance.get(feat, 0), 2),
                    "Policy Action": "Increase / optimise via government schemes",
                })
            for feat in ctrl.get("uncontrollable", []):
                rows.append({
                    "Feature":       feat,
                    "Category":      "Uncontrollable",
                    "SHAP Impact":   round(feat_importance.get(feat, 0), 2),
                    "Policy Action": "Adapt via crop insurance / drought-resistant varieties",
                })
            tbl = pd.DataFrame(rows).sort_values("SHAP Impact", ascending=False)
            st.dataframe(tbl.set_index("Feature"), use_container_width=True)

        st.markdown("---")
        st.subheader(f"● Policy Waterfall — {sel_dist}")

        if not sel_pred.empty:
            row_feat = [c for c in FEAT if c in pred_df.columns]
            if row_feat:
                latest_row = sel_pred.sort_values(YC, ascending=False).iloc[0]
                X_row      = latest_row[row_feat].values.reshape(1, -1)
                explainer  = shap.TreeExplainer(model)
                sv_live    = explainer(pd.DataFrame(X_row, columns=row_feat))
                sv_arr     = sv_live[0].values

                ctrl_set   = set(ctrl.get("controllable",   []))
                unctrl_set = set(ctrl.get("uncontrollable", []))
                colors     = [
                    CHART_COLORS["muted"]    if fn in ctrl_set   else
                    CHART_COLORS["negative"] if fn in unctrl_set else
                    "#9a9585"
                    for fn in row_feat
                ]

                top_k  = 15
                order  = np.argsort(np.abs(sv_arr))[::-1][:top_k]
                vals   = sv_arr[order]
                clrs   = [colors[i] for i in order]
                fnames = [row_feat[i] for i in order]

                fig, ax = plt.subplots(figsize=(12, 5.5))
                ax.barh(np.arange(len(order)), vals, color=clrs,
                        edgecolor="#f9f6f0", height=0.65)
                ax.set_yticks(np.arange(len(order)))
                ax.set_yticklabels(fnames, fontsize=8)
                ax.axvline(0, color=CHART_COLORS["primary"], linewidth=0.8, alpha=0.5)
                ax.set_xlabel("SHAP Value (Kg/ha impact on yield)")
                ax.set_title(
                    f"Policy waterfall  ·  {sel_dist}  ·  {int(latest_row[YC])}"
                    "\n[green] Controllable   [dark] Uncontrollable   [grey] Structural",
                    fontsize=9, loc="left",
                )
                fig.tight_layout()
                st.pyplot(fig); plt.close()

                ctrl_idx = [i for i, fn in enumerate(row_feat) if fn in ctrl_set]
                if ctrl_idx:
                    ctrl_shap_vals = sv_arr[ctrl_idx]
                    best_i         = ctrl_idx[int(np.argmax(np.abs(ctrl_shap_vals)))]
                    best_ctrl_fn   = row_feat[best_i]
                    best_ctrl_val  = sv_arr[best_i]
                    st.success(
                        f"**◆ Highest-impact policy lever for {sel_dist}:**  "
                        f"`{best_ctrl_fn}`  →  SHAP: **{best_ctrl_val:+.1f} Kg/ha**\n\n"
                        "Prioritise this input in agricultural extension programmes."
                    )
        else:
            st.info("Select a district with test-set predictions to see the live waterfall.")
    else:
        st.info(
            "Controllability data not found. "
            "Run `python analysis.py` to generate `data/processed/controllability.json`."
        )
