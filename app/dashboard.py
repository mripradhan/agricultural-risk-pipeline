"""
ICRISAT Crop Yield Interactive Dashboard
Run with: streamlit run app/dashboard.py
"""

import matplotlib
matplotlib.use('Agg')          # must be set before pyplot is imported anywhere
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, shap
from pathlib import Path

BASE    = Path(__file__).parent.parent
DATA    = BASE / "data/processed"
FIGS    = BASE / "outputs/figures"
MODELS  = BASE / "outputs/models"

st.set_page_config(page_title="ICRISAT Yield Explorer", layout="wide", page_icon="🌾")

@st.cache_data
def load_data():
    df      = pd.read_csv(DATA / "master_clean.csv")
    pred    = pd.read_csv(DATA / "test_predictions.csv")
    df_long = pd.read_csv(DATA / "long_format.csv")
    shap_df = pd.read_csv(DATA / "shap_values.csv")
    with open(DATA / "meta.json") as f:
        meta = json.load(f)
    return df, pred, df_long, shap_df, meta

@st.cache_resource
def load_model():
    with open(MODELS / "xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

df, pred_df, df_long, shap_df, meta = load_data()
model = load_model()

T    = meta["TARGET"]
DC   = meta["DISTRICT_COL"]
SC   = meta["STATE_COL"]
YC   = meta["YEAR_COL"]
FEAT = meta["feature_cols"]
RMSE = meta["model_rmse"]
R2   = meta["model_r2"]

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/thumb/e/e9/ICRISAT.svg/200px-ICRISAT.svg.png",
    width=140
)
st.sidebar.title("🌾 ICRISAT Explorer")
st.sidebar.markdown(f"**Model R²:** {R2}   |   **RMSE:** {RMSE} Kg/ha")
st.sidebar.markdown("---")

states   = sorted(df[SC].unique())
sel_state = st.sidebar.selectbox("State", states)

dists    = sorted(df[df[SC] == sel_state][DC].unique())
sel_dist  = st.sidebar.selectbox("District", dists)

crops_av = [c.replace(" YIELD (Kg per ha)", "") for c in meta.get("yield_cols", [])]
sel_crop  = st.sidebar.selectbox("Crop (for trend)", crops_av if crops_av else ["Rice"])

# ─── Filter ──────────────────────────────────────────────────────────────────
sel_df   = df[(df[SC] == sel_state) & (df[DC] == sel_dist)]
sel_pred = pred_df[(pred_df[SC] == sel_state) & (pred_df[DC] == sel_dist)] if SC in pred_df.columns else pd.DataFrame()

# ─── Top metric cards ─────────────────────────────────────────────────────────
st.title("ICRISAT District-Level Crop Yield Analysis")
c1, c2, c3, c4 = st.columns(4)

if not sel_pred.empty:
    latest = sel_pred.sort_values(YC, ascending=False).iloc[0]
    c1.metric("Actual Yield (latest)", f"{latest[T]:.0f} Kg/ha")
    c2.metric("Predicted Yield",       f"{latest['predicted']:.0f} Kg/ha",
              delta=f"{latest['predicted'] - latest[T]:.0f}")
    c3.metric("Abs Error",             f"{latest['error']:.0f} Kg/ha")
elif not sel_df.empty:
    c1.metric("Avg Yield (selection)", f"{sel_df[T].mean():.0f} Kg/ha")

c4.metric("Years of data", len(sel_df))
st.markdown("---")

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🗺️ Maps", "🔍 SHAP"])

# ── Tab 1: EDA ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"District Deep-Dive — {sel_dist}, {sel_state}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Rice Yield Distribution — {sel_dist}**")
        if not sel_df.empty and not sel_df[T].dropna().empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(sel_df[T].dropna(), bins=20, color="steelblue", edgecolor="white")
            ax.set_xlabel("Rice Yield (Kg/ha)")
            ax.set_ylabel("Years")
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No rice yield data for this district.")

    with col2:
        st.markdown(f"**{sel_crop} Yield Trend — {sel_dist}**")
        crop_yield_col = f"{sel_crop} YIELD (Kg per ha)"
        if not sel_df.empty and crop_yield_col in sel_df.columns:
            trend = sel_df.groupby(YC)[crop_yield_col].mean()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(trend.index, trend.values, marker="o", color="royalblue", linewidth=2)
            ax.fill_between(trend.index, trend.values, alpha=0.15, color="royalblue")
            ax.set_xlabel("Year")
            ax.set_ylabel("Yield (Kg/ha)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        else:
            st.info(f"No {sel_crop} yield data for {sel_dist}.")

    st.markdown("---")
    st.subheader("Pre-computed EDA Figures")
    for label, fname in [
        ("Yield Distribution by Crop",       "yield_distribution.png"),
        ("Year-wise Yield Trend (All Crops)", "yield_trend_yearly.png"),
        ("Top/Bottom Districts",              "district_yield_top_bottom.png"),
        ("Correlation Heatmap",               "correlation_heatmap.png"),
        ("Rainfall vs Yield",                 "rainfall_vs_yield.png"),
        ("Yield by Season",                   "yield_by_season.png"),
        ("Actual vs Predicted",               "actual_vs_predicted.png"),
        ("XGBoost Feature Importance",        "feature_importance.png"),
    ]:
        fp = FIGS / fname
        if fp.exists():
            st.image(str(fp), caption=label, use_container_width=True)

# ── Tab 2: Maps ───────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Choropleth Maps — India District Level")
    for label, fname in [
        ("Average Rice Yield per District (1990–2015)", "choropleth_yield.html"),
        ("Rice Yield Trend per District (OLS slope)",   "choropleth_trend.html"),
    ]:
        fp = FIGS / fname
        if fp.exists():
            st.markdown(f"**{label}**")
            with open(fp) as f:
                html = f.read()
            st.components.v1.html(html, height=580, scrolling=True)
        else:
            st.warning(f"{fname} not found. Ensure geospatial section ran successfully.")

# ── Tab 3: SHAP ───────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Global SHAP Explainability")
    for label, fname in [
        ("Global Feature Importance (Beeswarm)",  "shap/shap_beeswarm.png"),
        ("Waterfall — Worst Prediction",           "shap/shap_waterfall_worst.png"),
        ("Waterfall — Best Prediction",            "shap/shap_waterfall_best.png"),
    ]:
        fp = FIGS / fname
        if fp.exists():
            st.image(str(fp), caption=label, use_container_width=True)

    for feat in meta.get("top2_shap", []):
        safe = feat.replace("/","_").replace(" ","_").replace("(","").replace(")","")
        fp = FIGS / f"shap/shap_dependence_{safe}.png"
        if fp.exists():
            st.image(str(fp), caption=f"SHAP Dependence — {feat}", use_container_width=True)

    st.markdown("---")
    st.subheader(f"Live SHAP Waterfall — {sel_dist} (most recent year)")

    if not sel_pred.empty:
        row_feat = [c for c in FEAT if c in pred_df.columns]
        if row_feat:
            latest_row = sel_pred.sort_values(YC, ascending=False).iloc[0]
            X_row = latest_row[row_feat].values.reshape(1, -1)
            explainer = shap.TreeExplainer(model)
            sv = explainer(X_row)
            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(sv[0], show=False, max_display=15)
            plt.title(f"SHAP Waterfall — {sel_dist} ({int(latest_row[YC])})", fontsize=11)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
        else:
            st.info("Feature columns not available in prediction file.")
    else:
        st.info("No predictions found for this district (may be in training set).")
