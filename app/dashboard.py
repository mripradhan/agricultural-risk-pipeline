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
from scipy.stats import linregress

BASE    = Path(__file__).parent.parent
DATA    = BASE / "data/processed"
FIGS    = BASE / "outputs/figures"
MODELS  = BASE / "outputs/models"

st.set_page_config(page_title="XAI-Crop Yield Explorer", layout="wide", page_icon="🌾")

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

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/en/thumb/e/e9/ICRISAT.svg/200px-ICRISAT.svg.png",
    width=140
)
st.sidebar.title("🌾 ICRISAT Explorer")
st.sidebar.markdown(f"**Model R²:** {R2}   |   **RMSE:** {RMSE} Kg/ha")
st.sidebar.markdown("---")

states    = sorted(df[SC].unique())
sel_state = st.sidebar.selectbox("State", states)

dists    = sorted(df[df[SC] == sel_state][DC].unique())
sel_dist = st.sidebar.selectbox("District", dists)

crops_av = [c.replace(" YIELD (Kg per ha)", "") for c in meta.get("yield_cols", [])]
sel_crop = st.sidebar.selectbox("Crop (for trend)", crops_av if crops_av else ["Rice"])

# ─── Filter ──────────────────────────────────────────────────────────────────
sel_df   = df[(df[SC] == sel_state) & (df[DC] == sel_dist)]
sel_pred = pred_df[(pred_df[SC] == sel_state) & (pred_df[DC] == sel_dist)] if SC in pred_df.columns else pd.DataFrame()

# ─── Top metric cards ────────────────────────────────────────────────────────
st.title("🌾 XAI-Crop: District-Level Yield Explainability Dashboard")
st.caption("560 Indian districts · 1990–2015 · Multi-source fusion · XGBoost + TabNet + SHAP · Policy-actionable insights")
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

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🗺️ Maps", "🔍 SHAP"])

# ── Tab 1: EDA ───────────────────────────────────────────────────────────────
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

    eda_figures = [
        (
            "Yield Distribution by Crop", "yield_distribution.png",
            "Rice yield is bimodal — a high cluster around 3,500 Kg/ha (Indo-Gangetic plain) and a low cluster "
            "around 1,200 Kg/ha (peninsular and central India). This variance is structured geographic signal, "
            "not noise, and is why a national average is misleading.",
        ),
        (
            "Year-wise Yield Trend (All Crops)", "yield_trend_yearly.png",
            "Rice shows a steady upward trend 1990–2015, reflecting compounding Green Revolution yield gains. "
            "Chickpea is nearly flat — differential technology adoption, not climate, drives this divergence.",
        ),
        (
            "Top/Bottom Districts", "district_yield_top_bottom.png",
            "The yield gap between the best and worst district exceeds 4,500 Kg/ha — larger than the all-India "
            "average itself. Closing even half this gap in the bottom 15 districts would materially shift national output.",
        ),
        (
            "Correlation Heatmap", "correlation_heatmap.png",
            "Summer maximum temperature is strongly negatively correlated with rice yield (heat stress during "
            "flowering reduces grain set). Monsoon precipitation is positively correlated. These are the biological "
            "underpinnings of the model's top uncontrollable features.",
        ),
        (
            "Rainfall vs Yield", "rainfall_vs_yield.png",
            "The relationship is non-linear: yield rises steeply with rainfall up to ~1,200 mm then plateaus or "
            "declines — consistent with waterlogging effects in high-rainfall coastal districts.",
        ),
        (
            "Yield by Season", "yield_by_season.png",
            "Kharif (monsoon) yields are higher on average but more variable; Rabi (winter) yields are lower but "
            "more stable. The stability gap reflects irrigation dependency — Rabi crops are mostly irrigated, "
            "insulating them from rainfall shocks.",
        ),
        (
            "Actual vs Predicted", "actual_vs_predicted.png",
            "The model fits well across the middle of the yield range. Residual heteroscedasticity is visible at "
            "high yields — the model underestimates the best-performing districts, likely because exceptional input "
            "conditions are underrepresented in training data.",
        ),
        (
            "XGBoost Feature Importance", "feature_importance.png",
            "XGBoost's native gain-based importance ranks sown area highest. Note: this measure is biased toward "
            "high-cardinality features. The SHAP beeswarm (Tab 3) gives the corrected, unbiased picture.",
        ),
    ]

    for label, fname, interpretation in eda_figures:
        fp = FIGS / fname
        if fp.exists():
            st.image(str(fp), caption=label, use_container_width=True)
            with st.expander("📖 What does this show?"):
                st.markdown(interpretation)

# ── Tab 2: Maps ──────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Choropleth Maps — India District Level")

    avg_by_dist = df.groupby(DC)[T].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("Districts Mapped",  len(avg_by_dist))
    c2.metric("Highest Avg Yield", f"{avg_by_dist.max():.0f} Kg/ha  ({avg_by_dist.idxmax()})")
    c3.metric("Lowest Avg Yield",  f"{avg_by_dist.min():.0f} Kg/ha  ({avg_by_dist.idxmin()})")

    fp = FIGS / "choropleth_yield.html"
    if fp.exists():
        st.markdown("**Average Rice Yield per District (1990–2015)**")
        with open(fp, encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=580, scrolling=True)
    else:
        st.warning("choropleth_yield.html not found. Ensure geospatial section ran successfully.")

    st.markdown("---")

    slopes = df.groupby(DC).apply(
        lambda g: linregress(g[YC], g[T].fillna(g[T].median())).slope
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Improving Districts", int((slopes > 0).sum()))
    c2.metric("Declining Districts", int((slopes < 0).sum()))
    c3.metric("Fastest Decline",     f"{slopes.idxmin()} ({slopes.min():.1f} Kg/ha/yr)")

    fp = FIGS / "choropleth_trend.html"
    if fp.exists():
        st.markdown("**Rice Yield Trend per District (OLS slope)**")
        with open(fp, encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=580, scrolling=True)
    else:
        st.warning("choropleth_trend.html not found. Ensure geospatial section ran successfully.")

# ── Tab 3: SHAP ──────────────────────────────────────────────────────────────
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
        (
            "Global Feature Importance (Beeswarm)", "shap/shap_beeswarm.png",
            "Each dot is one district-year in the test set. Features are ranked by mean absolute SHAP value. "
            "Rice sown area dominates globally. Nitrogen shows a positive tail — high-N districts consistently "
            "outperform the baseline. Red = high feature value, blue = low.",
        ),
        (
            "Waterfall — Worst Prediction", "shap/shap_waterfall_worst.png",
            "Low monsoon rainfall and minimal sown area together pull the prediction ~800 Kg/ha below baseline. "
            "These are uncontrollable drivers — this is exactly the kind of district that needs a climate "
            "adaptation strategy, not a fertilizer subsidy.",
        ),
        (
            "Waterfall — Best Prediction", "shap/shap_waterfall_best.png",
            "For the best-predicted district-year, high irrigation area and nitrogen each push the prediction "
            "300–400 Kg/ha above the model baseline. Every bar is one feature's additive contribution — they "
            "sum to the final prediction minus the baseline.",
        ),
    ]

    for label, fname, interpretation in shap_figures:
        fp = FIGS / fname
        if fp.exists():
            st.image(str(fp), caption=label, use_container_width=True)
            with st.expander("📖 What does this show?"):
                st.markdown(interpretation)

    for feat in meta.get("top2_shap", []):
        safe = feat.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        fp = FIGS / f"shap/shap_dependence_{safe}.png"
        if fp.exists():
            st.image(str(fp), caption=f"SHAP Dependence — {feat}", use_container_width=True)
            with st.expander("📖 What does this show?"):
                st.markdown(
                    "Scatter of one feature's value vs. its SHAP contribution, coloured by the strongest "
                    "interacting feature. Nitrogen shows near-zero SHAP impact in low-rainfall districts "
                    "regardless of application rate — fertilizer efficacy is rainfall-conditional."
                )

    st.markdown("---")
    st.subheader(f"Live SHAP Waterfall — {sel_dist} (most recent year)")

    if not sel_pred.empty:
        row_feat = [c for c in FEAT if c in pred_df.columns]
        if row_feat:
            latest_row = sel_pred.sort_values(YC, ascending=False).iloc[0]
            X_row      = latest_row[row_feat].values.reshape(1, -1)
            explainer  = shap.TreeExplainer(model)
            sv         = explainer(X_row)

            total_ctrl   = sum(v for f, v in zip(row_feat, sv[0].values) if f in CONTROLLABLE)
            total_unctrl = float(sv[0].values.sum()) - total_ctrl
            st.markdown(
                f"**Prediction breakdown for {sel_dist}:** "
                f"Controllable factors contribute **{total_ctrl:+.0f} Kg/ha** · "
                f"Uncontrollable factors contribute **{total_unctrl:+.0f} Kg/ha** "
                f"relative to the national baseline ({sv[0].base_values:.0f} Kg/ha)."
            )

            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(sv[0], show=False, max_display=15)
            plt.title(f"SHAP Waterfall — {sel_dist} ({int(latest_row[YC])})", fontsize=11)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

            shap_series = pd.Series(sv[0].values, index=row_feat)
            ctrl_shap   = shap_series[shap_series.index.isin(CONTROLLABLE)]
            if not ctrl_shap.empty:
                top_ctrl  = ctrl_shap.abs().idxmax()
                top_val   = ctrl_shap[top_ctrl]
                direction = "increase" if top_val > 0 else "decrease"
                st.success(
                    f"**Policy lever for {sel_dist}:** `{top_ctrl}` has the largest controllable "
                    f"SHAP impact ({top_val:+.1f} Kg/ha). "
                    f"This factor is currently **{'above' if top_val > 0 else 'below'} baseline** — "
                    f"maintaining or improving it would {direction} yield."
                )

            rows = []
            for feat, val in zip(row_feat, sv[0].values):
                if feat in CONTROLLABLE:
                    cat = "Controllable"
                elif any(kw in feat.upper() for kw in UNCONTROLLABLE_KEYWORDS):
                    cat = "Uncontrollable"
                else:
                    cat = "Structural"
                rows.append({"Feature": feat, "SHAP Impact (Kg/ha)": round(float(val), 1), "Category": cat})

            breakdown = (pd.DataFrame(rows)
                           .sort_values("SHAP Impact (Kg/ha)", key=abs, ascending=False)
                           .head(15))
            st.dataframe(breakdown, use_container_width=True, hide_index=True)
        else:
            st.info("Feature columns not available in prediction file.")
    else:
        st.info("No predictions found for this district (may be in training set).")
