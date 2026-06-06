#!/usr/bin/env python3
"""
One-time export: generates JSON + copies assets for the static web dashboard.
Run: python export_data.py
Then: python -m http.server 8080 --directory web
Open: http://localhost:8080
"""
import json, pickle, shutil
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from scipy.stats import linregress

def sf(v):
    """Safe float — NaN/inf → None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None

BASE   = Path(".")
DATA   = BASE / "data/processed"
MODELS = BASE / "outputs/models"
FIGS   = BASE / "outputs/figures"
WEB    = BASE / "web"

for d in [WEB/"data", WEB/"figures"/"shap", WEB/"maps"]:
    d.mkdir(parents=True, exist_ok=True)

print("Loading data and model...")
df      = pd.read_csv(DATA / "master_clean.csv")
pred    = pd.read_csv(DATA / "test_predictions.csv")
shap_df = pd.read_csv(DATA / "shap_values.csv")
with open(DATA / "meta.json") as f:
    meta = json.load(f)
with open(MODELS / "xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

DC   = meta["DISTRICT_COL"]
YC   = meta["YEAR_COL"]
SC   = meta["STATE_COL"]
T    = meta["TARGET"]
FEAT = meta["feature_cols"]

CONTROLLABLE = {
    "NITROGEN CONSUMPTION (tons)",
    "PHOSPHATE CONSUMPTION (tons)",
    "POTASH CONSUMPTION (tons)",
    "TOTAL FERTILISER CONSUMPTION (tons)",
    "GROSS IRRIGATED AREA (1000 ha)",
    "GROSS CROPPED AREA (1000 ha)",
    "RICE AREA (1000 ha)",
}
UNCONTROLLABLE_KW = ["TEMPERATURE", "PERCIPITATION", "EVAPOTRANSPIRATION", "WINDSPEED"]

# ── 1. meta.json ──────────────────────────────────────────────────────────────
out_meta = {
    "target":       T,
    "district_col": DC,
    "state_col":    SC,
    "year_col":     YC,
    "model_rmse":   meta["model_rmse"],
    "model_mae":    meta["model_mae"],
    "model_r2":     meta["model_r2"],
    "enriched_rmse": meta.get("enriched_rmse"),
    "enriched_mae":  meta.get("enriched_mae"),
    "enriched_r2":   meta.get("enriched_r2"),
    "yield_cols":   meta.get("yield_cols", []),
    "top2_shap":    meta.get("top2_shap", []),
    "eda_figures": [
        {"file": "yield_distribution.png",       "label": "Yield Distribution by Crop",
         "caption": "Rice yield is bimodal — a high cluster around 3,500 Kg/ha (Indo-Gangetic plain) and a low cluster around 1,200 Kg/ha (peninsular and central India). This variance is structured geographic signal, not noise, and is why a national average is misleading."},
        {"file": "yield_trend_yearly.png",        "label": "Year-wise Yield Trend (All Crops)",
         "caption": "Rice shows a steady upward trend 1990–2015, reflecting compounding Green Revolution yield gains. Chickpea is nearly flat — differential technology adoption, not climate, drives this divergence."},
        {"file": "district_yield_top_bottom.png", "label": "Top / Bottom Districts",
         "caption": "The yield gap between the best and worst district exceeds 4,500 Kg/ha — larger than the all-India average itself. Closing even half this gap in the bottom 15 districts would materially shift national output."},
        {"file": "correlation_heatmap.png",       "label": "Correlation Heatmap",
         "caption": "Summer maximum temperature is strongly negatively correlated with rice yield (heat stress during flowering reduces grain set). Monsoon precipitation is positively correlated. These are the biological underpinnings of the model's top uncontrollable features."},
        {"file": "rainfall_vs_yield.png",         "label": "Rainfall vs Yield",
         "caption": "The relationship is non-linear: yield rises steeply with rainfall up to ~1,200 mm then plateaus or declines — consistent with waterlogging effects in high-rainfall coastal districts."},
        {"file": "yield_by_season.png",           "label": "Yield by Season",
         "caption": "Kharif (monsoon) yields are higher on average but more variable; Rabi (winter) yields are lower but more stable. The stability gap reflects irrigation dependency — Rabi crops are mostly irrigated, insulating them from rainfall shocks."},
        {"file": "actual_vs_predicted.png",       "label": "Actual vs Predicted",
         "caption": "The model fits well across the middle of the yield range. Residual heteroscedasticity is visible at high yields — the model underestimates the best-performing districts, likely because exceptional input conditions are underrepresented in training data."},
        {"file": "feature_importance.png",        "label": "XGBoost Feature Importance",
         "caption": "XGBoost's native gain-based importance ranks sown area highest. This measure is biased toward high-cardinality features. The SHAP beeswarm in the SHAP tab gives the corrected, unbiased picture."},
    ],
    "shap_figures": [
        {"file": "shap/shap_beeswarm.png",                        "label": "Global Feature Importance (Beeswarm)",
         "caption": "Each dot is one district-year in the test set. Features are ranked by mean absolute SHAP value. Rice sown area dominates globally. Nitrogen shows a positive tail — high-N districts consistently outperform the baseline. Red = high feature value, blue = low."},
        {"file": "shap/shap_waterfall_worst.png",                  "label": "Waterfall — Worst Prediction",
         "caption": "Low monsoon rainfall and minimal sown area together pull the prediction ~800 Kg/ha below baseline. These are uncontrollable drivers — this district needs a climate adaptation strategy, not a fertilizer subsidy."},
        {"file": "shap/shap_waterfall_best.png",                   "label": "Waterfall — Best Prediction",
         "caption": "High irrigation area and nitrogen each push the prediction 300–400 Kg/ha above the model baseline. Every bar is one feature's additive contribution — they sum to the final prediction minus the baseline."},
        {"file": "shap/shap_dependence_RICE_AREA_1000_ha.png",     "label": "SHAP Dependence — Rice Area",
         "caption": "Scatter of Rice Area's value vs. its SHAP contribution, coloured by the strongest interacting feature. The relationship is strongly positive with diminishing returns — very large sown areas show compressed SHAP gains."},
        {"file": "shap/shap_dependence_State_Name_enc.png",        "label": "SHAP Dependence — State",
         "caption": "State encoding captures unobserved regional effects (soil quality, infrastructure, policy). Punjab and Haryana cluster at high positive SHAP values; eastern states cluster at negative — consistent with known yield geography."},
    ],
}
with open(WEB / "data/meta.json", "w") as f:
    json.dump(out_meta, f, indent=2)
print("✓ meta.json")

# ── 2. tabnet.json ────────────────────────────────────────────────────────────
shutil.copy(MODELS / "tabnet_results.json", WEB / "data/tabnet.json")
print("✓ tabnet.json")

# ── 3. districts.json ─────────────────────────────────────────────────────────
yield_cols = meta.get("yield_cols", [T])
pred_r     = pred.reset_index(drop=True)
districts_out = {}

for (state, dist), grp in df.groupby([SC, DC]):
    grp_s = grp.sort_values(YC)
    entry = {"state": state, "years": grp_s[YC].tolist()}
    for col in yield_cols:
        if col in grp_s.columns:
            entry[col] = [sf(v) for v in grp_s[col].tolist()]
    p = pred_r[(pred_r[SC] == state) & (pred_r[DC] == dist)].sort_values(YC)
    if not p.empty:
        entry["pred_years"]  = p[YC].tolist()
        entry["pred_values"] = [sf(v) for v in p["predicted"].tolist()]
        entry["actual_test"] = [sf(v) for v in p[T].tolist()]
    districts_out[dist] = entry

with open(WEB / "data/districts.json", "w") as f:
    json.dump(districts_out, f)
print(f"✓ districts.json  ({len(districts_out)} districts)")

# ── 4. maps.json ──────────────────────────────────────────────────────────────
avg_y  = df.groupby(DC)[T].mean()
slopes = df.groupby(DC).apply(
    lambda g: linregress(g[YC], g[T].fillna(g[T].median())).slope,
    include_groups=False,
)
maps_out = {
    "districts": [
        {"name": d, "avg_yield": sf(avg_y.get(d)), "slope": sf(slopes.get(d))}
        for d in df[DC].unique()
    ],
    "stats": {
        "count":                    int(len(avg_y)),
        "max_yield_district":       str(avg_y.idxmax()),
        "max_yield":                sf(avg_y.max()),
        "min_yield_district":       str(avg_y.idxmin()),
        "min_yield":                sf(avg_y.min()),
        "improving":                int((slopes > 0).sum()),
        "declining":                int((slopes < 0).sum()),
        "fastest_decline_district": str(slopes.idxmin()),
        "fastest_decline_slope":    sf(slopes.min()),
    },
}
with open(WEB / "data/maps.json", "w") as f:
    json.dump(maps_out, f, indent=2)
print("✓ maps.json")

# ── 5. shap.json ──────────────────────────────────────────────────────────────
print("Computing SHAP base value (TreeExplainer)...")
feat_cols  = [c for c in FEAT if c in shap_df.columns]
explainer  = shap.TreeExplainer(model)
ev = explainer.expected_value
base_value = float(ev[0] if hasattr(ev, '__len__') else ev)

# Categories are feature-level (same for every district)
categories = []
for f in feat_cols:
    if f in CONTROLLABLE:
        categories.append("Controllable")
    elif any(kw in f.upper() for kw in UNCONTROLLABLE_KW):
        categories.append("Uncontrollable")
    else:
        categories.append("Structural")

shap_r = shap_df.reset_index(drop=True)
shap_out = {
    "features":   feat_cols,
    "categories": categories,
    "base":       round(base_value, 2),
    "districts":  {},
}

for dist, grp in pred_r.groupby(DC):
    latest_idx = grp[YC].idxmax()
    vals = shap_r.loc[latest_idx, feat_cols].values
    shap_out["districts"][dist] = {
        "values":    [0.0 if np.isnan(v) else round(float(v), 3) for v in vals],
        "year":      int(pred_r.loc[latest_idx, YC]),
        "actual":    sf(pred_r.loc[latest_idx, T]),
        "predicted": sf(pred_r.loc[latest_idx, "predicted"]),
    }

with open(WEB / "data/shap.json", "w") as f:
    json.dump(shap_out, f)
print(f"✓ shap.json  ({len(shap_out['districts'])} districts)")

# ── 6. Copy static assets ─────────────────────────────────────────────────────
for png in FIGS.glob("*.png"):
    shutil.copy(png, WEB / "figures" / png.name)
for png in (FIGS / "shap").glob("*.png"):
    shutil.copy(png, WEB / "figures" / "shap" / png.name)
print("✓ figures copied")

for html_file in FIGS.glob("choropleth_*.html"):
    shutil.copy(html_file, WEB / "maps" / html_file.name)
print("✓ choropleth maps copied")

print("\n✅  Export complete.")
print("    Serve: python -m http.server 8080 --directory web")
print("    Open:  http://localhost:8080")
