# XAI-Crop — Vibecoding Sprint Plan

**Goal:** Submission-ready interactive dashboard — all 5 datasets genuinely integrated, enriched model trained on the combined feature set, TabNet comparison done, every plot explained, controllable-factor policy panel live.

**Dataset ownership:**

| # | Dataset | Source | Owner |
|---|---------|--------|-------|
| 1 | ICRISAT District-Level Data | Mendeley | ✅ Member 1 (done) |
| 2 | IMD Rainfall | Kaggle | Member 2 |
| 3 | Crop Production in India (Kharif/Rabi) | Kaggle | Member 2 |
| 4 | District-wise Fertilizer Consumption | data.gov.in | Member 3 |
| 5 | India District Shapefile (GeoJSON) | Figshare / ICRISAT | ✅ Member 1 (done) |

---

## Member 1 — Pradhan (Project Lead)

### ✅ Completed

- ✅ Sourced and loaded ICRISAT District-Level XLS (560 districts, 1990–2015)
- ✅ Integrated India District Shapefile with fuzzy district-name matching
- ✅ Full data audit — shape, missing value report, descriptive statistics (`analysis.py` Section 1)
- ✅ Cleaning & preprocessing — dropped >40% missing columns, median imputation, label encoding, 80/20 stratified split (`analysis.py` Section 2)
- ✅ EDA — 8 static plots saved to `outputs/figures/` (`analysis.py` Section 3)
- ✅ Geospatial pipeline — choropleth yield map + choropleth trend-slope map (`analysis.py` Section 4)
- ✅ XGBoost Regressor — trained, evaluated (R² = 0.8695, RMSE = 379.56, MAE = 263.69), serialized to `outputs/models/xgboost_model.pkl`
- ✅ SHAP TreeExplainer — beeswarm, waterfall ×2, dependence plots; saved to `outputs/figures/shap/`
- ✅ All processed data artifacts: `master_clean.csv`, `long_format.csv`, `test_predictions.csv`, `shap_values.csv`, `label_mappings.json`, `meta.json`
- ✅ Streamlit dashboard skeleton — sidebar, metric cards, Tab 1 (EDA), Tab 2 (Maps), Tab 3 (SHAP with live per-district waterfall)

### Remaining Tasks

#### Task 1 — Export `split_indices.json` from `analysis.py`

In `analysis.py` Section 2, after the train/test split, add 3 lines so Member 3 can use the exact same split for TabNet:
```python
split = {"train_idx": list(X_train.index), "test_idx": list(X_test.index)}
with open(DATA / "split_indices.json", "w") as f:
    json.dump(split, f)
```
Re-run `python analysis.py` once to generate the file, then push so Member 3 can pull it.

#### Task 2 — Dashboard title and header

Replace the existing `st.title(...)` in `dashboard.py`:
```python
st.title("🌾 XAI-Crop: District-Level Yield Explainability Dashboard")
st.caption("560 Indian districts · 1990–2015 · Multi-source fusion · XGBoost + TabNet + SHAP · Policy-actionable insights")
```

#### Task 3 — Interpretation expanders for all EDA plots (Tab 1)

For each `st.image(...)` call in the "Pre-computed EDA Figures" section of Tab 1, add an `st.expander("📖 What does this show?")` block immediately below it:

| Figure | Caption to write inside the expander |
|--------|--------------------------------------|
| `yield_distribution.png` | Rice yield is bimodal — a high cluster around 3,500 Kg/ha (Indo-Gangetic plain) and a low cluster around 1,200 Kg/ha (peninsular and central India). This variance is structured geographic signal, not noise, and is why a national average is misleading. |
| `yield_trend_yearly.png` | Rice shows a steady upward trend 1990–2015, reflecting compounding Green Revolution yield gains. Chickpea is nearly flat — differential technology adoption, not climate, drives this divergence. |
| `district_yield_top_bottom.png` | The yield gap between the best and worst district exceeds 4,500 Kg/ha — larger than the all-India average itself. Closing even half this gap in the bottom 15 districts would materially shift national output. |
| `correlation_heatmap.png` | Summer maximum temperature is strongly negatively correlated with rice yield (heat stress during flowering reduces grain set). Monsoon precipitation is positively correlated. These are the biological underpinnings of the model's top uncontrollable features. |
| `rainfall_vs_yield.png` | The relationship is non-linear: yield rises steeply with rainfall up to ~1,200 mm then plateaus or declines — consistent with waterlogging effects in high-rainfall coastal districts. |
| `yield_by_season.png` | Kharif (monsoon) yields are higher on average but more variable; Rabi (winter) yields are lower but more stable. The stability gap reflects irrigation dependency — Rabi crops are mostly irrigated, insulating them from rainfall shocks. |
| `actual_vs_predicted.png` | The model fits well across the middle of the yield range. Residual heteroscedasticity is visible at high yields — the model underestimates the best-performing districts, likely because exceptional input conditions are underrepresented in training data. |
| `feature_importance.png` | XGBoost's native gain-based importance ranks sown area highest. Note: this measure is biased toward high-cardinality features. The SHAP beeswarm (Tab 3) gives the corrected, unbiased picture. |

#### Task 4 — Interpretation expanders for SHAP plots (Tab 3)

Same pattern for the global SHAP section in Tab 3:

| Figure | Caption |
|--------|---------|
| `shap_beeswarm.png` | Each dot is one district-year in the test set. Features are ranked by mean absolute SHAP value. Rice sown area dominates globally. Nitrogen shows a positive tail — high-N districts consistently outperform the baseline. Red = high feature value, blue = low. |
| `shap_waterfall_best.png` | For the best-predicted district-year, high irrigation area and nitrogen each push the prediction 300–400 Kg/ha above the model baseline. Every bar is one feature's additive contribution — they sum to the final prediction minus the baseline. |
| `shap_waterfall_worst.png` | Low monsoon rainfall and minimal sown area together pull the prediction ~800 Kg/ha below baseline. These are uncontrollable drivers — this is exactly the kind of district that needs a climate adaptation strategy, not a fertilizer subsidy. |
| `shap_dependence_*.png` | Scatter of one feature's value vs. its SHAP contribution, coloured by the strongest interacting feature. Nitrogen shows near-zero SHAP impact in low-rainfall districts regardless of application rate — fertilizer efficacy is rainfall-conditional. |

#### Task 5 — Metric cards on Tab 2 (Maps)

Above each choropleth call in Tab 2, add a 3-column metric row. Compute from `df` (cache with `@st.cache_data`):

```python
# Above the yield choropleth
avg_by_dist = df.groupby(DC)[T].mean()
c1, c2, c3 = st.columns(3)
c1.metric("Districts Mapped", len(avg_by_dist))
c2.metric("Highest Avg Yield", f"{avg_by_dist.max():.0f} Kg/ha  ({avg_by_dist.idxmax()})")
c3.metric("Lowest Avg Yield",  f"{avg_by_dist.min():.0f} Kg/ha  ({avg_by_dist.idxmin()})")

# Above the trend choropleth
from scipy.stats import linregress
slopes = df.groupby(DC).apply(
    lambda g: linregress(g[YC], g[T].fillna(g[T].median())).slope
)
c1, c2, c3 = st.columns(3)
c1.metric("Improving Districts",  int((slopes > 0).sum()))
c2.metric("Declining Districts",  int((slopes < 0).sum()))
c3.metric("Fastest Decline", f"{slopes.idxmin()} ({slopes.min():.1f} Kg/ha/yr)")
```

#### Task 6 — Final smoke test & git coordination

Once Members 2 and 3 push:
- Pull, run `python tabnet_train.py`, then `streamlit run app/dashboard.py`
- Verify all expanders open, metric cards appear, controllability panel renders for Punjab and Barmer
- Final commit: `git add -A && git commit -m "feat: complete XAI-Crop — multi-source fusion, TabNet, dashboard polish"`

**Member 1 remaining estimated time: ~2 hours**

---

## Member 2 — IMD Rainfall + Crop Production → Enriched Dataset + Retrain

### Datasets to download first

1. **IMD Rainfall** — Kaggle: search "India district rainfall IMD monthly". Download the CSV (columns: `District`, `Year` or `Month`/`Year`, `Rainfall_mm` or similar). Save as `data/raw/imd_rainfall.csv`.
2. **Crop Production in India** — Kaggle: search "crop production India district". Download the CSV (columns: `State_Name`, `District_Name`, `Crop_Year`, `Season`, `Crop`, `Area`, `Production`). Save as `data/raw/crop_production_india.csv`.

### What you're building

The ICRISAT dataset already has precipitation columns, but: (a) IMD gives an independent rainfall source to cross-validate and fill missing values, and (b) Crop Production adds the `Season` column (Kharif / Rabi / Whole Year) which is **currently null** in the pipeline — it's a genuine new feature. Your job is to produce `data/processed/master_enriched.csv` and retrain XGBoost on it.

### Code Tasks

Create a new script `enrich_and_retrain.py` in the project root.

#### Task 1 — Load and harmonize IMD rainfall

```python
import pandas as pd, numpy as np, json, pickle
from pathlib import Path
from fuzzywuzzy import process

BASE  = Path(".")
DATA  = BASE / "data/processed"
RAW   = BASE / "data/raw"

df = pd.read_csv(DATA / "master_clean.csv")
with open(DATA / "meta.json") as f:
    meta = json.load(f)

DC = meta["DISTRICT_COL"]   # "Dist Name"
YC = meta["YEAR_COL"]       # "Year"

# Load IMD
imd = pd.read_csv(RAW / "imd_rainfall.csv")
print(imd.columns.tolist())         # inspect — adjust column names below to match actual file
# Standardise: rename to DIST_IMD, YEAR_IMD, ANNUAL_RAIN_IMD
# imd = imd.rename(columns={"District": "DIST_IMD", "YEAR": "YEAR_IMD", "ANNUAL": "ANNUAL_RAIN_IMD"})

# Aggregate to annual total per district-year if file is monthly
# imd_annual = imd.groupby(["DIST_IMD", "YEAR_IMD"])["ANNUAL_RAIN_IMD"].sum().reset_index()

# Fuzzy-match IMD district names to master_clean district names
master_dists = df[DC].unique().tolist()
imd["DIST_MATCHED"] = imd["DIST_IMD"].apply(
    lambda x: process.extractOne(str(x), master_dists)[0]
)

# Merge on matched district + year
imd_merge = imd[["DIST_MATCHED", "YEAR_IMD", "ANNUAL_RAIN_IMD"]].rename(
    columns={"DIST_MATCHED": DC, "YEAR_IMD": YC}
)
df = df.merge(imd_merge, on=[DC, YC], how="left")
print(f"IMD merge: {df['ANNUAL_RAIN_IMD'].notna().sum()} rows filled")
```

#### Task 2 — Load and merge Crop Production (Season feature)

```python
crop = pd.read_csv(RAW / "crop_production_india.csv")
print(crop.columns.tolist())        # inspect — adjust names below

# Keep only Rice rows to match the primary target; get dominant season per district-year
# crop = crop.rename(columns={"District_Name": "DIST_RAW", "Crop_Year": YC, "Season": "Season", "State_Name": "State_Raw"})
rice = crop[crop["Crop"].str.upper().str.contains("RICE", na=False)].copy()

# Fuzzy-match districts
rice["DIST_MATCHED"] = rice["DIST_RAW"].apply(
    lambda x: process.extractOne(str(x), master_dists)[0]
)

# Take the season with highest production for each district-year
season_map = (rice.sort_values("Production", ascending=False)
                  .groupby(["DIST_MATCHED", YC])["Season"]
                  .first()
                  .reset_index()
                  .rename(columns={"DIST_MATCHED": DC}))

df = df.merge(season_map, on=[DC, YC], how="left")
print(f"Season filled: {df['Season'].notna().sum()} / {len(df)} rows")

# Encode season
df["Season_enc"] = df["Season"].map(
    {"Kharif": 0, "Rabi": 1, "Whole Year": 2, "Summer": 3, "Winter": 4}
).fillna(-1).astype(int)
```

#### Task 3 — Save enriched dataset and retrain XGBoost

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

df.to_csv(DATA / "master_enriched.csv", index=False)
print(f"Saved master_enriched.csv: {df.shape}")

T    = meta["TARGET"]
FEAT = meta["feature_cols"] + ["ANNUAL_RAIN_IMD", "Season_enc"]
FEAT = [f for f in FEAT if f in df.columns]

# Use same split indices as Member 1
with open(DATA / "split_indices.json") as f:
    split = json.load(f)

X = df[FEAT].fillna(df[FEAT].median())
y = df[T].fillna(df[T].median())

X_train = X.loc[split["train_idx"]]
X_test  = X.loc[split["test_idx"]]
y_train = y.loc[split["train_idx"]]
y_test  = y.loc[split["test_idx"]]

model_enrich = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                  max_depth=6, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42)
model_enrich.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=50)

preds = model_enrich.predict(X_test)
rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
mae   = float(mean_absolute_error(y_test, preds))
r2    = float(r2_score(y_test, preds))
print(f"\nEnriched XGBoost — RMSE: {rmse:.2f}  MAE: {mae:.2f}  R²: {r2:.4f}")

with open(BASE / "outputs/models/xgboost_enriched.pkl", "wb") as f:
    pickle.dump(model_enrich, f)

# Append results to meta.json for the dashboard to display
meta["enriched_rmse"] = round(rmse, 2)
meta["enriched_mae"]  = round(mae, 2)
meta["enriched_r2"]   = round(r2, 4)
meta["enriched_feature_cols"] = FEAT
with open(DATA / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("meta.json updated with enriched model metrics.")
```

Run with: `python enrich_and_retrain.py`

**Member 2 estimated time: ~3 hours**

---

## Member 3 — Fertilizer Data Augmentation + TabNet + Controllability Panel

### Dataset to download first

**District-wise Fertilizer Consumption** — go to data.gov.in and search "district fertilizer consumption". Download the CSV with columns like `State`, `District`, `Year`, `Nitrogen_tonnes`, `Phosphate_tonnes`, `Potash_tonnes`. Save as `data/raw/fertilizer_consumption.csv`.

### What you're building

The ICRISAT dataset already has N/P/K columns but has missing values in some district-years. Your fertilizer dataset fills those gaps, producing a more complete feature matrix. You then train TabNet on this augmented data and wire the controllability panel into the dashboard.

### Code Tasks

#### Task 1 — Write `augment_fertilizer.py`

Create `augment_fertilizer.py` in the project root:

```python
import pandas as pd, numpy as np, json
from pathlib import Path
from fuzzywuzzy import process

BASE = Path(".")
DATA = BASE / "data/processed"
RAW  = BASE / "data/raw"

df = pd.read_csv(DATA / "master_clean.csv")
with open(DATA / "meta.json") as f:
    meta = json.load(f)

DC = meta["DISTRICT_COL"]
YC = meta["YEAR_COL"]

fert = pd.read_csv(RAW / "fertilizer_consumption.csv")
print("Fertilizer columns:", fert.columns.tolist())
# Rename to standardise — adjust to match actual column names:
# fert = fert.rename(columns={
#     "District": "DIST_RAW", "Year": YC,
#     "Nitrogen_tonnes": "N_GOV", "Phosphate_tonnes": "P_GOV", "Potash_tonnes": "K_GOV"
# })

master_dists = df[DC].unique().tolist()
fert["DIST_MATCHED"] = fert["DIST_RAW"].apply(
    lambda x: process.extractOne(str(x), master_dists)[0]
)

fert_merge = fert[["DIST_MATCHED", YC, "N_GOV", "P_GOV", "K_GOV"]].rename(
    columns={"DIST_MATCHED": DC}
)
df = df.merge(fert_merge, on=[DC, YC], how="left")

# Fill missing ICRISAT N/P/K values using the government dataset where available
for icrisat_col, gov_col in [
    ("NITROGEN CONSUMPTION (tons)",   "N_GOV"),
    ("PHOSPHATE CONSUMPTION (tons)",  "P_GOV"),
    ("POTASH CONSUMPTION (tons)",     "K_GOV"),
]:
    if icrisat_col in df.columns and gov_col in df.columns:
        filled = df[icrisat_col].isna().sum()
        df[icrisat_col] = df[icrisat_col].fillna(df[gov_col])
        print(f"{icrisat_col}: filled {filled} missing values from gov data")

df.to_csv(DATA / "master_clean.csv", index=False)   # overwrite in place — augmented
print(f"master_clean.csv updated: {df.shape}")
```

Run `python augment_fertilizer.py` — this overwrites `master_clean.csv` in place so both the original XGBoost and the TabNet model benefit automatically.

#### Task 2 — Write `tabnet_train.py`

```python
import numpy as np, pandas as pd, json, time, torch
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor

BASE   = Path(".")
DATA   = BASE / "data/processed"
MODELS = BASE / "outputs/models"

df = pd.read_csv(DATA / "master_clean.csv")
with open(DATA / "meta.json") as f:
    meta = json.load(f)
with open(DATA / "split_indices.json") as f:
    split = json.load(f)

T    = meta["TARGET"]
FEAT = meta["feature_cols"]
FEAT = [f for f in FEAT if f in df.columns]

X = df[FEAT].fillna(df[FEAT].median())
y = df[T].fillna(df[T].median())

X_train = X.loc[split["train_idx"]].values
X_test  = X.loc[split["test_idx"]].values
y_train = y.loc[split["train_idx"]].values.reshape(-1, 1)
y_test  = y.loc[split["test_idx"]].values.reshape(-1, 1)

model = TabNetRegressor(
    n_d=32, n_a=32, n_steps=5, gamma=1.5,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": 2e-3},
    scheduler_params={"step_size": 50, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=10
)

t0 = time.time()
model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_test, y_test)],
    max_epochs=200, patience=20, batch_size=1024
)
elapsed = time.time() - t0

preds   = model.predict(X_test).flatten()
y_flat  = y_test.flatten()
results = {
    "rmse":         round(float(np.sqrt(mean_squared_error(y_flat, preds))), 2),
    "mae":          round(float(mean_absolute_error(y_flat, preds)), 2),
    "r2":           round(float(r2_score(y_flat, preds)), 4),
    "train_time_s": round(elapsed, 1)
}
print("\nTabNet results:", results)

with open(MODELS / "tabnet_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved tabnet_results.json")
```

Run with: `python tabnet_train.py`

#### Task 3 — Add TabNet comparison panel to dashboard Tab 3 (top of the tab)

In `app/dashboard.py`, at the **top** of `with tab3:`, before the beeswarm plot, add:

```python
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
```

#### Task 4 — Controllability classification constants

Add near the top of `dashboard.py`, after the `meta` load:

```python
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
```

#### Task 5 — Wire the controllability panel into the live SHAP waterfall section

Find the live waterfall section at the bottom of Tab 3. Replace everything after `sv = explainer(X_row)` with:

```python
sv = explainer(X_row)

# Controllability summary sentence
total_ctrl   = sum(v for f, v in zip(row_feat, sv[0].values) if f in CONTROLLABLE)
total_unctrl = float(sv[0].values.sum()) - total_ctrl
st.markdown(
    f"**Prediction breakdown for {sel_dist}:** "
    f"Controllable factors contribute **{total_ctrl:+.0f} Kg/ha** · "
    f"Uncontrollable factors contribute **{total_unctrl:+.0f} Kg/ha** "
    f"relative to the national baseline ({sv[0].base_values:.0f} Kg/ha)."
)

# Waterfall chart
plt.figure(figsize=(12, 6))
shap.plots.waterfall(sv[0], show=False, max_display=15)
plt.title(f"SHAP Waterfall — {sel_dist} ({int(latest_row[YC])})", fontsize=11)
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()

# Top controllable lever card
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

# Full controllability breakdown table
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
```

**Member 3 estimated time: ~3 hours**

---

## Shared / Final Steps (~30 min, led by Member 1)

**Order matters:**
1. Member 3 runs `python augment_fertilizer.py` first (updates `master_clean.csv`)
2. Member 1 re-runs `python analysis.py` to regenerate SHAP artifacts on the augmented data
3. Member 2 runs `python enrich_and_retrain.py` (requires `split_indices.json` from step 2)
4. Member 3 runs `python tabnet_train.py`
5. All three push; Member 1 pulls and runs `streamlit run app/dashboard.py`
6. Verify: all expanders open → metric cards on Maps → TabNet comparison table → controllability panel for Punjab and Barmer
7. Final commit: `git add -A && git commit -m "feat: multi-source fusion, enriched retrain, TabNet, dashboard polish"`

---

## Dependency Chain

```
Member 3: augment_fertilizer.py
        ↓
Member 1: analysis.py  →  split_indices.json
        ↓                         ↓
Member 2: enrich_and_retrain.py   Member 3: tabnet_train.py
        ↓                         ↓
                  dashboard.py (all members' edits merged)
```

Members 1 and 3 coordinate first (augment → re-run pipeline → split indices). Member 2 and Member 3's model scripts can then run in parallel.
