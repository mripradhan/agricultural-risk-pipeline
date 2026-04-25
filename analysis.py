"""
ICRISAT District-Level Crop Yield Analysis
Wide-format dataset: district × year rows, multiple crop yield columns
End-to-end: audit → cleaning → EDA → geospatial → XGBoost → SHAP → Streamlit
"""

import os, sys, warnings, pickle, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as sc_stats

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE   = Path("/mnt/c/Users/pradh/Projects/ICRISAT District-Level Data Heterogeneous Climate Effect on Crop Yield and Associated Risks to Water Security in India")
DATA   = BASE / "data/processed"
FIGS   = BASE / "outputs/figures"
SHAPD  = FIGS / "shap"
MODELS = BASE / "outputs/models"
APPD   = BASE / "app"

for d in [DATA, FIGS, SHAPD, MODELS, APPD]:
    d.mkdir(parents=True, exist_ok=True)

XLS = BASE / "main merge (droped _merge==2) (560 dist 1990-2015).xls"

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 1 — DATA LOADING & AUDIT")
print("="*70)

df_raw = pd.read_excel(XLS, engine='xlrd')

print(f"\nShape: {df_raw.shape}")
print(f"\nColumn dtypes:\n{df_raw.dtypes.to_string()}")

miss    = df_raw.isnull().sum()
miss_pct = (miss / len(df_raw) * 100).round(2)
miss_df = pd.DataFrame({'missing_count': miss, 'missing_pct': miss_pct})
print(f"\nMissing value summary (columns with any missing):")
print(miss_df[miss_df.missing_count > 0].to_string())

print(f"\nSample (5 rows):\n{df_raw.sample(5, random_state=42).iloc[:, :10].to_string()}")

# Yield columns (all crop yield columns available)
YIELD_COLS   = [c for c in df_raw.columns if 'YIELD' in c.upper() and 'Kg per ha' in c]
AREA_COLS    = [c for c in df_raw.columns if 'AREA' in c.upper() and '1000 ha' in c]
PROD_COLS    = [c for c in df_raw.columns if 'PRODUCTION' in c.upper()]
CLIMATE_COLS = [c for c in df_raw.columns if any(kw in c.upper() for kw in
                ['TEMPERATURE', 'PERCIPITATION', 'EVAPOTRANSPIRATION', 'WINDSPEED'])]

print(f"\nYield columns  ({len(YIELD_COLS)}): {YIELD_COLS}")
print(f"Area columns   ({len(AREA_COLS)}):  {AREA_COLS}")
print(f"Climate columns ({len(CLIMATE_COLS)}): (first 5) {CLIMATE_COLS[:5]}")

print(f"\nDescriptive statistics (yield columns):")
print(df_raw[YIELD_COLS].describe().to_string())

# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEANING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 2 — CLEANING & PREPROCESSING")
print("="*70)

df = df_raw.copy()

# Drop columns with >40% missing
high_miss = [c for c in df.columns if df[c].isnull().mean() > 0.40]
print(f"\nDropping {len(high_miss)} columns (>40% missing): {high_miss}")
df.drop(columns=high_miss, inplace=True)

# Update column lists after dropping
YIELD_COLS   = [c for c in YIELD_COLS   if c in df.columns]
CLIMATE_COLS = [c for c in CLIMATE_COLS if c in df.columns]
AREA_COLS    = [c for c in AREA_COLS    if c in df.columns]
PROD_COLS    = [c for c in PROD_COLS    if c in df.columns]

# Fixed column names from this dataset
DISTRICT_COL = 'Dist Name'
STATE_COL    = 'State Name'
YEAR_COL     = 'Year'
DIST_CODE    = 'Dist Code'
STATE_CODE   = 'State Code'

# Categorical columns = only the object-dtype columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"\nCategorical columns: {cat_cols}")

# Numeric columns
num_cols = df.select_dtypes(include='number').columns.tolist()
print(f"Numeric columns: {len(num_cols)}")

# Impute
for col in cat_cols:
    if df[col].isnull().any():
        fv = df[col].mode()[0]
        df[col].fillna(fv, inplace=True)
        print(f"  Imputed '{col}' → mode: '{fv}'")

for col in num_cols:
    if df[col].isnull().any():
        fv = df[col].median()
        df[col].fillna(fv, inplace=True)

print(f"\nMissing after imputation: {df.isnull().sum().sum()}")

# Label encode State Name and Dist Name
from sklearn.preprocessing import LabelEncoder

label_mappings = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    label_mappings[col] = {int(i): str(c) for i, c in enumerate(le.classes_)}
    print(f"  Encoded '{col}': {len(le.classes_)} classes")

with open(DATA / "label_mappings.json", 'w') as f:
    json.dump(label_mappings, f, indent=2)
print(f"Label mappings saved → {DATA}/label_mappings.json")

# Primary target: RICE YIELD (best coverage in ICRISAT data)
TARGET = 'RICE YIELD (Kg per ha)'
if TARGET not in df.columns:
    TARGET = YIELD_COLS[0]
print(f"\nPrimary target: '{TARGET}'")

df_model = df.dropna(subset=[TARGET]).copy()
print(f"Rows with valid target: {len(df_model)} / {len(df)}")

df_model.to_csv(DATA / "master_clean.csv", index=False)
print(f"Cleaned CSV saved → {DATA}/master_clean.csv  {df_model.shape}")

# Also save a melted long-format for EDA across all crops
yield_id_cols = [YEAR_COL, STATE_COL, DISTRICT_COL, DIST_CODE, STATE_CODE]
yield_id_cols = [c for c in yield_id_cols if c in df.columns]
df_long = df[yield_id_cols + YIELD_COLS].melt(
    id_vars=yield_id_cols, var_name='Crop_Yield_Col', value_name='Yield_kg_ha'
)
df_long['Crop'] = df_long['Crop_Yield_Col'].str.replace(' YIELD (Kg per ha)', '', regex=False).str.strip()
df_long.dropna(subset=['Yield_kg_ha'], inplace=True)
df_long.to_csv(DATA / "long_format.csv", index=False)
print(f"Long-format CSV saved → {DATA}/long_format.csv  {df_long.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 3 — EDA PLOTS")
print("="*70)

# Seasonal precipitation (Kharif = Rainy JUN-SEP)
RAIN_COL = next((c for c in df.columns if 'Rainy JUN-SEP' in c and 'PERCIP' in c.upper()), None)
if RAIN_COL is None:
    RAIN_COL = next((c for c in df.columns if 'PERCIP' in c.upper()), None)
print(f"  Rainfall column: {RAIN_COL}")

# ── 3a. Yield distribution — overall + per crop (long-format) ────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(df_model[TARGET].dropna(), bins=60, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title(f'Rice Yield Distribution (all districts, 1990-2015)', fontsize=12)
axes[0].set_xlabel('Yield (Kg/ha)')
axes[0].set_ylabel('Count')

colors = plt.cm.tab10.colors
crops_u = df_long['Crop'].unique()
for i, crop in enumerate(crops_u):
    vals = df_long[df_long['Crop'] == crop]['Yield_kg_ha'].dropna()
    axes[1].hist(vals, bins=40, alpha=0.55, label=crop, color=colors[i % 10])
axes[1].set_title('Yield Distribution by Crop Type', fontsize=12)
axes[1].set_xlabel('Yield (Kg/ha)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIGS / "yield_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: yield_distribution.png")

# ── 3b. Top 10 / Bottom 10 districts by average rice yield ───────────────────
dist_avg = df_model.groupby(DISTRICT_COL)[TARGET].mean().sort_values()
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

dist_avg.head(10).plot(kind='barh', ax=axes[0], color='tomato',   edgecolor='white')
axes[0].set_title('Bottom 10 Districts — Avg Rice Yield', fontsize=12)
axes[0].set_xlabel('Avg Yield (Kg/ha)')

dist_avg.tail(10).plot(kind='barh', ax=axes[1], color='seagreen', edgecolor='white')
axes[1].set_title('Top 10 Districts — Avg Rice Yield', fontsize=12)
axes[1].set_xlabel('Avg Yield (Kg/ha)')

plt.tight_layout()
plt.savefig(FIGS / "district_yield_top_bottom.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: district_yield_top_bottom.png")

# ── 3c. Year-wise yield trend (all crops) ────────────────────────────────────
yearly_all = df_long.groupby([YEAR_COL, 'Crop'])['Yield_kg_ha'].mean().reset_index()
fig, ax = plt.subplots(figsize=(14, 6))
for i, crop in enumerate(crops_u):
    sub = yearly_all[yearly_all['Crop'] == crop]
    ax.plot(sub[YEAR_COL], sub['Yield_kg_ha'], marker='o', markersize=4,
            label=crop, color=colors[i % 10], linewidth=2)
ax.set_title('Year-wise Average Yield by Crop (All Districts)', fontsize=12)
ax.set_xlabel('Year')
ax.set_ylabel('Avg Yield (Kg/ha)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / "yield_trend_yearly.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: yield_trend_yearly.png")

# ── 3d. Correlation heatmap (climate + agri vars vs yield) ───────────────────
corr_cols = [TARGET] + CLIMATE_COLS[:20] + [
    c for c in ['NITROGEN CONSUMPTION (tons)', 'TOTAL FERTILISER CONSUMPTION (tons)',
                'GROSS IRRIGATED AREA (1000 ha)', 'GROSS CROPPED AREA (1000 ha)',
                'TOTAL AGRICULTURAL LABOUR POPULATION (1000 Number)']
    if c in df_model.columns
]
corr_df = df_model[corr_cols].dropna()
corr = corr_df.corr()
fig, ax = plt.subplots(figsize=(18, 14))
sns.heatmap(corr, cmap='RdYlGn', center=0, linewidths=0.3,
            cbar_kws={'shrink': 0.6}, ax=ax, fmt='.2f',
            annot=len(corr) <= 15)
ax.set_title('Correlation Heatmap — Climate & Agricultural Features vs Rice Yield', fontsize=12)
plt.tight_layout()
plt.savefig(FIGS / "correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: correlation_heatmap.png")

# ── 3e. Rainfall vs yield scatter (colored by state) ─────────────────────────
if RAIN_COL and RAIN_COL in df_model.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    states_u = df_model[STATE_COL].unique()[:15]
    state_colors = plt.cm.tab20.colors
    for i, st in enumerate(states_u):
        sub = df_model[df_model[STATE_COL] == st]
        ax.scatter(sub[RAIN_COL], sub[TARGET], alpha=0.3, s=8,
                   color=state_colors[i % 20], label=st)
    ax.set_xlabel(RAIN_COL)
    ax.set_ylabel(TARGET)
    ax.set_title('Kharif Rainfall vs Rice Yield (by State)', fontsize=12)
    ax.legend(fontsize=6, ncol=2, markerscale=2)
    plt.tight_layout()
    plt.savefig(FIGS / "rainfall_vs_yield.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: rainfall_vs_yield.png")

# ── 3f. Boxplot of yield by crop (Kharif crops vs Rabi crops) ────────────────
kharif = ['RICE', 'PEARL MILLET', 'GROUNDNUT']
rabi   = ['CHICKPEA']
all_crops_kv = {
    crop: 'Kharif' if crop in kharif else ('Rabi' if crop in rabi else 'Both')
    for crop in df_long['Crop'].unique()
}
df_long['Season'] = df_long['Crop'].map(all_crops_kv)

fig, ax = plt.subplots(figsize=(12, 6))
seasons = df_long['Season'].unique()
data_s  = [df_long[df_long['Season'] == s]['Yield_kg_ha'].dropna().values for s in seasons]
bp = ax.boxplot(data_s, labels=seasons, patch_artist=True,
                medianprops=dict(color='black', linewidth=2))
pal = ['#4ECDC4', '#FF6B6B', '#FFD93D']
for patch, c in zip(bp['boxes'], pal):
    patch.set_facecolor(c)
ax.set_title('Yield by Season (Kharif vs Rabi vs Both)', fontsize=12)
ax.set_ylabel('Yield (Kg/ha)')
plt.tight_layout()
plt.savefig(FIGS / "yield_by_season.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: yield_by_season.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. GEOSPATIAL VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 4 — GEOSPATIAL VISUALIZATION")
print("="*70)

import geopandas as gpd
import plotly.express as px
from rapidfuzz import process as rfprocess, fuzz
import urllib.request

GEO_URL   = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
GEO_LOCAL = DATA / "India_districts.geojson"

if not GEO_LOCAL.exists():
    print(f"Downloading district GeoJSON …")
    try:
        urllib.request.urlretrieve(GEO_URL, GEO_LOCAL)
        print("  Downloaded.")
    except Exception as e:
        print(f"  Download failed: {e}")
        GEO_LOCAL = None
else:
    print(f"  Using cached: {GEO_LOCAL}")

def fuzzy_match(name, choices, threshold=72):
    res = rfprocess.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    return res[0] if res and res[1] >= threshold else None

if GEO_LOCAL and GEO_LOCAL.exists():
    gdf = gpd.read_file(GEO_LOCAL)
    print(f"  GDF shape: {gdf.shape},  cols: {list(gdf.columns)}")

    # Find district name column in shapefile
    # geohacker dataset uses NAME_2 for district, NAME_1 for state
    geo_dist_col = next((c for c in gdf.columns if c in ['NAME_2', 'dtname', 'DISTRICT']), None)
    if geo_dist_col is None:
        geo_dist_col = next((c for c in gdf.columns
                             if 'name' in c.lower() and gdf[c].dtype == object
                             and gdf[c].nunique() > 50), None)
    if geo_dist_col is None:
        geo_dist_col = next((c for c in gdf.columns if gdf[c].dtype == object
                             and c not in ['geometry','ISO']), gdf.columns[1])
    print(f"  Geo district col: '{geo_dist_col}'")

    # District-level stats
    dist_stats = df_model.groupby(DISTRICT_COL)[TARGET].mean().reset_index()
    dist_stats.columns = [DISTRICT_COL, 'avg_yield']

    # Yield trend per district (OLS slope)
    slopes = {}
    for dist, grp in df_model.groupby(DISTRICT_COL):
        g = grp.dropna(subset=[YEAR_COL, TARGET]).sort_values(YEAR_COL)
        if len(g) >= 3:
            sl, *_ = sc_stats.linregress(g[YEAR_COL], g[TARGET])
            slopes[dist] = sl
    dist_stats['yield_trend'] = dist_stats[DISTRICT_COL].map(slopes)

    # Fuzzy match district names to shapefile
    geo_names = gdf[geo_dist_col].astype(str).tolist()
    dist_stats['geo_match'] = dist_stats[DISTRICT_COL].apply(
        lambda x: fuzzy_match(str(x), geo_names)
    )
    n_matched = dist_stats['geo_match'].notna().sum()
    print(f"  Matched {n_matched}/{len(dist_stats)} districts")

    gdf_m = gdf.merge(dist_stats, left_on=geo_dist_col, right_on='geo_match', how='left')
    gdf_m = gdf_m.to_crs(epsg=4326)

    # Choropleth — average yield
    fig_y = px.choropleth(
        gdf_m,
        geojson=gdf_m.__geo_interface__,
        locations=gdf_m.index,
        color='avg_yield',
        color_continuous_scale='YlOrRd',
        title='Average Rice Yield per District (1990–2015)',
        labels={'avg_yield': 'Avg Yield (Kg/ha)'},
        fitbounds='locations',
    )
    fig_y.update_geos(visible=False)
    fig_y.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig_y.write_html(str(FIGS / "choropleth_yield.html"))
    print("  Saved: choropleth_yield.html")

    # Choropleth — yield trend
    fig_t = px.choropleth(
        gdf_m,
        geojson=gdf_m.__geo_interface__,
        locations=gdf_m.index,
        color='yield_trend',
        color_continuous_scale='RdYlGn',
        color_continuous_midpoint=0,
        title='Rice Yield Trend per District (OLS Slope, 1990–2015)',
        labels={'yield_trend': 'Trend (Kg/ha/yr)'},
        fitbounds='locations',
    )
    fig_t.update_geos(visible=False)
    fig_t.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig_t.write_html(str(FIGS / "choropleth_trend.html"))
    print("  Saved: choropleth_trend.html")
else:
    print("  Geospatial skipped (GeoJSON not available).")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MODEL TRAINING — XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 5 — MODEL TRAINING")
print("="*70)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# Features: climate vars + fertilizer + irrigation + area + encoded identifiers
# Exclude other crop yields to prevent data leakage of the same domain
other_yields = [c for c in YIELD_COLS if c != TARGET]
other_prod   = [c for c in PROD_COLS  if 'RICE' not in c.upper()]
other_area   = [c for c in AREA_COLS  if 'RICE' not in c.upper() and 'GROSS' not in c.upper()]

FEATURE_COLS = (
    CLIMATE_COLS +
    [c for c in [
        'NITROGEN CONSUMPTION (tons)', 'PHOSPHATE CONSUMPTION (tons)',
        'POTASH CONSUMPTION (tons)',   'TOTAL FERTILISER CONSUMPTION (tons)',
        'TOTAL AGRICULTURAL LABOUR POPULATION (1000 Number)',
        'GROSS IRRIGATED AREA (1000 ha)', 'GROSS CROPPED AREA (1000 ha)',
        'RICE AREA (1000 ha)',
        'Dist Name_enc', 'State Name_enc', YEAR_COL,
    ] if c in df_model.columns]
)
# Remove duplicates
FEATURE_COLS = list(dict.fromkeys(FEATURE_COLS))

print(f"\nFeatures ({len(FEATURE_COLS)}): first 10 = {FEATURE_COLS[:10]}")
print(f"Target: {TARGET}")

X = df_model[FEATURE_COLS].copy().fillna(df_model[FEATURE_COLS].median())
y = df_model[TARGET].copy()

# Stratify by state for representative split
strat = df_model[STATE_COL].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=strat
)
print(f"Train: {X_train.shape},  Test: {X_test.shape}")

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\n{'─'*40}")
print(f"  RMSE : {rmse:.2f}")
print(f"  MAE  : {mae:.2f}")
print(f"  R²   : {r2:.4f}")
print(f"{'─'*40}")

with open(MODELS / "xgboost_model.pkl", 'wb') as f:
    pickle.dump(xgb, f)
print(f"  Model saved: {MODELS}/xgboost_model.pkl")

# Save test set with predictions
test_out = df_model.loc[X_test.index, [DISTRICT_COL, STATE_COL, YEAR_COL, TARGET]].copy()
test_out['predicted'] = y_pred
test_out['error']     = np.abs(y_test.values - y_pred)
test_out[FEATURE_COLS] = X_test[FEATURE_COLS].values
test_out.to_csv(DATA / "test_predictions.csv", index=False)

# ── 5a. Actual vs Predicted ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='royalblue')
lim = [min(y_test.min(), y_pred.min())*0.95, max(y_test.max(), y_pred.max())*1.05]
ax.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Actual Rice Yield (Kg/ha)')
ax.set_ylabel('Predicted Rice Yield (Kg/ha)')
ax.set_title(f'Actual vs Predicted  (R²={r2:.3f}, RMSE={rmse:.1f})', fontsize=12)
ax.legend()
plt.tight_layout()
plt.savefig(FIGS / "actual_vs_predicted.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: actual_vs_predicted.png")

# ── 5b. Feature Importance ────────────────────────────────────────────────────
imp = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(10, max(6, len(imp)*0.35)))
imp.plot(kind='barh', ax=ax, color='steelblue', edgecolor='white')
ax.set_title('XGBoost Feature Importance (Gain)', fontsize=12)
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(FIGS / "feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: feature_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 6 — SHAP EXPLAINABILITY")
print("="*70)

import shap

explainer = shap.TreeExplainer(xgb)
shap_vals  = explainer(X_test)

# ── 6a. Beeswarm ─────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 8))
shap.plots.beeswarm(shap_vals, show=False, max_display=20)
plt.title('SHAP Beeswarm — Global Feature Importance', fontsize=12)
plt.tight_layout()
plt.savefig(SHAPD / "shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: shap/shap_beeswarm.png")

# ── 6b. Waterfall — worst predicted ──────────────────────────────────────────
errors    = np.abs(y_test.values - y_pred)
worst_idx = int(np.argmax(errors))
best_idx  = int(np.argmin(errors))

plt.figure(figsize=(12, 7))
shap.plots.waterfall(shap_vals[worst_idx], show=False, max_display=15)
plt.title(f'SHAP Waterfall — Worst Prediction (error={errors[worst_idx]:.1f} Kg/ha)', fontsize=11)
plt.tight_layout()
plt.savefig(SHAPD / "shap_waterfall_worst.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: shap/shap_waterfall_worst.png")

# ── 6c. Waterfall — best predicted ───────────────────────────────────────────
plt.figure(figsize=(12, 7))
shap.plots.waterfall(shap_vals[best_idx], show=False, max_display=15)
plt.title(f'SHAP Waterfall — Best Prediction (error={errors[best_idx]:.2f} Kg/ha)', fontsize=11)
plt.tight_layout()
plt.savefig(SHAPD / "shap_waterfall_best.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: shap/shap_waterfall_best.png")

# ── 6d. Dependence plots — top 2 SHAP features ───────────────────────────────
mean_abs = np.abs(shap_vals.values).mean(0)
top2_idx  = np.argsort(mean_abs)[::-1][:2]
top2_feat = [FEATURE_COLS[i] for i in top2_idx]
print(f"  Top 2 SHAP features: {top2_feat}")

for feat in top2_feat:
    plt.figure(figsize=(9, 6))
    shap.plots.scatter(shap_vals[:, feat], show=False)
    plt.title(f'SHAP Dependence — {feat}', fontsize=12)
    plt.tight_layout()
    safe = feat.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(SHAPD / f"shap_dependence_{safe}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: shap/shap_dependence_{safe}.png")

# Save SHAP values CSV
shap_df = pd.DataFrame(shap_vals.values, columns=FEATURE_COLS)
shap_df['row_index'] = X_test.index.tolist()
shap_df.to_csv(DATA / "shap_values.csv", index=False)

# Top 5 features by mean |SHAP|
shap_importance = pd.Series(mean_abs, index=FEATURE_COLS).sort_values(ascending=False)
top5_shap = shap_importance.head(5)

# ══════════════════════════════════════════════════════════════════════════════
# 7. STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 7 — WRITING STREAMLIT DASHBOARD")
print("="*70)

# Save metadata for dashboard
meta = {
    "TARGET":       TARGET,
    "DISTRICT_COL": DISTRICT_COL,
    "STATE_COL":    STATE_COL,
    "YEAR_COL":     YEAR_COL,
    "RAIN_COL":     RAIN_COL,
    "CROP_COL":     None,
    "SEASON_COL":   None,
    "feature_cols": FEATURE_COLS,
    "top2_shap":    top2_feat,
    "yield_cols":   YIELD_COLS,
    "model_rmse":   round(rmse, 2),
    "model_mae":    round(mae, 2),
    "model_r2":     round(r2, 4),
}
with open(DATA / "meta.json", 'w') as f:
    json.dump(meta, f, indent=2)

dashboard = '''"""
ICRISAT Crop Yield Interactive Dashboard
Run with: streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, json, shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
'''

with open(APPD / "dashboard.py", 'w') as f:
    f.write(dashboard)
print(f"  Dashboard saved: {APPD}/dashboard.py")

# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 8 — FINAL SUMMARY")
print("="*70)

summary = f"""
# ICRISAT District-Level Crop Yield Analysis — Summary

## Dataset
| Metric | Value |
|---|---|
| Raw shape | {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns |
| After cleaning | {df_model.shape[0]:,} rows × {df_model.shape[1]} columns |
| Districts | 560 |
| States | {df_model[STATE_COL].nunique()} |
| Years | {df_model[YEAR_COL].min():.0f}–{df_model[YEAR_COL].max():.0f} |
| Target variable | `{TARGET}` |
| Feature count | {len(FEATURE_COLS)} |
| Crop yield columns | {YIELD_COLS} |

## Model Performance (XGBoost, 80/20 split, stratified by state)
| Metric | Value |
|---|---|
| RMSE | {rmse:.2f} Kg/ha |
| MAE | {mae:.2f} Kg/ha |
| R² | {r2:.4f} |

## Top 5 Features by Mean |SHAP|
{top5_shap.to_markdown()}

## Outputs
| Output | Path |
|---|---|
| Cleaned data | `data/processed/master_clean.csv` |
| Long-format data | `data/processed/long_format.csv` |
| Label mappings | `data/processed/label_mappings.json` |
| XGBoost model | `outputs/models/xgboost_model.pkl` |
| EDA figures | `outputs/figures/*.png` |
| Choropleth maps | `outputs/figures/choropleth_*.html` |
| SHAP figures | `outputs/figures/shap/*.png` |
| Dashboard | `app/dashboard.py` |

## Run the Dashboard
```bash
streamlit run app/dashboard.py
```
Then open **http://localhost:8501** in your browser.
"""

print(summary)
with open(BASE / "SUMMARY.md", 'w') as f:
    f.write(summary)

print(f"SUMMARY.md saved → {BASE}/SUMMARY.md")
print("\n✓  Pipeline complete.")
