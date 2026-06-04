"""
Train XGBRegressor on master_clean enriched with IMD annual rainfall.

Steps:
  1. Save enriched dataframe → data/processed/master_enriched.csv
  2. Load feature_cols + TARGET from meta.json, append ANNUAL_RAIN_IMD
  3. Reproduce original train/test split (test_size=0.20, random_state=42,
     stratify=state) — no split_indices.json exists, so we replicate the logic
  4. Fill NaNs with column medians
  5. Train XGBRegressor (same hyper-params as baseline)
  6. Evaluate → RMSE, MAE, R²  (compare to baseline in meta.json)
  7. Save model → outputs/models/xgboost_enriched.pkl
  8. Update meta.json with enriched_rmse/mae/r2/feature_cols
"""

import json
import pickle
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ── paths ─────────────────────────────────────────────────────────────────────
DATA    = pathlib.Path("data/processed")
OUTPUTS = pathlib.Path("outputs/models")
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ── 1. Build enriched dataframe ───────────────────────────────────────────────
master  = pd.read_csv(DATA / "master_clean.csv")
imd_rain = pd.read_csv(DATA / "master_with_imd_rain.csv")[
    ["Dist Name", "ANNUAL_RAIN_IMD"]
].drop_duplicates("Dist Name")

enriched = master.merge(imd_rain, on="Dist Name", how="left")

out_path = DATA / "master_enriched.csv"
enriched.to_csv(out_path, index=False)
print(f"Saved enriched dataframe → {out_path}  ({len(enriched):,} rows)")

# ── 2. Load meta and build feature list ───────────────────────────────────────
with open(DATA / "meta.json") as f:
    meta = json.load(f)

TARGET       = meta["TARGET"]
DISTRICT_COL = meta["DISTRICT_COL"]
STATE_COL    = meta["STATE_COL"]
feature_cols = list(meta["feature_cols"])          # original list

# Append ANNUAL_RAIN_IMD only if it landed in the dataframe
if "ANNUAL_RAIN_IMD" in enriched.columns:
    if "ANNUAL_RAIN_IMD" not in feature_cols:
        feature_cols.append("ANNUAL_RAIN_IMD")
    print("ANNUAL_RAIN_IMD appended to feature_cols.")
else:
    print("WARNING: ANNUAL_RAIN_IMD not found in enriched dataframe — skipping.")

# Keep only columns that actually exist
feature_cols = [c for c in feature_cols if c in enriched.columns]
print(f"Features: {len(feature_cols)}  (was {len(meta['feature_cols'])} before enrichment)")
print(f"Target  : {TARGET}")

# ── 3. Drop rows where target is null, build X / y ───────────────────────────
df_model = enriched.dropna(subset=[TARGET]).copy()

X = df_model[feature_cols].copy()
y = df_model[TARGET].copy()

# ── 4. Fill NaNs with column medians ─────────────────────────────────────────
medians = X.median()
X = X.fillna(medians)

# ── 5. Reproduce original split (stratify by state) ───────────────────────────
strat = df_model[STATE_COL].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=strat
)
print(f"\nTrain: {X_train.shape}   Test: {X_test.shape}")

# ── 6. Train XGBRegressor ─────────────────────────────────────────────────────
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
print("Training complete.")

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = xgb.predict(X_test)
rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae    = float(mean_absolute_error(y_test, y_pred))
r2     = float(r2_score(y_test, y_pred))

baseline_rmse = meta.get("model_rmse", None)
baseline_mae  = meta.get("model_mae",  None)
baseline_r2   = meta.get("model_r2",   None)

print("\n=== Evaluation (test set) ===")
print(f"{'Metric':<8}  {'Enriched':>12}  {'Baseline':>12}  {'Delta':>10}")
print("-" * 46)
for label, val, base in [("RMSE", rmse, baseline_rmse),
                          ("MAE",  mae,  baseline_mae),
                          ("R²",   r2,   baseline_r2)]:
    delta = f"{val - base:+.4f}" if base is not None else "n/a"
    print(f"{label:<8}  {val:>12.4f}  {str(base):>12}  {delta:>10}")

# ── 8. Save model ─────────────────────────────────────────────────────────────
model_path = OUTPUTS / "xgboost_enriched.pkl"
with open(model_path, "wb") as f:
    pickle.dump(xgb, f)
print(f"\nModel saved → {model_path}")

# ── 9. Update meta.json ───────────────────────────────────────────────────────
meta["enriched_rmse"]         = round(rmse, 4)
meta["enriched_mae"]          = round(mae,  4)
meta["enriched_r2"]           = round(r2,   4)
meta["enriched_feature_cols"] = feature_cols

with open(DATA / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"meta.json updated with enriched_rmse={rmse:.4f}, "
      f"enriched_mae={mae:.4f}, enriched_r2={r2:.4f}")
