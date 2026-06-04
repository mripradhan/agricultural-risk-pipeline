"""
Train XGBRegressor on master_enriched.csv (baseline features + ANNUAL_RAIN_IMD
+ Season_enc). Reproduces the original train/test split (test_size=0.20,
random_state=42, stratify=State Name) — no split_indices.json exists.
"""

import json
import pickle
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA    = pathlib.Path("data/processed")
OUTPUTS = pathlib.Path("outputs/models")
OUTPUTS.mkdir(parents=True, exist_ok=True)

# ── 1. Load enriched data + meta ──────────────────────────────────────────────
df = pd.read_csv(DATA / "master_enriched.csv")

with open(DATA / "meta.json") as f:
    meta = json.load(f)

TARGET       = meta["TARGET"]
STATE_COL    = meta["STATE_COL"]
feature_cols = list(meta["feature_cols"])

# ── 2. Append new columns if present ─────────────────────────────────────────
for col in ["ANNUAL_RAIN_IMD", "Season_enc"]:
    if col in df.columns and col not in feature_cols:
        feature_cols.append(col)
        print(f"Appended to feature_cols: '{col}'")
    elif col not in df.columns:
        print(f"WARNING: '{col}' not found in dataframe — skipping.")

# Keep only columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"Total features: {len(feature_cols)}")
print(f"Target        : {TARGET}")

# ── 3. Drop rows where target is null ─────────────────────────────────────────
df_model = df.dropna(subset=[TARGET]).copy()
print(f"\nModelling rows: {len(df_model):,}  "
      f"(Season_enc=-1 rows kept: "
      f"{(df_model['Season_enc'] == -1).sum():,})")

X = df_model[feature_cols].copy()
y = df_model[TARGET].copy()

# ── 4. Fill NaNs with column medians ─────────────────────────────────────────
X = X.fillna(X.median())

# ── 5. Reproduce original split (stratify by state) ───────────────────────────
strat = df_model[STATE_COL].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=strat
)
print(f"Train: {X_train.shape}   Test: {X_test.shape}")

# ── 6. Train ──────────────────────────────────────────────────────────────────
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

imd_rmse, imd_mae, imd_r2       = 377.35, 261.79, 0.8710
base_rmse, base_mae, base_r2    = 379.56, 263.69, 0.8695

print("\n" + "─" * 54)
print(f"{'Metric':<8} │ {'Season+IMD':>12} │ {'IMD Only':>10} │ {'Baseline':>10}")
print("─" * 54)
for label, val, imd_val, base_val in [
    ("RMSE", rmse, imd_rmse, base_rmse),
    ("MAE",  mae,  imd_mae,  base_mae),
    ("R²",   r2,   imd_r2,   base_r2),
]:
    delta = val - base_val
    arrow = "▼" if (label in ("RMSE", "MAE") and delta < 0) or \
                   (label == "R²" and delta > 0) else "▲" if delta != 0 else " "
    print(f"{label:<8} │ {val:>11.4f} │ {imd_val:>10} │ {base_val:>10}  {arrow}")
print("─" * 54)
print("▼ = improvement over baseline")

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
print(f"meta.json updated  →  enriched_rmse={rmse:.4f}, "
      f"enriched_mae={mae:.4f}, enriched_r2={r2:.4f}")
