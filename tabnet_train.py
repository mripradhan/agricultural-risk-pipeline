"""
TabNet Training — Task 2 (Umar)
Trains a TabNetRegressor on master_clean.csv and saves tabnet_results.json.

Always reproduces the exact same stratified split as analysis.py
(test_size=0.20, random_state=42, stratify=State Name) — no index
dependency on split_indices.json needed.

Usage:
    python tabnet_train.py
"""

import numpy as np
import pandas as pd
import json
import time
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor

BASE   = Path(__file__).resolve().parent
DATA   = BASE / "data/processed"
MODELS = BASE / "outputs/models"
MODELS.mkdir(parents=True, exist_ok=True)

# ── 1. Load data & meta ───────────────────────────────────────────────────────
df = pd.read_csv(DATA / "master_clean.csv")
with open(DATA / "meta.json") as f:
    meta = json.load(f)

T     = meta["TARGET"]
FEAT  = [f for f in meta["feature_cols"] if f in df.columns]
STATE = meta["STATE_COL"]

# Drop rows where target is missing (mirrors analysis.py)
df = df.dropna(subset=[T]).reset_index(drop=True)

X = df[FEAT].fillna(df[FEAT].median())
y = df[T]

# ── 2. Reproduce the exact same split as analysis.py ─────────────────────────
# We always use the dynamic split here.
# split_indices.json stores original-df indices which won't align with a
# freshly-loaded CSV that has been reset_index'd — so we just reproduce the
# split deterministically (same random_state + stratify = identical rows).
print("Reproducing stratified split (random_state=42, test_size=0.20).")
strat = df[STATE].astype(str)
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=strat
)

X_train = X_train_df.values
X_test  = X_test_df.values
y_train = y_train_df.values.reshape(-1, 1)
y_test  = y_test_df.values.reshape(-1, 1)

print(f"Train: {X_train.shape}   Test: {X_test.shape}")

# ── 3. Train TabNet ───────────────────────────────────────────────────────────
model = TabNetRegressor(
    n_d=32, n_a=32, n_steps=5, gamma=1.5,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": 2e-3},
    scheduler_params={"step_size": 50, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=10,
    seed=42,
)

t0 = time.time()
model.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_test, y_test)],
    max_epochs=200, patience=20, batch_size=1024,
)
elapsed = time.time() - t0

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
preds  = model.predict(X_test).flatten()
y_flat = y_test.flatten()

results = {
    "rmse":         round(float(np.sqrt(mean_squared_error(y_flat, preds))), 2),
    "mae":          round(float(mean_absolute_error(y_flat, preds)), 2),
    "r2":           round(float(r2_score(y_flat, preds)), 4),
    "train_time_s": round(elapsed, 1),
}
print("\nTabNet results:", results)

# ── 5. Print comparison with XGBoost baseline ─────────────────────────────────
xgb_rmse = meta.get("model_rmse", "N/A")
xgb_mae  = meta.get("model_mae",  "N/A")
xgb_r2   = meta.get("model_r2",   "N/A")

print(f"\n{'─'*52}")
print(f"{'Metric':<8} │ {'TabNet':>10} │ {'XGBoost':>10}")
print(f"{'─'*52}")
for label, tn_val, xgb_val in [
    ("RMSE", results["rmse"], xgb_rmse),
    ("MAE",  results["mae"],  xgb_mae),
    ("R²",   results["r2"],   xgb_r2),
]:
    print(f"{label:<8} │ {tn_val:>10} │ {str(xgb_val):>10}")
print(f"{'─'*52}")
print(f"Training time: {elapsed:.1f}s")

# ── 6. Save results & model ───────────────────────────────────────────────────
out_path = MODELS / "tabnet_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_path}")

model.save_model(str(MODELS / "tabnet_model"))
print(f"Saved → {MODELS}/tabnet_model.zip")

meta["tabnet_rmse"] = results["rmse"]
meta["tabnet_mae"]  = results["mae"]
meta["tabnet_r2"]   = results["r2"]
with open(DATA / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("meta.json updated with TabNet metrics.")