import numpy as np, pandas as pd, json, time, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pytorch_tabnet.tab_model import TabNetRegressor

BASE   = Path(".")
DATA   = BASE / "data/processed"
MODELS = BASE / "outputs/models"

df = pd.read_csv(DATA / "master_clean.csv")
with open(DATA / "meta.json") as f:
    meta = json.load(f)

T    = meta["TARGET"]
FEAT = meta["feature_cols"]
FEAT = [f for f in FEAT if f in df.columns]

X = df[FEAT].fillna(df[FEAT].median())
y = df[T].fillna(df[T].median())

# Perform the exact same split dynamically (avoids needing split_indices.json)
strat = df[meta["STATE_COL"]].astype(str)
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=strat
)

X_train = X_train_df.values
X_test  = X_test_df.values
y_train = y_train_df.values.reshape(-1, 1)
y_test  = y_test_df.values.reshape(-1, 1)

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