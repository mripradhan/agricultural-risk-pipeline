import sys, pickle, json
from pathlib import Path

BASE   = Path(__file__).parent
DATA   = BASE / "data/processed"
MODELS = BASE / "outputs/models"

print("Python:", sys.version)

import streamlit; print("Streamlit:", streamlit.__version__)
import shap;      print("SHAP:     ", shap.__version__)
import xgboost;   print("XGBoost:  ", xgboost.__version__)
import pandas;    print("Pandas:   ", pandas.__version__)

# Load model
with open(MODELS / "xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded OK:", type(model).__name__)

# Quick predict test
import numpy as np
with open(DATA / "meta.json") as f:
    meta = json.load(f)
pred_df = pandas.read_csv(DATA / "test_predictions.csv")
feat = [c for c in meta["feature_cols"] if c in pred_df.columns]
X_sample = pred_df[feat].head(3).values
y_hat = model.predict(X_sample)
print("Predict test OK:", y_hat.round(1))

print("\n✓ Environment is fully ready. Run the dashboard with:")
print('  streamlit run app\\dashboard.py')
