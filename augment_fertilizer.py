import pandas as pd
import numpy as np
import json
from pathlib import Path
from fuzzywuzzy import process
import warnings

warnings.filterwarnings('ignore')

BASE = Path(".")
DATA = BASE / "data/processed"
RAW  = BASE / "data/raw"

print("Loading baseline data...")
df = pd.read_csv(DATA / "master_clean.csv")
with open(DATA / "meta.json") as f:
    meta = json.load(f)

DC = meta["DISTRICT_COL"]
YC = meta["YEAR_COL"]

print("Loading fertilizer data...")
try:
    fert = pd.read_csv(RAW / "fertilizer_consumption.csv")
except FileNotFoundError:
    print("❌ fertilizer_consumption.csv not found in data/raw/. Skipping augmentation.")
    exit()

# Smart column mapping to handle variations in the government dataset
cols_lower = {c.lower(): c for c in fert.columns}

def get_col(keywords):
    for kw in keywords:
        for lo, orig in cols_lower.items():
            if kw in lo: return orig
    return None

dist_raw = get_col(["district", "dist", "dname"])
year_raw = get_col(["year", "yr"])
n_raw = get_col(["nitrogen", " n ", " n(", " n_", "n_consumption"])
p_raw = get_col(["phosphate", " p ", " p(", " p_", "p_consumption"])
k_raw = get_col(["potash", " k ", " k(", " k_", "k_consumption"])

if not dist_raw or not year_raw:
    print("❌ Could not detect District or Year columns in the fertilizer dataset.")
    exit()

fert = fert.rename(columns={
    dist_raw: "DIST_RAW",
    year_raw: YC,
    n_raw: "N_GOV",
    p_raw: "P_GOV",
    k_raw: "K_GOV"
})

print(f"Fuzzy matching {len(fert['DIST_RAW'].unique())} districts. This will take a few seconds...")
master_dists = df[DC].dropna().unique().tolist()
fert["DIST_MATCHED"] = fert["DIST_RAW"].apply(
    lambda x: process.extractOne(str(x), master_dists)[0] if pd.notnull(x) else None
)

fert_merge = fert[["DIST_MATCHED", YC, "N_GOV", "P_GOV", "K_GOV"]].rename(
    columns={"DIST_MATCHED": DC}
)

# Average out duplicates if the government data has multiple seasons per year
fert_merge = fert_merge.groupby([DC, YC], as_index=False).mean(numeric_only=True)

df = df.merge(fert_merge, on=[DC, YC], how="left")

print("\n--- Augmentation Results ---")
for icrisat_col, gov_col in [
    ("NITROGEN CONSUMPTION (tons)",   "N_GOV"),
    ("PHOSPHATE CONSUMPTION (tons)",  "P_GOV"),
    ("POTASH CONSUMPTION (tons)",     "K_GOV"),
]:
    if icrisat_col in df.columns and gov_col in df.columns:
        # Calculate how many NaNs were filled
        initial_missing = df[icrisat_col].isna().sum()
        df[icrisat_col] = df[icrisat_col].fillna(df[gov_col])
        new_missing = df[icrisat_col].isna().sum()
        filled = initial_missing - new_missing
        print(f"✅ {icrisat_col}: Filled {filled} missing values from government data.")

# Drop temporary columns
df = df.drop(columns=["N_GOV", "P_GOV", "K_GOV"], errors='ignore')

df.to_csv(DATA / "master_clean.csv", index=False)
print(f"\nmaster_clean.csv successfully updated and overwritten: {df.shape}")