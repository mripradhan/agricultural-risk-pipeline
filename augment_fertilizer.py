"""
Fertilizer Augmentation — Task 1 (Umar)
Fills missing Nitrogen / Phosphate / Potash values in master_clean.csv
using the government fertilizer_consumption.csv dataset.

The government CSV already uses clean ICRISAT-compatible column names,
so we do a direct merge first, then fall back to fuzzy matching for
district names that differ in casing or spelling.

Usage:
    python augment_fertilizer.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from fuzzywuzzy import process, fuzz
import warnings

warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/processed"
RAW  = BASE / "data/raw"

# ── 1. Load master ─────────────────────────────────────────────────────────
print("Loading baseline data...")
df = pd.read_csv(DATA / "master_clean.csv")
with open(DATA / "meta.json") as f:
    meta = json.load(f)

DC = meta["DISTRICT_COL"]   # "Dist Name"
YC = meta["YEAR_COL"]       # "Year"

# ── 2. Load fertilizer dataset ─────────────────────────────────────────────
print("Loading fertilizer data...")
fert_path = RAW / "fertilizer_consumption.csv"
if not fert_path.exists():
    print(f"❌  {fert_path} not found. Skipping augmentation.")
    raise SystemExit(1)

fert = pd.read_csv(fert_path)
fert.columns = fert.columns.str.strip()
print(f"  Fertilizer dataset: {fert.shape}  cols: {list(fert.columns)}")

# ── 3. Resolve column names ────────────────────────────────────────────────
# The government CSV uses "Dist Name" and "Year" (same as master).
# We also handle common variants gracefully.
cols_lower = {c.lower().strip(): c for c in fert.columns}

def _pick(candidates: list[str]) -> str | None:
    """Return the first column name whose lowercased form contains any candidate."""
    for kw in candidates:
        for lo, orig in cols_lower.items():
            if kw in lo:
                return orig
    return None

dist_raw = _pick(["dist name", "district", "dist ", "dname"])
year_raw = _pick(["year"])
n_raw    = _pick(["nitrogen"])
p_raw    = _pick(["phosphate"])
k_raw    = _pick(["potash"])

if dist_raw is None or year_raw is None:
    raise ValueError(
        f"Could not detect District/Year columns in fertilizer CSV.\n"
        f"Available columns: {list(fert.columns)}"
    )

print(f"  Detected  dist='{dist_raw}'  year='{year_raw}'  "
      f"N='{n_raw}'  P='{p_raw}'  K='{k_raw}'")

# Normalise year dtype
fert[year_raw] = pd.to_numeric(fert[year_raw], errors='coerce')
df[YC]         = pd.to_numeric(df[YC],         errors='coerce')

# ── 4. Fuzzy district matching ─────────────────────────────────────────────
master_dists  = df[DC].dropna().unique().tolist()
fert_dists    = fert[dist_raw].dropna().unique().tolist()

print(f"\nFuzzy-matching {len(fert_dists)} fertilizer districts "
      f"against {len(master_dists)} master districts …")

dist_map: dict[str, str | None] = {}
for name in fert_dists:
    # Try exact (case-insensitive) first — fast and reliable
    name_lo = str(name).strip().lower()
    exact = next((d for d in master_dists if d.strip().lower() == name_lo), None)
    if exact:
        dist_map[name] = exact
    else:
        result = process.extractOne(str(name), master_dists,
                                    scorer=fuzz.token_sort_ratio)
        dist_map[name] = result[0] if result and result[1] >= 70 else None

matched_n = sum(1 for v in dist_map.values() if v is not None)
print(f"  Matched {matched_n}/{len(fert_dists)} districts "
      f"({100*matched_n/len(fert_dists):.1f}%)")

fert["_dist_matched"] = fert[dist_raw].map(dist_map)

# ── 5. Build merge frame ───────────────────────────────────────────────────
rename_map = {"_dist_matched": DC, year_raw: YC}
if n_raw: rename_map[n_raw] = "N_GOV"
if p_raw: rename_map[p_raw] = "P_GOV"
if k_raw: rename_map[k_raw] = "K_GOV"

gov_cols = ["_dist_matched", year_raw] + [c for c in [n_raw, p_raw, k_raw] if c]
fert_merge = (
    fert[fert["_dist_matched"].notna()][gov_cols]
    .rename(columns=rename_map)
    .groupby([DC, YC], as_index=False)
    .mean(numeric_only=True)
)

df = df.merge(fert_merge, on=[DC, YC], how="left")

# ── 6. Fill missing values ─────────────────────────────────────────────────
fill_pairs = [
    ("NITROGEN CONSUMPTION (tons)",  "N_GOV"),
    ("PHOSPHATE CONSUMPTION (tons)", "P_GOV"),
    ("POTASH CONSUMPTION (tons)",    "K_GOV"),
]

print("\n--- Augmentation Results ---")
for icrisat_col, gov_col in fill_pairs:
    if icrisat_col not in df.columns:
        print(f"  ⚠  '{icrisat_col}' not in master — skipping.")
        continue
    if gov_col not in df.columns:
        print(f"  ⚠  '{gov_col}' not merged — skipping.")
        continue
    before = int(df[icrisat_col].isna().sum())
    df[icrisat_col] = df[icrisat_col].fillna(df[gov_col])
    after  = int(df[icrisat_col].isna().sum())
    filled = before - after
    pct    = 100 * filled / before if before else 0
    print(f"  ✅ {icrisat_col}: filled {filled}/{before} missing values "
          f"({pct:.1f}%) from government data.")

# Also recalculate TOTAL FERTILISER CONSUMPTION if all three components present
for col in ["TOTAL FERTILISER CONSUMPTION (tons)"]:
    if col in df.columns:
        components = ["NITROGEN CONSUMPTION (tons)",
                      "PHOSPHATE CONSUMPTION (tons)",
                      "POTASH CONSUMPTION (tons)"]
        if all(c in df.columns for c in components):
            missing_total = df[col].isna()
            recalc = df[components].sum(axis=1)
            df.loc[missing_total, col] = recalc[missing_total]
            n_recalc = int(missing_total.sum())
            print(f"  ✅ {col}: recalculated {n_recalc} missing rows from N+P+K sum.")

# Drop temporary government columns
df.drop(columns=["N_GOV", "P_GOV", "K_GOV"], errors='ignore', inplace=True)

# ── 7. Save ────────────────────────────────────────────────────────────────
df.to_csv(DATA / "master_clean.csv", index=False)
print(f"\n✅ master_clean.csv updated: {df.shape}")