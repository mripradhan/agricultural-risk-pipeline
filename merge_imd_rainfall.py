"""
Merge IMD rainfall data into master_clean.csv with fuzzy district matching.

The IMD file is semicolon-delimited with daily columns (1st…31st) and a
'month' column but NO year column.  Set IMD_YEAR below to the year the
data represents, or pass multiple files by running this script once per year.

Usage:
    python merge_imd_rainfall.py
"""

import json
import pandas as pd
from fuzzywuzzy import process, fuzz

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMD_YEAR    = 2022          # ← set to the year this rainfall file represents
SCORE_CUTOFF = 80           # fuzzy-match threshold (0–100); raise for stricter
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Load IMD file (semicolon-delimited) ────────────────────────────────────
imd_path = "data/raw/imd_rainfall.csv"
imd_raw = pd.read_csv(imd_path, sep=";")

# Strip accidental whitespace from column names and string values
imd_raw.columns = [c.strip().strip('"') for c in imd_raw.columns]
for c in imd_raw.select_dtypes("object").columns:
    imd_raw[c] = imd_raw[c].str.strip().str.strip('"')

print("=== IMD CSV — column names ===")
print(list(imd_raw.columns))
print("\n=== IMD CSV — first 3 rows ===")
print(imd_raw.head(3).to_string(index=False))
print()

# ── 2. Identify key columns ───────────────────────────────────────────────────
cols_lower = {c.lower(): c for c in imd_raw.columns}

def pick(keywords):
    for kw in keywords:
        for lo, orig in cols_lower.items():
            if kw in lo:
                return orig
    return None

dist_col  = pick(["district", "dist", "dname"])
month_col = pick(["month", "mon"])

if dist_col is None:
    raise ValueError(f"Could not find a district column. Columns: {list(imd_raw.columns)}")
if month_col is None:
    raise ValueError(f"Could not find a month column. Columns: {list(imd_raw.columns)}")

# Daily columns are ordinal strings like "1st", "2nd", … "31st"
day_cols = [c for c in imd_raw.columns
            if c.lower().rstrip("stndrdth").isdigit()]

print(f"District col : '{dist_col}'")
print(f"Month col    : '{month_col}'")
print(f"Day cols     : {day_cols[:5]} … ({len(day_cols)} total)")
print(f"Year assigned: {IMD_YEAR}")

# ── 3. Sum daily → monthly, then monthly → annual per district ────────────────
imd_raw[day_cols] = imd_raw[day_cols].apply(pd.to_numeric, errors="coerce")

# Monthly total = sum of day columns for that row
imd_raw["monthly_rain"] = imd_raw[day_cols].sum(axis=1)

# Annual total = sum across all months per district
imd = (
    imd_raw
    .groupby(dist_col, as_index=False)["monthly_rain"]
    .sum()
    .rename(columns={dist_col: "DIST_IMD", "monthly_rain": "ANNUAL_RAIN_IMD"})
)
imd["YEAR_IMD"] = IMD_YEAR

print(f"\nIMD after aggregation: {len(imd):,} districts, year={IMD_YEAR}")
print(imd.head(5).to_string(index=False))
print()

# ── 4. Load master and get district list ──────────────────────────────────────
with open("data/processed/meta.json") as f:
    meta = json.load(f)

DISTRICT_COL = meta["DISTRICT_COL"]   # "Dist Name"
YEAR_COL     = meta["YEAR_COL"]       # "Year"

master = pd.read_csv("data/processed/master_clean.csv")
master_districts = master[DISTRICT_COL].dropna().unique().tolist()

print(f"master_clean : {len(master):,} rows, "
      f"{len(master_districts)} unique districts (col: '{DISTRICT_COL}')")

# ── 5. Fuzzy-match IMD districts → master districts ───────────────────────────
def best_match(name, choices, cutoff=SCORE_CUTOFF):
    result = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= cutoff:
        return result[0], result[1]
    return None, None

mapping, scores = {}, {}
print(f"\nFuzzy-matching {imd['DIST_IMD'].nunique()} IMD districts "
      f"(cutoff={SCORE_CUTOFF})…")
for name in imd["DIST_IMD"].unique():
    matched, score = best_match(str(name), master_districts)
    mapping[name] = matched
    scores[name]  = score

matched_count   = sum(1 for v in mapping.values() if v is not None)
unmatched_names = [k for k, v in mapping.items() if v is None]

print(f"  Matched   : {matched_count}/{len(mapping)}")
if unmatched_names:
    print(f"  Unmatched ({len(unmatched_names)}): {unmatched_names[:20]}")

print("\nSample mappings (IMD district → master district, score):")
for name, matched in list(mapping.items())[:15]:
    print(f"  {name!r:35s} → {str(matched)!r:30s}  ({scores[name]})")

imd["DIST_MATCHED"] = imd["DIST_IMD"].map(mapping)

# ── 6. Left-join into master on district only (broadcast across all years) ────
# We have only one year of IMD data, so we drop YEAR_IMD and let the single
# rainfall value propagate to every year row for that district in master_clean.
imd_merge = (
    imd[imd["DIST_MATCHED"].notna()]
    [["DIST_MATCHED", "ANNUAL_RAIN_IMD"]]
    .drop_duplicates("DIST_MATCHED")
)

master_merged = master.merge(
    imd_merge,
    left_on=DISTRICT_COL,
    right_on="DIST_MATCHED",
    how="left",
)
master_merged.drop(columns=["DIST_MATCHED"], inplace=True, errors="ignore")

# ── 7. Coverage report ────────────────────────────────────────────────────────
non_null = master_merged["ANNUAL_RAIN_IMD"].notna().sum()
total    = len(master_merged)

filled_districts  = (
    master_merged[master_merged["ANNUAL_RAIN_IMD"].notna()][DISTRICT_COL]
    .nunique()
)
missing_districts = (
    master_merged[master_merged["ANNUAL_RAIN_IMD"].isna()][DISTRICT_COL]
    .nunique()
)

print(f"\n=== Merge result ===")
print(f"Total rows                            : {total:,}")
print(f"Rows with non-null ANNUAL_RAIN_IMD    : {non_null:,}  ({100*non_null/total:.1f}%)")
print(f"Rows still missing ANNUAL_RAIN_IMD    : {total - non_null:,}")
print(f"Unique districts WITH rainfall value  : {filled_districts}")
print(f"Unique districts MISSING rainfall     : {missing_districts}")

# ── 8. Save ───────────────────────────────────────────────────────────────────
out_path = "data/processed/master_with_imd_rain.csv"
master_merged.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
