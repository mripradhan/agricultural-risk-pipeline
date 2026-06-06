"""
Merge IMD rainfall data into master_clean.csv with fuzzy district matching.

The IMD file is semicolon-delimited with daily columns (1st … 31st) and a
'month' column but NO explicit year column.  Since the ICRISAT master spans
1990–2015 and the IMD dataset is a single-year snapshot, we broadcast the
annual district rainfall to every matching year-row in master.

Usage:
    python merge_imd_rainfall.py [--year 2010] [--cutoff 80]
"""

import json
import argparse
import pandas as pd
from fuzzywuzzy import process, fuzz
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--year",   type=int, default=2010,
                    help="Year the IMD rainfall snapshot represents (default: 2010)")
parser.add_argument("--cutoff", type=int, default=80,
                    help="Fuzzy-match score threshold 0–100 (default: 80)")
args, _ = parser.parse_known_args()

IMD_YEAR     = args.year
SCORE_CUTOFF = args.cutoff

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/processed"
RAW  = BASE / "data/raw"

# ── 1. Load IMD (semicolon-delimited) ─────────────────────────────────────────
imd_path = RAW / "imd_rainfall.csv"
imd_raw  = pd.read_csv(imd_path, sep=";")

# Strip whitespace / stray quotes from column names and string values
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

def _pick(keywords: list[str]) -> str | None:
    for kw in keywords:
        for lo, orig in cols_lower.items():
            if kw in lo:
                return orig
    return None

dist_col  = _pick(["district", "dist", "dname"])
month_col = _pick(["month", "mon"])

if dist_col is None:
    raise ValueError(f"Could not find district column. Columns: {list(imd_raw.columns)}")
if month_col is None:
    raise ValueError(f"Could not find month column. Columns: {list(imd_raw.columns)}")

# Daily columns are ordinal strings like "1st", "2nd", … "31st"
day_cols = [c for c in imd_raw.columns
            if c.lower().rstrip("stndrdth").isdigit()]

print(f"District col : '{dist_col}'")
print(f"Month col    : '{month_col}'")
print(f"Day cols     : {day_cols[:5]} … ({len(day_cols)} total)")
print(f"Year assigned: {IMD_YEAR}")

# ── 3. Aggregate daily → annual per district ──────────────────────────────────
imd_raw[day_cols] = imd_raw[day_cols].apply(pd.to_numeric, errors="coerce")
imd_raw["monthly_rain"] = imd_raw[day_cols].sum(axis=1)

imd = (
    imd_raw
    .groupby(dist_col, as_index=False)["monthly_rain"]
    .sum()
    .rename(columns={dist_col: "DIST_IMD", "monthly_rain": "ANNUAL_RAIN_IMD"})
)
imd["YEAR_IMD"] = IMD_YEAR

print(f"\nIMD after aggregation: {len(imd):,} districts, year={IMD_YEAR}")
print(imd.head(5).to_string(index=False))

# ── 4. Load master ─────────────────────────────────────────────────────────────
with open(DATA / "meta.json") as f:
    meta = json.load(f)

DISTRICT_COL = meta["DISTRICT_COL"]
YEAR_COL     = meta["YEAR_COL"]

master = pd.read_csv(DATA / "master_clean.csv")
master_districts = master[DISTRICT_COL].dropna().unique().tolist()

print(f"\nmaster_clean : {len(master):,} rows, "
      f"{len(master_districts)} unique districts (col: '{DISTRICT_COL}')")

# ── 5. Fuzzy-match IMD districts → master districts ───────────────────────────
print(f"\nFuzzy-matching {imd['DIST_IMD'].nunique()} IMD districts "
      f"(cutoff={SCORE_CUTOFF}) …")

mapping: dict[str, str | None] = {}
scores:  dict[str, int] = {}

for name in imd["DIST_IMD"].unique():
    name_s = str(name).strip()
    # Exact case-insensitive check first
    exact = next((d for d in master_districts
                  if d.strip().lower() == name_s.lower()), None)
    if exact:
        mapping[name] = exact
        scores[name]  = 100
    else:
        result = process.extractOne(name_s, master_districts,
                                    scorer=fuzz.token_sort_ratio)
        if result and result[1] >= SCORE_CUTOFF:
            mapping[name] = result[0]
            scores[name]  = result[1]
        else:
            mapping[name] = None
            scores[name]  = result[1] if result else 0

matched_n   = sum(1 for v in mapping.values() if v is not None)
unmatched   = [k for k, v in mapping.items() if v is None]

print(f"  Matched   : {matched_n}/{len(mapping)}")
if unmatched:
    print(f"  Unmatched ({len(unmatched)}): {unmatched[:20]}")

print("\nSample mappings (IMD district → master district, score):")
for name, matched in list(mapping.items())[:15]:
    print(f"  {name!r:35s} → {str(matched)!r:30s}  ({scores[name]})")

imd["DIST_MATCHED"] = imd["DIST_IMD"].map(mapping)

# ── 6. Left-join into master (broadcast single-year rainfall to all years) ────
# We intentionally drop YEAR_IMD: one rainfall value per district
# propagates to every year-row for that district in master_clean.
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

# ── 7. Coverage report ─────────────────────────────────────────────────────────
non_null = master_merged["ANNUAL_RAIN_IMD"].notna().sum()
total    = len(master_merged)

print(f"\n=== Merge result ===")
print(f"Total rows                            : {total:,}")
print(f"Rows with non-null ANNUAL_RAIN_IMD    : {non_null:,}  "
      f"({100*non_null/total:.1f}%)")
print(f"Rows still missing ANNUAL_RAIN_IMD    : {total - non_null:,}")
print(f"Unique districts WITH rainfall        : "
      f"{master_merged[master_merged['ANNUAL_RAIN_IMD'].notna()][DISTRICT_COL].nunique()}")
print(f"Unique districts MISSING rainfall     : "
      f"{master_merged[master_merged['ANNUAL_RAIN_IMD'].isna()][DISTRICT_COL].nunique()}")

# ── 8. Save ────────────────────────────────────────────────────────────────────
out_path = DATA / "master_with_imd_rain.csv"
master_merged.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")