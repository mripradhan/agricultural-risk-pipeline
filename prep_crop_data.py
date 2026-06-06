"""
Prepare rice crop production data (Gayathri — Dataset 2).
Filters the Kaggle crop production CSV for Rice, picks the dominant season
per district-year (highest production), and saves rice_dominant_season.csv.

Usage:
    python prep_crop_data.py
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW  = BASE / "data/raw"
DATA = BASE / "data/processed"
DATA.mkdir(parents=True, exist_ok=True)

print("Loading raw crop production data...")
df = pd.read_csv(RAW / "crop_production_india.csv")

# Normalise column names
df.columns = df.columns.str.strip()

print(f"  Raw shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# ── Locate required columns dynamically ───────────────────────────────────────
cols_upper = {c.upper(): c for c in df.columns}

def _find(keywords: list[str]) -> str | None:
    for kw in keywords:
        for up, orig in cols_upper.items():
            if kw in up:
                return orig
    return None

dist_col   = _find(["DISTRICT"])
year_col   = _find(["CROP_YEAR", "YEAR"])
season_col = _find(["SEASON"])
crop_col   = _find(["CROP"])
area_col   = _find(["AREA"])
prod_col   = _find(["PRODUCTION"])
yield_col  = _find(["YIELD"])

for name, val in [("district", dist_col), ("year", year_col),
                   ("crop", crop_col),     ("production", prod_col)]:
    if val is None:
        raise ValueError(f"Could not find '{name}' column.  "
                         f"Available: {list(df.columns)}")

print(f"\nMapped: dist='{dist_col}' year='{year_col}' "
      f"season='{season_col}' crop='{crop_col}' "
      f"area='{area_col}' production='{prod_col}' yield='{yield_col}'")

# ── Filter for Rice ───────────────────────────────────────────────────────────
print(f"\nFiltering for Rice crops …")
rice = df[df[crop_col].astype(str).str.upper().str.contains("RICE")].copy()
print(f"  Rice rows: {len(rice):,}")

# ── Compute Yield if not present ──────────────────────────────────────────────
if yield_col is None or yield_col not in rice.columns:
    if area_col and prod_col:
        rice["Yield"] = (
            rice[prod_col].replace(0, pd.NA) /
            rice[area_col].replace(0, pd.NA)
        )
        yield_col = "Yield"
        print("  Computed Yield = Production / Area")
    else:
        rice["Yield"] = pd.NA
        yield_col = "Yield"

# ── Rename to canonical names ─────────────────────────────────────────────────
rename_map = {
    dist_col: "District",
    year_col: "Crop_Year",
}
if season_col:
    rename_map[season_col] = "Season"
if area_col:
    rename_map[area_col] = "Area"
if prod_col:
    rename_map[prod_col] = "Production"
rename_map[yield_col] = "Yield"

rice = rice.rename(columns=rename_map)

if "Season" not in rice.columns:
    rice["Season"] = "Unknown"

# ── Keep dominant season per district-year (highest production) ───────────────
rice["Production"] = pd.to_numeric(rice["Production"], errors="coerce").fillna(0)
rice = (
    rice
    .sort_values("Production", ascending=False)
    .groupby(["District", "Crop_Year"])
    .first()
    .reset_index()
)

keep_cols = [c for c in ["District", "Crop_Year", "Season", "Area",
                          "Production", "Yield"] if c in rice.columns]
rice = rice[keep_cols]

out_path = DATA / "rice_dominant_season.csv"
rice.to_csv(out_path, index=False)
print(f"\n✅ Created {out_path.name} with {len(rice):,} dominant season records")
print(f"   Season distribution:\n{rice['Season'].value_counts().to_string()}")