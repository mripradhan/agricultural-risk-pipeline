"""
Merge dominant rice season into master_with_imd_rain and encode Season.
Saves to data/processed/master_enriched.csv.

Usage:
    python merge_season.py
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/processed"

SEASON_MAP: dict[str, int] = {
    "Kharif":     0,
    "Rabi":       1,
    "Whole Year": 2,
    "Summer":     3,
    "Winter":     4,
    "Autumn":     5,
}

# ── Load ──────────────────────────────────────────────────────────────────────
master = pd.read_csv(DATA / "master_with_imd_rain.csv")
rice   = pd.read_csv(DATA / "rice_dominant_season.csv")

# ── Rename to match master keys ───────────────────────────────────────────────
rice = rice.rename(columns={
    "District_matched": "Dist Name",
    "Crop_Year":        "Year",
})

rice["Year"]   = pd.to_numeric(rice["Year"],   errors="coerce").astype("Int64")
master["Year"] = pd.to_numeric(master["Year"], errors="coerce").astype("Int64")

rice_merge = (
    rice[["Dist Name", "Year", "Season"]]
    .drop_duplicates(subset=["Dist Name", "Year"])
)

# ── Left join ─────────────────────────────────────────────────────────────────
merged = master.merge(rice_merge, on=["Dist Name", "Year"], how="left")

# ── Encode Season_enc (NaN → -1) ──────────────────────────────────────────────
merged["Season_enc"] = (
    merged["Season"]
    .map(SEASON_MAP)
    .fillna(-1)
    .astype(int)
)

# ── Report ────────────────────────────────────────────────────────────────────
total    = len(merged)
non_null = merged["Season"].notna().sum()
pct      = 100 * non_null / total

print(f"Total rows                    : {total:,}")
print(f"Rows with non-null Season     : {non_null:,}  ({pct:.1f}%)")
print(f"Rows missing Season (enc=-1)  : {total - non_null:,}  ({100-pct:.1f}%)")

print("\nSeason_enc value_counts:")
rev_map = {v: k for k, v in SEASON_MAP.items()}
rev_map[-1] = "NaN / unmatched"
for enc, count in merged["Season_enc"].value_counts().sort_index().items():
    label = rev_map.get(int(enc), "?")
    print(f"  {enc:>3}  {label:<15}  {count:,}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = DATA / "master_enriched.csv"
merged.to_csv(out_path, index=False)
print(f"\n✅ Saved → {out_path}  ({total:,} rows, {len(merged.columns)} columns)")