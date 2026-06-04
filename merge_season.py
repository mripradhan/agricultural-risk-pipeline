"""
Merge dominant rice season into master_with_imd_rain and encode Season.
Saves to data/processed/master_enriched.csv.
"""

import pandas as pd

SEASON_MAP = {
    "Kharif":     0,
    "Rabi":       1,
    "Whole Year": 2,
    "Summer":     3,
    "Winter":     4,
    "Autumn":     5,
}

# ── Load ──────────────────────────────────────────────────────────────────────
master = pd.read_csv("data/processed/master_with_imd_rain.csv")
rice   = pd.read_csv("data/processed/rice_dominant_season.csv")

# ── 1. Rename to match master keys ────────────────────────────────────────────
rice = rice.rename(columns={
    "District_matched": "Dist Name",
    "Crop_Year":        "Year",
})
rice["Year"] = rice["Year"].astype(int)
master["Year"] = master["Year"].astype(int)

# Keep only the columns needed for the merge
rice_merge = rice[["Dist Name", "Year", "Season"]].drop_duplicates(
    subset=["Dist Name", "Year"]
)

# ── 2. Left join on Dist Name + Year ─────────────────────────────────────────
merged = master.merge(rice_merge, on=["Dist Name", "Year"], how="left")

# ── 3. Encode Season → Season_enc (NaN → -1) ─────────────────────────────────
merged["Season_enc"] = (
    merged["Season"]
    .map(SEASON_MAP)
    .fillna(-1)
    .astype(int)
)

# ── 4. Report ─────────────────────────────────────────────────────────────────
total     = len(merged)
non_null  = merged["Season"].notna().sum()
pct       = 100 * non_null / total

print(f"Total rows                    : {total:,}")
print(f"Rows with non-null Season     : {non_null:,}  ({pct:.1f}%)")
print(f"Rows missing Season (enc=-1)  : {total - non_null:,}  ({100-pct:.1f}%)")

print("\nSeason_enc value_counts:")
vc = merged["Season_enc"].value_counts().sort_index()
rev_map = {v: k for k, v in SEASON_MAP.items()}
rev_map[-1] = "NaN / unmatched"
for enc, count in vc.items():
    label = rev_map.get(enc, "?")
    print(f"  {enc:>3}  {label:<15}  {count:,}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
out_path = "data/processed/master_enriched.csv"
merged.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}  ({total:,} rows, {len(merged.columns)} columns)")
