import pandas as pd
from pathlib import Path

print("Loading raw crop production data...")
df = pd.read_csv("data/raw/crop_production_india.csv")

# Strip whitespace from column names just in case
df.columns = df.columns.str.strip()

# Dynamically find the District and Year columns
dist_col = next((c for c in df.columns if 'DISTRICT' in c.upper()), None)
year_col = next((c for c in df.columns if 'YEAR' in c.upper()), None)

print(f"Filtering for Rice crops... (found {dist_col} and {year_col})")
rice = df[df["Crop"].astype(str).str.upper().str.contains("RICE")].copy()

# Calculate Yield if it doesn't explicitly exist in the Kaggle file
if "Yield" not in rice.columns:
    rice["Yield"] = rice["Production"] / rice["Area"].replace(0, pd.NA)

# Rename to the exact column names expected by fix_crop_district_match.py
rice = rice.rename(columns={
    dist_col: "District",
    year_col: "Crop_Year"
})

# Keep only the dominant season for each district-year (highest production)
rice = rice.sort_values("Production", ascending=False).groupby(["District", "Crop_Year"]).first().reset_index()

out_path = Path("data/processed/rice_dominant_season.csv")
rice.to_csv(out_path, index=False)
print(f"✅ Created {out_path.name} with {len(rice)} dominant season records!")