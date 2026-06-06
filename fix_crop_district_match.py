"""
Improved fuzzy district matching for rice crop production data.
Applies normalisation → manual aliases → fuzzy match at cutoff=75.
Overwrites data/processed/rice_dominant_season.csv in-place.

Usage:
    python fix_crop_district_match.py [--cutoff 75]
"""

import re
import argparse
import pandas as pd
from fuzzywuzzy import process, fuzz
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cutoff", type=int, default=75,
                    help="Fuzzy-match score threshold (default: 75)")
args, _ = parser.parse_known_args()
SCORE_CUTOFF = args.cutoff

BASE = Path(__file__).resolve().parent
DATA = BASE / "data/processed"

# ── Manual aliases: known mismatches between Kaggle and ICRISAT ───────────────
ALIASES: dict[str, str] = {
    "PURBA BARDHAMAN":    "Purba Bardhaman",
    "MEDINIPUR WEST":     "Paschim Medinipur",
    "MEDINIPUR EAST":     "Purba Medinipur",
    "COOCHBEHAR":         "Koch Bihar",
    "MYSURU":             "Mysore",
    "BALLARI":            "Bellary",
    "SHIVAMOGGA":         "Shimoga",
    "PASHCHIM CHAMPARAN": "West Champaran",
    "KUSHI NAGAR":        "Kushinagar",
    "AMETHI":             "Sultanpur",
    "SIDDHARTH NAGAR":    "Siddharthnagar",
    "MUKTSAR":            "Sri Muktsar Sahib",
    "SANT RAVIDAS NAGAR": "Bhadohi",
    "LAHUL AND SPITI":    "Lahaul Spiti",
    "JYOTIBA PHULE NAGAR":"Amroha",
    "MAHAMAYA NAGAR":     "Hathras",
    "KANSHIRAM NAGAR":    "Kasganj",
}

def _normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()

# ── Load data ─────────────────────────────────────────────────────────────────
rice   = pd.read_csv(DATA / "rice_dominant_season.csv")
master = pd.read_csv(DATA / "master_with_imd_rain.csv")

rice["District"] = rice["District"].str.strip()
master_districts = master["Dist Name"].dropna().str.strip().unique().tolist()

# Build normalised lookup
master_norm      = {_normalize(d): d for d in master_districts}
master_norm_keys = list(master_norm.keys())

crop_districts = rice["District"].unique()
print(f"Crop districts to match : {len(crop_districts)}")
print(f"Master districts        : {len(master_districts)}")

# ── Match ─────────────────────────────────────────────────────────────────────
mapping: dict[str, str | None] = {}
scores:  dict[str, int]        = {}
methods: dict[str, str]        = {}

aliases_upper = {k.upper(): v for k, v in ALIASES.items()}

for raw in crop_districts:
    raw_stripped = raw.strip()

    # 1. Manual alias
    alias_target = aliases_upper.get(raw_stripped.upper())
    if alias_target and alias_target in master_districts:
        mapping[raw] = alias_target
        scores[raw]  = 100
        methods[raw] = "alias"
        continue

    # 2. Exact normalised match
    norm_raw = _normalize(raw_stripped)
    if norm_raw in master_norm:
        mapping[raw] = master_norm[norm_raw]
        scores[raw]  = 100
        methods[raw] = "exact-norm"
        continue

    # 3. Fuzzy match on normalised strings
    result = process.extractOne(norm_raw, master_norm_keys,
                                scorer=fuzz.token_sort_ratio)
    if result and result[1] >= SCORE_CUTOFF:
        mapping[raw] = master_norm[result[0]]
        scores[raw]  = result[1]
        methods[raw] = "fuzzy"
    else:
        mapping[raw] = None
        scores[raw]  = result[1] if result else 0
        methods[raw] = "unmatched"

# ── Report ────────────────────────────────────────────────────────────────────
matched   = {k: v for k, v in mapping.items() if v is not None}
unmatched = {k: v for k, v in mapping.items() if v is None}

print(f"\nMatched  : {len(matched)}/{len(crop_districts)}")
print(f"Unmatched: {len(unmatched)}")
print(f"\nMatch method breakdown:")
print(pd.Series(methods).value_counts().to_string())

if unmatched:
    print(f"\nStill unmatched ({len(unmatched)}) — top scores shown:")
    for name in sorted(unmatched, key=lambda x: -scores[x])[:20]:
        print(f"  {name!r:40s}  best score={scores[name]}")

print("\n10 sample mappings (raw → matched, score, method):")
for name in list(matched.keys())[:10]:
    print(f"  {name!r:35s} → {mapping[name]!r:35s}  "
          f"({scores[name]:3d}, {methods[name]})")

# ── Save ──────────────────────────────────────────────────────────────────────
rice["District_matched"] = rice["District"].map(mapping)

keep_cols = [c for c in ["District_matched", "Crop_Year", "Season",
                          "Area", "Production", "Yield"]
             if c in rice.columns]
out = (
    rice[rice["District_matched"].notna()][keep_cols]
    .reset_index(drop=True)
)

out_path = DATA / "rice_dominant_season.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out):,} matched rows → {out_path}")
print(f"Dropped {len(rice) - len(out):,} rows (unmatched districts)")