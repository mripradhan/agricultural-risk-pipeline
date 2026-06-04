"""
Improved fuzzy district matching for rice crop production data.
Applies normalization, manual aliases, then fuzzy match at cutoff=75.
Overwrites data/processed/rice_dominant_season.csv.
"""

import re
import pandas as pd
from fuzzywuzzy import process, fuzz

SCORE_CUTOFF = 75

ALIASES = {
    "PURBA BARDHAMAN":   "Purba Bardhaman",
    "MEDINIPUR WEST":    "Paschim Medinipur",
    "MEDINIPUR EAST":    "Purba Medinipur",
    "COOCHBEHAR":        "Koch Bihar",
    "MYSURU":            "Mysore",
    "BALLARI":           "Bellary",
    "SHIVAMOGGA":        "Shimoga",
    "PASHCHIM CHAMPARAN":"West Champaran",
    "KUSHI NAGAR":       "Kushinagar",
    "AMETHI":            "Sultanpur",
    "SIDDHARTH NAGAR":   "Siddharthnagar",
    "MUKTSAR":           "Sri Muktsar Sahib",
}

def normalize(name):
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()

# ── Load data ─────────────────────────────────────────────────────────────────
rice = pd.read_csv("data/processed/rice_dominant_season.csv")
rice["District"] = rice["District"].str.strip()

master = pd.read_csv("data/processed/master_with_imd_rain.csv")
master_districts = master["Dist Name"].dropna().str.strip().unique().tolist()

# Build normalised lookup: normalised master name → original master name
master_norm = {normalize(d): d for d in master_districts}
master_norm_keys = list(master_norm.keys())

crop_districts = rice["District"].unique()
print(f"Crop districts to match : {len(crop_districts)}")
print(f"Master districts        : {len(master_districts)}")

# ── Match ─────────────────────────────────────────────────────────────────────
mapping, scores, methods = {}, {}, {}

for raw in crop_districts:
    raw_stripped = raw.strip()

    # 1. Manual alias (checked against original casing)
    if raw_stripped.upper() in {k.upper(): k for k in ALIASES}:
        alias_key = next(k for k in ALIASES if k.upper() == raw_stripped.upper())
        target = ALIASES[alias_key]
        if target in master_districts:
            mapping[raw] = target
            scores[raw]  = 100
            methods[raw] = "alias"
            continue
        # alias target not in master — fall through to fuzzy

    # 2. Exact normalised match (handles pure case differences)
    norm_raw = normalize(raw_stripped)
    if norm_raw in master_norm:
        mapping[raw] = master_norm[norm_raw]
        scores[raw]  = 100
        methods[raw] = "exact-norm"
        continue

    # 3. Fuzzy match on normalised strings
    result = process.extractOne(
        norm_raw, master_norm_keys, scorer=fuzz.token_sort_ratio
    )
    if result and result[1] >= SCORE_CUTOFF:
        mapping[raw] = master_norm[result[0]]   # map back to original casing
        scores[raw]  = result[1]
        methods[raw] = "fuzzy"
    else:
        mapping[raw] = None
        scores[raw]  = result[1] if result else 0
        methods[raw] = "unmatched"

# ── Report ────────────────────────────────────────────────────────────────────
matched   = {k: v for k, v in mapping.items() if v is not None}
unmatched = {k: v for k, v in mapping.items() if v is None}

method_counts = pd.Series(methods).value_counts()
print(f"\nMatched  : {len(matched)}/{len(crop_districts)}")
print(f"Unmatched: {len(unmatched)}")
print(f"\nMatch method breakdown:\n{method_counts.to_string()}")

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

out = (
    rice[rice["District_matched"].notna()]
    [["District_matched", "Crop_Year", "Season", "Area", "Production", "Yield"]]
    .reset_index(drop=True)
)

out_path = "data/processed/rice_dominant_season.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved {len(out):,} matched rows → {out_path}")
print(f"Dropped {len(rice) - len(out):,} rows (unmatched districts)")
