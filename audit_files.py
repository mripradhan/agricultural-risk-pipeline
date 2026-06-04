"""
Audit and classify all project files.
Prints del commands for groups 2 and 3 — does NOT delete anything.
"""

from pathlib import Path

ROOT = Path(".")

GROUP2_SCRIPTS = [
    "inspect_crop_production.py",
    "merge_crop_production.py",
    "check_env.py",
]

GROUP3_CSV = [
    "data/processed/master_with_imd_rain.csv",
]

print("=" * 60)
print("GROUP 2 — Intermediate/debug scripts (safe to delete)")
print("=" * 60)
for f in GROUP2_SCRIPTS:
    p = ROOT / f
    exists = "EXISTS" if p.exists() else "NOT FOUND"
    print(f"  [{exists}] {f}")

print()
print("=" * 60)
print("GROUP 3 — Redundant CSV files (safe to delete)")
print("=" * 60)
for f in GROUP3_CSV:
    p = ROOT / f
    exists = "EXISTS" if p.exists() else "NOT FOUND"
    print(f"  [{exists}] {f}")

print()
print("=" * 60)
print("Windows del commands — paste into PowerShell to execute")
print("=" * 60)
for f in GROUP2_SCRIPTS + GROUP3_CSV:
    win_path = f.replace("/", "\\")
    print(f"del \"{win_path}\"")

print()
print("GROUP 4 — Unsure, flag for review:")
unsure = [r for r in ROOT.iterdir()
          if r.suffix in (".xls", ".xlsx")]
for f in unsure:
    print(f"  {f.name}  <- verify this is backed up before deleting")
