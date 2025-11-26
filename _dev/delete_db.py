"""Quick test to delete database and re-run analysis."""
from pathlib import Path

# Delete the database
db_path = Path(
    "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
)

if db_path.exists():
    db_path.unlink()
    print(f"✅ Deleted {db_path}")
else:
    print(f"❌ Database not found: {db_path}")

print("\nNow re-run your analysis to test the fixes!")
