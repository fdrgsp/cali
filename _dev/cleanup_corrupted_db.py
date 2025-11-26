#!/usr/bin/env python3
"""
Script to delete corrupted database and prepare for clean re-run.

The database at /Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali
is corrupted with:
1. Orphaned traces (traces pointing to non-existent ROIs)
2. Duplicate traces (multiple traces per ROI for the same analysis_result_id)

This script will:
1. Backup the corrupted database
2. Delete it
3. Display instructions for re-running the analysis
"""

import shutil
from pathlib import Path
from datetime import datetime

DB_PATH = Path("/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali")

def main():
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        return

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = DB_PATH.with_name(f"{DB_PATH.stem}_CORRUPTED_BACKUP_{timestamp}.cali")
    
    print(f"📦 Creating backup...")
    print(f"  From: {DB_PATH}")
    print(f"  To:   {backup_path}")
    shutil.copy2(DB_PATH, backup_path)
    print(f"✅ Backup created")

    # Delete original
    print(f"\n🗑️  Deleting corrupted database...")
    DB_PATH.unlink()
    print(f"✅ Database deleted")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("\n1. Re-run your analysis script")
    print("   The fixes are now in place to prevent:")
    print("   - Duplicate ROIs during detection")
    print("   - Duplicate Traces during analysis")
    print("   - Orphaned traces (via foreign key enforcement)")
    print("\n2. The new database will be created cleanly")
    print("\n3. Traces should now display properly in the GUI")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
