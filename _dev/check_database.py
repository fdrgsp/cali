"""Check what's actually in the database."""

from pathlib import Path
from sqlalchemy import create_engine
from sqlmodel import Session

from cali.sqlmodel._model import CaliResult, AnalysisSettings, Mask

# Path to test database
db_path = Path("tests/test_data/evoked/results.cali")
engine = create_engine(f"sqlite:///{db_path}")

with Session(engine) as session:
    # Check all CaliResults
    results = session.query(CaliResult).all()
    print(f"Found {len(results)} CaliResults:")
    for r in results:
        print(f"  Run #{r.id}: analysis_settings={r.analysis_settings}")
        
        if r.analysis_settings:
            settings = session.get(AnalysisSettings, r.analysis_settings)
            if settings:
                print(f"    stimulation_mask_id={settings.stimulation_mask_id}")
    
    # Check all Masks
    print("\nAll Masks in database:")
    masks = session.query(Mask).all()
    print(f"Found {len(masks)} masks:")
    for m in masks:
        print(f"  Mask #{m.id}: type={m.mask_type}, shape=({m.height}, {m.width})")
        if m.mask_type == "stimulation":
            print(f"    → STIMULATION MASK coords_y length={len(m.coords_y) if m.coords_y else 0}")
