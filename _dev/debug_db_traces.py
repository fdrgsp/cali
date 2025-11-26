import sys
from pathlib import Path
from sqlmodel import Session, create_engine, select, func

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from cali.sqlmodel._model import CaliResult, Traces, ROI, FOV, DataAnalysis

DB_PATH = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"

def check_db():
    if not Path(DB_PATH).exists():
        print(f"Database file not found at {DB_PATH}")
        return

    print(f"Connecting to database at {DB_PATH}")
    engine = create_engine(f"sqlite:///{DB_PATH}")

    with Session(engine) as session:
        # Check Traces for Analysis Result 1
        traces_1 = session.exec(select(Traces).where(Traces.analysis_result_id == 1)).all()
        print(f"Found {len(traces_1)} Trace(s) for Analysis Result 1")
        
        if traces_1:
            roi_ids_1 = set(t.roi_id for t in traces_1)
            print(f"  Unique ROI IDs in Traces for Run 1: {len(roi_ids_1)}")
            print(f"  Sample ROI IDs: {list(roi_ids_1)[:10]}")
            
            # Check if these ROI IDs exist in ROI table
            rois = session.exec(select(ROI).where(ROI.id.in_(roi_ids_1))).all()
            print(f"  Found {len(rois)} matching ROIs in ROI table")
            
            if len(rois) != len(roi_ids_1):
                print("  WARNING: Some traces point to non-existent ROIs!")
                existing_roi_ids = set(r.id for r in rois)
                missing = roi_ids_1 - existing_roi_ids
                print(f"  Missing ROI IDs: {missing}")
                
            # Check VALID traces (those with existing ROIs)
            valid_traces = [t for t in traces_1 if t.roi_id in existing_roi_ids]
            print(f"\n  VALID traces (with existing ROIs): {len(valid_traces)}")
            
            # Check for position B2
            fovs_b2 = session.exec(select(FOV).where(FOV.name.like("B2%"))).all()
            print(f"\n  FOVs matching 'B2%': {len(fovs_b2)}")
            for fov in fovs_b2:
                print(f"    FOV: {fov.name}, position_index: {fov.position_index}, ID: {fov.id}")
                # Get ROIs for this FOV
                rois_in_fov = session.exec(select(ROI).where(ROI.fov_id == fov.id)).all()
                print(f"      ROIs: {len(rois_in_fov)}")
                # Get traces for these ROIs
                roi_ids_in_fov = [r.id for r in rois_in_fov]
                traces_in_fov = session.exec(select(Traces).where(
                    Traces.roi_id.in_(roi_ids_in_fov),
                    Traces.analysis_result_id == 1
                )).all()
                print(f"      Traces for these ROIs: {len(traces_in_fov)}")
                if traces_in_fov:
                    sample_trace = traces_in_fov[0]
                    print(f"      Sample trace - ROI ID: {sample_trace.roi_id}, has dff: {sample_trace.dff is not None}, dff len: {len(sample_trace.dff) if sample_trace.dff else 0}")

if __name__ == "__main__":
    check_db()
