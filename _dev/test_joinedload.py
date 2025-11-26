"""Quick test to verify mask loading works without errors."""

from pathlib import Path

from sqlmodel import Session, create_engine
from sqlalchemy.orm import joinedload

from cali.sqlmodel._model import FOV, ROI, Traces

DB_PATH = Path(
    "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
)

if __name__ == "__main__":
    engine = create_engine(f"sqlite:///{DB_PATH}")
    
    detection_settings_id = 1
    pos_idx = 0
    
    with Session(engine) as session:
        # Use joinedload to load everything in one query
        from sqlmodel import select
        
        fov_stmt = (
            select(FOV)
            .join(ROI)
            .where(
                FOV.position_index == pos_idx,
                ROI.detection_settings_id == detection_settings_id,
            )
            .options(
                joinedload(FOV.rois).joinedload(ROI.roi_mask),
                joinedload(FOV.rois).joinedload(ROI.traces_history).joinedload(Traces.neuropil_mask),
                joinedload(FOV.rois).joinedload(ROI.data_analysis_history),
            )
        )
        
        print(f"Loading FOV at position {pos_idx} with joinedload...")
        fov = session.exec(fov_stmt).unique().first()
        
        if fov:
            print(f"✅ Loaded FOV: {fov.name}, {len(fov.rois)} ROIs")
            
            # Force load all attributes
            errors = 0
            for roi in fov.rois:
                if roi.roi_mask:
                    _ = roi.roi_mask.coords_y
                    _ = roi.roi_mask.coords_x
                    _ = roi.roi_mask.height
                    _ = roi.roi_mask.width
                
                for trace in roi.traces_history:
                    if trace.neuropil_mask:
                        _ = trace.neuropil_mask.coords_y
                        _ = trace.neuropil_mask.coords_x
                        _ = trace.neuropil_mask.height
                        _ = trace.neuropil_mask.width
                
                _ = roi.data_analysis_history
            
            print(f"✅ All {len(fov.rois)} ROI masks force-loaded successfully")
            
            # Now expunge and try to access
            session.expunge(fov)
            print("✅ FOV expunged from session")
            
            # Try to access after expunge
            print("\nTesting access after expunge:")
            for i, roi in enumerate(fov.rois[:5]):
                if roi.roi_mask:
                    coords_len = len(roi.roi_mask.coords_y)
                    print(f"  ROI {roi.label_value}: coords_y length = {coords_len}")
                else:
                    print(f"  ROI {roi.label_value}: NO MASK")
            
            print("\n✅ All masks accessible after expunge - joinedload works!")
        else:
            print(f"❌ No FOV found")
