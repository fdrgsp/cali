"""Test make_transient vs expunge for multi-threaded access."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from sqlmodel import Session, create_engine, select
from sqlalchemy.orm import joinedload, make_transient

from cali.sqlmodel._model import FOV, ROI, Traces

DB_PATH = Path(
    "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
)


def access_mask_in_thread(fov):
    """Access mask data in a different thread."""
    errors = []
    for roi in fov.rois[:5]:
        try:
            if roi.roi_mask:
                coords_len = len(roi.roi_mask.coords_y)
                print(f"  ✅ ROI {roi.label_value}: coords_y length = {coords_len}")
        except Exception as e:
            errors.append(f"  ❌ ROI {roi.label_value}: {type(e).__name__}: {e}")
    
    return errors


if __name__ == "__main__":
    engine = create_engine(f"sqlite:///{DB_PATH}")
    
    detection_settings_id = 1
    pos_idx = 0
    
    with Session(engine) as session:
        # Load with joinedload
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
        
        if not fov:
            print("❌ No FOV found")
            exit(1)
        
        print(f"✅ Loaded FOV: {fov.name}, {len(fov.rois)} ROIs")
        
        # Force load all attributes
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
        
        print("✅ All attributes force-loaded")
        
        # Make transient
        make_transient(fov)
        for roi in fov.rois:
            make_transient(roi)
            if roi.roi_mask:
                make_transient(roi.roi_mask)
            for trace in roi.traces_history:
                make_transient(trace)
                if trace.neuropil_mask:
                    make_transient(trace.neuropil_mask)
            for da in roi.data_analysis_history:
                make_transient(da)
        
        print("✅ All objects made transient")
    
    # Session is now closed - test access in different thread
    print("\n🧵 Testing access from ThreadPoolExecutor...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(access_mask_in_thread, fov)
        errors = future.result()
    
    if errors:
        print("\n❌ ERRORS OCCURRED:")
        for error in errors:
            print(error)
    else:
        print("\n✅ SUCCESS: No errors in multi-threaded access!")
