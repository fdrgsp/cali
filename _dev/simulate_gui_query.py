import sys
from pathlib import Path
from sqlmodel import Session, create_engine, select, col

sys.path.append(str(Path(__file__).parent.parent / "src"))

from cali.sqlmodel._model import CaliResult, Traces, ROI, FOV, DataAnalysis

DB_PATH = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"

def simulate_gui_query():
    """Simulate the exact query the GUI would run."""
    if not Path(DB_PATH).exists():
        print(f"Database file not found at {DB_PATH}")
        return

    print(f"Connecting to database at {DB_PATH}")
    engine = create_engine(f"sqlite:///{DB_PATH}")

    fov_name = "B2_0000"
    run_id = 1
    rois = None  # Plot all
    active_only = False

    with Session(engine) as session:
        # This is the EXACT query from _plot_traces_data
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Filter by active if requested
        if active_only:
            stmt = stmt.where(col(ROI.active) == True)

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        results = session.exec(stmt).all()
        
        print(f"\nQuery Results:")
        print(f"  Total rows returned: {len(results)}")
        
        # Count unique ROI IDs
        roi_ids = [roi.id for roi, _, _ in results]
        unique_roi_ids = set(roi_ids)
        print(f"  Unique ROI IDs: {len(unique_roi_ids)}")
        print(f"  Duplicate trace issue: Each ROI appears {len(roi_ids) / len(unique_roi_ids):.1f} times on average")
        
        if results:
            print(f"\n  Sample of first 5 results:")
            for i, (roi, trace, data_analysis) in enumerate(results[:5]):
                print(f"    Row {i+1}:")
                print(f"      ROI ID: {roi.id}, Label: {roi.label_value}, Active: {roi.active}")
                print(f"      Trace ID: {trace.id if trace else None}, has dff: {trace.dff is not None if trace else False}")
                if trace and trace.dff:
                    print(f"        dff length: {len(trace.dff)}")
                    print(f"        dff sample (first 5): {trace.dff[:5]}")
        else:
            print("  NO RESULTS RETURNED!")
            
        # Check if there are ANY traces for this FOV at all
        all_traces_stmt = (
            select(Traces)
            .join(ROI, Traces.roi_id == ROI.id)
            .join(FOV, ROI.fov_id == FOV.id)
            .where(col(FOV.name) == fov_name)
        )
        all_traces = session.exec(all_traces_stmt).all()
        print(f"\n  Total traces for FOV {fov_name} (any run_id): {len(all_traces)}")
        
        # Check distinct analysis_result_ids
        if all_traces:
            result_ids = set(t.analysis_result_id for t in all_traces)
            print(f"  Distinct analysis_result_ids: {result_ids}")

if __name__ == "__main__":
    simulate_gui_query()
