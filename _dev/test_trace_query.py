"""Test script to debug trace plotting query."""
from pathlib import Path

from sqlalchemy.engine import create_engine
from sqlmodel import Session, col, select

from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

# Database path
db_path = Path(
    "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
)

if not db_path.exists():
    print(f"❌ Database not found: {db_path}")
    exit(1)

# Create engine
engine = create_engine(f"sqlite:///{db_path}", echo=False)

# Test parameters
fov_name = "B2_0000"
run_id = 1

print(f"Testing trace query for FOV: {fov_name}, run_id: {run_id}")
print("=" * 80)

with Session(engine) as session:
    # Query database for ROI data
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
        .order_by(col(ROI.label_value))
    )

    results = session.exec(stmt).all()

    print(f"\n📊 Found {len(results)} results")
    print("=" * 80)

    if results:
        # Show first few results
        for i, (roi_model, trace_obj, data_analysis) in enumerate(results[:5], 1):
            print(f"\nResult {i}:")
            print(f"  ROI ID: {roi_model.id}, Label: {roi_model.label_value}")
            print(f"  Trace ID: {trace_obj.id if trace_obj else 'None'}")
            print(
                f"  Trace analysis_result_id: {trace_obj.analysis_result_id if trace_obj else 'None'}"
            )
            print(
                f"  DataAnalysis ID: {data_analysis.id if data_analysis else 'None'}"
            )
            print(
                f"  DataAnalysis analysis_result_id: {data_analysis.analysis_result_id if data_analysis else 'None'}"
            )

            if trace_obj:
                print(f"  Has raw_trace: {trace_obj.raw_trace is not None}")
                print(f"  Has corrected_trace: {trace_obj.corrected_trace is not None}")
                print(f"  Has dff: {trace_obj.dff is not None}")
                print(f"  Has dec_dff: {trace_obj.dec_dff is not None}")
                if trace_obj.raw_trace:
                    print(f"  Raw trace length: {len(trace_obj.raw_trace)}")
    else:
        print("\n❌ No results found!")
        print("\nLet's check what's in the database:")

        # Check FOVs
        fov_count = session.exec(select(FOV)).all()
        print(f"  Total FOVs: {len(fov_count)}")
        for fov in fov_count:
            print(f"    - {fov.name} (id={fov.id}, position={fov.position_index})")

        # Check ROIs
        roi_count = session.exec(select(ROI)).all()
        print(f"  Total ROIs: {len(roi_count)}")

        # Check Traces
        trace_count = session.exec(select(Traces)).all()
        print(f"  Total Traces: {len(trace_count)}")
        if trace_count:
            print("  First few traces:")
            for trace in trace_count[:5]:
                print(
                    f"    - Trace ID {trace.id}: roi_id={trace.roi_id}, "
                    f"analysis_result_id={trace.analysis_result_id}"
                )

engine.dispose()
print("\n✅ Test complete!")
