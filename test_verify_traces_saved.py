"""Test to verify that Traces are saved with analysis_result_id."""

from pathlib import Path

from sqlalchemy.orm import selectinload
from sqlmodel import Session, create_engine, select

from cali.sqlmodel._model import CaliResult, Traces

# Database path
output_path = Path(__file__).parent / "tests" / "test_data" / "evoked"
db_name = "test_neuropil_override.cali"
db_path = output_path / db_name

if not db_path.exists():
    print(f"Database not found at {db_path}")
    print("Run test_neuropil_override.py first")
    exit(1)

engine = create_engine(f"sqlite:///{db_path}")

with Session(engine) as session:
    # Get all CaliResults
    results = session.exec(select(CaliResult)).all()
    print(f"Found {len(results)} CaliResult entries:")
    for result in results:
        print(f"  ID={result.id}, detection_settings_id={result.detection_settings}, "
              f"analysis_settings_id={result.analysis_settings}")

    print()

    # Get all Traces
    all_traces = session.exec(
        select(Traces).options(selectinload(Traces.roi))  # type: ignore
    ).all()
    print(f"Found {len(all_traces)} Traces total:")
    for trace in all_traces:
        print(f"  Trace ID={trace.id}, ROI ID={trace.roi_id}, "
              f"analysis_result_id={trace.analysis_result_id}, "
              f"has_neuropil_mask={trace.neuropil_mask_id is not None}")

    print()

    # Check Traces for each analysis result
    for result in results:
        traces = session.exec(
            select(Traces)
            .where(Traces.analysis_result_id == result.id)
            .options(
                selectinload(Traces.roi),  # type: ignore
                selectinload(Traces.neuropil_mask),  # type: ignore
            )
        ).all()
        print(f"Analysis Result {result.id} has {len(traces)} Traces:")
        for trace in traces:
            neuropil_pixels = (
                len(trace.neuropil_mask.coords_y)
                if trace.neuropil_mask and trace.neuropil_mask.coords_y
                else 0
            )
            print(f"  ROI {trace.roi.label_value}: {neuropil_pixels} neuropil pixels")

engine.dispose()
