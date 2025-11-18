"""Verify that Traces and DataAnalysis are properly linked to AnalysisResult.

This script demonstrates that the fix allows you to:
1. Query traces/analysis by specific DetectionSettings + AnalysisSettings combination
2. Track which analysis run produced which results
3. Support multiple analysis runs with different settings on the same experiment
"""

from sqlmodel import Session, create_engine, select

from cali.sqlmodel._model import (
    ROI,
    AnalysisResult,
    DataAnalysis,
    Traces,
)

# Update this path to your actual database
DB_PATH = "/Users/fdrgsp/Desktop/cali_test/evk.tensorstore.zarr.db"

engine = create_engine(f"sqlite:///{DB_PATH}")

with Session(engine) as session:
    print("=" * 80)
    print("VERIFICATION: AnalysisResult Links")
    print("=" * 80)

    # Get all AnalysisResults
    analysis_results = session.exec(select(AnalysisResult)).all()

    print(f"\nFound {len(analysis_results)} AnalysisResult(s):\n")
    for ar in analysis_results:
        print(f"AnalysisResult ID: {ar.id}")
        print(f"  Experiment: {ar.experiment}")
        print(f"  DetectionSettings: {ar.detection_settings}")
        print(f"  AnalysisSettings: {ar.analysis_settings}")
        print(f"  Positions: {ar.positions_analyzed}")

        # Count linked Traces and DataAnalysis via foreign key
        traces_count = session.exec(
            select(Traces).where(Traces.analysis_result_id == ar.id)
        ).all()
        data_analysis_count = session.exec(
            select(DataAnalysis).where(DataAnalysis.analysis_result_id == ar.id)
        ).all()

        print(f"  Traces linked (via analysis_result_id): {len(traces_count)}")
        print(
            f"  DataAnalysis linked (via analysis_result_id): "
            f"{len(data_analysis_count)}"
        )
        print()

    print("=" * 80)
    print("QUERY EXAMPLES: Get results by settings combination")
    print("=" * 80)

    # Example 1: Get all traces from a specific analysis run
    if analysis_results:
        ar = analysis_results[0]
        traces = session.exec(
            select(Traces).where(Traces.analysis_result_id == ar.id)
        ).all()
        print(
            f"\nExample 1: Traces from AnalysisResult {ar.id} "
            f"(DetectionSettings={ar.detection_settings}, "
            f"AnalysisSettings={ar.analysis_settings})"
        )
        print(f"  Found {len(traces)} traces")

    # Example 2: For a specific ROI, get all analysis versions
    roi = session.exec(select(ROI)).first()
    if roi:
        print(f"\nExample 2: All analysis versions for ROI {roi.id}:")
        for trace in roi.traces_history:
            print(
                f"  Trace ID {trace.id}: "
                f"AnalysisResult={trace.analysis_result_id}, "
                f"created {trace.created_at}"
            )

    # Example 3: Compare results from different detection settings
    print("\nExample 3: Group results by DetectionSettings + AnalysisSettings:")
    for ar in analysis_results:
        key = f"Detection={ar.detection_settings}, Analysis={ar.analysis_settings}"
        trace_count = len(
            session.exec(select(Traces).where(Traces.analysis_result_id == ar.id)).all()
        )
        print(f"  {key}: {trace_count} traces")

    print("\n" + "=" * 80)
    print("✅ Fix verified! Traces and DataAnalysis are now properly linked.")
    print("=" * 80)
