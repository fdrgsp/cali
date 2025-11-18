"""Check if Traces and DataAnalysis are properly linked to AnalysisResult."""

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
    # Get all AnalysisResults
    analysis_results = session.exec(select(AnalysisResult)).all()

    print("=" * 80)
    print("AnalysisResult Records:")
    print("=" * 80)
    for ar in analysis_results:
        print(f"\nAnalysisResult ID: {ar.id}")
        print(f"  Experiment: {ar.experiment}")
        print(f"  DetectionSettings: {ar.detection_settings}")
        print(f"  AnalysisSettings: {ar.analysis_settings}")
        print(f"  Positions: {ar.positions_analyzed}")
        print(f"  Traces linked via relationship: {len(ar.traces)}")
        print(
            f"  DataAnalysis linked via relationship: {len(ar.data_analysis_results)}"
        )

    print("\n" + "=" * 80)
    print("Traces Records:")
    print("=" * 80)
    traces_list = session.exec(select(Traces)).all()
    print(f"Total Traces: {len(traces_list)}")

    traces_with_analysis_result = [
        t for t in traces_list if t.analysis_result_id is not None
    ]
    traces_without_analysis_result = [
        t for t in traces_list if t.analysis_result_id is None
    ]

    print(f"  With analysis_result_id: {len(traces_with_analysis_result)}")
    print(f"  WITHOUT analysis_result_id: {len(traces_without_analysis_result)}")

    if traces_without_analysis_result:
        print("\n  Sample Traces without analysis_result_id:")
        for t in traces_without_analysis_result[:5]:
            roi = session.exec(select(ROI).where(ROI.id == t.roi_id)).first()
            print(f"    Trace ID {t.id}: ROI {t.roi_id}, created {t.created_at}")
            if roi:
                print(f"      ROI detection_settings_id: {roi.detection_settings_id}")

    print("\n" + "=" * 80)
    print("DataAnalysis Records:")
    print("=" * 80)
    data_analysis_list = session.exec(select(DataAnalysis)).all()
    print(f"Total DataAnalysis: {len(data_analysis_list)}")

    da_with_analysis_result = [
        d for d in data_analysis_list if d.analysis_result_id is not None
    ]
    da_without_analysis_result = [
        d for d in data_analysis_list if d.analysis_result_id is None
    ]

    print(f"  With analysis_result_id: {len(da_with_analysis_result)}")
    print(f"  WITHOUT analysis_result_id: {len(da_without_analysis_result)}")

    if da_without_analysis_result:
        print("\n  Sample DataAnalysis without analysis_result_id:")
        for d in da_without_analysis_result[:5]:
            roi = session.exec(select(ROI).where(ROI.id == d.roi_id)).first()
            print(f"    DataAnalysis ID {d.id}: ROI {d.roi_id}, created {d.created_at}")
            if roi:
                print(f"      ROI detection_settings_id: {roi.detection_settings_id}")

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(
        f"The current implementation is {'CORRECT' if len(traces_without_analysis_result) == 0 else 'BROKEN'}!"
    )
    print("\nExpected behavior:")
    print(
        "  - Each Trace should have analysis_result_id pointing to the AnalysisResult"
    )
    print(
        "  - Each DataAnalysis should have analysis_result_id pointing to the AnalysisResult"
    )
    print(
        "  - This allows you to query which traces/analysis came from which analysis run"
    )
    print("\nCurrent issue:")
    if traces_without_analysis_result or da_without_analysis_result:
        print("  ❌ Traces and DataAnalysis are NOT linked to AnalysisResult")
        print("  ❌ You cannot determine which analysis run produced which results")
        print("  ❌ Multiple analysis runs will overwrite each other's data")
