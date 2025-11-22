from sqlmodel import Session, create_engine, select
from cali.sqlmodel._model import AnalysisResult, DetectionSettings, AnalysisSettings

db_path = "/Users/fdrgsp/Desktop/cali_test/testcalidb"
engine = create_engine(f"sqlite:///{db_path}")

with Session(engine) as session:
    # Get all AnalysisResults
    results = session.exec(select(AnalysisResult).order_by(AnalysisResult.id)).all()

    print("\n" + "=" * 100)
    print("ALL ANALYSIS RESULTS - TABLE OF RUNS")
    print("=" * 100)
    print(
        f"{'ID':<5} {'Created At':<20} {'Detection ID':<15} {'Analysis ID':<15} {'Positions':<15}"
    )
    print("-" * 100)

    for result in results:
        created_at = result.created_at.strftime("%Y-%m-%d %H:%M:%S")
        detection_id = (
            str(result.detection_settings) if result.detection_settings else "None"
        )
        analysis_id = (
            str(result.analysis_settings) if result.analysis_settings else "None"
        )
        positions = (
            str(result.positions_analyzed) if result.positions_analyzed else "None"
        )

        print(
            f"{result.id:<5} {created_at:<20} {detection_id:<15} {analysis_id:<15} {positions:<15}"
        )

    print("=" * 100)
    print(f"\nTotal AnalysisResults: {len(results)}")

    # # Now show details of each detection and analysis settings
    # print("\n" + "=" * 100)
    # print("DETECTION SETTINGS DETAILS")
    # print("=" * 100)
    # print(
    #     f"{'ID':<5} {'Method':<15} {'Model Type':<15} {'Diameter':<10} {'Flow Threshold':<15}"
    # )
    # print("-" * 100)

    # detection_settings = session.exec(
    #     select(DetectionSettings).order_by(DetectionSettings.id)
    # ).all()
    # for ds in detection_settings:
    #     diameter = str(ds.diameter) if ds.diameter else "default"
    #     flow_thresh = str(ds.flow_threshold) if ds.flow_threshold else "default"
    #     print(
    #         f"{ds.id:<5} {ds.method:<15} {ds.model_type:<15} {diameter:<10} {flow_thresh:<15}"
    #     )

    # print("\n" + "=" * 100)
    # print("ANALYSIS SETTINGS DETAILS")
    # print("=" * 100)
    # print(f"{'ID':<5} {'Threads':<10} {'DFF Window':<15} {'Burst Threshold':<20}")
    # print("-" * 100)

    # analysis_settings = session.exec(
    #     select(AnalysisSettings).order_by(AnalysisSettings.id)
    # ).all()
    # for anas in analysis_settings:
    #     print(
    #         f"{anas.id:<5} {anas.threads:<10} {anas.dff_window:<15} {anas.burst_threshold:<20}"
    #     )

    # print("=" * 100)
    # print("\n📊 SUMMARY:")
    # print(
    #     f"  - Detection-only runs (no analysis): {sum(1 for r in results if r.analysis_settings is None)}"
    # )
    # print(
    #     f"  - Full analysis runs: {sum(1 for r in results if r.analysis_settings is not None)}"
    # )
    # print(f"  - Unique detection settings: {len(detection_settings)}")
    # print(f"  - Unique analysis settings: {len(analysis_settings)}")

engine.dispose(close=True)
