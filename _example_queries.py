"""Example queries demonstrating the fixed AnalysisResult linking.

This shows practical examples of how to query your data now that
Traces and DataAnalysis are properly linked to AnalysisResult.
"""

from sqlmodel import Session, create_engine, select

from cali.sqlmodel._model import (
    ROI,
    AnalysisResult,
    AnalysisSettings,
    DetectionSettings,
    Traces,
)

# Update this path to your actual database
DB_PATH = "/path/to/your/experiment.db"


def example_1_get_traces_by_settings(
    session: Session, detection_id: int, analysis_id: int
) -> list[Traces]:
    """Get all traces from a specific DetectionSettings + AnalysisSettings combo."""
    # First find the AnalysisResult with these settings
    ar = session.exec(
        select(AnalysisResult).where(
            AnalysisResult.detection_settings == detection_id,
            AnalysisResult.analysis_settings == analysis_id,
        )
    ).first()

    if not ar:
        print(f"No AnalysisResult found for Det={detection_id}, Ana={analysis_id}")
        return []

    # Get all traces from this analysis run
    traces = session.exec(
        select(Traces).where(Traces.analysis_result_id == ar.id)
    ).all()

    return list(traces)


def example_2_compare_detection_methods(
    session: Session, roi_id: int
) -> dict[int, Traces]:
    """Get traces for same ROI from different detection methods."""
    roi = session.exec(select(ROI).where(ROI.id == roi_id)).first()
    if not roi:
        return {}

    # Group traces by detection_settings_id
    traces_by_detection = {}
    for trace in roi.traces_history:
        if trace.analysis_result_id:
            ar = session.exec(
                select(AnalysisResult).where(
                    AnalysisResult.id == trace.analysis_result_id
                )
            ).first()
            if ar and ar.detection_settings:
                traces_by_detection[ar.detection_settings] = trace

    return traces_by_detection


def example_3_get_latest_analysis(session: Session, roi_id: int) -> Traces | None:
    """Get the most recent analysis for an ROI."""
    roi = session.exec(select(ROI).where(ROI.id == roi_id)).first()
    if not roi or not roi.traces_history:
        return None

    # Sort by created_at and return latest
    latest = sorted(roi.traces_history, key=lambda t: t.created_at, reverse=True)[0]
    return latest


def example_4_get_analysis_metadata(session: Session, trace_id: int) -> dict[str, any]:
    """Get full metadata about how a trace was generated."""
    trace = session.exec(select(Traces).where(Traces.id == trace_id)).first()
    if not trace or not trace.analysis_result_id:
        return {}

    # Get the AnalysisResult
    ar = session.exec(
        select(AnalysisResult).where(AnalysisResult.id == trace.analysis_result_id)
    ).first()
    if not ar:
        return {}

    # Get the settings
    detection = None
    if ar.detection_settings:
        detection = session.exec(
            select(DetectionSettings).where(
                DetectionSettings.id == ar.detection_settings
            )
        ).first()

    analysis = session.exec(
        select(AnalysisSettings).where(AnalysisSettings.id == ar.analysis_settings)
    ).first()

    return {
        "analysis_result_id": ar.id,
        "experiment_id": ar.experiment,
        "detection_method": detection.method if detection else None,
        "detection_model": detection.model_type if detection else None,
        "analysis_decay_constant": analysis.decay_constant if analysis else None,
        "analysis_dff_window": analysis.dff_window if analysis else None,
        "positions_analyzed": ar.positions_analyzed,
        "trace_created_at": trace.created_at,
    }


def example_5_filter_by_analysis_params(
    session: Session, dff_window: int
) -> list[Traces]:
    """Get all traces generated with a specific analysis parameter."""
    # Find all AnalysisSettings with the specified dff_window
    settings = session.exec(
        select(AnalysisSettings).where(AnalysisSettings.dff_window == dff_window)
    ).all()

    if not settings:
        return []

    # Get all AnalysisResults using these settings
    results = []
    for setting in settings:
        ars = session.exec(
            select(AnalysisResult).where(AnalysisResult.analysis_settings == setting.id)
        ).all()
        results.extend(ars)

    # Get all traces from these AnalysisResults
    all_traces = []
    for ar in results:
        traces = session.exec(
            select(Traces).where(Traces.analysis_result_id == ar.id)
        ).all()
        all_traces.extend(traces)

    return all_traces


if __name__ == "__main__":
    engine = create_engine(f"sqlite:///{DB_PATH}")

    with Session(engine) as session:
        print("=" * 80)
        print("EXAMPLE QUERIES")
        print("=" * 80)

        # Example 1
        print("\n1. Get traces by settings combination:")
        traces = example_1_get_traces_by_settings(
            session, detection_id=1, analysis_id=1
        )
        print(
            f"   Found {len(traces)} traces with DetectionSettings=1, AnalysisSettings=1"
        )

        # Example 2
        print("\n2. Compare detection methods for same ROI:")
        roi_id = 1  # Change to actual ROI ID
        traces_by_det = example_2_compare_detection_methods(session, roi_id)
        for det_id, trace in traces_by_det.items():
            print(f"   DetectionSettings {det_id}: Trace ID {trace.id}")

        # Example 3
        print("\n3. Get latest analysis for ROI:")
        latest_trace = example_3_get_latest_analysis(session, roi_id=1)
        if latest_trace:
            print(
                f"   Latest: Trace ID {latest_trace.id}, "
                f"created {latest_trace.created_at}"
            )

        # Example 4
        print("\n4. Get full metadata for a trace:")
        metadata = example_4_get_analysis_metadata(session, trace_id=1)
        for key, value in metadata.items():
            print(f"   {key}: {value}")

        # Example 5
        print("\n5. Filter by analysis parameter (dff_window=300):")
        traces = example_5_filter_by_analysis_params(session, dff_window=300)
        print(f"   Found {len(traces)} traces with dff_window=300")
