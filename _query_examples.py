"""Query examples for versioned analysis results.

This module demonstrates how to query traces and data_analysis results
when you have multiple analysis runs with different settings.
"""

from sqlmodel import Session, select

from src.cali.sqlmodel._model import AnalysisResult, DataAnalysis, ROI, Traces

# ============================================================================
# Query Pattern 1: Get results for a specific AnalysisResult
# ============================================================================


def get_traces_for_analysis_result(
    session: Session, analysis_result_id: int
) -> list[Traces]:
    """Get all traces from a specific analysis run.

    Parameters
    ----------
    session : Session
        Database session
    analysis_result_id : int
        ID of the AnalysisResult to query

    Returns
    -------
    list[Traces]
        All trace results from this analysis run

    Example
    -------
    >>> # Get traces from analysis run #5
    >>> traces = get_traces_for_analysis_result(session, analysis_result_id=5)
    >>> for trace in traces:
    ...     print(f"ROI {trace.roi_id}: {len(trace.dff)} frames")
    """
    return session.exec(
        select(Traces).where(Traces.analysis_result_id == analysis_result_id)
    ).all()


def get_data_analysis_for_analysis_result(
    session: Session, analysis_result_id: int
) -> list[DataAnalysis]:
    """Get all analysis results from a specific analysis run.

    Parameters
    ----------
    session : Session
        Database session
    analysis_result_id : int
        ID of the AnalysisResult to query

    Returns
    -------
    list[DataAnalysis]
        All analysis results from this analysis run

    Example
    -------
    >>> # Get analysis results from run #5
    >>> results = get_data_analysis_for_analysis_result(session, 5)
    >>> for result in results:
    ...     print(f"ROI {result.roi_id}: {result.dec_dff_frequency} Hz")
    """
    return session.exec(
        select(DataAnalysis).where(
            DataAnalysis.analysis_result_id == analysis_result_id
        )
    ).all()


# ============================================================================
# Query Pattern 2: Get all analysis versions for an ROI
# ============================================================================


def get_all_traces_for_roi(session: Session, roi_id: int) -> list[Traces]:
    """Get all trace versions for an ROI across different analysis runs.

    Parameters
    ----------
    session : Session
        Database session
    roi_id : int
        ID of the ROI

    Returns
    -------
    list[Traces]
        All trace versions, ordered by creation time

    Example
    -------
    >>> # Get all analysis versions for ROI #123
    >>> traces = get_all_traces_for_roi(session, roi_id=123)
    >>> for trace in traces:
    ...     print(f"Analysis {trace.analysis_result_id} at {trace.created_at}")
    """
    return session.exec(
        select(Traces).where(Traces.roi_id == roi_id).order_by(Traces.created_at)
    ).all()


def get_latest_trace_for_roi(session: Session, roi_id: int) -> Traces | None:
    """Get the most recent trace for an ROI.

    Parameters
    ----------
    session : Session
        Database session
    roi_id : int
        ID of the ROI

    Returns
    -------
    Traces | None
        Most recent trace, or None if no traces exist

    Example
    -------
    >>> # Get latest trace for ROI #123
    >>> trace = get_latest_trace_for_roi(session, roi_id=123)
    >>> if trace:
    ...     print(f"Latest analysis: {trace.analysis_result_id}")
    """
    from sqlalchemy import desc

    return session.exec(
        select(Traces).where(Traces.roi_id == roi_id).order_by(desc(Traces.created_at))
    ).first()


# ============================================================================
# Query Pattern 3: Get results by experiment + settings combination
# ============================================================================


def get_traces_by_settings(
    session: Session,
    experiment_id: int,
    detection_settings_id: int | None = None,
    analysis_settings_id: int | None = None,
) -> list[Traces]:
    """Get traces for a specific experiment + settings combination.

    Parameters
    ----------
    session : Session
        Database session
    experiment_id : int
        ID of the experiment
    detection_settings_id : int | None
        Optional detection settings ID to filter by
    analysis_settings_id : int | None
        Optional analysis settings ID to filter by

    Returns
    -------
    list[Traces]
        All matching traces

    Example
    -------
    >>> # Get traces for experiment #1 with specific analysis settings
    >>> traces = get_traces_by_settings(
    ...     session, experiment_id=1, analysis_settings_id=3
    ... )
    """
    # First find matching AnalysisResults
    query = select(AnalysisResult).where(AnalysisResult.experiment == experiment_id)

    if detection_settings_id is not None:
        query = query.where(AnalysisResult.detection_settings == detection_settings_id)

    if analysis_settings_id is not None:
        query = query.where(AnalysisResult.analysis_settings == analysis_settings_id)

    analysis_results = session.exec(query).all()

    # Get traces for those AnalysisResults
    if not analysis_results:
        return []

    analysis_result_ids = [ar.id for ar in analysis_results]
    return session.exec(
        select(Traces).where(Traces.analysis_result_id.in_(analysis_result_ids))  # type: ignore
    ).all()


# ============================================================================
# Query Pattern 4: Load experiment with specific analysis results
# ============================================================================


def load_experiment_with_analysis_result(
    session: Session, experiment_id: int, analysis_result_id: int
) -> dict[int, tuple[Traces | None, DataAnalysis | None]]:
    """Load experiment ROIs with results from a specific analysis run.

    Parameters
    ----------
    session : Session
        Database session
    experiment_id : int
        ID of the experiment
    analysis_result_id : int
        ID of the specific analysis run

    Returns
    -------
    dict[int, tuple[Traces | None, DataAnalysis | None]]
        Dictionary mapping roi_id to (trace, data_analysis) tuples

    Example
    -------
    >>> # Load experiment #1 with analysis run #5
    >>> results = load_experiment_with_analysis_result(session, 1, 5)
    >>> for roi_id, (trace, analysis) in results.items():
    ...     if trace and analysis:
    ...         print(f"ROI {roi_id}: {analysis.dec_dff_frequency} Hz")
    """
    # Get all traces from this analysis run
    traces = session.exec(
        select(Traces).where(Traces.analysis_result_id == analysis_result_id)
    ).all()

    # Get all data_analysis from this analysis run
    data_analyses = session.exec(
        select(DataAnalysis).where(
            DataAnalysis.analysis_result_id == analysis_result_id
        )
    ).all()

    # Build lookup dictionaries
    traces_by_roi = {t.roi_id: t for t in traces}
    analyses_by_roi = {da.roi_id: da for da in data_analyses}

    # Get all ROI IDs
    all_roi_ids = set(traces_by_roi.keys()) | set(analyses_by_roi.keys())

    # Combine results
    return {
        roi_id: (traces_by_roi.get(roi_id), analyses_by_roi.get(roi_id))
        for roi_id in all_roi_ids
    }


# ============================================================================
# Query Pattern 5: Compare analysis results across different settings
# ============================================================================


def compare_analysis_results(
    session: Session,
    roi_id: int,
    analysis_result_ids: list[int],
) -> dict[int, tuple[Traces | None, DataAnalysis | None]]:
    """Compare results for the same ROI across different analysis runs.

    Parameters
    ----------
    session : Session
        Database session
    roi_id : int
        ID of the ROI to compare
    analysis_result_ids : list[int]
        List of AnalysisResult IDs to compare

    Returns
    -------
    dict[int, tuple[Traces | None, DataAnalysis | None]]
        Dictionary mapping analysis_result_id to (trace, data_analysis) tuples

    Example
    -------
    >>> # Compare ROI #123 across 3 different analysis runs
    >>> results = compare_analysis_results(
    ...     session, roi_id=123, analysis_result_ids=[1, 2, 3]
    ... )
    >>> for ar_id, (trace, analysis) in results.items():
    ...     if analysis:
    ...         print(f"Analysis {ar_id}: {analysis.dec_dff_frequency} Hz")
    """
    # Get traces for this ROI from specified analysis runs
    traces = session.exec(
        select(Traces)
        .where(Traces.roi_id == roi_id)
        .where(Traces.analysis_result_id.in_(analysis_result_ids))  # type: ignore
    ).all()

    # Get data_analysis for this ROI from specified analysis runs
    data_analyses = session.exec(
        select(DataAnalysis)
        .where(DataAnalysis.roi_id == roi_id)
        .where(DataAnalysis.analysis_result_id.in_(analysis_result_ids))  # type: ignore
    ).all()

    # Build lookup dictionaries
    traces_by_ar = {t.analysis_result_id: t for t in traces}
    analyses_by_ar = {da.analysis_result_id: da for da in data_analyses}

    # Combine results
    return {
        ar_id: (traces_by_ar.get(ar_id), analyses_by_ar.get(ar_id))
        for ar_id in analysis_result_ids
    }


# ============================================================================
# Query Pattern 6: Using ROI.traces_history and .data_analysis_history
# ============================================================================


def get_roi_with_all_history(session: Session, roi_id: int) -> ROI | None:
    """Load an ROI with all its analysis history.

    Parameters
    ----------
    session : Session
        Database session
    roi_id : int
        ID of the ROI

    Returns
    -------
    ROI | None
        ROI with traces_history and data_analysis_history relationships loaded

    Example
    -------
    >>> # Load ROI with all analysis versions
    >>> roi = get_roi_with_all_history(session, roi_id=123)
    >>> if roi:
    ...     print(f"ROI has {len(roi.traces_history)} trace versions")
    ...     print(f"ROI has {len(roi.data_analysis_history)} analysis versions")
    ...
    ...     # Get latest version
    ...     latest_trace = max(roi.traces_history, key=lambda t: t.created_at)
    ...     print(f"Latest analysis: {latest_trace.analysis_result_id}")
    """
    from sqlalchemy.orm import selectinload

    return session.exec(
        select(ROI)
        .where(ROI.id == roi_id)
        .options(
            selectinload(ROI.traces_history),
            selectinload(ROI.data_analysis_history),
        )
    ).first()


# ============================================================================
# Query Pattern 7: Compare ROIs by label value across analyses
# ============================================================================


def get_roi_by_label_and_fov(
    session: Session,
    fov_id: int,
    label_value: int,
    detection_settings_id: int | None = None,
) -> ROI | None:
    """Get a specific ROI by its label value within a FOV.

    Parameters
    ----------
    session : Session
        Database session
    fov_id : int
        ID of the FOV
    label_value : int
        The label value of the ROI (e.g., 1, 2, 3...)
    detection_settings_id : int | None
        Optional: specify which detection run if multiple exist

    Returns
    -------
    ROI | None
        The ROI with that label, or None if not found

    Example
    -------
    >>> # Get ROI with label 5 from FOV #1
    >>> roi = get_roi_by_label_and_fov(session, fov_id=1, label_value=5)
    >>> if roi:
    ...     print(f"Found ROI {roi.id} with label {roi.label_value}")
    """
    query = (
        select(ROI)
        .where(ROI.fov_id == fov_id)
        .where(ROI.label_value == label_value)
    )

    if detection_settings_id is not None:
        query = query.where(ROI.detection_settings_id == detection_settings_id)

    return session.exec(query).first()


def compare_roi_across_analyses(
    session: Session,
    fov_id: int,
    label_value: int,
    analysis_result_ids: list[int],
    detection_settings_id: int | None = None,
) -> dict[int, tuple[Traces | None, DataAnalysis | None]]:
    """Compare analysis results for the same ROI label across different analysis runs.

    This is useful when you want to see how different analysis settings affect
    the same cell (identified by its label value from segmentation).

    Parameters
    ----------
    session : Session
        Database session
    fov_id : int
        ID of the FOV containing the ROI
    label_value : int
        The label value of the ROI (e.g., 1, 2, 3...)
    analysis_result_ids : list[int]
        List of AnalysisResult IDs to compare
    detection_settings_id : int | None
        Optional: specify which detection run if multiple exist

    Returns
    -------
    dict[int, tuple[Traces | None, DataAnalysis | None]]
        Dictionary mapping analysis_result_id to (trace, data_analysis) tuples

    Example
    -------
    >>> # Compare how ROI label #5 was analyzed across 3 different analysis runs
    >>> results = compare_roi_across_analyses(
    ...     session,
    ...     fov_id=1,
    ...     label_value=5,
    ...     analysis_result_ids=[1, 2, 3]
    ... )
    >>> for ar_id, (trace, analysis) in results.items():
    ...     if analysis:
    ...         print(f"Analysis {ar_id}: {analysis.dec_dff_frequency} Hz")
    """
    # First, find the ROI with this label
    roi = get_roi_by_label_and_fov(
        session, fov_id, label_value, detection_settings_id
    )

    if roi is None or roi.id is None:
        return {}

    # Now use the existing comparison function with the ROI ID
    return compare_analysis_results(session, roi.id, analysis_result_ids)


def compare_all_rois_in_fov_across_analyses(
    session: Session,
    fov_id: int,
    analysis_result_ids: list[int],
    detection_settings_id: int | None = None,
) -> dict[int, dict[int, tuple[Traces | None, DataAnalysis | None]]]:
    """Compare all ROIs in a FOV across different analysis runs.

    This gives you a complete comparison showing how each cell (by label)
    was analyzed under different settings.

    Parameters
    ----------
    session : Session
        Database session
    fov_id : int
        ID of the FOV
    analysis_result_ids : list[int]
        List of AnalysisResult IDs to compare
    detection_settings_id : int | None
        Optional: specify which detection run if multiple exist

    Returns
    -------
    dict[int, dict[int, tuple[Traces | None, DataAnalysis | None]]]
        Nested dictionary: {label_value: {analysis_result_id: (trace, analysis)}}

    Example
    -------
    >>> # Compare all ROIs across 3 analysis runs
    >>> results = compare_all_rois_in_fov_across_analyses(
    ...     session, fov_id=1, analysis_result_ids=[1, 2, 3]
    ... )
    >>> # Check results for ROI label 5
    >>> for ar_id, (trace, analysis) in results[5].items():
    ...     if analysis:
    ...         print(f"Label 5, Analysis {ar_id}: {analysis.dec_dff_frequency} Hz")
    """
    # Get all ROIs for this FOV
    query = select(ROI).where(ROI.fov_id == fov_id)
    if detection_settings_id is not None:
        query = query.where(ROI.detection_settings_id == detection_settings_id)

    rois = session.exec(query).all()

    # For each ROI, get all analysis versions
    results: dict[int, dict[int, tuple[Traces | None, DataAnalysis | None]]] = {}

    for roi in rois:
        if roi.id is None:
            continue

        roi_results = compare_analysis_results(session, roi.id, analysis_result_ids)
        results[roi.label_value] = roi_results

    return results


# ============================================================================
# Query Pattern 8: Detection method comparison
# ============================================================================


def get_rois_by_detection_settings(
    session: Session, fov_id: int, detection_settings_id: int
) -> list[ROI]:
    """Get all ROIs from a specific detection run for a FOV.

    Parameters
    ----------
    session : Session
        Database session
    fov_id : int
        ID of the FOV
    detection_settings_id : int
        ID of the detection settings

    Returns
    -------
    list[ROI]
        All ROIs detected with those settings

    Example
    -------
    >>> # Get Cellpose ROIs for FOV #1
    >>> cellpose_rois = get_rois_by_detection_settings(
    ...     session, fov_id=1, detection_settings_id=1
    ... )
    >>> # Get CaImAn ROIs for the same FOV
    >>> caiman_rois = get_rois_by_detection_settings(
    ...     session, fov_id=1, detection_settings_id=2
    ... )
    >>> print(f"Cellpose found {len(cellpose_rois)} ROIs")
    >>> print(f"CaImAn found {len(caiman_rois)} ROIs")
    """
    return session.exec(
        select(ROI)
        .where(ROI.fov_id == fov_id)
        .where(ROI.detection_settings_id == detection_settings_id)
    ).all()


def compare_detection_methods(
    session: Session,
    experiment_id: int,
    detection_settings_ids: list[int],
) -> dict[int, dict[str, int]]:
    """Compare ROI counts across different detection methods.

    Parameters
    ----------
    session : Session
        Database session
    experiment_id : int
        ID of the experiment
    detection_settings_ids : list[int]
        List of detection settings IDs to compare

    Returns
    -------
    dict[int, dict[str, int]]
        Dictionary mapping detection_settings_id to
        {"total_rois": count, "active_rois": count}

    Example
    -------
    >>> # Compare Cellpose vs CaImAn detection
    >>> comparison = compare_detection_methods(
    ...     session, experiment_id=1, detection_settings_ids=[1, 2]
    ... )
    >>> for det_id, stats in comparison.items():
    ...     print(
    ...         f"Detection {det_id}: {stats['total_rois']} total, "
    ...         f"{stats['active_rois']} active"
    ...     )
    """
    from sqlalchemy import func

    results = {}
    for det_id in detection_settings_ids:
        # Count total ROIs
        total = session.exec(
            select(func.count(ROI.id)).where(ROI.detection_settings_id == det_id)
        ).one()

        # Count active ROIs
        active = session.exec(
            select(func.count(ROI.id))
            .where(ROI.detection_settings_id == det_id)
            .where(ROI.active == True)  # noqa: E712
        ).one()

        results[det_id] = {
            "total_rois": total,
            "active_rois": active,
        }

    return results


def get_analysis_for_detection_method(
    session: Session,
    experiment_id: int,
    detection_settings_id: int,
    analysis_settings_id: int | None = None,
) -> list[tuple[ROI, Traces | None, DataAnalysis | None]]:
    """Get analysis results for ROIs from a specific detection method.

    Parameters
    ----------
    session : Session
        Database session
    experiment_id : int
        ID of the experiment
    detection_settings_id : int
        ID of the detection settings
    analysis_settings_id : int | None
        Optional specific analysis settings to filter by

    Returns
    -------
    list[tuple[ROI, Traces | None, DataAnalysis | None]]
        List of (ROI, latest_trace, latest_analysis) tuples

    Example
    -------
    >>> # Get Cellpose results
    >>> cellpose_results = get_analysis_for_detection_method(
    ...     session, experiment_id=1, detection_settings_id=1
    ... )
    >>> # Get CaImAn results
    >>> caiman_results = get_analysis_for_detection_method(
    ...     session, experiment_id=1, detection_settings_id=2
    ... )
    >>> print(f"Cellpose: {len(cellpose_results)} analyzed ROIs")
    >>> print(f"CaImAn: {len(caiman_results)} analyzed ROIs")
    """
    from sqlalchemy import desc

    # Find analysis results matching criteria
    query = (
        select(AnalysisResult)
        .where(AnalysisResult.experiment == experiment_id)
        .where(AnalysisResult.detection_settings == detection_settings_id)
    )

    if analysis_settings_id is not None:
        query = query.where(AnalysisResult.analysis_settings == analysis_settings_id)

    # Get most recent analysis result
    analysis_result = session.exec(query.order_by(desc(AnalysisResult.id))).first()

    if not analysis_result or analysis_result.id is None:
        return []

    # Get all ROIs for this detection method
    rois = session.exec(
        select(ROI).where(ROI.detection_settings_id == detection_settings_id)
    ).all()

    # Get traces and analyses for this analysis result
    traces_by_roi = {
        t.roi_id: t
        for t in session.exec(
            select(Traces).where(Traces.analysis_result_id == analysis_result.id)
        ).all()
    }

    analyses_by_roi = {
        da.roi_id: da
        for da in session.exec(
            select(DataAnalysis).where(
                DataAnalysis.analysis_result_id == analysis_result.id
            )
        ).all()
    }

    # Combine results
    return [
        (roi, traces_by_roi.get(roi.id), analyses_by_roi.get(roi.id)) for roi in rois
    ]


# ============================================================================
# Helper: Get latest results for all ROIs (backward compatibility)
# ============================================================================


def get_latest_results_for_experiment(
    session: Session,
    experiment_id: int,
    detection_settings_id: int | None = None,
) -> dict[int, tuple[Traces | None, DataAnalysis | None]]:
    """Get the most recent analysis results for all ROIs in an experiment.

    This provides backward-compatible behavior similar to the old schema
    where each ROI had a single trace/data_analysis.

    Parameters
    ----------
    session : Session
        Database session
    experiment_id : int
        ID of the experiment
    detection_settings_id : int | None
        Optional: filter by specific detection settings

    Returns
    -------
    dict[int, tuple[Traces | None, DataAnalysis | None]]
        Dictionary mapping roi_id to latest (trace, data_analysis) tuples

    Example
    -------
    >>> # Get latest results for all ROIs in experiment #1
    >>> results = get_latest_results_for_experiment(session, experiment_id=1)
    >>> for roi_id, (trace, analysis) in results.items():
    ...     if analysis:
    ...         print(f"ROI {roi_id}: {analysis.dec_dff_frequency} Hz")
    >>>
    >>> # Get latest results for Cellpose ROIs only
    >>> results = get_latest_results_for_experiment(
    ...     session, experiment_id=1, detection_settings_id=1
    ... )
    """
    from sqlalchemy import desc

    # Find latest AnalysisResult for this experiment
    query = select(AnalysisResult).where(AnalysisResult.experiment == experiment_id)

    if detection_settings_id is not None:
        query = query.where(AnalysisResult.detection_settings == detection_settings_id)

    latest_analysis_result = session.exec(
        query.order_by(desc(AnalysisResult.id))
    ).first()

    if not latest_analysis_result or latest_analysis_result.id is None:
        return {}

    return load_experiment_with_analysis_result(
        session, experiment_id, latest_analysis_result.id
    )
