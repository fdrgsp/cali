from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d

from cali.logger import cali_logger
from cali.sqlmodel._model import ROI, CaliResult, DataAnalysis, Traces

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Helpers: retrieval from ROI histories
# -----------------------------------------------------------------------------#
def _get_traces_for_run(roi_model: ROI, run_id: int | None) -> Traces | None:
    """Get the Traces object for a specific run from the ROI's traces_history."""
    if not roi_model.traces_history:
        return None
    if run_id is None:
        return roi_model.traces_history[0]
    for trace in roi_model.traces_history:
        if trace.analysis_result_id == run_id:
            return trace
    return None


def _get_data_analysis_for_run(
    roi_model: ROI, run_id: int | None
) -> DataAnalysis | None:
    """Get DataAnalysis for a specific run from ROI's data_analysis_history."""
    if not roi_model.data_analysis_history:
        return None
    if run_id is None:
        return roi_model.data_analysis_history[0]
    # First try to find exact match
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    # Fall back to first entry (for backwards compatibility with data that has
    # analysis_result_id=None)
    return roi_model.data_analysis_history[0]


# -----------------------------------------------------------------------------#
# Main plotting entry point (pyqtgraph version)
# -----------------------------------------------------------------------------#
def _plot_inferred_spike_burst_activity(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot burst detection and network state analysis for inferred spikes (pyqtgraph).

    This analyzes population-level spike activity to detect synchronized burst events
    and displays burst statistics.
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()

    # Hide shared legend if you use one elsewhere
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Make sure viewbox is reset (important when switching from other plot types)
    vb = plot.getViewBox()
    vb.setAspectLocked(False)
    vb.enableAutoRange(x=True, y=True)

    # --- 1) Get burst parameters from AnalysisSettings ---
    burst_params = _get_burst_parameters(engine, fov_name, rois, run_id)
    if burst_params is None:
        cali_logger.warning("Burst parameters not found in ROI data.")
        plot.setTitle("Population Burst Activity\n(No burst parameters found)")
        plot.setLabel("bottom", "Time (s)")
        plot.setLabel("left", "Population Activity")
        return

    burst_threshold, min_burst_duration, smoothing_sigma = burst_params

    # --- 2) Get population spike data ---
    spike_trains, _roi_names, time_axis = _get_population_spike_data(
        engine, fov_name, rois, run_id
    )

    if spike_trains is None or spike_trains.shape[0] < 2:
        cali_logger.warning(
            "Not enough active ROIs with spikes to plot population activity."
        )
        plot.setTitle("Population Burst Activity\n(Not enough spike data)")
        plot.setLabel("bottom", "Time (s)")
        plot.setLabel("left", "Population Activity")
        return

    # --- 3) Population activity (mean over ROIs) ---
    population_activity = np.mean(spike_trains, axis=0)

    # --- 4) Smooth population activity for burst detection ---
    if smoothing_sigma > 0:
        smoothed_activity = gaussian_filter1d(
            population_activity, sigma=smoothing_sigma, mode="nearest"
        )
    else:
        smoothed_activity = population_activity

    # --- 5) Detect bursts (burst_threshold is in %) ---
    the_value = burst_threshold / 100.0
    bursts = _detect_population_bursts(smoothed_activity, the_value, min_burst_duration)

    # --- 6) Draw traces + threshold + burst regions ---
    _draw_population_activity_pg(
        plot,
        time_axis=time_axis,
        raw_activity=population_activity,
        smoothed_activity=smoothed_activity,
        bursts=bursts,
        threshold_value=the_value,
    )

    # --- 7) Stats text in title ---
    stats_text = _burst_statistics_text(bursts, time_axis)
    title = (
        "Population Activity and Burst Detection (Thresholded Spike Data)\n"
        f"{stats_text}"
    )
    plot.setTitle(title)

    plot.setLabel("bottom", "Time (s)")
    plot.setLabel("left", "Population Activity")

    # Auto-range once everything is added
    vb.enableAutoRange(x=True, y=True)


# -----------------------------------------------------------------------------#
# Burst parameter retrieval
# -----------------------------------------------------------------------------#
def _get_burst_parameters(
    engine: Engine,
    fov_name: str,  # kept for API symmetry; currently unused
    rois: list[int] | None = None,  # kept for API symmetry; currently unused
    run_id: int | None = None,
) -> tuple[float, int, float] | None:
    """Get burst detection parameters from AnalysisSettings.

    Returns (burst_threshold, burst_min_duration, burst_gaussian_sigma) if found.
    """
    from sqlmodel import Session, select

    from cali.sqlmodel._model import AnalysisSettings

    with Session(engine) as session:
        # Prefer settings from the given run_id
        if run_id is not None:
            result = session.get(CaliResult, run_id)
            if result and result.analysis_settings_id is not None:
                settings = session.get(AnalysisSettings, result.analysis_settings_id)
                if settings:
                    return (
                        settings.burst_threshold,
                        settings.burst_min_duration,
                        settings.burst_gaussian_sigma,
                    )

        # Fallback: get settings from the first available run that has them
        stmt = (
            select(CaliResult)
            .where(CaliResult.analysis_settings_id.is_not(None))  # type: ignore
            .limit(1)
        )
        result = session.exec(stmt).first()
        if result and result.analysis_settings_id is not None:
            settings = session.get(AnalysisSettings, result.analysis_settings_id)
            if settings:
                return (
                    settings.burst_threshold,
                    settings.burst_min_duration,
                    settings.burst_gaussian_sigma,
                )

    cali_logger.warning("No valid analysis settings found for burst parameters.")
    return None


# -----------------------------------------------------------------------------#
# Population spike data extraction
# -----------------------------------------------------------------------------#
def _get_population_spike_data(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[str], np.ndarray]:
    """Extract population spike data from database.

    Returns
    -------
    (spike_trains_array, roi_names, time_axis)
    """
    from sqlalchemy.orm import selectinload
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV

    with Session(engine) as session:
        # Get detection_settings_id from the run if run_id is provided
        detection_settings_id: int | None = None
        if run_id is not None:
            result = session.get(CaliResult, run_id)
            if result:
                detection_settings_id = result.detection_settings_id

        stmt = select(ROI).join(FOV).where(col(FOV.name) == fov_name)

        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Filter by detection settings if we have a run_id
        if detection_settings_id is not None:
            stmt = stmt.where(col(ROI.detection_settings_id) == detection_settings_id)

        stmt = stmt.where(col(ROI.active) == True).options(  # noqa: E712
            selectinload(ROI.traces_history),
            selectinload(ROI.data_analysis_history),
        )

        roi_results = session.exec(stmt).all()

    spike_trains: list[np.ndarray] = []
    roi_names: list[str] = []
    rois_rec_time: list[float] = []

    for roi in roi_results:
        traces = _get_traces_for_run(roi, run_id)
        if traces is None or not traces.inferred_spikes:
            continue

        spikes = np.asarray(traces.inferred_spikes, dtype=float)

        data_analysis = _get_data_analysis_for_run(roi, run_id)
        threshold = data_analysis.inferred_spikes_threshold if data_analysis else 0.0
        if threshold is None:
            threshold = 0.0

        # Threshold + binarize
        spikes[spikes <= threshold] = 0.0
        spike_train = (spikes > 0.0).astype(float)

        if spike_train.sum() == 0:
            continue

        spike_trains.append(spike_train)
        roi_names.append(str(roi.label_value))

        if data_analysis and data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

    if len(spike_trains) < 2:
        return None, [], np.array([])

    # Pad / truncate to common length
    lengths = np.array([len(t) for t in spike_trains], dtype=int)
    max_length = int(lengths.max())

    spike_trains_array = np.zeros((len(spike_trains), max_length), dtype=float)
    for i, train in enumerate(spike_trains):
        L = len(train)
        if L >= max_length:
            spike_trains_array[i, :] = train[:max_length]
        else:
            spike_trains_array[i, :L] = train

    # Time axis from recording time if available, else frames @ 10Hz
    if rois_rec_time:
        avg_rec_time = float(np.mean(rois_rec_time))
        time_axis = np.linspace(0.0, avg_rec_time, max_length)
    else:
        time_axis = np.arange(max_length) / 10.0

    return spike_trains_array, roi_names, time_axis


# -----------------------------------------------------------------------------#
# Burst detection (vectorized)
# -----------------------------------------------------------------------------#
def _detect_population_bursts(
    population_activity: np.ndarray,
    burst_threshold: float,
    min_duration: int,
) -> list[tuple[int, int]]:
    """Detect population bursts in the smoothed activity.

    Returns list of (start, end) indices; end is exclusive.
    """
    if population_activity.size == 0:
        return []

    above_threshold = population_activity > burst_threshold
    if not np.any(above_threshold):
        return []

    above_int = above_threshold.astype(int)
    changes = np.diff(above_int)

    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(above_threshold))

    durations = ends - starts
    valid = durations >= min_duration

    bursts: list[tuple[int, int]] = [
        (int(s), int(e)) for s, e, v in zip(starts, ends, valid) if v
    ]
    return bursts


# -----------------------------------------------------------------------------#
# pyqtgraph drawing helpers
# -----------------------------------------------------------------------------#
def _draw_population_activity_pg(
    plot: pg.PlotItem,
    time_axis: np.ndarray,
    raw_activity: np.ndarray,
    smoothed_activity: np.ndarray,
    bursts: list[tuple[int, int]],
    threshold_value: float,
) -> None:
    """Draw population activity + threshold + burst regions in pyqtgraph."""
    # Raw activity (light gray)
    plot.plot(
        time_axis,
        raw_activity,
        pen=pg.mkPen((200, 200, 200), width=1),
        name="Raw Population Activity",
    )

    # Smoothed activity (blue)
    plot.plot(
        time_axis,
        smoothed_activity,
        pen=pg.mkPen("c", width=2),
        name="Smoothed Population Activity",
    )

    # Threshold line
    the_line = pg.InfiniteLine(
        pos=threshold_value,
        angle=0,
        pen=pg.mkPen("y", width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
    )
    the_line.setZValue(5)
    plot.addItem(the_line)

    # Burst regions (green translucent)
    for start_idx, end_idx in bursts:
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(time_axis))
        if end_idx <= start_idx:
            continue
        t0 = float(time_axis[start_idx])
        t1 = float(time_axis[end_idx - 1])

        region = pg.LinearRegionItem(
            values=[t0, t1],
            brush=pg.mkBrush(0, 255, 0, 60),
            movable=False,
        )
        region.setZValue(1)
        plot.addItem(region)


# -----------------------------------------------------------------------------#
# Burst statistics → title text
# -----------------------------------------------------------------------------#
def _burst_statistics_text(
    bursts: list[tuple[int, int]],
    time_axis: np.ndarray,
) -> str:
    """Return a compact string with burst statistics."""
    if not bursts or time_axis.size == 0:
        return "Burst Statistics: No bursts detected"

    burst_durations: list[float] = []
    burst_intervals: list[float] = []

    for i, (start, end) in enumerate(bursts):
        start = max(start, 0)
        end = min(end, len(time_axis))
        if end <= start:
            continue

        duration = float(time_axis[end - 1] - time_axis[start])
        burst_durations.append(duration)

        if i > 0:
            prev_end = bursts[i - 1][1]
            prev_end = min(prev_end, len(time_axis))
            interval = float(time_axis[start] - time_axis[prev_end - 1])
            burst_intervals.append(interval)

    count = len(burst_durations)
    avg_duration = float(np.mean(burst_durations)) if burst_durations else 0.0
    avg_interval = float(np.mean(burst_intervals)) if burst_intervals else 0.0

    total_time = float(time_axis[-1] - time_axis[0])
    burst_rate = (count / total_time) * 60.0 if total_time > 0 else 0.0

    return (
        f"Count: {count}, "
        f"Avg Duration: {avg_duration:.2f}s, "
        f"Avg Interval: {avg_interval:.2f}s, "
        f"Rate: {burst_rate:.2f} bursts/min"
    )
