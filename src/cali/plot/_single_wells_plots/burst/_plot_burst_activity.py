from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from cali.logger import cali_logger
from cali.sqlmodel._model import ROI, CaliResult, DataAnalysis, FOVAnalysis, Traces

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Helpers: retrieval from ROI histories and FOV analysis
# -----------------------------------------------------------------------------#
def _get_fov_analysis_for_run(
    engine: Engine,
    fov_name: str,
    run_id: int | None = None,
) -> FOVAnalysis | None:
    """Get FOVAnalysis with pre-computed burst data for the given FOV and run.

    Returns None if no FOVAnalysis exists for this run.
    """
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV

    with Session(engine) as session:
        # Get FOV
        stmt = select(FOV).where(col(FOV.name) == fov_name)
        fov = session.exec(stmt).first()
        if not fov:
            return None

        # Get FOVAnalysis for this run
        if not fov.fov_analysis_history:
            return None

        # If run_id specified, find matching FOVAnalysis
        if run_id is not None:
            for fov_analysis in fov.fov_analysis_history:
                if fov_analysis.analysis_result_id == run_id:
                    return fov_analysis

        # Otherwise return the most recent one
        result = fov.fov_analysis_history[-1] if fov.fov_analysis_history else None
        return result


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

    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)
    vb.enableAutoRange(x=True, y=True)

    # Disconnect any hover handlers from previous plots
    scene = plot.scene()
    handler_names = [
        "sync_hover_handler",
        "ccorr_hover_handler",
        "spike_sync_hover_handler",
        "spike_ccorr_hover_handler",
        "spike_maxlag_hover_handler",
        "spike_maxlag_values_hover_handler",
        "dff_corr_hover_handler",
        "evoked_hover_handler",
    ]
    for handler_name in handler_names:
        old_handler = plot.property(handler_name)
        if old_handler is not None:
            try:
                scene.sigMouseMoved.disconnect(old_handler)
            except (TypeError, RuntimeError):
                pass
            plot.setProperty(handler_name, None)

    # Hide shared legend if you use one elsewhere
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    # Initialize variables that will be used in drawing
    population_activity: np.ndarray | None = None
    time_axis: np.ndarray = np.array([])
    bursts: list[tuple[int, int]] = []

    # --- Try to get pre-computed burst data from FOVAnalysis ---
    fov_analysis = _get_fov_analysis_for_run(engine, fov_name, run_id)

    # Check if pre-computed population activity data exists (regardless of burst count)
    if fov_analysis is not None and fov_analysis.spike_population_activity is not None:
        # Use stored burst data - much faster!
        burst_starts = fov_analysis.spike_burst_starts or []
        burst_ends = fov_analysis.spike_burst_ends or []
        population_activity_list = fov_analysis.spike_population_activity

        if population_activity_list:
            # spike_population_activity contains smoothed normalized activity
            population_activity = np.array(population_activity_list)

            # Get raw activity if available
            population_activity_raw_list = fov_analysis.spike_population_activity_raw
            population_activity_raw = (
                np.array(population_activity_raw_list)
                if population_activity_raw_list
                else None
            )

            # Get time axis from ALL ROIs
            # (population activity was computed from all active ROIs)
            _, _, time_axis = _get_population_spike_data(
                engine, fov_name, rois=None, run_id=run_id
            )

            if time_axis.size == 0 or len(time_axis) != len(population_activity):
                cali_logger.warning("Time axis length mismatch with stored data")
                # Fall back to frame indices
                time_axis = np.arange(len(population_activity)) / 10.0

            # Convert stored frame indices to (start, end) tuples
            bursts = list(zip(burst_starts, burst_ends))

            # Get threshold from AnalysisSettings for display
            burst_params = _get_burst_parameters(engine, fov_name, rois, run_id)
            the_value = (burst_params[0] / 100.0) if burst_params else 0.5

            # --- Draw raw + smoothed activity + threshold + burst regions ---
            # Plot raw population activity (normalized, black)
            if population_activity_raw is not None:
                plot.plot(
                    time_axis,
                    population_activity_raw,
                    pen=pg.mkPen("black", width=2),
                    name="Raw Activity (Normalized)",
                )

            # Plot smoothed population activity (normalized, magenta)
            plot.plot(
                time_axis,
                population_activity,
                pen=pg.mkPen("magenta", width=3),
                name="Smoothed Activity (Normalized)",
            )

            # Threshold line
            threshold_line = pg.InfiniteLine(
                pos=the_value,
                angle=0,
                pen=pg.mkPen("magenta", width=3, style=pg.QtCore.Qt.PenStyle.DashLine),
            )
            threshold_line.setZValue(5)
            plot.addItem(threshold_line)

            # Burst regions (green translucent)
            # Burst indices are frame indices; map them to time_axis values
            for start_idx, end_idx in bursts:
                # Clamp indices to valid range
                start_idx = max(0, min(start_idx, len(time_axis) - 1))
                end_idx = max(0, min(end_idx, len(time_axis)))
                if end_idx <= start_idx:
                    continue
                # Map frame indices to time values
                t0 = float(time_axis[start_idx])
                t1 = float(time_axis[min(end_idx - 1, len(time_axis) - 1)])

                region = pg.LinearRegionItem(
                    values=[t0, t1],
                    brush=pg.mkBrush(0, 255, 0, 90),
                    pen=pg.mkPen(None),
                    movable=False,
                )
                region.setZValue(1)
                plot.addItem(region)

            # Add legend
            if hasattr(widget, "legend") and widget.legend is not None:
                widget.legend.clear()
                raw_item = pg.PlotDataItem(pen=pg.mkPen("black", width=2))
                widget.legend.addItem(raw_item, "Raw Activity (Normalized)")
                smoothed_item = pg.PlotDataItem(pen=pg.mkPen("magenta", width=3))
                widget.legend.addItem(smoothed_item, "Smoothed Activity (Normalized)")
                threshold_item = pg.PlotDataItem(
                    pen=pg.mkPen(
                        "magenta", width=3, style=pg.QtCore.Qt.PenStyle.DashLine
                    )
                )
                widget.legend.addItem(
                    threshold_item, f"Burst Threshold ({the_value:.2f})"
                )
                burst_item = pg.PlotDataItem(pen=pg.mkPen((0, 255, 0), width=3))
                widget.legend.addItem(burst_item, "Detected Bursts")
                widget.legend.setVisible(True)

            # --- Stats text in title ---
            stats_text = _burst_statistics_text(bursts, time_axis)
            title = f"Population Burst Activity (Inferred Spikes)\n{stats_text}"
            plot.setTitle(title)

            plot.setLabel("bottom", "Time (s)")
            plot.setLabel("left", "Population Activity (Normalized [0,1])")

            # Auto-range once everything is added
            vb.enableAutoRange(x=True, y=True)
        else:
            cali_logger.warning(
                "Pre-computed spike burst data is incomplete (missing population activity)"
            )
            plot.setTitle(
                "Population Burst Activity\n(No pre-computed data available - please re-run analysis)"
            )
            plot.setLabel("bottom", "Time (s)")
            plot.setLabel("left", "Population Activity")
    else:
        cali_logger.warning("No spike burst analysis data found in database")
        plot.setTitle(
            "Population Burst Activity\n(No data found - please run analysis first)"
        )
        plot.setLabel("bottom", "Time (s)")
        plot.setLabel("left", "Population Activity")


# -----------------------------------------------------------------------------#
# Burst parameter retrieval
# -----------------------------------------------------------------------------#
def _get_burst_parameters(
    engine: Engine,
    fov_name: str,  # kept for API symmetry; currently unused
    rois: list[int] | None = None,  # kept for API symmetry; currently unused
    run_id: int | None = None,
) -> tuple[float, float, float] | None:
    """Get burst detection parameters from AnalysisSettings.

    Returns (burst_threshold, burst_min_duration_ms, burst_gaussian_sigma) if found.
    All returned in original units from database (threshold in %, duration in ms, sigma
    in seconds).
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
                        settings.burst_min_duration,  # milliseconds
                        settings.burst_gaussian_sigma,  # seconds
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
                    settings.burst_min_duration,  # milliseconds
                    settings.burst_gaussian_sigma,  # seconds
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
    legend: pg.LegendItem | None = None,
) -> None:
    """Draw population activity + threshold + burst regions in pyqtgraph.

    Parameters
    ----------
    plot : pg.PlotItem
        The plot item to draw on.
    time_axis : np.ndarray
        Time axis values.
    raw_activity : np.ndarray
        Raw population activity.
    smoothed_activity : np.ndarray
        Smoothed population activity.
    bursts : list[tuple[int, int]]
        List of (start, end) indices for detected bursts.
    threshold_value : float
        Threshold value for burst detection.
    legend : pg.LegendItem | None
        Optional shared legend item from the widget. If provided, will be used
        instead of creating a new legend on the plot.
    """
    # Raw activity (light gray)
    plot.plot(
        time_axis,
        raw_activity,
        pen=pg.mkPen((0, 0, 0), width=2),
        name="Raw Activity",
    )

    # Smoothed activity (magenta)
    plot.plot(
        time_axis,
        smoothed_activity,
        pen=pg.mkPen("magenta", width=3),
        name="Smoothed Activity",
    )

    # Threshold line
    threshold_line = pg.InfiniteLine(
        pos=threshold_value,
        angle=0,
        pen=pg.mkPen("magenta", width=3, style=pg.QtCore.Qt.PenStyle.DashLine),
    )
    threshold_line.setZValue(5)
    plot.addItem(threshold_line)

    # Dummy item so dashed line appears in legend
    plot.plot(
        [time_axis[0]],
        [threshold_value],
        pen=pg.mkPen("magenta", width=3, style=pg.QtCore.Qt.PenStyle.DashLine),
        name="Burst Threshold",
    )

    # Burst regions (green translucent)
    # Burst indices are frame indices; map them to time_axis values
    for start_idx, end_idx in bursts:
        # Clamp indices to valid range
        start_idx = max(0, min(start_idx, len(time_axis) - 1))
        end_idx = max(0, min(end_idx, len(time_axis)))
        if end_idx <= start_idx:
            continue
        # Map frame indices to time values
        t0 = float(time_axis[start_idx])
        t1 = float(time_axis[min(end_idx - 1, len(time_axis) - 1)])

        region = pg.LinearRegionItem(
            values=[t0, t1],
            brush=pg.mkBrush(0, 255, 0, 90),
            pen=pg.mkPen(None),
            movable=False,
        )
        region.setZValue(1)
        plot.addItem(region)

    # --- Use shared legend if provided, otherwise create plot-local legend ---
    if legend is not None:
        # Using widget's shared legend - manually add items
        legend.clear()

        # Add legend items manually
        raw_item = pg.PlotDataItem(pen=pg.mkPen((0, 0, 0), width=3))
        legend.addItem(raw_item, "Raw Activity")

        smoothed_item = pg.PlotDataItem(pen=pg.mkPen("magenta", width=3))
        legend.addItem(smoothed_item, "Smoothed Activity")

        threshold_item = pg.PlotDataItem(
            pen=pg.mkPen("magenta", width=3, style=pg.QtCore.Qt.PenStyle.DashLine)
        )
        legend.addItem(threshold_item, "Burst Threshold")

        # Only add "Detected Bursts" legend item if there are bursts
        if bursts:
            burst_item = pg.PlotDataItem(pen=pg.mkPen((0, 255, 0), width=3))
            legend.addItem(burst_item, "Detected Bursts")

        legend.setVisible(True)


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


# -----------------------------------------------------------------------------#
# Normalized spikes + global bursts overlay
# -----------------------------------------------------------------------------#
def _plot_inferred_spikes_normalized_with_bursts(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot normalized inferred spikes with superimposed *global* burst periods.

    Network bursts are always computed from ALL active ROIs for the given run
    (global network activity). The ROI selection only affects which traces are
    drawn, not how bursts are defined.
    """
    plot = widget.plot_item
    assert plot is not None

    # Disconnect any hover handlers from previous plots
    scene = plot.scene()
    handler_names = [
        "sync_hover_handler",
        "ccorr_hover_handler",
        "spike_sync_hover_handler",
        "spike_ccorr_hover_handler",
        "spike_maxlag_hover_handler",
        "spike_maxlag_values_hover_handler",
        "dff_corr_hover_handler",
        "evoked_hover_handler",
    ]
    for handler_name in handler_names:
        old_handler = plot.property(handler_name)
        if old_handler is not None:
            try:
                scene.sigMouseMoved.disconnect(old_handler)
            except (TypeError, RuntimeError):
                pass
            plot.setProperty(handler_name, None)

    # ---- remove ROI legend for this plot ----
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.clear()
        cali_logger.warning(
            "No run_id provided for inferred spikes normalized with bursts plot."
        )
        plot.setTitle(
            "No analysis run selected.\nPlease select a run from the dropdown."
        )
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "Inferred Spikes (a.u.)")
        return

    # ------------- Get pre-computed burst data from FOVAnalysis -------------#
    fov_analysis = _get_fov_analysis_for_run(engine, fov_name, run_id)

    bursts: list[tuple[int, int]] = []
    if fov_analysis is not None and fov_analysis.spike_burst_starts is not None:
        # Use stored burst data
        burst_starts = fov_analysis.spike_burst_starts
        burst_ends = fov_analysis.spike_burst_ends or []
        bursts = list(zip(burst_starts, burst_ends))

    # -------------------- Plot normalized spikes (subset) -------------------#
    from cali.plot._single_wells_plots.spikes._plot_inferred_spikes import (
        _plot_inferred_spikes,
    )

    _plot_inferred_spikes(
        widget,
        engine,
        fov_name,
        rois,
        run_id=run_id,
        raw=False,
        normalize=True,
        active_only=False,
        dec_dff=False,
        thresholds=False,
    )

    # ------------------------ Overlay global bursts ------------------------ #
    if bursts and len(bursts) > 0:
        plot = widget.plot_item
        assert plot is not None

        # Burst indices are frame indices, and the underlying raster plot
        # uses frame indices (0, 1, 2, ...) for x-axis, so we use them directly
        for start_idx, end_idx in bursts:
            if end_idx <= start_idx:
                continue

            # Use frame indices directly (matching the underlying plot)
            region = pg.LinearRegionItem(
                values=(float(start_idx), float(end_idx - 1)),
                brush=pg.mkBrush(0, 255, 0, 90),
                pen=pg.mkPen(None),
                movable=False,
            )
            region.setZValue(-5)  # behind traces
            plot.addItem(region)

        # Add legend for detected bursts
        legend = getattr(widget, "legend", None)
        if legend is not None:
            legend.clear()
            # Use solid green line with width 3 for burst legend
            burst_item = pg.PlotDataItem(pen=pg.mkPen((0, 255, 0), width=3))
            legend.addItem(burst_item, "Detected Bursts")
            legend.setVisible(True)


def _plot_inferred_spike_raster_with_bursts(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot thresholded spike raster with superimposed *global* burst periods.

    Network bursts are always computed from ALL active ROIs for the given run
    (global network activity). The ROI selection only affects which raster rows
    are drawn, not how bursts are defined.
    """
    from cali.plot._single_wells_plots.raster._plot_inferred_spike_raster_plots import (
        _generate_spike_raster_plot,
    )

    plot = widget.plot_item
    assert plot is not None

    # Disconnect any hover handlers from previous plots
    scene = plot.scene()
    handler_names = [
        "sync_hover_handler",
        "ccorr_hover_handler",
        "spike_sync_hover_handler",
        "spike_ccorr_hover_handler",
        "spike_maxlag_hover_handler",
        "spike_maxlag_values_hover_handler",
        "dff_corr_hover_handler",
        "evoked_hover_handler",
    ]
    for handler_name in handler_names:
        old_handler = plot.property(handler_name)
        if old_handler is not None:
            try:
                scene.sigMouseMoved.disconnect(old_handler)
            except (TypeError, RuntimeError):
                pass
            plot.setProperty(handler_name, None)

    # ---- remove ROI legend for this plot ----
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.clear()
        cali_logger.warning(
            "No run_id provided for inferred spike raster with bursts plot."
        )
        plot.setTitle(
            "No analysis run selected.\nPlease select a run from the dropdown."
        )
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "ROI")
        return

    # ------------- Get pre-computed burst data from FOVAnalysis -------------#
    fov_analysis = _get_fov_analysis_for_run(engine, fov_name, run_id)

    bursts: list[tuple[int, int]] = []
    if fov_analysis is not None and fov_analysis.spike_burst_starts is not None:
        # Use stored burst data
        burst_starts = fov_analysis.spike_burst_starts
        burst_ends = fov_analysis.spike_burst_ends or []
        bursts = list(zip(burst_starts, burst_ends))

    # -------------------- Plot raster (subset) -------------------#
    _generate_spike_raster_plot(widget, engine, fov_name, rois=rois, run_id=run_id)

    # ------------------------ Overlay global bursts ------------------------ #
    if bursts and len(bursts) > 0:
        plot = widget.plot_item
        assert plot is not None

        # Burst indices are frame indices, and the underlying raster plot
        # uses frame indices (0, 1, 2, ...) for x-axis, so we use them directly
        for start_idx, end_idx in bursts:
            if end_idx <= start_idx:
                continue

            # Use frame indices directly (matching the underlying plot)
            region = pg.LinearRegionItem(
                values=(float(start_idx), float(end_idx - 1)),
                brush=pg.mkBrush(0, 255, 0, 90),
                pen=pg.mkPen(None),
                movable=False,
            )
            region.setZValue(-5)  # behind raster dots
            plot.addItem(region)

        # Update title to indicate burst overlay
        plot.setTitle(
            "Inferred Spike Events (binary) Raster Plot (Thresholded) "
            "with Network Bursts"
        )

        # Add legend for detected bursts
        legend = getattr(widget, "legend", None)
        if legend is not None:
            legend.clear()
            # Use solid green line with width 3 for burst legend
            burst_item = pg.PlotDataItem(pen=pg.mkPen((0, 255, 0), width=3))
            legend.addItem(burst_item, "Detected Bursts")
            legend.setVisible(True)


# -----------------------------------------------------------------------------#
# Normalized calcium traces + global bursts overlay
# -----------------------------------------------------------------------------#
def _plot_calcium_normalized_with_bursts(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot normalized calcium traces with superimposed *global* burst periods.

    Network bursts are always computed from ALL active ROIs for the given run
    (global network activity). The ROI selection only affects which traces are
    drawn, not how bursts are defined.
    """
    from cali.plot._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
        _plot_traces_data,
    )

    plot = widget.plot_item
    assert plot is not None

    # Disconnect any hover handlers from previous plots
    scene = plot.scene()
    handler_names = [
        "sync_hover_handler",
        "ccorr_hover_handler",
        "spike_sync_hover_handler",
        "spike_ccorr_hover_handler",
        "spike_maxlag_hover_handler",
        "spike_maxlag_values_hover_handler",
        "dff_corr_hover_handler",
        "evoked_hover_handler",
    ]
    for handler_name in handler_names:
        old_handler = plot.property(handler_name)
        if old_handler is not None:
            try:
                scene.sigMouseMoved.disconnect(old_handler)
            except (TypeError, RuntimeError):
                pass
            plot.setProperty(handler_name, None)

    # ---- remove ROI legend for this plot ----
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.clear()
        cali_logger.warning(
            "No run_id provided for calcium traces normalized with bursts plot."
        )
        plot.setTitle(
            "No analysis run selected.\nPlease select a run from the dropdown."
        )
        plot.setLabel("bottom", "Time (s)")
        plot.setLabel("left", "Deconvolved ΔF/F0 (a.u.)")
        return

    # ------------- Get pre-computed burst data from FOVAnalysis -------------#
    fov_analysis = _get_fov_analysis_for_run(engine, fov_name, run_id)

    bursts: list[tuple[int, int]] = []
    if fov_analysis is not None and fov_analysis.calcium_burst_starts is not None:
        # Use stored burst data
        burst_starts = fov_analysis.calcium_burst_starts
        burst_ends = fov_analysis.calcium_burst_ends or []
        bursts = list(zip(burst_starts, burst_ends))

    # -------------------- Plot normalized calcium traces (subset) -------------------#
    _plot_traces_data(
        widget,
        engine,
        fov_name,
        rois,
        run_id=run_id,
        raw=False,
        dff=False,
        dec=True,
        normalize=True,
        with_peaks=False,
        active_only=False,
        thresholds=False,
    )

    # ------------------------ Overlay global bursts ------------------------ #
    if bursts and len(bursts) > 0:
        plot = widget.plot_item
        assert plot is not None

        # Burst indices are frame indices, and the underlying raster plot
        # uses frame indices (0, 1, 2, ...) for x-axis, so we use them directly
        for start_idx, end_idx in bursts:
            if end_idx <= start_idx:
                continue

            # Use frame indices directly (matching the underlying plot)
            region = pg.LinearRegionItem(
                values=(float(start_idx), float(end_idx - 1)),
                brush=pg.mkBrush(0, 255, 0, 90),
                pen=pg.mkPen(None),
                movable=False,
            )
            region.setZValue(-5)  # behind traces
            plot.addItem(region)

        # Add legend for detected bursts
        legend = getattr(widget, "legend", None)
        if legend is not None:
            legend.clear()
            # Use solid green line with width 3 for burst legend
            burst_item = pg.PlotDataItem(pen=pg.mkPen((0, 255, 0), width=3))
            legend.addItem(burst_item, "Detected Bursts")
            legend.setVisible(True)


def _plot_calcium_raster_with_bursts(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot calcium peaks raster with superimposed *global* burst periods.

    Network bursts are always computed from ALL active ROIs for the given run
    (global network activity). The ROI selection only affects which raster rows
    are drawn, not how bursts are defined.
    """
    from cali.plot._single_wells_plots.raster._plot_calcium_peaks_raster_plots import (
        _generate_raster_plot,
    )

    plot = widget.plot_item
    assert plot is not None

    # Disconnect any hover handlers from previous plots
    scene = plot.scene()
    handler_names = [
        "sync_hover_handler",
        "ccorr_hover_handler",
        "spike_sync_hover_handler",
        "spike_ccorr_hover_handler",
        "spike_maxlag_hover_handler",
        "spike_maxlag_values_hover_handler",
        "dff_corr_hover_handler",
        "evoked_hover_handler",
    ]
    for handler_name in handler_names:
        old_handler = plot.property(handler_name)
        if old_handler is not None:
            try:
                scene.sigMouseMoved.disconnect(old_handler)
            except (TypeError, RuntimeError):
                pass
            plot.setProperty(handler_name, None)

    # ---- remove ROI legend for this plot ----
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        plot.clear()
        cali_logger.warning(
            "No run_id provided for calcium peaks raster with bursts plot."
        )
        plot.setTitle(
            "No analysis run selected.\nPlease select a run from the dropdown."
        )
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "ROI")
        return

    # ------------- Get pre-computed burst data from FOVAnalysis -------------#
    fov_analysis = _get_fov_analysis_for_run(engine, fov_name, run_id)

    bursts: list[tuple[int, int]] = []
    if fov_analysis is not None and fov_analysis.calcium_burst_starts is not None:
        # Use stored burst data
        burst_starts = fov_analysis.calcium_burst_starts
        burst_ends = fov_analysis.calcium_burst_ends or []
        bursts = list(zip(burst_starts, burst_ends))

    # -------------------- Plot raster (subset) -------------------#
    _generate_raster_plot(
        widget,
        engine,
        fov_name,
        rois=rois,
        run_id=run_id,
        amplitude_colors=False,
        colorbar=False,
    )

    # ------------------------ Overlay global bursts ------------------------ #
    if bursts and len(bursts) > 0:
        plot = widget.plot_item
        assert plot is not None

        for start_idx, end_idx in bursts:
            start_idx = max(start_idx, 0)
            # Use frame indices directly (matching raster x-axis)
            if end_idx <= start_idx:
                continue

            # Frame indices for x-axis (matching raster)
            x_start = float(start_idx)
            x_end = float(end_idx - 1)

            region = pg.LinearRegionItem(
                values=(x_start, x_end),
                brush=pg.mkBrush(0, 255, 0, 90),
                pen=pg.mkPen(None),
                movable=False,
            )
            region.setZValue(-5)  # behind raster dots
            plot.addItem(region)

        # Update title to indicate burst overlay
        plot.setTitle("Calcium Peaks Raster Plot with Network Bursts")

        # Add legend for detected bursts
        legend = getattr(widget, "legend", None)
        if legend is not None:
            legend.clear()
            # Use solid green line with width 3 for burst legend
            burst_item = pg.PlotDataItem(pen=pg.mkPen((0, 255, 0), width=3))
            legend.addItem(burst_item, "Detected Bursts")
            legend.setVisible(True)


# -----------------------------------------------------------------------------#
# Calcium burst plotting (using deconvolved DF/F)
# -----------------------------------------------------------------------------#
def _plot_calcium_burst_activity(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot burst detection and network state analysis for calcium activity (pyqtgraph).

    This analyzes population-level calcium activity (deconvolved DF/F) to detect
    synchronized burst events and displays burst statistics.
    """
    plot = widget.plot_item
    assert plot is not None

    # Clear previous plot
    plot.clear()

    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)
    vb.enableAutoRange(x=True, y=True)

    # Disconnect any hover handlers from previous plots
    scene = plot.scene()
    handler_names = [
        "sync_hover_handler",
        "ccorr_hover_handler",
        "spike_sync_hover_handler",
        "spike_ccorr_hover_handler",
        "spike_maxlag_hover_handler",
        "spike_maxlag_values_hover_handler",
        "dff_corr_hover_handler",
        "evoked_hover_handler",
    ]
    for handler_name in handler_names:
        old_handler = plot.property(handler_name)
        if old_handler is not None:
            try:
                scene.sigMouseMoved.disconnect(old_handler)
            except (TypeError, RuntimeError):
                pass
            plot.setProperty(handler_name, None)

    # Hide shared legend if you use one elsewhere
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    # --- Try to get pre-computed burst data from FOVAnalysis ---
    fov_analysis = _get_fov_analysis_for_run(engine, fov_name, run_id)

    # Check if pre-computed population activity data exists (regardless of burst count)
    if (
        fov_analysis is not None
        and fov_analysis.calcium_population_activity is not None
    ):
        # Use stored burst data - much faster!
        burst_starts = fov_analysis.calcium_burst_starts or []
        burst_ends = fov_analysis.calcium_burst_ends or []
        population_activity_list = fov_analysis.calcium_population_activity

        if population_activity_list:
            # calcium_population_activity contains smoothed+normalized [0,1] trace
            population_activity = np.array(population_activity_list)

            # Get raw activity if available
            population_activity_raw_list = fov_analysis.calcium_population_activity_raw
            population_activity_raw = (
                np.array(population_activity_raw_list)
                if population_activity_raw_list
                else None
            )

            # Get time axis from ALL ROIs (population activity was computed from
            # all active ROIs)
            _, _, time_axis = _get_population_calcium_data(
                engine, fov_name, rois=None, run_id=run_id
            )

            if time_axis.size == 0 or len(time_axis) != len(population_activity):
                cali_logger.warning("Time axis length mismatch with stored data")
                # Fall back to frame indices
                time_axis = np.arange(len(population_activity)) / 10.0

            # Convert stored frame indices to (start, end) tuples
            bursts = list(zip(burst_starts, burst_ends))

            # Get threshold from AnalysisSettings for display
            calcium_burst_params = _get_calcium_burst_parameters(engine, run_id)
            the_value = (
                (calcium_burst_params[0] / 100.0) if calcium_burst_params else 0.5
            )

            # --- Draw raw + smoothed+normalized activity + threshold + burst regions ---
            # Plot raw population activity (normalized, black)
            if population_activity_raw is not None:
                plot.plot(
                    time_axis,
                    population_activity_raw,
                    pen=pg.mkPen("black", width=2),
                    name="Raw Activity (Normalized)",
                )

            # Plot smoothed population activity (normalized, magenta)
            plot.plot(
                time_axis,
                population_activity,
                pen=pg.mkPen("magenta", width=3),
                name="Smoothed Activity (Normalized)",
            )

            # Threshold line
            threshold_line = pg.InfiniteLine(
                pos=the_value,
                angle=0,
                pen=pg.mkPen("magenta", width=3, style=pg.QtCore.Qt.PenStyle.DashLine),
            )
            threshold_line.setZValue(5)
            plot.addItem(threshold_line)

            # Burst regions (green translucent)
            # Burst indices are frame indices; map them to time_axis values
            for start_idx, end_idx in bursts:
                # Clamp indices to valid range
                start_idx = max(0, min(start_idx, len(time_axis) - 1))
                end_idx = max(0, min(end_idx, len(time_axis)))
                if end_idx <= start_idx:
                    continue
                # Map frame indices to time values
                t0 = float(time_axis[start_idx])
                t1 = float(time_axis[min(end_idx - 1, len(time_axis) - 1)])

                region = pg.LinearRegionItem(
                    values=[t0, t1],
                    brush=pg.mkBrush(0, 255, 0, 90),
                    pen=pg.mkPen(None),
                    movable=False,
                )
                region.setZValue(1)
                plot.addItem(region)

            # Add legend
            if hasattr(widget, "legend") and widget.legend is not None:
                widget.legend.clear()
                raw_item = pg.PlotDataItem(pen=pg.mkPen("black", width=2))
                widget.legend.addItem(raw_item, "Raw Activity (Normalized)")
                smoothed_item = pg.PlotDataItem(pen=pg.mkPen("magenta", width=3))
                widget.legend.addItem(smoothed_item, "Smoothed Activity (Normalized)")
                threshold_item = pg.PlotDataItem(
                    pen=pg.mkPen(
                        "magenta", width=3, style=pg.QtCore.Qt.PenStyle.DashLine
                    )
                )
                widget.legend.addItem(
                    threshold_item, f"Burst Threshold ({the_value:.2f})"
                )
                burst_item = pg.PlotDataItem(pen=pg.mkPen((0, 255, 0), width=3))
                widget.legend.addItem(burst_item, "Detected Bursts")
                widget.legend.setVisible(True)

            # --- Stats text in title ---
            stats_text = _burst_statistics_text(bursts, time_axis)
            title = (
                "Calcium Population Activity and Burst Detection (Deconvolved ΔF/F0)\n"
                f"{stats_text}"
            )
            plot.setTitle(title)

            plot.setLabel("bottom", "Time (s)")
            plot.setLabel("left", "Population Activity (Normalized [0,1])")

            # Auto-range once everything is added
            vb.enableAutoRange(x=True, y=True)
        else:
            cali_logger.warning(
                "Pre-computed calcium burst data is incomplete (missing population activity)"
            )
            plot.setTitle(
                "Calcium Population Burst Activity\n(No pre-computed data available - please re-run analysis)"
            )
            plot.setLabel("bottom", "Time (s)")
            plot.setLabel("left", "Population Activity (Normalized)")
    else:
        cali_logger.warning("No calcium burst analysis data found in database")
        plot.setTitle(
            "Calcium Population Burst Activity\n(No data found - please run analysis first)"
        )
        plot.setLabel("bottom", "Time (s)")
        plot.setLabel("left", "Population Activity (Normalized)")


# -----------------------------------------------------------------------------#
# Helper functions for calcium burst plotting
# -----------------------------------------------------------------------------#
def _get_calcium_burst_parameters(
    engine: Engine,
    run_id: int | None = None,
) -> tuple[float, float, float] | None:
    """Get calcium burst detection parameters from AnalysisSettings.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine
    run_id : int | None
        CaliResult ID to get settings from, or None for most recent settings

    Returns
    -------
    tuple[float, float, float] | None
        (calcium_burst_threshold, calcium_burst_min_duration_ms,
         calcium_burst_gaussian_sigma) if found, else None.
        All returned in original units from database (threshold in %, duration in ms,
        sigma in seconds).
    """
    from sqlmodel import Session, select

    from cali.sqlmodel._model import AnalysisSettings

    with Session(engine) as session:
        if run_id is not None:
            from cali.sqlmodel._model import CaliResult

            stmt = select(CaliResult).where(CaliResult.id == run_id)
            result = session.exec(stmt).first()
            if result and result.analysis_settings_id:
                # Query AnalysisSettings by ID
                settings_stmt = select(AnalysisSettings).where(
                    AnalysisSettings.id == result.analysis_settings_id
                )
                settings = session.exec(settings_stmt).first()
                if settings:
                    return (
                        settings.calcium_burst_threshold,
                        settings.calcium_burst_min_duration,
                        settings.calcium_burst_gaussian_sigma,
                    )

        # Otherwise get most recent settings
        stmt = select(AnalysisSettings).order_by(AnalysisSettings.created_at.desc())
        settings_obj = session.exec(stmt).first()
        if settings_obj:
            return (
                settings_obj.calcium_burst_threshold,
                settings_obj.calcium_burst_min_duration,
                settings_obj.calcium_burst_gaussian_sigma,
            )

    cali_logger.warning(
        "No valid analysis settings found for calcium burst parameters."
    )
    return None


def _get_population_calcium_data(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[str], np.ndarray]:
    """Extract population calcium data (deconvolved DF/F) from database.

    Returns
    -------
    (calcium_traces_array, roi_names, time_axis)
    """
    from sqlalchemy.orm import selectinload
    from sqlmodel import Session, col, select

    from cali.sqlmodel._model import FOV

    with Session(engine) as session:
        stmt = (
            select(FOV)
            .where(col(FOV.name) == fov_name)
            .options(
                selectinload(FOV.rois).selectinload(ROI.traces_history),
                selectinload(FOV.rois).selectinload(ROI.data_analysis_history),
            )
        )
        fov = session.exec(stmt).first()
        if not fov or not fov.rois:
            return None, [], np.array([])

        roi_results = [r for r in fov.rois if rois is None or r.label_value in rois]
        if not roi_results:
            return None, [], np.array([])

    calcium_traces: list[np.ndarray] = []
    roi_names: list[str] = []
    rois_rec_time: list[float] = []

    for roi in roi_results:
        traces_obj = _get_traces_for_run(roi, run_id)
        if traces_obj is None:
            continue

        dec_dff = traces_obj.dec_dff
        if dec_dff is None or len(dec_dff) == 0:
            continue

        calcium_traces.append(np.array(dec_dff, dtype=float))
        roi_names.append(str(roi.label_value))

        # Get recording time from data_analysis (more reliable than x_axis)
        data_analysis = _get_data_analysis_for_run(roi, run_id)
        if data_analysis and data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

    if len(calcium_traces) < 2:
        return None, roi_names, np.array([])

    # Pad / truncate to common length
    lengths = np.array([len(t) for t in calcium_traces], dtype=int)
    max_length = int(lengths.max())

    calcium_traces_array = np.zeros((len(calcium_traces), max_length), dtype=float)
    for i, trace in enumerate(calcium_traces):
        current_length = len(trace)
        if current_length >= max_length:
            calcium_traces_array[i, :] = trace[:max_length]
        else:
            calcium_traces_array[i, :current_length] = trace

    # Time axis from recording time if available, else frames @ 10Hz
    if rois_rec_time:
        max_time = max(rois_rec_time)
        time_axis = np.linspace(0, max_time, max_length)
    else:
        time_axis = np.arange(max_length) / 10.0

    return calcium_traces_array, roi_names, time_axis
