"""Plot raw, neuropil, and corrected traces together using pyqtgraph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, ROI, CaliResult, DataAnalysis, Traces

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# max number of time points we will draw per trace (automatic downsampling)
MAX_POINTS = 2000


# -----------------------------------------------------------------------------#
# Helpers: retrieval from ROI histories
# -----------------------------------------------------------------------------#
def _get_traces_for_run(roi_model: ROI, run_id: int | None) -> Traces | None:
    """Get the Traces object for a specific run from the ROI's traces_history."""
    if not roi_model.traces_history:
        return None
    if run_id is None:
        return roi_model.traces_history[0] if roi_model.traces_history else None
    for trace in roi_model.traces_history:
        if trace.analysis_result_id == run_id:
            return trace
    return None


def _get_neuropil_traces(
    corrected: bool, trace_obj: Traces
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Get raw, neuropil, and corrected traces based on corrected flag.

    Returns (raw_trace, neuropil_trace, corrected_trace) tuple.
    If corrected=True, only returns corrected_trace (others are None).
    If corrected=False, returns all three traces.
    """
    if corrected:
        # Only return corrected trace
        corr = trace_obj.corrected_trace
        if corr is None:
            return None, None, None
        return None, None, np.asarray(corr, dtype=float)
    else:
        # Return all three traces
        raw = trace_obj.raw_trace
        neu = trace_obj.neuropil_trace
        corr = trace_obj.corrected_trace

        # All three must exist for full neuropil visualization
        if raw is None or neu is None or corr is None:
            return None, None, None

        return (
            np.asarray(raw, dtype=float),
            np.asarray(neu, dtype=float),
            np.asarray(corr, dtype=float),
        )


def _get_data_analysis_for_run(
    roi_model: ROI, run_id: int | None
) -> DataAnalysis | None:
    """Get DataAnalysis for a specific run from ROI's data_analysis_history."""
    if not roi_model.data_analysis_history:
        return None
    if run_id is None:
        return (
            roi_model.data_analysis_history[0]
            if roi_model.data_analysis_history
            else None
        )
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    return (
        roi_model.data_analysis_history[0] if roi_model.data_analysis_history else None
    )


# -----------------------------------------------------------------------------#
# Main plotting entry point (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_neuropil_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    corrected: bool = False,
) -> None:
    """Plot neuropil-related traces using pyqtgraph.

    When corrected=True: Only plots corrected traces.
    When corrected=False: Plots raw, neuropil, and corrected traces together.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        The widget containing the pyqtgraph plot item
    engine : Engine
        Database engine for querying traces
    fov_name : str
        Name of the field of view to plot
    rois : list[int] | None
        List of specific ROI IDs (label_values) to plot, or None for all
    run_id : int | None
        Analysis result ID to filter traces by
    corrected : bool
        If True, only plot corrected traces.
        If False, plot raw, neuropil, and corrected.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    # Reset ViewBox settings that might have been set by raster plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)

    # clear_plot already hid & cleared widget.legend, but just in case:
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    if run_id is None:
        cali_logger.warning("No run_id provided for neuropil traces plot.")
        plot.setTitle("No run selected.")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "Fluorescence (a.u.)")
        return

    # ---------- QUERY DATABASE ----------
    with Session(engine) as session:
        # Get detection_settings_id from the run
        result = session.get(CaliResult, run_id)

        detection_settings_id: int | None = None
        if result:
            detection_settings_id = result.detection_settings_id

        # Build query to get ROIs for this FOV with eager loading of related data
        stmt = (
            select(ROI)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .options(
                selectinload(ROI.traces_history),
                selectinload(ROI.data_analysis_history),
            )
        )

        # Filter by detection settings if we have a run_id + linked settings
        if detection_settings_id is not None:
            stmt = stmt.where(col(ROI.detection_settings_id) == detection_settings_id)

        # Filter by specific ROIs if requested (using label_value, as elsewhere)
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        roi_models = session.exec(stmt).all()

    # ---------- COLLECT VALID ROIS & TRACES ----------
    labels: list[int] = []
    raw_traces: list[np.ndarray] = []
    neuropil_traces: list[np.ndarray] = []
    corrected_traces: list[np.ndarray] = []
    rois_rec_time: list[float] = []

    for roi in roi_models:
        traces = _get_traces_for_run(roi, run_id)
        if traces is None:
            continue

        raw, neu, corr = _get_neuropil_traces(corrected, traces)

        # Skip if no valid traces returned
        if corr is None:
            continue

        # If corrected=False, all three should be present and same length
        if not corrected:
            if raw is None or neu is None:
                continue
            if not (len(raw) == len(neu) == len(corr)):
                continue

        labels.append(roi.label_value)
        if raw is not None:
            raw_traces.append(raw)
        if neu is not None:
            neuropil_traces.append(neu)
        corrected_traces.append(corr)

        data_analysis = _get_data_analysis_for_run(roi, run_id)
        if (
            data_analysis is not None
            and data_analysis.total_recording_time_sec is not None
        ):
            rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not corrected_traces:
        plot.setTitle("No neuropil traces available.")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "Fluorescence (a.u.)")
        return

    # Stack into arrays: shape = (n_rois, T_orig)
    Y_corr = np.vstack(corrected_traces)
    Y_corr = np.nan_to_num(Y_corr, nan=0.0)

    # Only stack raw and neuropil if corrected=False
    if not corrected and raw_traces and neuropil_traces:
        Y_raw = np.vstack(raw_traces)
        Y_neu = np.vstack(neuropil_traces)
        Y_raw = np.nan_to_num(Y_raw, nan=0.0)
        Y_neu = np.nan_to_num(Y_neu, nan=0.0)
    else:
        Y_raw = None
        Y_neu = None

    n_rois, T_orig = Y_corr.shape
    x_full = np.arange(T_orig, dtype=float)

    # ---------- DOWNSAMPLING ----------
    stride = 1
    if T_orig > MAX_POINTS:
        stride = int(np.ceil(T_orig / MAX_POINTS))

    x = x_full[::stride]
    Y_corr_ds = Y_corr[:, ::stride]

    if Y_raw is not None and Y_neu is not None:
        Y_raw_ds = Y_raw[:, ::stride]
        Y_neu_ds = Y_neu[:, ::stride]
    else:
        Y_raw_ds = None
        Y_neu_ds = None

    # ---------- COLORS ----------
    if corrected:
        # When showing only corrected traces, use multi-color like calcium traces
        pen_raw = None
        pen_neu = None
    else:
        # Fixed semantics when showing all: magenta = raw, yellow = neuropil
        pen_raw = pg.mkPen("magenta", width=1)
        pen_neu = pg.mkPen("yellow", width=1)

    # ---------- PLOTTING ----------
    for i in range(n_rois):
        if Y_raw_ds is not None:
            plot.plot(x, Y_raw_ds[i], pen=pen_raw)
        if Y_neu_ds is not None:
            plot.plot(x, Y_neu_ds[i], pen=pen_neu)

        # Corrected trace: multi-color if corrected=True, green otherwise
        if corrected:
            # Multi-trace → distinct colors (same logic as calcium traces)
            if n_rois == 1:
                color = "w"  # Single trace → white
            else:
                color = pg.intColor(i, hues=max(n_rois, 16))
            pen_corr = pg.mkPen(color, width=1)
        else:
            # All three traces shown → green for corrected
            pen_corr = pg.mkPen("green", width=1)

        plot.plot(x, Y_corr_ds[i], pen=pen_corr)

    # ---------- AXES / TITLES ----------
    if corrected:
        plot.setTitle("Corrected Traces")
    else:
        plot.setTitle("Raw, Neuropil, and Corrected Traces")
    plot.setLabel("left", "Fluorescence (a.u.)")

    _update_time_axis_pg_for_neuropil(
        plot,
        rois_rec_time=rois_rec_time,
        T_orig=T_orig,
    )

    plot.getViewBox().enableAutoRange(x=True, y=True)

    # ---------- LEGEND (single, reused from widget.legend) ----------
    legend = getattr(widget, "legend", None)
    if legend is not None:
        # ensure attached to this plot
        if legend.parentItem() is None:
            legend.setParentItem(plot.graphicsItem())

        # Clear old items
        if hasattr(legend, "clear"):
            legend.clear()

        if corrected:
            # When showing only corrected traces with multi-color, hide legend
            # (same behavior as calcium traces - colors are self-explanatory)
            legend.setVisible(False)
        else:
            # Show all three in legend with fixed colors
            legend.setVisible(True)
            _populate_neuropil_legend(
                legend, pen_raw, pen_neu, pg.mkPen("green", width=1)
            )


# -----------------------------------------------------------------------------#
# Legend helpers
# -----------------------------------------------------------------------------#
def _populate_neuropil_legend(
    legend: pg.LegendItem,
    pen_raw: pg.mkPen,
    pen_neu: pg.mkPen,
    pen_corr: pg.mkPen,
) -> None:
    """Ensure legend shows exactly three entries: Raw / Neuropil / Corrected."""
    # Represent each type with a tiny sample curve
    raw_sample = pg.PlotDataItem([0], [0], pen=pen_raw)
    neu_sample = pg.PlotDataItem([0], [0], pen=pen_neu)
    corr_sample = pg.PlotDataItem([0], [0], pen=pen_corr)

    legend.addItem(raw_sample, "Raw")
    legend.addItem(neu_sample, "Neuropil")
    legend.addItem(corr_sample, "Corrected")


# -----------------------------------------------------------------------------#
# Time axis helper
# -----------------------------------------------------------------------------#
def _update_time_axis_pg_for_neuropil(
    plot: pg.PlotItem,
    rois_rec_time: list[float],
    T_orig: int,
) -> None:
    """Configure bottom axis as time in seconds if recording time is available."""
    if not rois_rec_time or sum(rois_rec_time) <= 0 or T_orig <= 1:
        plot.setLabel("bottom", "Frames")
        return

    avg_rec_time = int(np.mean(rois_rec_time))
    x_ticks = np.linspace(0, T_orig, num=5, dtype=int)
    tick_interval = avg_rec_time / T_orig
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]

    axis = plot.getAxis("bottom")
    axis.setTicks([list(zip(x_ticks.tolist(), x_labels))])
    plot.setLabel("bottom", "Time (s)")
