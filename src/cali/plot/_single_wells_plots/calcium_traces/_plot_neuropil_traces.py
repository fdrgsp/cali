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
) -> None:
    """Plot raw, neuropil, and corrected traces using pyqtgraph.

    Raw, neuropil, and corrected traces are plotted on the same axes for each ROI.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
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
        if (
            traces is None
            or traces.raw_trace is None
            or traces.neuropil_trace is None
            or traces.corrected_trace is None
        ):
            continue

        raw = np.asarray(traces.raw_trace, dtype=float)
        neu = np.asarray(traces.neuropil_trace, dtype=float)
        corr = np.asarray(traces.corrected_trace, dtype=float)

        # Ensure same length; otherwise skip ROI
        if not (len(raw) == len(neu) == len(corr)):
            continue

        labels.append(roi.label_value)
        raw_traces.append(raw)
        neuropil_traces.append(neu)
        corrected_traces.append(corr)

        data_analysis = _get_data_analysis_for_run(roi, run_id)
        if (
            data_analysis is not None
            and data_analysis.total_recording_time_sec is not None
        ):
            rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not raw_traces:
        plot.setTitle("No neuropil traces available.")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "Fluorescence (a.u.)")
        return

    # Stack into arrays: shape = (n_rois, T_orig)
    Y_raw = np.vstack(raw_traces)
    Y_neu = np.vstack(neuropil_traces)
    Y_corr = np.vstack(corrected_traces)

    Y_raw = np.nan_to_num(Y_raw, nan=0.0)
    Y_neu = np.nan_to_num(Y_neu, nan=0.0)
    Y_corr = np.nan_to_num(Y_corr, nan=0.0)

    n_rois, T_orig = Y_raw.shape
    x_full = np.arange(T_orig, dtype=float)

    # ---------- DOWNSAMPLING ----------
    stride = 1
    if T_orig > MAX_POINTS:
        stride = int(np.ceil(T_orig / MAX_POINTS))

    x = x_full[::stride]
    Y_raw_ds = Y_raw[:, ::stride]
    Y_neu_ds = Y_neu[:, ::stride]
    Y_corr_ds = Y_corr[:, ::stride]

    # ---------- COLORS ----------
    # Fixed semantics: magenta = raw, yellow = neuropil, green = corrected
    pen_raw = pg.mkPen("magenta", width=1)
    pen_neu = pg.mkPen("yellow", width=1)
    pen_corr = pg.mkPen("green", width=1)

    # ---------- PLOTTING ----------
    for i in range(n_rois):
        plot.plot(x, Y_raw_ds[i], pen=pen_raw)
        plot.plot(x, Y_neu_ds[i], pen=pen_neu)
        plot.plot(x, Y_corr_ds[i], pen=pen_corr)

    # ---------- AXES / TITLES ----------
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

        legend.setVisible(True)
        _populate_neuropil_legend(legend, pen_raw, pen_neu, pen_corr)


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
