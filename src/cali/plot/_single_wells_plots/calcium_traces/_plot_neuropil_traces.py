"""Plot raw traces with neuropil correction visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cmap
import numpy as np
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._hover_utils import setup_pick_click
from cali.sqlmodel._model import FOV, ROI, CaliResult, DataAnalysis, Traces

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget

from matplotlib.lines import Line2D


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
    # First try to find exact match
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    # Fall back to first entry (for backwards compatibility with data that has
    # analysis_result_id=None)
    return (
        roi_model.data_analysis_history[0] if roi_model.data_analysis_history else None
    )


def _plot_neuropil_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot all raw and neuropil traces together on widget canvas.

    ...by querying database directly.

    Raw traces and neuropil traces are plotted on the same axes,
    allowing the filtering logic to isolate specific ROI pairs.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        The widget containing the matplotlib figure and canvas
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV (e.g., "B5_0000")
    rois : list[int] | None
        List of specific ROI IDs to plot, or None for all
    run_id : int | None
        The CaliResult.id of the selected run. If provided, only data from this run
        will be plotted.
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    if run_id is None:
        cali_logger.warning("No run_id provided for neuropil traces plot.")
        ax.text(
            0.5,
            0.5,
            "No run selected.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
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

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        roi_models = session.exec(stmt).all()

    # ---------- COLLECT VALID ROIS & TRACES (VECTOR-FRIENDLY) ----------
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

        # Ensure they all have same length; if not, you could pad/crop here.
        if not (len(raw) == len(neu) == len(corr)):
            # For safety, skip inconsistent entries
            continue

        labels.append(roi.label_value)
        raw_traces.append(raw)
        neuropil_traces.append(neu)
        corrected_traces.append(corr)

        # Recording time if available
        data_analysis = _get_data_analysis_for_run(roi, run_id)
        if (
            data_analysis is not None
            and data_analysis.total_recording_time_sec is not None
        ):
            rois_rec_time.append(data_analysis.total_recording_time_sec)

    if not raw_traces:
        # No valid data to plot
        ax.text(
            0.5,
            0.5,
            "No neuropil traces available.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Stack into arrays: shape = (n_rois, T)
    Y_raw = np.vstack(raw_traces)
    Y_neu = np.vstack(neuropil_traces)
    Y_corr = np.vstack(corrected_traces)
    n_rois, T = Y_raw.shape
    frames = np.arange(T)

    last_trace = Y_raw[0]  # for time axis ticks

    # ---------- COLORS (GLASBEY) ----------
    glasbey_cmap = cmap.Colormap("glasbey").to_matplotlib()
    color_indices = np.linspace(0.05, 1, n_rois)  # avoid very dark colors
    colors = glasbey_cmap(color_indices)

    # ---------- VECTORIZED PLOTTING ----------
    # One plot call per type (raw / neuropil / corrected)
    # ax.plot(frames, Y.T) => multiple lines
    lines_raw = ax.plot(
        frames,
        Y_raw.T,
        linewidth=1,
        linestyle="--",
        picker=3,
    )
    lines_neu = ax.plot(
        frames,
        Y_neu.T,
        linewidth=1,
        linestyle=":",
        picker=3,
    )
    lines_corr = ax.plot(
        frames,
        Y_corr.T,
        linewidth=1,
        linestyle="-",
        picker=3,
    )

    # Assign colors, labels & metadata for hover
    for i, roi_id in enumerate(labels):
        color = colors[i]

        # Raw
        line_r = lines_raw[i]
        line_r.set_color(color)
        line_r.set_label(f"Raw ROI {roi_id}")
        line_r._roi_label = roi_id
        line_r._trace_type = "raw"

        # Neuropil
        line_n = lines_neu[i]
        line_n.set_color(color)
        line_n.set_label(f"Neuropil ROI {roi_id}")
        line_n._roi_label = roi_id
        line_n._trace_type = "neuropil"

        # Corrected
        line_c = lines_corr[i]
        line_c.set_color(color)
        line_c.set_label(f"Corrected ROI {roi_id}")
        line_c._roi_label = roi_id
        line_c._trace_type = "corrected"

    # ---------- FORMATTING ----------
    ax.set_ylabel("Fluorescence (a.u.)", fontsize=11)
    ax.set_title("Raw, Neuropil, and Corrected Traces", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Custom legend explaining line styles
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=1, linestyle="--", label="Raw"),
        Line2D([0], [0], color="gray", linewidth=1, linestyle=":", label="Neuropil"),
        Line2D([0], [0], color="gray", linewidth=1, linestyle="-", label="Corrected"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        framealpha=0.9,
        fontsize=9,
    )

    # ---------- TIME AXIS ----------
    _update_time_axis(ax, rois_rec_time, last_trace)

    # ---------- HOVER FUNCTIONALITY ----------
    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using efficient pick events."""
    setup_pick_click(ax, widget, picker_tolerance=3)


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: np.ndarray | None
) -> None:
    """Update x-axis to show time instead of frames if recording time is available."""
    if trace is None or sum(rois_rec_time) <= 0:
        ax.set_xlabel("Frame", fontsize=11)
        return
    # Get the average total recording time in seconds
    avg_rec_time = int(np.mean(rois_rec_time))
    # Get total number of frames from the trace
    total_frames = len(trace)
    # Compute tick positions
    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)", fontsize=11)
