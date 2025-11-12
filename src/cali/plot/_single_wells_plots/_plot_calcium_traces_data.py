from __future__ import annotations

from typing import TYPE_CHECKING, cast

import mplcursors
import numpy as np
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, create_engine, select

from cali.sqlmodel._model import FOV, ROI

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes

    from cali.gui._graph_widgets import _SingleWellGraphWidget


COUNT_INCREMENT = 1
P1 = 5
P2 = 100


def _plot_traces_data(
    widget: _SingleWellGraphWidget,
    db_path: str | Path,
    fov_name: str,
    rois: list[int] | None = None,
    raw: bool = False,
    dff: bool = False,
    dec: bool = False,
    normalize: bool = False,
    with_peaks: bool = False,
    active_only: bool = False,
    thresholds: bool = False,
) -> None:
    """Plot traces data by querying database directly.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Graph widget to plot on
    db_path : str | Path
        Path to the SQLite database
    fov_name : str
        Name of the FOV (e.g., "B5_0000")
    rois : list[int] | None
        List of ROI label values to plot. If None, plots all ROIs.
    raw : bool
        Plot raw traces
    dff : bool
        Plot ΔF/F traces
    dec : bool
        Plot deconvolved ΔF/F traces
    normalize : bool
        Normalize traces using percentile method
    with_peaks : bool
        Show detected peaks
    active_only : bool
        Only plot active ROIs
    thresholds : bool
        Show peak detection thresholds (only if single ROI selected)
    """
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # show peaks thresholds only if only 1 roi is selected
    thresholds = thresholds if rois and len(rois) == 1 else False

    # Query database for ROI data
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    with Session(engine) as session:
        # Build query to get ROIs for this FOV with eager loading of related data
        stmt = (
            select(ROI)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .options(
                selectinload(ROI.traces),
                selectinload(ROI.data_analysis),
            )
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Filter by active if requested
        if active_only or with_peaks:
            stmt = stmt.where(col(ROI.active) == True)  # noqa: E712

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        roi_models = session.exec(stmt).all()

    engine.dispose(close=True)

    if not roi_models:
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # compute nth and nth percentiles globally
    p1 = p2 = 0.0
    if normalize:
        all_values = []
        for roi_model in roi_models:
            if trace := _get_trace_from_model(roi_model, dff, dec, raw):
                all_values.extend(trace)
        if all_values:
            percentiles = np.percentile(all_values, [P1, P2])
            p1, p2 = float(percentiles[0]), float(percentiles[1])
        else:
            p1, p2 = 0.0, 1.0

    count = 0
    rois_rec_time: list[float] = []
    last_trace: list[float] | None = None

    for roi_model in roi_models:
        trace = _get_trace_from_model(roi_model, dff, dec, raw)

        if not trace:
            continue

        # Get recording time from data_analysis if available
        if roi_model.data_analysis and (
            ttime := roi_model.data_analysis.total_recording_time_sec
        ) is not None:
            rois_rec_time.append(ttime)

        _plot_trace(
            ax,
            str(roi_model.label_value),
            trace,
            normalize,
            with_peaks,
            roi_model,
            count,
            p1,
            p2,
            thresholds,
        )
        last_trace = trace
        count += COUNT_INCREMENT

    _set_graph_title_and_labels(ax, dff, dec, normalize, with_peaks)

    _update_time_axis(ax, rois_rec_time, last_trace)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _get_trace_from_model(
    roi_model: ROI, dff: bool, dec: bool, raw: bool
) -> list[float] | None:
    """Get the appropriate trace from ROI model based on the flags."""
    if not roi_model.traces:
        return None

    try:
        if dff:
            data = roi_model.traces.dff
        elif dec:
            data = roi_model.traces.dec_dff
        elif raw:
            data = roi_model.traces.raw_trace
        else:
            data = roi_model.traces.corrected_trace
        return data or None
    except AttributeError:
        return None


def _plot_trace(
    ax: Axes,
    roi_key: str,
    trace: list[float] | np.ndarray,
    normalize: bool,
    with_peaks: bool,
    roi_model: ROI,
    count: int,
    p1: float,
    p2: float,
    thresholds: bool = False,
) -> None:
    """Plot trace data with optional percentile-based normalization and peaks."""
    offset = count * 1.1  # vertical offset

    if normalize:
        trace = _normalize_trace_percentile(trace, p1, p2) + offset
        ax.plot(trace, label=f"ROI {roi_key}")
        ax.set_yticks([])
        ax.set_yticklabels([])
    else:
        ax.plot(trace, label=f"ROI {roi_key}")

    # Get peaks data from data_analysis if available
    if with_peaks and roi_model.data_analysis and roi_model.data_analysis.peaks_dec_dff:
        peaks_indices = [int(p) for p in roi_model.data_analysis.peaks_dec_dff]
        ax.plot(peaks_indices, np.array(trace)[peaks_indices], "x")

        # Add vertical lines for peaks height and prominence thresholds
        if thresholds:
            if roi_model.data_analysis.peaks_height_dec_dff is not None:
                # Horizontal dashed line for height threshold
                ph = roi_model.data_analysis.peaks_height_dec_dff
                ax.axhline(
                    y=ph,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.6,
                    label=f"Peaks Height threshold\n(ROI {roi_key} - {ph:.4f})",
                )
            if roi_model.data_analysis.peaks_prominence_dec_dff is not None:
                # Vertical line from 0 to prominence threshold value
                pp = roi_model.data_analysis.peaks_prominence_dec_dff
                ax.plot(
                    [-3, -3],
                    [0, pp],
                    color="orange",
                    linestyle="-",
                    linewidth=5,
                    alpha=0.8,
                    label=f"Peaks Prominence Threshold \n(ROI {roi_key} - {pp:.4f})",
                )


def _normalize_trace_percentile(
    trace: list[float] | np.ndarray, p1: float, p2: float
) -> np.ndarray:
    """Normalize a trace using p1th-p2th percentile, clipped to [0, 1]."""
    tr = np.array(trace) if isinstance(trace, list) else trace
    denom = p2 - p1
    if denom == 0:
        return np.zeros_like(tr)
    normalized = (tr - p1) / denom
    return np.clip(normalized, 0, 1)


def _set_graph_title_and_labels(
    ax: Axes,
    dff: bool,
    dec: bool,
    normalize: bool,
    with_peaks: bool,
) -> None:
    """Set axis labels based on the plotted data."""
    if dff:
        title = (
            "Normalized Calcium Traces (ΔF/F)" if normalize else "Calcium Traces (ΔF/F)"
        )
        y_lbl = "ROIs" if normalize else "ΔF/F"
    elif dec:
        title = (
            "Normalized Calcium Traces (Deconvolved ΔF/F)"
            if normalize
            else "Calcium Traces (Deconvolved ΔF/F)"
        )
        y_lbl = "ROIs" if normalize else "Deconvolved ΔF/F"
    else:
        title = "Normalized Calcium Traces" if normalize else "Raw Calcium Traces"
        y_lbl = "ROIs" if normalize else "Fluorescence Intensity"
    if with_peaks:
        title += " with Peaks"

    ax.set_title(title)
    ax.set_ylabel(y_lbl)


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    if trace is None or sum(rois_rec_time) <= 0:
        ax.set_xlabel("Frames")
        return
    # get the average total recording time in seconds
    avg_rec_time = int(np.mean(rois_rec_time))
    # get total number of frames from the trace
    total_frames = len(trace) if trace is not None else 1
    # compute tick positions
    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the label of the artist
        label = sel.artist.get_label()

        # Only show hover for ROI traces, not for peaks or other elements
        if label and "ROI" in label and not label.startswith("_"):
            sel.annotation.set(text=label, fontsize=8, color="black")
            roi = cast("str", label.split(" ")[1])
            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)
