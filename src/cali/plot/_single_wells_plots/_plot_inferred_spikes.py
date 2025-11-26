from __future__ import annotations

from typing import TYPE_CHECKING

import mplcursors
import numpy as np
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, select

from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:

    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


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
    """Get the DataAnalysis object for a specific run from the ROI's data_analysis_history."""
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


def _plot_inferred_spikes(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    raw: bool = False,
    normalize: bool = False,
    active_only: bool = False,
    dec_dff: bool = False,
    thresholds: bool = False,
) -> None:
    """Plot inferred spikes data by querying database directly.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Graph widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV (e.g., "B5_0000")
    rois : list[int] | None
        List of ROI label values to plot. If None, plots all ROIs.
    run_id : int | None
        The CaliResult.id of the selected run. If provided, only data from this run
        will be plotted.
    raw : bool
        Plot raw inferred spikes
    normalize : bool
        Normalize traces using percentile method
    active_only : bool
        Only plot active ROIs
    dec_dff : bool
        Show deconvolved ΔF/F traces
    thresholds : bool
        Show peak detection thresholds (only if single ROI selected)
    """
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # show peaks thresholds only if only 1 roi is selected
    thresholds = thresholds if rois and len(rois) == 1 else False

    # Query database for ROI data
    with Session(engine) as session:
        roi_data = []  # List of (ROI, Traces, DataAnalysis)

        if run_id is not None:
            # Optimized query
            stmt = (
                select(ROI, Traces, DataAnalysis)
                .join(FOV, ROI.fov_id == FOV.id)
                .join(
                    Traces,
                    (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
                )
                .join(
                    DataAnalysis,
                    (DataAnalysis.roi_id == ROI.id)
                    & (DataAnalysis.analysis_result_id == run_id),
                )
                .where(col(FOV.name) == fov_name)
            )

            if rois is not None:
                stmt = stmt.where(col(ROI.label_value).in_(rois))

            if active_only:
                stmt = stmt.where(col(ROI.active) == True)  # noqa: E712

            stmt = stmt.order_by(col(ROI.label_value))

            results = session.exec(stmt).all()
            roi_data = results
        else:
            # Legacy behavior
            stmt = (
                select(ROI)
                .join(FOV)
                .where(col(FOV.name) == fov_name)
                .options(
                    selectinload(ROI.traces_history),
                    selectinload(ROI.data_analysis_history),
                )
            )

            if rois is not None:
                stmt = stmt.where(col(ROI.label_value).in_(rois))

            if active_only:
                stmt = stmt.where(col(ROI.active) == True)  # noqa: E712

            stmt = stmt.order_by(col(ROI.label_value))

            roi_models = session.exec(stmt).all()

            for r in roi_models:
                t = _get_traces_for_run(r, None)
                da = _get_data_analysis_for_run(r, None)
                if t and da:
                    roi_data.append((r, t, da))

    # compute percentiles for normalization if needed
    p1 = p2 = 0.0
    if normalize:
        all_values = []
        for _, _, data_analysis in roi_data:
            if data_analysis and data_analysis.inferred_spikes:
                # Use inferred_spikes as the spike data
                spike_data = data_analysis.inferred_spikes
                if raw:
                    # For raw, use all values above 0
                    spike_values = [s for s in spike_data if s > 0]
                else:
                    # For thresholded, use values above threshold
                    threshold = data_analysis.inferred_spikes_threshold or 0
                    spike_values = [s for s in spike_data if s > threshold]
                all_values.extend(spike_values)

        if all_values:
            percentiles = np.percentile(all_values, [5, 100])
            p1, p2 = float(percentiles[0]), float(percentiles[1])
        else:
            p1, p2 = 0.0, 1.0

    count = 0
    rois_rec_time: list[float] = []
    last_trace: list[float] | None = None

    for roi, traces, data_analysis in roi_data:
        if data_analysis is None or not data_analysis.inferred_spikes:
            continue

        if data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

        # Get spike data based on raw/thresholded mode
        if raw:
            spike_data = [s if s > 0 else 0 for s in data_analysis.inferred_spikes]
        else:
            threshold = data_analysis.inferred_spikes_threshold or 0
            spike_data = [
                s if s > threshold else 0 for s in data_analysis.inferred_spikes
            ]

        _plot_trace(
            ax,
            str(roi.label_value),
            spike_data,
            normalize,
            count,
            p1,
            p2,
            thresholds,
            data_analysis.inferred_spikes_threshold,
        )
        if dec_dff and traces and traces.dec_dff:
            _plot_trace(
                ax, str(roi.label_value), traces.dec_dff, normalize, count, p1, p2
            )
        last_trace = data_analysis.inferred_spikes
        count += 1

    _set_graph_title_and_labels(ax, normalize, raw)

    _update_time_axis(ax, rois_rec_time, last_trace)

    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_trace(
    ax: Axes,
    roi_key: str,
    trace: list[float] | None,
    normalize: bool,
    count: int,
    p1: float,
    p2: float,
    thresholds: bool = False,
    spikes_threshold: float | None = None,
) -> None:
    """Plot inferred spikes trace with optional percentile-based normalization."""
    if trace is None or not trace:
        return
    if normalize:
        offset = count * 1.1  # vertical offset
        spike_trace = _normalize_trace_percentile(trace, p1, p2) + offset
        ax.plot(spike_trace, label=f"ROI {roi_key}")
        ax.set_yticks([])
        ax.set_yticklabels([])
    else:
        ax.plot(trace, label=f"ROI {roi_key}")

    # Add horizontal line for spike detection threshold
    if thresholds and spikes_threshold is not None and spikes_threshold > 0.0:
        ax.axhline(
            y=spikes_threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.6,
            label=f"Spike threshold (ROI {roi_key} - {spikes_threshold:.4f})",
        )


def _normalize_trace_percentile(trace: list[float], p1: float, p2: float) -> np.ndarray:
    """Normalize a trace using p1th-p2th percentile, clipped to [0, 1]."""
    tr = np.array(trace)
    denom = p2 - p1
    if denom == 0:
        return np.zeros_like(tr)
    normalized = (tr - p1) / denom
    return np.clip(normalized, 0, 1)


def _set_graph_title_and_labels(ax: Axes, normalize: bool, raw: bool) -> None:
    """Set axis labels based on the plotted data."""
    title = ("Normalized Inferred Spikes" if normalize else "Inferred Spikes") + (
        " (Raw)" if raw else " (Thresholded Spike Data)"
    )
    y_lbl = "ROIs" if normalize else "Inferred Spikes (magnitude)"

    ax.set_title(title)
    ax.set_ylabel(y_lbl)


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    """Update the time axis based on recording time."""
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

        # Show hover for anything with ROI in the label (traces and thresholds)
        if label and "ROI" in label and not label.startswith("_"):
            sel.annotation.set(text=label, fontsize=8, color="black")
            # Extract ROI number for selection (works for both traces and thresholds)
            roi_parts = label.split("ROI ")
            if len(roi_parts) > 1:
                roi_num = roi_parts[1].split()[0] if roi_parts[1].split() else ""
                if roi_num.isdigit():
                    widget.roiSelected.emit(roi_num)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)


def _plot_inferred_spikes_normalized_with_bursts(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot normalized inferred spikes with superimposed burst periods.

    This combines the normalized spike traces visualization with burst detection
    to show when network bursts occur overlaid on the individual ROI traces.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        List of ROI indices to include, None for all active ROIs
    run_id : int | None
        The run ID to filter by, None for latest
    """
    # Clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # For now, delegate to the basic normalized spikes plot
    # TODO: Add burst detection overlay
    _plot_inferred_spikes(widget, engine, fov_name, rois, run_id=run_id, normalize=True)

    # Add note that burst overlay is not yet implemented
    ax.text(
        0.5,
        0.98,
        "Note: Network burst overlay not yet implemented",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        color="gray",
    )


def _detect_population_bursts(
    population_activity: np.ndarray,
    burst_threshold: float,
    min_duration: int,
) -> list[tuple[int, int]]:
    """Detect population bursts in the smoothed activity."""
    # Find regions above threshold
    above_threshold = population_activity > burst_threshold

    # Find start and end points of bursts
    bursts = []
    in_burst = False
    burst_start = 0

    for i, is_active in enumerate(above_threshold):
        if is_active and not in_burst:
            # Start of new burst
            burst_start = i
            in_burst = True
        elif not is_active and in_burst:
            # End of burst
            burst_duration = i - burst_start
            if burst_duration >= min_duration:
                bursts.append((burst_start, i))
            in_burst = False

    # Handle case where burst extends to the end
    if in_burst and (len(population_activity) - burst_start) >= min_duration:
        bursts.append((burst_start, len(population_activity)))

    return bursts


def _overlay_burst_periods(
    ax: Axes, bursts: list[tuple[int, int]], num_rois: int
) -> None:
    """Overlay burst periods as shaded regions on the plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    bursts : list[tuple[int, int]]
        List of (start, end) indices for bursts
    num_rois : int
        Number of ROIs plotted (for determining y-axis span)
    """
    if not bursts:
        return

    for i, (start, end) in enumerate(bursts):
        label = "Network Burst" if i == 0 else ""
        ax.axvspan(start, end, alpha=0.2, color="green", label=label)
