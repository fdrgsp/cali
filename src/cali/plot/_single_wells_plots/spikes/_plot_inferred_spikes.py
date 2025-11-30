from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._hover_utils import setup_pick_click
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


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
    for analysis in roi_model.data_analysis_history:
        if analysis.analysis_result_id == run_id:
            return analysis
    return roi_model.data_analysis_history[0]


# -----------------------------------------------------------------------------#
# Main plotting: inferred spikes
# -----------------------------------------------------------------------------#
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
        Plot raw inferred spikes (values > 0)
    normalize : bool
        Normalize spike traces globally using percentiles
    active_only : bool
        Only plot active ROIs
    dec_dff : bool
        Optionally overlay deconvolved ΔF/F traces
    thresholds : bool
        Show spike detection thresholds (only if single ROI selected)
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status-bar XY readout
    ax.format_coord = lambda x, y: ""

    # thresholds only if a single ROI is selected
    thresholds = thresholds if rois and len(rois) == 1 else False

    if run_id is None:
        cali_logger.warning("No run_id provided for inferred spikes plot.")
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # ------------------------ Query DB ------------------------ #
    with Session(engine) as session:
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
        roi_data: list[tuple[ROI, Traces, DataAnalysis]] = session.exec(stmt).all()

    if not roi_data:
        ax.text(
            0.5,
            0.5,
            "No ROI spike data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # ---------------- Global percentiles (for normalization) ---------------- #
    p1 = p2 = 0.0
    if normalize:
        all_values: list[float] = []
        for _, traces, data_analysis in roi_data:
            if data_analysis and traces.inferred_spikes:
                spike_data = traces.inferred_spikes
                if raw:
                    # Raw: all > 0
                    spike_values = [float(s) for s in spike_data if s > 0]
                else:
                    # Thresholded: > threshold
                    the = data_analysis.inferred_spikes_threshold or 0.0
                    spike_values = [float(s) for s in spike_data if s > the]
                all_values.extend(spike_values)

        if all_values:
            p1, p2 = map(float, np.percentile(all_values, [5, 100]))
        else:
            p1, p2 = 0.0, 1.0

    # ------------------------ Plot traces ------------------------ #
    count = 0
    rois_rec_time: list[float] = []
    last_trace: list[float] | None = None

    for roi, traces, data_analysis in roi_data:
        if data_analysis is None or not traces.inferred_spikes:
            continue

        if data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

        # Raw vs thresholded spikes
        if raw:
            spike_data = [float(s) if s > 0 else 0.0 for s in traces.inferred_spikes]
        else:
            the = data_analysis.inferred_spikes_threshold or 0.0
            spike_data = [float(s) if s > the else 0.0 for s in traces.inferred_spikes]

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

        # Optional overlay of deconvolved ΔF/F
        if dec_dff and traces.dec_dff:
            _plot_trace(
                ax,
                str(roi.label_value),
                traces.dec_dff,
                normalize,
                count,
                p1,
                p2,
                thresholds=False,
                spikes_threshold=None,
            )

        last_trace = list(traces.inferred_spikes)
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
        offset = count * 1.1  # vertical offset per ROI
        spike_trace = _normalize_trace_percentile(trace, p1, p2) + offset
        ax.plot(spike_trace, label=f"ROI {roi_key}")
        ax.set_yticks([])
        ax.set_yticklabels([])

        # Threshold line in normalized coordinates
        if thresholds and spikes_threshold is not None and spikes_threshold > 0.0:
            denom = p2 - p1
            if denom > 0:
                the_norm = (spikes_threshold - p1) / denom
                the_norm = float(np.clip(the_norm, 0.0, 1.0) + offset)
                ax.axhline(
                    y=the_norm,
                    color="black",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.6,
                    label=f"Spike threshold (ROI {roi_key} - {spikes_threshold:.4f})",
                )
    else:
        ax.plot(trace, label=f"ROI {roi_key}")
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
    tr = np.array(trace, dtype=float)
    denom = p2 - p1
    if denom == 0:
        return np.zeros_like(tr)
    normalized = (tr - p1) / denom
    return np.clip(normalized, 0, 1)


def _set_graph_title_and_labels(ax: Axes, normalize: bool, raw: bool) -> None:
    """Set axis labels based on the plotted data."""
    title = "Normalized Inferred Spikes" if normalize else "Inferred Spikes"
    title += " (Raw)" if raw else " (Thresholded Spike Data)"
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

    # Average total recording time in seconds
    avg_rec_time = int(np.mean(rois_rec_time))
    total_frames = len(trace) if trace is not None else 1

    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")


def _add_hover_functionality(ax: Axes, widget: _SingleWellGraphWidget) -> None:
    """Add hover functionality using efficient pick events."""
    setup_pick_click(ax, widget, picker_tolerance=3)


# -----------------------------------------------------------------------------#
# Normalized spikes + global bursts
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
    if run_id is None:
        widget.figure.clear()
        ax = widget.figure.add_subplot(111)
        ax.format_coord = lambda x, y: ""
        cali_logger.warning(
            "No run_id provided for inferred spikes normalized with bursts plot."
        )
        ax.text(
            0.5,
            0.5,
            "No analysis run selected.\nPlease select a run from the dropdown.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # ------------- Burst detection (GLOBAL, ignore ROI subset) -------------#
    from cali.plot._single_wells_plots.burst._plot_inferred_spike_burst_activity import (  # noqa: E501
        _detect_population_bursts,
        _get_burst_parameters,
        _get_population_spike_data,
    )

    bursts: list[tuple[int, int]] = []

    # Use global ROI set for burst parameters and population data
    burst_params = _get_burst_parameters(engine, fov_name, rois=None, run_id=run_id)
    if burst_params is not None:
        burst_threshold, min_burst_duration, smoothing_sigma = burst_params

        spike_trains_array, _, _time_axis = _get_population_spike_data(
            engine, fov_name, rois=None, run_id=run_id
        )

        if spike_trains_array is not None:
            population_activity = np.mean(spike_trains_array, axis=0)

            # Smooth before detection
            if smoothing_sigma > 0:
                smoothed_activity = gaussian_filter1d(
                    population_activity, sigma=smoothing_sigma, mode="nearest"
                )
            else:
                smoothed_activity = population_activity

            # Detect bursts (threshold passed as fraction, not %)
            bursts = _detect_population_bursts(
                smoothed_activity, burst_threshold / 100.0, min_burst_duration
            )

    # -------------------- Plot normalized spikes (subset) -------------------#
    # This call will clear the figure and draw normalized traces
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
    if bursts:
        axes = widget.figure.get_axes()
        if axes:
            ax = axes[0]
            _overlay_burst_periods(ax, bursts)
            widget.figure.tight_layout()
            widget.canvas.draw()


def _overlay_burst_periods(ax: Axes, bursts: list[tuple[int, int]]) -> None:
    """Overlay burst periods as shaded regions on the plot."""
    if not bursts:
        return

    for i, (start, end) in enumerate(bursts):
        label = "Network Burst" if i == 0 else ""
        ax.axvspan(start, end, alpha=0.2, color="green", label=label)
