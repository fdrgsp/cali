from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._hover_utils import setup_pick_click_for_raster
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


def _generate_spike_raster_plot(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
    amplitude_colors: bool = False,
    colorbar: bool = False,
) -> None:
    """Generate a spike raster plot using thresholded spike data.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Graph widget to plot on
    engine : Engine
        SQLAlchemy Engine connected to the database
    fov_name : str
        Name of the FOV (e.g., "B5_0000")
    run_id : int
        The run ID to filter by
    rois : list[int] | None
        List of ROI label values to plot. If None, plots all ROIs.
    amplitude_colors : bool
        Whether to color spikes by their amplitude
    colorbar : bool
        Whether to show a colorbar for amplitudes
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    ax.format_coord = lambda x, y: ""

    ax.set_title("Inferred Spikes Raster Plot")

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

        stmt = stmt.order_by(col(ROI.label_value))
        roi_data: list[tuple[ROI, Traces, DataAnalysis]] = session.exec(stmt).all()

    if not roi_data:
        cali_logger.warning("No ROI data found for spike raster plot.")
        _draw_centered_message(
            ax,
            "No ROI data found for this FOV.\nPlease check the selected run and FOV.",
        )
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # ------------------------ Collect events ------------------------ #
    event_data: list[list[int]] = []  # per-ROI list of spike indices
    colors: list = []  # matches event_data shape
    rois_rec_time: list[float] = []
    active_rois: list[int] = []
    sample_trace: list[float] | None = None

    min_amp = float("inf")
    max_amp = float("-inf")

    # First pass: collect spike times, amplitudes, and min/max for normalization
    per_roi_spike_amplitudes: list[list[float]] = []

    for roi, traces, data_analysis in roi_data:
        if data_analysis is None or not traces.inferred_spikes:
            continue

        threshold = data_analysis.inferred_spikes_threshold or 0.0
        inferred = np.asarray(traces.inferred_spikes, dtype=float)

        # Thresholded spikes
        above_the = inferred > threshold
        if not np.any(above_the):
            continue

        spike_times = np.where(above_the)[0].tolist()
        spike_amplitudes = inferred[above_the].tolist()

        if not spike_times:
            continue

        active_rois.append(roi.label_value)
        event_data.append(spike_times)
        per_roi_spike_amplitudes.append(spike_amplitudes)

        if data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

        if sample_trace is None and traces.corrected_trace is not None:
            sample_trace = traces.corrected_trace

        if amplitude_colors and spike_amplitudes:
            min_amp = min(min_amp, min(spike_amplitudes))
            max_amp = max(max_amp, max(spike_amplitudes))

    if not event_data:
        cali_logger.warning(
            "No spike data above threshold for the selected ROIs and run."
        )
        _draw_centered_message(
            ax,
            "No spike data above threshold.\n"
            "Try adjusting thresholds or selecting different ROIs.",
        )
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # ------------------------ Colors ------------------------ #
    if amplitude_colors and np.isfinite(min_amp) and np.isfinite(max_amp):
        vmin, vmax = _compute_amp_norm_bounds(min_amp, max_amp)
        colors = _generate_spike_amplitude_colors(per_roi_spike_amplitudes, vmin, vmax)
    else:
        # Fallback: ALL black
        n_rois = len(event_data)
        colors = ["black"] * n_rois
        amplitude_colors = False  # no valid amp range → plain raster

    # ------------------------ Plot raster ------------------------ #
    ax.eventplot(event_data, colors=colors, linewidth=2)

    ax.set_ylabel("ROIs")
    ax.set_yticks([])
    ax.set_yticklabels([])

    _update_time_axis(ax, rois_rec_time, sample_trace)

    # ------------------------ Colorbar ------------------------ #
    if amplitude_colors and colorbar:
        vmin, vmax = _compute_amp_norm_bounds(min_amp, max_amp)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar = widget.figure.colorbar(
            cm.ScalarMappable(norm=norm, cmap="viridis"),
            ax=ax,
        )
        cbar.set_label("Spike Amplitude")

    widget.figure.tight_layout()
    _add_hover_functionality(ax, widget, active_rois)
    widget.canvas.draw()


def _draw_centered_message(ax: Axes, text: str) -> None:
    """Utility to draw a centered multi-line message and hide axes."""
    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.axis("off")


def _compute_amp_norm_bounds(min_amp: float, max_amp: float) -> tuple[float, float]:
    """Compute robust vmin/vmax for amplitude normalization."""
    if not np.isfinite(min_amp) or not np.isfinite(max_amp):
        return 0.0, 1.0

    # Use a reduced range (e.g. 60% of full) to make mid/high values more visible
    vmax = min_amp + (max_amp - min_amp) * 0.6
    if vmax <= min_amp:
        vmax = max_amp
    if vmax <= min_amp:
        vmax = min_amp + 0.1  # tiny positive range if everything is almost equal

    vmin = min_amp
    return vmin, vmax


def _generate_spike_amplitude_colors(
    per_roi_spike_amplitudes: list[list[float]],
    vmin: float,
    vmax: float,
) -> list:
    """Assign colors based on individual spike amplitudes per ROI.

    Parameters
    ----------
    per_roi_spike_amplitudes : list[list[float]]
        For each ROI (row), the list of spike amplitudes (in temporal order).
        Must be aligned with event_data rows.
    vmin, vmax : float
        Normalization bounds for amplitudes.

    Returns
    -------
    list
        A list of lists of RGBA colors, matching event_data shape.
    """
    norm_amp_color = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps.get_cmap("viridis")

    colors: list[list[tuple[float, float, float, float]]] = []
    for amps in per_roi_spike_amplitudes:
        row_colors = [cmap(norm_amp_color(a)) for a in amps]
        colors.append(row_colors)
    return colors


def _add_hover_functionality(
    ax: Axes, widget: _SingleWellGraphWidget, active_rois: list[int]
) -> None:
    """Add hover functionality using efficient pick events."""
    setup_pick_click_for_raster(ax, widget, active_rois, picker_tolerance=5)


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    """Update the x-axis to show time in seconds if recording time is available."""
    if trace is None or sum(rois_rec_time) <= 0:
        ax.set_xlabel("Frames")
        return

    avg_rec_time = int(np.mean(rois_rec_time))
    total_frames = len(trace) if trace is not None else 1

    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")
