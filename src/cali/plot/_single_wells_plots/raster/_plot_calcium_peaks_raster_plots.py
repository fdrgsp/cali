from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from sqlmodel import Session, col, select

from cali.plot._hover_utils import setup_pick_click_for_raster
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget


def _generate_raster_plot(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
    amplitude_colors: bool = False,
    colorbar: bool = False,
) -> None:
    """Generate a raster plot by querying database directly.

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
        Whether to color lines by peak amplitude
    colorbar : bool
        Whether to show a colorbar for amplitude coloring
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    # Disable status bar x/y display
    ax.format_coord = lambda x, y: ""

    ax.set_title(
        "Calcium Peaks Raster Plot Colored by Amplitude"
        if amplitude_colors
        else "Calcium Peaks Raster Plot"
    )

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
        ax.text(
            0.5,
            0.5,
            "No ROI data found for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # ------------------------ Collect events & metadata ------------------------ #
    event_data: list[list[float]] = []
    rois_rec_time: list[float] = []
    active_rois: list[int] = []
    sample_trace: list[float] | None = None

    min_amp = float("inf")
    max_amp = float("-inf")

    for roi, traces, data_analysis in roi_data:
        if (
            not data_analysis.peaks_dec_dff
            or not data_analysis.peaks_amplitudes_dec_dff
        ):
            continue

        # Active ROIs in raster order
        active_rois.append(roi.label_value)

        # Recording time (for time axis)
        if data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(data_analysis.total_recording_time_sec)

        # Use first valid trace as sample for frame count
        if sample_trace is None and traces.corrected_trace is not None:
            sample_trace = traces.corrected_trace

        # Peaks indices per ROI (frames)
        event_data.append(list(data_analysis.peaks_dec_dff))

        # Track amp range for coloring
        if amplitude_colors:
            amps = data_analysis.peaks_amplitudes_dec_dff
            min_amp = min(min_amp, min(amps))
            max_amp = max(max_amp, max(amps))

    if not event_data:
        ax.text(
            0.5,
            0.5,
            "No peak data available for this FOV.",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # ------------------------ Colors ------------------------ #
    colors: list = []

    if amplitude_colors and np.isfinite(min_amp) and np.isfinite(max_amp):
        # Compute normalization bounds shared by line colors and colorbar
        vmin, vmax = _compute_amp_norm_bounds(min_amp, max_amp)
        colors = _generate_amplitude_colors(roi_data, vmin, vmax)
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
        cbar.set_label("Amplitude")

    widget.figure.tight_layout()

    _add_hover_functionality(ax, widget, active_rois)
    widget.canvas.draw()


def _compute_amp_norm_bounds(min_amp: float, max_amp: float) -> tuple[float, float]:
    """Compute robust vmin/vmax for amplitude normalization."""
    if not np.isfinite(min_amp) or not np.isfinite(max_amp):
        return 0.0, 1.0

    # Try to compress the top end a bit (to avoid single huge outliers dominating)
    vmax = max_amp * 0.5
    if vmax <= min_amp:
        vmax = max_amp
    if vmax <= min_amp:
        vmax = min_amp + 0.1  # tiny range if everything is almost equal

    vmin = min_amp
    return vmin, vmax


def _generate_amplitude_colors(
    roi_data: list[tuple[ROI, Traces, DataAnalysis]],
    vmin: float,
    vmax: float,
) -> list:
    """Assign one color per ROI based on average amplitude."""
    norm_amp_color = Normalize(vmin=vmin, vmax=vmax)
    cmap = colormaps.get_cmap("viridis")

    colors: list = []
    for _, _traces, data_analysis in roi_data:
        if not (data_analysis and data_analysis.peaks_amplitudes_dec_dff):
            continue

        avg_amp = float(np.mean(data_analysis.peaks_amplitudes_dec_dff))
        colors.append(cmap(norm_amp_color(avg_amp)))

    # Note: this assumes roi_data was filtered in the same way as event_data
    # (only ROIs with valid peaks), which is true in _generate_raster_plot.
    return colors


def _add_hover_functionality(
    ax: Axes, widget: _SingleWellGraphWidget, active_rois: list[int]
) -> None:
    """Add hover functionality using efficient pick events."""
    setup_pick_click_for_raster(ax, widget, active_rois, picker_tolerance=5)


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    """Set x-axis as time (s) if total recording time is available, else frames."""
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
