from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.cm as cm
import mplcursors
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from sqlalchemy.orm import selectinload
from sqlmodel import Session, col, create_engine, select

from cali.sqlmodel._model import FOV, ROI

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes

    from cali.gui._graph_widgets import _SingleWellGraphWidget


def _generate_raster_plot(
    widget: _SingleWellGraphWidget,
    db_path: str | Path,
    fov_name: str,
    rois: list[int] | None = None,
    amplitude_colors: bool = False,
    colorbar: bool = False,
) -> None:
    """Generate a raster plot by querying database directly.

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
    amplitude_colors : bool
        Whether to color by amplitude
    colorbar : bool
        Whether to show colorbar
    """
    # clear the figure
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    ax.set_title(
        "Calcium Peaks Raster Plot Colored by Amplitude"
        if amplitude_colors
        else "Calcium Peaks Raster Plot"
    )

    # Query database for ROI data
    engine = create_engine(f"sqlite:///{db_path}", echo=False)

    with Session(engine) as session:
        # Build query to get ROIs for this FOV with eager loading of related data
        stmt = (
            select(ROI)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .options(
                selectinload(ROI.traces),  # type: ignore
                selectinload(ROI.data_analysis),  # type: ignore
            )
        )

        # Filter by specific ROIs if requested
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        # Order by label_value for consistent plotting
        stmt = stmt.order_by(col(ROI.label_value))

        roi_models = session.exec(stmt).all()

    engine.dispose(close=True)

    # initialize required lists and variables
    event_data: list[list[float]] = []
    colors: list = []  # Colors for eventplot (can be strings or tuples)
    rois_rec_time: list[float] = []
    total_frames: int = 0

    # if amplitude colors are used, determine min/max amplitude range
    min_amp, max_amp = (float("inf"), float("-inf")) if amplitude_colors else (0, 0)

    active_rois = []
    # loop over the ROI models and get the peaks and their colors for each ROI
    for roi in roi_models:
        if roi.data_analysis is None or roi.traces is None:
            continue

        if (
            not roi.data_analysis.peaks_dec_dff
            or not roi.data_analysis.peaks_amplitudes_dec_dff
        ):
            continue

        # store the active ROIs
        active_rois.append(roi.label_value)

        # convert the x-axis frames to seconds
        if roi.data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(roi.data_analysis.total_recording_time_sec)

        # assuming all traces have the same number of frames
        if not total_frames and roi.traces.corrected_trace is not None:
            total_frames = len(roi.traces.corrected_trace)

        # store event data
        event_data.append(roi.data_analysis.peaks_dec_dff)

        if amplitude_colors:
            # calculate min and max amplitudes for color normalization
            min_amp = min(min_amp, min(roi.data_analysis.peaks_amplitudes_dec_dff))
            max_amp = max(max_amp, max(roi.data_analysis.peaks_amplitudes_dec_dff))
        else:
            # assign default color if not using amplitude-based coloring
            colors.append(f"C{roi.label_value - 1}")

    # create the color palette for the raster plot
    if amplitude_colors:
        _generate_amplitude_colors(roi_models, min_amp, max_amp, colors)

    # plot the raster plot
    ax.eventplot(event_data, colors=colors)

    # set the axis labels
    ax.set_ylabel("ROIs")
    ax.set_yticklabels([])
    ax.set_yticks([])

    # use any trace to get total number of frames (they should all be the same)
    sample_trace = None
    for roi in roi_models:
        if roi.traces and roi.traces.corrected_trace is not None:
            sample_trace = roi.traces.corrected_trace
            break

    _update_time_axis(ax, rois_rec_time, sample_trace)

    # add the colorbar if amplitude colors are used
    if amplitude_colors and colorbar:
        # Ensure valid normalization range for colorbar
        vmax = max_amp * 0.5
        if vmax <= min_amp:
            vmax = max_amp
        if vmax <= min_amp:
            vmax = min_amp + 0.1

        cbar = widget.figure.colorbar(
            cm.ScalarMappable(norm=Normalize(vmin=min_amp, vmax=vmax), cmap="viridis"),
            ax=ax,
        )
        cbar.set_label("Amplitude")

    widget.figure.tight_layout()
    _add_hover_functionality(ax, widget, active_rois)
    widget.canvas.draw()


def _generate_amplitude_colors(
    roi_models: Sequence[ROI],
    min_amp: float,
    max_amp: float,
    colors: list,
) -> None:
    """Assign colors based on amplitude for raster plot."""
    # Ensure valid normalization range
    vmax = max_amp * 0.5
    if vmax <= min_amp:
        # If max_amp * 0.5 is not greater than min_amp, use full range
        vmax = max_amp
    # Ensure minimum range for color mapping
    if vmax <= min_amp:
        # If still equal, add small offset to create range
        vmax = min_amp + 0.1

    norm_amp_color = Normalize(vmin=min_amp, vmax=vmax)
    cmap = colormaps.get_cmap("viridis")
    for roi in roi_models:
        if roi.data_analysis and roi.data_analysis.peaks_amplitudes_dec_dff:
            # Use average amplitude for ROI color
            avg_amp = np.mean(roi.data_analysis.peaks_amplitudes_dec_dff)
            color = cmap(norm_amp_color(avg_amp))
            colors.append(color)


def _add_hover_functionality(
    ax: Axes, widget: _SingleWellGraphWidget, active_rois: list[int]
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        # Get the label of the artist
        label = sel.artist.get_label()

        # Only show hover for valid ROI elements
        if label and "ROI" in label and not label.startswith("_"):
            sel.annotation.set(text=label, fontsize=8, color="black")
            roi_parts = label.split(" ")
            if len(roi_parts) > 1 and roi_parts[1].isdigit():
                widget.roiSelected.emit(roi_parts[1])
        else:
            # For raster plots, map the position to an ROI
            if hasattr(sel, "target") and len(active_rois) > 0:
                try:
                    y_pos = int(sel.target[1])  # Get y-coordinate (ROI index)
                    if 0 <= y_pos < len(active_rois):
                        roi_id = active_rois[y_pos]
                        hover_text = f"ROI {roi_id}"
                        sel.annotation.set(text=hover_text, fontsize=8, color="black")
                        widget.roiSelected.emit(str(roi_id))
                        return
                except (ValueError, AttributeError, IndexError):
                    pass

            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)


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
