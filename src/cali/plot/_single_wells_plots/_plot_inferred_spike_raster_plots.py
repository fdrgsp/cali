from __future__ import annotations

import contextlib
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

from cali.logger import cali_logger


def _generate_spike_raster_plot(
    widget: _SingleWellGraphWidget,
    db_path: str | Path,
    fov_name: str,
    rois: list[int] | None = None,
    amplitude_colors: bool = False,
    colorbar: bool = False,
) -> None:
    """Generate a spike raster plot using thresholded spike data
    by querying database directly.

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

    ax.set_title("Inferred Spikes Raster Plot")

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
    event_data: list[list[int]] = []
    colors: list = []  # Colors for eventplot (can be strings or tuples)
    rois_rec_time: list[float] = []
    total_frames: int = 0

    # if amplitude colors are used, determine min/max amplitude range
    min_amp, max_amp = (float("inf"), float("-inf")) if amplitude_colors else (0, 0)

    active_rois = []
    # loop over the ROI models and get the spike events for each ROI
    for roi in roi_models:
        if roi.data_analysis is None or not roi.data_analysis.inferred_spikes:
            continue

        # Get thresholded spikes (values above threshold)
        threshold = roi.data_analysis.inferred_spikes_threshold or 0
        thresholded_spikes = [
            s if s > threshold else 0 for s in roi.data_analysis.inferred_spikes
        ]

        if not any(s > 0 for s in thresholded_spikes):
            continue

        # Find spike event times (indices where spike values are above threshold)
        spike_times = []
        spike_amplitudes = []

        for i, spike_val in enumerate(thresholded_spikes):
            if spike_val > 0:  # Above threshold
                spike_times.append(i)
                spike_amplitudes.append(spike_val)

        if not spike_times:
            continue

        # store the active ROIs
        active_rois.append(roi.label_value)

        # convert the x-axis frames to seconds
        if roi.data_analysis.total_recording_time_sec is not None:
            rois_rec_time.append(roi.data_analysis.total_recording_time_sec)

        # assuming all traces have the same number of frames
        if (
            not total_frames
            and roi.traces is not None
            and roi.traces.corrected_trace is not None
        ):
            total_frames = len(roi.traces.corrected_trace)

        # store event data (spike times)
        event_data.append(spike_times)

        if amplitude_colors and spike_amplitudes:
            # calculate min and max amplitudes for color normalization
            min_amp = min(min_amp, min(spike_amplitudes))
            max_amp = max(max_amp, max(spike_amplitudes))
        else:
            # assign default color if not using amplitude-based coloring
            colors.append(f"C{roi.label_value - 1}")

    if not event_data:
        cali_logger.warning(
            "No spike data available for the selected ROIs. "
            "Please check the data or ROI selection."
        )
        return

    # create the color palette for the raster plot
    if amplitude_colors:
        _generate_spike_amplitude_colors(roi_models, min_amp, max_amp, colors)

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
        # Use the same logic as in color generation
        vmax = min_amp + (max_amp - min_amp) * 0.6  # Use 50% of the range
        cbar = widget.figure.colorbar(
            cm.ScalarMappable(norm=Normalize(vmin=min_amp, vmax=vmax), cmap="viridis"),
            ax=ax,
        )
        cbar.set_label("Spike Amplitude")

    widget.figure.tight_layout()
    _add_hover_functionality(ax, widget, active_rois)
    widget.canvas.draw()


def _generate_spike_amplitude_colors(
    roi_models: Sequence[ROI],
    min_amp: float,
    max_amp: float,
    colors: list,
) -> None:
    """Assign colors based on individual spike amplitudes for raster plot."""
    # Always use a reduced range to make yellow colors more visible
    # Use the midpoint between min and max as vmax for better color distribution
    vmax = min_amp + (max_amp - min_amp) * 0.5  # Use 50% of the range
    norm_amp_color = Normalize(vmin=min_amp, vmax=vmax)
    cmap = colormaps.get_cmap("viridis")

    for roi in roi_models:
        if roi.data_analysis and roi.data_analysis.inferred_spikes:
            threshold = roi.data_analysis.inferred_spikes_threshold or 0
            thresholded_spikes = [
                s if s > threshold else 0 for s in roi.data_analysis.inferred_spikes
            ]

            # Get individual spike amplitudes and create colors for each spike event
            spike_colors = []
            for spike_val in thresholded_spikes:
                if spike_val > 0:  # This is a spike event
                    # Color each spike based on its individual amplitude
                    color = cmap(norm_amp_color(spike_val))
                    spike_colors.append(color)

            if spike_colors:
                colors.append(spike_colors)


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
            if hasattr(sel, "target") and active_rois:
                with contextlib.suppress(ValueError, AttributeError, IndexError):
                    y_pos = int(sel.target[1])  # Get y-coordinate (ROI index)
                    if 0 <= y_pos < len(active_rois):
                        roi_id = active_rois[y_pos]
                        hover_text = f"ROI {roi_id}"
                        sel.annotation.set(text=hover_text, fontsize=8, color="black")
                        widget.roiSelected.emit(str(roi_id))
                        return
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)


def _update_time_axis(
    ax: Axes, rois_rec_time: list[float], trace: list[float] | None
) -> None:
    """Update the x-axis to show time in seconds if recording time is available."""
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
