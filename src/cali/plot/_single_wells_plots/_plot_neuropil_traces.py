"""Plot raw traces with neuropil correction visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cmap
import mplcursors
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from cali.gui._graph_widgets import _SingleWellGraphWidget
    from cali.sqlmodel._util import ROIData


def _plot_neuropil_traces(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot all raw and neuropil traces together on widget canvas.

    Raw traces and neuropil traces are plotted on the same axes,
    allowing the filtering logic to isolate specific ROI pairs.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        The widget containing the matplotlib figure and canvas
    data : dict[str, ROIData]
        Dictionary of ROI data
    rois : list[int] | None
        List of specific ROI IDs to plot, or None for all
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    # Filter data by selected ROIs if specified
    filtered_data = data
    if rois is not None:
        filtered_data = {k: v for k, v in data.items() if int(k) in rois}

    # Filter ROIs that have both raw_trace and neuropil traces
    valid_rois = {
        roi_id: roi_data
        for roi_id, roi_data in filtered_data.items()
        if roi_data.raw_trace is not None
        and roi_data.neuropil_trace is not None
        and roi_data.corrected_trace is not None
    }

    if not valid_rois:
        # No valid data to plot
        ax.text(
            0.5,
            0.5,
            "No neuropil traces available.\nNeuropil correction may not be enabled.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Generate colors using glasbey colormap
    n_rois = len(valid_rois)
    glasbey_cmap = cmap.Colormap("glasbey").to_matplotlib()
    # Skip the first color (often black/dark) and use from 0.05 to skip dark
    color_indices = np.linspace(0.05, 1, n_rois)
    colors = glasbey_cmap(color_indices)

    # Store lines for hover functionality
    lines = []
    roi_ids = []
    rois_rec_time: list[float] = []
    last_trace: np.ndarray | None = None

    # Plot all traces on the same axes
    for idx, (roi_id, roi_data) in enumerate(valid_rois.items()):
        color = colors[idx]

        raw_trace = np.array(roi_data.raw_trace)
        neuropil_trace = np.array(roi_data.neuropil_trace)
        corrected_trace = np.array(roi_data.corrected_trace)
        frames = np.arange(len(raw_trace))

        # Collect recording time if available
        if (ttime := roi_data.total_recording_time_sec) is not None:
            rois_rec_time.append(ttime)

        # Keep track of last trace for time axis calculation
        last_trace = raw_trace

        # Plot raw trace (solid line)
        line_raw = ax.plot(
            frames,
            raw_trace,
            label=f"Raw ROI {roi_id}",
            color=color,
            linewidth=1,
            linestyle="--",
        )[0]
        lines.append(line_raw)
        roi_ids.append(roi_id)

        # Plot neuropil trace (dashed line, same color)
        line_neuropil = ax.plot(
            frames,
            neuropil_trace,
            label=f"Neuropil ROI {roi_id}",
            color=color,
            linewidth=1,
            linestyle=":",
        )[0]
        lines.append(line_neuropil)
        roi_ids.append(roi_id)

        # Plot corrected trace (dotted line, same color)
        line_corrected = ax.plot(
            frames,
            corrected_trace,
            label=f"Corrected ROI {roi_id}",
            color=color,
            linewidth=1,
            linestyle="-",
        )[0]
        lines.append(line_corrected)
        roi_ids.append(roi_id)

    # Formatting
    ax.set_ylabel("Fluorescence (a.u.)", fontsize=11)
    ax.set_title("Raw, Neuropil, and Corrected Traces", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add a custom legend explaining the line styles
    from matplotlib.lines import Line2D

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

    # Update time axis if recording time is available
    _update_time_axis(ax, rois_rec_time, last_trace)

    # Add hover functionality
    _add_hover_functionality(ax, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


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
            roi = cast("str", label.split(" ")[-1])
            if roi.isdigit():
                widget.roiSelected.emit(roi)
        else:
            # Hide the annotation for non-ROI elements
            sel.annotation.set_visible(False)


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
