from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import mplcursors
import numpy as np
import tifffile
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from skimage.measure import find_contours

from cali._constants import MWCM
from cali.plot._util import (
    _get_spikes_over_threshold,
    equation_from_str,
    get_stimulated_amplitudes_from_roi_data,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sqlalchemy.engine import Engine

    from cali.gui._graph_widgets import _SingleWellGraphWidget
    from cali.sqlmodel._util import ROIData

from cali.logger import cali_logger

DEFAULT_COLOR = "gray"
STIMULATED_COLOR = "green"
NON_STIMULATED_COLOR = "magenta"
P1 = 5
P2 = 100


def _plot_evoked_experiment_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    stimulated_area: bool = False,
    with_rois: bool = False,
    stimulated: bool = False,
    with_peaks: bool = False,
) -> None:
    """Plot evoked experiment data.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        Widget to plot on
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        List of ROI indices to include, None for all
    run_id : int | None
        The run ID to filter by, None for latest
    stimulated_area : bool
        Whether to show stimulated area
    with_rois : bool
        Whether to show ROIs
    stimulated : bool
        Whether to show stimulated peaks
    with_peaks : bool
        Whether to show peaks
    """
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    ax.text(
        0.5,
        0.5,
        "Evoked Experiment Plots\n\n"
        "These visualizations require evoked experiment data\n"
        "that is not yet fully integrated with the new database schema.\n\n"
        "Please use the legacy viewer or contact support.",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.axis("off")

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_stim_or_not_stim_peaks_amplitude(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    stimulated: bool = False,
) -> None:
    """Visualize stimulated peak amplitudes per ROI per stimulation parameters."""
    # TODO: Integrate with new database schema
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    ax.text(
        0.5,
        0.5,
        "Evoked Experiment Amplitude Analysis\n\n"
        "This feature requires integration with the new\n"
        "three-stage pipeline (Detection → Extraction → Analysis).\n\n"
        "Coming soon!",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.axis("off")
    widget.figure.tight_layout()
    widget.canvas.draw()


def extract_leading_number(key: str) -> float:
    """Extract leading number from key (before '_'), stripping units if present."""
    if match := re.match(r"(\d+(?:\.\d+)?)", key.split("_")[0]):
        return float(match[1])
    raise ValueError(f"Could not extract a valid number from key: {key}")


def _add_hover_to_stimulated_amp_plot(
    widget: _SingleWellGraphWidget,
    artists: list,
    metadata: list[tuple[list[int], list[float]]],
) -> None:
    """Add hover tooltips to amplitude scatter plot points."""
    cursor = mplcursors.cursor(artists, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore
    def on_add(sel: mplcursors.Selection) -> None:
        artist = sel.artist
        index = sel.index
        group_index = artists.index(artist)
        rois_, amps = metadata[group_index]
        roi = rois_[index]
        amp_val = amps[index]

        sel.annotation.set(
            text=f"ROI {roi}\nAmp: {amp_val:.3f}", fontsize=8, color="black"
        )
        sel.annotation.arrow_patch.set_alpha(0.5)

        widget.roiSelected.emit(str(roi))


def _visualize_stimulated_area(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    with_rois: bool = False,
    stimulated_area: bool = False,
) -> None:
    """Visualize Stimulated area - STUB for future implementation."""
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)

    ax.text(
        0.5,
        0.5,
        "Stimulated Area Visualization\n\n"
        "This requires evoked experiment metadata\n"
        "not yet fully integrated with the new schema.\n\n"
        "Use the Detection Viewer for ROI visualization.",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.axis("off")

    widget.figure.tight_layout()
    widget.canvas.draw()


def _plot_stimulated_rois(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None,
    stim_mask: np.ndarray,
    with_stimulated_area: bool,
) -> None:
    """Plot the ROIs with stimulated and non-stimulated areas."""
    # get the labels file path
    labels_image_path = widget._plate_viewer.pv_labels_path
    if labels_image_path is None:
        return

    stim, non_stim = _group_rois(data, rois)

    # open label image
    r = str(rois[0]) if rois is not None else "1"
    label_name = f"{data[r].well_fov_position}.tif"
    if not label_name:
        return
    # todo: maybe get it form ROIData.mask_coord_and_shape
    labels = tifffile.imread(Path(labels_image_path) / label_name)

    # create a color mapping for the labels
    color_mapping = _generate_color_mapping(labels, stim, non_stim)

    # plot the labels image with the color mapping
    unique_labels = np.unique(labels)
    colors = [color_mapping.get(lbl, DEFAULT_COLOR) for lbl in unique_labels]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        boundaries=np.append(unique_labels, unique_labels[-1] + 1),
        ncolors=len(colors),
    )

    if with_stimulated_area:
        stim_area_contours = find_contours(stim_mask.astype(float), level=0.5)
        for contour in stim_area_contours:
            ax.plot(contour[:, 1], contour[:, 0], color="yellow", linewidth=1)
    ax.imshow(labels, cmap=cmap, norm=norm)

    _add_legend(ax)
    _add_hover_functionality_plot_stim_roi(ax, widget, labels, stim_mask)


def _group_rois(data: dict, rois: list[int] | None) -> tuple[list[int], list[int]]:
    """To group the ROIs based on stimulated state."""
    stimulated_rois: list[int] = []
    non_stimulated_rois: list[int] = []

    for roi_key in data:
        if rois is not None and int(roi_key) not in rois:
            continue

        roi_data = cast("ROIData", data[roi_key])

        if roi_data.stimulated:
            stimulated_rois.append(int(roi_key))
        else:
            non_stimulated_rois.append(int(roi_key))

    return stimulated_rois, non_stimulated_rois


def _generate_color_mapping(
    labels: np.ndarray, stim: list[int], non_stim: list[int]
) -> dict[int, str]:
    """Generate a color mapping for the labels."""
    color_mapping = {0: "black", 1: "white"}  # 0: background, 1: stimulated area
    labels_range = np.unique(labels[labels != 0])
    for roi in labels_range:
        if roi in stim:
            color_mapping[roi] = STIMULATED_COLOR
        elif roi in non_stim:
            color_mapping[roi] = NON_STIMULATED_COLOR
        else:
            color_mapping[roi] = DEFAULT_COLOR
    return color_mapping


def _add_legend(ax: Axes) -> None:
    """Add legend to the plot."""
    legend_patches = [
        Patch(color="green", label="Stimulated ROIs"),
        Patch(color="magenta", label="Non-Stimulated ROIs"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),  # moves it above the plot (x, y)
        ncol=2,  # single row
        frameon=True,
        fontsize="small",
        edgecolor="black",
    )


def _add_hover_functionality_plot_stim_roi(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    labels: np.ndarray,
    stim_mask: np.ndarray,
) -> None:
    """Add hover functionality using mplcursors."""
    cursor = mplcursors.cursor(ax, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_add(sel: mplcursors.Selection) -> None:
        roi_val = None
        x, y = int(sel.target[0]), int(sel.target[1])
        if 0 <= y < stim_mask.shape[0] and 0 <= x < stim_mask.shape[1]:
            roi_val = str(labels[y, x]) if labels[y, x] > 0 else None
        if roi_val and "ROI" in roi_val:
            sel.annotation.set(text=f"ROI {roi_val}", fontsize=8, color="yellow")
            sel.annotation.arrow_patch.set_color("yellow")
            sel.annotation.arrow_patch.set_alpha(1)  # arrow is visible
        else:
            sel.annotation.set_visible(False)  # hide annotation
            sel.annotation.arrow_patch.set_alpha(0)  # hide arrow
        if roi_val and roi_val.isdigit():
            widget.roiSelected.emit(roi_val)


def _plot_stimulated_vs_non_stimulated_roi_amp(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    with_peaks: bool = False,
) -> None:
    """Plot dec ΔF/F traces with global percentile normalization (5th-100th)."""
    # TODO: Integrate with new database schema
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    ax.text(
        0.5,
        0.5,
        "Stimulated vs Non-Stimulated Traces\n\n"
        "This feature requires integration with the new\n"
        "three-stage pipeline (Detection → Extraction → Analysis).\n\n"
        "Coming soon!",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.axis("off")
    widget.figure.tight_layout()
    widget.canvas.draw()


def _normalize_trace_percentile(
    trace: list[float], p1: float, p2: float
) -> list[float]:
    """Normalize a trace using the global 5th and 100th percentiles."""
    tr = np.array(trace)
    denom = p2 - p1
    if denom == 0:
        return cast("list[float]", np.zeros_like(tr).tolist())
    normalized = (tr - p1) / denom
    normalized = np.clip(normalized, 0, 1)  # ensure values in [0, 1]
    return cast("list[float]", normalized.tolist())


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


def _add_hover_functionality_stim_vs_non_stim(
    ax: Axes, widget: _SingleWellGraphWidget
) -> None:
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


def _plot_stimulated_vs_non_stimulated_spike_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot thresholded spike traces: green=stimulated, magenta=non-stimulated."""
    # TODO: Integrate with new database schema
    widget.figure.clear()
    ax = widget.figure.add_subplot(111)
    ax.text(
        0.5,
        0.5,
        "Stimulated vs Non-Stimulated Spike Traces\n\n"
        "This feature requires integration with the new\n"
        "three-stage pipeline (Detection → Extraction → Analysis).\n\n"
        "Coming soon!",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax.transAxes,
    )
    ax.axis("off")
    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality_spike_traces(
    ax: Axes, widget: _SingleWellGraphWidget, active_rois: list[int]
) -> None:
    """Add hover functionality using mplcursors for spike traces."""
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


def _update_time_axis_spike_traces(
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
