"""Utilities for efficient hover/pick functionality in matplotlib plots.

This module provides optimized event-based picking instead of mplcursors
for better performance with hundreds of traces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from cali.gui._graph_widgets import _SingleWellGraphWidget


def setup_pick_click(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    picker_tolerance: int = 3,
) -> None:
    """Set up click-based interaction for ROI traces.

    Uses pick events for click detection.
    Shows ROI info in the status bar.

    Parameters
    ----------
    ax : Axes
        The axes containing the plot
    widget : _SingleWellGraphWidget
        Graph widget that will receive ROI selection signals
    picker_tolerance : int, optional
        Picking tolerance in pixels (default: 3). Lower values are faster
        but require more precise mouse positioning.
    """
    # Enable picking on all artists with ROI labels
    for artist in ax.get_children():
        if hasattr(artist, "get_label"):
            label = artist.get_label()
            if isinstance(label, str) and "ROI" in label and not label.startswith("_"):
                artist.set_picker(picker_tolerance)

    def on_pick(event: Any) -> None:
        """Handle pick events on lines."""
        artist = event.artist
        label = artist.get_label()

        label_str = label if isinstance(label, str) else None
        if label_str and "ROI" in label_str and not label_str.startswith("_"):
            # Extract ROI number from label (e.g., "ROI 42" -> "42")
            parts = label_str.split()
            roi_idx = next((i for i, p in enumerate(parts) if p == "ROI"), None)
            if roi_idx is not None and roi_idx + 1 < len(parts):
                roi = parts[roi_idx + 1]
                if roi.isdigit():
                    widget.roiSelected.emit(roi)
                    # Update status bar
                    _update_status_bar(ax, label_str, event)

    def on_motion(event: Any) -> None:
        """Clear status bar when mouse moves."""
        _set_format_coord(ax, lambda x, y: "")
        ax.figure.canvas.draw_idle()

    # Connect events
    ax.figure.canvas.mpl_connect("pick_event", on_pick)
    ax.figure.canvas.mpl_connect("motion_notify_event", on_motion)

    # Set default format_coord behavior
    _set_format_coord(ax, lambda x, y: "")


def _update_status_bar(
    ax: Axes,
    label: str,
    event: Any,
) -> None:
    """Update matplotlib status bar with ROI info."""

    def format_coord(x: float, y: float) -> str:
        return label

    _set_format_coord(ax, format_coord)
    # Force immediate toolbar update by calling the format_coord and setting the text
    if hasattr(ax.figure.canvas, "toolbar") and ax.figure.canvas.toolbar is not None:
        toolbar = ax.figure.canvas.toolbar
        # Update toolbar message immediately
        if hasattr(toolbar, "set_message"):
            toolbar.set_message(label)


def _set_format_coord(ax: Axes, func: Any) -> None:
    """Set format_coord on axes (type-safe wrapper)."""
    ax.format_coord = func


def setup_pick_click_for_raster(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    active_rois: list[int],
    picker_tolerance: int = 5,
) -> None:
    """Set up click-based interaction for raster plots.

    Raster plots need special handling since y-position maps to ROI index.

    Parameters
    ----------
    ax : Axes
        The axes containing the raster plot
    widget : _SingleWellGraphWidget
        Graph widget that will receive ROI selection signals
    active_rois : list[int]
        List of active ROI IDs in plot order
    picker_tolerance : int, optional
        Picking tolerance in pixels (default: 5)
    """
    # Enable picking on all plot elements
    for artist in ax.get_children():
        if hasattr(artist, "set_picker"):
            artist.set_picker(picker_tolerance)

    def on_pick(event: Any) -> None:
        """Handle pick events for raster plots."""
        artist = event.artist
        label = artist.get_label()

        # Try to get ROI from label first
        label_str = label if isinstance(label, str) else None
        if label_str and "ROI" in label_str and not label_str.startswith("_"):
            parts = label_str.split()
            if len(parts) > 1 and parts[1].isdigit():
                roi = parts[1]
                _update_status_and_emit(ax, widget, roi)
                return

        # For raster plots, map y-position to ROI
        if hasattr(event, "mouseevent") and active_rois:
            try:
                ydata = event.mouseevent.ydata
                if ydata is not None:
                    y_pos = int(ydata)
                    if 0 <= y_pos < len(active_rois):
                        roi_id = active_rois[y_pos]
                        _update_status_and_emit(ax, widget, str(roi_id))
            except (ValueError, AttributeError, TypeError):
                pass

    def on_motion(event: Any) -> None:
        """Clear status bar when mouse moves."""
        _set_format_coord(ax, lambda x, y: "")
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("pick_event", on_pick)
    ax.figure.canvas.mpl_connect("motion_notify_event", on_motion)
    _set_format_coord(ax, lambda x, y: "")


def _update_status_and_emit(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    roi: str,
) -> None:
    """Helper to update status bar and emit ROI selection signal."""
    widget.roiSelected.emit(roi)
    _set_format_coord(ax, lambda x, y: f"ROI {roi}")
    # Update status bar immediately using toolbar
    if hasattr(ax.figure.canvas, "toolbar"):
        toolbar = ax.figure.canvas.toolbar
        if toolbar and hasattr(toolbar, "set_message"):
            toolbar.set_message(f"ROI {roi}")
    ax.figure.canvas.draw_idle()


def setup_pick_click_for_heatmap(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    rois: list[int],
    data_matrix: object,  # np.ndarray but avoid import
) -> None:
    """Set up click-based interaction for heatmap/correlation matrix plots.

    Parameters
    ----------
    ax : Axes
        The axes containing the heatmap
    widget : _SingleWellGraphWidget
        Graph widget that will receive ROI selection signals
    rois : list[int]
        List of ROI IDs (database IDs)
    data_matrix : array-like
        The data matrix being displayed
    """

    def on_click(event: Any) -> None:
        """Handle click events to show info and emit ROI selection signal."""
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                try:
                    x_idx = max(0, min(int(np.round(x)), len(rois) - 1))
                    y_idx = max(0, min(int(np.round(y)), len(rois) - 1))
                    if (
                        0 <= x_idx < len(rois)
                        and 0 <= y_idx < len(rois)
                        and hasattr(data_matrix, "shape")
                    ):
                        value = data_matrix[y_idx, x_idx]  # type: ignore[index]
                        roi_x, roi_y = rois[x_idx], rois[y_idx]
                        status = f"ROI {roi_y} ↔ ROI {roi_x}: {value:.3f}"
                        _set_format_coord(ax, lambda x, y: status)
                        # Update status bar immediately using toolbar
                        if hasattr(ax.figure.canvas, "toolbar"):
                            toolbar = ax.figure.canvas.toolbar
                            if toolbar and hasattr(toolbar, "set_message"):
                                toolbar.set_message(status)
                        ax.figure.canvas.draw_idle()
                        # Emit signal for the ROI pair
                        widget.roiSelected.emit([str(roi_x), str(roi_y)])
                except (IndexError, ValueError, AttributeError):
                    pass

    def on_motion(event: Any) -> None:
        """Clear status bar when mouse moves."""
        _set_format_coord(ax, lambda x, y: "")
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("button_press_event", on_click)
    ax.figure.canvas.mpl_connect("motion_notify_event", on_motion)
    _set_format_coord(ax, lambda x, y: "")
