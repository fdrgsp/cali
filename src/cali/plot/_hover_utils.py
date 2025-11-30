"""Utilities for efficient hover/pick functionality in matplotlib plots.

This module provides optimized event-based picking instead of mplcursors
for better performance with hundreds of traces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from cali.gui._graph_widgets import _SingleWellGraphWidget


def setup_pick_hover(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    picker_tolerance: int = 3,
    *,
    show_coordinates: bool = False,
) -> None:
    """Set up efficient pick-based hover for ROI traces.

    Uses native matplotlib pick events instead of mplcursors for better
    performance with hundreds of traces. Shows ROI info in the status bar
    instead of floating annotations.

    Parameters
    ----------
    ax : Axes
        The axes containing the plot
    widget : _SingleWellGraphWidget
        Graph widget that will receive ROI selection signals
    picker_tolerance : int, optional
        Picking tolerance in pixels (default: 3). Lower values are faster
        but require more precise mouse positioning.
    show_coordinates : bool, optional
        Whether to show x,y coordinates in status bar (default: False).
        If False, only ROI info is shown.
    """
    # Enable picking on all ROI lines
    for line in ax.get_lines():
        label = line.get_label()
        if label and "ROI" in label and not label.startswith("_"):
            line.set_picker(picker_tolerance)

    # Track currently hovered line for status bar updates
    current_hover: dict[str, str | None] = {"roi": None}

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
                    current_hover["roi"] = roi
                    widget.roiSelected.emit(roi)
                    # Update status bar
                    _update_status_bar(ax, label_str, show_coordinates, event)

    def on_motion(event: Any) -> None:
        """Clear status bar when mouse moves away from lines."""
        if event.inaxes != ax:
            current_hover["roi"] = None
            _set_format_coord(ax, _default_format_coord(show_coordinates))
            ax.figure.canvas.draw_idle()

    # Connect events
    ax.figure.canvas.mpl_connect("pick_event", on_pick)
    ax.figure.canvas.mpl_connect("motion_notify_event", on_motion)

    # Set default format_coord behavior
    _set_format_coord(ax, _default_format_coord(show_coordinates))


def _update_status_bar(
    ax: Axes,
    label: str,
    show_coordinates: bool,
    event: Any,
) -> None:
    """Update matplotlib status bar with ROI info."""

    def format_coord(x: float, y: float) -> str:
        if show_coordinates:
            return f"{label} | x={x:.2f}, y={y:.2f}"
        return label

    _set_format_coord(ax, format_coord)
    # Force immediate toolbar update by calling the format_coord and setting the text
    if hasattr(ax.figure.canvas, "toolbar") and ax.figure.canvas.toolbar is not None:
        toolbar = ax.figure.canvas.toolbar
        # Update toolbar message immediately
        if hasattr(toolbar, "set_message"):
            toolbar.set_message(label)


def _default_format_coord(show_coordinates: bool) -> Any:
    """Return default coordinate formatter."""
    if show_coordinates:
        return lambda x, y: f"x={x:.2f}, y={y:.2f}"
    return lambda x, y: ""


def _set_format_coord(ax: Axes, func: Any) -> None:
    """Set format_coord on axes (type-safe wrapper)."""
    ax.format_coord = func


def setup_pick_hover_for_raster(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    active_rois: list[int],
    picker_tolerance: int = 5,
) -> None:
    """Set up pick-based hover for raster plots.

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
                widget.roiSelected.emit(roi)
                _set_format_coord(ax, lambda x, y: f"ROI {roi}")
                ax.figure.canvas.draw_idle()
                return

        # For raster plots, map y-position to ROI
        if hasattr(event, "mouseevent") and active_rois:
            try:
                ydata = event.mouseevent.ydata
                if ydata is not None:
                    y_pos = int(ydata)
                    if 0 <= y_pos < len(active_rois):
                        roi_id = active_rois[y_pos]
                        widget.roiSelected.emit(str(roi_id))
                        _set_format_coord(ax, lambda x, y: f"ROI {roi_id}")
                        ax.figure.canvas.draw_idle()
            except (ValueError, AttributeError, TypeError):
                pass

    def on_motion(event: Any) -> None:
        """Clear status bar when mouse moves away."""
        if event.inaxes != ax:
            _set_format_coord(ax, lambda x, y: "")
            ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("pick_event", on_pick)
    ax.figure.canvas.mpl_connect("motion_notify_event", on_motion)
    _set_format_coord(ax, lambda x, y: "")


def setup_pick_hover_for_heatmap(
    ax: Axes,
    widget: _SingleWellGraphWidget,
    rois: list[str],
    data_matrix: object,  # np.ndarray but avoid import
) -> None:
    """Set up pick-based hover for heatmap/correlation matrix plots.

    Parameters
    ----------
    ax : Axes
        The axes containing the heatmap
    widget : _SingleWellGraphWidget
        Graph widget that will receive ROI selection signals
    rois : list[str]
        List of ROI labels for x and y axes
    data_matrix : array-like
        The data matrix being displayed
    """
    import numpy as np

    # Enable picking on the image
    for image in ax.get_images():
        image.set_picker(True)

    # Track hover state for click detection
    hover_state: dict[str, Any] = {"roi_x": None, "roi_y": None}

    def on_pick(event: Any) -> None:
        """Handle pick events on heatmap."""
        if hasattr(event, "mouseevent"):
            x, y = event.mouseevent.xdata, event.mouseevent.ydata
            if x is not None and y is not None:
                try:
                    x_idx, y_idx = int(np.round(x)), int(np.round(y))
                    if (
                        0 <= x_idx < len(rois)
                        and 0 <= y_idx < len(rois)
                        and hasattr(data_matrix, "shape")
                    ):
                        value = data_matrix[y_idx, x_idx]  # type: ignore[index]
                        roi_x, roi_y = rois[x_idx], rois[y_idx]
                        hover_state["roi_x"] = roi_x
                        hover_state["roi_y"] = roi_y
                        status = f"ROI {roi_y} ↔ ROI {roi_x}: {value:.3f}"
                        _set_format_coord(ax, lambda x, y: status)
                        ax.figure.canvas.draw_idle()
                except (IndexError, ValueError, AttributeError):
                    pass

    def on_click(event: Any) -> None:
        """Handle click events to emit ROI selection signal."""
        if event.inaxes == ax:
            # Recompute hover state from click position for immediate feedback
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                try:
                    x_idx, y_idx = int(np.round(x)), int(np.round(y))
                    if (
                        0 <= x_idx < len(rois)
                        and 0 <= y_idx < len(rois)
                        and hasattr(data_matrix, "shape")
                    ):
                        value = data_matrix[y_idx, x_idx]  # type: ignore[index]
                        roi_x, roi_y = rois[x_idx], rois[y_idx]
                        hover_state["roi_x"] = roi_x
                        hover_state["roi_y"] = roi_y
                        status = f"ROI {roi_y} ↔ ROI {roi_x}: {value:.3f}"
                        _set_format_coord(ax, lambda x, y: status)
                        # Update status bar immediately using toolbar
                        if hasattr(ax.figure.canvas, "toolbar"):
                            toolbar = ax.figure.canvas.toolbar
                            if toolbar and hasattr(toolbar, "set_message"):
                                toolbar.set_message(status)
                        ax.figure.canvas.draw_idle()
                        # Emit signal for the ROI pair
                        widget.roiSelected.emit(str(roi_y))
                        widget.roiSelected.emit(str(roi_x))
                except (IndexError, ValueError, AttributeError):
                    pass

    def on_motion(event: Any) -> None:
        """Clear status bar when mouse moves away."""
        if event.inaxes != ax:
            _set_format_coord(ax, lambda x, y: "")
            ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("pick_event", on_pick)
    ax.figure.canvas.mpl_connect("motion_notify_event", on_motion)
    _set_format_coord(ax, lambda x, y: "")
