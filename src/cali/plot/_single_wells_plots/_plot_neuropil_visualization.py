"""Plot neuropil and ROI masks visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mplcursors
import numpy as np
from matplotlib.patches import Polygon

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from cali.gui._graph_widgets import _SingleWellGraphWidget
    from cali.sqlmodel._util import ROIData


def _plot_neuropil_masks(
    widget: _SingleWellGraphWidget,
    data: dict[str, ROIData],
    rois: list[int] | None = None,
) -> None:
    """Plot neuropil and ROI masks on widget canvas.

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

    # Filter ROIs that have both ROI and neuropil masks
    valid_rois = {
        roi_id: roi_data
        for roi_id, roi_data in filtered_data.items()
        if roi_data.mask_coord_and_shape is not None
        and roi_data.neuropil_mask_coord_and_shape is not None
    }

    if not valid_rois:
        # No valid data to plot
        ax.text(
            0.5,
            0.5,
            "No neuropil masks available.\nNeuropil correction may not be enabled.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        widget.figure.tight_layout()
        widget.canvas.draw()
        return

    # Determine image size from the first ROI
    first_roi = next(iter(valid_rois.values()))
    _, img_shape = first_roi.mask_coord_and_shape  # type: ignore

    # Create a labels array for hover detection
    # Use filled masks instead of sparse coordinates
    labels_array = np.zeros(img_shape, dtype=int)
    for roi_id, roi_data in valid_rois.items():
        roi_coords, roi_shape = roi_data.mask_coord_and_shape  # type: ignore
        # Create filled mask from coordinates
        roi_mask = coordinates_to_mask(roi_coords, roi_shape)
        # Fill the labels array with ROI ID wherever the mask is True
        labels_array[roi_mask] = int(roi_id)

    # Setup axes
    ax.set_aspect("equal")
    ax.set_xlim(0, img_shape[1])
    ax.set_ylim(img_shape[0], 0)
    ax.set_title("ROI and Neuropil Masks")
    ax.axis("off")

    # Add invisible labels image for robust hover detection
    # This ensures hover works everywhere in the plot area
    ax.imshow(
        labels_array,
        origin="upper",
        interpolation="nearest",
        extent=(0, img_shape[1], img_shape[0], 0),
        alpha=0,  # invisible, but still receives mouse events
        zorder=0,  # behind everything else
    )

    # Generate colors for each ROI using glasbey colormap
    import cmap

    glasbey_cmap = cmap.Colormap("glasbey").to_matplotlib()
    # Skip the first color (often black/dark) and use from 0.05 to skip dark
    color_indices = np.linspace(0.05, 1, len(valid_rois))
    colors = glasbey_cmap(color_indices)

    # Plot each ROI and its neuropil
    for idx, (_roi_id, roi_data) in enumerate(valid_rois.items()):
        color = colors[idx]

        # Reconstruct ROI mask
        roi_coords, roi_shape = roi_data.mask_coord_and_shape  # type: ignore
        roi_mask = coordinates_to_mask(roi_coords, roi_shape)

        # Reconstruct neuropil mask
        neuropil_coords, neuropil_shape = roi_data.neuropil_mask_coord_and_shape  # type: ignore
        neuropil_mask = coordinates_to_mask(neuropil_coords, neuropil_shape)

        # Plot ROI contour (filled)
        roi_contours = _get_contours(roi_mask)
        for contour in roi_contours:
            polygon = Polygon(
                contour,
                fill=True,
                facecolor=color,
                edgecolor="black",
                alpha=0.7,
                linewidth=1.5,
            )
            ax.add_patch(polygon)

        # Plot neuropil contour (outline only)
        neuropil_contours = _get_contours(neuropil_mask)
        for contour in neuropil_contours:
            polygon = Polygon(
                contour,
                fill=False,
                edgecolor=color,
                alpha=0.9,
                linewidth=2,
                linestyle="--",
            )
            ax.add_patch(polygon)

    # Add hover functionality
    _add_hover_functionality(ax, labels_array, widget)

    widget.figure.tight_layout()
    widget.canvas.draw()


def _add_hover_functionality(
    ax: Axes, labels_array: np.ndarray, widget: _SingleWellGraphWidget
) -> None:
    """Add hover functionality to show ROI labels and emit signal."""
    # Attach cursor to the invisible image artist for pixel-precise hover detection
    img_artist = next(art for art in ax.get_images())
    cursor = mplcursors.cursor(img_artist, hover=mplcursors.HoverMode.Transient)

    @cursor.connect("add")  # type: ignore [misc]
    def on_hover(sel: mplcursors.Selection) -> None:
        """Handle hover events on the ROI masks."""
        # Get coordinates from the hover event
        x, y = int(sel.target[0]), int(sel.target[1])

        # Check bounds and get ROI ID from labels array
        if 0 <= y < labels_array.shape[0] and 0 <= x < labels_array.shape[1]:
            label_value = labels_array[y, x]
            roi_val = str(label_value) if label_value > 0 else None
        else:
            roi_val = None

        if roi_val:
            sel.annotation.set(text=f"ROI {roi_val}", fontsize=8, color="black")
            sel.annotation.arrow_patch.set_alpha(0.5)
            widget.roiSelected.emit(roi_val)
        else:
            sel.annotation.set_visible(False)
            sel.annotation.arrow_patch.set_alpha(0)


def coordinates_to_mask(
    coords: tuple[list[int], list[int]], shape: tuple[int, int]
) -> np.ndarray:
    """Reconstruct a binary mask from coordinates and shape.

    Parameters
    ----------
    coords : tuple[list[int], list[int]]
        Tuple of (y_coords, x_coords) lists
    shape : tuple[int, int]
        Shape of the mask (height, width)

    Returns
    -------
    np.ndarray
        Binary mask array
    """
    mask = np.zeros(shape, dtype=bool)
    y_coords, x_coords = coords
    mask[y_coords, x_coords] = True
    return mask


def _get_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Extract contours from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask

    Returns
    -------
    list[np.ndarray]
        List of contour arrays, each of shape (N, 2) with (x, y) coordinates
    """
    from skimage import measure

    # Find contours at 0.5 level (for binary masks)
    contours = measure.find_contours(mask.astype(float), 0.5)

    # Convert from (row, col) to (x, y) coordinates
    contours_xy = [np.column_stack([c[:, 1], c[:, 0]]) for c in contours]

    return contours_xy
