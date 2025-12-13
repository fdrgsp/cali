from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlalchemy.exc import OperationalError
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, FOVAnalysis

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Database query for pre-computed spike correlation matrix
# -----------------------------------------------------------------------------#
def _get_spike_correlation_matrix_from_db(
    engine: Engine,
    fov_name: str,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Get the pre-computed spike correlation matrix from database.

    Parameters
    ----------
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    run_id : int | None
        Filter by specific analysis run

    Returns
    -------
    tuple[np.ndarray | None, list[int] | None]
        (correlation_matrix, roi_labels) or (None, None) if not found
    """
    if run_id is None:
        cali_logger.warning("No run ID specified for spike correlation plot.")
        return None, None

    try:
        with Session(engine) as session:
            stmt = (
                select(FOVAnalysis)
                .join(FOV, FOVAnalysis.fov_id == FOV.id)
                .where(col(FOV.name) == fov_name)
                .where(col(FOVAnalysis.analysis_result_id) == run_id)
            )

            fov_analysis = session.exec(stmt).first()

            if fov_analysis is None:
                cali_logger.info(
                    f"No FOVAnalysis found for FOV {fov_name} and run {run_id}"
                )
                return None, None

            if (
                fov_analysis.spike_correlation_matrix is None
                or fov_analysis.active_roi_labels is None
            ):
                cali_logger.info(
                    f"FOVAnalysis for {fov_name} has no spike correlation matrix"
                )
                return None, None

            corr_matrix = np.asarray(fov_analysis.spike_correlation_matrix, dtype=float)
            roi_labels = list(fov_analysis.active_roi_labels)

            return corr_matrix, roi_labels
    except OperationalError:
        # Table doesn't exist in older databases
        cali_logger.info("FOVAnalysis table not found in database")
        return None, None


def _filter_matrix_by_rois(
    matrix: np.ndarray,
    roi_labels: list[int],
    selected_rois: list[int] | None,
) -> tuple[np.ndarray, list[int]]:
    """Filter a correlation matrix to only include selected ROIs."""
    if selected_rois is None:
        return matrix, roi_labels

    indices = []
    filtered_labels = []
    for i, label in enumerate(roi_labels):
        if label in selected_rois:
            indices.append(i)
            filtered_labels.append(label)

    if len(indices) < 2:
        return matrix, roi_labels

    indices_arr = np.array(indices)
    filtered_matrix = matrix[np.ix_(indices_arr, indices_arr)]

    return filtered_matrix, filtered_labels


# -----------------------------------------------------------------------------#
# Heatmap plot (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_spike_cross_correlation_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    title_suffix: str = "",
) -> None:
    """Plot the pairwise cross-correlation matrix as a heatmap (pyqtgraph).

    title_suffix : str
        Optional suffix to add to plot titles (e.g., " - Stimulated")
    """
    plot = widget.plot_item
    assert plot is not None

    # Full reset handled by widget.clear_plot(), but clear again just in case
    plot.clear()
    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Disconnect any hover handlers from previous plots (except our own spike_ccorr_hover_handler)
    scene = plot.scene()
    handler_names = [
        "sync_hover_handler",
        "ccorr_hover_handler",
        "spike_sync_hover_handler",
        "spike_maxlag_hover_handler",
        "spike_maxlag_values_hover_handler",
        "dff_corr_hover_handler",
        "evoked_hover_handler",
    ]
    for handler_name in handler_names:
        old_handler = plot.property(handler_name)
        if old_handler is not None:
            try:
                scene.sigMouseMoved.disconnect(old_handler)
            except (TypeError, RuntimeError):
                pass
            plot.setProperty(handler_name, None)

    # Hide shared legend if present (we don't want it here)
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Query pre-computed correlation matrix from database
    correlation_matrix, roi_labels = _get_spike_correlation_matrix_from_db(
        engine, fov_name, run_id
    )

    if correlation_matrix is None or roi_labels is None:
        cali_logger.warning(
            "No spike correlation data found for this FOV. "
            "Ensure analysis has been run."
        )
        plot.setTitle(f"Pairwise Pearson Correlation (No data){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # Filter to selected ROIs if specified
    corr, rois_idxs = _filter_matrix_by_rois(correlation_matrix, roi_labels, rois)

    if len(rois_idxs) < 2:
        cali_logger.warning("Need at least 2 ROIs for correlation plot.")
        plot.setTitle(f"Pairwise Pearson Correlation (Need ≥2 ROIs){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # ---------------- IMAGE ITEM ---------------- #
    img = pg.ImageItem(corr)

    # viridis colormap
    cmap = pg.colormap.get("viridis")
    img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
    img.setLevels((0.0, 1.0))  # fixed [0, 1]

    plot.addItem(img)

    # ViewBox & geometry
    vb = plot.getViewBox()
    vb.invertY(True)  # make (0,0) top-left
    vb.setAspectLocked(True)  # keep it square
    vb.enableAutoRange(x=True, y=True)

    # Calculate median of off-diagonal elements
    mask = ~np.eye(corr.shape[0], dtype=bool)
    median_corr = np.median(corr[mask])

    title = (
        f"Pairwise Pearson Correlation (Zero-Lag - Thresholded Inferred Spikes) "
        f"(median: {median_corr:.3f})"
        f"{title_suffix}"
    )
    plot.setTitle(title)
    plot.setLabel("bottom", "ROI")
    plot.setLabel("left", "ROI")

    # Hide axis tick labels (like your MPL version)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar
    _add_colorbar_to_widget(widget, vmin=0.0, vmax=1.0, label="Correlation")

    # ---------------- Hover + Click interaction ---------------- #
    _attach_spike_corr_interaction(widget, plot, vb, rois_idxs, corr, title)


def _attach_spike_corr_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
    base_title: str = "",
) -> None:
    """
    Attach interaction to the spike correlation heatmap.

    - Hover: show ROI_i, ROI_j, value in the title
    - Click: emit widget.roiSelected with [roi_i, roi_j] as strings

    title_suffix : str
        Optional suffix to add to plot titles (e.g., " - Stimulated")
    """
    n_rows, n_cols = values.shape
    scene = plot.scene()

    # If we reconnect many times, avoid stacking multiple handlers
    old_hover = plot.property("spike_ccorr_hover_handler")
    old_click = plot.property("spike_ccorr_click_handler")
    if old_hover is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseMoved.disconnect(old_hover)
    if old_click is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseClicked.disconnect(old_click)

    def _on_mouse_moved(pos: pg.QtCore.QPointF) -> None:
        if not plot.sceneBoundingRect().contains(pos):
            plot.setTitle(base_title)
            return
        mouse_point = viewbox.mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = rois[row]
            roi_j = rois[col]
            val = float(values[row, col])
            plot.setTitle(f"{base_title} | ROI {roi_i} vs ROI {roi_j}: {val:.3f}")
        else:
            plot.setTitle(base_title)

    def _on_mouse_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = viewbox.mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = rois[row]
            roi_j = rois[col]
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)

    # Remember handlers so we can disconnect on next call
    plot.setProperty("spike_ccorr_hover_handler", _on_mouse_moved)
    plot.setProperty("spike_ccorr_click_handler", _on_mouse_clicked)


def _add_colorbar_to_widget(
    widget: _SingleWellGraphWidget,
    vmin: float,
    vmax: float,
    label: str = "Correlation",
) -> None:
    """Add a ColorBarItem to the widget layout."""
    # Remove any existing colorbar
    if widget.colorbar is not None:
        widget.plot_item.layout.removeItem(widget.colorbar)
        widget.colorbar = None

    # Create ColorBarItem
    widget.colorbar = pg.ColorBarItem(
        values=(vmin, vmax),
        colorMap=pg.colormap.get("viridis"),
        width=15,
        label=label,
        interactive=False,
    )

    # Add to plot layout (row 2, column 3 = right side)
    widget.plot_item.layout.addItem(widget.colorbar, 2, 3)
