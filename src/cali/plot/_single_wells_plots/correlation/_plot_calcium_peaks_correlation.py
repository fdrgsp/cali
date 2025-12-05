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
# Database query for pre-computed correlation matrix
# -----------------------------------------------------------------------------#
def _get_correlation_matrix_from_db(
    engine: Engine,
    fov_name: str,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Get the pre-computed calcium peaks correlation matrix from database.

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
        cali_logger.warning("No run ID specified for cross-correlation plot.")
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
                cali_logger.debug(
                    f"No FOVAnalysis found for FOV {fov_name} and run {run_id}"
                )
                return None, None

            if (
                fov_analysis.calcium_peaks_max_lag_correlation_matrix is None
                or fov_analysis.active_roi_labels is None
            ):
                cali_logger.debug(
                    f"FOVAnalysis for {fov_name} has no correlation matrix"
                )
                return None, None

            corr_matrix = np.asarray(
                fov_analysis.calcium_peaks_max_lag_correlation_matrix, dtype=float
            )
            roi_labels = list(fov_analysis.active_roi_labels)

            return corr_matrix, roi_labels
    except OperationalError:
        # Table doesn't exist in older databases
        cali_logger.debug("FOVAnalysis table not found in database")
        return None, None


def _filter_matrix_by_rois(
    matrix: np.ndarray,
    roi_labels: list[int],
    selected_rois: list[int] | None,
) -> tuple[np.ndarray, list[int]]:
    """Filter a correlation/synchrony matrix to only include selected ROIs.

    Parameters
    ----------
    matrix : np.ndarray
        Full NxN matrix
    roi_labels : list[int]
        ROI labels corresponding to matrix indices
    selected_rois : list[int] | None
        ROIs to filter to, or None to keep all

    Returns
    -------
    tuple[np.ndarray, list[int]]
        (filtered_matrix, filtered_roi_labels)
    """
    if selected_rois is None:
        return matrix, roi_labels

    # Find indices of selected ROIs in the full matrix
    indices = []
    filtered_labels = []
    for i, label in enumerate(roi_labels):
        if label in selected_rois:
            indices.append(i)
            filtered_labels.append(label)

    if len(indices) < 2:
        return matrix, roi_labels  # Return full matrix if too few ROIs selected

    # Extract submatrix
    indices_arr = np.array(indices)
    filtered_matrix = matrix[np.ix_(indices_arr, indices_arr)]

    return filtered_matrix, filtered_labels


# -----------------------------------------------------------------------------#
# Plotting with pyqtgraph
# -----------------------------------------------------------------------------#
def _plot_cross_correlation_data(
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

    # Clear previous plot
    plot.clear()
    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend if present (we don't want it here)
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Query pre-computed correlation matrix from database
    correlation_matrix, roi_labels = _get_correlation_matrix_from_db(
        engine, fov_name, run_id
    )

    if correlation_matrix is None or roi_labels is None:
        plot.setTitle(f"Pairwise Cross-Correlation Matrix\n(No data){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # Filter to selected ROIs if specified
    corr, rois_idxs = _filter_matrix_by_rois(correlation_matrix, roi_labels, rois)

    if len(rois_idxs) < 2:
        plot.setTitle(
            f"Pairwise Cross-Correlation Matrix\n(Need ≥2 ROIs){title_suffix}"
        )
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # ---------------- IMAGE ITEM (centered, full view) ---------------- #
    img = pg.ImageItem(corr)

    # viridis colormap
    cmap = pg.colormap.get("viridis")
    img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
    img.setLevels((0.0, 1.0))  # fixed [0, 1]

    plot.addItem(img)

    # ViewBox & geometry
    vb = plot.getViewBox()

    # Make (0,0) top-left like imshow
    vb.invertY(True)

    # keep it square
    vb.setAspectLocked(True)  # or vb.setAspectLocked(True, ratio=1)

    title = f"Pairwise Cross-Correlation Matrix\n(Calcium Peaks Events){title_suffix}"
    plot.setTitle(title)
    plot.setLabel("bottom", "ROI index")
    plot.setLabel("left", "ROI index")

    # Hide axis tick labels (like the MPL version)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar
    _add_colorbar_to_widget(widget, vmin=0.0, vmax=1.0, label="Correlation")

    # ---------------- Hover + Click interaction ---------------- #
    _attach_heatmap_interaction(widget, plot, vb, rois_idxs, corr)


# -----------------------------------------------------------------------------#
# Hover + click helper
# -----------------------------------------------------------------------------#
def _attach_heatmap_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """
    Attach interaction to the heatmap.

    - Hover: show ROI_i, ROI_j, value in the title
    - Click: emit widget.roiSelected with a tuple (roi_i, roi_j)
    """
    n_rows, n_cols = values.shape
    scene = plot.scene()

    # If we reconnect many times, avoid stacking multiple handlers
    old_hover = plot.property("ccorr_hover_handler")
    old_click = plot.property("ccorr_click_handler")
    if old_hover is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseMoved.disconnect(old_hover)
    if old_click is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseClicked.disconnect(old_click)

    base_title = "Pairwise Cross-Correlation Matrix\n(Calcium Peaks Events)"

    def _on_mouse_moved(pos: pg.Point) -> None:
        if not plot.sceneBoundingRect().contains(pos):
            plot.setTitle(base_title)
            return
        mouse_point = viewbox.mapSceneToView(pos)
        col = round(mouse_point.x())
        row = round(mouse_point.y())
        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = rois[row]
            roi_j = rois[col]
            val = float(values[row, col])
            plot.setTitle(f"{base_title}\nROI {roi_i} vs ROI {roi_j}: {val:.3f}")
        else:
            plot.setTitle(base_title)

    def _on_mouse_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = viewbox.mapSceneToView(pos)
        col = round(mouse_point.x())
        row = round(mouse_point.y())
        if 0 <= row < n_rows and 0 <= col < n_cols:
            roi_i = rois[row]
            roi_j = rois[col]
            # Emit tuple; roiSelected is Signal(object), so this is fine
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)

    # Remember handlers so we can disconnect on next call
    plot.setProperty("ccorr_hover_handler", _on_mouse_moved)
    plot.setProperty("ccorr_click_handler", _on_mouse_clicked)


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
