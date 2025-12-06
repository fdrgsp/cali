from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform
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
                cali_logger.debug(
                    f"No FOVAnalysis found for FOV {fov_name} and run {run_id}"
                )
                return None, None

            if (
                fov_analysis.spike_correlation_matrix is None
                or fov_analysis.active_roi_labels is None
            ):
                cali_logger.debug(
                    f"FOVAnalysis for {fov_name} has no spike correlation matrix"
                )
                return None, None

            corr_matrix = np.asarray(fov_analysis.spike_correlation_matrix, dtype=float)
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
        plot.setTitle(f"Pairwise Cross-Correlation Matrix\n(No data){title_suffix}")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # Filter to selected ROIs if specified
    corr, rois_idxs = _filter_matrix_by_rois(correlation_matrix, roi_labels, rois)

    if len(rois_idxs) < 2:
        cali_logger.warning("Need at least 2 ROIs for correlation plot.")
        plot.setTitle(
            f"Pairwise Cross-Correlation Matrix\n(Need ≥2 ROIs){title_suffix}"
        )
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

    title = f"Pairwise Cross-Correlation Matrix\n(Thresholded Spike Data){title_suffix}"
    plot.setTitle(title)
    plot.setLabel("bottom", "ROI index")
    plot.setLabel("left", "ROI index")

    # Hide axis tick labels (like your MPL version)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar
    _add_colorbar_to_widget(widget, vmin=0.0, vmax=1.0, label="Correlation")

    # ---------------- Hover + Click interaction ---------------- #
    _attach_spike_corr_interaction(widget, plot, vb, rois_idxs, corr, title_suffix)


def _attach_spike_corr_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
    title_suffix: str = "",
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

    base_title = (
        f"Pairwise Cross-Correlation Matrix\n(Thresholded Spike Data){title_suffix}"
    )

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
            plot.setTitle(f"{base_title}\nROI {roi_i} vs ROI {roi_j}: {val:.3f}")
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


# -----------------------------------------------------------------------------#
# Hierarchical clustering plots (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_spike_hierarchical_clustering_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
    use_dendrogram: bool = False,
) -> None:
    """Plot hierarchical clustering analysis for spike correlation data."""
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Get correlation matrix from database
    full_correlation_matrix, roi_labels = _get_spike_correlation_matrix_from_db(
        engine, fov_name, run_id
    )

    if full_correlation_matrix is None or roi_labels is None:
        cali_logger.warning(
            "No spike correlation data found for this FOV. "
            "Ensure analysis has been run."
        )
        plot.setTitle("Pairwise Cross-Correlation - Hierarchical Clustering\n(No data)")
        plot.setLabel("bottom", "ROI")
        return

    # Filter to selected ROIs if specified
    correlation_matrix, rois_idxs = _filter_matrix_by_rois(
        full_correlation_matrix, roi_labels, rois
    )

    if len(rois_idxs) < 2:
        cali_logger.warning(
            "Insufficient spike data for hierarchical clustering analysis. "
            "Ensure at least two ROIs with spikes are selected."
        )
        plot.setTitle("Pairwise Cross-Correlation - Hierarchical Clustering\n(No data)")
        plot.setLabel("bottom", "ROI")
        return

    if use_dendrogram:
        _plot_spike_hierarchical_clustering_dendrogram(
            plot, correlation_matrix, rois_idxs
        )
    else:
        _plot_spike_hierarchical_clustering_map(
            widget, plot, correlation_matrix, rois_idxs
        )


def _plot_spike_hierarchical_clustering_dendrogram(
    plot: pg.PlotItem,
    correlation_matrix: np.ndarray,
    rois_idxs: list[int],
) -> None:
    """Plot the hierarchical clustering dendrogram for spike correlation data."""
    plot.clear()
    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    plot.setTitle(
        "Pairwise Cross-Correlation - Hierarchical Clustering Dendrogram\n"
        "(Thresholded Spike Data)"
    )
    plot.setLabel("left", "Distance")
    plot.setLabel("bottom", "ROI")

    # Stabilize numerics
    correlation_matrix = np.round(correlation_matrix, decimals=8)

    # Convert correlation to distance (1 - |corr|)
    dist_condensed = squareform(1.0 - np.abs(correlation_matrix))

    # Complete-linkage clustering
    Z = linkage(dist_condensed, method="complete")

    labels = [str(i) for i in rois_idxs]

    # Use scipy to compute dendrogram coordinates, but don't plot into MPL
    d = dendrogram(Z, labels=labels, no_plot=True)

    # Draw each branch as a polyline
    for xs, ys in zip(d["icoord"], d["dcoord"]):
        plot.plot(xs, ys, pen=pg.mkPen("w", width=1))

    # Put ROI labels on the bottom axis (approximate positions: 5, 15, 25, ...)
    tick_positions = [5 + 10 * i for i in range(len(d["ivl"]))]
    axis = plot.getAxis("bottom")
    axis.setTicks([list(zip(tick_positions, d["ivl"]))])

    vb = plot.getViewBox()
    vb.invertY(False)  # distance increases upward
    vb.enableAutoRange(x=True, y=True)


def _plot_spike_hierarchical_clustering_map(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    correlation_matrix: np.ndarray,
    rois_idxs: list[int],
) -> None:
    """Plot the hierarchical clustering heatmap for spike correlation data."""
    plot.clear()
    # Reset ViewBox settings that might have been set by previous plots
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    # Stabilize numerics
    correlation_matrix = np.round(correlation_matrix, decimals=8)

    # Distance → clustering → leaf order
    dist_condensed = squareform(1.0 - np.abs(correlation_matrix))
    linkage_mat = linkage(dist_condensed, method="complete")
    order = leaves_list(linkage_mat)

    # Reorder matrix and ROI IDs
    reordered_matrix = correlation_matrix[order][:, order]
    reordered_roi_ids = [rois_idxs[i] for i in order]

    plot.setTitle(
        "Pairwise Cross-Correlation - Hierarchical Clustering Map\n"
        "(Thresholded Spike Data)"
    )
    plot.setLabel("bottom", "ROI index")
    plot.setLabel("left", "ROI index")

    img = pg.ImageItem(reordered_matrix)
    cmap = pg.colormap.get("viridis")
    img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
    img.setLevels((0.0, 1.0))

    plot.addItem(img)

    vb = plot.getViewBox()
    vb.invertY(True)
    vb.setAspectLocked(True)
    vb.enableAutoRange(x=True, y=True)

    # Hide ticks (cluster map is more about pattern than axes)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    _attach_spike_cluster_interaction(
        widget, plot, vb, reordered_roi_ids, reordered_matrix
    )


def _attach_spike_cluster_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """
    Attach interaction to the clustering heatmap.

    - Hover: show ROI_i, ROI_j, value in the title
    - Click: emit widget.roiSelected with [roi_i, roi_j]
    """
    n_rows, n_cols = values.shape
    scene = plot.scene()

    old_hover = plot.property("spike_cluster_hover_handler")
    old_click = plot.property("spike_cluster_click_handler")
    if old_hover is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseMoved.disconnect(old_hover)
    if old_click is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseClicked.disconnect(old_click)

    base_title = (
        "Pairwise Cross-Correlation - Hierarchical Clustering Map\n"
        "(Thresholded Spike Data)"
    )

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
            plot.setTitle(f"{base_title}\nROI {roi_i} vs ROI {roi_j}: {val:.3f}")
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

    plot.setProperty("spike_cluster_hover_handler", _on_mouse_moved)
    plot.setProperty("spike_cluster_click_handler", _on_mouse_clicked)
