from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.signal import correlate
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# -----------------------------------------------------------------------------#
# Cross-correlation computation
# -----------------------------------------------------------------------------#
def _calculate_spike_cross_correlation(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> tuple[np.ndarray | None, list[int] | None]:
    """Calculate the cross-correlation matrix for spike trains from active ROIs.

    Uses thresholded inferred_spikes → binary spike trains, then computes
    maximum normalized cross-correlation over all lags.
    """
    spike_trains: list[np.ndarray] = []
    rois_idxs: list[int] = []

    # Query ROIs from database with optimized joins
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .join(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )

        # IMPORTANT: use label_value for ROI subset, not ROI.id
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        stmt = stmt.order_by(col(ROI.label_value))
        roi_results: list[tuple[ROI, Traces, DataAnalysis]] = session.exec(stmt).all()

    # Extract spike trains for the active ROIs
    for roi, traces, data_analysis in roi_results:
        if traces is None or data_analysis is None:
            continue

        inferred_spikes = traces.inferred_spikes
        inferred_spikes_threshold = data_analysis.inferred_spikes_threshold

        if inferred_spikes is None or inferred_spikes_threshold is None:
            continue

        spikes = np.asarray(inferred_spikes, dtype=float)
        the = float(inferred_spikes_threshold)

        # Threshold and binarize (vectorized)
        spikes[spikes <= the] = 0.0
        spike_train = (spikes > 0.0).astype(float)

        if spike_train.sum() <= 0:
            # Skip ROIs with no spikes
            continue

        if roi.label_value is None:
            continue

        rois_idxs.append(int(roi.label_value))
        spike_trains.append(spike_train)

    if len(rois_idxs) <= 1:
        cali_logger.warning(
            "Insufficient spike data for correlation analysis. "
            "Need at least 2 ROIs with spikes."
        )
        return None, None

    # Convert to array: shape (n_rois, n_frames)
    spike_trains_array = np.vstack(spike_trains)

    # Z-score per ROI (handle varying firing rates)
    spike_trains_zscore = zscore(spike_trains_array, axis=1, nan_policy="omit")
    spike_trains_zscore = np.nan_to_num(
        spike_trains_zscore, nan=0.0, posinf=0.0, neginf=0.0
    )

    n_rois = len(rois_idxs)
    correlation_matrix = np.empty((n_rois, n_rois), dtype=float)

    # Precompute norms, avoid repeated work
    norms = np.linalg.norm(spike_trains_zscore, axis=1)
    norms[norms == 0] = np.finfo(float).eps  # avoid division by zero

    # Diagonal is self-correlation = 1
    np.fill_diagonal(correlation_matrix, 1.0)

    # Compute only upper triangle, mirror to lower
    for i in range(n_rois):
        x = spike_trains_zscore[i]
        for j in range(i + 1, n_rois):
            y = spike_trains_zscore[j]

            # FFT-based cross-correlation over all lags
            corr = correlate(x, y, mode="full", method="fft")
            corr /= norms[i] * norms[j]

            # Use max absolute correlation
            max_corr = float(np.max(np.abs(corr)))
            correlation_matrix[i, j] = max_corr
            correlation_matrix[j, i] = max_corr

    return correlation_matrix, rois_idxs


# -----------------------------------------------------------------------------#
# Heatmap plot (pyqtgraph)
# -----------------------------------------------------------------------------#
def _plot_spike_cross_correlation_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot pairwise cross-correlation matrix for spike trains (pyqtgraph)."""
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

    correlation_matrix, rois_idxs = _calculate_spike_cross_correlation(
        engine, fov_name, rois, run_id
    )

    if correlation_matrix is None or rois_idxs is None:
        cali_logger.warning(
            "Insufficient spike data for cross-correlation analysis. "
            "Ensure at least two ROIs with spikes are selected."
        )
        plot.setTitle("Pairwise Cross-Correlation Matrix\n(No data)")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    corr = correlation_matrix

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

    plot.setTitle("Pairwise Cross-Correlation Matrix\n(Thresholded Spike Data)")
    plot.setLabel("bottom", "ROI index")
    plot.setLabel("left", "ROI index")

    # Hide axis tick labels (like your MPL version)
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    # Add colorbar
    _add_colorbar_to_widget(widget, vmin=0.0, vmax=1.0, label="Correlation")

    # ---------------- Hover + Click interaction ---------------- #
    _attach_spike_corr_interaction(widget, plot, vb, rois_idxs, corr)


def _attach_spike_corr_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    viewbox: pg.ViewBox,
    rois: list[int],
    values: np.ndarray,
) -> None:
    """
    Attach interaction to the spike correlation heatmap.

    - Hover: show ROI_i, ROI_j, value in the title
    - Click: emit widget.roiSelected with [roi_i, roi_j] as strings
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

    base_title = "Pairwise Cross-Correlation Matrix\n(Thresholded Spike Data)"

    def _on_mouse_moved(pos: pg.QtCore.QPointF) -> None:
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

    correlation_matrix, rois_idxs = _calculate_spike_cross_correlation(
        engine, fov_name, rois, run_id
    )

    if correlation_matrix is None or rois_idxs is None:
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
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)

    plot.setProperty("spike_cluster_hover_handler", _on_mouse_moved)
    plot.setProperty("spike_cluster_click_handler", _on_mouse_clicked)
