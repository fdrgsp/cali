"""Cluster analysis visualization plots.

Provides three cluster visualization types:
1. Cluster-sorted correlation heatmap
2. Cluster-colored calcium peaks raster
3. Cluster-colored denoised ΔF/F traces
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.plot._util import add_colorbar_to_widget, disconnect_hover_handlers
from cali.sqlmodel._model import FOV, ROI, DataAnalysis, FOVAnalysis, Traces

if TYPE_CHECKING:
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget


# ---- Color utilities ---- #

# Qualitative color palette (distinct, colorblind-friendly)
CLUSTER_COLORS = [
    (31, 119, 180, 255),  # blue
    (255, 127, 14, 255),  # orange
    (44, 160, 44, 255),  # green
    (214, 39, 40, 255),  # red
    (148, 103, 189, 255),  # purple
    (140, 86, 75, 255),  # brown
    (227, 119, 194, 255),  # pink
    (127, 127, 127, 255),  # gray
    (188, 189, 34, 255),  # olive
    (23, 190, 207, 255),  # cyan
]

CORR_CMAP_NAME = "viridis"
CORR_CMAP = pg.colormap.get(CORR_CMAP_NAME)


def _get_cluster_color(cluster_id: int) -> tuple[int, int, int, int]:
    """Get color for a cluster, cycling through palette if needed."""
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def _get_cluster_data_from_db(
    engine: Engine,
    fov_name: str,
    run_id: int | None,
) -> tuple[
    np.ndarray | None,  # corr_matrix
    list[int] | None,  # roi_labels
    list[int] | None,  # cluster_labels
    list[int] | None,  # cluster_order
    str | None,  # cluster_method
    int | None,  # cluster_n_clusters
    float | None,  # cluster_silhouette_score
]:
    """Query FOVAnalysis for cluster data.

    Returns
    -------
    tuple
        (corr_matrix, roi_labels, cluster_labels, cluster_order,
         method, n_clusters, silhouette_score)
        All None if not found.
    """
    if run_id is None:
        return None, None, None, None, None, None, None

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
                return None, None, None, None, None, None, None

            if (
                fov_analysis.cluster_labels is None
                or fov_analysis.active_roi_labels is None
            ):
                return None, None, None, None, None, None, None

            corr_matrix = (
                np.asarray(fov_analysis.calcium_den_dff_corr_matrix, dtype=float)
                if fov_analysis.calcium_den_dff_corr_matrix is not None
                else None
            )

            return (
                corr_matrix,
                list(fov_analysis.active_roi_labels),
                list(fov_analysis.cluster_labels),
                (
                    list(fov_analysis.cluster_order)
                    if fov_analysis.cluster_order is not None
                    else None
                ),
                fov_analysis.cluster_method,
                fov_analysis.cluster_n_clusters,
                fov_analysis.cluster_silhouette_score,
            )
    except Exception:
        cali_logger.exception("Error loading cluster data from database")
        return None, None, None, None, None, None, None


# ---- Plot 1: Cluster-Sorted Correlation Heatmap ---- #


def _plot_cluster_sorted_correlation_heatmap(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Plot correlation heatmap with rows/columns sorted by cluster.

    Shows the denoised ΔF/F correlation matrix reordered so ROIs in the
    same cluster are adjacent. Cluster boundaries drawn as white lines.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)

    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    (
        corr_matrix,
        roi_labels,
        cluster_labels,
        cluster_order,
        method,
        n_clusters,
        sil_score,
    ) = _get_cluster_data_from_db(engine, fov_name, run_id)

    if (
        corr_matrix is None
        or cluster_labels is None
        or cluster_order is None
        or roi_labels is None
    ):
        plot.setTitle("Cluster-Sorted Correlation (No cluster data)")
        plot.setLabel("bottom", "ROI")
        plot.setLabel("left", "ROI")
        return

    # Filter by selected ROIs if needed
    if rois is not None and roi_labels is not None:
        # Build index mapping for selected ROIs
        indices = [i for i, lbl in enumerate(roi_labels) if lbl in rois]
        if len(indices) < 3:
            plot.setTitle("Cluster-Sorted Correlation (Need ≥3 ROIs)")
            return
        # Re-sort by cluster within selected ROIs
        sub_labels = [cluster_labels[i] for i in indices]
        sorted_idx = sorted(range(len(indices)), key=lambda x: sub_labels[x])
        reorder = [indices[s] for s in sorted_idx]
        cluster_labels_ordered = [cluster_labels[i] for i in reorder]
    else:
        reorder = cluster_order
        cluster_labels_ordered = [cluster_labels[i] for i in reorder]

    # Reorder correlation matrix
    sorted_corr = corr_matrix[np.ix_(reorder, reorder)]

    # Display heatmap
    img = pg.ImageItem(sorted_corr)
    img.setLookupTable(CORR_CMAP.getLookupTable(0, 1, 256))
    img.setLevels((-1.0, 1.0))
    plot.addItem(img)

    vb = plot.getViewBox()
    vb.invertY(True)
    vb.setAspectLocked(True)

    # Draw cluster boundary lines
    n = len(reorder)
    boundaries = []
    for i in range(1, n):
        if cluster_labels_ordered[i] != cluster_labels_ordered[i - 1]:
            boundaries.append(i)

    for b in boundaries:
        # Horizontal line
        h_line = pg.InfiniteLine(pos=b, angle=0, pen=pg.mkPen("w", width=2))
        plot.addItem(h_line)
        # Vertical line
        v_line = pg.InfiniteLine(pos=b, angle=90, pen=pg.mkPen("w", width=2))
        plot.addItem(v_line)

    method_str = method or "unknown"
    k_str = n_clusters or "?"
    sil_str = f"{sil_score:.3f}" if sil_score is not None else "N/A"
    base_title = (
        f"Cluster-Sorted Correlation ({method_str}, k={k_str}, silhouette={sil_str})"
    )
    plot.setTitle(base_title)
    plot.setLabel("bottom", "ROI (sorted by cluster)")
    plot.setLabel("left", "ROI (sorted by cluster)")
    plot.getAxis("bottom").setTicks([])
    plot.getAxis("left").setTicks([])

    add_colorbar_to_widget(
        widget, vmin=-1.0, vmax=1.0, label="Correlation", colormap=CORR_CMAP_NAME
    )

    _attach_cluster_heatmap_interaction(
        widget, plot, vb, roi_labels, reorder, sorted_corr, base_title
    )


def _attach_cluster_heatmap_interaction(
    widget: _SingleWellGraphWidget,
    plot: pg.PlotItem,
    vb: pg.ViewBox,
    roi_labels: list[int],
    reorder: list[int],
    sorted_corr: np.ndarray,
    base_title: str,
) -> None:
    """Attach hover and click interaction to the cluster-sorted correlation heatmap."""
    n = len(reorder)
    scene = plot.scene()

    old_hover = plot.property("cluster_heatmap_hover_handler")
    old_click = plot.property("cluster_heatmap_click_handler")
    if old_hover is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseMoved.disconnect(old_hover)
    if old_click is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseClicked.disconnect(old_click)

    def _on_mouse_moved(pos: pg.Point) -> None:
        if not plot.sceneBoundingRect().contains(pos):
            plot.setTitle(base_title)
            return
        mouse_point = vb.mapSceneToView(pos)
        col_idx = int(mouse_point.x())
        row_idx = int(mouse_point.y())
        if 0 <= row_idx < n and 0 <= col_idx < n:
            roi_i = roi_labels[reorder[row_idx]]
            roi_j = roi_labels[reorder[col_idx]]
            val = float(sorted_corr[row_idx, col_idx])
            plot.setTitle(f"{base_title} | ROI {roi_i} vs ROI {roi_j}: r = {val:.3f}")
        else:
            plot.setTitle(base_title)

    def _on_mouse_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = vb.mapSceneToView(pos)
        col_idx = int(mouse_point.x())
        row_idx = int(mouse_point.y())
        if 0 <= row_idx < n and 0 <= col_idx < n:
            roi_i = roi_labels[reorder[row_idx]]
            roi_j = roi_labels[reorder[col_idx]]
            widget.roiSelected.emit([str(roi_i), str(roi_j)])

    scene.sigMouseMoved.connect(_on_mouse_moved)
    scene.sigMouseClicked.connect(_on_mouse_clicked)
    plot.setProperty("cluster_heatmap_hover_handler", _on_mouse_moved)
    plot.setProperty("cluster_heatmap_click_handler", _on_mouse_clicked)


# ---- Plot 2: Cluster-Colored Raster ---- #


def _plot_cluster_colored_raster(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Plot calcium peaks raster with events colored by cluster.

    Each ROI's peak events are plotted as scatter points, colored by
    cluster assignment. ROIs are sorted by cluster on the y-axis.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(False)
    vb.invertY(True)

    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Get cluster data
    (
        _,
        roi_labels,
        cluster_labels,
        cluster_order,
        method,
        n_clusters,
        _sil_score,
    ) = _get_cluster_data_from_db(engine, fov_name, run_id)

    if cluster_labels is None or roi_labels is None or cluster_order is None:
        plot.setTitle("Cluster-Colored Raster (No cluster data)")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "ROI")
        return

    # Build label->cluster mapping
    label_to_cluster = dict(zip(roi_labels, cluster_labels))

    # Query peak data from DB
    with Session(engine) as session:
        stmt = (
            select(ROI, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))
        roi_data = session.exec(stmt).all()

    if not roi_data:
        plot.setTitle("Cluster-Colored Raster (No ROI data)")
        return

    # Sort ROIs by cluster assignment
    roi_cluster_list = []
    for roi, data_analysis in roi_data:
        if roi.label_value not in label_to_cluster:
            continue
        if not data_analysis.peaks_den_dff:
            continue
        roi_cluster_list.append((roi, data_analysis, label_to_cluster[roi.label_value]))

    roi_cluster_list.sort(key=lambda x: x[2])

    # Plot events per ROI
    for row_idx, (_roi, data_analysis, cluster_id) in enumerate(roi_cluster_list):
        peaks = np.array(data_analysis.peaks_den_dff, dtype=float)
        if len(peaks) == 0:
            continue

        color = _get_cluster_color(cluster_id)
        scatter = pg.ScatterPlotItem(
            x=peaks,
            y=np.full(len(peaks), row_idx),
            pen=pg.mkPen(None),
            brush=pg.mkBrush(*color),
            symbol="s",
            size=3,
        )
        plot.addItem(scatter)

    # Add legend entries (one per cluster)
    if n_clusters:
        widget.legend.clear()
        for c in range(n_clusters):
            color = _get_cluster_color(c)
            widget.legend.addItem(
                pg.ScatterPlotItem(
                    pen=pg.mkPen(None),
                    brush=pg.mkBrush(*color),
                    symbol="s",
                    size=8,
                ),
                f"Cluster {c}",
            )
        widget.legend.setVisible(True)

    method_str = method or "unknown"
    plot.setTitle(f"Cluster-Colored Raster ({method_str}, k={n_clusters})")
    plot.setLabel("bottom", "Frames")
    plot.setLabel("left", "ROI (sorted by cluster)")
    plot.getAxis("left").setTicks([])

    # Attach click handler: clicking a row selects that ROI
    scene = plot.scene()
    old_click = plot.property("cluster_raster_click_handler")
    if old_click is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseClicked.disconnect(old_click)

    n_rows = len(roi_cluster_list)
    _vb = plot.getViewBox()

    def _on_raster_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = _vb.mapSceneToView(pos)
        row_idx = int(mouse_point.y())
        if 0 <= row_idx < n_rows:
            roi = roi_cluster_list[row_idx][0]
            widget.roiSelected.emit(str(roi.label_value))

    scene.sigMouseClicked.connect(_on_raster_clicked)
    plot.setProperty("cluster_raster_click_handler", _on_raster_clicked)


# ---- Plot 3: Cluster-Colored Traces ---- #


def _plot_cluster_colored_traces(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    *,
    run_id: int,
) -> None:
    """Plot denoised ΔF/F traces colored by cluster assignment.

    Traces are normalized and vertically offset, sorted by cluster.
    Each trace is colored according to its cluster.
    """
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    disconnect_hover_handlers(plot)
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.invertY(False)
    vb.setAspectLocked(False)

    if hasattr(widget, "legend") and widget.legend is not None:
        widget.legend.clear()
        widget.legend.setVisible(False)

    # Get cluster data
    (
        _,
        roi_labels,
        cluster_labels,
        _cluster_order,
        method,
        n_clusters,
        _sil_score,
    ) = _get_cluster_data_from_db(engine, fov_name, run_id)

    if cluster_labels is None or roi_labels is None:
        plot.setTitle("Cluster-Colored Traces (No cluster data)")
        plot.setLabel("bottom", "Frames")
        plot.setLabel("left", "")
        return

    label_to_cluster = dict(zip(roi_labels, cluster_labels))

    # Query traces from DB
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        stmt = stmt.order_by(col(ROI.label_value))
        roi_data = session.exec(stmt).all()

    if not roi_data:
        plot.setTitle("Cluster-Colored Traces (No data)")
        return

    # Collect and sort by cluster
    trace_items = []
    for roi, traces in roi_data:
        if roi.label_value not in label_to_cluster:
            continue
        if traces.den_dff is None:
            continue
        trace_items.append((roi, traces, label_to_cluster[roi.label_value]))

    trace_items.sort(key=lambda x: x[2])

    # Plot traces with vertical offset, colored by cluster
    offset = 0.0
    plotted_roi_labels: list[int] = []
    for _roi, traces, cluster_id in trace_items:
        trace = np.asarray(traces.den_dff, dtype=float)
        if trace.size == 0:
            continue

        # Normalize: min-max scale to [0, 1]
        t_min, t_max = trace.min(), trace.max()
        if t_max > t_min:
            trace_norm = (trace - t_min) / (t_max - t_min)
        else:
            trace_norm = np.zeros_like(trace)

        color = _get_cluster_color(cluster_id)
        x = np.arange(len(trace_norm))
        plot.plot(x, trace_norm + offset, pen=pg.mkPen(color, width=1))
        plotted_roi_labels.append(_roi.label_value)
        offset += 1.1  # vertical spacing between traces

    # Add legend
    if n_clusters:
        widget.legend.clear()
        for c in range(n_clusters):
            color = _get_cluster_color(c)
            widget.legend.addItem(
                pg.PlotDataItem(pen=pg.mkPen(color, width=2)),
                f"Cluster {c}",
            )
        widget.legend.setVisible(True)

    method_str = method or "unknown"
    plot.setTitle(f"Cluster-Colored Traces ({method_str}, k={n_clusters})")
    plot.setLabel("bottom", "Frames")
    plot.setLabel("left", "Denoised ΔF/F (normalized, offset)")
    plot.getAxis("left").setTicks([])

    # Attach click handler: clicking a trace selects that ROI
    scene = plot.scene()
    old_click = plot.property("cluster_traces_click_handler")
    if old_click is not None:
        with contextlib.suppress(TypeError, RuntimeError):
            scene.sigMouseClicked.disconnect(old_click)

    n_plotted = len(plotted_roi_labels)
    _vb = plot.getViewBox()

    def _on_traces_clicked(ev: MouseClickEvent) -> None:
        pos = ev.scenePos()
        if not plot.sceneBoundingRect().contains(pos):
            return
        mouse_point = _vb.mapSceneToView(pos)
        # Traces are spaced 1.1 apart starting at offset 0
        row_idx = max(0, min(n_plotted - 1, round(mouse_point.y() / 1.1)))
        if 0 <= row_idx < n_plotted:
            widget.roiSelected.emit(str(plotted_roi_labels[row_idx]))

    scene.sigMouseClicked.connect(_on_traces_clicked)
    plot.setProperty("cluster_traces_click_handler", _on_traces_clicked)
