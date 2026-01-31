"""Connectivity graph plotting for calcium imaging analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from sqlmodel import Session, col, select

from cali.plot._util import disconnect_hover_handlers

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
    from cali.sqlmodel._model import FOVAnalysis


SELECTED_COLOR = (25, 255, 25, 230)  # green
CORRELATED_COLOR = (255, 255, 0, 255)  # yellow
CORRELATED_WIDTH = 2.5  # width of edges to correlated neighbors
NODES_COLOR = (200, 200, 200, 255)  # light gray
NODES_COLOR_OUTLINE = (50, 50, 50, 255)  # dark gray
EDGE_COLOR = (100, 100, 100, 220)  # gray
EDGE_WIDTH = 5


def plot_connectivity_graph(
    widget: _SingleWellGraphWidget,
    adjacency: np.ndarray,
    weights: np.ndarray,
    roi_labels: list[int],
    roi_positions: np.ndarray | None = None,
) -> None:
    plot = widget.plot_item
    assert plot is not None

    plot.clear()
    vb = plot.getViewBox()
    vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
    vb.setAspectLocked(True)
    vb.enableAutoRange(x=True, y=True)

    # Disconnect any hover handlers from previous plots
    disconnect_hover_handlers(plot)

    # Hide shared legend if present
    if hasattr(widget, "legend") and widget.legend is not None:
        if hasattr(widget.legend, "clear"):
            widget.legend.clear()
        widget.legend.setVisible(False)

    layout = "spatial" if roi_positions is not None else "circular"
    graph_item = _create_pyqtgraph_connectivity_item(
        adjacency=adjacency,
        weights=weights,
        roi_labels=roi_labels,
        layout=layout,
        roi_positions=roi_positions,
    )
    plot.addItem(graph_item)
    plot.setTitle("Functional Connectivity")
    plot.setLabel("bottom", "")
    plot.setLabel("left", "")

    for axis in ("bottom", "left"):
        ax = plot.getAxis(axis)
        ax.setTicks([])
        ax.setStyle(showValues=False)

    scatter = graph_item.scatter

    # Grab base brushes from graph_item (the ones we set originally)
    base_brushes = graph_item.property("base_brushes")
    if base_brushes is None:
        n = adjacency.shape[0]
        base_brushes = [pg.mkBrush(*NODES_COLOR)] * n

    # Store references for highlight / clear
    plot.setProperty("connectivity_graph_item", graph_item)
    plot.setProperty("connectivity_base_brushes", base_brushes)
    plot.setProperty("connectivity_highlight_edges", [])

    def on_node_click(scatter_plot: pg.ScatterPlotItem, points: list) -> None:
        if len(points) == 0:
            return
        roi_label = points[0].data()
        if roi_label is not None:
            # Find the index of this ROI in the graph_item for highlighting
            stored_labels = graph_item.property("roi_labels")
            if stored_labels:
                try:
                    idx = stored_labels.index(str(roi_label))
                    # Get all neighbors (correlated ROIs)
                    neighbors = np.where(adjacency[idx] != 0)[0]
                    neighbor_labels = [stored_labels[j] for j in neighbors]
                    # Emit selected ROI + all correlated neighbors as a list
                    roi_list = [str(roi_label), *neighbor_labels]
                    widget.roiSelected.emit(roi_list)
                    _highlight_node_and_neighbors(plot, graph_item, idx)
                except ValueError:
                    # ROI not found in stored labels, emit just the selected ROI
                    widget.roiSelected.emit([str(roi_label)])

    def on_background_click(event: Any) -> None:
        # Clear only when click is not on a node
        scene_pos = event.scenePos()
        items_at_pos = plot.scene().items(scene_pos)
        if scatter in items_at_pos:
            # node click → handled by on_node_click
            return
        _clear_connectivity_highlight(plot)

    # Disconnect previous handlers
    old_node_handler = plot.property("connectivity_click_handler")
    if old_node_handler is not None:
        try:
            scatter.sigClicked.disconnect(old_node_handler)
        except (TypeError, RuntimeError):
            pass

    old_bg_handler = plot.property("connectivity_bg_click_handler")
    if old_bg_handler is not None:
        try:
            plot.scene().sigMouseClicked.disconnect(old_bg_handler)
        except (TypeError, RuntimeError):
            pass

    scatter.sigClicked.connect(on_node_click)
    plot.scene().sigMouseClicked.connect(on_background_click)

    plot.setProperty("connectivity_click_handler", on_node_click)
    plot.setProperty("connectivity_bg_click_handler", on_background_click)


def _clear_connectivity_highlight(plot: pg.PlotItem) -> None:
    """Restore original node colors and remove overlay edges."""
    graph_item = plot.property("connectivity_graph_item")
    base_brushes = plot.property("connectivity_base_brushes")

    if isinstance(graph_item, pg.GraphItem) and base_brushes is not None:
        # Block signals while updating visual appearance to prevent spurious events
        scatter = graph_item.scatter
        scatter.blockSignals(True)
        try:
            # Update both the scatter AND GraphItem's internal data
            scatter.setBrush(base_brushes)
            try:
                graph_item.data["symbolBrush"] = base_brushes
            except Exception:
                pass
        finally:
            scatter.blockSignals(False)

    edge_items = plot.property("connectivity_highlight_edges") or []
    for item in edge_items:
        try:
            plot.removeItem(item)
        except Exception:
            pass
    plot.setProperty("connectivity_highlight_edges", [])


def _highlight_node_and_neighbors(
    plot: pg.PlotItem,
    graph_item: pg.GraphItem,
    node_index: int,
) -> None:
    """Highlight clicked node + neighbors by recoloring the original scatter."""
    adjacency: np.ndarray = graph_item.property("adjacency")
    base_brushes = plot.property("connectivity_base_brushes")

    if adjacency is None or base_brushes is None:
        return

    pos = np.asarray(graph_item.pos)
    n = adjacency.shape[0]
    if node_index < 0 or node_index >= n or pos.shape[0] != n:
        return

    neighbors = np.where(adjacency[node_index] != 0)[0]

    # Build new brush list from base
    new_brushes = list(base_brushes)
    new_brushes[node_index] = pg.mkBrush(*SELECTED_COLOR)  # clicked
    for j in neighbors:
        new_brushes[j] = pg.mkBrush(*CORRELATED_COLOR)  # neighbors

    # Block signals while updating visual appearance to prevent spurious events
    scatter = graph_item.scatter
    scatter.blockSignals(True)
    try:
        # Apply to scatter AND GraphItem internal data (so zoom doesn't reset)
        scatter.setBrush(new_brushes)
        try:
            graph_item.data["symbolBrush"] = new_brushes
        except Exception:
            pass
    finally:
        scatter.blockSignals(False)

    # Remove old edges
    old_edges = plot.property("connectivity_highlight_edges") or []
    for item in old_edges:
        try:
            plot.removeItem(item)
        except Exception:
            pass

    # Add new edges (clicked node ↔ neighbors)
    edge_items: list[pg.PlotDataItem] = []
    x0, y0 = pos[node_index]
    for j in neighbors:
        x1, y1 = pos[j]
        edge_item = pg.PlotDataItem(
            [float(x0), float(x1)],
            [float(y0), float(y1)],
            pen=pg.mkPen(CORRELATED_COLOR, width=CORRELATED_WIDTH),
        )
        plot.addItem(edge_item)
        edge_items.append(edge_item)

    plot.setProperty("connectivity_highlight_edges", edge_items)


def _create_pyqtgraph_connectivity_item(
    adjacency: np.ndarray,
    weights: np.ndarray,
    roi_labels: Iterable[int],
    layout: str = "circular",
    node_size: float = 15.0,
    roi_positions: np.ndarray | None = None,
) -> pg.GraphItem:
    """
    Create a pyqtgraph.GraphItem representing a connectivity graph.

    Parameters
    ----------
    adjacency : np.ndarray
        Binary adjacency matrix (N, N). adjacency[i, j] != 0 means an edge.
        Expected symmetric for undirected graphs. Diagonal is ignored.
    weights : np.ndarray
        Metric values (N, N) corresponding to edges (e.g. correlation or synchrony).
        Used to scale edge width (stronger weight → thicker edge).
    roi_labels : Iterable[int]
        ROI labels in the same order as the adjacency/weights matrices.
    layout : {"circular", "spatial"}, default "circular"
        Node layout strategy. "circular" places nodes in a circle,
        "spatial" uses actual ROI positions from roi_positions.
    node_size : float, default 15.0
        Radius of node symbols in pixels.
    roi_positions : np.ndarray | None, optional
        ROI centroid positions (N, 2) for spatial layout.
        Required if layout="spatial".

    Returns
    -------
    pg.GraphItem
        A pyqtgraph GraphItem ready to be added to a PlotItem.
    """
    adjacency = np.asarray(adjacency, dtype=int)
    weights = np.asarray(weights, dtype=float)
    roi_labels = list(roi_labels)

    if adjacency.shape != weights.shape:
        raise ValueError(
            f"adjacency shape {adjacency.shape} != weights shape {weights.shape}"
        )

    n = adjacency.shape[0]
    if len(roi_labels) != n:
        raise ValueError(
            f"Number of roi_labels ({len(roi_labels)}) != adjacency size ({n})"
        )

    # 1) Node positions
    if layout == "circular":
        pos = _compute_circular_layout(n_nodes=n, radius=1.0)
    elif layout == "spatial":
        if roi_positions is None:
            raise ValueError("roi_positions required for spatial layout")
        pos = _compute_spatial_layout(roi_positions)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")

    # 2) Edge list (M, 2)
    edges = _build_undirected_edge_list(adjacency)

    # 3) Extract weights for existing edges
    if edges.size > 0:
        edge_weights = np.array([weights[i, j] for (i, j) in edges], dtype=float)
    else:
        edge_weights = np.array([], dtype=float)

    # 4) Edge widths
    edge_widths = _normalize_edge_widths(
        edge_weights, min_width=1.0, max_width=EDGE_WIDTH
    )

    # 5) Node symbols/labels
    # symbolBrush: fill color
    brushes = [pg.mkBrush(*NODES_COLOR)] * n
    # symbolPen: outline
    pens = [pg.mkPen(*NODES_COLOR_OUTLINE, width=1.0)] * n

    # Edge pens (one per edge)
    edge_pens = [pg.mkPen(*EDGE_COLOR, width=w) for w in edge_widths]

    # 6) Node text labels (ROI labels as strings)
    labels = [str(lbl) for lbl in roi_labels]

    graph_item = pg.GraphItem()
    graph_item.setData(
        pos=pos,
        adj=edges,
        size=node_size,
        symbol="o",
        symbolBrush=brushes,
        symbolPen=pens,
        pxMode=True,
        pens=edge_pens,
        texts=labels,
        textSize="10pt",
        data=roi_labels,  # Set per-point data for click handling
    )

    # ---- store metadata for interaction / highlighting ----
    graph_item.setProperty("adjacency", adjacency)
    graph_item.setProperty("roi_labels", labels)
    graph_item.setProperty("node_size", node_size)
    graph_item.setProperty("base_brushes", brushes)

    return graph_item


def _build_undirected_edge_list(adjacency: np.ndarray) -> np.ndarray:
    """Return edge list (M, 2) from a symmetric adjacency matrix (N, N)."""
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square 2D array")

    n = adjacency.shape[0]
    edges: list[tuple[int, int]] = []

    # Only use upper triangle to avoid duplicates
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] != 0:
                edges.append((i, j))

    return np.array(edges, dtype=int) if edges else np.empty((0, 2), dtype=int)


def _compute_circular_layout(n_nodes: int, radius: float = 1.0) -> np.ndarray:
    """Return positions (N, 2) for nodes placed on a circle."""
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.vstack([x, y]).T  # shape (N, 2)


def _compute_spatial_layout(roi_positions: np.ndarray) -> np.ndarray:
    """Normalize ROI positions to fit in a standard coordinate space.

    Parameters
    ----------
    roi_positions : np.ndarray
        ROI centroid positions (N, 2)

    Returns
    -------
    np.ndarray
        Normalized positions (N, 2)
    """
    positions = np.asarray(roi_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) positions, got {positions.shape}")

    # Normalize to [-1, 1] range for better visualization
    # Note: Y-axis is typically inverted in images (0 at top)
    # so we flip Y to match standard plot coordinates
    x = positions[:, 0]
    y = positions[:, 1]

    # Center and normalize
    x_center = (x.max() + x.min()) / 2
    y_center = (y.max() + y.min()) / 2
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    max_range = max(x_range, y_range)

    if max_range > 0:
        x_norm = (x - x_center) / max_range * 2  # Scale to [-1, 1]
        y_norm = -(y - y_center) / max_range * 2  # Flip and scale
    else:
        x_norm = np.zeros_like(x)
        y_norm = np.zeros_like(y)

    return np.vstack([x_norm, y_norm]).T


def _normalize_edge_widths(
    raw_weights: np.ndarray,
    min_width: float = 1.0,
    max_width: float = 5.0,
) -> np.ndarray:
    """Map nonzero weights to a line width in [min_width, max_width]."""
    if raw_weights.size == 0:
        return np.array([], dtype=float)

    w_min = float(np.min(raw_weights))
    w_max = float(np.max(raw_weights))

    if w_max <= w_min:
        # All edges same weight → constant width
        return np.full_like(raw_weights, (min_width + max_width) / 2.0, dtype=float)

    norm = (raw_weights - w_min) / (w_max - w_min)
    return min_width + norm * (max_width - min_width)


def get_fov_analysis_from_db(
    engine: Engine, fov_name: str, run_id: int
) -> FOVAnalysis | None:
    """Retrieve FOVAnalysis for a given FOV and run.

    Parameters
    ----------
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    run_id : int
        Analysis result ID

    Returns
    -------
    FOVAnalysis | None
        FOVAnalysis object or None if not found
    """
    from cali.sqlmodel._model import FOV, FOVAnalysis

    with Session(engine) as session:
        stmt = (
            select(FOVAnalysis)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .where(col(FOVAnalysis.analysis_result_id) == run_id)
        )
        return session.exec(stmt).first()  # type: ignore


def _plot_connectivity_network_data(
    widget: _SingleWellGraphWidget,
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> None:
    """Plot connectivity graph from FOVAnalysis data.

    Parameters
    ----------
    widget : _SingleWellGraphWidget
        The widget to plot in
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    rois : list[int] | None
        Selected ROI labels to display, or None for all active ROIs
    run_id : int | None
        Analysis result ID
    """
    if run_id is None:
        return

    # Get FOVAnalysis from database
    fov_analysis = get_fov_analysis_from_db(engine, fov_name, run_id)
    if fov_analysis is None:
        return

    # Get threshold and method from widget (stored as attributes)
    threshold = widget._connectivity_threshold
    method = widget._connectivity_method

    # Show the threshold control widget
    widget._connectivity_threshold_widget.setVisible(True)

    # Compute connectivity metrics with current threshold and method
    from typing import cast

    from cali.analysis._fov_metrics import (
        ConnectivityMethod,
        _compute_connectivity_metrics,
    )

    try:
        adjacency, weights, roi_labels = _compute_connectivity_metrics(
            fov_analysis,
            method=cast("ConnectivityMethod", method),
            threshold=threshold,
            use_absolute_for_corr=True,
        )
    except ValueError:
        # FOVAnalysis doesn't have the required data
        return

    # Filter to selected ROIs if specified
    adjacency, weights, roi_labels = _filter_connectivity_by_rois(
        adjacency, weights, roi_labels, rois
    )

    if len(roi_labels) < 2:
        # Need at least 2 ROIs to show connectivity - clear and show message
        plot = widget.plot_item
        assert plot is not None
        plot.clear()
        plot.setTitle("Functional Connectivity (Need ≥2 ROIs)")
        plot.setLabel("bottom", "")
        plot.setLabel("left", "")
        return

    # Get ROI positions for spatial layout
    roi_positions = _get_roi_positions(engine, fov_name, roi_labels)

    # Plot the connectivity graph
    plot_connectivity_graph(
        widget=widget,
        adjacency=adjacency,
        weights=weights,
        roi_labels=roi_labels,
        roi_positions=roi_positions,
    )


def _filter_connectivity_by_rois(
    adjacency: np.ndarray,
    weights: np.ndarray,
    roi_labels: list[int],
    selected_rois: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Filter connectivity matrices to only include selected ROIs.

    Parameters
    ----------
    adjacency : np.ndarray
        Full NxN adjacency matrix
    weights : np.ndarray
        Full NxN weights matrix
    roi_labels : list[int]
        ROI labels corresponding to matrix indices
    selected_rois : list[int] | None
        ROIs to filter to, or None to keep all

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[int]]
        (filtered_adjacency, filtered_weights, filtered_roi_labels)
    """
    if selected_rois is None:
        return adjacency, weights, roi_labels

    # Find indices of selected ROIs in the full matrices
    indices = []
    filtered_labels = []
    for i, label in enumerate(roi_labels):
        if label in selected_rois:
            indices.append(i)
            filtered_labels.append(label)

    if len(indices) < 2:
        # Return full matrices if too few ROIs selected
        return adjacency, weights, roi_labels

    # Extract submatrices
    indices_arr = np.array(indices)
    filtered_adjacency = adjacency[np.ix_(indices_arr, indices_arr)]
    filtered_weights = weights[np.ix_(indices_arr, indices_arr)]

    return filtered_adjacency, filtered_weights, filtered_labels


def _get_roi_positions(
    engine: Engine, fov_name: str, roi_labels: list[int]
) -> np.ndarray | None:
    """Get ROI centroid positions from database.

    Parameters
    ----------
    engine : Engine
        Database engine
    fov_name : str
        Name of the FOV
    roi_labels : list[int]
        ROI label values

    Returns
    -------
    np.ndarray | None
        ROI positions (N, 2) or None if not available
    """
    from cali.sqlmodel._model import FOV, ROI

    with Session(engine) as session:
        # Get ROIs for this FOV with their masks
        stmt = (
            select(ROI)
            .join(FOV)
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.label_value).in_(roi_labels))
        )
        rois = session.exec(stmt).all()

        if not rois:
            return None

        # Create a mapping from label_value to ROI
        roi_map = {roi.label_value: roi for roi in rois}

        # Get positions in the same order as roi_labels
        positions = []
        for label in roi_labels:
            roi = roi_map.get(label)
            if roi is None or roi.roi_mask is None:
                # If any ROI is missing mask data, fall back to circular layout
                return None

            # Compute centroid from mask coordinates
            mask = roi.roi_mask
            if mask.coords_x is None or mask.coords_y is None:
                return None

            if len(mask.coords_x) == 0 or len(mask.coords_y) == 0:
                return None

            # Centroid is the mean of all coordinates
            centroid_x = np.mean(mask.coords_x)
            centroid_y = np.mean(mask.coords_y)
            positions.append([centroid_x, centroid_y])

        return np.array(positions, dtype=float)
