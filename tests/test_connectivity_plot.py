"""Tests for connectivity plot functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6.QtWidgets import QWidget

from cali.sqlmodel import FOVAnalysis

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytestqt.qtbot import QtBot


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def plot_widget_setup(qtbot: QtBot) -> tuple[MagicMock, pg.PlotItem]:
    """Create Qt widgets for plot testing with mock signal."""
    widget = QWidget()
    qtbot.addWidget(widget)
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem

    assert plot_widget.plotItem is not None
    return mock_widget, plot_widget.plotItem


@pytest.fixture
def simple_connectivity_data() -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Create simple 3-node connectivity data for testing."""
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    weights = np.array([[0.0, 0.8, 0.0], [0.8, 0.0, 0.7], [0.0, 0.7, 0.0]])
    roi_labels = [1, 2, 3]
    return adjacency, weights, roi_labels


@pytest.fixture
def four_node_data() -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Create 4-node connectivity data with specific correlation values."""
    corr_matrix = np.array(
        [
            [1.0, 0.9, 0.5, 0.2],
            [0.9, 1.0, 0.6, 0.3],
            [0.5, 0.6, 1.0, 0.4],
            [0.2, 0.3, 0.4, 1.0],
        ]
    )
    return corr_matrix, corr_matrix, [0, 1, 2, 3]


# ============================================================================
# Basic Graph Creation Tests
# ============================================================================


def test_plot_connectivity_graph_creates_items() -> None:
    """Test that connectivity graph creates plot items."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        plot_connectivity_graph,
    )

    mock_widget = MagicMock()
    mock_plot_item = MagicMock()
    mock_widget.plot_item = mock_plot_item

    roi_labels = [0, 1, 2]
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    weights = np.array([[0.0, 0.8, 0.0], [0.8, 0.0, 0.5], [0.0, 0.5, 0.0]])

    plot_connectivity_graph(
        widget=mock_widget,
        roi_labels=roi_labels,
        adjacency=adjacency,
        weights=weights,
    )

    assert mock_plot_item.addItem.called


# ============================================================================
# Connectivity Metrics Tests (Parametrized)
# ============================================================================


@pytest.mark.parametrize(
    ("threshold", "expected_edges"),
    [
        (0.0, 12),  # All off-diagonal elements (4*3)
        (0.5, 6),  # Values >= 0.5: (0,1), (1,0), (0,2), (2,0), (1,2), (2,1)
        (0.9, 2),  # Only 0.9 values (bidirectional)
    ],
)
def test_compute_connectivity_metrics_thresholds(
    four_node_data: tuple[np.ndarray, np.ndarray, list[int]],
    threshold: float,
    expected_edges: int,
) -> None:
    """Test connectivity metrics with various thresholds."""
    from cali.analysis._util import _compute_connectivity_metrics

    corr_matrix, _, roi_labels = four_node_data

    fov_analysis = FOVAnalysis(
        id=1,
        calcium_dec_dff_corr_matrix=corr_matrix.tolist(),
        active_roi_labels=roi_labels,
    )

    adjacency, _, returned_labels = _compute_connectivity_metrics(
        fov_analysis, method="calcium_dec_dff_corr", threshold=threshold
    )

    assert adjacency.sum() == expected_edges
    assert returned_labels == roi_labels


def test_compute_connectivity_specific_threshold_values(
    four_node_data: tuple[np.ndarray, np.ndarray, list[int]],
) -> None:
    """Test specific adjacency values with threshold 0.5."""
    from cali.analysis._util import _compute_connectivity_metrics

    corr_matrix, _, roi_labels = four_node_data

    fov_analysis = FOVAnalysis(
        id=1,
        calcium_dec_dff_corr_matrix=corr_matrix.tolist(),
        active_roi_labels=roi_labels,
    )

    adjacency, _, _ = _compute_connectivity_metrics(
        fov_analysis, method="calcium_dec_dff_corr", threshold=0.5
    )

    assert adjacency[0, 1] == 1  # 0.9 >= 0.5
    assert adjacency[1, 2] == 1  # 0.6 >= 0.5
    assert adjacency[0, 3] == 0  # 0.2 < 0.5


@pytest.mark.parametrize(
    "method",
    ["calcium_dff_corr", "calcium_dec_dff_corr", "calcium_peaks_maxlag"],
)
def test_compute_connectivity_different_methods(method: str) -> None:
    """Test computing connectivity with different correlation methods."""
    from cali.analysis._util import _compute_connectivity_metrics

    fov_analysis = FOVAnalysis(
        id=1,
        calcium_dff_correlation_matrix=[[1.0, 0.8], [0.8, 1.0]],
        calcium_dec_dff_corr_matrix=[[1.0, 0.7], [0.7, 1.0]],
        calcium_peaks_max_lag_correlation_matrix=[[1.0, 0.9], [0.9, 1.0]],
        active_roi_labels=[0, 1],
    )

    adjacency, weights, roi_labels = _compute_connectivity_metrics(
        fov_analysis,
        method=method,  # type: ignore[arg-type]
        threshold=0.5,
    )

    assert adjacency.shape == (2, 2)
    assert weights.shape == (2, 2)
    assert len(roi_labels) == 2


# ============================================================================
# Filtering Tests
# ============================================================================


@pytest.mark.parametrize(
    ("selected_rois", "expected_shape", "expected_labels"),
    [
        (None, (4, 4), [0, 1, 2, 3]),  # No filtering
        ([1, 2], (2, 2), [1, 2]),  # Filter to subset
        ([1], (4, 4), [0, 1, 2, 3]),  # Too few (returns original)
    ],
)
def test_filter_connectivity_by_rois(
    selected_rois: list[int] | None,
    expected_shape: tuple[int, int],
    expected_labels: list[int],
) -> None:
    """Test filtering connectivity matrices by selected ROIs."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _filter_connectivity_by_rois,
    )

    adjacency = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ]
    )
    weights = np.array(
        [
            [0.0, 0.8, 0.7, 0.0],
            [0.8, 0.0, 0.6, 0.5],
            [0.7, 0.6, 0.0, 0.9],
            [0.0, 0.5, 0.9, 0.0],
        ]
    )
    roi_labels = [0, 1, 2, 3]

    filtered_adj, filtered_weights, filtered_labels = _filter_connectivity_by_rois(
        adjacency, weights, roi_labels, selected_rois
    )

    assert filtered_adj.shape == expected_shape
    assert filtered_weights.shape == expected_shape
    assert filtered_labels == expected_labels


def test_filter_connectivity_values() -> None:
    """Test that filtered connectivity preserves correct values."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _filter_connectivity_by_rois,
    )

    adjacency = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ]
    )
    weights = np.array(
        [
            [0.0, 0.8, 0.7, 0.0],
            [0.8, 0.0, 0.6, 0.5],
            [0.7, 0.6, 0.0, 0.9],
            [0.0, 0.5, 0.9, 0.0],
        ]
    )

    filtered_adj, filtered_weights, _ = _filter_connectivity_by_rois(
        adjacency, weights, [0, 1, 2, 3], [1, 2]
    )

    # Check values are correctly extracted for ROI 1->2 connection
    assert filtered_adj[0, 1] == 1
    assert filtered_weights[0, 1] == 0.6


# ============================================================================
# Click Handler Tests
# ============================================================================


def test_node_click_emits_roi_signal(qtbot: QtBot) -> None:
    """Test that clicking a node emits the roiSelected signal with correct ROI."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        plot_connectivity_graph,
    )

    # Create fresh widgets for this test
    widget = QWidget()
    qtbot.addWidget(widget)
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem
    plot_item = plot_widget.plotItem
    assert plot_item is not None

    roi_labels = [5, 10, 15]
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    weights = np.array([[0.0, 0.8, 0.0], [0.8, 0.0, 0.7], [0.0, 0.7, 0.0]])

    plot_connectivity_graph(
        widget=mock_widget, roi_labels=roi_labels, adjacency=adjacency, weights=weights
    )

    click_handler = plot_item.property("connectivity_click_handler")
    assert click_handler is not None

    # Simulate clicking on node 1 (ROI label 10)
    mock_point = MagicMock()
    mock_point.index.return_value = 1
    click_handler(None, [mock_point])

    mock_widget.roiSelected.emit.assert_called_once_with("10")


@pytest.mark.parametrize(
    ("index_value", "should_emit"),
    [
        (None, False),  # Empty points (will test with [])
        (999, False),  # Invalid index
    ],
)
def test_node_click_edge_cases(
    qtbot: QtBot,
    index_value: int | None,
    should_emit: bool,
) -> None:
    """Test node click handler with edge cases."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        plot_connectivity_graph,
    )

    widget = QWidget()
    qtbot.addWidget(widget)
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem
    plot_item = plot_widget.plotItem
    assert plot_item is not None

    roi_labels = [1, 2, 3]
    adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    weights = np.array([[0.0, 0.8, 0.0], [0.8, 0.0, 0.7], [0.0, 0.7, 0.0]])

    plot_connectivity_graph(
        widget=mock_widget, roi_labels=roi_labels, adjacency=adjacency, weights=weights
    )

    click_handler = plot_item.property("connectivity_click_handler")

    # Test empty points list or invalid index
    if index_value is None:
        points = []
    else:
        mock_point = MagicMock()
        mock_point.index.return_value = index_value
        points = [mock_point]

    click_handler(None, points)

    if should_emit:
        mock_widget.roiSelected.emit.assert_called_once()
    else:
        mock_widget.roiSelected.emit.assert_not_called()


def test_background_click_clears_highlight(qtbot: QtBot) -> None:
    """Test that clicking background clears node highlighting."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        plot_connectivity_graph,
    )

    widget = QWidget()
    qtbot.addWidget(widget)
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem
    plot_item = plot_widget.plotItem
    assert plot_item is not None

    roi_labels = [1, 2, 3]
    adjacency = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    weights = np.array([[0.0, 0.8, 0.7], [0.8, 0.0, 0.9], [0.7, 0.9, 0.0]])

    plot_connectivity_graph(
        widget=mock_widget, roi_labels=roi_labels, adjacency=adjacency, weights=weights
    )

    # First, click a node to create highlight
    click_handler = plot_item.property("connectivity_click_handler")
    mock_point = MagicMock()
    mock_point.index.return_value = 1
    click_handler(None, [mock_point])

    initial_edges = plot_item.property("connectivity_highlight_edges") or []
    assert len(initial_edges) > 0

    # Now simulate background click
    bg_click_handler = plot_item.property("connectivity_bg_click_handler")
    mock_event = MagicMock()
    mock_event.scenePos.return_value = MagicMock()

    mock_scene = MagicMock()
    mock_scene.items.return_value = []
    plot_item.scene = lambda: mock_scene

    bg_click_handler(mock_event)

    final_edges = plot_item.property("connectivity_highlight_edges") or []
    assert len(final_edges) == 0


# ============================================================================
# Highlight/Clear Tests
# ============================================================================


def test_highlight_node_and_neighbors(
    qtbot: QtBot,
    simple_connectivity_data: tuple[np.ndarray, np.ndarray, list[int]],
) -> None:
    """Test that highlighting a node creates correct edge overlays."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _create_pyqtgraph_connectivity_item,
        _highlight_node_and_neighbors,
    )

    widget = QWidget()
    qtbot.addWidget(widget)
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot_item = plot_widget.plotItem
    assert plot_item is not None
    adjacency, weights, roi_labels = simple_connectivity_data

    graph_item = _create_pyqtgraph_connectivity_item(
        adjacency=adjacency, weights=weights, roi_labels=roi_labels, layout="circular"
    )

    plot_item.addItem(graph_item)

    base_brushes = graph_item.property("base_brushes")
    plot_item.setProperty("connectivity_base_brushes", base_brushes)
    plot_item.setProperty("connectivity_graph_item", graph_item)
    plot_item.setProperty("connectivity_highlight_edges", [])

    # Highlight node 1 (connected to nodes 0 and 2)
    _highlight_node_and_neighbors(plot_item, graph_item, node_index=1)

    edge_items = plot_item.property("connectivity_highlight_edges")
    assert edge_items is not None
    assert len(edge_items) == 2  # Node 1 has 2 connections


def test_clear_connectivity_highlight(qtbot: QtBot) -> None:
    """Test that clearing highlight removes edge overlays."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _clear_connectivity_highlight,
        _create_pyqtgraph_connectivity_item,
        _highlight_node_and_neighbors,
    )

    widget = QWidget()
    qtbot.addWidget(widget)
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot_item = plot_widget.plotItem
    assert plot_item is not None

    adjacency = np.array([[0, 1], [1, 0]])
    weights = np.array([[0.0, 0.8], [0.8, 0.0]])
    roi_labels = [1, 2]

    graph_item = _create_pyqtgraph_connectivity_item(
        adjacency=adjacency, weights=weights, roi_labels=roi_labels, layout="circular"
    )

    plot_item.addItem(graph_item)

    base_brushes = graph_item.property("base_brushes")
    plot_item.setProperty("connectivity_base_brushes", base_brushes)
    plot_item.setProperty("connectivity_graph_item", graph_item)
    plot_item.setProperty("connectivity_highlight_edges", [])

    # Add highlight
    _highlight_node_and_neighbors(plot_item, graph_item, node_index=0)
    edge_items = plot_item.property("connectivity_highlight_edges")
    assert len(edge_items) > 0

    # Clear highlight
    _clear_connectivity_highlight(plot_item)
    edge_items = plot_item.property("connectivity_highlight_edges")
    assert len(edge_items) == 0


# ============================================================================
# Graph Item Creation Tests
# ============================================================================


@pytest.mark.parametrize(
    ("layout", "roi_positions"),
    [
        ("circular", None),
        ("spatial", np.array([[100, 200], [300, 400], [500, 600]])),
    ],
)
def test_create_graph_with_layouts(
    simple_connectivity_data: tuple[np.ndarray, np.ndarray, list[int]],
    layout: str,
    roi_positions: np.ndarray | None,
) -> None:
    """Test creating graph items with different layout strategies."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _create_pyqtgraph_connectivity_item,
    )

    adjacency, weights, roi_labels = simple_connectivity_data

    graph_item = _create_pyqtgraph_connectivity_item(
        adjacency=adjacency,
        weights=weights,
        roi_labels=roi_labels,
        layout=layout,
        roi_positions=roi_positions,
    )

    assert graph_item is not None
    assert graph_item.property("adjacency") is not None
    assert len(graph_item.property("roi_labels")) == 3


@pytest.mark.parametrize(
    ("adjacency", "weights", "roi_labels", "layout", "roi_positions", "error_match"),
    [
        # Mismatched shapes
        (
            np.array([[0, 1], [1, 0]]),
            np.array([[0.0, 0.8, 0.0], [0.8, 0.0, 0.7], [0.0, 0.7, 0.0]]),
            [1, 2],
            "circular",
            None,
            r"adjacency shape.*!= weights shape",
        ),
        # Mismatched labels
        (
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            np.array([[0.0, 0.8, 0.0], [0.8, 0.0, 0.7], [0.0, 0.7, 0.0]]),
            [1, 2],
            "circular",
            None,
            r"Number of roi_labels.*!= adjacency size",
        ),
        # Spatial layout without positions
        (
            np.array([[0, 1], [1, 0]]),
            np.array([[0.0, 0.8], [0.8, 0.0]]),
            [1, 2],
            "spatial",
            None,
            "roi_positions required",
        ),
    ],
)
def test_create_graph_validation_errors(
    adjacency: np.ndarray,
    weights: np.ndarray,
    roi_labels: list[int],
    layout: str,
    roi_positions: np.ndarray | None,
    error_match: str,
) -> None:
    """Test that graph creation validates input parameters."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _create_pyqtgraph_connectivity_item,
    )

    with pytest.raises(ValueError, match=error_match):
        _create_pyqtgraph_connectivity_item(
            adjacency=adjacency,
            weights=weights,
            roi_labels=roi_labels,
            layout=layout,
            roi_positions=roi_positions,
        )


# ============================================================================
# Helper Function Tests
# ============================================================================


def test_build_undirected_edge_list() -> None:
    """Test building edge list from adjacency matrix."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _build_undirected_edge_list,
    )

    adjacency = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    edges = _build_undirected_edge_list(adjacency)

    assert edges.shape == (3, 2)
    edge_set = {tuple(e) for e in edges}
    assert (0, 1) in edge_set
    assert (0, 2) in edge_set
    assert (1, 2) in edge_set


def test_build_undirected_edge_list_empty() -> None:
    """Test building edge list from adjacency with no edges."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _build_undirected_edge_list,
    )

    adjacency = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    edges = _build_undirected_edge_list(adjacency)

    assert edges.shape == (0, 2)


def test_compute_circular_layout() -> None:
    """Test circular layout positioning."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _compute_circular_layout,
    )

    positions = _compute_circular_layout(n_nodes=4, radius=1.0)

    assert positions.shape == (4, 2)
    distances = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    np.testing.assert_allclose(distances, 1.0, rtol=1e-10)


@pytest.mark.parametrize(
    ("roi_positions", "expected_range"),
    [
        (np.array([[0, 0], [100, 0], [100, 100], [0, 100]]), (-1.1, 1.1)),
        (np.array([[50, 50]]), (0.0, 0.0)),  # Single point edge case
    ],
)
def test_compute_spatial_layout(
    roi_positions: np.ndarray, expected_range: tuple[float, float]
) -> None:
    """Test spatial layout normalization."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _compute_spatial_layout,
    )

    normalized = _compute_spatial_layout(roi_positions)

    assert normalized.shape == roi_positions.shape

    min_val, max_val = expected_range
    if min_val == max_val:  # Single point case
        np.testing.assert_array_equal(normalized, [[0.0, 0.0]])
    else:
        assert normalized.min() >= min_val
        assert normalized.max() <= max_val


@pytest.mark.parametrize(
    ("weights", "min_width", "max_width", "check_fn"),
    [
        # Normal case
        (
            np.array([0.2, 0.5, 0.8, 1.0]),
            1.0,
            5.0,
            lambda w: w[0] == pytest.approx(1.0) and w[3] == pytest.approx(5.0),
        ),
        # Constant weights
        (np.array([0.7, 0.7, 0.7]), 2.0, 6.0, lambda w: np.allclose(w, 4.0)),
        # Empty array
        (np.array([]), 1.0, 5.0, lambda w: w.shape == (0,)),
    ],
)
def test_normalize_edge_widths(
    weights: np.ndarray,
    min_width: float,
    max_width: float,
    check_fn: Callable[[np.ndarray], bool],
) -> None:
    """Test edge width normalization with various inputs."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _normalize_edge_widths,
    )

    widths = _normalize_edge_widths(weights, min_width=min_width, max_width=max_width)
    assert check_fn(widths)


# ============================================================================
# Function Existence Tests (for backwards compatibility)
# ============================================================================


@pytest.mark.parametrize(
    "function_name",
    [
        "_plot_connectivity_network_data",
        "get_fov_analysis_from_db",
    ],
)
def test_function_exists(function_name: str) -> None:
    """Test that expected functions exist and are importable."""
    from cali.plot._single_wells_plots.correlation import _plot_connectivity

    func = getattr(_plot_connectivity, function_name)
    assert func is not None
    assert callable(func)


def test_plot_connectivity_wrapper_signature() -> None:
    """Test that wrapper function has correct signature."""
    from inspect import signature

    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _plot_connectivity_network_data,
    )

    sig = signature(_plot_connectivity_network_data)
    params = list(sig.parameters.keys())

    assert "widget" in params
    assert "engine" in params
    assert "fov_name" in params
