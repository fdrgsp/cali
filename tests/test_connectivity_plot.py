"""Tests for connectivity plot functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from cali.sqlmodel import FOVAnalysis


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock database engine."""
    return MagicMock()


def test_plot_connectivity_graph_pg_creates_nodes() -> None:
    """Test that connectivity graph creates nodes for all ROIs."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        plot_connectivity_graph,
    )

    # Create mock widget with required attributes
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

    # Verify that plot items were added
    assert mock_plot_item.addItem.called


def test_compute_connectivity_metrics_with_threshold() -> None:
    """Test computing connectivity metrics with threshold."""
    from cali.analysis._util import _compute_connectivity_metrics

    # Create FOVAnalysis with correlation matrix
    fov_analysis = FOVAnalysis(
        id=1,
        calcium_dec_dff_corr_matrix=[
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.5],
            [0.3, 0.5, 1.0],
        ],
        active_roi_labels=[0, 1, 2],
    )

    # Test with threshold 0.6
    adjacency, _, roi_labels = _compute_connectivity_metrics(
        fov_analysis, method="calcium_dec_dff_corr", threshold=0.6
    )

    # Check adjacency: only values >= 0.6 should have edges
    expected_adjacency = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(adjacency, expected_adjacency)

    # Check that ROI labels were returned
    assert roi_labels == [0, 1, 2]


def test_compute_connectivity_metrics_threshold_0_0() -> None:
    """Test connectivity metrics with threshold 0.0 (all edges)."""
    from cali.analysis._util import _compute_connectivity_metrics

    fov_analysis = FOVAnalysis(
        id=1,
        calcium_dff_correlation_matrix=[
            [1.0, 0.9, 0.5, 0.2],
            [0.9, 1.0, 0.6, 0.3],
            [0.5, 0.6, 1.0, 0.4],
            [0.2, 0.3, 0.4, 1.0],
        ],
        active_roi_labels=[0, 1, 2, 3],
    )

    # Threshold 0.0 - all edges
    adjacency, _, _ = _compute_connectivity_metrics(
        fov_analysis, method="calcium_dff_corr", threshold=0.0
    )
    assert adjacency.sum() == 12  # All off-diagonal elements (4*3)


def test_compute_connectivity_metrics_threshold_0_5() -> None:
    """Test connectivity metrics with threshold 0.5 (moderate edges)."""
    from cali.analysis._util import _compute_connectivity_metrics

    fov_analysis = FOVAnalysis(
        id=1,
        calcium_dec_dff_corr_matrix=[
            [1.0, 0.9, 0.5, 0.2],
            [0.9, 1.0, 0.6, 0.3],
            [0.5, 0.6, 1.0, 0.4],
            [0.2, 0.3, 0.4, 1.0],
        ],
        active_roi_labels=[0, 1, 2, 3],
    )

    # Threshold 0.5 - moderate edges
    adjacency, _, _ = _compute_connectivity_metrics(
        fov_analysis, method="calcium_dec_dff_corr", threshold=0.5
    )
    assert adjacency[0, 1] == 1  # 0.9 >= 0.5
    assert adjacency[1, 2] == 1  # 0.6 >= 0.5
    assert adjacency[0, 3] == 0  # 0.2 < 0.5


def test_compute_connectivity_metrics_threshold_0_9() -> None:
    """Test connectivity metrics with threshold 0.9 (strongest edges)."""
    from cali.analysis._util import _compute_connectivity_metrics

    fov_analysis = FOVAnalysis(
        id=1,
        calcium_peaks_max_lag_correlation_matrix=[
            [1.0, 0.9, 0.5, 0.2],
            [0.9, 1.0, 0.6, 0.3],
            [0.5, 0.6, 1.0, 0.4],
            [0.2, 0.3, 0.4, 1.0],
        ],
        active_roi_labels=[0, 1, 2, 3],
    )

    # Threshold 0.9 - only strongest edges
    adjacency, _, _ = _compute_connectivity_metrics(
        fov_analysis, method="calcium_peaks_maxlag", threshold=0.9
    )
    assert adjacency[0, 1] == 1  # 0.9 >= 0.9
    assert adjacency[1, 0] == 1  # Symmetric
    assert adjacency.sum() == 2  # Only one edge (bidirectional)


def test_connectivity_plot_wrapper_exists() -> None:
    """Test that connectivity plot wrapper function exists and is importable."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _plot_connectivity_network_data,
    )

    assert _plot_connectivity_network_data is not None
    assert callable(_plot_connectivity_network_data)


def test_get_fov_analysis_from_db_exists() -> None:
    """Test that get_fov_analysis_from_db function exists."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        get_fov_analysis_from_db,
    )

    assert get_fov_analysis_from_db is not None
    assert callable(get_fov_analysis_from_db)


def test_plot_connectivity_wrapper_function_signature() -> None:
    """Test that wrapper function has correct signature."""
    from inspect import signature

    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _plot_connectivity_network_data,
    )

    sig = signature(_plot_connectivity_network_data)
    params = list(sig.parameters.keys())

    # Check required parameters exist
    assert "widget" in params
    assert "engine" in params
    assert "fov_name" in params


def test_compute_connectivity_different_methods() -> None:
    """Test computing connectivity with different correlation methods."""
    from cali.analysis._util import _compute_connectivity_metrics

    # Create FOVAnalysis with multiple correlation matrices
    fov_analysis = FOVAnalysis(
        id=1,
        calcium_dff_correlation_matrix=[[1.0, 0.8], [0.8, 1.0]],
        calcium_dec_dff_corr_matrix=[[1.0, 0.7], [0.7, 1.0]],
        calcium_peaks_max_lag_correlation_matrix=[[1.0, 0.9], [0.9, 1.0]],
        active_roi_labels=[0, 1],
    )

    # Test each method
    for method in [
        "calcium_dff_corr",
        "calcium_dec_dff_corr",
        "calcium_peaks_maxlag",
    ]:
        adjacency, weights, roi_labels = _compute_connectivity_metrics(
            fov_analysis,
            method=method,
            threshold=0.5,  # type: ignore[arg-type]
        )

        # All should produce adjacency matrices with edges
        assert adjacency.shape == (2, 2)
        assert weights.shape == (2, 2)
        assert len(roi_labels) == 2


def test_filter_connectivity_by_rois() -> None:
    """Test filtering connectivity matrices by selected ROIs."""
    from cali.plot._single_wells_plots.correlation._plot_connectivity import (
        _filter_connectivity_by_rois,
    )

    # Create test data for 4 ROIs
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

    # Test 1: No filtering (None)
    filtered_adj, filtered_weights, filtered_labels = _filter_connectivity_by_rois(
        adjacency, weights, roi_labels, None
    )
    assert filtered_adj.shape == (4, 4)
    assert filtered_weights.shape == (4, 4)
    assert filtered_labels == [0, 1, 2, 3]

    # Test 2: Filter to subset [1, 2]
    filtered_adj, filtered_weights, filtered_labels = _filter_connectivity_by_rois(
        adjacency, weights, roi_labels, [1, 2]
    )
    assert filtered_adj.shape == (2, 2)
    assert filtered_weights.shape == (2, 2)
    assert filtered_labels == [1, 2]
    # Check values are correctly extracted
    assert filtered_adj[0, 1] == 1  # ROI 1->2 connection
    assert filtered_weights[0, 1] == 0.6

    # Test 3: Too few ROIs selected (returns original)
    filtered_adj, filtered_weights, filtered_labels = _filter_connectivity_by_rois(
        adjacency, weights, roi_labels, [1]
    )
    assert filtered_adj.shape == (4, 4)
    assert filtered_labels == [0, 1, 2, 3]
