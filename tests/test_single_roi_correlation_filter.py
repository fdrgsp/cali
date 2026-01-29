"""Test that correlation plots correctly handle single ROI selection.

This test ensures that when a user selects a single ROI in correlation heatmap
plots, no plot is displayed (Need ≥2 ROIs message), rather than showing all ROIs.
"""

from pathlib import Path
from unittest.mock import MagicMock


def test_single_roi_returns_empty_matrix_calcium_correlation(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that selecting a single ROI in calcium correlation returns empty matrix."""
    import numpy as np

    from cali.plot._single_wells_plots.correlation._plot_calcium_traces_correlation import (  # noqa: E501
        _filter_matrix_by_rois,
    )

    # Create a 3x3 correlation matrix
    matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    roi_labels = [1, 2, 3]

    # Test selecting single ROI - should return empty matrix
    filtered_matrix, filtered_labels = _filter_matrix_by_rois(
        matrix, roi_labels, selected_rois=[1]
    )

    assert filtered_matrix.size == 0, "Single ROI selection should return empty matrix"
    assert len(filtered_labels) == 1, (
        "Single ROI selection should return that ROI label"
    )
    assert filtered_labels == [1]


def test_single_roi_returns_empty_matrix_spike_correlation(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that selecting a single ROI in spike correlation returns empty matrix."""
    import numpy as np

    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_correlation import (  # noqa: E501
        _filter_matrix_by_rois,
    )

    # Create a 3x3 correlation matrix
    matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
    roi_labels = [1, 2, 3]

    # Test selecting single ROI - should return empty matrix
    filtered_matrix, filtered_labels = _filter_matrix_by_rois(
        matrix, roi_labels, selected_rois=[1]
    )

    assert filtered_matrix.size == 0, "Single ROI selection should return empty matrix"
    assert len(filtered_labels) == 1, (
        "Single ROI selection should return that ROI label"
    )
    assert filtered_labels == [1]


def test_single_roi_returns_empty_matrix_spike_lag_values(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that selecting a single ROI in spike lag values returns empty matrix."""
    import numpy as np

    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_values import (
        _filter_matrix_by_rois,
    )

    # Create a 3x3 lag values matrix
    matrix = np.array([[0, 2, -1], [-2, 0, 3], [1, -3, 0]])
    roi_labels = [1, 2, 3]

    # Test selecting single ROI - should return empty matrix
    filtered_matrix, filtered_labels = _filter_matrix_by_rois(
        matrix, roi_labels, selected_rois=[1]
    )

    assert filtered_matrix.size == 0, "Single ROI selection should return empty matrix"
    assert len(filtered_labels) == 1, (
        "Single ROI selection should return that ROI label"
    )
    assert filtered_labels == [1]


def test_single_roi_returns_empty_matrix_spike_synchrony(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that selecting a single ROI in spike synchrony returns empty matrix."""
    import numpy as np

    from cali.plot._single_wells_plots.correlation._plot_inferred_spike_synchrony import (  # noqa: E501
        _filter_matrix_by_rois,
    )

    # Create a 3x3 synchrony matrix
    matrix = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.5], [0.2, 0.5, 1.0]])
    roi_labels = [1, 2, 3]

    # Test selecting single ROI - should return empty matrix
    filtered_matrix, filtered_labels = _filter_matrix_by_rois(
        matrix, roi_labels, selected_rois=[1]
    )

    assert filtered_matrix.size == 0, "Single ROI selection should return empty matrix"
    assert len(filtered_labels) == 1, (
        "Single ROI selection should return that ROI label"
    )
    assert filtered_labels == [1]


def test_two_rois_returns_filtered_matrix(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that selecting two ROIs correctly filters the matrix."""
    import numpy as np

    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_correlation import (  # noqa: E501
        _filter_matrix_by_rois,
    )

    # Create a 4x4 correlation matrix
    matrix = np.array(
        [
            [1.0, 0.5, 0.3, 0.2],
            [0.5, 1.0, 0.4, 0.6],
            [0.3, 0.4, 1.0, 0.7],
            [0.2, 0.6, 0.7, 1.0],
        ]
    )
    roi_labels = [1, 2, 3, 4]

    # Test selecting two ROIs - should return 2x2 matrix
    filtered_matrix, filtered_labels = _filter_matrix_by_rois(
        matrix, roi_labels, selected_rois=[2, 4]
    )

    assert filtered_matrix.shape == (2, 2), "Two ROI selection should return 2x2 matrix"
    assert filtered_labels == [2, 4]
    # Check that the values are from the correct positions
    # ROI 2 is index 1, ROI 4 is index 3
    assert filtered_matrix[0, 0] == 1.0  # ROI 2 self-correlation
    assert filtered_matrix[1, 1] == 1.0  # ROI 4 self-correlation
    assert filtered_matrix[0, 1] == 0.6  # ROI 2 vs ROI 4
    assert filtered_matrix[1, 0] == 0.6  # ROI 4 vs ROI 2


def test_zero_rois_returns_empty_matrix(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that selecting zero ROIs (empty list) returns empty matrix."""
    import numpy as np

    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_correlation import (  # noqa: E501
        _filter_matrix_by_rois,
    )

    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    roi_labels = [1, 2]

    # Test selecting no ROIs - should return empty matrix
    filtered_matrix, filtered_labels = _filter_matrix_by_rois(
        matrix, roi_labels, selected_rois=[]
    )

    assert filtered_matrix.size == 0, "Zero ROI selection should return empty matrix"
    assert filtered_labels == []


def test_none_rois_returns_full_matrix(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that None for selected ROIs returns the full matrix."""
    import numpy as np

    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_correlation import (  # noqa: E501
        _filter_matrix_by_rois,
    )

    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    roi_labels = [1, 2]

    # Test None - should return full matrix
    filtered_matrix, filtered_labels = _filter_matrix_by_rois(
        matrix, roi_labels, selected_rois=None
    )

    assert filtered_matrix.shape == (2, 2)
    assert filtered_labels == [1, 2]
    assert np.array_equal(filtered_matrix, matrix)
