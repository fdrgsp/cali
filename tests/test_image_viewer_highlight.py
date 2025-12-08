"""Tests for image viewer ROI highlighting functionality.

Tests the dual-color highlighting system where:
- Green: Selected ROI(s)
- Yellow: Connected/correlated ROIs
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from cali.gui._image_viewer import _ImageViewer

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def image_viewer(qtbot: QtBot) -> _ImageViewer:
    """Create an image viewer widget for testing."""
    viewer = _ImageViewer()
    qtbot.addWidget(viewer)
    return viewer


@pytest.fixture
def sample_image_data() -> np.ndarray:
    """Create sample image data."""
    return np.random.rand(100, 100).astype(np.float32)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Create sample label mask with 5 ROIs."""
    labels = np.zeros((100, 100), dtype=np.uint8)
    # Create simple labeled regions
    labels[10:20, 10:20] = 1
    labels[30:40, 30:40] = 2
    labels[50:60, 50:60] = 3
    labels[70:80, 70:80] = 4
    labels[20:30, 60:70] = 5
    return labels


def test_image_viewer_initialization(image_viewer: _ImageViewer) -> None:
    """Test that image viewer initializes properly."""
    assert image_viewer._viewer.highlight_roi is None
    assert image_viewer._viewer.highlight_connected_roi is None


def test_set_data_with_labels(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test setting image data with labels."""
    image_viewer.setData(sample_image_data, sample_labels)

    assert image_viewer._viewer.image is not None
    assert image_viewer._viewer.labels_image is not None
    assert image_viewer._labels.isEnabled()


def test_highlight_single_roi(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test highlighting a single ROI (green only)."""
    image_viewer.setData(sample_image_data, sample_labels)

    # Highlight ROI 1
    image_viewer._highlight_rois(roi=1)

    # Should have green highlight, but no yellow
    assert image_viewer._viewer.highlight_roi is not None
    assert image_viewer._viewer.highlight_connected_roi is None


def test_highlight_with_connected_rois(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test highlighting selected ROI (green) and connected ROIs (yellow)."""
    image_viewer.setData(sample_image_data, sample_labels)

    # Highlight ROI 1 (green) with ROIs 2, 3 as connected (yellow)
    image_viewer._highlight_rois(roi=1, connected_rois=[2, 3])

    # Should have both green and yellow highlights
    assert image_viewer._viewer.highlight_roi is not None
    assert image_viewer._viewer.highlight_connected_roi is not None


def test_highlight_with_invalid_connected_rois(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test that invalid connected ROIs are filtered out."""
    image_viewer.setData(sample_image_data, sample_labels)

    # ROI 99 doesn't exist (max is 5)
    image_viewer._highlight_rois(roi=1, connected_rois=[2, 99])

    # Should still have yellow highlight for valid ROI 2
    assert image_viewer._viewer.highlight_roi is not None
    assert image_viewer._viewer.highlight_connected_roi is not None


def test_highlight_with_only_invalid_connected_rois(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test that no yellow highlight is created if all connected ROIs are invalid."""
    image_viewer.setData(sample_image_data, sample_labels)

    # All invalid ROIs
    image_viewer._highlight_rois(roi=1, connected_rois=[99, 100])

    # Should have green highlight, but no yellow
    assert image_viewer._viewer.highlight_roi is not None
    assert image_viewer._viewer.highlight_connected_roi is None


def test_clear_highlight_clears_both_layers(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test that clearing highlights removes both green and yellow layers."""
    image_viewer.setData(sample_image_data, sample_labels)

    # Create highlights
    image_viewer._highlight_rois(roi=1, connected_rois=[2, 3])
    assert image_viewer._viewer.highlight_roi is not None
    assert image_viewer._viewer.highlight_connected_roi is not None

    # Clear highlights
    image_viewer._clear_highlight()
    assert image_viewer._viewer.highlight_roi is None
    assert image_viewer._viewer.highlight_connected_roi is None
    assert image_viewer._roi_number_le.text() == ""


def test_highlight_replaces_previous_highlights(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test that new highlights replace previous ones."""
    image_viewer.setData(sample_image_data, sample_labels)

    # Create first highlight
    image_viewer._highlight_rois(roi=1, connected_rois=[2])
    first_green = image_viewer._viewer.highlight_roi
    first_yellow = image_viewer._viewer.highlight_connected_roi

    # Create second highlight
    image_viewer._highlight_rois(roi=3, connected_rois=[4, 5])
    second_green = image_viewer._viewer.highlight_roi
    second_yellow = image_viewer._viewer.highlight_connected_roi

    # Should be different objects
    assert first_green is not second_green
    assert first_yellow is not second_yellow


def test_highlight_without_labels_shows_error(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
) -> None:
    """Test that highlighting without labels shows an error."""
    # Set data without labels
    image_viewer.setData(sample_image_data, None)

    with patch("cali.gui._image_viewer.show_error_dialog") as mock_error:
        image_viewer._highlight_rois(roi=1)
        mock_error.assert_called_once()


def test_highlight_out_of_range_roi_shows_error(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test that highlighting ROI out of range shows an error."""
    image_viewer.setData(sample_image_data, sample_labels)

    with patch("cali.gui._image_viewer.show_error_dialog") as mock_error:
        image_viewer._highlight_rois(roi=99)  # Max label is 5
        mock_error.assert_called_once()


@pytest.mark.parametrize(
    ("roi", "connected", "expect_green", "expect_yellow"),
    [
        (1, None, True, False),  # Single ROI, no connections
        (1, [], True, False),  # Empty connected list
        (1, [2], True, True),  # Single connection
        (1, [2, 3, 4], True, True),  # Multiple connections
        (3, [1, 2, 4, 5], True, True),  # Many connections
    ],
)
def test_highlight_combinations(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
    roi: int,
    connected: list[int] | None,
    expect_green: bool,
    expect_yellow: bool,
) -> None:
    """Test various combinations of ROI highlighting."""
    image_viewer.setData(sample_image_data, sample_labels)

    image_viewer._highlight_rois(roi=roi, connected_rois=connected)

    if expect_green:
        assert image_viewer._viewer.highlight_roi is not None
    else:  # pragma: no cover
        assert image_viewer._viewer.highlight_roi is None

    if expect_yellow:
        assert image_viewer._viewer.highlight_connected_roi is not None
    else:
        assert image_viewer._viewer.highlight_connected_roi is None


def test_highlight_from_line_edit(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test highlighting ROIs from line edit input."""
    image_viewer.setData(sample_image_data, sample_labels)

    # Set text in line edit
    image_viewer._roi_number_le.setText("1,2,3")

    # Call without explicit ROI (should parse from line edit)
    image_viewer._highlight_rois()

    assert image_viewer._viewer.highlight_roi is not None


def test_show_labels_clears_highlights(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test that showing labels clears highlights."""
    image_viewer.setData(sample_image_data, sample_labels)

    # Create highlights
    image_viewer._highlight_rois(roi=1, connected_rois=[2])
    assert image_viewer._viewer.highlight_roi is not None

    # Show labels
    image_viewer._show_labels(True)

    # Highlights should be cleared
    assert image_viewer._viewer.highlight_roi is None
    assert image_viewer._viewer.highlight_connected_roi is None


def test_highlight_properties_set_correctly(
    image_viewer: _ImageViewer,
    sample_image_data: np.ndarray,
    sample_labels: np.ndarray,
) -> None:
    """Test that highlight images have correct properties."""
    image_viewer.setData(sample_image_data, sample_labels)

    image_viewer._highlight_rois(roi=1, connected_rois=[2])

    # Green highlight
    green = image_viewer._viewer.highlight_roi
    assert green is not None
    assert green.interactive is True

    # Yellow highlight
    yellow = image_viewer._viewer.highlight_connected_roi
    assert yellow is not None
    assert yellow.interactive is True
