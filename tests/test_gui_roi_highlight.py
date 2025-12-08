"""Tests for CaliGui ROI highlighting integration.

Tests the integration between roiSelected signals and image viewer highlighting,
specifically testing the dual-color highlighting from connectivity plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def mock_image_viewer() -> MagicMock:
    """Create a mock image viewer with the necessary methods."""
    viewer = MagicMock()
    viewer._roi_number_le = MagicMock()
    viewer._roi_number_le.setText = MagicMock()
    viewer._highlight_rois = MagicMock()
    return viewer


def test_highlight_roi_with_single_string(
    mock_image_viewer: MagicMock, qtbot: QtBot
) -> None:
    """Test highlighting a single ROI passed as a string."""
    from cali.gui._cali_gui import CaliGui

    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._image_viewer = mock_image_viewer

    gui._highlight_roi("5")

    mock_image_viewer._roi_number_le.setText.assert_called_once_with("5")
    mock_image_viewer._highlight_rois.assert_called_once_with(5)


def test_highlight_roi_with_single_item_list(
    mock_image_viewer: MagicMock, qtbot: QtBot
) -> None:
    """Test highlighting when list has only one ROI (selected, no connections)."""
    from cali.gui._cali_gui import CaliGui

    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._image_viewer = mock_image_viewer

    gui._highlight_roi(["3"])

    mock_image_viewer._roi_number_le.setText.assert_called_once_with("3")
    mock_image_viewer._highlight_rois.assert_called_once_with(3, connected_rois=None)


def test_highlight_roi_with_empty_list(
    mock_image_viewer: MagicMock, qtbot: QtBot
) -> None:
    """Test that empty list is handled gracefully."""
    from cali.gui._cali_gui import CaliGui

    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._image_viewer = mock_image_viewer

    gui._highlight_roi([])

    # Should not call any methods on empty list
    mock_image_viewer._roi_number_le.setText.assert_not_called()
    mock_image_viewer._highlight_rois.assert_not_called()


def test_highlight_roi_with_selected_and_one_connected(
    mock_image_viewer: MagicMock,
    qtbot: QtBot,
) -> None:
    """Test highlighting selected ROI + 1 connected ROI (from connectivity plot)."""
    from cali.gui._cali_gui import CaliGui

    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._image_viewer = mock_image_viewer

    # First ROI is selected (green), second is connected (yellow)
    gui._highlight_roi(["1", "2"])

    mock_image_viewer._roi_number_le.setText.assert_called_once_with("1")
    mock_image_viewer._highlight_rois.assert_called_once_with(1, connected_rois=[2])


def test_highlight_roi_with_multiple_connected(
    mock_image_viewer: MagicMock, qtbot: QtBot
) -> None:
    """Test highlighting selected ROI + multiple connected ROIs."""
    from cali.gui._cali_gui import CaliGui

    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._image_viewer = mock_image_viewer

    # First is selected, rest are connected
    gui._highlight_roi(["5", "1", "2", "3"])

    mock_image_viewer._roi_number_le.setText.assert_called_once_with("5")
    mock_image_viewer._highlight_rois.assert_called_once_with(
        5, connected_rois=[1, 2, 3]
    )


@pytest.mark.parametrize(
    ("input_roi", "expected_text", "expected_roi", "expected_connected"),
    [
        ("10", "10", 10, None),  # String input
        (["7"], "7", 7, None),  # Single item list
        (["3", "4"], "3", 3, [4]),  # Two items (selected + 1 connected)
        (
            ["1", "2", "3", "4"],
            "1",
            1,
            [2, 3, 4],
        ),  # Multiple connected
        (
            ["8", "9", "10", "11", "12"],
            "8",
            8,
            [9, 10, 11, 12],
        ),  # Many connected
    ],
)
def test_highlight_roi_parametrized(
    mock_image_viewer: MagicMock,
    qtbot: QtBot,
    input_roi: str | list[str],
    expected_text: str,
    expected_roi: int,
    expected_connected: list[int] | None,
) -> None:
    """Test various input formats for _highlight_roi."""
    from cali.gui._cali_gui import CaliGui

    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._image_viewer = mock_image_viewer

    gui._highlight_roi(input_roi)

    mock_image_viewer._roi_number_le.setText.assert_called_once_with(expected_text)

    if expected_connected is None:
        if isinstance(input_roi, str):
            mock_image_viewer._highlight_rois.assert_called_once_with(expected_roi)
        else:
            mock_image_viewer._highlight_rois.assert_called_once_with(
                expected_roi, connected_rois=None
            )
    else:
        mock_image_viewer._highlight_rois.assert_called_once_with(
            expected_roi, connected_rois=expected_connected
        )


def test_highlight_roi_integration_with_connectivity_plot(qtbot: QtBot) -> None:
    """Test that connectivity plot signal properly triggers dual highlighting.

    This simulates the full workflow:
    1. Connectivity plot emits roiSelected with [selected, neighbor1, neighbor2, ...]
    2. CaliGui._highlight_roi receives the list
    3. Image viewer highlights selected (green) and neighbors (yellow)
    """
    from cali.gui._cali_gui import CaliGui

    gui = CaliGui()
    qtbot.addWidget(gui)

    # Mock the image viewer
    mock_viewer = MagicMock()
    gui._image_viewer = mock_viewer

    # Simulate connectivity plot emitting signal
    # ROI 5 is selected, ROIs 2 and 8 are connected
    gui._highlight_roi(["5", "2", "8"])

    # Verify correct calls
    mock_viewer._roi_number_le.setText.assert_called_once_with("5")
    mock_viewer._highlight_rois.assert_called_once_with(5, connected_rois=[2, 8])
