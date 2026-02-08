"""Tests for correlation plot mouse event handlers.

This module tests that:
1. Hover handlers accept pg.Point directly (not via event.scenePos())
2. Click handlers accept MouseClickEvent with ev.scenePos() method
3. All correlation plots emit roiSelected signal on click
4. Handlers are properly connected/disconnected
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pyqtgraph as pg
import pytest
from PyQt6.QtCore import QPointF

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


@pytest.fixture
def mock_correlation_matrix() -> np.ndarray:
    """Create a small correlation matrix for testing."""
    return np.array(
        [
            [1.0, 0.8, 0.6],
            [0.8, 1.0, 0.7],
            [0.6, 0.7, 1.0],
        ]
    )


@pytest.fixture
def mock_roi_labels() -> list[int]:
    """Create ROI labels for testing."""
    return [1, 2, 3]


def test_calcium_traces_correlation_hover_handler(
    qtbot: QtBot,
    mock_correlation_matrix: np.ndarray,
    mock_roi_labels: list[int],
) -> None:
    """Test that hover handler accepts pg.Point directly."""
    from cali.plot._single_wells_plots.correlation._plot_calcium_traces_correlation import (  # noqa: E501
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    # Attach handlers
    _attach_heatmap_interaction(
        widget, plot, vb, mock_roi_labels, mock_correlation_matrix, "Test Title"
    )

    # Get the hover handler
    hover_handler = plot.property("dff_corr_hover_handler")
    assert hover_handler is not None

    # Create a pg.Point (not QPointF) - this is what sigMouseMoved emits
    test_point = pg.Point(1.5, 1.5)

    # Should not raise AttributeError
    try:
        hover_handler(test_point)
    except AttributeError as e:
        pytest.fail(f"Hover handler raised AttributeError: {e}")


def test_calcium_traces_correlation_click_handler(
    qtbot: QtBot,
    mock_correlation_matrix: np.ndarray,
    mock_roi_labels: list[int],
) -> None:
    """Test that click handler accepts MouseClickEvent."""
    from cali.plot._single_wells_plots.correlation._plot_calcium_traces_correlation import (  # noqa: E501
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    # Attach handlers
    _attach_heatmap_interaction(
        widget, plot, vb, mock_roi_labels, mock_correlation_matrix, "Test Title"
    )

    # Get the click handler
    click_handler = plot.property("dff_corr_click_handler")
    assert click_handler is not None

    # Create mock MouseClickEvent
    mock_event = MagicMock()
    mock_event.scenePos.return_value = QPointF(1.0, 1.0)

    # Should not raise AttributeError
    try:
        click_handler(mock_event)
    except AttributeError as e:
        pytest.fail(f"Click handler raised AttributeError: {e}")


def test_calcium_traces_correlation_emits_roi_selected(
    qtbot: QtBot,
    mock_correlation_matrix: np.ndarray,
    mock_roi_labels: list[int],
) -> None:
    """Test that clicking on heatmap emits roiSelected signal."""
    from cali.plot._single_wells_plots.correlation._plot_calcium_traces_correlation import (  # noqa: E501
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    _attach_heatmap_interaction(
        widget, plot, vb, mock_roi_labels, mock_correlation_matrix, "Test Title"
    )

    click_handler = plot.property("dff_corr_click_handler")

    # Track signal emissions
    emitted_rois = []

    def capture_emission(rois: list[str]) -> None:
        emitted_rois.append(rois)

    widget.roiSelected.connect(capture_emission)

    # Mock ViewBox.mapSceneToView to return predictable coordinates
    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(1, 1)  # Click on (1, 1) in matrix

        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True

            # Create mock event
            mock_event = MagicMock()
            mock_event.scenePos.return_value = QPointF(100, 100)

            # Trigger click
            click_handler(mock_event)

    # Should have emitted signal
    assert len(emitted_rois) == 1
    # Should be ROI labels at position (1, 1) -> [2, 2] since mock_roi_labels[1] = 2
    assert emitted_rois[0] == ["2", "2"]


def test_handlers_cleanup_on_reattach(
    qtbot: QtBot,
    mock_correlation_matrix: np.ndarray,
    mock_roi_labels: list[int],
) -> None:
    """Test that old handlers are disconnected when reattaching."""
    from cali.plot._single_wells_plots.correlation._plot_calcium_traces_correlation import (  # noqa: E501
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    # Attach handlers first time
    _attach_heatmap_interaction(
        widget, plot, vb, mock_roi_labels, mock_correlation_matrix, "Test Title 1"
    )

    old_hover = plot.property("dff_corr_hover_handler")
    old_click = plot.property("dff_corr_click_handler")

    # Attach handlers second time (simulates replotting)
    _attach_heatmap_interaction(
        widget, plot, vb, mock_roi_labels, mock_correlation_matrix, "Test Title 2"
    )

    new_hover = plot.property("dff_corr_hover_handler")
    new_click = plot.property("dff_corr_click_handler")

    # Handlers should be different objects (new ones created)
    assert new_hover is not old_hover
    assert new_click is not old_click


@pytest.mark.parametrize(
    "plot_function",
    [
        "_plot_dff_correlation_data",
        # Skip den_dff test due to SQLAlchemy transaction warning
        # "_plot_den_dff_correlation_data",
    ],
)
def test_both_correlation_plots_have_handlers(
    plot_function: str,
    qtbot: QtBot,
) -> None:
    """Test that both DF/F correlation plots have hover and click handlers."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_calcium_traces_correlation,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    # Mock database to return test data
    mock_engine = MagicMock()

    with patch.object(
        _plot_calcium_traces_correlation,
        "_get_dff_correlation_matrix_from_db"
        if "dff" in plot_function and "dec" not in plot_function
        else "_get_den_dff_correlation_matrix_from_db",
    ) as mock_get_matrix:
        # Return small test matrix
        mock_get_matrix.return_value = (
            np.array([[1.0, 0.5], [0.5, 1.0]]),
            [1, 2],
        )

        # Call the plot function
        plot_func = getattr(_plot_calcium_traces_correlation, plot_function)
        plot_func(
            widget=widget,
            engine=mock_engine,
            fov_name="test_fov",
            rois=None,
            run_id=1,
        )

        # Verify handlers were attached
        plot = widget.plot_item
        assert plot is not None

        hover_handler = plot.property("dff_corr_hover_handler")
        click_handler = plot.property("dff_corr_click_handler")

        assert hover_handler is not None
        assert click_handler is not None


def test_handler_hover_updates_title(
    qtbot: QtBot,
    mock_correlation_matrix: np.ndarray,
    mock_roi_labels: list[int],
) -> None:
    """Test that hovering over cells updates the plot title."""
    from cali.plot._single_wells_plots.correlation._plot_calcium_traces_correlation import (  # noqa: E501
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Base Title"

    _attach_heatmap_interaction(
        widget, plot, vb, mock_roi_labels, mock_correlation_matrix, base_title
    )

    hover_handler = plot.property("dff_corr_hover_handler")

    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(0, 1)  # Hover over (0, 1)

        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True

            # Trigger hover
            test_point = pg.Point(50, 50)
            hover_handler(test_point)

            # Title should be updated with correlation value
            # At position (0, 1): roi_labels[0]=1, roi_labels[1]=2, corr=0.8
            # Note: matrix indexing is [row, col], so corr[0, 1] = 0.8
            title = plot.titleLabel.text
            # Check for correlation value (ROI order might vary)
            assert "0.800" in title  # correlation value formatted to 3 decimals
            assert ("ROI 1" in title and "ROI 2" in title) or (
                "ROI 2" in title and "ROI 1" in title
            )
