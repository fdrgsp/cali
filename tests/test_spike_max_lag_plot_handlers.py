"""Tests for spike max-lag correlation/values plot handlers.

Tests hover and click handlers for:
1. Spike max-lag correlation heatmaps
2. Spike max-lag values heatmaps (with directional lag display)
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
def test_corr_matrix() -> np.ndarray:
    """Small test correlation matrix."""
    return np.array([[1.0, 0.8, 0.5], [0.8, 1.0, 0.6], [0.5, 0.6, 1.0]], dtype=float)


@pytest.fixture
def test_lag_matrix() -> np.ndarray:
    """Small test lag matrix with positive, negative, and zero values."""
    return np.array([[0, 3, -2], [-3, 0, 5], [2, -5, 0]], dtype=int)


@pytest.fixture
def test_roi_labels() -> list[int]:
    """Test ROI labels (non-sequential to test mapping)."""
    return [5, 12, 20]


# -----------------------------------------------------------------------------
# Spike Max-Lag Correlation Handler Tests
# -----------------------------------------------------------------------------
def test_spike_maxlag_corr_hover_handler(
    qtbot: QtBot, test_corr_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test hover handler for spike max-lag correlation plot."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_spike_max_lag_correlation,
    )

    _attach_heatmap_interaction = (
        _plot_spike_max_lag_correlation._attach_heatmap_interaction
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Spike Max-Lag Correlation"
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_corr_matrix, base_title
    )

    hover_handler = plot.property("spike_maxlag_hover_handler")
    assert hover_handler is not None

    # Test hover inside bounds
    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(1, 0)  # col=1, row=0
        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True
            hover_handler(pg.Point(50, 50))

            title = plot.titleLabel.text
            assert "ROI 5" in title
            assert "ROI 12" in title
            assert "0.800" in title


def test_spike_maxlag_corr_hover_outside_bounds(
    qtbot: QtBot, test_corr_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test hover handler resets title when outside bounds."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_spike_max_lag_correlation,
    )

    _attach_heatmap_interaction = (
        _plot_spike_max_lag_correlation._attach_heatmap_interaction
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Spike Max-Lag Correlation"
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_corr_matrix, base_title
    )

    hover_handler = plot.property("spike_maxlag_hover_handler")

    # Test hover outside scene bounds
    with patch.object(plot, "sceneBoundingRect") as mock_rect:
        mock_rect.return_value.contains.return_value = False
        hover_handler(pg.Point(50, 50))

        assert plot.titleLabel.text == base_title


def test_spike_maxlag_corr_hover_outside_matrix(
    qtbot: QtBot, test_corr_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test hover handler resets title when outside matrix indices."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_spike_max_lag_correlation,
    )

    _attach_heatmap_interaction = (
        _plot_spike_max_lag_correlation._attach_heatmap_interaction
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Spike Max-Lag Correlation"
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_corr_matrix, base_title
    )

    hover_handler = plot.property("spike_maxlag_hover_handler")

    # Test hover at invalid matrix position
    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(10, 10)  # Outside 3x3 matrix
        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True
            hover_handler(pg.Point(50, 50))

            assert plot.titleLabel.text == base_title


def test_spike_maxlag_corr_click_emits_roi_labels(
    qtbot: QtBot, test_corr_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test click handler emits correct ROI labels."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_spike_max_lag_correlation,
    )

    _attach_heatmap_interaction = (
        _plot_spike_max_lag_correlation._attach_heatmap_interaction
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_corr_matrix, "Test"
    )

    emitted = []
    widget.roiSelected.connect(lambda r: emitted.append(r))

    click_handler = plot.property("spike_maxlag_click_handler")
    assert click_handler is not None

    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(2, 1)  # col=2, row=1
        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True
            mock_event = MagicMock()
            mock_event.scenePos.return_value = QPointF(10, 20)
            click_handler(mock_event)

    assert len(emitted) == 1
    # row=1 -> ROI 12, col=2 -> ROI 20
    assert emitted[0] == ["12", "20"]


def test_spike_maxlag_corr_click_outside_bounds(
    qtbot: QtBot, test_corr_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test click handler ignores clicks outside bounds."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_spike_max_lag_correlation,
    )

    _attach_heatmap_interaction = (
        _plot_spike_max_lag_correlation._attach_heatmap_interaction
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_corr_matrix, "Test"
    )

    emitted = []
    widget.roiSelected.connect(lambda r: emitted.append(r))

    click_handler = plot.property("spike_maxlag_click_handler")

    with patch.object(plot, "sceneBoundingRect") as mock_rect:
        mock_rect.return_value.contains.return_value = False
        mock_event = MagicMock()
        mock_event.scenePos.return_value = QPointF(10, 20)
        click_handler(mock_event)

    assert len(emitted) == 0


def test_spike_maxlag_corr_handlers_cleanup_on_reattach(
    qtbot: QtBot, test_corr_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test handlers are disconnected when reattaching."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_spike_max_lag_correlation,
    )

    _attach_heatmap_interaction = (
        _plot_spike_max_lag_correlation._attach_heatmap_interaction
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    # Attach first time
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_corr_matrix, "First"
    )
    old_hover = plot.property("spike_maxlag_hover_handler")

    # Attach second time
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_corr_matrix, "Second"
    )
    new_hover = plot.property("spike_maxlag_hover_handler")

    assert new_hover is not old_hover


# -----------------------------------------------------------------------------
# Spike Max-Lag Values Handler Tests
# -----------------------------------------------------------------------------
def test_spike_maxlag_values_hover_positive_lag(
    qtbot: QtBot, test_lag_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test hover shows correct message for positive lag (j lags)."""
    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_values import (
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Max-Lag Values"
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_lag_matrix, base_title
    )

    hover_handler = plot.property("spike_maxlag_values_hover_handler")
    assert hover_handler is not None

    # Position (0, 1) has lag=3 (positive)
    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(1, 0)  # col=1, row=0
        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True
            hover_handler(pg.Point(50, 50))

            title = plot.titleLabel.text
            assert "ROI 5" in title
            assert "ROI 12" in title
            assert "+3" in title
            assert "lags" in title


def test_spike_maxlag_values_hover_negative_lag(
    qtbot: QtBot, test_lag_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test hover shows correct message for negative lag (j leads)."""
    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_values import (
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Max-Lag Values"
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_lag_matrix, base_title
    )

    hover_handler = plot.property("spike_maxlag_values_hover_handler")

    # Position (0, 2) has lag=-2 (negative)
    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(2, 0)  # col=2, row=0
        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True
            hover_handler(pg.Point(50, 50))

            title = plot.titleLabel.text
            assert "ROI 5" in title
            assert "ROI 20" in title
            assert "-2" in title
            assert "leads" in title


def test_spike_maxlag_values_hover_zero_lag(
    qtbot: QtBot, test_lag_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test hover shows correct message for zero lag (sync)."""
    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_values import (
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Max-Lag Values"
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_lag_matrix, base_title
    )

    hover_handler = plot.property("spike_maxlag_values_hover_handler")

    # Diagonal positions have lag=0
    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(0, 0)  # col=0, row=0 (diagonal)
        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True
            hover_handler(pg.Point(50, 50))

            title = plot.titleLabel.text
            assert "sync" in title


def test_spike_maxlag_values_hover_outside_matrix(
    qtbot: QtBot, test_lag_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test hover resets title when outside matrix bounds."""
    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_values import (
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Max-Lag Values"
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_lag_matrix, base_title
    )

    hover_handler = plot.property("spike_maxlag_values_hover_handler")

    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(10, 10)
        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True
            hover_handler(pg.Point(50, 50))

            assert plot.titleLabel.text == base_title


def test_spike_maxlag_values_click_emits_roi_labels(
    qtbot: QtBot, test_lag_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test click handler emits correct ROI labels."""
    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_values import (
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_lag_matrix, "Test"
    )

    emitted = []
    widget.roiSelected.connect(lambda r: emitted.append(r))

    click_handler = plot.property("spike_maxlag_values_click_handler")
    assert click_handler is not None

    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(0, 2)  # col=0, row=2
        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True
            mock_event = MagicMock()
            mock_event.scenePos.return_value = QPointF(10, 20)
            click_handler(mock_event)

    assert len(emitted) == 1
    # row=2 -> ROI 20, col=0 -> ROI 5
    assert emitted[0] == ["20", "5"]


def test_spike_maxlag_values_click_outside_bounds(
    qtbot: QtBot, test_lag_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test click ignores clicks outside bounds."""
    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_values import (
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_lag_matrix, "Test"
    )

    emitted = []
    widget.roiSelected.connect(lambda r: emitted.append(r))

    click_handler = plot.property("spike_maxlag_values_click_handler")

    with patch.object(plot, "sceneBoundingRect") as mock_rect:
        mock_rect.return_value.contains.return_value = False
        mock_event = MagicMock()
        mock_event.scenePos.return_value = QPointF(10, 20)
        click_handler(mock_event)

    assert len(emitted) == 0


def test_spike_maxlag_values_handlers_cleanup_on_reattach(
    qtbot: QtBot, test_lag_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test handlers are disconnected when reattaching."""
    from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_values import (
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    # Attach first time
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_lag_matrix, "First"
    )
    old_hover = plot.property("spike_maxlag_values_hover_handler")

    # Attach second time
    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_lag_matrix, "Second"
    )
    new_hover = plot.property("spike_maxlag_values_hover_handler")

    assert new_hover is not old_hover
