"""Tests for roiSelected signal emission across all correlation plots.

This module tests that all correlation plot click handlers emit roiSelected signal:
1. Calcium traces correlation (DF/F and denoised DF/F)
2. Calcium peaks synchrony
3. Inferred spike synchrony
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
def test_matrix() -> np.ndarray:
    """Small test matrix."""
    return np.array([[1.0, 0.8], [0.8, 1.0]])


@pytest.fixture
def test_roi_labels() -> list[int]:
    """Test ROI labels."""
    return [5, 10]


def test_inferred_spike_synchrony_emits_roi_selected(
    qtbot: QtBot, test_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test inferred spike synchrony plot emits roiSelected on click."""
    from cali.plot._single_wells_plots.correlation._plot_inferred_spike_synchrony import (  # noqa: E501
        _plot_spike_synchrony_data,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    mock_engine = MagicMock()

    with patch(
        "cali.plot._single_wells_plots.correlation._plot_inferred_spike_synchrony."
        "_get_spike_synchrony_matrix_from_db"
    ) as mock_get:
        mock_get.return_value = (
            test_matrix,
            test_roi_labels,
            0.6,  # global_synchrony
            300.0,  # jitter_window_ms
        )

        _plot_spike_synchrony_data(
            widget=widget,
            engine=mock_engine,
            fov_name="test",
            rois=None,
            run_id=1,
        )

        # Track signal emissions
        emitted_rois = []
        widget.roiSelected.connect(lambda rois: emitted_rois.append(rois))

        # Get click handler
        plot = widget.plot_item
        assert plot is not None
        click_handler = plot.property("spike_sync_click_handler")
        assert click_handler is not None

        # Mock scene position -> view position mapping
        vb = plot.getViewBox()
        with patch.object(vb, "mapSceneToView") as mock_map:
            mock_map.return_value = pg.Point(1, 0)  # Click at (1, 0)

            with patch.object(plot, "sceneBoundingRect") as mock_rect:
                mock_rect.return_value.contains.return_value = True

                # Simulate click event
                mock_event = MagicMock()
                mock_event.scenePos.return_value = QPointF(100, 100)

                click_handler(mock_event)

        # Should emit signal with ROIs at position (1, 0)
        assert len(emitted_rois) == 1
        # Point(1, 0) = col=1, row=0 -> roi_labels[0], roi_labels[1]
        assert emitted_rois[0] == ["5", "10"]


@pytest.mark.parametrize(
    "plot_module,plot_function,db_function,handler_name",
    [
        (
            "_plot_calcium_traces_correlation",
            "_plot_dff_correlation_data",
            "_get_dff_correlation_matrix_from_db",
            "dff_corr_click_handler",
        ),
        (
            "_plot_calcium_traces_correlation",
            "_plot_den_dff_correlation_data",
            "_get_den_dff_correlation_matrix_from_db",
            "dff_corr_click_handler",
        ),
    ],
)
def test_all_correlation_plots_emit_signal(
    qtbot: QtBot,
    test_matrix: np.ndarray,
    test_roi_labels: list[int],
    plot_module: str,
    plot_function: str,
    db_function: str,
    handler_name: str,
) -> None:
    """Parametrized test to verify all correlation plots emit roiSelected."""
    # Import the module dynamically
    import importlib

    module = importlib.import_module(
        f"cali.plot._single_wells_plots.correlation.{plot_module}"
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    mock_engine = MagicMock()

    with patch.object(module, db_function) as mock_get:
        mock_get.return_value = (test_matrix, test_roi_labels)

        # Call plot function
        func = getattr(module, plot_function)
        func(
            widget=widget,
            engine=mock_engine,
            fov_name="test",
            rois=None,
            run_id=1,
        )

        # Track emissions
        emitted = []
        widget.roiSelected.connect(lambda r: emitted.append(r))

        # Get handler
        plot = widget.plot_item
        assert plot is not None
        handler = plot.property(handler_name)
        assert handler is not None

        # Simulate click
        vb = plot.getViewBox()
        with patch.object(vb, "mapSceneToView") as mock_map:
            mock_map.return_value = pg.Point(0, 0)

            event = MagicMock()
            event.scenePos.return_value = QPointF(25, 25)
            handler(event)

        # Verify signal was emitted
        assert len(emitted) == 1
        assert emitted[0] == ["5", "5"]  # Position (0,0) -> roi_labels[0]


def test_clicking_outside_matrix_does_not_emit(
    qtbot: QtBot, test_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test that clicking outside matrix bounds doesn't emit signal."""
    from cali.plot._single_wells_plots.correlation._plot_calcium_traces_correlation import (  # noqa: E501
        _plot_dff_correlation_data,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    mock_engine = MagicMock()

    with patch(
        "cali.plot._single_wells_plots.correlation._plot_calcium_traces_correlation."
        "_get_dff_correlation_matrix_from_db"
    ) as mock_get:
        mock_get.return_value = (test_matrix, test_roi_labels)

        _plot_dff_correlation_data(
            widget=widget,
            engine=mock_engine,
            fov_name="test",
            rois=None,
            run_id=1,
        )

        emitted = []
        widget.roiSelected.connect(lambda r: emitted.append(r))

        plot = widget.plot_item
        assert plot is not None
        handler = plot.property("dff_corr_click_handler")

        vb = plot.getViewBox()

        # Click outside bounds (matrix is 2x2, click at (5, 5))
        with patch.object(vb, "mapSceneToView") as mock_map:
            mock_map.return_value = pg.Point(5, 5)

            event = MagicMock()
            event.scenePos.return_value = QPointF(200, 200)
            handler(event)

        # Should NOT emit signal
        assert len(emitted) == 0
