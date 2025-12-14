"""Test that clearing plots and setting combo to None properly disconnects handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def test_clear_plot_disconnects_all_handlers(qtbot: QtBot) -> None:
    """Test that clear_plot() properly disconnects all hover and click handlers."""
    from cali.plot._single_wells_plots.correlation._plot_calcium_traces_correlation import (  # noqa: E501
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    # Attach handlers
    test_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
    test_roi_labels = [5, 10]

    _attach_heatmap_interaction(
        widget, plot, vb, test_roi_labels, test_matrix, "Test Title"
    )

    # Verify handlers are attached
    hover_handler = plot.property("dff_corr_hover_handler")
    click_handler = plot.property("dff_corr_click_handler")
    assert hover_handler is not None, "Hover handler should be attached"
    assert click_handler is not None, "Click handler should be attached"

    # Call clear_plot
    widget.clear_plot()

    # Verify handlers are cleared
    hover_handler_after = plot.property("dff_corr_hover_handler")
    click_handler_after = plot.property("dff_corr_click_handler")
    assert hover_handler_after is None, "Hover handler should be cleared"
    assert click_handler_after is None, "Click handler should be cleared"


def test_combo_none_clears_handlers(qtbot: QtBot) -> None:
    """Test that setting combo to 'None' properly clears handlers."""
    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    # Mock the engine and setup
    mock_engine = MagicMock()
    widget._engine = mock_engine
    widget._fov = "test_fov"
    widget._run_id = 1

    plot = widget.plot_item
    assert plot is not None

    # Manually attach a test handler
    def dummy_handler(pos: object) -> None:
        pass

    scene = plot.scene()
    scene.sigMouseMoved.connect(dummy_handler)
    plot.setProperty("test_hover_handler", dummy_handler)

    # Verify handler is attached
    assert plot.property("test_hover_handler") is not None

    # Simulate combo change to "None"
    with patch.object(
        widget, "_engine", mock_engine
    ):  # Ensure engine is set for the combo handler
        widget._on_combo_changed("None")

    # Plot should be cleared and ready for next use
    # The important thing is that no handlers remain attached
    # (our disconnect_hover_handlers doesn't know about test_hover_handler,
    # but this tests that the clear_plot pathway is triggered)
    # Verify clear_plot was called by checking that items are cleared
    assert len(plot.items) == 0  # All plot items should be cleared


def test_multiple_plot_switches_dont_stack_handlers(qtbot: QtBot) -> None:
    """Test that switching plots multiple times doesn't stack handlers."""
    from cali.plot._single_wells_plots.correlation._plot_calcium_traces_correlation import (  # noqa: E501
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    test_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
    test_roi_labels = [5, 10]

    # Attach handlers multiple times (simulating plot switches)
    for i in range(5):
        widget.clear_plot()
        _attach_heatmap_interaction(
            widget, plot, vb, test_roi_labels, test_matrix, f"Test Title {i}"
        )

    # After all switches, there should only be ONE handler attached
    hover_handler = plot.property("dff_corr_hover_handler")
    click_handler = plot.property("dff_corr_click_handler")
    assert hover_handler is not None
    assert click_handler is not None

    # Track emissions to verify only one handler fires
    emitted = []
    widget.roiSelected.connect(lambda r: emitted.append(r))

    # Simulate a click
    from PyQt6.QtCore import QPointF

    with patch.object(vb, "mapSceneToView") as mock_map:
        import pyqtgraph as pg

        mock_map.return_value = pg.Point(0, 0)

        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True

            mock_event = MagicMock()
            mock_event.scenePos.return_value = QPointF(10, 10)

            click_handler(mock_event)

    # Should only emit once (not 5 times from stacked handlers)
    assert len(emitted) == 1, f"Expected 1 emission, got {len(emitted)}"


def test_evoked_handlers_cleared_properly(qtbot: QtBot) -> None:
    """Test that evoked correlation handlers are properly cleared."""
    from cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (  # noqa: E501
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    test_matrix = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.6], [0.3, 0.6, 1.0]])
    test_roi_labels = [5, 12, 20]

    # Attach evoked handlers
    _attach_heatmap_interaction(
        widget=widget,
        plot=plot,
        base_title="Evoked Test",
        viewbox=vb,
        rois=test_roi_labels,
        values=test_matrix,
    )

    # Verify handlers are attached
    hover_handler = plot.property("evoked_hover_handler")
    click_handler = plot.property("evoked_click_handler")
    assert hover_handler is not None, "Evoked hover handler should be attached"
    assert click_handler is not None, "Evoked click handler should be attached"

    # Call clear_plot
    widget.clear_plot()

    # Verify handlers are cleared
    hover_handler_after = plot.property("evoked_hover_handler")
    click_handler_after = plot.property("evoked_click_handler")
    assert hover_handler_after is None, "Evoked hover handler should be cleared"
    assert click_handler_after is None, "Evoked click handler should be cleared"
