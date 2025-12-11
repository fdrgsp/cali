"""Tests for proper cleanup of heatmap interaction handlers in evoked plots.

This module tests that hover/click handlers are properly disconnected when
switching between sorted evoked plots to prevent ghost title updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg

from cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (  # noqa: E501
    _detach_heatmap_interaction,
)

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def test_detach_heatmap_interaction_clears_handlers(qtbot: QtBot) -> None:
    """Test that _detach_heatmap_interaction removes stored handler references."""
    # Create a mock plot widget
    plot = pg.PlotItem()
    plot.getViewBox()

    # Simulate attaching handlers
    plot.setProperty("evoked_hover_handler", lambda x: None)
    plot.setProperty("evoked_click_handler", lambda x: None)

    # Verify handlers are set
    assert plot.property("evoked_hover_handler") is not None
    assert plot.property("evoked_click_handler") is not None

    # Detach handlers
    _detach_heatmap_interaction(plot)

    # Verify handlers are cleared
    assert plot.property("evoked_hover_handler") is None
    assert plot.property("evoked_click_handler") is None


def test_detach_heatmap_interaction_safe_with_no_handlers(qtbot: QtBot) -> None:
    """Test that _detach_heatmap_interaction is safe when no handlers exist."""
    # Create a mock plot widget with no handlers
    plot = pg.PlotItem()

    # Should not raise any errors
    _detach_heatmap_interaction(plot)

    # Verify properties remain None
    assert plot.property("evoked_hover_handler") is None
    assert plot.property("evoked_click_handler") is None


def test_detach_heatmap_interaction_safe_with_none_scene(qtbot: QtBot) -> None:
    """Test that _detach_heatmap_interaction handles None scene gracefully."""
    # Create a plot that hasn't been added to a scene yet
    plot = pg.PlotItem()

    # Manually clear the scene to simulate edge case
    # (In practice this shouldn't happen, but we test defensive code)
    plot.setProperty("evoked_hover_handler", lambda x: None)

    # Should not raise even if scene is None
    _detach_heatmap_interaction(plot)
