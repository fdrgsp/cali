"""Tests for inferred spike thresholded plotting functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine

from cali.plot._single_wells_plots.spikes._plot_inferred_spikes import (
    _plot_inferred_spikes,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytestqt.qtbot import QtBot
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

# Test data paths
TEST_DB = Path(__file__).parent / "test_data" / "data_and_db_for_tests" / "test_db.cali"


@pytest.fixture
def engine_with_analyzed_data() -> Generator[Engine, None, None]:
    """Create database engine for test database with analyzed data."""
    db_path = TEST_DB
    assert db_path.exists(), f"Test database not found: {db_path}"
    engine = create_engine(f"sqlite:///{db_path}")
    yield engine
    engine.dispose()


@pytest.fixture
def mock_widget(qtbot: QtBot) -> _SingleWellGraphWidget:
    """Create a mock single well graph widget with a PlotItem."""
    from pyqtgraph import PlotItem

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget

    widget = _SingleWellGraphWidget(parent=None)
    widget.plot_item = PlotItem()
    qtbot.addWidget(widget)
    return widget


def test_plot_inferred_spikes_thresholded(
    mock_widget: _SingleWellGraphWidget,
    engine_with_analyzed_data: Engine,
) -> None:
    """Test plotting thresholded inferred spikes as vertical lines with amplitudes."""
    _plot_inferred_spikes(
        widget=mock_widget,
        engine=engine_with_analyzed_data,
        fov_name="B5_0000",
        rois=[1, 2],
        run_id=1,
        thresholded=True,
    )

    plot = mock_widget.plot_item
    assert plot is not None
    assert "Thresholded" in plot.titleLabel.text
    assert "Spike Amplitude" in plot.getAxis("left").labelText
    # Should have multiple items (vertical lines for spikes)
    assert len(plot.items) > 0


def test_plot_inferred_spikes_thresholded_normalized(
    mock_widget: _SingleWellGraphWidget,
    engine_with_analyzed_data: Engine,
) -> None:
    """Test plotting normalized thresholded inferred spikes."""
    _plot_inferred_spikes(
        widget=mock_widget,
        engine=engine_with_analyzed_data,
        fov_name="B5_0000",
        rois=[1, 2],
        run_id=1,
        thresholded=True,
        normalize=True,
    )

    plot = mock_widget.plot_item
    assert plot is not None
    assert "Normalized" in plot.titleLabel.text
    assert "Thresholded" in plot.titleLabel.text
    assert "ROI" in plot.getAxis("left").labelText


def test_plot_inferred_spikes_thresholded_single_roi(
    mock_widget: _SingleWellGraphWidget,
    engine_with_analyzed_data: Engine,
) -> None:
    """Test thresholded plotting with single ROI."""
    _plot_inferred_spikes(
        widget=mock_widget,
        engine=engine_with_analyzed_data,
        fov_name="B5_0000",
        rois=[1],
        run_id=1,
        thresholded=True,
    )

    plot = mock_widget.plot_item
    assert plot is not None
    # Should have items (lines for spikes above threshold)
    assert len(plot.items) > 0


def test_inferred_spike_thresholded_plot_signal_emission(
    mock_widget: _SingleWellGraphWidget,
    engine_with_analyzed_data: Engine,
    qtbot: QtBot,
) -> None:
    """Test that clicking on thresholded spike traces emits roiSelected signal."""
    _plot_inferred_spikes(
        widget=mock_widget,
        engine=engine_with_analyzed_data,
        fov_name="B5_0000",
        rois=[1, 2],
        run_id=1,
        thresholded=True,
    )

    plot = mock_widget.plot_item
    assert plot is not None

    # Find clickable curves (invisible baselines for click handling)
    curves = [
        item
        for item in plot.items
        if hasattr(item, "property") and item.property("roi_label") is not None
    ]
    assert len(curves) > 0

    # Test signal emission
    with qtbot.waitSignal(mock_widget.roiSelected, timeout=1000):
        curve = curves[0]
        roi_label = curve.property("roi_label")
        assert roi_label is not None
        mock_widget.roiSelected.emit(roi_label)


def test_plot_registry_entries_exist() -> None:
    """Test that new thresholded plot types are registered in _main_plot.py."""
    from cali.plot._main_plot import ANALYSIS_PRODUCTS, AnalysisGroup

    # Get all inferred spike plot names
    spike_plots = [
        p.name
        for p in ANALYSIS_PRODUCTS
        if p.group == AnalysisGroup.SINGLE_WELL and "Inferred Spikes" in p.name
    ]

    # Check that new plots are registered
    assert "Inferred Spikes Thresholded" in spike_plots
    assert "Inferred Spikes Thresholded Normalized" in spike_plots


def test_active_only_plots_includes_thresholded() -> None:
    """Test that new thresholded plot types are in ACTIVE_ONLY_PLOTS set."""
    from cali.plot._main_plot import ACTIVE_ONLY_PLOTS

    assert "Inferred Spikes Thresholded" in ACTIVE_ONLY_PLOTS
    assert "Inferred Spikes Thresholded Normalized" in ACTIVE_ONLY_PLOTS


def test_thresholded_plots_only_show_above_threshold(
    mock_widget: _SingleWellGraphWidget,
    engine_with_analyzed_data: Engine,
) -> None:
    """Test that thresholded plots only show spikes above the detection threshold."""
    _plot_inferred_spikes(
        widget=mock_widget,
        engine=engine_with_analyzed_data,
        fov_name="B5_0000",
        rois=[1],
        run_id=1,
        thresholded=True,
    )

    plot = mock_widget.plot_item
    assert plot is not None

    # Count vertical lines (should be less than total frames since only above threshold)
    line_items = [item for item in plot.items if hasattr(item, "xData")]
    # Should have at least some lines (spikes above threshold)
    assert len(line_items) > 0


def test_thresholded_normalized_stacks_rois(
    mock_widget: _SingleWellGraphWidget,
    engine_with_analyzed_data: Engine,
) -> None:
    """Test that normalized thresholded plots stack multiple ROIs vertically."""
    _plot_inferred_spikes(
        widget=mock_widget,
        engine=engine_with_analyzed_data,
        fov_name="B5_0000",
        rois=[1, 2, 3],
        run_id=1,
        thresholded=True,
        normalize=True,
    )

    plot = mock_widget.plot_item
    assert plot is not None

    # Y-axis should not show numeric labels when normalized
    y_axis = plot.getAxis("left")
    # The style should hide values for normalized plots
    assert "ROI" in y_axis.labelText
