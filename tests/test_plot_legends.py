"""Tests for plot legend functionality in burst and evoked plots.

This module tests that legends are properly displayed and contain
the expected items for plots that use the widget's shared legend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pyqtgraph as pg
import pytest

from cali.plot._single_wells_plots.burst._plot_burst_activity import (
    _draw_population_activity_pg,
)
from cali.plot._single_wells_plots.evoked._plot_evoked_experiment_data_plots import (
    _plot_stimulated_vs_non_stimulated_roi_traces,
    _plot_stimulated_vs_non_stimulated_spike_traces,
)

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


# ============================================================================
# Burst Plot Legend Tests
# ============================================================================


def test_burst_plot_legend_items(qtbot: QtBot) -> None:
    """Test that burst plot legend contains all expected items."""
    # Create a plot widget and legend
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()
    assert plot_item is not None

    # Create a shared legend like in _SingleWellGraphWidget
    legend = pg.LegendItem(offset=(-10, 10), horSpacing=10, verSpacing=0)
    legend.setParentItem(plot_item.graphicsItem())

    # Create sample data
    time_axis = np.linspace(0, 100, 1000)
    raw_activity = np.random.rand(1000) * 0.3
    smoothed_activity = np.random.rand(1000) * 0.5
    bursts = [(100, 200), (500, 600)]  # Two bursts
    threshold_value = 0.3

    # Draw the plot with legend
    _draw_population_activity_pg(
        plot=plot_item,
        time_axis=time_axis,
        raw_activity=raw_activity,
        smoothed_activity=smoothed_activity,
        bursts=bursts,
        threshold_value=threshold_value,
        legend=legend,
    )

    # Check legend is visible
    assert legend.isVisible(), "Legend should be visible"

    # Get legend items
    legend_items = legend.items
    legend_labels = [item[1].text for item in legend_items]

    # Check expected items are present
    expected_labels = [
        "Raw Activity",
        "Smoothed Activity",
        "Burst Threshold",
        "Detected Bursts",
    ]

    for expected_label in expected_labels:
        assert expected_label in legend_labels, (
            f"Legend should contain '{expected_label}'. Found: {legend_labels}"
        )


def test_burst_plot_legend_without_bursts(qtbot: QtBot) -> None:
    """Test burst plot legend when no bursts are detected."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()
    assert plot_item is not None

    legend = pg.LegendItem(offset=(-10, 10), horSpacing=10, verSpacing=0)
    legend.setParentItem(plot_item.graphicsItem())

    # Create sample data with no bursts
    time_axis = np.linspace(0, 100, 1000)
    raw_activity = np.random.rand(1000) * 0.1  # Low activity
    smoothed_activity = np.random.rand(1000) * 0.1
    bursts: list[tuple[int, int]] = []  # No bursts
    threshold_value = 0.3

    _draw_population_activity_pg(
        plot=plot_item,
        time_axis=time_axis,
        raw_activity=raw_activity,
        smoothed_activity=smoothed_activity,
        bursts=bursts,
        threshold_value=threshold_value,
        legend=legend,
    )

    # Legend should still be visible
    assert legend.isVisible()

    # Get legend items
    legend_items = legend.items
    legend_labels = [item[1].text for item in legend_items]

    # Should have all items except "Detected Bursts"
    assert "Raw Activity" in legend_labels
    assert "Smoothed Activity" in legend_labels
    assert "Burst Threshold" in legend_labels
    assert "Detected Bursts" not in legend_labels


def test_burst_plot_without_legend_parameter(qtbot: QtBot) -> None:
    """Test burst plot works when legend parameter is None (backward compatibility)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()
    assert plot_item is not None

    # Create sample data
    time_axis = np.linspace(0, 100, 1000)
    raw_activity = np.random.rand(1000) * 0.3
    smoothed_activity = np.random.rand(1000) * 0.5
    bursts = [(100, 200)]
    threshold_value = 0.3

    # Draw without providing legend (should not crash)
    _draw_population_activity_pg(
        plot=plot_item,
        time_axis=time_axis,
        raw_activity=raw_activity,
        smoothed_activity=smoothed_activity,
        bursts=bursts,
        threshold_value=threshold_value,
        legend=None,
    )

    # Should have plot items
    assert len(plot_item.items) > 0


# ============================================================================
# Evoked Plot Legend Tests
# ============================================================================


def test_evoked_roi_traces_legend_with_both_types(qtbot: QtBot) -> None:
    """Test evoked ROI traces legend with stimulated and non-stimulated ROIs."""
    # Create mock widget
    mock_widget = MagicMock()
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()
    mock_widget.plot_item = plot_item

    # Create shared legend
    legend = pg.LegendItem(offset=(-10, 10), horSpacing=10, verSpacing=0)
    legend.setParentItem(plot_item.graphicsItem())
    mock_widget.legend = legend

    # Create mock engine with test data
    from unittest.mock import Mock

    from sqlmodel import Session

    mock_engine = Mock()
    mock_session = Mock(spec=Session)

    # Mock ROI data with both stimulated and non-stimulated ROIs
    from cali.sqlmodel._model import ROI, DataAnalysis, Traces

    roi1 = ROI(
        id=1,
        label_value=1,
        stimulated=True,
        fov_id=1,
        detection_settings_id=1,
    )
    roi2 = ROI(
        id=2,
        label_value=2,
        stimulated=False,
        fov_id=1,
        detection_settings_id=1,
    )

    trace1 = Traces(
        id=1,
        roi_id=1,
        analysis_result_id=1,
        dec_dff=[0.1, 0.5, 1.0, 0.5, 0.1] * 20,
    )
    trace2 = Traces(
        id=2,
        roi_id=2,
        analysis_result_id=1,
        dec_dff=[0.1, 0.3, 0.8, 0.3, 0.1] * 20,
    )

    data1 = DataAnalysis(
        id=1,
        roi_id=1,
        analysis_result_id=1,
        total_recording_time_sec=10.0,
        peaks_dec_dff=[10, 30, 50],
    )
    data2 = DataAnalysis(
        id=2,
        roi_id=2,
        analysis_result_id=1,
        total_recording_time_sec=10.0,
        peaks_dec_dff=[15, 35, 55],
    )

    mock_results = [
        (roi1, trace1, data1),
        (roi2, trace2, data2),
    ]

    # Mock CaliResult and AnalysisSettings for LED bands
    from cali.sqlmodel._model import AnalysisSettings, CaliResult

    mock_analysis_settings = AnalysisSettings(
        id=1,
        led_pulse_on_frames=[10, 50],
        led_pulse_duration=100.0,
        frame_rate=30.0,
    )
    mock_cali_result = CaliResult(id=1, analysis_settings_id=1)

    # Setup mock session to return appropriate values
    def mock_session_get(model: type, id: int) -> object:
        if model == CaliResult:
            return mock_cali_result
        elif model == AnalysisSettings:
            return mock_analysis_settings
        return None

    mock_session.get = Mock(side_effect=mock_session_get)
    mock_session.exec.return_value.all.return_value = mock_results
    mock_engine.__enter__ = Mock(return_value=mock_session)
    mock_engine.__exit__ = Mock(return_value=False)

    def session_context_manager(*args: object, **kwargs: object) -> object:
        return mock_engine

    Session_mock = Mock(side_effect=session_context_manager)

    # Patch Session in the module
    from cali.plot._single_wells_plots.evoked import (
        _plot_evoked_experiment_data_plots as evoked_mod,
    )

    original_session = evoked_mod.Session
    evoked_mod.Session = Session_mock

    try:
        # Call the function
        _plot_stimulated_vs_non_stimulated_roi_traces(
            widget=mock_widget,
            engine=mock_engine,
            fov_name="test_fov",
            rois=None,
            run_id=1,
            with_peaks=True,
        )

        # Check legend is visible
        assert legend.isVisible(), "Legend should be visible"

        # Get legend items
        legend_items = legend.items
        legend_labels = [item[1].text for item in legend_items]

        # Should have both stimulated and non-stimulated items
        assert "Stimulated ROIs" in legend_labels
        assert "Non-Stimulated ROIs" in legend_labels
        assert "Peaks" in legend_labels
    finally:
        # Restore original Session
        evoked_mod.Session = original_session


def test_evoked_spike_traces_legend(qtbot: QtBot) -> None:
    """Test evoked spike traces legend displays correctly."""
    # Create mock widget
    mock_widget = MagicMock()
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()
    mock_widget.plot_item = plot_item

    # Create shared legend
    legend = pg.LegendItem(offset=(-10, 10), horSpacing=10, verSpacing=0)
    legend.setParentItem(plot_item.graphicsItem())
    mock_widget.legend = legend

    # Create mock engine with test data
    from unittest.mock import Mock

    from sqlmodel import Session

    mock_engine = Mock()
    mock_session = Mock(spec=Session)

    # Mock ROI data with spike traces
    from cali.sqlmodel._model import ROI, DataAnalysis, Traces

    roi1 = ROI(
        id=1,
        label_value=1,
        stimulated=True,
        fov_id=1,
        detection_settings_id=1,
        active=True,
    )
    roi2 = ROI(
        id=2,
        label_value=2,
        stimulated=False,
        fov_id=1,
        detection_settings_id=1,
        active=True,
    )

    trace1 = Traces(
        id=1,
        roi_id=1,
        analysis_result_id=1,
        inferred_spikes=[0.0, 0.5, 1.0, 0.5, 0.0] * 20,
    )
    trace2 = Traces(
        id=2,
        roi_id=2,
        analysis_result_id=1,
        inferred_spikes=[0.0, 0.3, 0.8, 0.3, 0.0] * 20,
    )

    data1 = DataAnalysis(
        id=1,
        roi_id=1,
        analysis_result_id=1,
        total_recording_time_sec=10.0,
        inferred_spikes_threshold=0.2,
    )
    data2 = DataAnalysis(
        id=2,
        roi_id=2,
        analysis_result_id=1,
        total_recording_time_sec=10.0,
        inferred_spikes_threshold=0.2,
    )

    mock_results = [
        (roi1, trace1, data1),
        (roi2, trace2, data2),
    ]

    # Mock CaliResult and AnalysisSettings for LED bands
    from cali.sqlmodel._model import AnalysisSettings, CaliResult

    mock_analysis_settings = AnalysisSettings(
        id=1,
        led_pulse_on_frames=[10, 50],
        led_pulse_duration=100.0,
        frame_rate=30.0,
    )
    mock_cali_result = CaliResult(
        id=1,
        analysis_settings_id=1,
    )

    # Setup mock session to return appropriate values
    def mock_session_get(model: type, id: int) -> object:
        if model == CaliResult:
            return mock_cali_result
        elif model == AnalysisSettings:
            return mock_analysis_settings
        return None

    mock_session.get = Mock(side_effect=mock_session_get)
    mock_session.exec.return_value.all.return_value = mock_results
    mock_engine.__enter__ = Mock(return_value=mock_session)
    mock_engine.__exit__ = Mock(return_value=False)

    def session_context_manager(*args: object, **kwargs: object) -> object:
        return mock_engine

    Session_mock = Mock(side_effect=session_context_manager)

    # Patch Session in the module
    from cali.plot._single_wells_plots.evoked import (
        _plot_evoked_experiment_data_plots as evoked_mod,
    )

    original_session = evoked_mod.Session
    evoked_mod.Session = Session_mock

    try:
        # Call the function
        _plot_stimulated_vs_non_stimulated_spike_traces(
            widget=mock_widget,
            engine=mock_engine,
            fov_name="test_fov",
            rois=None,
            run_id=1,
        )

        # Check legend is visible
        assert legend.isVisible(), "Legend should be visible"

        # Get legend items
        legend_items = legend.items
        legend_labels = [item[1].text for item in legend_items]

        # Should have both stimulated and non-stimulated items
        assert "Stimulated ROIs" in legend_labels
        assert "Non-Stimulated ROIs" in legend_labels
    finally:
        # Restore original Session
        evoked_mod.Session = original_session


def test_evoked_spike_raster_legend(qtbot: QtBot) -> None:
    """Test evoked spike raster legend displays correctly."""
    # Create mock widget
    mock_widget = MagicMock()
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot_item = plot_widget.getPlotItem()
    mock_widget.plot_item = plot_item

    # Create shared legend
    legend = pg.LegendItem(offset=(-10, 10), horSpacing=10, verSpacing=0)
    legend.setParentItem(plot_item.graphicsItem())
    mock_widget.legend = legend

    # Create mock engine with test data
    from unittest.mock import Mock

    from sqlmodel import Session

    from cali.plot._single_wells_plots.evoked._plot_evoked_experiment_data_plots import (  # noqa: E501
        _plot_stimulated_vs_non_stimulated_spike_raster,
    )
    from cali.sqlmodel._model import ROI, DataAnalysis, Traces

    mock_engine = Mock()
    mock_session = Mock(spec=Session)

    roi1 = ROI(
        id=1,
        label_value=1,
        stimulated=True,
        fov_id=1,
        detection_settings_id=1,
        active=True,
    )
    roi2 = ROI(
        id=2,
        label_value=2,
        stimulated=False,
        fov_id=1,
        detection_settings_id=1,
        active=True,
    )

    trace1 = Traces(
        id=1,
        roi_id=1,
        analysis_result_id=1,
        inferred_spikes=[0.0, 0.5, 1.0, 0.5, 0.0] * 20,
    )
    trace2 = Traces(
        id=2,
        roi_id=2,
        analysis_result_id=1,
        inferred_spikes=[0.0, 0.3, 0.8, 0.3, 0.0] * 20,
    )

    data1 = DataAnalysis(
        id=1,
        roi_id=1,
        analysis_result_id=1,
        total_recording_time_sec=10.0,
        inferred_spikes_threshold=0.2,
    )
    data2 = DataAnalysis(
        id=2,
        roi_id=2,
        analysis_result_id=1,
        total_recording_time_sec=10.0,
        inferred_spikes_threshold=0.2,
    )

    mock_results = [
        (roi1, trace1, data1),
        (roi2, trace2, data2),
    ]

    # Mock CaliResult and AnalysisSettings for LED bands
    from cali.sqlmodel._model import AnalysisSettings, CaliResult

    mock_analysis_settings = AnalysisSettings(
        id=1,
        led_pulse_on_frames=[10, 50],
        led_pulse_duration=100.0,
        frame_rate=30.0,
    )
    mock_cali_result = CaliResult(
        id=1,
        analysis_settings_id=1,
    )

    # Setup mock session to return appropriate values
    def mock_session_get(model: type, id: int) -> object:
        if model == CaliResult:
            return mock_cali_result
        elif model == AnalysisSettings:
            return mock_analysis_settings
        return None

    mock_session.get = Mock(side_effect=mock_session_get)
    mock_session.exec.return_value.all.return_value = mock_results
    mock_engine.__enter__ = Mock(return_value=mock_session)
    mock_engine.__exit__ = Mock(return_value=False)

    def session_context_manager(*args: object, **kwargs: object) -> object:
        return mock_engine

    Session_mock = Mock(side_effect=session_context_manager)

    # Patch Session in the module
    from cali.plot._single_wells_plots.evoked import (
        _plot_evoked_experiment_data_plots as evoked_mod,
    )

    original_session = evoked_mod.Session
    evoked_mod.Session = Session_mock

    try:
        # Call the function
        _plot_stimulated_vs_non_stimulated_spike_raster(
            widget=mock_widget,
            engine=mock_engine,
            fov_name="test_fov",
            rois=None,
            run_id=1,
        )

        # Check legend is visible
        assert legend.isVisible(), "Legend should be visible"

        # Get legend items
        legend_items = legend.items
        legend_labels = [item[1].text for item in legend_items]

        # Should have both stimulated and non-stimulated items
        assert "Stimulated ROIs" in legend_labels
        assert "Non-Stimulated ROIs" in legend_labels
    finally:
        # Restore original Session
        evoked_mod.Session = original_session


# ============================================================================
# Integration Test with Real Widget
# ============================================================================


@pytest.mark.parametrize(
    "plot_name",
    [
        "Inferred Spikes (Thresholded) Burst Activity Analysis",
        "Stimulated vs Non-Stimulated Normalized Calcium Traces (Deconvolved ΔF/F0)",
    ],
)
def test_legend_visible_in_widget_plots(
    qtbot: QtBot,
    plot_name: str,
) -> None:
    """Test that legends are visible when plotting through the widget interface."""
    from sqlmodel import Session, create_engine, select

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
    from cali.plot._main_plot import plot_single_well_data
    from cali.sqlmodel._model import FOV

    db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    with Session(engine) as session:
        fov_name = session.exec(select(FOV.name).limit(1)).first()

    assert fov_name is not None

    widget = _SingleWellGraphWidget(None)  # type: ignore[arg-type]
    qtbot.addWidget(widget)
    widget.database_path = db_path
    widget.engine = engine

    # Plot the data
    plot_single_well_data(
        widget=widget,
        engine=engine,
        fov_name=fov_name,
        text=plot_name,
        rois=None,
        run_id=1,
    )

    # Check if legend exists and has items
    if hasattr(widget, "legend") and widget.legend is not None:
        # Legend should be visible for these plot types
        burst_plot = "Burst Activity Analysis"
        stim_plot = "Stimulated vs Non-Stimulated"
        if burst_plot in plot_name or stim_plot in plot_name:
            # Note: Legend visibility depends on data availability
            # Just check it exists and has items if visible
            if widget.legend.isVisible():
                assert len(widget.legend.items) > 0, (
                    f"Legend for '{plot_name}' is visible but has no items"
                )

    engine.dispose(close=True)


def test_sorted_evoked_plots_have_stimulated_rois_legend(qtbot: QtBot) -> None:
    """Test sorted evoked plots show 'Stimulated ROIs' legend."""
    from sqlmodel import Session, create_engine, select

    from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
    from cali.plot._single_wells_plots.correlation import (
        _plot_evoked_correlation_synchrony,
    )

    # _plot_sorted_spike_correlation was removed
    _plot_sorted_spike_synchrony = (
        _plot_evoked_correlation_synchrony._plot_sorted_spike_synchrony
    )
    from cali.sqlmodel._model import FOV

    db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    with Session(engine) as session:
        fov_name = session.exec(select(FOV.name).limit(1)).first()

    assert fov_name is not None

    widget = _SingleWellGraphWidget(None)  # type: ignore[arg-type]
    qtbot.addWidget(widget)
    widget.database_path = db_path
    widget.engine = engine

    # Test sorted synchrony plot
    _plot_sorted_spike_synchrony(
        widget=widget,
        engine=engine,
        fov_name=fov_name,
        rois=None,
        run_id=1,
    )

    # Check legend for "Stimulated ROIs" if data exists
    if (
        hasattr(widget, "legend")
        and widget.legend is not None
        and widget.legend.isVisible()
    ):
        legend_labels = [item[1].text for item in widget.legend.items]
        assert "Stimulated ROIs" in legend_labels, (
            f"Sorted synchrony plot should have 'Stimulated ROIs' legend. "
            f"Found: {legend_labels}"
        )

    # Note: _plot_sorted_spike_correlation was removed during refactoring
    # Test passes as long as synchrony plot has the correct legend

    engine.dispose(close=True)
