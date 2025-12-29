"""Test performance of thresholded spike plotting."""

import numpy as np
import pyqtgraph as pg
from pytestqt.qtbot import QtBot

from cali.plot._single_wells_plots.spikes._plot_inferred_spikes import (
    _plot_thresholded_spikes,
)


def test_plot_thresholded_spikes_performance(qtbot: QtBot) -> None:
    """Test that plotting many spikes completes in reasonable time.

    This test ensures the optimization using NaN separators instead of
    individual PlotDataItem objects is working correctly.
    """
    # Create a plot widget
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.getPlotItem()

    # Simulate data with many spikes (e.g., 10000 frames, 30% spike rate)
    n_frames = 10000
    x = np.arange(n_frames, dtype=float)

    # Create spike data with many spikes above threshold
    spike_data = np.random.exponential(scale=0.5, size=n_frames)
    spike_data[spike_data < 0.3] = 0  # Set low values to 0
    threshold = 0.3

    # Count expected spikes
    expected_spikes = np.sum(spike_data > threshold)
    assert expected_spikes > 1000, "Test needs sufficient spikes"

    # Plot should complete quickly (< 1 second for 10k frames)
    import time

    start = time.time()

    result = _plot_thresholded_spikes(
        plot=plot,
        roi_key="test_roi",
        x=x,
        spike_data=spike_data,
        threshold=threshold,
        normalize=False,
        index=0,
        n_rois=1,
        p1=0.0,
        p2=1.0,
    )

    elapsed = time.time() - start

    # Should complete in under 1 second (old implementation took 10+ seconds)
    assert elapsed < 1.0, f"Plotting took too long: {elapsed:.2f}s"
    assert result is not None, "Should return baseline curve"


def test_plot_thresholded_spikes_normalized_performance(qtbot: QtBot) -> None:
    """Test normalized plotting with multiple ROIs performs well."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.getPlotItem()

    n_frames = 5000
    n_rois = 10
    x = np.arange(n_frames, dtype=float)

    import time

    start = time.time()

    # Plot multiple ROIs
    for i in range(n_rois):
        spike_data = np.random.exponential(scale=0.5, size=n_frames)
        spike_data[spike_data < 0.3] = 0

        _plot_thresholded_spikes(
            plot=plot,
            roi_key=f"roi_{i}",
            x=x,
            spike_data=spike_data,
            threshold=0.3,
            normalize=True,
            index=i,
            n_rois=n_rois,
            p1=0.0,
            p2=1.0,
        )

    elapsed = time.time() - start

    # Should complete in under 2 seconds for 10 ROIs
    assert elapsed < 2.0, f"Plotting {n_rois} ROIs took too long: {elapsed:.2f}s"


def test_plot_thresholded_spikes_no_spikes(qtbot: QtBot) -> None:
    """Test that plotting with no spikes above threshold works."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.getPlotItem()

    x = np.arange(1000, dtype=float)
    spike_data = np.random.uniform(0, 0.2, size=1000)  # All below threshold

    result = _plot_thresholded_spikes(
        plot=plot,
        roi_key="test_roi",
        x=x,
        spike_data=spike_data,
        threshold=0.5,
        normalize=False,
        index=0,
        n_rois=1,
        p1=0.0,
        p2=1.0,
    )

    # Should return None when no spikes
    assert result is None


def test_plot_thresholded_spikes_clickable_curve(qtbot: QtBot) -> None:
    """Test that the spike lines are created with correct properties for clicking."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.getPlotItem()

    x = np.arange(1000, dtype=float)
    spike_data = np.random.exponential(scale=0.5, size=1000)
    spike_data[spike_data < 0.3] = 0

    result = _plot_thresholded_spikes(
        plot=plot,
        roi_key="test_roi_123",
        x=x,
        spike_data=spike_data,
        threshold=0.3,
        normalize=False,
        index=5,
        n_rois=10,
        p1=0.0,
        p2=1.0,
    )

    # Should return a PlotDataItem (the spike lines themselves)
    assert result is not None
    assert isinstance(result, pg.PlotDataItem)

    # Check that properties are set correctly for click handling
    assert result.property("roi_label") == "test_roi_123"
    assert result.property("roi_index") == 5
    assert result.name() == "ROI test_roi_123"


def test_plot_thresholded_spikes_normalized_clickable_curve(qtbot: QtBot) -> None:
    """Test that normalized plotting creates clickable spike lines."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.getPlotItem()

    x = np.arange(500, dtype=float)
    spike_data = np.random.exponential(scale=0.5, size=500)
    spike_data[spike_data < 0.3] = 0

    n_rois = 5
    index = 2  # Middle ROI

    result = _plot_thresholded_spikes(
        plot=plot,
        roi_key="roi_middle",
        x=x,
        spike_data=spike_data,
        threshold=0.3,
        normalize=True,
        index=index,
        n_rois=n_rois,
        p1=0.0,
        p2=1.0,
    )

    # Check curve properties
    assert result is not None
    assert result.property("roi_label") == "roi_middle"
    assert result.property("roi_index") == 2

    # Verify it's the actual spike lines (has data with NaN separators)
    x_data = result.xData
    y_data = result.yData
    assert x_data is not None
    assert y_data is not None
    # Should have NaN values separating the vertical line segments
    assert np.any(np.isnan(x_data))
    assert np.any(np.isnan(y_data))
