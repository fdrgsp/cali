"""Tests for plotting functions.

This module tests all plotting functions to ensure they correctly query the database
and generate visualizations without errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from sqlmodel import create_engine

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy.engine import Engine


@pytest.fixture
def evoked_db_path() -> Path:
    """Return path to evoked experiment test database."""
    path = Path("tests/test_data/evoked/results.cali")
    if not path.exists():
        pytest.skip(f"Evoked test database not found at {path}")
    return path


@pytest.fixture
def evoked_engine(evoked_db_path: Path) -> Generator[Engine, None, None]:
    """Create engine for evoked experiment database."""
    engine = create_engine(f"sqlite:///{evoked_db_path}")
    yield engine
    engine.dispose()


@pytest.fixture
def mock_widget() -> MagicMock:
    """Create a mock graph widget for plotting tests."""
    widget = MagicMock()
    widget.figure = Figure()
    widget.canvas = MagicMock()
    widget.roiSelected = MagicMock()
    widget.roiSelected.emit = MagicMock()

    # Mock the plate viewer attribute (needed for some plots)
    widget._plate_viewer = MagicMock()
    widget._plate_viewer.pv_labels_path = None

    # Wrap Figure methods to track calls while still executing them
    original_clear = widget.figure.clear
    original_add_subplot = widget.figure.add_subplot
    original_tight_layout = widget.figure.tight_layout

    widget.figure.clear = MagicMock(side_effect=original_clear)
    widget.figure.add_subplot = MagicMock(side_effect=original_add_subplot)
    widget.figure.tight_layout = MagicMock(side_effect=original_tight_layout)

    return widget


def test_plot_stimulated_vs_non_stimulated_traces(
    mock_widget: MagicMock, evoked_engine: Engine
) -> None:
    """Test plotting stimulated vs non-stimulated normalized traces."""
    from cali.plot._single_wells_plots._plolt_evoked_experiment_data_plots import (
        _plot_stimulated_vs_non_stimulated_roi_amp,
    )

    # Test with valid data
    _plot_stimulated_vs_non_stimulated_roi_amp(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        rois=None,
        run_id=1,
        with_peaks=False,
    )

    # Verify figure was drawn
    assert mock_widget.canvas.draw.called
    assert len(mock_widget.figure.axes) > 0


def test_plot_stimulated_vs_non_stimulated_traces_with_peaks(
    mock_widget: MagicMock, evoked_engine: Engine
) -> None:
    """Test plotting stimulated vs non-stimulated traces with peaks."""
    from cali.plot._single_wells_plots._plolt_evoked_experiment_data_plots import (
        _plot_stimulated_vs_non_stimulated_roi_amp,
    )

    _plot_stimulated_vs_non_stimulated_roi_amp(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        rois=[1, 2],
        run_id=1,
        with_peaks=True,
    )

    assert mock_widget.canvas.draw.called


def test_plot_stimulated_vs_non_stimulated_traces_no_run(
    mock_widget: MagicMock, evoked_engine: Engine
) -> None:
    """Test plotting with no run_id selected."""
    from cali.plot._single_wells_plots._plolt_evoked_experiment_data_plots import (
        _plot_stimulated_vs_non_stimulated_roi_amp,
    )

    _plot_stimulated_vs_non_stimulated_roi_amp(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        rois=None,
        run_id=None,
        with_peaks=False,
    )

    # Should show message about no run selected
    assert mock_widget.canvas.draw.called


def test_plot_stimulated_vs_non_stimulated_spike_traces(
    mock_widget: MagicMock, evoked_engine: Engine
) -> None:
    """Test plotting stimulated vs non-stimulated spike traces."""
    from cali.plot._single_wells_plots._plolt_evoked_experiment_data_plots import (
        _plot_stimulated_vs_non_stimulated_spike_traces,
    )

    _plot_stimulated_vs_non_stimulated_spike_traces(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        rois=None,
        run_id=1,
    )

    assert mock_widget.canvas.draw.called
    assert len(mock_widget.figure.axes) > 0


def test_plot_stimulated_peak_amplitudes(
    mock_widget: MagicMock, evoked_engine: Engine
) -> None:
    """Test plotting stimulated ROI peak amplitudes."""
    from cali.plot._single_wells_plots._plolt_evoked_experiment_data_plots import (
        _plot_stim_or_not_stim_peaks_amplitude,
    )

    # Test stimulated ROIs
    _plot_stim_or_not_stim_peaks_amplitude(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        rois=None,
        run_id=1,
        stimulated=True,
    )

    assert mock_widget.canvas.draw.called


def test_plot_non_stimulated_peak_amplitudes(
    mock_widget: MagicMock, evoked_engine: Engine
) -> None:
    """Test plotting non-stimulated ROI peak amplitudes."""
    from cali.plot._single_wells_plots._plolt_evoked_experiment_data_plots import (
        _plot_stim_or_not_stim_peaks_amplitude,
    )

    # Test non-stimulated ROIs
    _plot_stim_or_not_stim_peaks_amplitude(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        rois=None,
        run_id=1,
        stimulated=False,
    )

    assert mock_widget.canvas.draw.called


def test_visualize_stimulated_area(
    mock_widget: MagicMock, evoked_engine: Engine
) -> None:
    """Test visualizing stimulated area."""
    from cali.plot._single_wells_plots._plolt_evoked_experiment_data_plots import (
        _visualize_stimulated_area,
    )

    _visualize_stimulated_area(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        rois=None,
        run_id=1,
        with_rois=True,
        stimulated_area=False,
    )

    assert mock_widget.canvas.draw.called


def test_plot_evoked_experiment_data(
    mock_widget: MagicMock, evoked_engine: Engine
) -> None:
    """Test main evoked experiment data plotting function."""
    from cali.plot._single_wells_plots._plolt_evoked_experiment_data_plots import (
        _plot_evoked_experiment_data,
    )

    _plot_evoked_experiment_data(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        rois=None,
        run_id=1,
        stimulated_area=False,
        with_rois=False,
        stimulated=False,
        with_peaks=False,
    )

    assert mock_widget.canvas.draw.called


# ============================================================================
# CALCIUM TRACES TESTS
# ============================================================================


def test_plot_calcium_traces_raw(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting raw calcium traces."""
    from cali.plot._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
        _plot_traces_data,
    )

    _plot_traces_data(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        raw=True,
        dff=False,
        dec=False,
        normalize=False,
        with_peaks=False,
        active_only=False,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called_once()
    mock_widget.figure.add_subplot.assert_called_once()
    mock_widget.canvas.draw.assert_called_once()


def test_plot_calcium_traces_dff(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting ΔF/F calcium traces."""
    from cali.plot._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
        _plot_traces_data,
    )

    _plot_traces_data(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        raw=False,
        dff=True,
        dec=False,
        normalize=False,
        with_peaks=False,
        active_only=False,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called_once()
    mock_widget.canvas.draw.assert_called_once()


def test_plot_calcium_traces_dec_dff(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting deconvolved ΔF/F traces."""
    from cali.plot._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
        _plot_traces_data,
    )

    _plot_traces_data(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        raw=False,
        dff=False,
        dec=True,
        normalize=False,
        with_peaks=False,
        active_only=False,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called_once()
    mock_widget.canvas.draw.assert_called_once()


def test_plot_calcium_traces_normalized(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting normalized traces with global percentile scaling."""
    from cali.plot._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
        _plot_traces_data,
    )

    _plot_traces_data(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        raw=False,
        dff=False,
        dec=True,
        normalize=True,
        with_peaks=False,
        active_only=False,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called_once()
    mock_widget.canvas.draw.assert_called_once()


def test_plot_calcium_traces_with_peaks(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting traces with detected peaks."""
    from cali.plot._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
        _plot_traces_data,
    )

    _plot_traces_data(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        raw=False,
        dff=False,
        dec=True,
        normalize=False,
        with_peaks=True,
        active_only=False,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called_once()
    mock_widget.canvas.draw.assert_called_once()


def test_plot_calcium_traces_active_only(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting only active ROIs."""
    from cali.plot._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
        _plot_traces_data,
    )

    _plot_traces_data(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        raw=False,
        dff=False,
        dec=True,
        normalize=False,
        with_peaks=False,
        active_only=True,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called_once()
    mock_widget.canvas.draw.assert_called_once()


def test_plot_calcium_traces_with_thresholds(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting single ROI with threshold visualization."""
    from cali.plot._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
        _plot_traces_data,
    )

    # Must specify a single ROI for thresholds to show
    _plot_traces_data(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        rois=[1],
        raw=False,
        dff=False,
        dec=True,
        normalize=False,
        with_peaks=True,
        active_only=False,
        thresholds=True,
    )

    mock_widget.figure.clear.assert_called_once()
    mock_widget.canvas.draw.assert_called_once()


def test_plot_calcium_traces_specific_rois(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting specific ROIs."""
    from cali.plot._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
        _plot_traces_data,
    )

    _plot_traces_data(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        rois=[1, 2],
        raw=False,
        dff=False,
        dec=True,
        normalize=False,
        with_peaks=False,
        active_only=False,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called_once()
    mock_widget.canvas.draw.assert_called_once()


# ============================================================================
# INFERRED SPIKES TESTS
# ============================================================================


def test_plot_inferred_spikes_raw(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting raw inferred spikes."""
    from cali.plot._single_wells_plots._plot_inferred_spikes import (
        _plot_inferred_spikes,
    )

    _plot_inferred_spikes(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        raw=True,
        normalize=False,
        active_only=False,
        dec_dff=False,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called()
    mock_widget.canvas.draw.assert_called()


def test_plot_inferred_spikes_normalized(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting normalized inferred spikes."""
    from cali.plot._single_wells_plots._plot_inferred_spikes import (
        _plot_inferred_spikes,
    )

    _plot_inferred_spikes(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        raw=True,
        normalize=True,
        active_only=False,
        dec_dff=False,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called()
    mock_widget.canvas.draw.assert_called()


def test_plot_inferred_spikes_with_dec_dff(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting inferred spikes with deconvolved dff traces."""
    from cali.plot._single_wells_plots._plot_inferred_spikes import (
        _plot_inferred_spikes,
    )

    _plot_inferred_spikes(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        raw=True,
        normalize=False,
        active_only=False,
        dec_dff=True,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called()
    mock_widget.canvas.draw.assert_called()


def test_plot_inferred_spikes_active_only(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting inferred spikes for active ROIs only."""
    from cali.plot._single_wells_plots._plot_inferred_spikes import (
        _plot_inferred_spikes,
    )

    _plot_inferred_spikes(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        raw=True,
        normalize=False,
        active_only=True,
        dec_dff=False,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called()
    mock_widget.canvas.draw.assert_called()


def test_plot_inferred_spikes_with_threshold(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting inferred spikes with threshold for single ROI."""
    from cali.plot._single_wells_plots._plot_inferred_spikes import (
        _plot_inferred_spikes,
    )

    _plot_inferred_spikes(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        rois=[1],
        raw=True,
        normalize=False,
        active_only=False,
        dec_dff=False,
        thresholds=True,
    )

    mock_widget.figure.clear.assert_called()
    mock_widget.canvas.draw.assert_called()


def test_plot_inferred_spikes_specific_rois(
    evoked_engine: Engine,
    mock_widget: MagicMock,
) -> None:
    """Test plotting inferred spikes for specific ROIs."""
    from cali.plot._single_wells_plots._plot_inferred_spikes import (
        _plot_inferred_spikes,
    )

    _plot_inferred_spikes(
        widget=mock_widget,
        engine=evoked_engine,
        fov_name="B5_0000",
        run_id=1,
        rois=[1, 2],
        raw=True,
        normalize=False,
        active_only=False,
        dec_dff=False,
        thresholds=False,
    )

    mock_widget.figure.clear.assert_called()
    mock_widget.canvas.draw.assert_called()
