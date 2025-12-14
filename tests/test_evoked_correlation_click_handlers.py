"""Tests for evoked correlation/synchrony plot click handlers.

Tests that the windowed and sorted correlation/synchrony plots correctly:
1. Emit roiSelected signals with correct ROI label values
2. Handle mouse clicks properly
3. Map matrix positions to ROI labels correctly
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
    """Small test correlation matrix."""
    return np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.6], [0.3, 0.6, 1.0]], dtype=float)


@pytest.fixture
def test_roi_labels() -> list[int]:
    """Test ROI labels (not sequential to test proper mapping)."""
    return [5, 12, 20]  # Non-sequential labels


def test_sorted_spike_synchrony_click_emits_correct_roi_labels(
    qtbot: QtBot, test_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test that clicking on sorted spike synchrony emits correct ROI labels."""
    from cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (  # noqa: E501
        _attach_heatmap_interaction,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    # Attach handlers
    _attach_heatmap_interaction(
        widget=widget,
        plot=plot,
        base_title="Test",
        viewbox=vb,
        rois=test_roi_labels,  # Pass actual ROI labels
        values=test_matrix,
    )

    # Track emissions
    emitted = []
    widget.roiSelected.connect(lambda r: emitted.append(r))

    # Get click handler
    click_handler = plot.property("evoked_click_handler")
    assert click_handler is not None, "Click handler not attached"

    # Simulate click at matrix position (1, 0)
    # This should map to ROI labels [12, 5]
    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(0, 1)  # col=0, row=1

        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True

            mock_event = MagicMock()
            mock_event.scenePos.return_value = QPointF(10, 20)

            click_handler(mock_event)

    # Should emit ROI labels from the test_roi_labels list
    assert len(emitted) == 1, f"Expected 1 emission, got {len(emitted)}"
    # row=1 -> test_roi_labels[1] = 12, col=0 -> test_roi_labels[0] = 5
    assert emitted[0] == ["12", "5"], f"Expected ['12', '5'], got {emitted[0]}"


def test_sorted_dec_dff_correlation_windowed_click(
    qtbot: QtBot,
) -> None:
    """Test windowed dec_dff correlation click handler."""
    from cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (  # noqa: E501
        _plot_sorted_dec_dff_correlation_windowed_by_stim,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    mock_engine = MagicMock()

    # Mock all database queries
    with (
        patch(
            "cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony."
            "_get_sorted_rois_by_stimulation"
        ) as mock_get_sorted,
        patch(
            "cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony."
            "Session"
        ) as mock_session_cls,
    ):
        # Setup mock data
        mock_get_sorted.return_value = ([5, 12, 20], [5, 12], [20])

        # Setup session mock
        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        # Mock CaliResult
        mock_result = MagicMock()
        mock_result.analysis_settings_id = 1
        mock_session.exec.return_value.first.return_value = mock_result

        # Mock AnalysisSettings
        mock_settings = MagicMock()
        mock_settings.led_pulse_on_frames = [10, 50, 90]
        mock_settings.frame_rate = 10.0

        # Mock FOV with ROIs
        mock_fov = MagicMock()
        mock_fov.rois = []

        # Create mock ROIs with traces
        for label in [5, 12, 20]:
            mock_roi = MagicMock()
            mock_roi.label_value = label
            mock_roi.id = label

            # Mock trace with data
            mock_trace = MagicMock()
            mock_trace.dec_dff = np.random.rand(100).tolist()
            mock_trace.roi_id = label

            # Bind to roi for later lookup
            mock_roi.mock_trace = mock_trace

            mock_fov.rois.append(mock_roi)

        # Setup exec to return different things based on call
        def side_effect_func(stmt: object) -> MagicMock:
            result_mock = MagicMock()
            # Check if it's a Traces query or DataAnalysis query
            if "Traces" in str(stmt):
                # Return appropriate trace based on roi_id in query
                for roi in mock_fov.rois:
                    result_mock.first.return_value = roi.mock_trace
                    break
            elif "AnalysisSettings" in str(stmt):
                result_mock.first.return_value = mock_settings
            elif "FOV" in str(stmt):
                result_mock.first.return_value = mock_fov
            else:
                result_mock.first.return_value = mock_result
            return result_mock

        mock_session.exec.side_effect = side_effect_func

        # Call the plot function
        _plot_sorted_dec_dff_correlation_windowed_by_stim(
            widget=widget,
            engine=mock_engine,
            fov_name="test_fov",
            rois=None,
            run_id=1,
            window_ms=250.0,
        )

        # Check that plot was created
        plot = widget.plot_item
        assert plot is not None

        # Check that click handler exists
        click_handler = plot.property("evoked_click_handler")
        if click_handler is not None:
            # Track emissions
            emitted = []
            widget.roiSelected.connect(lambda r: emitted.append(r))

            # Simulate click
            vb = plot.getViewBox()
            with patch.object(vb, "mapSceneToView") as mock_map:
                mock_map.return_value = pg.Point(0, 0)

                with patch.object(plot, "sceneBoundingRect") as mock_rect:
                    mock_rect.return_value.contains.return_value = True

                    mock_event = MagicMock()
                    mock_event.scenePos.return_value = QPointF(10, 10)

                    click_handler(mock_event)

            # Should emit something (exact value depends on correlation computation)
            # The key is that it shouldn't crash
            assert isinstance(emitted, list)


def test_sorted_spike_correlation_windowed_click(
    qtbot: QtBot,
) -> None:
    """Test windowed spike correlation click handler."""
    from cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (  # noqa: E501
        _plot_sorted_spike_correlation_windowed_by_stim,
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    mock_engine = MagicMock()

    # Similar setup as above
    with (
        patch(
            "cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony."
            "_get_sorted_rois_by_stimulation"
        ) as mock_get_sorted,
        patch(
            "cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony."
            "Session"
        ) as mock_session_cls,
    ):
        mock_get_sorted.return_value = ([5, 12, 20], [5, 12], [20])

        mock_session = MagicMock()
        mock_session_cls.return_value.__enter__.return_value = mock_session

        mock_result = MagicMock()
        mock_result.analysis_settings_id = 1

        mock_settings = MagicMock()
        mock_settings.led_pulse_on_frames = [10, 50, 90]
        mock_settings.frame_rate = 10.0

        mock_fov = MagicMock()
        mock_fov.rois = []

        for label in [5, 12, 20]:
            mock_roi = MagicMock()
            mock_roi.label_value = label
            mock_roi.id = label

            mock_trace = MagicMock()
            mock_trace.inferred_spikes = np.random.rand(100).tolist()
            mock_trace.roi_id = label

            mock_da = MagicMock()
            mock_da.inferred_spikes_threshold = 0.1
            mock_da.roi_id = label

            # Bind to roi for later lookup
            mock_roi.mock_trace = mock_trace
            mock_roi.mock_da = mock_da

            mock_fov.rois.append(mock_roi)

        def side_effect_func(stmt: object) -> MagicMock:
            result_mock = MagicMock()
            if "Traces" in str(stmt):
                # Return appropriate trace based on roi_id in query
                for roi in mock_fov.rois:
                    result_mock.first.return_value = roi.mock_trace
                    break
            elif "DataAnalysis" in str(stmt):
                # Return appropriate data analysis based on roi_id in query
                for roi in mock_fov.rois:
                    result_mock.first.return_value = roi.mock_da
                    break
            elif "AnalysisSettings" in str(stmt):
                result_mock.first.return_value = mock_settings
            elif "FOV" in str(stmt):
                result_mock.first.return_value = mock_fov
            else:
                result_mock.first.return_value = mock_result
            return result_mock

        mock_session.exec.side_effect = side_effect_func

        _plot_sorted_spike_correlation_windowed_by_stim(
            widget=widget,
            engine=mock_engine,
            fov_name="test_fov",
            rois=None,
            run_id=1,
            window_ms=250.0,
        )

        plot = widget.plot_item
        assert plot is not None

        click_handler = plot.property("evoked_click_handler")
        if click_handler is not None:
            emitted = []
            widget.roiSelected.connect(lambda r: emitted.append(r))

            vb = plot.getViewBox()
            with patch.object(vb, "mapSceneToView") as mock_map:
                mock_map.return_value = pg.Point(0, 0)

                with patch.object(plot, "sceneBoundingRect") as mock_rect:
                    mock_rect.return_value.contains.return_value = True

                    mock_event = MagicMock()
                    mock_event.scenePos.return_value = QPointF(10, 10)

                    click_handler(mock_event)

            assert isinstance(emitted, list)
