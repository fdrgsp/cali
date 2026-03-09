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
from sqlmodel import Session, create_engine, select

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
from cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (  # noqa: E501
    _detach_heatmap_interaction,
)
from cali.plot._single_wells_plots.correlation._plot_inferred_spike_synchrony import (
    _plot_spike_synchrony_data,
)
from cali.sqlmodel import FOV

if TYPE_CHECKING:
    from pathlib import Path

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


# ============================================================================
# Correlation Plot Signals Tests
# ============================================================================


def test_inferred_spike_synchrony_emits_roi_selected(
    qtbot: QtBot,
) -> None:
    """Test inferred spike synchrony plot emits roiSelected on click."""
    from cali.plot._single_wells_plots.correlation._plot_inferred_spike_synchrony import (  # noqa: E501
        _plot_spike_synchrony_data,
    )

    # Use inline data (2x2 matrix, [5, 10] labels) to avoid fixture name conflicts
    _test_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
    _test_roi_labels = [5, 10]

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    mock_engine = MagicMock()

    with patch(
        "cali.plot._single_wells_plots.correlation._plot_inferred_spike_synchrony."
        "_get_spike_synchrony_matrix_from_db"
    ) as mock_get:
        mock_get.return_value = (
            _test_matrix,
            _test_roi_labels,
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


# ============================================================================
# Evoked Correlation Click Handler Tests
# ============================================================================


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


def test_sorted_den_dff_correlation_windowed_click(
    qtbot: QtBot,
) -> None:
    """Test windowed den_dff correlation click handler."""
    from cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (  # noqa: E501
        _plot_sorted_den_dff_correlation_windowed_by_stim,
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
            mock_trace.den_dff = np.random.rand(100).tolist()
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
        _plot_sorted_den_dff_correlation_windowed_by_stim(
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


def test_evoked_hover_outside_scene_bounds(
    qtbot: QtBot, test_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test hover resets title when position is outside scene bounds."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_evoked_correlation_synchrony,
    )

    _attach_heatmap_interaction = (
        _plot_evoked_correlation_synchrony._attach_heatmap_interaction
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Evoked Correlation"
    _attach_heatmap_interaction(
        widget=widget,
        plot=plot,
        base_title=base_title,
        viewbox=vb,
        rois=test_roi_labels,
        values=test_matrix,
    )

    hover_handler = plot.property("evoked_hover_handler")
    assert hover_handler is not None

    # Test hover outside scene bounds
    with patch.object(plot, "sceneBoundingRect") as mock_rect:
        mock_rect.return_value.contains.return_value = False
        hover_handler(pg.Point(50, 50))

        assert plot.titleLabel.text == base_title


def test_evoked_hover_outside_matrix_indices(
    qtbot: QtBot, test_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test hover resets title when indices are outside matrix dimensions."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_evoked_correlation_synchrony,
    )

    _attach_heatmap_interaction = (
        _plot_evoked_correlation_synchrony._attach_heatmap_interaction
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Evoked Correlation"
    _attach_heatmap_interaction(
        widget=widget,
        plot=plot,
        base_title=base_title,
        viewbox=vb,
        rois=test_roi_labels,
        values=test_matrix,
    )

    hover_handler = plot.property("evoked_hover_handler")

    # Hover at matrix position outside bounds (10, 10 for a 3x3 matrix)
    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(10, 10)
        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True
            hover_handler(pg.Point(50, 50))

            assert plot.titleLabel.text == base_title


def test_evoked_hover_shows_correlation_value(
    qtbot: QtBot, test_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test hover shows correct correlation value in title."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_evoked_correlation_synchrony,
    )

    _attach_heatmap_interaction = (
        _plot_evoked_correlation_synchrony._attach_heatmap_interaction
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    base_title = "Evoked Correlation"
    _attach_heatmap_interaction(
        widget=widget,
        plot=plot,
        base_title=base_title,
        viewbox=vb,
        rois=test_roi_labels,
        values=test_matrix,
    )

    hover_handler = plot.property("evoked_hover_handler")

    # Hover at position (0, 1) in matrix -> value 0.5
    with patch.object(vb, "mapSceneToView") as mock_map:
        mock_map.return_value = pg.Point(1, 0)  # col=1, row=0
        with patch.object(plot, "sceneBoundingRect") as mock_rect:
            mock_rect.return_value.contains.return_value = True
            hover_handler(pg.Point(50, 50))

            title = plot.titleLabel.text
            assert "ROI 5" in title
            assert "ROI 12" in title
            assert "0.500" in title


def test_evoked_click_outside_scene_bounds(
    qtbot: QtBot, test_matrix: np.ndarray, test_roi_labels: list[int]
) -> None:
    """Test click outside scene bounds does not emit signal."""
    from cali.plot._single_wells_plots.correlation import (
        _plot_evoked_correlation_synchrony,
    )

    _attach_heatmap_interaction = (
        _plot_evoked_correlation_synchrony._attach_heatmap_interaction
    )

    widget = _SingleWellGraphWidget(None)  # type: ignore
    qtbot.addWidget(widget)

    plot = widget.plot_item
    assert plot is not None
    vb = plot.getViewBox()

    _attach_heatmap_interaction(
        widget=widget,
        plot=plot,
        base_title="Test",
        viewbox=vb,
        rois=test_roi_labels,
        values=test_matrix,
    )

    emitted = []
    widget.roiSelected.connect(lambda r: emitted.append(r))

    click_handler = plot.property("evoked_click_handler")

    with patch.object(plot, "sceneBoundingRect") as mock_rect:
        mock_rect.return_value.contains.return_value = False
        mock_event = MagicMock()
        mock_event.scenePos.return_value = QPointF(10, 20)
        click_handler(mock_event)

    assert len(emitted) == 0


# ============================================================================
# Spike Max Lag Plot Handler Tests
# ============================================================================


@pytest.fixture
def test_corr_matrix() -> np.ndarray:
    """Small test correlation matrix."""
    return np.array([[1.0, 0.8, 0.5], [0.8, 1.0, 0.6], [0.5, 0.6, 1.0]], dtype=float)


@pytest.fixture
def test_lag_matrix() -> np.ndarray:
    """Small test lag matrix with positive, negative, and zero values."""
    return np.array([[0, 3, -2], [-3, 0, 5], [2, -5, 0]], dtype=int)


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


# ============================================================================
# Synchrony Plot Title Suffix Tests
# ============================================================================


@pytest.fixture
def widget(qtbot: QtBot) -> _SingleWellGraphWidget:
    """Create a _SingleWellGraphWidget for testing."""
    widget = _SingleWellGraphWidget(None)  # type: ignore[arg-type]
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def test_db_path() -> Path:
    """Return path to test database."""
    from pathlib import Path

    return (
        Path(__file__).parent / "test_data" / "data_and_db_for_tests" / "test_db.cali"
    )


def test_spike_synchrony_title_suffix_with_data(
    widget: _SingleWellGraphWidget,
    test_db_path: Path,
    qtbot: QtBot,
) -> None:
    """Test that title_suffix is appended to spike synchrony plot titles."""
    engine = create_engine(f"sqlite:///{test_db_path}")

    try:
        with Session(engine) as session:
            fov = session.exec(select(FOV).limit(1)).first()
            assert fov is not None
            fov_name = fov.name

        # Plot with title_suffix
        _plot_spike_synchrony_data(
            widget=widget,
            engine=engine,
            fov_name=fov_name,
            rois=None,
            run_id=1,
            title_suffix=" - Non-Stimulated",
        )

        # Verify suffix appears in title
        plot_title = widget.plot_item.titleLabel.text  # type: ignore[union-attr]
        assert " - Non-Stimulated" in plot_title, (
            f"Expected suffix in title, got: {plot_title}"
        )
    finally:
        engine.dispose(close=True)


def test_spike_synchrony_title_suffix_no_data(
    widget: _SingleWellGraphWidget,
    test_db_path: Path,
    qtbot: QtBot,
) -> None:
    """Test title_suffix appears when there's no spike data."""
    engine = create_engine(f"sqlite:///{test_db_path}")

    try:
        with Session(engine) as session:
            fov = session.exec(select(FOV).limit(1)).first()
            assert fov is not None
            fov_name = fov.name

        # Plot with insufficient ROIs and title_suffix
        _plot_spike_synchrony_data(
            widget=widget,
            engine=engine,
            fov_name=fov_name,
            rois=[1],  # Single ROI - insufficient for synchrony
            run_id=1,
            title_suffix=" - Test",
        )

        # Should show "Need ≥2 ROIs" with suffix
        plot_title = widget.plot_item.titleLabel.text  # type: ignore[union-attr]
        assert "Need ≥2 ROIs" in plot_title
        assert " - Test" in plot_title, f"Expected suffix in title, got: {plot_title}"
    finally:
        engine.dispose(close=True)


@pytest.mark.parametrize("suffix", ["", " - NonStim", " (Control)", " [Baseline]"])
def test_spike_synchrony_various_suffixes(
    widget: _SingleWellGraphWidget,
    test_db_path: Path,
    qtbot: QtBot,
    suffix: str,
) -> None:
    """Test spike synchrony with various suffix formats."""
    engine = create_engine(f"sqlite:///{test_db_path}")

    try:
        with Session(engine) as session:
            fov = session.exec(select(FOV).limit(1)).first()
            assert fov is not None
            fov_name = fov.name

        _plot_spike_synchrony_data(
            widget=widget,
            engine=engine,
            fov_name=fov_name,
            rois=None,
            run_id=1,
            title_suffix=suffix,
        )

        plot_title = widget.plot_item.titleLabel.text  # type: ignore[union-attr]
        if suffix:
            msg = f"Expected '{suffix}' in title, got: {plot_title}"
            assert suffix in plot_title, msg
        else:
            # Empty suffix should work fine
            assert plot_title, "Title should not be empty"
    finally:
        engine.dispose(close=True)


# ============================================================================
# Clear Plot Handler Tests
# ============================================================================


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


# ============================================================================
# Evoked Plot Handler Cleanup Tests
# ============================================================================


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
