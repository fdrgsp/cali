"""Tests for cluster analysis plot helper functions.

Covers:
- _get_cluster_data_from_db: early-return and data-retrieval paths
- Plot function "no data" early-return branches
- Interactive hover/click handler closures
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pyqtgraph as pg
import pytest
from qtpy.QtCore import QPointF
from sqlmodel import Session, create_engine

from cali.plot._single_wells_plots.cluster._plot_cluster_analysis import (
    _attach_cluster_heatmap_interaction,
    _get_cluster_data_from_db,
    _plot_cluster_colored_raster,
    _plot_cluster_colored_traces,
    _plot_cluster_connectivity_graph,
    _plot_cluster_sorted_correlation_heatmap,
)
from cali.sqlmodel import FOV, ROI
from cali.sqlmodel._model import (
    CaliResult,
    DataAnalysis,
    Experiment,
    FOVAnalysis,
    Traces,
)
from cali.sqlmodel._util import create_database_and_tables

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytestqt.qtbot import QtBot
    from sqlalchemy.engine import Engine


# Real test database (read-only, never modify)
TEST_DB = Path(__file__).parent / "test_data" / "data_and_db_for_tests" / "test_db.cali"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_engine() -> Generator[Engine, None, None]:
    """Read-only connection to the shared test database."""
    engine = create_engine(f"sqlite:///{TEST_DB}")
    yield engine
    engine.dispose(close=True)


@pytest.fixture
def empty_engine() -> Generator[Engine, None, None]:
    """In-memory DB with tables but no data."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    yield engine
    engine.dispose(close=True)
    gc.collect()


@pytest.fixture
def no_cluster_engine() -> Generator[Engine, None, None]:
    """In-memory DB with one FOV that has NO cluster data."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="no_cluster_exp")
        session.add(exp)
        session.flush()

        run = CaliResult(experiment=exp.id)
        session.add(run)
        session.flush()

        fov = FOV(name="test_fov", position_index=0)
        session.add(fov)
        session.flush()

        fa = FOVAnalysis(
            fov_id=fov.id,
            analysis_result_id=run.id,
            active_roi_labels=[1, 2, 3],
            cluster_labels=None,  # no cluster data
        )
        session.add(fa)
        session.commit()

    yield engine
    engine.dispose(close=True)
    gc.collect()


@pytest.fixture
def null_corr_engine() -> Generator[Engine, None, None]:
    """In-memory DB with cluster data but NULL corr matrix."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="null_corr_exp")
        session.add(exp)
        session.flush()

        run = CaliResult(experiment=exp.id)
        session.add(run)
        session.flush()

        fov = FOV(name="test_fov", position_index=0)
        session.add(fov)
        session.flush()

        fa = FOVAnalysis(
            fov_id=fov.id,
            analysis_result_id=run.id,
            active_roi_labels=[1, 2, 3],
            cluster_labels=[0, 0, 1],
            cluster_method="hierarchical",
            cluster_n_clusters=2,
            cluster_silhouette_score=0.6,
            cluster_order=[0, 1, 2],
            calcium_den_dff_corr_matrix=None,  # no corr matrix
        )
        session.add(fa)
        session.commit()

    yield engine
    engine.dispose(close=True)
    gc.collect()


@pytest.fixture
def single_roi_engine() -> Generator[Engine, None, None]:
    """In-memory DB with a single ROI that has cluster data (n < 2 after filter)."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="single_roi_exp")
        session.add(exp)
        session.flush()

        run = CaliResult(experiment=exp.id)
        session.add(run)
        session.flush()

        fov = FOV(name="test_fov", position_index=0)
        session.add(fov)
        session.flush()

        fa = FOVAnalysis(
            fov_id=fov.id,
            analysis_result_id=run.id,
            active_roi_labels=[1],
            cluster_labels=[0],
            cluster_method="hierarchical",
            cluster_n_clusters=1,
            cluster_silhouette_score=None,
            cluster_order=[0],
        )
        session.add(fa)
        session.commit()

    yield engine
    engine.dispose(close=True)
    gc.collect()


def _make_mock_widget() -> MagicMock:
    """Return a minimal mock _SingleWellGraphWidget."""
    widget = MagicMock()
    plot = MagicMock()
    widget.plot_item = plot
    return widget


# ---------------------------------------------------------------------------
# _get_cluster_data_from_db
# ---------------------------------------------------------------------------


def test_get_cluster_data_returns_all_none_on_database_error(
    empty_engine: Engine,
) -> None:
    """Returns all None when an unexpected database error occurs."""
    from unittest.mock import patch

    with patch(
        "cali.plot._single_wells_plots.cluster._plot_cluster_analysis.Session",
        side_effect=RuntimeError("simulated DB error"),
    ):
        result = _get_cluster_data_from_db(empty_engine, "any_fov", run_id=1)

    assert result == (None, None, None, None, None, None, None)


def test_get_cluster_data_returns_all_none_when_run_id_is_none(
    empty_engine: Engine,
) -> None:
    """Returns all None when run_id is None (fast early-exit path)."""
    result = _get_cluster_data_from_db(empty_engine, "any_fov", run_id=None)
    assert result == (None, None, None, None, None, None, None)


def test_get_cluster_data_returns_all_none_when_fov_not_found(
    empty_engine: Engine,
) -> None:
    """Returns all None when the queried FOV is absent from the database."""
    result = _get_cluster_data_from_db(empty_engine, "nonexistent_fov", run_id=1)
    assert result == (None, None, None, None, None, None, None)


def test_get_cluster_data_returns_all_none_when_cluster_labels_none(
    no_cluster_engine: Engine,
) -> None:
    """Returns all None when FOVAnalysis exists but cluster_labels is None."""
    result = _get_cluster_data_from_db(no_cluster_engine, "test_fov", run_id=1)
    assert result == (None, None, None, None, None, None, None)


def test_get_cluster_data_returns_none_corr_matrix_when_matrix_missing(
    null_corr_engine: Engine,
) -> None:
    """corr_matrix is None when calcium_den_dff_corr_matrix is not stored."""
    corr_matrix, roi_labels, cluster_labels, *_ = _get_cluster_data_from_db(
        null_corr_engine, "test_fov", run_id=1
    )
    assert corr_matrix is None
    assert roi_labels == [1, 2, 3]
    assert cluster_labels == [0, 0, 1]


def test_get_cluster_data_returns_full_data(test_engine: Engine) -> None:
    """Returns full cluster data when all fields are present in FOVAnalysis."""
    corr_matrix, roi_labels, cluster_labels, cluster_order, method, n_k, sil = (
        _get_cluster_data_from_db(test_engine, "B5_0000", run_id=1)
    )

    assert corr_matrix is not None
    assert corr_matrix.shape == (4, 4)
    assert roi_labels == [1, 2, 3, 4]
    assert cluster_labels == [0, 0, 1, 0]
    assert cluster_order == [0, 1, 3, 2]
    assert method == "hierarchical"
    assert n_k == 2
    assert pytest.approx(sil, abs=1e-3) == 0.417


# ---------------------------------------------------------------------------
# Plot function "no data" early returns (mocked widget, no Qt event loop)
# ---------------------------------------------------------------------------


def test_plot_cluster_sorted_correlation_heatmap_no_data(
    no_cluster_engine: Engine,
) -> None:
    """plot_cluster_sorted_correlation_heatmap sets 'No cluster data' title
    when the FOVAnalysis has no cluster fields."""
    widget = _make_mock_widget()
    _plot_cluster_sorted_correlation_heatmap(
        widget, no_cluster_engine, "test_fov", run_id=1
    )
    widget.plot_item.setTitle.assert_called_with(
        "Cluster-Sorted Correlation (No cluster data)"
    )


def test_plot_cluster_sorted_correlation_heatmap_needs_three_rois(
    test_engine: Engine,
) -> None:
    """When rois filter leaves < 3 ROIs the plot shows the 'Need ≥3 ROIs' title."""
    widget = _make_mock_widget()
    # Pass only 2 of the 4 ROIs → < 3 after filtering
    _plot_cluster_sorted_correlation_heatmap(
        widget, test_engine, "B5_0000", rois=[1, 2], run_id=1
    )
    widget.plot_item.setTitle.assert_called_with(
        "Cluster-Sorted Correlation (Need \u22653 ROIs)"
    )


def test_plot_cluster_colored_raster_no_data(
    no_cluster_engine: Engine,
) -> None:
    """plot_cluster_colored_raster shows 'No cluster data' when cluster fields
    are absent."""
    widget = _make_mock_widget()
    _plot_cluster_colored_raster(widget, no_cluster_engine, "test_fov", run_id=1)
    widget.plot_item.setTitle.assert_called_with(
        "Cluster-Colored Raster (No cluster data)"
    )


def test_plot_cluster_connectivity_graph_no_data(
    no_cluster_engine: Engine,
) -> None:
    """plot_cluster_connectivity_graph shows 'No cluster data' title when there
    are no cluster labels in the database."""
    widget = _make_mock_widget()
    _plot_cluster_connectivity_graph(widget, no_cluster_engine, "test_fov", run_id=1)
    widget.plot_item.setTitle.assert_called_with(
        "Functional Connectivity (Clustering) - No cluster data"
    )


def test_plot_cluster_colored_traces_no_data(
    no_cluster_engine: Engine,
) -> None:
    """plot_cluster_colored_traces shows 'No cluster data' title when cluster
    fields are absent."""
    widget = _make_mock_widget()
    _plot_cluster_colored_traces(widget, no_cluster_engine, "test_fov", run_id=1)
    widget.plot_item.setTitle.assert_called_with(
        "Cluster-Colored Traces (No cluster data)"
    )


def test_plot_cluster_connectivity_graph_needs_two_rois(
    single_roi_engine: Engine,
) -> None:
    """Connectivity graph shows 'Need >=2 ROIs' when n < 2 after filtering."""
    widget = _make_mock_widget()
    _plot_cluster_connectivity_graph(widget, single_roi_engine, "test_fov", run_id=1)
    widget.plot_item.setTitle.assert_called_with(
        "Functional Connectivity (Clustering) - Need >=2 ROIs"
    )


def test_plot_cluster_connectivity_graph_rois_below_two_falls_back_to_all(
    single_roi_engine: Engine,
) -> None:
    """When rois filter has no matches the function falls back to all ROIs.

    With a single-ROI database, the fallback still yields n=1 < 2.
    """
    widget = _make_mock_widget()
    # Pass an ROI label that doesn't exist → pairs=[] → else fallback → n=1 < 2
    _plot_cluster_connectivity_graph(
        widget, single_roi_engine, "test_fov", rois=[999], run_id=1
    )
    widget.plot_item.setTitle.assert_called_with(
        "Functional Connectivity (Clustering) - Need >=2 ROIs"
    )


# ---------------------------------------------------------------------------
# Additional fixtures for handler / success-path tests
# ---------------------------------------------------------------------------


@pytest.fixture
def raster_filter_engine() -> Generator[Engine, None, None]:
    """DB with cluster data + ROI records that exercise the 'continue' filters.

    ROI label=99  → NOT in cluster mapping (covers line 367 skip)
    ROI label=10  → in cluster mapping but peaks_den_dff=None (covers line 369 skip)
    """
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="raster_filter_exp")
        session.add(exp)
        session.flush()

        run = CaliResult(experiment=exp.id)
        session.add(run)
        session.flush()

        fov = FOV(name="test_fov", position_index=0)
        session.add(fov)
        session.flush()

        fa = FOVAnalysis(
            fov_id=fov.id,
            analysis_result_id=run.id,
            active_roi_labels=[10, 20],
            cluster_labels=[0, 1],
            cluster_method="hierarchical",
            cluster_n_clusters=2,
            cluster_silhouette_score=0.7,
            cluster_order=[0, 1],
        )
        session.add(fa)

        # ROI whose label is NOT in the cluster mapping
        roi_unmatched = ROI(fov_id=fov.id, label_value=99)
        session.add(roi_unmatched)
        session.flush()
        session.add(
            DataAnalysis(
                roi_id=roi_unmatched.id,
                analysis_result_id=run.id,
                peaks_den_dff=[1.0, 2.0],
            )
        )

        # ROI whose label IS in the mapping but has no peaks
        roi_no_peaks = ROI(fov_id=fov.id, label_value=10)
        session.add(roi_no_peaks)
        session.flush()
        session.add(
            DataAnalysis(
                roi_id=roi_no_peaks.id,
                analysis_result_id=run.id,
                peaks_den_dff=None,
            )
        )

        session.commit()

    yield engine
    engine.dispose(close=True)
    gc.collect()


@pytest.fixture
def traces_filter_engine() -> Generator[Engine, None, None]:
    """DB with cluster data + ROI + Traces records that exercise 'continue' filters.

    ROI label=99  → NOT in cluster mapping (covers line 659 skip)
    ROI label=10  → in mapping but Traces.den_dff=None (covers line 661 skip)
    """
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="traces_filter_exp")
        session.add(exp)
        session.flush()

        run = CaliResult(experiment=exp.id)
        session.add(run)
        session.flush()

        fov = FOV(name="test_fov", position_index=0)
        session.add(fov)
        session.flush()

        session.add(
            FOVAnalysis(
                fov_id=fov.id,
                analysis_result_id=run.id,
                active_roi_labels=[10, 20],
                cluster_labels=[0, 1],
                cluster_method="hierarchical",
                cluster_n_clusters=2,
                cluster_silhouette_score=0.7,
                cluster_order=[0, 1],
            )
        )

        roi_unmatched = ROI(fov_id=fov.id, label_value=99)
        session.add(roi_unmatched)
        session.flush()
        session.add(
            Traces(
                roi_id=roi_unmatched.id,
                analysis_result_id=run.id,
                den_dff=[0.1, 0.2, 0.3],
            )
        )

        roi_null_trace = ROI(fov_id=fov.id, label_value=10)
        session.add(roi_null_trace)
        session.flush()
        session.add(
            Traces(
                roi_id=roi_null_trace.id,
                analysis_result_id=run.id,
                den_dff=None,
            )
        )

        session.commit()

    yield engine
    engine.dispose(close=True)
    gc.collect()


# ---------------------------------------------------------------------------
# _attach_cluster_heatmap_interaction: disconnect + hover + click handlers
# ---------------------------------------------------------------------------


def test_attach_heatmap_disconnects_old_handlers(qtbot: QtBot) -> None:
    """Calling _attach_cluster_heatmap_interaction a second time disconnects
    the previously registered hover and click handlers (covers lines 253-257)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.plotItem
    vb = plot.getViewBox()

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()

    corr = np.eye(3)
    roi_labels = [1, 2, 3]
    reorder = [0, 1, 2]

    # First call: registers handlers
    _attach_cluster_heatmap_interaction(
        mock_widget, plot, vb, roi_labels, reorder, corr, "Title"
    )
    assert plot.property("cluster_heatmap_hover_handler") is not None
    assert plot.property("cluster_heatmap_click_handler") is not None

    # Second call: triggers disconnect branch for both old handlers
    _attach_cluster_heatmap_interaction(
        mock_widget, plot, vb, roi_labels, reorder, corr, "Title"
    )
    # After second call the new handlers are still registered
    assert plot.property("cluster_heatmap_hover_handler") is not None
    assert plot.property("cluster_heatmap_click_handler") is not None


def test_attach_heatmap_hover_outside_scene(qtbot: QtBot) -> None:
    """Hover handler sets base_title and returns when pos is outside scene rect
    (covers lines 260-262)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.plotItem
    mock_vb = MagicMock()

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()

    corr = np.eye(3)
    roi_labels = [1, 2, 3]
    reorder = [0, 1, 2]

    _attach_cluster_heatmap_interaction(
        mock_widget, plot, mock_vb, roi_labels, reorder, corr, "BaseTitle"
    )

    hover_handler = plot.property("cluster_heatmap_hover_handler")
    assert hover_handler is not None

    # A point far outside any reasonable scene bounding rect → early return
    hover_handler(QPointF(-999999.0, -999999.0))

    # Title should be reset to base_title (the real plot.setTitle was called)
    assert "BaseTitle" in plot.titleLabel.text


def test_attach_heatmap_hover_inside_valid_idx(qtbot: QtBot) -> None:
    """Hover handler updates title with ROI info when pos maps to a valid index
    (covers lines 263-270)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.plotItem

    mock_vb = MagicMock()
    mock_pt = MagicMock()
    mock_pt.x.return_value = 1.5  # col_idx = int(1.5) = 1
    mock_pt.y.return_value = 1.5  # row_idx = int(1.5) = 1
    mock_vb.mapSceneToView.return_value = mock_pt

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()

    corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]])
    roi_labels = [10, 20, 30]
    reorder = [0, 1, 2]

    _attach_cluster_heatmap_interaction(
        mock_widget, plot, mock_vb, roi_labels, reorder, corr, "Base"
    )

    hover_handler = plot.property("cluster_heatmap_hover_handler")
    assert hover_handler is not None

    # Patch sceneBoundingRect so that contains() returns True (point "inside")
    mock_rect = MagicMock()
    mock_rect.contains.return_value = True
    with patch.object(plot, "sceneBoundingRect", return_value=mock_rect):
        hover_handler(QPointF(100.0, 100.0))

    # Title should include ROI labels
    assert "ROI" in plot.titleLabel.text


def test_attach_heatmap_hover_inside_invalid_idx(qtbot: QtBot) -> None:
    """Hover handler resets to base_title when pos maps to an out-of-range index
    (covers line 272)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.plotItem

    mock_vb = MagicMock()
    mock_pt = MagicMock()
    mock_pt.x.return_value = 9999.0  # int(9999) = 9999, out of range for n=3
    mock_pt.y.return_value = 9999.0
    mock_vb.mapSceneToView.return_value = mock_pt

    mock_widget = MagicMock()

    corr = np.eye(3)
    roi_labels = [1, 2, 3]
    reorder = [0, 1, 2]

    _attach_cluster_heatmap_interaction(
        mock_widget, plot, mock_vb, roi_labels, reorder, corr, "BaseOOB"
    )

    hover_handler = plot.property("cluster_heatmap_hover_handler")
    mock_rect = MagicMock()
    mock_rect.contains.return_value = True
    with patch.object(plot, "sceneBoundingRect", return_value=mock_rect):
        hover_handler(QPointF(100.0, 100.0))

    # Out-of-range: title reverts to base_title
    assert "BaseOOB" in plot.titleLabel.text


def test_attach_heatmap_click_outside_scene(qtbot: QtBot) -> None:
    """Click handler returns without emitting when pos is outside scene rect
    (covers lines 275-278)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.plotItem
    mock_vb = MagicMock()

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()

    corr = np.eye(3)
    roi_labels = [1, 2, 3]
    reorder = [0, 1, 2]

    _attach_cluster_heatmap_interaction(
        mock_widget, plot, mock_vb, roi_labels, reorder, corr, "Base"
    )

    click_handler = plot.property("cluster_heatmap_click_handler")
    assert click_handler is not None

    mock_ev = MagicMock()
    # scenePos returns a real QPointF far outside the scene
    mock_ev.scenePos.return_value = QPointF(-999999.0, -999999.0)
    click_handler(mock_ev)

    mock_widget.roiSelected.emit.assert_not_called()


def test_attach_heatmap_click_inside_valid_idx(qtbot: QtBot) -> None:
    """Click handler emits roiSelected with ROI pair when pos maps to a valid cell
    (covers lines 279-284)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)
    plot = plot_widget.plotItem

    mock_vb = MagicMock()
    mock_pt = MagicMock()
    mock_pt.x.return_value = 1.5  # col_idx = 1
    mock_pt.y.return_value = 0.5  # row_idx = 0
    mock_vb.mapSceneToView.return_value = mock_pt

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()

    corr = np.eye(3)
    roi_labels = [10, 20, 30]
    reorder = [0, 1, 2]

    _attach_cluster_heatmap_interaction(
        mock_widget, plot, mock_vb, roi_labels, reorder, corr, "Base"
    )

    click_handler = plot.property("cluster_heatmap_click_handler")
    assert click_handler is not None

    mock_ev = MagicMock()
    mock_ev.scenePos.return_value = QPointF(50.0, 50.0)
    mock_rect = MagicMock()
    mock_rect.contains.return_value = True
    with patch.object(plot, "sceneBoundingRect", return_value=mock_rect):
        click_handler(mock_ev)

    mock_widget.roiSelected.emit.assert_called_once()
    emitted = mock_widget.roiSelected.emit.call_args[0][0]
    assert len(emitted) == 2


# ---------------------------------------------------------------------------
# _plot_cluster_colored_raster: path coverage
# ---------------------------------------------------------------------------


def test_plot_raster_has_cluster_data_but_no_roi_records(
    test_engine: Engine,
) -> None:
    """When cluster data exists but ROI filter returns no records, the
    'No ROI data' title is shown (covers lines 360-361)."""
    widget = _make_mock_widget()
    # rois=[999] matches no existing label_value → DataAnalysis query returns empty
    _plot_cluster_colored_raster(widget, test_engine, "B5_0000", rois=[999], run_id=1)
    widget.plot_item.setTitle.assert_called_with("Cluster-Colored Raster (No ROI data)")


def test_plot_raster_skips_roi_not_in_cluster_mapping(
    raster_filter_engine: Engine,
) -> None:
    """ROIs whose label_value is not in the cluster mapping are skipped (line 367)."""
    widget = _make_mock_widget()
    # The fixture has ROI label=99 (not in mapping) and ROI label=10 (no peaks).
    # After filtering all ROIs are skipped → empty roi_cluster_list → runs to
    # completion.
    _plot_cluster_colored_raster(widget, raster_filter_engine, "test_fov", run_id=1)
    # Function completes without crashing; title uses method/k from cluster data
    call_args_list = widget.plot_item.setTitle.call_args_list
    titles = [c[0][0] for c in call_args_list]
    assert any("Raster" in t for t in titles)


def test_plot_raster_success_with_click_handler(
    qtbot: QtBot, test_engine: Engine
) -> None:
    """Full raster render with real PlotWidget: handler is registered and callable
    (covers lines 417-418, 424-431)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem
    mock_widget.legend = MagicMock()

    _plot_cluster_colored_raster(mock_widget, test_engine, "B5_0000", run_id=1)

    handler = plot_widget.plotItem.property("cluster_raster_click_handler")
    assert handler is not None

    # Test: click outside scene → no emit
    mock_ev = MagicMock()
    mock_ev.scenePos.return_value = QPointF(-999999.0, -999999.0)
    handler(mock_ev)
    mock_widget.roiSelected.emit.assert_not_called()

    # Test: click at valid row position using patched sceneBoundingRect + vb
    mock_rect = MagicMock()
    mock_rect.contains.return_value = True
    mock_pt = MagicMock()
    mock_pt.y.return_value = 0.5  # row_idx = int(0.5) = 0
    vb = plot_widget.plotItem.getViewBox()
    mock_ev2 = MagicMock()
    mock_ev2.scenePos.return_value = QPointF(50.0, 50.0)
    with (
        patch.object(plot_widget.plotItem, "sceneBoundingRect", return_value=mock_rect),
        patch.object(vb, "mapSceneToView", return_value=mock_pt),
    ):
        handler(mock_ev2)

    mock_widget.roiSelected.emit.assert_called_once()


def test_plot_raster_disconnect_on_second_call(
    qtbot: QtBot, test_engine: Engine
) -> None:
    """Second call to _plot_cluster_colored_raster disconnects the old handler
    (covers lines 417-418 disconnect branch)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem
    mock_widget.legend = MagicMock()

    _plot_cluster_colored_raster(mock_widget, test_engine, "B5_0000", run_id=1)
    first_handler = plot_widget.plotItem.property("cluster_raster_click_handler")
    assert first_handler is not None

    _plot_cluster_colored_raster(mock_widget, test_engine, "B5_0000", run_id=1)
    second_handler = plot_widget.plotItem.property("cluster_raster_click_handler")
    assert second_handler is not None
    # A new closure is created on each call
    assert first_handler is not second_handler


# ---------------------------------------------------------------------------
# _plot_cluster_connectivity_graph: click handler body
# ---------------------------------------------------------------------------


def test_connectivity_on_node_click_emits_signal(
    qtbot: QtBot, test_engine: Engine
) -> None:
    """on_node_click emits roiSelected with the clicked ROI label
    (covers lines 577-580)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem
    mock_widget.legend = MagicMock()

    _plot_cluster_connectivity_graph(mock_widget, test_engine, "B5_0000", run_id=1)

    handler = plot_widget.plotItem.property("cluster_connectivity_click_handler")
    assert handler is not None

    # Point with valid data()
    mock_point = MagicMock()
    mock_point.data.return_value = 42
    handler(None, [mock_point])
    mock_widget.roiSelected.emit.assert_called_once_with(["42"])

    # Empty points list → no emit
    mock_widget.roiSelected.reset_mock()
    handler(None, [])
    mock_widget.roiSelected.emit.assert_not_called()

    # Point with data() == None → no emit
    mock_point_none = MagicMock()
    mock_point_none.data.return_value = None
    handler(None, [mock_point_none])
    mock_widget.roiSelected.emit.assert_not_called()


def test_connectivity_disconnect_on_second_call(
    qtbot: QtBot, test_engine: Engine
) -> None:
    """Second call disconnects old handler (covers lines 573-574)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem
    mock_widget.legend = MagicMock()

    _plot_cluster_connectivity_graph(mock_widget, test_engine, "B5_0000", run_id=1)
    first_handler = plot_widget.plotItem.property("cluster_connectivity_click_handler")
    assert first_handler is not None

    _plot_cluster_connectivity_graph(mock_widget, test_engine, "B5_0000", run_id=1)
    second_handler = plot_widget.plotItem.property("cluster_connectivity_click_handler")
    assert second_handler is not None
    assert first_handler is not second_handler


# ---------------------------------------------------------------------------
# _plot_cluster_colored_traces: path coverage
# ---------------------------------------------------------------------------


def test_plot_traces_has_cluster_data_but_no_trace_records(
    test_engine: Engine,
) -> None:
    """When cluster data exists but ROI filter returns no trace records, the
    'No data' title is shown (covers lines 652-653)."""
    widget = _make_mock_widget()
    # rois=[999] matches no existing label_value → Traces query returns empty
    _plot_cluster_colored_traces(widget, test_engine, "B5_0000", rois=[999], run_id=1)
    widget.plot_item.setTitle.assert_called_with("Cluster-Colored Traces (No data)")


def test_plot_traces_skips_roi_not_in_cluster_mapping(
    traces_filter_engine: Engine,
) -> None:
    """ROIs not in the cluster mapping are skipped (line 659) and ROIs with
    den_dff=None are skipped (line 661). Function completes without crashing."""
    widget = _make_mock_widget()
    _plot_cluster_colored_traces(widget, traces_filter_engine, "test_fov", run_id=1)
    call_args_list = widget.plot_item.setTitle.call_args_list
    titles = [c[0][0] for c in call_args_list]
    assert any("Traces" in t for t in titles)


def test_plot_traces_success_with_click_handler(
    qtbot: QtBot, test_engine: Engine
) -> None:
    """Full traces render: normalization + plot call (lines 672, 679) and
    handler registered and callable (lines 708-709, 715-722)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem
    mock_widget.legend = MagicMock()

    _plot_cluster_colored_traces(mock_widget, test_engine, "B5_0000", run_id=1)

    handler = plot_widget.plotItem.property("cluster_traces_click_handler")
    assert handler is not None

    # Click outside scene → no emit
    mock_ev = MagicMock()
    mock_ev.scenePos.return_value = QPointF(-999999.0, -999999.0)
    handler(mock_ev)
    mock_widget.roiSelected.emit.assert_not_called()

    # Click at a valid row position
    mock_rect = MagicMock()
    mock_rect.contains.return_value = True
    mock_pt = MagicMock()
    mock_pt.y.return_value = 0.0  # row_idx = 0
    vb = plot_widget.plotItem.getViewBox()
    mock_ev2 = MagicMock()
    mock_ev2.scenePos.return_value = QPointF(50.0, 50.0)
    with (
        patch.object(plot_widget.plotItem, "sceneBoundingRect", return_value=mock_rect),
        patch.object(vb, "mapSceneToView", return_value=mock_pt),
    ):
        handler(mock_ev2)

    mock_widget.roiSelected.emit.assert_called_once()


def test_plot_traces_disconnect_on_second_call(
    qtbot: QtBot, test_engine: Engine
) -> None:
    """Second call to _plot_cluster_colored_traces disconnects old handler
    (covers lines 708-709 disconnect branch)."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem
    mock_widget.legend = MagicMock()

    _plot_cluster_colored_traces(mock_widget, test_engine, "B5_0000", run_id=1)
    first_handler = plot_widget.plotItem.property("cluster_traces_click_handler")
    assert first_handler is not None

    _plot_cluster_colored_traces(mock_widget, test_engine, "B5_0000", run_id=1)
    second_handler = plot_widget.plotItem.property("cluster_traces_click_handler")
    assert second_handler is not None
    assert first_handler is not second_handler


@pytest.fixture
def traces_edge_engine() -> Generator[Engine, None, None]:
    """DB with cluster data + ROI + Traces that hit edge-case branches.

    ROI label=10 → den_dff=[] (empty list: size==0 → continue, covers line 672)
    ROI label=20 → den_dff=[0.5, 0.5, 0.5] (constant trace: t_max==t_min → zeros_like,
                                               covers line 679)
    """
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="traces_edge_exp")
        session.add(exp)
        session.flush()

        run = CaliResult(experiment=exp.id)
        session.add(run)
        session.flush()

        fov = FOV(name="test_fov", position_index=0)
        session.add(fov)
        session.flush()

        session.add(
            FOVAnalysis(
                fov_id=fov.id,
                analysis_result_id=run.id,
                active_roi_labels=[10, 20],
                cluster_labels=[0, 1],
                cluster_method="hierarchical",
                cluster_n_clusters=2,
                cluster_silhouette_score=0.7,
                cluster_order=[0, 1],
            )
        )

        roi1 = ROI(fov_id=fov.id, label_value=10)
        session.add(roi1)
        session.flush()
        session.add(
            Traces(
                roi_id=roi1.id,
                analysis_result_id=run.id,
                den_dff=[],  # empty → size == 0 → continue
            )
        )

        roi2 = ROI(fov_id=fov.id, label_value=20)
        session.add(roi2)
        session.flush()
        session.add(
            Traces(
                roi_id=roi2.id,
                analysis_result_id=run.id,
                den_dff=[0.5, 0.5, 0.5],  # constant → t_max == t_min → zeros_like
            )
        )

        session.commit()

    yield engine
    engine.dispose(close=True)
    gc.collect()


def test_plot_traces_edge_cases_empty_and_constant(
    qtbot: QtBot, traces_edge_engine: Engine
) -> None:
    """Traces loop handles empty den_dff (size==0 continue, line 672) and
    constant den_dff (zeros_like normalization, line 679) without crashing."""
    plot_widget = pg.PlotWidget()
    qtbot.addWidget(plot_widget)

    mock_widget = MagicMock()
    mock_widget.roiSelected = Mock()
    mock_widget.plot_item = plot_widget.plotItem
    mock_widget.legend = MagicMock()

    _plot_cluster_colored_traces(mock_widget, traces_edge_engine, "test_fov", run_id=1)

    # ROI 10 (empty trace) is skipped; ROI 20 (constant) is plotted with zeros_like.
    # The function completes successfully and registers a click handler.
    handler = plot_widget.plotItem.property("cluster_traces_click_handler")
    assert handler is not None
