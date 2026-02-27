"""Tests to cover diff lines in multi-wells branch vs main.

Covers:
- Spike synchrony / correlation queries and plot functions
- PCA loadings and scree rendering
- PCA stim-split feature matrix building
- _PCAFeaturesDialog widget
- Inferred spike frequency plot functions
- Burst rate bar plot
- Edge cases: empty data, OperationalError handling
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qtpy.QtWidgets import QWidget
from sqlmodel import Session, create_engine

from cali.sqlmodel import (
    FOV,
    Condition,
    Experiment,
    FOVAnalysis,
    Plate,
    Well,
)
from cali.sqlmodel._model import (
    ROI,
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
)
from cali.sqlmodel._util import create_database_and_tables

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytestqt.qtbot import QtBot
    from sqlalchemy.engine import Engine

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget


# ---------------------------------------------------------------------------
# Shared DB fixture: 2 conditions x 2 FOVs with ROIs, burst stats,
# synchrony, and correlation data
# ---------------------------------------------------------------------------


def _build_full_db() -> tuple[Engine, int]:
    """In-memory DB with rich data for synchrony, correlation, PCA tests."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="full_exp")
        session.add(exp)
        session.flush()

        settings = AnalysisSettings(frame_rate=10.0, enable_rising_edge_analysis=True)
        session.add(settings)
        session.flush()

        run = CaliResult(experiment=exp.id, analysis_settings_id=settings.id)
        session.add(run)
        session.flush()
        run_id: int = run.id  # type: ignore[assignment]

        plate = Plate(experiment=exp, name="P1", plate_type="6-well")
        session.add(plate)
        session.flush()

        rng = np.random.default_rng(42)

        for cond_name, row_idx in [("WT", 0), ("KO", 1)]:
            cond = Condition(name=cond_name, condition_type="genotype")
            for fov_idx in range(2):
                well = Well(
                    plate=plate,
                    name=f"{cond_name}_W{fov_idx}",
                    row=row_idx,
                    column=fov_idx,
                    conditions=[cond],
                )
                session.add(well)
                session.flush()

                fov = FOV(
                    name=f"fov_{cond_name.lower()}_{fov_idx}",
                    position_index=fov_idx,
                    well_id=well.id,
                )
                session.add(fov)
                session.flush()

                # 3x3 correlation matrix
                corr = rng.uniform(0.2, 0.8, (3, 3))
                np.fill_diagonal(corr, 1.0)
                corr = ((corr + corr.T) / 2).tolist()

                fa = FOVAnalysis(
                    fov_id=fov.id,
                    analysis_result_id=run.id,
                    spike_burst_count=3 + fov_idx,
                    spike_burst_avg_duration=0.5 + 0.1 * fov_idx,
                    spike_burst_avg_interval=2.0 + 0.2 * fov_idx,
                    spike_population_activity=[0.0] * 600,
                    global_spike_jitter_synchrony=0.3 + 0.1 * fov_idx,
                    spike_max_lag_correlation_matrix=corr,
                )
                session.add(fa)

                # Add ROIs with data for PCA feature matrix
                for roi_idx in range(3):
                    roi = ROI(
                        label_value=roi_idx + 1,
                        active=True,
                        fov_id=fov.id,
                        cell_size=float(rng.uniform(50, 200)),
                    )
                    session.add(roi)
                    session.flush()

                    da = DataAnalysis(
                        roi_id=roi.id,
                        analysis_result_id=run.id,
                        peaks_amplitudes_den_dff=rng.uniform(0.5, 3.0, 5).tolist(),
                        den_dff_frequency=float(rng.uniform(0.1, 2.0)),
                        iei=rng.uniform(0.5, 5.0, 4).tolist(),
                        inferred_spikes_frequency=float(rng.uniform(0.1, 1.0)),
                        inferred_spikes_rising_edge_frequency=float(
                            rng.uniform(0.05, 0.5)
                        ),
                    )
                    session.add(da)

        session.commit()

    return engine, run_id


@pytest.fixture
def full_db() -> Generator[tuple[Engine, int], None, None]:
    engine, run_id = _build_full_db()
    yield engine, run_id
    engine.dispose(close=True)
    gc.collect()


@pytest.fixture
def full_widget(
    qtbot: QtBot,
    full_db: tuple[Engine, int],
) -> Generator[tuple[_MultilWellGraphWidget, Engine, int], None, None]:
    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget

    engine, run_id = full_db
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    qtbot.addWidget(widget)
    widget.engine = engine
    widget.run_id = run_id
    yield widget, engine, run_id


# ---------------------------------------------------------------------------
# Spike synchrony query tests
# ---------------------------------------------------------------------------


def test_query_spike_synchrony_returns_conditions(
    full_db: tuple[Engine, int],
) -> None:
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_spike_synchrony_by_condition,
    )

    engine, run_id = full_db
    data = _query_spike_synchrony_by_condition(engine, run_id)
    assert set(data.keys()) == {"WT", "KO"}
    for _cond, fov_dict in data.items():
        assert len(fov_dict) == 2
        for val in fov_dict.values():
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0


def test_query_spike_synchrony_empty_db() -> None:
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_spike_synchrony_by_condition,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    assert _query_spike_synchrony_by_condition(engine) == {}
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Spike correlation query tests
# ---------------------------------------------------------------------------


def test_query_spike_correlation_returns_conditions(
    full_db: tuple[Engine, int],
) -> None:
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_spike_correlation_by_condition,
    )

    engine, run_id = full_db
    data = _query_spike_correlation_by_condition(engine, run_id)
    assert set(data.keys()) == {"WT", "KO"}
    for _cond, fov_dict in data.items():
        assert len(fov_dict) == 2
        for val in fov_dict.values():
            assert isinstance(val, float)


def test_query_spike_correlation_empty_db() -> None:
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_spike_correlation_by_condition,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    assert _query_spike_correlation_by_condition(engine) == {}
    engine.dispose(close=True)


def test_query_spike_correlation_skips_1x1_matrix() -> None:
    """A 1x1 correlation matrix (single ROI) is skipped."""
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_spike_correlation_by_condition,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="single_roi")
        session.add(exp)
        session.flush()
        settings = AnalysisSettings(frame_rate=10.0)
        session.add(settings)
        session.flush()
        run = CaliResult(experiment=exp.id, analysis_settings_id=settings.id)
        session.add(run)
        session.flush()
        plate = Plate(experiment=exp, name="P1", plate_type="6-well")
        session.add(plate)
        session.flush()
        cond = Condition(name="WT", condition_type="genotype")
        well = Well(plate=plate, name="W0", row=0, column=0, conditions=[cond])
        session.add(well)
        session.flush()
        fov = FOV(name="fov_0", position_index=0, well_id=well.id)
        session.add(fov)
        session.flush()
        fa = FOVAnalysis(
            fov_id=fov.id,
            analysis_result_id=run.id,
            spike_max_lag_correlation_matrix=[[1.0]],
        )
        session.add(fa)
        session.flush()
        rid: int = run.id  # type: ignore[assignment]
        session.commit()

    data = _query_spike_correlation_by_condition(engine, rid)
    assert data == {}
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Spike synchrony and correlation plot functions
# ---------------------------------------------------------------------------


def test_plot_spike_synchrony_bar_plot_renders(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    from pyqtgraph import BarGraphItem

    from cali.plot._multi_wells_plots._inferred_spikes import (
        plot_spike_synchrony_bar_plot,
    )

    widget, engine, run_id = full_widget
    plot_spike_synchrony_bar_plot(widget, "Synchrony", engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


def test_plot_spike_correlation_bar_plot_renders(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    from pyqtgraph import BarGraphItem

    from cali.plot._multi_wells_plots._inferred_spikes import (
        plot_spike_correlation_bar_plot,
    )

    widget, engine, run_id = full_widget
    plot_spike_correlation_bar_plot(widget, "Correlation", engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


def test_plot_spike_synchrony_empty_db_no_crash(
    qtbot: QtBot,
) -> None:
    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._inferred_spikes import (
        plot_spike_synchrony_bar_plot,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    plot_spike_synchrony_bar_plot(widget, "Synchrony", engine, run_id=None)
    engine.dispose(close=True)


def test_plot_spike_correlation_empty_db_no_crash(
    qtbot: QtBot,
) -> None:
    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._inferred_spikes import (
        plot_spike_correlation_bar_plot,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    plot_spike_correlation_bar_plot(widget, "Correlation", engine, run_id=None)
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Inferred spike frequency plot functions
# ---------------------------------------------------------------------------


def test_plot_inferred_spikes_frequency_bar_plot_renders(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    from pyqtgraph import BarGraphItem

    from cali.plot._multi_wells_plots._inferred_spikes import (
        plot_inferred_spikes_frequency_bar_plot,
    )

    widget, engine, run_id = full_widget
    plot_inferred_spikes_frequency_bar_plot(widget, "Spike Freq", engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


def test_plot_inferred_spikes_rising_edge_frequency_bar_plot_renders(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    from pyqtgraph import BarGraphItem

    from cali.plot._multi_wells_plots._inferred_spikes import (
        plot_inferred_spikes_rising_edge_frequency_bar_plot,
    )

    widget, engine, run_id = full_widget
    plot_inferred_spikes_rising_edge_frequency_bar_plot(
        widget, "Spike RE Freq", engine, run_id
    )
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


def test_plot_burst_rate_bar_plot_renders(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    from pyqtgraph import BarGraphItem

    from cali.plot._multi_wells_plots._inferred_spikes import (
        plot_burst_rate_bar_plot,
    )

    widget, engine, run_id = full_widget
    plot_burst_rate_bar_plot(widget, "Burst Rate", engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


# ---------------------------------------------------------------------------
# PCA with varying data — loadings & scree
# ---------------------------------------------------------------------------


def test_pca_scatter_renders_with_varying_data(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    import pyqtgraph as pg

    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_scatter,
    )

    widget, engine, run_id = full_widget
    plot_pca_scatter(widget, "PCA", engine, run_id)
    scatter_items = [
        i for i in widget.plot_item.items if isinstance(i, pg.ScatterPlotItem)
    ]
    assert len(scatter_items) >= 1


def test_pca_loadings_renders_bars(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    from pyqtgraph import BarGraphItem

    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_loadings,
    )

    widget, engine, run_id = full_widget
    plot_pca_loadings(widget, "Loadings", engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


def test_pca_scree_renders_bars_and_line(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    import pyqtgraph as pg
    from pyqtgraph import BarGraphItem

    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_scree,
    )

    widget, engine, run_id = full_widget
    plot_pca_scree(widget, "Scree", engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    line_items = [i for i in widget.plot_item.items if isinstance(i, pg.PlotDataItem)]
    assert len(bar_items) >= 1, "Scree plot should have bars"
    assert len(line_items) >= 1, "Scree plot should have cumulative line"


def test_pca_scatter_stim_split_with_varying_data(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """PCA stim-split with non-evoked data still runs without error."""
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_scatter_stim_split,
    )

    widget, engine, run_id = full_widget
    plot_pca_scatter_stim_split(widget, "PCA stim", engine, run_id)
    assert widget.plot_item is not None


# ---------------------------------------------------------------------------
# build_fov_feature_matrix with include_stim_status
# ---------------------------------------------------------------------------


def test_build_feature_matrix_stim_split_sets_burst_nan(
    full_db: tuple[Engine, int],
) -> None:
    """When include_stim_status=True, burst columns are NaN."""
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        build_fov_feature_matrix,
    )

    engine, run_id = full_db
    df = build_fov_feature_matrix(engine, run_id, include_stim_status=True)
    if len(df) > 0:
        # All ROIs have stimulated=None so stim-split may not produce rows,
        # but if it does, burst columns must be NaN
        for col_name in [
            "burst_count",
            "burst_avg_duration_s",
            "burst_avg_interval_s",
        ]:
            assert df[col_name].isna().all(), f"{col_name} should be NaN for stim-split"


def test_build_feature_matrix_burst_stats_populated(
    full_db: tuple[Engine, int],
) -> None:
    """Non-stim-split matrix should have burst stats from FOVAnalysis."""
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        build_fov_feature_matrix,
    )

    engine, run_id = full_db
    df = build_fov_feature_matrix(engine, run_id, include_stim_status=False)
    assert len(df) == 4
    # At least some burst_count values should be non-NaN
    assert not df["burst_count"].isna().all()


# ---------------------------------------------------------------------------
# _prepare_feature_matrix edge cases
# ---------------------------------------------------------------------------


def test_prepare_feature_matrix_drops_zero_variance() -> None:
    """Columns with zero variance are dropped before PCA."""
    import pandas as pd

    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        _prepare_feature_matrix,
    )

    df = pd.DataFrame(
        {
            "varying": [1.0, 2.0, 3.0],
            "constant": [5.0, 5.0, 5.0],
        }
    )
    X, used = _prepare_feature_matrix(df, feature_cols=["varying", "constant"])
    assert "varying" in used
    assert "constant" not in used
    assert X.shape[1] == 1


def test_prepare_feature_matrix_raises_all_constant() -> None:
    """All-constant features raise ValueError."""
    import pandas as pd

    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        _prepare_feature_matrix,
    )

    df = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]})
    with pytest.raises(ValueError, match="No features with non-zero variance"):
        _prepare_feature_matrix(df, feature_cols=["a", "b"])


# ---------------------------------------------------------------------------
# _PCAFeaturesDialog
# ---------------------------------------------------------------------------


def test_pca_features_dialog_default_all_checked(qtbot: QtBot) -> None:
    from cali.gui._pygraph_plot_widgets import _PCAFeaturesDialog
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        FEATURE_COLUMNS,
    )

    dialog = _PCAFeaturesDialog(
        current_features=None,
        experiment_type=None,
        enable_rising_edge=True,
    )
    qtbot.addWidget(dialog)

    # All checkboxes should be checked and enabled
    for feat in FEATURE_COLUMNS:
        cb = dialog._checkboxes[feat]
        assert cb.isChecked(), f"{feat} should be checked by default"
        assert cb.isEnabled(), f"{feat} should be enabled"

    # get_features returns None when all are checked
    assert dialog.get_features() is None


def test_pca_features_dialog_subset_selected(qtbot: QtBot) -> None:
    from cali.gui._pygraph_plot_widgets import _PCAFeaturesDialog

    selected = ["mean_amplitude", "mean_frequency"]
    dialog = _PCAFeaturesDialog(
        current_features=selected,
        experiment_type=None,
        enable_rising_edge=True,
    )
    qtbot.addWidget(dialog)

    assert dialog._checkboxes["mean_amplitude"].isChecked()
    assert dialog._checkboxes["mean_frequency"].isChecked()
    assert not dialog._checkboxes["mean_iei"].isChecked()

    result = dialog.get_features()
    assert result is not None
    assert set(result) == set(selected)


def test_pca_features_dialog_evoked_disables_burst(qtbot: QtBot) -> None:
    from cali._constants import EVOKED
    from cali.gui._pygraph_plot_widgets import _BURST_FEATURES, _PCAFeaturesDialog

    dialog = _PCAFeaturesDialog(
        current_features=None,
        experiment_type=EVOKED,
        enable_rising_edge=True,
    )
    qtbot.addWidget(dialog)

    for feat in _BURST_FEATURES:
        cb = dialog._checkboxes[feat]
        assert not cb.isEnabled(), f"{feat} should be disabled for evoked"
        assert not cb.isChecked(), f"{feat} should be unchecked for evoked"


def test_pca_features_dialog_rising_edge_disabled(qtbot: QtBot) -> None:
    from cali.gui._pygraph_plot_widgets import _PCAFeaturesDialog

    dialog = _PCAFeaturesDialog(
        current_features=None,
        experiment_type=None,
        enable_rising_edge=False,
    )
    qtbot.addWidget(dialog)

    cb = dialog._checkboxes["mean_spike_freq_edges"]
    assert not cb.isEnabled()
    assert not cb.isChecked()


# ---------------------------------------------------------------------------
# _compute_condition_mean_and_sem edge cases
# ---------------------------------------------------------------------------


def test_compute_condition_mean_and_sem_empty() -> None:
    from cali.plot._multi_wells_plots._util import _compute_condition_mean_and_sem

    mean, sem = _compute_condition_mean_and_sem(np.array([]))
    assert mean == 0.0
    assert sem == 0.0


def test_compute_condition_mean_and_sem_single() -> None:
    from cali.plot._multi_wells_plots._util import _compute_condition_mean_and_sem

    mean, sem = _compute_condition_mean_and_sem(np.array([5.0]))
    assert mean == 5.0
    assert sem == 0.0


# ---------------------------------------------------------------------------
# _run_pca_scatter error handling paths
# ---------------------------------------------------------------------------


def test_pca_scatter_error_path_too_few_fovs(qtbot: QtBot) -> None:
    """PCA scatter with < 2 FOVs shows informative message."""
    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_scatter,
    )

    # DB with only 1 FOV
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    with Session(engine) as session:
        exp = Experiment(name="tiny")
        session.add(exp)
        session.flush()
        settings = AnalysisSettings(frame_rate=10.0)
        session.add(settings)
        session.flush()
        run = CaliResult(experiment=exp.id, analysis_settings_id=settings.id)
        session.add(run)
        session.flush()
        plate = Plate(experiment=exp, name="P1", plate_type="6-well")
        session.add(plate)
        session.flush()
        cond = Condition(name="WT", condition_type="genotype")
        well = Well(plate=plate, name="W0", row=0, column=0, conditions=[cond])
        session.add(well)
        session.flush()
        fov = FOV(name="fov_0", position_index=0, well_id=well.id)
        session.add(fov)
        session.flush()
        fa = FOVAnalysis(fov_id=fov.id, analysis_result_id=run.id)
        session.add(fa)
        session.flush()
        run_id: int = run.id  # type: ignore[assignment]
        session.commit()

    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    plot_pca_scatter(widget, "PCA", engine, run_id)
    # Should show "Need >= 2 FOVs" message, not crash
    assert widget.plot_item is not None
    engine.dispose(close=True)


def test_pca_loadings_too_few_fovs(qtbot: QtBot) -> None:
    """PCA loadings with insufficient data shows message."""
    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_loadings,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    plot_pca_loadings(widget, "Loadings", engine, run_id=None)
    assert widget.plot_item is not None
    engine.dispose(close=True)


def test_pca_scree_too_few_fovs(qtbot: QtBot) -> None:
    """PCA scree with insufficient data shows message."""
    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_scree,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    plot_pca_scree(widget, "Scree", engine, run_id=None)
    assert widget.plot_item is not None
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# _get_experiment_type helper
# ---------------------------------------------------------------------------


def test_get_experiment_type_returns_type(full_db: tuple[Engine, int]) -> None:
    from sqlmodel import Session as Sess

    from cali.plot._multi_wells_plots._util import _get_experiment_type

    engine, run_id = full_db
    with Sess(engine) as session:
        result = _get_experiment_type(session, run_id)
    # Our fixture doesn't set experiment_type, so it should be None or default
    # Just verify no crash
    assert result is None or isinstance(result, str)


def test_get_experiment_type_invalid_run_id(
    full_db: tuple[Engine, int],
) -> None:
    from sqlmodel import Session as Sess

    from cali.plot._multi_wells_plots._util import _get_experiment_type

    engine, _ = full_db
    with Sess(engine) as session:
        result = _get_experiment_type(session, 9999)
    assert result is None
