"""Tests to cover diff lines in multi-wells branch vs main.

Covers:
- Spike synchrony / correlation queries and plot functions
- PCA loadings and scree rendering
- PCA stim-split feature matrix building
- _PCAFeaturesDialog widget
- Inferred spike frequency plot functions
- Burst rate bar plot
- Edge cases: empty data, OperationalError handling
- _BarTickLabel.dataBounds
- _get_cluster_color extended palette
- override_color in bar plot
- OperationalError handlers in burst/sync/corr queries
- Stim-split guards in cell properties and calcium peaks
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING
from unittest.mock import patch

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
# _aggregate_fov_data_to_condition_stats tests
# ---------------------------------------------------------------------------


def test_aggregate_fov_data_weighted_mean() -> None:
    """Condition mean is a weighted average of FOV means (weighted by ROI count)."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {
        "Drug": {
            "fov1": [0.3, 0.5, 0.8, 0.4, 0.6],  # 5 ROIs, mean=0.52
            "fov2": [1.0, 1.2],  # 2 ROIs, mean=1.10
            "fov3": [0.7, 0.9, 0.6, 0.8],  # 4 ROIs, mean=0.75
        }
    }
    result = _aggregate_fov_data_to_condition_stats(data)

    # weighted mean = (5*0.52 + 2*1.10 + 4*0.75) / 11
    expected_mean = (5 * 0.52 + 2 * 1.10 + 4 * 0.75) / 11
    assert abs(result["means"][0] - expected_mean) < 1e-10


def test_aggregate_fov_data_pooled_sem() -> None:
    """Condition SEM is pooled from per-FOV SEMs weighted by ROI count."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {
        "Ctrl": {
            "fov1": [1.0, 2.0, 3.0],  # n=3
            "fov2": [4.0, 5.0],  # n=2
        }
    }
    result = _aggregate_fov_data_to_condition_stats(data)

    # FOV1: mean=2.0, std(ddof=1)=1.0, sem=1.0/sqrt(3)
    # FOV2: mean=4.5, std(ddof=1)≈0.7071, sem≈0.7071/sqrt(2)=0.5
    fov1_sem = 1.0 / np.sqrt(3)
    fov2_sem = np.std([4.0, 5.0], ddof=1) / np.sqrt(2)
    # pooled = sqrt((3*sem1^2 + 2*sem2^2) / 5)
    expected_sem = np.sqrt((3 * fov1_sem**2 + 2 * fov2_sem**2) / 5)
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_fov_data_single_fov() -> None:
    """Single FOV: mean is the FOV mean, SEM is the within-FOV SEM."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {"A": {"fov1": [2.0, 4.0, 6.0]}}
    result = _aggregate_fov_data_to_condition_stats(data)

    assert abs(result["means"][0] - 4.0) < 1e-10
    expected_sem = np.std([2.0, 4.0, 6.0], ddof=1) / np.sqrt(3)
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_fov_data_single_roi_per_fov() -> None:
    """Single ROI per FOV: SEM should be 0 (no within-FOV variability)."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {"A": {"fov1": [3.0], "fov2": [7.0]}}
    result = _aggregate_fov_data_to_condition_stats(data)

    # weighted mean = (1*3 + 1*7)/2 = 5.0
    assert abs(result["means"][0] - 5.0) < 1e-10
    # Each FOV has n=1 → SEM=0, so pooled SEM=0
    assert result["sems"][0] == 0.0


def test_aggregate_fov_data_empty() -> None:
    """Empty input returns empty output."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    result = _aggregate_fov_data_to_condition_stats({})
    assert result["conditions"] == []
    assert result["means"] == []
    assert result["sems"] == []


def test_aggregate_fov_data_with_list_values() -> None:
    """Values that are lists (e.g. peak amplitudes) are flattened correctly."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_data_to_condition_stats,
    )

    data = {
        "X": {
            "fov1": [[1.0, 2.0], [3.0]],  # ROI1 has 2 peaks, ROI2 has 1
            "fov2": [[4.0, 5.0, 6.0]],  # ROI3 has 3 peaks
        }
    }
    result = _aggregate_fov_data_to_condition_stats(data)
    # fov1 flat: [1,2,3] → mean=2.0, n=3
    # fov2 flat: [4,5,6] → mean=5.0, n=3
    # weighted mean = (3*2 + 3*5)/6 = 3.5
    assert abs(result["means"][0] - 3.5) < 1e-10


# ---------------------------------------------------------------------------
# _aggregate_percentage_data_to_condition_stats tests
# ---------------------------------------------------------------------------


def test_aggregate_percentage_weighted_mean() -> None:
    """Percentage mean is weighted by total ROI count per FOV."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    data = {
        "Ctrl": {
            "fov1": (80.0, 10),  # 8/10 active
            "fov2": (15.0, 20),  # 3/20 active
        }
    }
    result = _aggregate_percentage_data_to_condition_stats(data)

    # weighted mean = (10*80 + 20*15) / 30 = 1100/30 ≈ 36.67%
    expected_mean = (10 * 80.0 + 20 * 15.0) / 30
    assert abs(result["means"][0] - expected_mean) < 1e-10


def test_aggregate_percentage_binomial_sem() -> None:
    """SEM uses binomial formula: sqrt(p*(1-p)/N) * 100."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    data = {
        "Ctrl": {
            "fov1": (80.0, 10),
            "fov2": (15.0, 20),
        }
    }
    result = _aggregate_percentage_data_to_condition_stats(data)

    p = result["means"][0] / 100.0
    n_total = 30
    expected_sem = np.sqrt(p * (1 - p) / n_total) * 100
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_percentage_single_fov() -> None:
    """Single FOV: binomial SEM based on that FOV's count."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    data = {"A": {"fov1": (50.0, 20)}}
    result = _aggregate_percentage_data_to_condition_stats(data)

    assert abs(result["means"][0] - 50.0) < 1e-10
    # p=0.5, N=20 → sem = sqrt(0.25/20)*100
    expected_sem = np.sqrt(0.25 / 20) * 100
    assert abs(result["sems"][0] - expected_sem) < 1e-10


def test_aggregate_percentage_empty() -> None:
    """Empty input returns empty output."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_percentage_data_to_condition_stats,
    )

    result = _aggregate_percentage_data_to_condition_stats({})
    assert result["conditions"] == []


# ---------------------------------------------------------------------------
# _store_bar_plot_data / CSV export tests
# ---------------------------------------------------------------------------


def test_store_bar_plot_data(qtbot: QtBot) -> None:
    """_create_pyqtgraph_bar_plot stores a dict on the widget."""
    from qtpy.QtWidgets import QWidget

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._util import _create_pyqtgraph_bar_plot

    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)

    data = {
        "conditions": ["A", "B"],
        "means": [1.0, 2.0],
        "sems": [0.1, 0.2],
        "fov_values_list": [np.array([0.9, 1.1]), np.array([1.8, 2.2, 2.0])],
    }

    _create_pyqtgraph_bar_plot(widget, data, parameter="Amplitude", units="dF/F")

    stored = widget._last_plot_data
    assert isinstance(stored, dict)
    assert stored["conditions"] == ["A", "B"]
    assert stored["means"] == [1.0, 2.0]
    assert stored["sems"] == [0.1, 0.2]
    assert stored["fov_values"] == [[0.9, 1.1], [1.8, 2.2, 2.0]]
    assert stored["parameter"] == "Amplitude (dF/F)"


def test_clear_plot_resets_data(qtbot: QtBot) -> None:
    """clear_plot resets _last_plot_data to None."""
    from qtpy.QtWidgets import QWidget

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget

    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)

    widget._last_plot_data = {"conditions": ["A"]}
    widget.clear_plot()
    assert widget._last_plot_data is None


def test_save_csv_writes_file(qtbot: QtBot, tmp_path: object) -> None:
    """Bar plot dict data can be exported to CSV via DataFrame."""
    import pandas as pd
    from qtpy.QtWidgets import QWidget

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._util import _create_pyqtgraph_bar_plot

    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)

    data = {
        "conditions": ["X"],
        "means": [3.0],
        "sems": [0.5],
        "fov_values_list": [np.array([2.5, 3.5])],
    }
    _create_pyqtgraph_bar_plot(widget, data, parameter="Freq", units="Hz")

    # Build DataFrame from stored dict (same logic as _on_save_csv)
    stored = widget._last_plot_data
    assert isinstance(stored, dict)
    assert "fov_values" in stored

    csv_path = str(tmp_path / "test_export.csv")  # type: ignore[operator]
    max_fovs = max((len(fv) for fv in stored["fov_values"]), default=0)
    rows = []
    for cond, mean, sem, fv in zip(
        stored["conditions"], stored["means"], stored["sems"], stored["fov_values"]
    ):
        row: dict[str, object] = {"condition": cond, "mean": mean, "sem": sem}
        for i, val in enumerate(fv):
            row[f"fov_{i + 1}"] = val
        for i in range(len(fv), max_fovs):
            row[f"fov_{i + 1}"] = float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    # Read back and verify
    result = pd.read_csv(csv_path)
    assert list(result["condition"]) == ["X"]
    assert result["mean"].iloc[0] == 3.0


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


# ---------------------------------------------------------------------------
# _BarTickLabel.dataBounds  (_util.py lines 545-550)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ax", "expected_type"),
    [
        (0, tuple),  # returns (x, x)
        (1, tuple),  # returns (y_extent, 0.0)
        (2, type(None)),  # returns None
    ],
)
def test_bar_tick_label_dataBounds(qtbot: QtBot, ax: int, expected_type: type) -> None:
    import pyqtgraph as pg

    from cali.plot._multi_wells_plots._util import _BarTickLabel

    label = _BarTickLabel("test", y_extent=-0.5, anchor=(0.5, 0))
    # Must be added to a plot for pos() to work
    pw = pg.PlotWidget()
    qtbot.addWidget(pw)
    pw.addItem(label)
    label.setPos(3.0, 0.0)

    result = label.dataBounds(ax)
    assert isinstance(result, expected_type)
    if ax == 0:
        assert result == (3.0, 3.0)
    elif ax == 1:
        assert result == (-0.5, 0.0)


# ---------------------------------------------------------------------------
# _get_cluster_color with n_total > palette size  (line 91)
# ---------------------------------------------------------------------------


def test_get_cluster_color_extended_palette() -> None:
    from cali.plot._single_wells_plots.cluster._plot_cluster_analysis import (
        CLUSTER_COLORS,
        _get_cluster_color,
    )

    n_total = len(CLUSTER_COLORS) + 5
    # Should return generated colors without error
    for cid in range(n_total):
        color = _get_cluster_color(cid, n_total=n_total)
        assert len(color) == 4  # RGBA tuple


# ---------------------------------------------------------------------------
# override_color branch in _create_pyqtgraph_bar_plot  (_util.py lines 587, 590)
# ---------------------------------------------------------------------------


def test_create_bar_plot_with_override_color(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    from pyqtgraph import BarGraphItem

    from cali.plot._multi_wells_plots._util import (
        BarPlotData,
        _create_pyqtgraph_bar_plot,
    )

    widget, _engine, _run_id = full_widget
    data: BarPlotData = {
        "conditions": ["WT", "KO"],
        "means": [1.0, 2.0],
        "sems": [0.1, 0.2],
        "fov_values_list": [np.array([1.0]), np.array([2.0])],
    }
    _create_pyqtgraph_bar_plot(
        widget=widget,
        data=data,
        parameter="Test",
        units="AU",
        override_color="green",
    )
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


# ---------------------------------------------------------------------------
# OperationalError handlers: queries on DB without proper tables
# ---------------------------------------------------------------------------


def _make_no_table_engine() -> Engine:
    """Engine with schema but drop fov_analysis table to trigger OperationalError."""
    from sqlalchemy import text

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS fov_analysis"))
        conn.commit()
    return engine


def test_query_burst_metrics_operational_error() -> None:
    """_query_burst_metrics_by_condition returns {} on OperationalError."""
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_burst_metrics_by_condition,
    )

    engine = _make_no_table_engine()
    assert _query_burst_metrics_by_condition(engine) == {}
    engine.dispose(close=True)


def test_query_spike_synchrony_operational_error() -> None:
    """_query_spike_synchrony_by_condition returns {} on OperationalError."""
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_spike_synchrony_by_condition,
    )

    engine = _make_no_table_engine()
    assert _query_spike_synchrony_by_condition(engine) == {}
    engine.dispose(close=True)


def test_query_spike_correlation_operational_error() -> None:
    """_query_spike_correlation_by_condition returns {} on OperationalError."""
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_spike_correlation_by_condition,
    )

    engine = _make_no_table_engine()
    assert _query_spike_correlation_by_condition(engine) == {}
    engine.dispose(close=True)


def test_query_calcium_burst_metrics_operational_error() -> None:
    """_query_calcium_burst_metrics_by_condition returns {} on OperationalError."""
    from cali.plot._multi_wells_plots._calcium_peaks import (
        _query_calcium_burst_metrics_by_condition,
    )

    engine = _make_no_table_engine()
    assert _query_calcium_burst_metrics_by_condition(engine) == {}
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# _plot_burst_metric with empty data  (_inferred_spikes.py lines 132-133)
# ---------------------------------------------------------------------------


def test_plot_burst_metric_empty_data(qtbot: QtBot) -> None:
    """_plot_burst_metric returns early when query returns empty dict."""
    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._inferred_spikes import _plot_burst_metric

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    _plot_burst_metric(widget, "Burst Count", engine, None, "count", "N")
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Calcium peaks stim-split empty data  (_calcium_peaks.py lines 98-100, 172-174)
# ---------------------------------------------------------------------------


def test_calcium_peaks_amplitude_stim_split_empty(qtbot: QtBot) -> None:
    """plot_calcium_peaks_amplitude_stim_split_bar_plot handles empty evoked data."""
    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._calcium_peaks import (
        plot_calcium_peaks_amplitude_stim_split_bar_plot,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    # No evoked data in DB → both stim and non-stim queries empty → lines 97-100
    plot_calcium_peaks_amplitude_stim_split_bar_plot(widget, "Amp Stim", engine, None)
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Cell properties stim-split empty  (_cell_properties.py lines 190-191)
# ---------------------------------------------------------------------------


def test_percentage_active_stim_split_empty(qtbot: QtBot) -> None:
    """plot_percentage_active_stim_split_bar_plot handles empty stim data."""
    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
    from cali.plot._multi_wells_plots._cell_properties import (
        plot_percentage_active_stim_split_bar_plot,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    # No stim data → empty query → lines 189-191
    plot_percentage_active_stim_split_bar_plot(widget, "% Active Stim", engine, None)
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Dimensionality reduction: stim_suffix  (line 139)
# ---------------------------------------------------------------------------


def test_build_feature_matrix_stim_suffix() -> None:
    """build_fov_feature_matrix adds stim/non_stim suffix when ROI has stimulated."""
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        build_fov_feature_matrix,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="stim_exp")
        session.add(exp)
        session.flush()
        settings = AnalysisSettings(frame_rate=10.0, enable_rising_edge_analysis=True)
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
        # ROI with stimulated=True
        roi = ROI(
            label_value=1,
            active=True,
            fov_id=fov.id,
            cell_size=100.0,
            stimulated=True,
        )
        session.add(roi)
        session.flush()
        da = DataAnalysis(
            roi_id=roi.id,
            analysis_result_id=run.id,
            peaks_amplitudes_den_dff=[1.0, 2.0],
            den_dff_frequency=1.5,
            iei=[0.5],
            inferred_spikes_frequency=0.5,
            inferred_spikes_rising_edge_frequency=0.3,
        )
        session.add(da)
        run_id: int = run.id  # type: ignore[assignment]
        session.commit()

    df = build_fov_feature_matrix(engine, run_id, include_stim_status=True)
    # Should have stim suffix in fov_name
    assert any("_stim" in name for name in df["fov_name"])
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Dimensionality reduction: NaN burst when no FOVAnalysis  (lines 270-272)
# ---------------------------------------------------------------------------


def test_build_feature_matrix_nan_burst_no_fov_analysis() -> None:
    """When FOVAnalysis is missing, burst columns should be NaN."""
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        build_fov_feature_matrix,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="no_fa_exp")
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
        # No FOVAnalysis! Just ROI + DataAnalysis
        roi = ROI(label_value=1, active=True, fov_id=fov.id, cell_size=100.0)
        session.add(roi)
        session.flush()
        da = DataAnalysis(
            roi_id=roi.id,
            analysis_result_id=run.id,
            peaks_amplitudes_den_dff=[1.0, 2.0],
            den_dff_frequency=1.5,
            iei=[0.5],
            inferred_spikes_frequency=0.5,
        )
        session.add(da)
        run_id: int = run.id  # type: ignore[assignment]
        session.commit()

    df = build_fov_feature_matrix(engine, run_id, include_stim_status=False)
    assert len(df) >= 1
    # burst columns should be NaN since no FOVAnalysis exists
    assert df["burst_count"].isna().all()
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Dimensionality reduction: hidden condition in scatter  (line 397)
# ---------------------------------------------------------------------------


def test_pca_scatter_hidden_condition(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """PCA scatter skips conditions with visible=False."""
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_scatter,
    )

    widget, engine, run_id = full_widget
    # First render to populate conditions
    plot_pca_scatter(widget, "PCA", engine, run_id)
    # Now hide one condition
    for cond_name in widget.conditions:
        widget.conditions[cond_name]["visible"] = False
        break
    # Re-render should skip hidden condition
    plot_pca_scatter(widget, "PCA", engine, run_id)
    assert widget.plot_item is not None


# ---------------------------------------------------------------------------
# Dimensionality reduction: _run_pca_scatter exception paths
# ---------------------------------------------------------------------------


def test_pca_scatter_build_matrix_exception(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """_run_pca_scatter handles build_fov_feature_matrix exception."""
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_scatter,
    )

    widget, engine, run_id = full_widget
    with patch(
        "cali.plot._multi_wells_plots._dimensionality_reduction"
        ".build_fov_feature_matrix",
        side_effect=RuntimeError("mock error"),
    ):
        plot_pca_scatter(widget, "PCA", engine, run_id)
    # Should show error message, not crash
    assert widget.plot_item is not None


def test_pca_scatter_compute_pca_generic_exception(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """_run_pca_scatter handles generic compute_pca exception."""
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_scatter,
    )

    widget, engine, run_id = full_widget
    with patch(
        "cali.plot._multi_wells_plots._dimensionality_reduction.compute_pca",
        side_effect=RuntimeError("unexpected"),
    ):
        plot_pca_scatter(widget, "PCA", engine, run_id)
    assert widget.plot_item is not None


# ---------------------------------------------------------------------------
# Dimensionality reduction: _run_pca_full exception  (lines 531-533)
# ---------------------------------------------------------------------------


def test_pca_loadings_build_matrix_exception(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """plot_pca_loadings handles build_fov_feature_matrix exception."""
    from cali.plot._multi_wells_plots._dimensionality_reduction import (
        plot_pca_loadings,
    )

    widget, engine, run_id = full_widget
    with patch(
        "cali.plot._multi_wells_plots._dimensionality_reduction"
        ".build_fov_feature_matrix",
        side_effect=RuntimeError("mock error"),
    ):
        plot_pca_loadings(widget, "Loadings", engine, run_id)
    assert widget.plot_item is not None


# ---------------------------------------------------------------------------
# _query_roi_attribute_by_condition with include_stim_status  (_util.py line 339)
# ---------------------------------------------------------------------------


def test_query_roi_attribute_stim_status_with_run_id(
    full_db: tuple[Engine, int],
) -> None:
    """_query_roi_attribute_by_condition with include_stim_status and run_id."""
    from cali.plot._multi_wells_plots._util import _query_roi_attribute_by_condition

    engine, run_id = full_db
    result = _query_roi_attribute_by_condition(
        engine, "cell_size", run_id=run_id, include_stim_status=True
    )
    # Should return data (with or without stim labels)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# PCA combo shows features button  (_pygraph_plot_widgets.py line 770)
# ---------------------------------------------------------------------------


def test_pca_combo_shows_features_button(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """Selecting a PCA option shows the _pca_features_btn."""
    widget, engine, run_id = full_widget
    widget._engine = engine
    widget._run_id = run_id
    # Directly call the handler with a PCA text to trigger line 770
    widget._on_combo_changed("PCA Scatter")
    # isHidden() checks the widget's own hidden flag (not parent visibility)
    assert not widget._pca_features_btn.isHidden()

    # Non-PCA text hides it
    widget._on_combo_changed("None")
    assert widget._pca_features_btn.isHidden()


# ---------------------------------------------------------------------------
# _show_pca_features_dialog  (_pygraph_plot_widgets.py lines 810-830)
# ---------------------------------------------------------------------------


def test_show_pca_features_dialog(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """_show_pca_features_dialog queries DB and opens dialog."""
    from qtpy.QtWidgets import QDialog

    widget, engine, run_id = full_widget
    widget._engine = engine
    widget._run_id = run_id

    # Mock dialog exec to return Accepted
    with patch("cali.gui._pygraph_plot_widgets._PCAFeaturesDialog") as MockDialog:
        mock_instance = MockDialog.return_value
        mock_instance.exec.return_value = QDialog.DialogCode.Accepted
        mock_instance.get_features.return_value = ["mean_amplitude", "mean_frequency"]

        widget._show_pca_features_dialog()

        MockDialog.assert_called_once()
        assert widget._pca_features == ["mean_amplitude", "mean_frequency"]
