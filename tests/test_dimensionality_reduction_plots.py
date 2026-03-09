"""Tests for PCA and UMAP scatter plot functions.

Covers:
- plot_pca_scatter renders ScatterPlotItem objects per condition
- plot_umap_scatter works when umap-learn is installed, gracefully fails otherwise
- Empty database → no crash
- Feature matrix building returns expected columns
"""

from __future__ import annotations

import gc
import importlib.util
from typing import TYPE_CHECKING
from unittest.mock import patch

import pandas as pd
import pytest
from qtpy.QtWidgets import QWidget
from sqlmodel import Session, create_engine

from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
from cali.plot._multi_wells_plots._dimensionality_reduction import (
    FEATURE_COLUMNS,
    _prepare_feature_matrix,
    build_fov_feature_matrix,
    compute_pca,
    plot_pca_loadings,
    plot_pca_scatter,
    plot_pca_scatter_stim_split,
    plot_pca_scree,
)
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

HAS_PG = importlib.util.find_spec("pyqtgraph") is not None

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytestqt.qtbot import QtBot
    from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# DB fixture: 2 conditions, 2 FOVs each, minimal FOVAnalysis data
# ---------------------------------------------------------------------------


def _build_dim_red_db() -> tuple[Engine, int]:
    """Return (engine, run_id) with 4 FOVs across 2 conditions."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="dim_red_exp")
        session.add(exp)
        session.flush()

        settings = AnalysisSettings(frame_rate=10.0)
        session.add(settings)
        session.flush()

        run = CaliResult(experiment=exp.id, analysis_settings_id=settings.id)
        session.add(run)
        session.flush()
        run_id: int = run.id  # type: ignore[assignment]

        plate = Plate(experiment=exp, name="P1", plate_type="6-well")
        session.add(plate)
        session.flush()

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

                fa = FOVAnalysis(
                    fov_id=fov.id,
                    analysis_result_id=run.id,
                    spike_burst_count=3 + fov_idx,
                    spike_burst_avg_duration=0.5 + 0.1 * fov_idx,
                    spike_burst_avg_interval=2.0 + 0.2 * fov_idx,
                )
                session.add(fa)

        session.commit()

    return engine, run_id


@pytest.fixture
def dim_red_db() -> Generator[tuple[Engine, int], None, None]:
    engine, run_id = _build_dim_red_db()
    yield engine, run_id
    engine.dispose(close=True)
    gc.collect()


@pytest.fixture
def pca_widget(
    qtbot: QtBot,
    dim_red_db: tuple[Engine, int],
) -> Generator[tuple[_MultilWellGraphWidget, Engine, int], None, None]:
    engine, run_id = dim_red_db
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    qtbot.addWidget(widget)
    widget.engine = engine
    widget.run_id = run_id
    yield widget, engine, run_id
    engine.dispose(close=True)
    gc.collect()


@pytest.fixture
def empty_pca_widget(
    qtbot: QtBot,
) -> Generator[_MultilWellGraphWidget, None, None]:
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    qtbot.addWidget(widget)
    widget.engine = engine
    widget.run_id = 1
    yield widget
    engine.dispose(close=True)
    gc.collect()


# ---------------------------------------------------------------------------
# build_fov_feature_matrix (pure DB — no Qt)
# ---------------------------------------------------------------------------


def test_build_fov_feature_matrix_columns(dim_red_db: tuple[Engine, int]) -> None:
    """Feature matrix has fov_name, condition, and all FEATURE_COLUMNS."""
    engine, run_id = dim_red_db
    df = build_fov_feature_matrix(engine, run_id)
    assert "fov_name" in df.columns
    assert "condition" in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_build_fov_feature_matrix_row_count(dim_red_db: tuple[Engine, int]) -> None:
    """One row per FOV (4 FOVs in fixture)."""
    engine, run_id = dim_red_db
    df = build_fov_feature_matrix(engine, run_id)
    assert len(df) == 4


def test_build_fov_feature_matrix_conditions(dim_red_db: tuple[Engine, int]) -> None:
    """Condition column contains the two configured condition names."""
    engine, run_id = dim_red_db
    df = build_fov_feature_matrix(engine, run_id)
    assert set(df["condition"].unique()) == {"WT", "KO"}


def test_build_fov_feature_matrix_empty_db() -> None:
    """Returns an empty DataFrame (not None) for an empty database."""
    import pandas as pd

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    df = build_fov_feature_matrix(engine)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# compute_pca (pure math — no Qt)
# ---------------------------------------------------------------------------


def test_compute_pca_shape(dim_red_db: tuple[Engine, int]) -> None:
    """PCA coords have shape (n_fovs, 2)."""
    engine, run_id = dim_red_db
    df = build_fov_feature_matrix(engine, run_id)
    coords, _pca, _features = compute_pca(df)
    assert coords.shape == (len(df), 2)


def test_compute_pca_explained_variance(dim_red_db: tuple[Engine, int]) -> None:
    """Explained variance ratios are positive and sum to ≤ 1.0."""
    engine, run_id = dim_red_db
    df = build_fov_feature_matrix(engine, run_id)
    _, pca, _ = compute_pca(df)
    ratios = pca.explained_variance_ratio_
    assert all(r >= 0.0 for r in ratios)
    assert sum(ratios) <= 1.0 + 1e-9


def test_compute_pca_used_features_subset(dim_red_db: tuple[Engine, int]) -> None:
    """Used features are a subset of FEATURE_COLUMNS."""
    engine, run_id = dim_red_db
    df = build_fov_feature_matrix(engine, run_id)
    _, _, used = compute_pca(df)
    assert set(used).issubset(set(FEATURE_COLUMNS))


# ---------------------------------------------------------------------------
# plot_pca_scatter (Qt required)
# ---------------------------------------------------------------------------


def test_plot_pca_scatter_renders_scatter_items(
    pca_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """PCA scatter plot adds at least one ScatterPlotItem to plot_item."""
    import pyqtgraph as pg

    widget, engine, run_id = pca_widget
    plot_pca_scatter(widget, "PCA", engine, run_id)

    assert widget.plot_item is not None
    scatter_items = [
        item for item in widget.plot_item.items if isinstance(item, pg.ScatterPlotItem)
    ]
    assert len(scatter_items) >= 1, "Expected at least one ScatterPlotItem"


def test_plot_pca_scatter_one_scatter_per_condition(
    pca_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """Each visible condition gets its own ScatterPlotItem."""
    import pyqtgraph as pg

    widget, engine, run_id = pca_widget
    plot_pca_scatter(widget, "PCA", engine, run_id)

    n_scatter = sum(
        1 for item in widget.plot_item.items if isinstance(item, pg.ScatterPlotItem)
    )
    # fixture has 2 conditions (WT, KO)
    assert n_scatter == 2


def test_plot_pca_scatter_conditions_set(
    pca_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """widget.conditions is populated after plot."""
    widget, engine, run_id = pca_widget
    plot_pca_scatter(widget, "PCA", engine, run_id)
    assert set(widget.conditions.keys()) == {"WT", "KO"}


def test_plot_pca_scatter_axis_labels_contain_var(
    pca_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """Bottom axis label contains 'PC1' and left axis label contains 'PC2'."""
    widget, engine, run_id = pca_widget
    plot_pca_scatter(widget, "PCA", engine, run_id)
    bottom = widget.plot_item.getAxis("bottom").labelText
    left = widget.plot_item.getAxis("left").labelText
    assert "PC1" in bottom
    assert "PC2" in left


def test_plot_pca_scatter_no_crash_empty_db(
    empty_pca_widget: _MultilWellGraphWidget,
) -> None:
    """Empty DB → no crash; plot remains clear."""
    plot_pca_scatter(
        empty_pca_widget, "PCA", empty_pca_widget.engine, empty_pca_widget.run_id
    )
    # Should not raise


def test_plot_pca_scatter_legend_cleared_on_replot(
    pca_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """Calling plot_pca_scatter twice does not double-accumulate legend items."""
    import pyqtgraph as pg

    widget, engine, run_id = pca_widget
    plot_pca_scatter(widget, "PCA", engine, run_id)
    widget.clear_plot()
    plot_pca_scatter(widget, "PCA", engine, run_id)

    n_scatter = sum(
        1 for item in widget.plot_item.items if isinstance(item, pg.ScatterPlotItem)
    )
    assert n_scatter == 2  # still one per condition, not doubled


# ---------------------------------------------------------------------------
# plot_pca_scatter_stim_split
# ---------------------------------------------------------------------------


def test_plot_pca_scatter_stim_split_no_crash(
    pca_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """plot_pca_scatter_stim_split runs without error (graceful on non-evoked DB)."""
    widget, engine, run_id = pca_widget
    # Should not raise — fixture is non-evoked so ROIs have no stimulated attr;
    # the function should still complete (possibly clearing the plot).
    plot_pca_scatter_stim_split(widget, "PCA stim", engine, run_id)


def test_plot_pca_scatter_stim_split_empty_db(
    empty_pca_widget: _MultilWellGraphWidget,
) -> None:
    """Empty DB → no crash for stim-split PCA scatter."""
    plot_pca_scatter_stim_split(
        empty_pca_widget,
        "PCA stim",
        empty_pca_widget.engine,
        empty_pca_widget.run_id,
    )


def test_build_fov_feature_matrix_include_stim_status_no_crash(
    dim_red_db: tuple[Engine, int],
) -> None:
    """build_fov_feature_matrix with include_stim_status=True does not crash."""
    import pandas as pd

    engine, run_id = dim_red_db
    df = build_fov_feature_matrix(engine, run_id, include_stim_status=True)
    # The fixture has no ROI/DataAnalysis records (only FOVAnalysis burst stats),
    # so the stim-split matrix is empty — that is a valid result, just must
    # not raise an exception.
    assert isinstance(df, pd.DataFrame)


# def test_pca_stim_split_registered_in_analysis_products() -> None:
#     """plot_pca_scatter_stim_split is registered as an AnalysisProduct."""
#     from cali.plot._main_plot import ANALYSIS_PRODUCTS

#     names = [p.name for p in ANALYSIS_PRODUCTS]
#     assert "PCA Scatter (Stim vs NonStim)" in names


def test_inferred_spike_burst_combo_names_renamed() -> None:
    """Inferred spike burst AnalysisProducts include 'Inferred Spikes' in
    their names.
    """
    from cali.plot._main_plot import ANALYSIS_PRODUCTS

    names = [p.name for p in ANALYSIS_PRODUCTS]
    assert "Inferred Spikes Burst Count Bar Plot" in names
    assert "Inferred Spikes Burst Average Duration Bar Plot" in names
    assert "Inferred Spikes Burst Average Interval Bar Plot" in names
    assert "Inferred Spikes Burst Rate Bar Plot" in names
    # Old bare names must no longer exist
    assert "Burst Count Bar Plot" not in names
    assert "Burst Rate Bar Plot" not in names


def test_build_fov_feature_matrix_with_roi_data_populates_metrics() -> None:
    """build_fov_feature_matrix accumulates ROI-level metrics correctly.

    Regression test for the bug where d = fov_roi_data[fov.id] was used
    instead of d = fov_roi_data[key] — which caused a KeyError (keys are
    tuples, not bare ints) and silently produced an empty feature matrix.
    """
    import gc

    import pandas as pd
    from sqlmodel import Session, create_engine

    from cali.sqlmodel import FOV, Condition, Experiment, FOVAnalysis, Plate, Well
    from cali.sqlmodel._model import (
        ROI,
        AnalysisSettings,
        CaliResult,
        DataAnalysis,
    )
    from cali.sqlmodel._util import create_database_and_tables

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    try:
        with Session(engine) as session:
            exp = Experiment(name="roi_test_exp")
            session.add(exp)
            session.flush()

            settings = AnalysisSettings(frame_rate=10.0)
            session.add(settings)
            session.flush()

            run = CaliResult(experiment=exp.id, analysis_settings_id=settings.id)
            session.add(run)
            session.flush()
            run_id: int = run.id  # type: ignore[assignment]

            plate = Plate(experiment=exp, name="P1", plate_type="6-well")
            session.add(plate)
            session.flush()

            cond = Condition(name="ctrl", condition_type="genotype")
            well = Well(
                plate=plate,
                name="W1",
                row=0,
                column=0,
                conditions=[cond],
            )
            session.add(well)
            session.flush()

            fov = FOV(name="fov_ctrl_0", position_index=0, well_id=well.id)
            session.add(fov)
            session.flush()

            # Two active ROIs with amplitude and frequency data
            for i in range(2):
                roi = ROI(label_value=i + 1, active=True, fov_id=fov.id)
                session.add(roi)
                session.flush()

                da = DataAnalysis(
                    roi_id=roi.id,
                    analysis_result_id=run.id,
                    peaks_amplitudes_den_dff=[1.0 + i, 2.0 + i],
                    den_dff_frequency=0.5 + 0.1 * i,
                )
                session.add(da)

            fa = FOVAnalysis(fov_id=fov.id, analysis_result_id=run.id)
            session.add(fa)
            session.commit()

        df = build_fov_feature_matrix(engine, run_id)

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1, "Expected at least one row in the feature matrix"
        # ROI-level metrics must be populated (not all-NaN)
        assert not df["mean_amplitude"].isna().all(), (
            "mean_amplitude is all-NaN — ROI metric accumulation is broken"
        )
        assert not df["mean_frequency"].isna().all(), (
            "mean_frequency is all-NaN — ROI metric accumulation is broken"
        )
    finally:
        engine.dispose(close=True)
        gc.collect()


# ---------------------------------------------------------------------------
# Additional widget fixture using shared full_db from conftest
# ---------------------------------------------------------------------------


@pytest.fixture
def full_widget(
    qtbot: QtBot,
    full_db: tuple[Engine, int],
) -> Generator[tuple[_MultilWellGraphWidget, Engine, int], None, None]:
    engine, run_id = full_db
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    qtbot.addWidget(widget)
    widget.engine = engine
    widget.run_id = run_id
    yield widget, engine, run_id


# ---------------------------------------------------------------------------
# _prepare_feature_matrix edge cases
# ---------------------------------------------------------------------------


def test_prepare_feature_matrix_drops_zero_variance() -> None:
    """Columns with zero variance are dropped before PCA."""
    df = pd.DataFrame({"varying": [1.0, 2.0, 3.0], "constant": [5.0, 5.0, 5.0]})
    X, used = _prepare_feature_matrix(df, feature_cols=["varying", "constant"])
    assert "varying" in used
    assert "constant" not in used
    assert X.shape[1] == 1


def test_prepare_feature_matrix_raises_all_constant() -> None:
    """All-constant features raise ValueError."""
    df = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]})
    with pytest.raises(ValueError, match="No features with non-zero variance"):
        _prepare_feature_matrix(df, feature_cols=["a", "b"])


# ---------------------------------------------------------------------------
# PCA rendering with rich full_db (loadings, scree, stim-split)
# ---------------------------------------------------------------------------


def test_pca_loadings_renders_bars(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    from pyqtgraph import BarGraphItem

    widget, engine, run_id = full_widget
    plot_pca_loadings(widget, "Loadings", engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


def test_pca_scree_renders_bars_and_line(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    import pyqtgraph as pg
    from pyqtgraph import BarGraphItem

    widget, engine, run_id = full_widget
    plot_pca_scree(widget, "Scree", engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    line_items = [i for i in widget.plot_item.items if isinstance(i, pg.PlotDataItem)]
    assert len(bar_items) >= 1, "Scree plot should have bars"
    assert len(line_items) >= 1, "Scree plot should have cumulative line"


def test_pca_scatter_renders_with_full_db(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    import pyqtgraph as pg

    widget, engine, run_id = full_widget
    plot_pca_scatter(widget, "PCA", engine, run_id)
    scatter_items = [
        i for i in widget.plot_item.items if isinstance(i, pg.ScatterPlotItem)
    ]
    assert len(scatter_items) >= 1


def test_pca_scatter_stim_split_no_crash_full_db(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """PCA stim-split with non-evoked data still runs without error."""
    widget, engine, run_id = full_widget
    plot_pca_scatter_stim_split(widget, "PCA stim", engine, run_id)
    assert widget.plot_item is not None


# ---------------------------------------------------------------------------
# build_fov_feature_matrix edge cases (stim, no FOVAnalysis)
# ---------------------------------------------------------------------------


def test_build_feature_matrix_burst_stats_populated(
    full_db: tuple[Engine, int],
) -> None:
    """Non-stim-split matrix should have burst stats from FOVAnalysis."""
    engine, run_id = full_db
    df = build_fov_feature_matrix(engine, run_id, include_stim_status=False)
    assert len(df) == 4
    assert not df["burst_count"].isna().all()


def test_build_feature_matrix_stim_split_sets_burst_nan(
    full_db: tuple[Engine, int],
) -> None:
    """When include_stim_status=True, burst columns are NaN."""
    engine, run_id = full_db
    df = build_fov_feature_matrix(engine, run_id, include_stim_status=True)
    if len(df) > 0:
        for col_name in ["burst_count", "burst_avg_duration_s", "burst_avg_interval_s"]:
            assert df[col_name].isna().all(), f"{col_name} should be NaN for stim-split"


def test_build_feature_matrix_stim_suffix() -> None:
    """build_fov_feature_matrix adds _stim suffix when ROI has stimulated=True."""
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
    assert any("_stim" in name for name in df["fov_name"])
    engine.dispose(close=True)


def test_build_feature_matrix_nan_burst_no_fov_analysis() -> None:
    """When FOVAnalysis is missing, burst columns should be NaN."""
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
    assert df["burst_count"].isna().all()
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# PCA error / edge paths
# ---------------------------------------------------------------------------


def test_pca_scatter_error_path_too_few_fovs(qtbot: QtBot) -> None:
    """PCA scatter with < 2 FOVs shows informative message, does not crash."""
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
    assert widget.plot_item is not None
    engine.dispose(close=True)


@pytest.mark.parametrize(
    "plot_fn",
    ["plot_pca_loadings", "plot_pca_scree"],
)
def test_pca_full_too_few_fovs(qtbot: QtBot, plot_fn: str) -> None:
    """PCA loadings/scree with insufficient data shows message without crash."""
    import importlib

    mod = importlib.import_module(
        "cali.plot._multi_wells_plots._dimensionality_reduction"
    )
    fn = getattr(mod, plot_fn)
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    fn(widget, "test", engine, run_id=None)
    assert widget.plot_item is not None
    engine.dispose(close=True)


def test_pca_scatter_hidden_condition(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """PCA scatter skips conditions with visible=False."""
    widget, engine, run_id = full_widget
    plot_pca_scatter(widget, "PCA", engine, run_id)
    # Hide one condition
    for cond_name in widget.conditions:
        widget.conditions[cond_name]["visible"] = False
        break
    plot_pca_scatter(widget, "PCA", engine, run_id)
    assert widget.plot_item is not None


@pytest.mark.parametrize(
    "patch_target,plot_fn",
    [
        (
            "cali.plot._multi_wells_plots._dimensionality_reduction"
            ".build_fov_feature_matrix",
            "plot_pca_scatter",
        ),
        (
            "cali.plot._multi_wells_plots._dimensionality_reduction.compute_pca",
            "plot_pca_scatter",
        ),
        (
            "cali.plot._multi_wells_plots._dimensionality_reduction"
            ".build_fov_feature_matrix",
            "plot_pca_loadings",
        ),
    ],
)
def test_pca_handles_exception(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
    patch_target: str,
    plot_fn: str,
) -> None:
    """PCA scatter/loadings handle RuntimeError from build_matrix or compute_pca."""
    import importlib

    mod = importlib.import_module(
        "cali.plot._multi_wells_plots._dimensionality_reduction"
    )
    fn = getattr(mod, plot_fn)
    widget, engine, run_id = full_widget
    with patch(patch_target, side_effect=RuntimeError("mock")):
        fn(widget, "test", engine, run_id)
    assert widget.plot_item is not None


# ---------------------------------------------------------------------------
# _query_roi_attribute_by_condition with include_stim_status
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
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _PCAFeaturesDialog widget tests
# ---------------------------------------------------------------------------


def test_pca_features_dialog_default_all_checked(qtbot: QtBot) -> None:
    from cali.gui._pygraph_plot_widgets import _PCAFeaturesDialog

    dialog = _PCAFeaturesDialog(
        current_features=None, experiment_type=None, enable_rising_edge=True
    )
    qtbot.addWidget(dialog)

    for feat in FEATURE_COLUMNS:
        cb = dialog._checkboxes[feat]
        assert cb.isChecked(), f"{feat} should be checked by default"
        assert cb.isEnabled(), f"{feat} should be enabled"

    assert dialog.get_features() is None


def test_pca_features_dialog_subset_selected(qtbot: QtBot) -> None:
    from cali.gui._pygraph_plot_widgets import _PCAFeaturesDialog

    selected = ["mean_amplitude", "mean_frequency"]
    dialog = _PCAFeaturesDialog(
        current_features=selected, experiment_type=None, enable_rising_edge=True
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
        current_features=None, experiment_type=EVOKED, enable_rising_edge=True
    )
    qtbot.addWidget(dialog)
    for feat in _BURST_FEATURES:
        cb = dialog._checkboxes[feat]
        assert not cb.isEnabled(), f"{feat} should be disabled for evoked"
        assert not cb.isChecked(), f"{feat} should be unchecked for evoked"


def test_pca_features_dialog_rising_edge_disabled(qtbot: QtBot) -> None:
    from cali.gui._pygraph_plot_widgets import _PCAFeaturesDialog

    dialog = _PCAFeaturesDialog(
        current_features=None, experiment_type=None, enable_rising_edge=False
    )
    qtbot.addWidget(dialog)
    cb = dialog._checkboxes["mean_spike_freq_edges"]
    assert not cb.isEnabled()
    assert not cb.isChecked()


# ---------------------------------------------------------------------------
# _MultilWellGraphWidget PCA UI interactions
# ---------------------------------------------------------------------------


def test_pca_combo_shows_features_button(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """Selecting a PCA option shows the _pca_features_btn."""
    widget, engine, run_id = full_widget
    widget._engine = engine
    widget._run_id = run_id
    widget._on_combo_changed("PCA Scatter")
    assert not widget._pca_features_btn.isHidden()
    widget._on_combo_changed("None")
    assert widget._pca_features_btn.isHidden()


def test_show_pca_features_dialog(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """_show_pca_features_dialog queries DB and processes dialog result."""
    from qtpy.QtWidgets import QDialog

    widget, engine, run_id = full_widget
    widget._engine = engine
    widget._run_id = run_id

    with patch("cali.gui._pygraph_plot_widgets._PCAFeaturesDialog") as MockDialog:
        mock_instance = MockDialog.return_value
        mock_instance.exec.return_value = QDialog.DialogCode.Accepted
        mock_instance.get_features.return_value = ["mean_amplitude", "mean_frequency"]

        widget._show_pca_features_dialog()

        MockDialog.assert_called_once()
        assert widget._pca_features == ["mean_amplitude", "mean_frequency"]
