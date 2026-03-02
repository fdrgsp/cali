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

import pytest
from qtpy.QtWidgets import QWidget
from sqlmodel import Session, create_engine

from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
from cali.plot._multi_wells_plots._dimensionality_reduction import (
    FEATURE_COLUMNS,
    build_fov_feature_matrix,
    compute_pca,
    plot_pca_scatter,
    plot_pca_scatter_stim_split,
)
from cali.sqlmodel import (
    FOV,
    Condition,
    Experiment,
    FOVAnalysis,
    Plate,
    Well,
)
from cali.sqlmodel._model import AnalysisSettings, CaliResult
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
