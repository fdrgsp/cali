"""Tests for calcium burst bar plot functions.

Covers:
- plot_calcium_burst_count_bar_plot
- plot_calcium_burst_avg_duration_bar_plot
- plot_calcium_burst_avg_interval_bar_plot
- Empty database / missing data → no crash
- _query_calcium_burst_metrics_by_condition returns correct structure
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pytest
from pyqtgraph import BarGraphItem
from qtpy.QtWidgets import QWidget
from sqlmodel import Session, create_engine

from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget
from cali.plot._multi_wells_plots._calcium_peaks import (
    _query_calcium_burst_metrics_by_condition,
    plot_calcium_burst_avg_duration_bar_plot,
    plot_calcium_burst_avg_interval_bar_plot,
    plot_calcium_burst_count_bar_plot,
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

if TYPE_CHECKING:
    from collections.abc import Generator

    from pytestqt.qtbot import QtBot
    from sqlalchemy.engine import Engine


# ---------------------------------------------------------------------------
# DB fixture: 2 conditions, 2 FOVs each, with calcium burst data
# ---------------------------------------------------------------------------


def _build_calcium_burst_db() -> tuple[Engine, int]:
    """Return (engine, run_id) with 4 FOVs across 2 conditions with burst data."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="calcium_burst_exp")
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
                    calcium_burst_count=3 + fov_idx,
                    calcium_burst_avg_duration=0.8 + 0.1 * fov_idx,
                    calcium_burst_avg_interval=3.0 + 0.5 * fov_idx,
                )
                session.add(fa)

        session.commit()

    return engine, run_id


@pytest.fixture
def calcium_burst_db() -> Generator[tuple[Engine, int], None, None]:
    engine, run_id = _build_calcium_burst_db()
    yield engine, run_id
    engine.dispose(close=True)
    gc.collect()


@pytest.fixture
def calcium_burst_widget(
    qtbot: QtBot,
    calcium_burst_db: tuple[Engine, int],
) -> Generator[tuple[_MultilWellGraphWidget, Engine, int], None, None]:
    engine, run_id = calcium_burst_db
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
def empty_widget(qtbot: QtBot) -> Generator[_MultilWellGraphWidget, None, None]:
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
# _query_calcium_burst_metrics_by_condition
# ---------------------------------------------------------------------------


def test_query_returns_two_conditions(calcium_burst_db: tuple[Engine, int]) -> None:
    engine, run_id = calcium_burst_db
    data = _query_calcium_burst_metrics_by_condition(engine, run_id)
    assert set(data.keys()) == {"WT", "KO"}


def test_query_returns_two_fovs_per_condition(
    calcium_burst_db: tuple[Engine, int],
) -> None:
    engine, run_id = calcium_burst_db
    data = _query_calcium_burst_metrics_by_condition(engine, run_id)
    for cond, fov_dict in data.items():
        assert len(fov_dict) == 2, f"Expected 2 FOVs for {cond}, got {len(fov_dict)}"


def test_query_metrics_keys(calcium_burst_db: tuple[Engine, int]) -> None:
    engine, run_id = calcium_burst_db
    data = _query_calcium_burst_metrics_by_condition(engine, run_id)
    for cond, fov_dict in data.items():
        for fov_name, metrics in fov_dict.items():
            assert "count" in metrics, f"Missing 'count' for {cond}/{fov_name}"
            assert "avg_duration_s" in metrics
            assert "avg_interval_s" in metrics


def test_query_returns_empty_for_empty_db() -> None:
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    data = _query_calcium_burst_metrics_by_condition(engine)
    assert data == {}
    engine.dispose(close=True)


def test_query_excludes_zero_count_fovs() -> None:
    """FOVs with calcium_burst_count=0 must NOT appear in the query output.

    Regression test: previously only None was filtered out, so FOVs with
    count=0 (no bursts detected) produced bars at height 0 in the plots.
    """
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="zero_burst_exp")
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

        cond = Condition(name="WT", condition_type="genotype")
        well = Well(plate=plate, name="W0", row=0, column=0, conditions=[cond])
        session.add(well)
        session.flush()

        fov_zero = FOV(name="fov_zero", position_index=0, well_id=well.id)
        session.add(fov_zero)
        session.flush()

        # burst_count = 0 — no bursts at all
        fa_zero = FOVAnalysis(
            fov_id=fov_zero.id,
            analysis_result_id=run.id,
            calcium_burst_count=0,
            calcium_burst_avg_duration=None,
            calcium_burst_avg_interval=None,
        )
        session.add(fa_zero)
        session.commit()

    data = _query_calcium_burst_metrics_by_condition(engine, run_id)
    # The FOV with count=0 must be completely absent
    assert data == {}, f"Expected empty dict, got {data}"
    engine.dispose(close=True)


def test_plot_calcium_burst_count_renders_bars(
    calcium_burst_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    widget, engine, run_id = calcium_burst_widget
    plot_calcium_burst_count_bar_plot(widget, "Calcium Burst Count", engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


def test_plot_calcium_burst_count_two_bars(
    calcium_burst_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    """One bar per condition (2 conditions in fixture)."""
    widget, engine, run_id = calcium_burst_widget
    plot_calcium_burst_count_bar_plot(widget, "Calcium Burst Count", engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    total_bars = sum(len(b.opts.get("height", [])) for b in bar_items)
    assert total_bars == 2


def test_plot_calcium_burst_count_no_crash_empty_db(
    empty_widget: _MultilWellGraphWidget,
) -> None:
    plot_calcium_burst_count_bar_plot(
        empty_widget, "Calcium Burst Count", empty_widget.engine, empty_widget.run_id
    )


# ---------------------------------------------------------------------------
# plot_calcium_burst_avg_duration_bar_plot
# ---------------------------------------------------------------------------


def test_plot_calcium_burst_avg_duration_renders_bars(
    calcium_burst_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    widget, engine, run_id = calcium_burst_widget
    plot_calcium_burst_avg_duration_bar_plot(
        widget, "Calcium Burst Duration", engine, run_id
    )
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


def test_plot_calcium_burst_avg_duration_no_crash_empty_db(
    empty_widget: _MultilWellGraphWidget,
) -> None:
    plot_calcium_burst_avg_duration_bar_plot(
        empty_widget, "Calcium Burst Duration", empty_widget.engine, empty_widget.run_id
    )


# ---------------------------------------------------------------------------
# plot_calcium_burst_avg_interval_bar_plot
# ---------------------------------------------------------------------------


def test_plot_calcium_burst_avg_interval_renders_bars(
    calcium_burst_widget: tuple[_MultilWellGraphWidget, Engine, int],
) -> None:
    widget, engine, run_id = calcium_burst_widget
    plot_calcium_burst_avg_interval_bar_plot(
        widget, "Calcium Burst Interval", engine, run_id
    )
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


def test_plot_calcium_burst_avg_interval_no_crash_empty_db(
    empty_widget: _MultilWellGraphWidget,
) -> None:
    plot_calcium_burst_avg_interval_bar_plot(
        empty_widget,
        "Calcium Burst Interval",
        empty_widget.engine,
        empty_widget.run_id,
    )


# ---------------------------------------------------------------------------
# Calcium burst headless compute
# ---------------------------------------------------------------------------


def _build_db_with_calcium_bursts() -> tuple[Engine, int]:
    """In-memory DB with one condition/FOV with calcium burst data."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="burst_exp")
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
        well = Well(plate=plate, name="W1", row=0, column=0, conditions=[cond])
        session.add(well)
        session.flush()

        fov = FOV(name="fov_0", position_index=0, well_id=well.id)
        session.add(fov)
        session.flush()

        fa = FOVAnalysis(
            fov_id=fov.id,
            analysis_result_id=run.id,
            calcium_burst_count=5,
            calcium_burst_avg_duration=1.2,
            calcium_burst_avg_interval=3.5,
        )
        session.add(fa)
        session.commit()
        run_id: int = run.id  # type: ignore[assignment]

    return engine, run_id


def test_compute_calcium_burst_count_data() -> None:
    from cali.plot._multi_wells_plots._calcium_peaks import (
        compute_calcium_burst_count_data,
    )

    engine, run_id = _build_db_with_calcium_bursts()
    result = compute_calcium_burst_count_data(engine, run_id)
    assert result is not None
    bar_data, name, _units = result
    assert name == "Calcium Burst Count"
    assert bar_data["means"][0] == 5.0
    engine.dispose(close=True)


def test_compute_calcium_burst_avg_duration_data() -> None:
    from cali.plot._multi_wells_plots._calcium_peaks import (
        compute_calcium_burst_avg_duration_data,
    )

    engine, run_id = _build_db_with_calcium_bursts()
    result = compute_calcium_burst_avg_duration_data(engine, run_id)
    assert result is not None
    engine.dispose(close=True)


def test_compute_calcium_burst_empty_db() -> None:
    """compute_calcium_burst_count_data returns None for empty DB."""
    from cali.plot._multi_wells_plots._calcium_peaks import (
        compute_calcium_burst_count_data,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    assert compute_calcium_burst_count_data(engine, None) is None
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Stim-split empty data paths
# ---------------------------------------------------------------------------


def test_calcium_peaks_amplitude_stim_split_empty(qtbot: QtBot) -> None:
    """plot_calcium_peaks_amplitude_stim_split_bar_plot handles empty evoked data."""
    from cali.plot._multi_wells_plots._calcium_peaks import (
        plot_calcium_peaks_amplitude_stim_split_bar_plot,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    plot_calcium_peaks_amplitude_stim_split_bar_plot(widget, "Amp Stim", engine, None)
    engine.dispose(close=True)
