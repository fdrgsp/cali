"""Tests for inferred spike burst bar plot functions.

Covers:
- _query_burst_metrics_by_condition returns correct structure from stored FOVAnalysis
- FOVs with zero / None spike_burst_count are excluded
- Burst rate is computed from spike_population_activity length and frame_rate
- Empty database → empty dict (no crash)
- _query_fov_scalar_by_condition: synchrony and correlation queries
- Spike/calcium plot render functions (synchrony, correlation, CCG, rising edges)
- Headless compute helpers (compute_spike_synchrony_data, etc.)
- OperationalError handlers
- _aggregate_fov_scalar_to_condition_stats edge cases
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pytest
from qtpy.QtWidgets import QWidget
from sqlmodel import Session, create_engine

from cali.plot._multi_wells_plots._inferred_spikes import (
    _query_burst_metrics_by_condition,
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

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget


# ---------------------------------------------------------------------------
# DB fixture helpers
# ---------------------------------------------------------------------------

_FRAME_RATE = 10.0  # frames per second
_N_FRAMES = 600  # 60 seconds = 1 minute recording


def _build_spike_burst_db() -> tuple[Engine, int]:
    """Return (engine, run_id) with 4 FOVs across 2 conditions with burst data."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="spike_burst_exp")
        session.add(exp)
        session.flush()

        settings = AnalysisSettings(frame_rate=_FRAME_RATE)
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

                burst_count = 3 + fov_idx
                fa = FOVAnalysis(
                    fov_id=fov.id,
                    analysis_result_id=run.id,
                    spike_burst_count=burst_count,
                    spike_burst_avg_duration=0.5 + 0.1 * fov_idx,
                    spike_burst_avg_interval=2.0 + 0.5 * fov_idx,
                    spike_population_activity=[0.0] * _N_FRAMES,
                )
                session.add(fa)

        session.commit()

    return engine, run_id


@pytest.fixture
def spike_burst_db() -> Generator[tuple[Engine, int], None, None]:
    engine, run_id = _build_spike_burst_db()
    yield engine, run_id
    engine.dispose(close=True)
    gc.collect()


# ---------------------------------------------------------------------------
# _query_burst_metrics_by_condition — structure
# ---------------------------------------------------------------------------


def test_query_returns_two_conditions(spike_burst_db: tuple[Engine, int]) -> None:
    """Returns one entry per condition in the DB."""
    engine, run_id = spike_burst_db
    data = _query_burst_metrics_by_condition(engine, run_id)
    assert set(data.keys()) == {"WT", "KO"}


def test_query_returns_two_fovs_per_condition(
    spike_burst_db: tuple[Engine, int],
) -> None:
    """Returns one entry per FOV within each condition."""
    engine, run_id = spike_burst_db
    data = _query_burst_metrics_by_condition(engine, run_id)
    for cond, fov_dict in data.items():
        assert len(fov_dict) == 2, f"Expected 2 FOVs for {cond}, got {len(fov_dict)}"


def test_query_metrics_keys_present(spike_burst_db: tuple[Engine, int]) -> None:
    """Each FOV entry contains the expected metric keys."""
    engine, run_id = spike_burst_db
    data = _query_burst_metrics_by_condition(engine, run_id)
    for cond, fov_dict in data.items():
        for fov_name, metrics in fov_dict.items():
            assert "count" in metrics, f"Missing 'count' for {cond}/{fov_name}"
            assert "avg_duration_sec" in metrics
            assert "avg_interval_sec" in metrics
            assert "rate_per_min" in metrics


def test_query_count_value(spike_burst_db: tuple[Engine, int]) -> None:
    """Stored spike_burst_count is returned as a float."""
    engine, run_id = spike_burst_db
    data = _query_burst_metrics_by_condition(engine, run_id)
    for _cond, fov_dict in data.items():
        for _fov, metrics in fov_dict.items():
            assert metrics["count"] >= 3.0


# ---------------------------------------------------------------------------
# _query_burst_metrics_by_condition — rate computation
# ---------------------------------------------------------------------------


def test_query_rate_computed_from_population_activity(
    spike_burst_db: tuple[Engine, int],
) -> None:
    """Burst rate is derived from spike_population_activity length and frame_rate.

    With _N_FRAMES=600 and frame_rate=10 fps → 1 minute recording.
    With burst_count=3 → expected rate = 3 bursts/min.
    """
    engine, run_id = spike_burst_db
    data = _query_burst_metrics_by_condition(engine, run_id)
    # fov_wt_0 has burst_count=3; recording = 600/10/60 = 1 min → rate = 3.0
    fov_wt_0 = data["WT"]["fov_wt_0"]
    expected_rate = 3.0 / (_N_FRAMES / _FRAME_RATE / 60.0)
    assert abs(fov_wt_0["rate_per_min"] - expected_rate) < 1e-9


def test_query_rate_zero_when_no_population_activity() -> None:
    """When spike_population_activity is None/empty, rate falls back to 0.0."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="no_activity_exp")
        session.add(exp)
        session.flush()

        settings = AnalysisSettings(frame_rate=_FRAME_RATE)
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

        fov = FOV(name="fov_0", position_index=0, well_id=well.id)
        session.add(fov)
        session.flush()

        fa = FOVAnalysis(
            fov_id=fov.id,
            analysis_result_id=run.id,
            spike_burst_count=5,
            spike_burst_avg_duration=0.5,
            spike_burst_avg_interval=2.0,
            spike_population_activity=None,  # no activity stored
        )
        session.add(fa)
        session.commit()

    data = _query_burst_metrics_by_condition(engine, run_id)
    assert data["WT"]["fov_0"]["rate_per_min"] == 0.0
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# _query_burst_metrics_by_condition — filtering
# ---------------------------------------------------------------------------


def test_query_excludes_fovs_with_zero_burst_count() -> None:
    """FOVs with spike_burst_count=0 are excluded from results."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="zero_burst_exp")
        session.add(exp)
        session.flush()

        settings = AnalysisSettings(frame_rate=_FRAME_RATE)
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

        fov = FOV(name="fov_zero", position_index=0, well_id=well.id)
        session.add(fov)
        session.flush()

        fa = FOVAnalysis(
            fov_id=fov.id,
            analysis_result_id=run.id,
            spike_burst_count=0,
        )
        session.add(fa)
        session.commit()

    data = _query_burst_metrics_by_condition(engine, run_id)
    assert data == {}, f"Expected empty dict, got {data}"
    engine.dispose(close=True)


def test_query_excludes_fovs_with_none_burst_count() -> None:
    """FOVs with spike_burst_count=None are excluded from results."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="none_burst_exp")
        session.add(exp)
        session.flush()

        settings = AnalysisSettings(frame_rate=_FRAME_RATE)
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

        fov = FOV(name="fov_none", position_index=0, well_id=well.id)
        session.add(fov)
        session.flush()

        fa = FOVAnalysis(
            fov_id=fov.id,
            analysis_result_id=run.id,
            spike_burst_count=None,
        )
        session.add(fa)
        session.commit()

    data = _query_burst_metrics_by_condition(engine, run_id)
    assert data == {}, f"Expected empty dict for None burst count, got {data}"
    engine.dispose(close=True)


def test_query_returns_empty_for_empty_db() -> None:
    """Empty database returns empty dict without raising."""
    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    data = _query_burst_metrics_by_condition(engine)
    assert data == {}
    engine.dispose(close=True)


def test_query_filters_by_run_id(spike_burst_db: tuple[Engine, int]) -> None:
    """Non-existent run_id returns empty dict."""
    engine, _run_id = spike_burst_db
    data = _query_burst_metrics_by_condition(engine, run_id=9999)
    assert data == {}


def test_query_uses_stored_avg_duration(spike_burst_db: tuple[Engine, int]) -> None:
    """avg_duration_sec reflects the stored spike_burst_avg_duration value."""
    engine, run_id = spike_burst_db
    data = _query_burst_metrics_by_condition(engine, run_id)
    # fov_wt_0 was stored with spike_burst_avg_duration=0.5
    fov_wt_0 = data["WT"]["fov_wt_0"]
    assert abs(fov_wt_0["avg_duration_sec"] - 0.5) < 1e-9


def test_query_uses_stored_avg_interval(spike_burst_db: tuple[Engine, int]) -> None:
    """avg_interval_sec reflects the stored spike_burst_avg_interval value."""
    engine, run_id = spike_burst_db
    data = _query_burst_metrics_by_condition(engine, run_id)
    assert data["WT"]["fov_wt_0"]["avg_interval_sec"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Widget fixture for Qt-dependent plot tests
# ---------------------------------------------------------------------------


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
# _query_fov_scalar_by_condition: synchrony and correlation queries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column",
    ["global_spike_jitter_synchrony", "global_spike_max_lag_correlation"],
)
def test_query_fov_scalar_returns_conditions(
    full_db: tuple[Engine, int],
    column: str,
) -> None:
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_fov_scalar_by_condition,
    )

    engine, run_id = full_db
    data = _query_fov_scalar_by_condition(engine, run_id, column)
    assert set(data.keys()) == {"WT", "KO"}
    for _cond, fov_dict in data.items():
        assert len(fov_dict) == 2
        for scalar, weight in fov_dict.values():
            assert isinstance(scalar, float)
            assert isinstance(weight, int)


@pytest.mark.parametrize(
    "column",
    ["global_spike_jitter_synchrony", "global_spike_max_lag_correlation"],
)
def test_query_fov_scalar_empty_db(column: str) -> None:
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_fov_scalar_by_condition,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    assert _query_fov_scalar_by_condition(engine, None, column) == {}
    engine.dispose(close=True)


def test_query_fov_scalar_no_weight(full_db: tuple[Engine, int]) -> None:
    """use_n_pairs_weight=False → each FOV has weight=1."""
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_fov_scalar_by_condition,
    )

    engine, run_id = full_db
    data = _query_fov_scalar_by_condition(
        engine, run_id, "global_spike_jitter_synchrony", use_n_pairs_weight=False
    )
    for fov_dict in data.values():
        for _scalar, weight in fov_dict.values():
            assert weight == 1


# ---------------------------------------------------------------------------
# OperationalError handlers (missing fov_analysis table)
# ---------------------------------------------------------------------------


def _make_no_table_engine() -> Engine:
    """Engine with schema but fov_analysis table dropped → triggers OperationalError."""
    from sqlalchemy import text

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS fov_analysis"))
        conn.commit()
    return engine


def test_query_burst_metrics_operational_error() -> None:
    engine = _make_no_table_engine()
    assert _query_burst_metrics_by_condition(engine) == {}
    engine.dispose(close=True)


def test_query_fov_scalar_operational_error() -> None:
    from cali.plot._multi_wells_plots._inferred_spikes import (
        _query_fov_scalar_by_condition,
    )

    engine = _make_no_table_engine()
    assert (
        _query_fov_scalar_by_condition(engine, None, "global_spike_jitter_synchrony")
        == {}
    )
    assert (
        _query_fov_scalar_by_condition(engine, None, "global_spike_max_lag_correlation")
        == {}
    )
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# Qt plot render tests (sync, correlation, freq, burst rate, CCG, rising edges)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plot_fn_name,title",
    [
        ("plot_spike_synchrony_bar_plot", "Synchrony"),
        ("plot_spike_correlation_bar_plot", "Correlation"),
        ("plot_inferred_spikes_frequency_bar_plot", "Spike Freq"),
        ("plot_inferred_spikes_rising_edge_frequency_bar_plot", "Spike RE Freq"),
        ("plot_burst_rate_bar_plot", "Burst Rate"),
        ("plot_calcium_dff_correlation_bar_plot", "DFF Corr"),
        ("plot_calcium_den_dff_correlation_bar_plot", "Den DFF Corr"),
        ("plot_spike_synchrony_rising_edges_bar_plot", "Sync RE"),
        ("plot_spike_correlation_rising_edges_bar_plot", "Corr RE"),
        ("plot_fraction_significant_ccg_pairs_bar_plot", "CCG"),
        ("plot_fraction_significant_ccg_pairs_rising_edges_bar_plot", "CCG RE"),
    ],
)
def test_inferred_spike_bar_plot_renders(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
    plot_fn_name: str,
    title: str,
) -> None:
    import importlib

    from pyqtgraph import BarGraphItem

    mod = importlib.import_module("cali.plot._multi_wells_plots._inferred_spikes")
    plot_fn = getattr(mod, plot_fn_name)

    widget, engine, run_id = full_widget
    plot_fn(widget, title, engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


@pytest.mark.parametrize(
    "plot_fn_name",
    ["plot_spike_synchrony_bar_plot", "plot_spike_correlation_bar_plot"],
)
def test_inferred_spike_plot_empty_db_no_crash(qtbot: QtBot, plot_fn_name: str) -> None:
    import importlib

    from cali.gui._pygraph_plot_widgets import _MultilWellGraphWidget

    mod = importlib.import_module("cali.plot._multi_wells_plots._inferred_spikes")
    plot_fn = getattr(mod, plot_fn_name)

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    parent = QWidget()
    widget = _MultilWellGraphWidget(parent)
    qtbot.addWidget(parent)
    plot_fn(widget, "test", engine, run_id=None)
    engine.dispose(close=True)


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
# Headless compute helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "compute_fn_name,expected_name",
    [
        ("compute_burst_count_data", "Burst Count"),
        ("compute_burst_avg_duration_data", None),
        ("compute_spike_synchrony_data", "Spike Jitter Synchrony"),
        ("compute_spike_correlation_data", None),
        ("compute_calcium_dff_correlation_data", "Calcium ΔF/F Correlation"),
        ("compute_calcium_den_dff_correlation_data", None),
        ("compute_spike_synchrony_rising_edges_data", None),
        ("compute_spike_correlation_rising_edges_data", None),
        ("compute_fraction_significant_ccg_pairs_data", None),
        ("compute_fraction_significant_ccg_pairs_rising_edges_data", None),
    ],
)
def test_headless_compute_returns_result(
    full_db: tuple[Engine, int],
    compute_fn_name: str,
    expected_name: str | None,
) -> None:
    import importlib

    mod = importlib.import_module("cali.plot._multi_wells_plots._inferred_spikes")
    fn = getattr(mod, compute_fn_name)

    engine, run_id = full_db
    result = fn(engine, run_id)
    assert result is not None
    if expected_name is not None:
        _bar_data, name, _units = result
        assert name == expected_name


def test_compute_fov_scalar_data_empty_db() -> None:
    """compute_spike_synchrony_data returns None for empty DB."""
    from cali.plot._multi_wells_plots._inferred_spikes import (
        compute_spike_synchrony_data,
    )

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    assert compute_spike_synchrony_data(engine, None) is None
    engine.dispose(close=True)


# ---------------------------------------------------------------------------
# _aggregate_fov_scalar_to_condition_stats edge cases
# ---------------------------------------------------------------------------


def test_aggregate_fov_scalar_empty_fov_dict_skipped() -> None:
    """Empty fov_dict in a condition is skipped."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_scalar_to_condition_stats,
    )

    data: dict[str, dict[str, tuple[float, int]]] = {
        "WT": {},
        "KO": {"fov_0": (0.5, 3)},
    }
    result = _aggregate_fov_scalar_to_condition_stats(data)
    assert result["conditions"] == ["KO"]


def test_aggregate_fov_scalar_single_fov_zero_sem() -> None:
    """Single FOV → SEM=0."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_scalar_to_condition_stats,
    )

    data: dict[str, dict[str, tuple[float, int]]] = {"WT": {"fov_0": (0.7, 3)}}
    result = _aggregate_fov_scalar_to_condition_stats(data)
    assert result["conditions"] == ["WT"]
    assert result["sems"][0] == 0.0


def test_aggregate_fov_scalar_equal_weights_zero_variance() -> None:
    """Two FOVs with identical values → variance=0, SEM=0."""
    from cali.plot._multi_wells_plots._util import (
        _aggregate_fov_scalar_to_condition_stats,
    )

    data: dict[str, dict[str, tuple[float, int]]] = {
        "WT": {"fov_0": (0.5, 1), "fov_1": (0.5, 1)},
    }
    result = _aggregate_fov_scalar_to_condition_stats(data)
    assert result["means"][0] == 0.5
    assert result["sems"][0] == 0.0


# ---------------------------------------------------------------------------
# Additional render tests: calcium DFF correlation + rising-edge plots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plot_fn_name",
    [
        "plot_calcium_dff_correlation_bar_plot",
        "plot_calcium_den_dff_correlation_bar_plot",
        "plot_spike_synchrony_rising_edges_bar_plot",
        "plot_spike_correlation_rising_edges_bar_plot",
        "plot_fraction_significant_ccg_pairs_bar_plot",
        "plot_fraction_significant_ccg_pairs_rising_edges_bar_plot",
    ],
)
def test_inferred_spike_extra_plot_renders(
    full_widget: tuple[_MultilWellGraphWidget, Engine, int],
    plot_fn_name: str,
) -> None:
    """Render tests for DFF-correlation and rising-edge plot functions."""
    import importlib

    from pyqtgraph import BarGraphItem

    mod = importlib.import_module("cali.plot._multi_wells_plots._inferred_spikes")
    fn = getattr(mod, plot_fn_name)
    widget, engine, run_id = full_widget
    fn(widget, plot_fn_name, engine, run_id)
    bar_items = [i for i in widget.plot_item.items if isinstance(i, BarGraphItem)]
    assert len(bar_items) >= 1


# ---------------------------------------------------------------------------
# Calcium burst OperationalError safety
# ---------------------------------------------------------------------------


def test_query_calcium_burst_metrics_operational_error() -> None:
    """_query_calcium_burst_metrics_by_condition returns {} on OperationalError."""
    from cali.plot._multi_wells_plots._calcium_peaks import (
        _query_calcium_burst_metrics_by_condition,
    )

    engine = _make_no_table_engine()
    assert _query_calcium_burst_metrics_by_condition(engine) == {}
    engine.dispose(close=True)
