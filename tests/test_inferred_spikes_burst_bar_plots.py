"""Tests for inferred spike burst bar plot functions.

Covers:
- _query_burst_metrics_by_condition returns correct structure from stored FOVAnalysis
- FOVs with zero / None spike_burst_count are excluded
- Burst rate is computed from spike_population_activity length and frame_rate
- Empty database → empty dict (no crash)
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pytest
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

    from sqlalchemy.engine import Engine


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
    # fov_wt_0 was stored with spike_burst_avg_interval=2.0
    fov_wt_0 = data["WT"]["fov_wt_0"]
    assert abs(fov_wt_0["avg_interval_sec"] - 2.0) < 1e-9
