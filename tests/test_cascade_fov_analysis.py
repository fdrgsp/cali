"""FOV populations, timing and persistence stay independent across spike outputs."""

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine, event
from sqlmodel import Session, select

from cali.analysis._fov_analysis import compute_fov_analysis
from cali.analysis._fov_analysis_parallel import (
    _compute_ccg_for_pair,
    _compute_jitter_for_pair,
    compute_fov_analysis_parallel,
)
from cali.analysis._fov_inputs import collect_spikes
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    Experiment,
    ExtractionFrameWindow,
    FOVAnalysis,
    SpikeAnalysis,
    SpikeAnalysisSettings,
    SpikeFOVAnalysis,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    _engine,
    create_cali_engine,
    create_database_and_tables,
    ensure_schema_current,
)
from cali.sqlmodel._spike_fov_analysis import SPIKE_FOV_METRICS


def _settings(
    methods: tuple = ("oasis", "cascade"), **kwargs: object
) -> AnalysisSettings:
    return AnalysisSettings(
        frame_rate=25,  # OASIS's legacy FOV rate; CASCADE must use recorded 10 Hz.
        spike_settings=[
            SpikeAnalysisSettings(
                method=method,
                threshold_mode="global",
                threshold_value=0.2,
                enable_rising_edge_analysis=method == "oasis",
                spikes_sync_cross_corr_lag=200 if method == "oasis" else 300,
                spikes_sync_jitter_window=40 if method == "oasis" else 200,
                ccg_n_shuffles=3 if method == "oasis" else 4,
                burst_threshold=25 if method == "oasis" else 50,
                burst_min_duration=40 if method == "oasis" else 100,
                burst_gaussian_sigma=0.1,
            )
            for method in methods
        ],
        **kwargs,
    )


def _fov(methods: tuple = ("oasis", "cascade"), count: int = 3) -> FOV:
    fov = FOV(name="independent populations", position_index=0)
    runs = {
        method: SpikeInferenceRun(
            method=method,
            units="a.u." if method == "oasis" else "spikes/frame",
            backend_package="oasis-deconv" if method == "oasis" else "CascadeTorch",
            model_sampling_rate_hz=10 if method == "cascade" else None,
        )
        for method in methods
    }
    for label in range(1, count + 1):
        roi = ROI(label_value=label, active=label != 3, fov=fov)
        calcium = np.sin(np.arange(100) / 10 + label / 4)
        spikes = []
        metrics = []
        for method in methods:
            active = label in (1, 3) if method == "oasis" else label != 1
            start, stop = (
                (0, 100)
                if method == "oasis"
                else ((20, 85) if label == 3 else (10, 90))
            )
            values = np.zeros(100)
            if active:
                values[[start, 30, 70]] = 0.5
            if method == "cascade":
                values[:start] = 100
                values[stop:] = 100
            spike = SpikeTrace(
                values=values.tolist(),
                valid_start=start,
                valid_stop=stop,
                inference_run=runs[method],
            )
            spikes.append(spike)
            metrics.append(
                SpikeAnalysis(
                    spike_trace=spike,
                    method=method,
                    units=runs[method].units,
                    threshold=0.2,
                    threshold_mode="global",
                    spike_active=active,
                )
            )
        trace = Traces(
            roi=roi,
            dff=calcium.tolist(),
            den_dff=calcium.tolist(),
            x_axis=(np.arange(100) * 100.0).tolist(),
            x_axis_units="ms",
            extraction_frame_window=ExtractionFrameWindow(
                source_start_frame=10,
                source_start_time_ms=1000,
                retained_frame_count=100,
                original_frame_count=110,
                acquisition_frame_rate_hz=10,
            ),
            spike_traces=spikes,
        )
        analysis = DataAnalysis(
            roi=roi,
            calcium_active=label in (1, 2),
            peaks_den_dff=[30, 70] if label in (1, 2) else [],
            spike_analyses=metrics,
        )
        roi._new_traces = [trace]
        roi._new_data_analysis = [analysis]
    return fov


@pytest.mark.parametrize(
    "compute", [compute_fov_analysis, compute_fov_analysis_parallel]
)
def test_populations_use_their_own_flags_intervals_units_and_rates(
    compute: Callable[[FOV, AnalysisSettings], FOVAnalysis | None],
) -> None:
    fov = _fov()
    result = compute(fov, _settings())
    assert result.calcium_active_roi_labels == [1, 2]
    assert np.asarray(result.calcium_dff_correlation_matrix).shape == (2, 2)
    oasis = result.get_spike_analysis("oasis")
    cascade = result.get_spike_analysis("cascade")
    assert oasis.active_roi_labels == [1, 3]
    assert cascade.active_roi_labels == [
        2,
        3,
    ]  # ROI 3's union flag was deliberately False.
    assert (oasis.valid_start, oasis.valid_stop, oasis.frame_rate_hz) == (0, 100, 25)
    assert (cascade.valid_start, cascade.valid_stop, cascade.frame_rate_hz) == (
        20,
        85,
        10,
    )
    assert oasis.units == "a.u." and cascade.units == "spikes/frame"
    for child in (oasis, cascade):
        for name in (
            "spike_max_lag_correlation_matrix",
            "spike_jitter_synchrony_matrix",
        ):
            assert np.asarray(getattr(child, name)).shape == (2, 2)
        assert (
            len(child.spike_population_activity_raw)
            == child.valid_stop - child.valid_start
        )
        assert all(
            child.valid_start <= x <= child.valid_stop
            for x in child.spike_burst_starts or []
        )
    assert oasis.spike_max_lag_correlation_matrix_rising_edges is not None
    assert cascade.spike_max_lag_correlation_matrix_rising_edges is None
    population = collect_spikes(fov, _settings(), "cascade")
    assert population.binary["3"][0] == 1
    assert population.onsets["3"][0] == 0  # no onset created by intersecting intervals
    assert np.flatnonzero(population.onsets["3"]).tolist() == [10, 50]
    assert (
        sum(population.binary["2"]) == 2
    )  # pre-intersection event and padding excluded


def test_calcium_and_each_methods_deterministic_metrics_match_single_output() -> None:
    dual = compute_fov_analysis(_fov(), _settings())
    for method in ("oasis", "cascade"):
        single = compute_fov_analysis(_fov((method,)), _settings((method,)))
        assert single.calcium_active_roi_labels == dual.calcium_active_roi_labels
        assert (
            single.calcium_dff_correlation_matrix == dual.calcium_dff_correlation_matrix
        )
        assert (
            single.calcium_population_activity_raw
            == dual.calcium_population_activity_raw
        )
        for name in SPIKE_FOV_METRICS:
            # Shift-predictor scores are stochastic; all input-derived metrics match.
            if "zscore" not in name and "significant" not in name:
                assert single.get_spike_metric(method, name) == dual.get_spike_metric(
                    method, name
                )


def test_spikes_only_do_not_require_calcium_traces_or_calcium_activity() -> None:
    fov = _fov(("cascade",))
    for roi in fov.rois:
        roi._new_traces[-1].dff = None
        roi._new_traces[-1].den_dff = None
    result = compute_fov_analysis(fov, _settings(("cascade",), enable_calcium=False))
    assert result.calcium_active_roi_labels == []
    assert result.calcium_dff_correlation_matrix is None
    assert result.get_spike_roi_labels("cascade") == [2, 3]


@pytest.mark.parametrize("active_labels", [(), (2,)])
def test_insufficient_method_still_records_membership_and_source(
    active_labels: tuple,
) -> None:
    fov = _fov()
    for roi in fov.rois:
        roi._new_data_analysis[-1].get_spike_analysis("cascade").spike_active = (
            roi.label_value in active_labels
        )
    result = compute_fov_analysis(fov, _settings())
    child = result.get_spike_analysis("cascade")
    assert child.active_roi_labels == list(active_labels)
    assert (
        child.inference_run
        is fov.rois[0]._new_traces[-1].get_spike_trace("cascade").inference_run
    )
    assert (child.valid_start, child.valid_stop, child.frame_rate_hz) == (
        (10, 90, 10) if active_labels else (None, None, None)
    )
    assert child.spike_max_lag_correlation_matrix is None


@pytest.mark.parametrize(
    "problem, message",
    [
        ("run", "inference run"),
        ("time", "time axis"),
        ("window", "frame window"),
        ("length", "matching lengths"),
        ("interval", "common valid"),
        ("rate", "acquisition rate"),
        ("units", "method and units"),
        ("source", "selected stored trace"),
    ],
)
def test_incompatible_inputs_fail_before_any_fov_product_is_attached(
    problem: str, message: str
) -> None:
    fov = _fov()
    trace = fov.rois[2]._new_traces[-1]
    spike = trace.get_spike_trace("cascade")
    if problem == "run":
        spike.inference_run = SpikeInferenceRun(
            method="cascade", units="spikes/frame", backend_package="CascadeTorch"
        )
    elif problem == "time":
        trace.x_axis = [x + 1 for x in trace.x_axis]
    elif problem == "window":
        trace.extraction_frame_window.source_start_frame += 1
    elif problem == "length":
        spike.values = [*spike.values, 0]
    elif problem == "interval":
        spike.valid_start, spike.valid_stop = 90, 100
    elif problem == "rate":
        trace.extraction_frame_window.acquisition_frame_rate_hz = 10.05
    elif problem == "units":
        fov.rois[2]._new_data_analysis[-1].get_spike_analysis("cascade").units = "a.u."
    elif problem == "source":
        fov.rois[2]._new_data_analysis[-1].get_spike_analysis("cascade").spike_trace = (
            fov.rois[1]._new_traces[-1].get_spike_trace("cascade")
        )
    with pytest.raises(ValueError, match=message):
        compute_fov_analysis(fov, _settings())
    assert not hasattr(fov, "_new_fov_analysis")


def test_each_methods_settings_reach_pairwise_and_population_calculations() -> None:
    calls = []
    burst_calls = []

    def pairwise(data: dict, **kwargs: Any) -> tuple:
        calls.append((list(data), len(next(iter(data.values()))), kwargs))
        return np.eye(2), np.zeros((2, 2), dtype=int), np.eye(2)

    def bursts(**kwargs: Any) -> tuple:
        burst_calls.append(kwargs)
        size = len(kwargs["spike_trains"][0])
        return 1, 0.1, None, [10], [11], np.zeros(size), np.zeros(size)

    with (
        patch("cali.analysis._fov_metrics._get_spike_correlations_matrix", pairwise),
        patch(
            "cali.analysis._fov_analysis_parallel._detect_spikes_population_bursts",
            bursts,
        ),
    ):
        result = compute_fov_analysis(_fov(), _settings())
    oasis, cascade = calls[0], calls[4]
    assert oasis[:2] == (["1", "3"], 100)
    assert oasis[2] == {"method": "cross_correlation", "max_lag": 5, "n_shuffles": 3}
    assert cascade[:2] == (["2", "3"], 65)
    assert cascade[2] == {"method": "cross_correlation", "max_lag": 3, "n_shuffles": 4}
    assert len(calls) == 6  # OASIS binary and onsets; CASCADE binary only
    assert [
        (x["frame_rate"], x["burst_threshold_percent"], x["min_duration_ms"])
        for x in burst_calls
    ] == [(25, 25, 40), (10, 50, 100)]
    assert result.get_spike_analysis("cascade").spike_burst_starts == [30]
    assert result.get_spike_analysis("cascade").spike_burst_ends == [31]


def test_large_method_population_drives_pool_dimensions_and_valid_worker_inputs() -> (
    None
):
    fov = _fov(("cascade",), count=11)
    captured = []

    def mapped(worker: Callable, args: list) -> list:
        captured.append((worker, args))
        return [worker(arg) for arg in args]

    pool = MagicMock()
    pool.__enter__.return_value.map.side_effect = mapped
    with patch("cali.analysis._fov_analysis_parallel.mp.get_context") as context:
        context.return_value.Pool.return_value = pool
        result = compute_fov_analysis_parallel(fov, _settings(("cascade",)))
    child = result.get_spike_analysis("cascade")
    assert child.active_roi_labels == list(range(2, 12))
    assert np.asarray(child.spike_ccg_zscore_matrix).shape == (10, 10)
    assert [worker for worker, _ in captured] == [
        _compute_ccg_for_pair,
        _compute_jitter_for_pair,
    ]
    assert [len(args) for _, args in captured] == [45, 45]
    assert all(len(arg[2]) == len(arg[3]) == 65 for _, args in captured for arg in args)
    assert all(arg[-2:] == (3, 4) for arg in captured[0][1])


def test_persisted_population_coordinates_and_provenance_roundtrip(
    tmp_path: Path,
) -> None:
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'fov.cali'}")
    create_database_and_tables(engine)
    fov = _fov()
    result = compute_fov_analysis(fov, _settings())
    result.fov = fov
    with Session(engine) as session:
        session.add_all([fov, result])
        session.commit()
    with Session(engine) as session:
        reloaded = session.exec(select(FOVAnalysis)).one()
        snapshot = FOVAnalysis.model_validate_json(reloaded.model_dump_json())
        for parent in (reloaded, snapshot):
            child = parent.get_spike_analysis("cascade")
            assert (child.valid_start, child.valid_stop, child.frame_rate_hz) == (
                20,
                85,
                10,
            )
            assert child.active_roi_labels == [2, 3]
            assert child.inference_run.method == "cascade"
            assert child.spike_inference_run_id is not None
            assert len(child.spike_population_activity_raw) == 65
    engine.dispose()


@pytest.mark.parametrize("flush_each", [True, False])
def test_multiple_fovs_retarget_results_to_canonical_inference_runs(
    flush_each: bool,
    tmp_path: Path,
) -> None:
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'plate.cali'}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            experiment = Experiment(name="shared inference provenance")
            session.add(experiment)
            session.flush()
            owner = CaliResult(experiment=experiment.id, positions_extracted=[0, 1, 2])
            session.add(owner)
            session.flush()
            for position in range(3):
                fov = _fov()
                fov.name = f"A1_{position}"
                fov.position_index = position
                for roi in fov.rois:
                    roi._new_traces[-1].analysis_result = owner
                    roi._new_data_analysis[-1].analysis_result = owner
                parent = compute_fov_analysis(fov, _settings())
                parent.fov = fov
                parent.analysis_result = owner
                session.add_all([fov, parent])
                if flush_each:
                    session.flush()
            session.commit()
            runs = session.exec(select(SpikeInferenceRun)).all()
            assert len(runs) == len(owner.spike_inference_runs) == 2
            canonical = {run.method: run for run in runs}
            parents = session.exec(select(FOVAnalysis)).all()
            assert len(parents) == 3
            for parent in parents:
                for child in parent.spike_analyses:
                    assert child.inference_run is canonical[child.method]
                    assert child.spike_inference_run_id == canonical[child.method].id
    finally:
        engine.dispose()


def test_v11_upgrade_preserves_metrics_and_unknown_coordinates_with_atomic_retry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v11.cali"
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        session.add(FOVAnalysis(id=1))
        session.commit()
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO spike_fov_analysis "
            "(fov_analysis_id,method,units,spike_burst_count,provenance_source) "
            "VALUES (1,'oasis','a.u.',7,'legacy_import')"
        )
        for name in ("valid_start", "valid_stop", "frame_rate_hz"):
            connection.exec_driver_sql(
                f"ALTER TABLE spike_fov_analysis DROP COLUMN {name}"
            )
        connection.exec_driver_sql("PRAGMA user_version=11")
    engine.dispose()
    engine = create_engine(f"sqlite:///{path}")

    def interrupt(*args: Any) -> None:
        if "ADD COLUMN valid_stop" in args[2]:
            raise RuntimeError("interrupted interval migration")

    event.listen(engine, "after_cursor_execute", interrupt)
    with pytest.raises(RuntimeError, match="interrupted interval"):
        ensure_schema_current(engine)
    event.remove(engine, "after_cursor_execute", interrupt)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 11
        assert "valid_start" not in {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info(spike_fov_analysis)"
            )
        }
    ensure_schema_current(engine)
    with engine.connect() as connection:
        assert (
            connection.exec_driver_sql("PRAGMA user_version").scalar()
            == _engine.SCHEMA_VERSION
        )
        assert connection.exec_driver_sql(
            "SELECT spike_burst_count,valid_start,valid_stop,frame_rate_hz "
            "FROM spike_fov_analysis"
        ).one() == (7, None, None, None)
    engine.dispose()


@pytest.mark.parametrize(
    "change",
    [
        {"valid_stop": None},
        {"frame_rate_hz": 0},
        {"valid_start": -1},
        {"spike_population_activity_raw": [1]},
        {"spike_burst_starts": [5]},
    ],
)
def test_invalid_population_coordinates_rejected_by_snapshots_and_database(
    change: dict,
    tmp_path: Path,
) -> None:
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'invalid.cali'}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            child = SpikeFOVAnalysis(
                valid_start=10,
                valid_stop=20,
                frame_rate_hz=10,
                provenance_source="legacy_import",
            )
            parent = FOVAnalysis(spike_analyses=[child])
            session.add(parent)
            session.commit()
            for name, value in change.items():
                setattr(child, name, value)
            with pytest.raises(ValueError, match="FOV spike"):
                SpikeFOVAnalysis.model_validate(child.model_dump())
            with pytest.raises(ValueError, match="FOV spike"):
                session.commit()
            session.rollback()
    finally:
        engine.dispose()
