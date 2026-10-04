"""Noise QC stays independent of activity, model scale and historical results."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, event
from sqlmodel import Session

from cali._constants import CASCADE_EXPECTED_SPIKES_TRACES
from cali.analysis._fov_analysis import compute_fov_analysis
from cali.analysis._fov_analysis_parallel import compute_fov_analysis_parallel
from cali.analysis._noise_qc import NoiseQCCollection, summarize_noise
from cali.analysis._roi_analysis import analyze_roi_calcium
from cali.sqlmodel import (
    CaliResult,
    DataAnalysis,
    Experiment,
    FOVAnalysis,
    SpikeFOVAnalysis,
    SpikeInferenceRun,
    _engine,
    create_cali_engine,
    create_database_and_tables,
    ensure_schema_current,
)
from cali.sqlmodel._spike_fov_analysis import SPIKE_FOV_METRICS
from cali.util import export_noise_qc_to_csv
from cali.util._database_to_csv import export_traces_to_csv

from .test_cascade_fov_analysis import _fov, _settings


@pytest.mark.parametrize("parallel", [False, True])
def test_known_noise_includes_inactive_rois_without_changing_metrics(
    parallel: bool,
) -> None:
    compute = compute_fov_analysis_parallel if parallel else compute_fov_analysis
    baseline = compute(_fov(), _settings())
    fov = _fov()
    for roi, calcium, model in zip(fov.rois, [1, 3, 9], [2, 4, 12], strict=True):
        trace = roi._new_traces[-1]
        trace.calcium_noise = calcium
        trace.get_spike_trace("oasis").noise = 1000
        trace.get_spike_trace("cascade").noise = model
    result = compute(fov, _settings())
    assert (result.calcium_noise_median, result.calcium_noise_iqr) == (3, 4)
    assert result.calcium_noise_roi_count == 3
    cascade = result.get_spike_analysis("cascade")
    assert (cascade.model_noise_median, cascade.model_noise_iqr) == (4, 5)
    assert cascade.model_noise_roi_count == 3
    assert result.get_spike_analysis("oasis").model_noise_median is None
    assert result.get_spike_analysis("oasis").model_noise_roi_count is None
    # Existing calcium and deterministic spike fields are unaffected by summaries.
    for name in type(result).model_fields:
        if name not in {"id", "created_at"} and not name.startswith("calcium_noise_"):
            assert getattr(result, name) == getattr(baseline, name)
    for method in ("oasis", "cascade"):
        a, b = result.get_spike_analysis(method), baseline.get_spike_analysis(method)
        assert a.active_roi_labels == b.active_roi_labels
        for name in SPIKE_FOV_METRICS:
            if "ccg_zscore" not in name and "fraction_significant" not in name:
                np.testing.assert_equal(getattr(a, name), getattr(b, name))


def test_summary_keeps_zero_and_excludes_unknown_invalid_estimates() -> None:
    assert summarize_noise([0, 4, -1, np.nan, np.inf]) == (2, 2, 2)
    assert summarize_noise([]) == (None, None, 0)
    assert summarize_noise([-1, np.nan]) == (None, None, 0)
    assert summarize_noise([3]) == (3, 0, 1)


def test_staged_actual_calcium_noise_wins_over_extraction_estimate() -> None:
    fov = _fov()
    for roi, noise in zip(fov.rois, [0.1, 0.4, 0.7], strict=True):
        roi._new_traces[-1].calcium_noise = 50
        roi._new_data_analysis[-1].calcium_noise = noise
    result = compute_fov_analysis(fov, _settings())
    assert result.calcium_noise_median == 0.4
    assert result.calcium_noise_iqr == pytest.approx(0.3)


def test_calcium_reanalysis_records_its_actual_getsn_fallback() -> None:
    trace = _fov().rois[0]._new_traces[-1]
    assert trace.calcium_noise is None
    with patch("cali.analysis._roi_analysis.GetSn", return_value=1.25) as estimator:
        result = analyze_roi_calcium(trace, _settings(), duration_s=9.9)
    assert result.calcium_noise == 1.25
    estimator.assert_called_once()
    assert trace.calcium_noise is None  # The historical extraction is not rewritten.
    disabled = analyze_roi_calcium(
        trace, _settings(enable_calcium=False), duration_s=9.9
    )
    assert disabled.calcium_noise is None


def test_unrelated_stored_analysis_noise_does_not_replace_selected_trace() -> None:
    fov = _fov()
    for roi, noise in zip(fov.rois, [1, 3, 9], strict=True):
        roi._new_traces[-1].calcium_noise = noise
        delattr(roi, "_new_data_analysis")
        DataAnalysis(roi=roi, calcium_noise=1000, calcium_active=True)
    result = compute_fov_analysis(fov, _settings(enable_spikes=False))
    assert result.calcium_noise_median == 3
    assert result.calcium_noise_roi_count == 3


def test_all_inactive_rois_still_produce_qc_without_population_metrics() -> None:
    fov = _fov()
    for roi in fov.rois:
        roi.active = False
        analysis = roi._new_data_analysis[-1]
        analysis.calcium_active = False
        for child in analysis.spike_analyses:
            child.spike_active = False
        roi._new_traces[-1].calcium_noise = 0.5
        roi._new_traces[-1].get_spike_trace("cascade").noise = 3
    result = compute_fov_analysis(fov, _settings())
    assert result.calcium_active_roi_labels == []
    assert result.calcium_noise_roi_count == 3
    assert result.calcium_dff_correlation_matrix is None
    assert result.get_spike_analysis("cascade").model_noise_median == 3
    assert result.get_spike_analysis("cascade").active_roi_labels == []
    assert result.get_spike_analysis("cascade").spike_population_activity is None


def _qc_result(calcium: float | None, model: float | None = None) -> FOVAnalysis:
    return FOVAnalysis(
        calcium_noise_median=calcium,
        spike_analyses=[
            SpikeFOVAnalysis(
                method="cascade",
                units="spikes/frame",
                model_noise_median=model,
                inference_run=SpikeInferenceRun(
                    method="cascade",
                    units="spikes/frame",
                    resolved_model="same model",
                    weights_manifest_sha256="a" * 64,
                    model_sampling_rate_hz=30,
                ),
            )
        ],
    )


def test_batch_warnings_compare_separate_scales_without_mutating_results() -> None:
    qc = NoiseQCCollection()
    values = [1, 1.1, 1.2, 1.3, 1.4, 100]
    results = [_qc_result(value, value * 10) for value in values]
    before = [result.model_dump_json() for result in results]
    for index, result in enumerate(results):
        qc.add(f"A1_{index}", result)
    with patch("cali.analysis._noise_qc.cali_logger.warning") as warning:
        qc.warn_outliers()
    assert warning.call_count == 2
    messages = [call.args[0] for call in warning.call_args_list]
    assert all("A1_5" in message for message in messages)
    assert any("calcium/GetSn" in message for message in messages)
    assert any("cascade/model noise" in message for message in messages)
    assert before == [result.model_dump_json() for result in results]


@pytest.mark.parametrize(
    "reason", ["too_few", "zero_iqr", "different_model", "unknown"]
)
def test_unusable_batch_baselines_do_not_warn(reason: str) -> None:
    qc = NoiseQCCollection()
    count = 3 if reason == "too_few" else 6
    for index in range(count):
        value = 1 if reason == "zero_iqr" else (100 if index == 5 else 1 + index / 10)
        result = _qc_result(None, value)
        if reason == "different_model" and index == 5:
            result.spike_analyses[0].inference_run.resolved_model = "other model"
        if reason == "unknown":
            result.spike_analyses[0].inference_run.model_sampling_rate_hz = None
        qc.add(str(index), result)
    with patch("cali.analysis._noise_qc.cali_logger.warning") as warning:
        qc.warn_outliers()
    warning.assert_not_called()


def _version_thirteen(path: Path) -> None:
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as c:
        for table in ("data_analysis", "fov_analysis", "spike_fov_analysis"):
            c.exec_driver_sql(
                f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, value FLOAT)"
            )
            c.exec_driver_sql(f"INSERT INTO {table} VALUES (1, 0.123)")
        c.exec_driver_sql("PRAGMA user_version=13")
    engine.dispose()


def test_schema_fourteen_preserves_old_metrics_and_leaves_qc_unknown(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.cali"
    _version_thirteen(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        with engine.connect() as c:
            assert c.exec_driver_sql("PRAGMA user_version").scalar() == 14
            assert (
                c.exec_driver_sql("SELECT calcium_noise FROM data_analysis").scalar()
                is None
            )
            for table, prefix in (
                ("fov_analysis", "calcium"),
                ("spike_fov_analysis", "model"),
            ):
                assert c.exec_driver_sql(
                    f"SELECT value,{prefix}_noise_median,{prefix}_noise_iqr,"
                    f"{prefix}_noise_roi_count FROM {table}"
                ).one() == (0.123, None, None, None)
    finally:
        engine.dispose()


def test_noise_qc_migration_ddl_rolls_back_and_retries(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.cali"
    _version_thirteen(path)
    engine = create_engine(f"sqlite:///{path}")

    def interrupt(*args: object) -> None:
        if str(args[2]).startswith("ALTER TABLE fov_analysis"):
            raise RuntimeError("interrupted QC migration")

    try:
        event.listen(engine, "after_cursor_execute", interrupt)
        with pytest.raises(RuntimeError, match="QC migration"):
            ensure_schema_current(engine)
        event.remove(engine, "after_cursor_execute", interrupt)
        with engine.connect() as c:
            assert c.exec_driver_sql("PRAGMA user_version").scalar() == 13
            assert len(c.exec_driver_sql("PRAGMA table_info(data_analysis)").all()) == 2
        ensure_schema_current(engine)
        with engine.connect() as c:
            assert (
                c.exec_driver_sql("PRAGMA user_version").scalar()
                == _engine.SCHEMA_VERSION
            )
    finally:
        engine.dispose()


def test_qc_persistence_json_and_run_scoped_csv(tmp_path: Path) -> None:
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'qc.cali'}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            experiment = Experiment(name="noise QC")
            session.add(experiment)
            session.flush()
            owner = CaliResult(experiment=experiment.id)
            session.add(owner)
            session.flush()
            fov = _fov()
            for roi, calcium, model in zip(
                fov.rois, [1, 3, 9], [2, 4, 12], strict=True
            ):
                trace = roi._new_traces[-1]
                trace.calcium_noise = calcium
                trace.analysis_result = owner
                roi._new_data_analysis[-1].analysis_result = owner
                trace.get_spike_trace("cascade").noise = model
            parent = compute_fov_analysis(fov, _settings())
            parent.fov, parent.analysis_result = fov, owner
            session.add_all([fov, parent])
            session.commit()
            run_id, result_id = owner.id, parent.id
            session.expunge_all()
            stored = session.get(FOVAnalysis, result_id)
            assert stored.calcium_noise_median == 3
            snapshot = FOVAnalysis.model_validate_json(stored.model_dump_json())
            assert snapshot.get_spike_analysis("cascade").model_noise_iqr == 5
        path = tmp_path / "noise.csv"
        assert export_noise_qc_to_csv(engine, path, run_id=run_id)
        rows = pd.read_csv(path)
        assert rows.method.tolist() == ["oasis_denoising", "cascade"]
        assert rows.noise_median.tolist() == [3, 4]
        assert rows.noise_units.tolist() == ["dF/F", "CASCADE model noise"]
        assert rows.known_noise_roi_count.tolist() == [3, 3]
        assert not export_noise_qc_to_csv(
            engine, tmp_path / "absent.csv", run_id=run_id + 1
        )
        assert not export_noise_qc_to_csv(
            engine, tmp_path / "filtered.csv", run_id=run_id, position_indices=[99]
        )
        export_traces_to_csv(
            engine,
            {CASCADE_EXPECTED_SPIKES_TRACES: True},
            run_id,
            tmp_path / "qc.cali",
        )
        automatic = tmp_path / "qc_exports" / f"run_{run_id}" / "noise_qc.csv"
        assert automatic.read_bytes() == path.read_bytes()
        assert not export_noise_qc_to_csv(
            engine, path, run_id=run_id, position_indices=[99]
        )
        assert not path.exists()  # A filtered re-export must not leave stale QC.
    finally:
        engine.dispose()
