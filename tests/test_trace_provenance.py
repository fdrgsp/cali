"""Normalized timing/inference storage and exact, retryable legacy backfill."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from cali.sqlmodel import (
    FOV,
    ROI,
    CaliResult,
    Experiment,
    ExtractionFrameWindow,
    ExtractionSettings,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    _engine,
    create_cali_engine,
    create_database_and_tables,
    ensure_schema_current,
)
from cali.sqlmodel._trace_migration import migrate_trace_provenance


def _legacy_file(path: Path, crop: int = 0, analysis_copy: bool = False) -> None:
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        experiment = Experiment(name="legacy")
        settings = ExtractionSettings(discard_initial_value=crop)
        fov = FOV(name="A1", position_index=0)
        session.add_all([experiment, settings, fov])
        session.flush()
        result = CaliResult(
            experiment=experiment.id,
            extraction_settings_id=settings.id,
            positions_extracted=[0],
        )
        session.add(result)
        session.flush()
        for label in (1, 2):
            roi = ROI(fov=fov, label_value=label)
            session.add(
                Traces(
                    roi=roi,
                    analysis_result=result,
                    raw_trace=[1.25, 2.5, 3.75, 4.0],
                    dff=[0, 0.1, 0.2, 0],
                    den_dff=[0, 0.09, 0.18, 0],
                    inferred_spikes=[0, label * 0.125, 0, 0.75],
                    x_axis=[0, 100, 200, 300],
                    x_axis_units="ms",
                    source_start_frame=crop,
                    source_start_time_ms=crop * 100,
                    original_frame_count=4 + crop,
                    discarded_duration_ms=crop * 100,
                    discard_timing_source="runner_time" if crop else None,
                )
            )
        session.commit()
        snapshots = [
            (trace.id, trace.inferred_spikes) for trace in session.exec(select(Traces))
        ]
        if analysis_copy:
            copy = CaliResult(
                experiment=experiment.id,
                extraction_settings_id=settings.id,
                positions_analyzed=[0],
            )
            session.add(copy)
            session.commit()
            owner_id = copy.id
        else:
            owner_id = result.id
    with engine.begin() as connection:
        for trace_id, spikes in snapshots:
            connection.exec_driver_sql(
                "UPDATE trace SET inferred_spikes = ?, analysis_result_id = ?, "
                "source_start_frame = ?, source_start_time_ms = ?, "
                "original_frame_count = ?, discarded_duration_ms = ?, "
                "discard_timing_source = ?, extraction_frame_window_id = NULL "
                "WHERE id = ?",
                (
                    json.dumps(spikes),
                    owner_id,
                    crop,
                    crop * 100,
                    4 + crop,
                    crop * 100,
                    "runner_time" if crop else None,
                    trace_id,
                ),
            )
        columns = connection.exec_driver_sql("PRAGMA table_info(trace)").all()
        columns = [
            column for column in columns if column[1] != "extraction_frame_window_id"
        ]
        declarations = []
        for _, name, datatype, required, default, primary in columns:
            declaration = f'"{name}" {datatype}'
            if primary:
                declaration += " PRIMARY KEY"
            elif required:
                declaration += " NOT NULL"
            if default is not None:
                declaration += f" DEFAULT {default}"
            declarations.append(declaration)
        connection.exec_driver_sql(
            "CREATE TABLE trace_v3 (" + ", ".join(declarations) + ")"
        )
        names = ", ".join(f'"{column[1]}"' for column in columns)
        connection.exec_driver_sql(f"INSERT INTO trace_v3 SELECT {names} FROM trace")
        for table in (
            "spike_trace",
            "spike_inference_run",
            "extraction_frame_window",
            "trace",
        ):
            connection.exec_driver_sql(f"DROP TABLE {table}")
        connection.exec_driver_sql("ALTER TABLE trace_v3 RENAME TO trace")
        connection.exec_driver_sql("PRAGMA user_version = 3")
    engine.dispose()


@pytest.mark.parametrize("crop", [0, 2])
def test_exact_legacy_backfill_and_shared_window(tmp_path: Path, crop: int) -> None:
    path = tmp_path / "legacy.cali"
    _legacy_file(path, crop)
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        with Session(engine) as session:
            traces = list(session.exec(select(Traces).order_by(Traces.id)).all())
            assert len(session.exec(select(ExtractionFrameWindow)).all()) == 1
            assert len(session.exec(select(SpikeInferenceRun)).all()) == 1
            assert len(session.exec(select(SpikeTrace)).all()) == 2
            session.expunge_all()
        window = traces[0].extraction_frame_window
        assert window is traces[1].extraction_frame_window
        assert window.requested_discard_value == crop
        assert window.source_start_frame == crop
        assert window.original_frame_count == 4 + crop
        assert window.retained_frame_count == 4
        assert window.source_start_time_ms == crop * 100
        assert window.source_start_timestamp_ms == (None if crop else 0)
        for label, trace in enumerate(traces, 1):
            child = trace.get_spike_trace("oasis")
            assert child.values == [0, label * 0.125, 0, 0.75]
            assert child.valid_start == 0
            assert child.valid_stop is None
            assert child.resolved_valid_stop == 4
            assert child.inference_run.backend_version is None
            assert child.inference_run.units == "a.u."
        with patch.object(_engine, "_MIGRATIONS", (None,) * 4):
            ensure_schema_current(engine)
    finally:
        engine.dispose()


def test_unresolved_analysis_copy_has_synthetic_provenance(tmp_path: Path) -> None:
    path = tmp_path / "copy.cali"
    _legacy_file(path, analysis_copy=True)
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        with Session(engine) as session:
            trace = session.exec(select(Traces)).first()
            run = trace.get_spike_trace("oasis").inference_run
            assert run.extraction_result_id is None
            assert run.legacy_owner_result_id == trace.analysis_result_id
            assert run.provenance_source == "legacy_analysis_copy_unresolved"
    finally:
        engine.dispose()


def test_trace_migration_rolls_back_ddl_and_arrays(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.cali"
    _legacy_file(path)
    engine = create_engine(f"sqlite:///{path}")

    def interrupted(connection: object) -> None:
        migrate_trace_provenance(connection)
        raise RuntimeError("interrupted trace backfill")

    try:
        with patch.object(
            _engine, "_MIGRATIONS", (*_engine._MIGRATIONS[:3], interrupted)
        ):
            with pytest.raises(RuntimeError, match="interrupted"):
                ensure_schema_current(engine)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 3
            assert not connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE name = 'spike_trace'"
            ).all()
        ensure_schema_current(engine)
        with Session(engine) as session:
            assert len(session.exec(select(SpikeTrace)).all()) == 2
    finally:
        engine.dispose()


def test_dual_outputs_are_explicit_and_legacy_accessor_is_read_only() -> None:
    trace = Traces(
        spike_traces=[
            SpikeTrace(values=[0, 1], inference_run=SpikeInferenceRun()),
            SpikeTrace(
                values=[0.1, 0.2],
                inference_run=SpikeInferenceRun(
                    method="cascade",
                    units="spikes/frame",
                    backend_package="synthetic",
                ),
            ),
        ]
    )
    assert trace.get_spike_values("oasis") == [0, 1]
    assert trace.get_spike_values("cascade") == [0.1, 0.2]
    with pytest.raises(ValueError, match="Multiple"):
        _ = trace.inferred_spikes
    with pytest.raises(AttributeError):
        trace.inferred_spikes = [1, 2]


def test_existing_window_id_is_preserved_and_loaded() -> None:
    engine = create_cali_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            window = ExtractionFrameWindow(
                original_frame_count=3, retained_frame_count=2, source_start_frame=1
            )
            session.add(window)
            session.commit()
            trace = Traces(raw_trace=[1, 2], extraction_frame_window_id=window.id)
            session.add(trace)
            session.commit()
            assert trace.extraction_frame_window is window
            assert trace.source_start_frame == 1
            assert len(session.exec(select(ExtractionFrameWindow)).all()) == 1
    finally:
        engine.dispose()


def test_dual_copy_survives_detachment_and_source_deletion() -> None:
    engine = create_cali_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            experiment = Experiment(name="dual")
            session.add(experiment)
            session.flush()
            source = CaliResult(experiment=experiment.id, positions_extracted=[0])
            copy = CaliResult(experiment=experiment.id, positions_analyzed=[0])
            trace = Traces(
                analysis_result=source,
                raw_trace=[1, 2],
                spike_traces=[
                    SpikeTrace(values=[0, 1], inference_run=SpikeInferenceRun()),
                    SpikeTrace(
                        values=[0.1, 0.2],
                        inference_run=SpikeInferenceRun(
                            method="cascade",
                            units="spikes/frame",
                            backend_package="synthetic",
                        ),
                    ),
                ],
            )
            session.add(trace)
            session.commit()
            copied = Traces(
                analysis_result=copy,
                raw_trace=list(trace.raw_trace),
                extraction_frame_window=trace.extraction_frame_window,
                spike_traces=[
                    SpikeTrace(
                        values=list(child.values), inference_run=child.inference_run
                    )
                    for child in trace.spike_traces
                ],
            )
            session.add(copied)
            session.commit()
            copied_id = copied.id
            assert len(session.exec(select(SpikeInferenceRun)).all()) == 2
            assert len(session.exec(select(ExtractionFrameWindow)).all()) == 1
            session.delete(trace)
            session.delete(source)
            session.commit()
        with Session(engine) as session:
            reloaded = session.get(Traces, copied_id)
            session.expunge_all()
        assert reloaded.get_spike_values("oasis") == [0, 1]
        assert reloaded.get_spike_values("cascade") == [0.1, 0.2]
        assert (
            reloaded.get_spike_trace("oasis").inference_run.extraction_result_id is None
        )
        assert reloaded.extraction_frame_window.retained_frame_count == 2
    finally:
        engine.dispose()


def test_conflicting_legacy_windows_roll_back(tmp_path: Path) -> None:
    path = tmp_path / "conflict.cali"
    _legacy_file(path)
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "UPDATE trace SET source_start_frame = 1, original_frame_count = 5 "
                "WHERE id = (SELECT MAX(id) FROM trace)"
            )
        with pytest.raises(ValueError, match="Conflicting legacy frame windows"):
            ensure_schema_current(engine)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 3
            assert not connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE name = 'spike_trace'"
            ).all()
    finally:
        engine.dispose()


def test_new_writes_deduplicate_provenance_and_leave_old_columns_empty() -> None:
    engine = create_cali_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            experiment = Experiment(name="normalized")
            fov = FOV(name="A1", position_index=0)
            session.add_all([experiment, fov])
            session.flush()
            result = CaliResult(experiment=experiment.id, positions_extracted=[0])
            session.add(result)
            session.flush()
            for label in (1, 2):
                session.add(
                    Traces(
                        roi=ROI(fov=fov, label_value=label),
                        analysis_result=result,
                        raw_trace=[1, 2],
                        inferred_spikes=[0, 0.5],
                    )
                )
            session.commit()
            assert len(session.exec(select(ExtractionFrameWindow)).all()) == 1
            assert len(session.exec(select(SpikeInferenceRun)).all()) == 1
            assert len(result.spike_inference_runs) == 1
        with engine.connect() as connection:
            assert all(
                row[0] is None
                for row in connection.exec_driver_sql(
                    "SELECT inferred_spikes FROM trace"
                )
            )
            with pytest.raises(IntegrityError):
                connection.exec_driver_sql(
                    "INSERT INTO spike_trace SELECT NULL, trace_id, "
                    'spike_inference_run_id, "values", valid_start, valid_stop, '
                    "noise, selected_noise_level, ar_coefficients "
                    "FROM spike_trace LIMIT 1"
                )
    finally:
        engine.dispose()


@pytest.mark.parametrize("start, stop", [(-1, None), (2, 1), (0, 3)])
def test_invalid_valid_interval_rejected(start: int, stop: int | None) -> None:
    engine = create_cali_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            session.add(
                Traces(
                    spike_traces=[
                        SpikeTrace(
                            values=[0, 1],
                            valid_start=start,
                            valid_stop=stop,
                            inference_run=SpikeInferenceRun(),
                        )
                    ]
                )
            )
            with pytest.raises(ValueError, match="valid interval"):
                session.commit()
    finally:
        engine.dispose()
