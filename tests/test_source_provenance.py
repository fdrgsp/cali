"""Source pinning and audited, transactional source-link backfill."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, select

from cali.analysis import AnalysisRunner
from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    MigrationIssue,
    Plate,
    Traces,
    Well,
    _engine,
    create_cali_engine,
    create_database_and_tables,
    ensure_schema_current,
)
from cali.sqlmodel._source_migration import migrate_source_links
from cali.sqlmodel._source_provenance import record_result_sources

from ._legacy_spike_json import write_legacy_spike_json


def _graph(
    session: Session,
) -> tuple[Experiment, DetectionSettings, ExtractionSettings, FOV, ROI]:
    experiment = Experiment(name="sources")
    detection = DetectionSettings()
    extraction = ExtractionSettings()
    plate = Plate(name="plate", experiment=experiment)
    well = Well(name="A1", row=0, column=0, plate=plate)
    fov = FOV(name="A1", position_index=0, well=well)
    session.add_all([experiment, detection, extraction, fov])
    session.flush()
    roi = ROI(fov=fov, label_value=1, detection_settings_id=detection.id)
    session.add(roi)
    session.flush()
    return experiment, detection, extraction, fov, roi


def _result(
    session: Session, graph: tuple, day: int = 0, analysis: bool = False
) -> CaliResult:
    experiment, detection, extraction, _, _ = graph
    result = CaliResult(
        experiment=experiment.id,
        detection_settings_id=detection.id,
        extraction_settings_id=extraction.id,
        created_at=datetime(2020, 1, 1) + timedelta(days=day),
        positions_extracted=None if analysis else [0],
        positions_analyzed=[0] if analysis else None,
    )
    session.add(result)
    session.flush()
    return result


def _trace(
    session: Session, roi: ROI, result: CaliResult, amplitude: float = 1
) -> Traces:
    trace = Traces(
        roi=roi,
        analysis_result=result,
        raw_trace=[1, 2, 1],
        dff=[0, 0.1, 0],
        den_dff=[0, 0.09, 0],
        inferred_spikes=[0, amplitude, 0],
        x_axis=[0, 100, 200],
        x_axis_units="ms",
    )
    session.add(trace)
    session.flush()
    return trace


def _legacy(path: Path, kind: str) -> tuple[int, int]:
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        graph = _graph(session)
        source = _result(session, graph)
        _trace(session, graph[-1], source)
        if kind == "latest":
            source = _result(session, graph, day=1)
            _trace(session, graph[-1], source)
        elif kind == "ambiguous":
            _trace(session, graph[-1], _result(session, graph))
        copy = _result(session, graph, day=2, analysis=True)
        copied = _trace(
            session, graph[-1], copy, amplitude=2 if kind == "mismatch" else 1
        )
        session.commit()
        source_id, copy_id = source.id, copy.id
        window_id = copied.extraction_frame_window_id
        run_id = copied.get_spike_trace("oasis").spike_inference_run_id
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE extraction_frame_window SET extraction_result_id = NULL, "
            "legacy_owner_result_id = ?, "
            "provenance_source = 'legacy_analysis_copy_unresolved' "
            "WHERE id = ?",
            (copy_id, window_id),
        )
        connection.exec_driver_sql(
            "UPDATE spike_inference_run SET extraction_result_id = NULL, "
            "legacy_owner_result_id = ?, "
            "provenance_source = 'legacy_analysis_copy_unresolved' "
            "WHERE id = ?",
            (copy_id, run_id),
        )
        if kind == "missing":
            connection.exec_driver_sql(
                "UPDATE analysis_result SET positions_extracted = NULL"
            )
        connection.exec_driver_sql("DROP TABLE migration_issue")
        write_legacy_spike_json(connection)
        connection.exec_driver_sql("PRAGMA user_version = 4")
    engine.dispose()
    return source_id, copy_id


@pytest.mark.parametrize("kind", ["unique", "latest"])
def test_unique_preceding_source_rebinds_verified_copies(
    tmp_path: Path, kind: str
) -> None:
    path = tmp_path / "legacy.cali"
    source_id, copy_id = _legacy(path, kind)
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        with Session(engine) as session:
            copy = session.get(CaliResult, copy_id)
            assert copy.source_extraction_result_id == source_id
            assert copy.legacy_trace_resolution == "latest_preceding_inferred"
            trace = session.exec(
                select(Traces).where(Traces.analysis_result_id == copy_id)
            ).one()
            assert trace.get_spike_values("oasis") == [0, 1, 0]
            assert trace.extraction_frame_window.extraction_result_id == source_id
            assert (
                trace.get_spike_trace("oasis").inference_run.extraction_result_id
                == source_id
            )
            assert not session.exec(select(MigrationIssue)).all()
        ensure_schema_current(engine)
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "kind,code",
    [
        ("ambiguous", "unresolved_ambiguous"),
        ("missing", "unresolved_missing"),
        ("mismatch", "unresolved_payload_mismatch"),
    ],
)
def test_unresolved_source_is_audited_and_arrays_stay_readable(
    tmp_path: Path, kind: str, code: str
) -> None:
    path = tmp_path / "unresolved.cali"
    _, copy_id = _legacy(path, kind)
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        with Session(engine) as session:
            copy = session.get(CaliResult, copy_id)
            assert copy.source_extraction_result_id is None
            assert copy.legacy_trace_resolution == code
            issue = session.exec(
                select(MigrationIssue).where(
                    MigrationIssue.analysis_result_id == copy_id
                )
            ).one()
            assert issue.code == code and not issue.resolved
            trace = session.exec(
                select(Traces).where(Traces.analysis_result_id == copy_id)
            ).one()
            assert trace.get_spike_values("oasis") == [
                0,
                2 if kind == "mismatch" else 1,
                0,
            ]
    finally:
        engine.dispose()


def test_source_migration_interruption_rolls_back_and_retries(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.cali"
    _, copy_id = _legacy(path, "unique")
    engine = create_engine(f"sqlite:///{path}")

    def interrupted(connection: object) -> None:
        migrate_source_links(connection)
        raise RuntimeError("interrupted source migration")

    try:
        with patch.object(
            _engine, "_MIGRATIONS", (*_engine._MIGRATIONS[:4], interrupted)
        ):
            with pytest.raises(RuntimeError, match="interrupted"):
                ensure_schema_current(engine)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 4
            assert (
                connection.exec_driver_sql(
                    "SELECT source_extraction_result_id FROM analysis_result "
                    "WHERE id = ?",
                    (copy_id,),
                ).scalar_one()
                is None
            )
            assert not connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE name = 'migration_issue'"
            ).all()
        ensure_schema_current(engine)
    finally:
        engine.dispose()


def test_analysis_and_persistence_use_same_pinned_trace() -> None:
    engine = create_cali_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            graph = _graph(session)
            _experiment, detection, extraction, fov, roi = graph
            source = _result(session, graph)
            original = _trace(session, roi, source)
            target = _result(session, graph, day=3, analysis=True)
            session.commit()
            runner = CaliRunner()
            runner._pin_analysis_source_traces(
                session, [fov], extraction.id, detection.id, source.id
            )
            _trace(session, roi, _result(session, graph, day=1), amplitude=10)
            settings = AnalysisSettings(
                enable_calcium=False,
                enable_spikes=True,
                spike_threshold_mode="global",
                spike_threshold_value=5,
            )
            AnalysisRunner()._analyze_fov(settings, fov)
            assert roi._new_data_analysis[-1].inferred_spikes_frequency is None
            assert not roi.active
            runner._process_fov_results(fov, session, target.id, include_traces=False)
            session.commit()
            copied = session.exec(
                select(Traces).where(Traces.analysis_result_id == target.id)
            ).one()
            assert copied.get_spike_values("oasis") == original.get_spike_values(
                "oasis"
            )
            assert target.source_extraction_result_id == source.id
            assert not hasattr(roi, "_analysis_source_trace")
    finally:
        engine.dispose()


def test_multiple_sources_are_audited_instead_of_overwritten() -> None:
    engine = create_cali_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            graph = _graph(session)
            first = _result(session, graph)
            second = _result(session, graph, day=1)
            target = _result(session, graph, day=2, analysis=True)
            record_result_sources(session, target, {first.id})
            record_result_sources(session, target, {second.id})
            session.commit()
            assert target.source_extraction_result_id is None
            assert target.legacy_trace_resolution == "multiple_sources"
            issue = session.exec(select(MigrationIssue)).one()
            assert issue.details["source_result_ids"] == sorted([first.id, second.id])
    finally:
        engine.dispose()


def test_public_runner_pins_source_and_force_preserves_original(tmp_path: Path) -> None:
    path = tmp_path / "selected.cali"
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            graph = _graph(session)
            experiment, detection, extraction, _, roi = graph
            source = _result(session, graph)
            original = _trace(session, roi, source)
            _trace(session, roi, _result(session, graph, day=1), amplitude=10)
            session.commit()
            source_id, original_id = source.id, original.id
            detection_id, extraction_id = detection.id, extraction.id
            session.refresh(experiment)
            session.expunge(experiment)
        settings = AnalysisSettings(
            enable_calcium=False,
            enable_spikes=True,
            spike_threshold_mode="global",
            spike_threshold_value=5,
            threads=1,
        )
        runner = CaliRunner()
        for force in (False, True):
            runner.run(
                experiment=experiment,
                dataset_path=None,
                detection_settings=detection_id,
                extraction_settings=extraction_id,
                analysis_settings=settings,
                source_extraction_result_id=source_id,
                database_name=path.name,
                output_path=tmp_path,
                global_position_indices=[0],
                force=force,
            )
            with Session(engine) as session:
                original = session.get(Traces, original_id)
                assert original.get_spike_values("oasis") == [0, 1, 0]
                targets = session.exec(
                    select(CaliResult).where(
                        CaliResult.source_extraction_result_id == source_id,
                        CaliResult.id != source_id,
                    )
                ).all()
                assert len(targets) == 1
                assert targets[0].positions_extracted is None
                assert targets[0].positions_analyzed == [0]
                copied = session.exec(
                    select(Traces).where(
                        Traces.analysis_result_id == targets[0].id,
                    )
                ).one()
                assert copied.get_spike_values("oasis") == [0, 1, 0]
    finally:
        engine.dispose()


def test_unresolved_history_blocks_implicit_reanalysis_but_source_can_be_selected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ambiguous.cali"
    source_id, _ = _legacy(path, "ambiguous")
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        with Session(engine) as session:
            source = session.get(CaliResult, source_id)
            fov = session.exec(select(FOV)).one()
            runner = CaliRunner()
            with pytest.raises(ValueError, match="source is unresolved"):
                runner._pin_analysis_source_traces(
                    session,
                    [fov],
                    source.extraction_settings_id,
                    source.detection_settings_id,
                )
            runner._pin_analysis_source_traces(
                session,
                [fov],
                source.extraction_settings_id,
                source.detection_settings_id,
                source.id,
            )
            assert fov.rois[0]._analysis_source_trace.analysis_result_id == source.id
    finally:
        engine.dispose()


def test_source_deletion_clears_links_on_migration_connection(tmp_path: Path) -> None:
    path = tmp_path / "delete.cali"
    source_id, copy_id = _legacy(path, "unique")
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        with engine.begin() as connection:
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
            connection.exec_driver_sql(
                "DELETE FROM analysis_result WHERE id = ?", (source_id,)
            )
        with Session(engine) as session:
            copy = session.get(CaliResult, copy_id)
            assert copy.source_extraction_result_id is None
            trace = session.exec(
                select(Traces).where(
                    Traces.analysis_result_id == copy_id,
                )
            ).one()
            assert trace.get_spike_values("oasis") == [0, 1, 0]
            assert trace.extraction_frame_window.extraction_result_id is None
            assert (
                trace.get_spike_trace("oasis").inference_run.extraction_result_id
                is None
            )
    finally:
        engine.dispose()


def test_source_column_ddl_rolls_back_with_version(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'v4.cali'}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE analysis_result (id INTEGER PRIMARY KEY, experiment INTEGER, "
            "created_at VARCHAR, positions_extracted JSON, positions_analyzed JSON)"
        )
        connection.exec_driver_sql(
            "INSERT INTO analysis_result VALUES (1, 1, '2020-01-01', '[0]', NULL)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE trace (id INTEGER PRIMARY KEY, analysis_result_id INTEGER, "
            "extraction_frame_window_id INTEGER)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE extraction_frame_window "
            "(id INTEGER PRIMARY KEY, extraction_result_id INTEGER)"
        )
        connection.exec_driver_sql(
            'CREATE TABLE spike_trace (id INTEGER PRIMARY KEY, "values" JSON NOT NULL)'
        )
        connection.exec_driver_sql("PRAGMA user_version = 4")

    def interrupted(connection: object) -> None:
        migrate_source_links(connection)
        raise RuntimeError("interrupt after source column DDL")

    try:
        with patch.object(
            _engine, "_MIGRATIONS", (*_engine._MIGRATIONS[:4], interrupted)
        ):
            with pytest.raises(RuntimeError, match="interrupt after"):
                ensure_schema_current(engine)
        with engine.connect() as connection:
            names = {
                row[1]
                for row in connection.exec_driver_sql(
                    "PRAGMA table_info(analysis_result)"
                )
            }
            assert "source_extraction_result_id" not in names
            assert "legacy_trace_resolution" not in names
            assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 4
        ensure_schema_current(engine)
        with engine.connect() as connection:
            assert (
                connection.exec_driver_sql(
                    "SELECT source_extraction_result_id FROM analysis_result"
                ).scalar_one()
                == 1
            )
    finally:
        engine.dispose()
