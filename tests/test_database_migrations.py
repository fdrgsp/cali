from pathlib import Path
from unittest.mock import patch

import pytest
from pytestqt.qtbot import QtBot
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection
from sqlmodel import Session, select

from cali.sqlmodel import (
    AnalysisSettings,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    Traces,
    _engine,
    create_cali_engine,
    create_database_and_tables,
    ensure_schema_current,
)
from cali.sqlmodel._engine import SCHEMA_VERSION
from cali.util._database_to_csv import _get_default_run_id

from ._legacy_spike_json import write_legacy_spike_json


def _legacy_database(path: Path) -> None:
    engine = create_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        session.add(Experiment(name="legacy"))
        session.add(DetectionSettings())
        session.add(
            CaliResult(
                experiment=1,
                detection_settings_id=1,
                extraction_settings_id=1,
                analysis_settings_id=1,
            )
        )
        session.add(ExtractionSettings(frame_rate=17))
        session.add(AnalysisSettings(enable_calcium=False))
        session.add(
            Traces(
                raw_trace=[1.2, 3.4], inferred_spikes=[0.0, 0.5], analysis_result_id=1
            )
        )
        session.commit()
    with engine.begin() as conn:
        for name in (
            "frame_rate_verified",
            "discard_initial_value",
            "discard_initial_unit",
        ):
            conn.exec_driver_sql(f"ALTER TABLE extraction_settings DROP COLUMN {name}")
        for name in (
            "source_start_frame",
            "source_start_time_ms",
            "original_frame_count",
            "discarded_duration_ms",
            "discard_timing_source",
        ):
            conn.exec_driver_sql(f"ALTER TABLE trace DROP COLUMN {name}")
        for name in ("enable_calcium", "enable_spikes"):
            conn.exec_driver_sql(f"ALTER TABLE analysis_settings DROP COLUMN {name}")
        write_legacy_spike_json(conn)
        conn.exec_driver_sql("PRAGMA user_version = 0")
    engine.dispose()


def test_factory_migrates_before_orm_and_preserves_arrays(tmp_path: Path) -> None:
    path = tmp_path / "legacy.cali"
    _legacy_database(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        with Session(engine) as session:
            settings = session.exec(select(ExtractionSettings)).one()
            assert settings.frame_rate == 17
            assert settings.discard_initial_value == 0
            assert settings.discard_initial_unit == "frames"
            assert settings.frame_rate_verified is False
            analysis = session.exec(select(AnalysisSettings)).one()
            assert analysis.enable_calcium is True
            assert analysis.enable_spikes is True
            trace = session.exec(select(Traces)).one()
            assert trace.raw_trace == [1.2, 3.4]
            assert trace.inferred_spikes == [0.0, 0.5]
            assert trace.source_start_frame == 0
        with engine.connect() as conn:
            assert (
                conn.exec_driver_sql("PRAGMA user_version").scalar_one()
                == SCHEMA_VERSION
            )
        with patch.object(_engine, "_MIGRATIONS", (None, None)):
            ensure_schema_current(engine)
    finally:
        engine.dispose()


def test_external_session_loader_migrates(tmp_path: Path) -> None:
    path = tmp_path / "legacy.cali"
    _legacy_database(path)
    engine = create_engine(f"sqlite:///{path}")
    try:
        with Session(engine) as session:
            settings = ExtractionSettings.load_from_database(
                path, id=1, session=session
            )
            assert settings.discard_initial_value == 0
    finally:
        engine.dispose()


def test_external_engine_export_migrates(tmp_path: Path) -> None:
    path = tmp_path / "legacy.cali"
    _legacy_database(path)
    engine = create_engine(f"sqlite:///{path}")
    try:
        assert _get_default_run_id(engine) == 1
        with Session(engine) as session:
            assert session.exec(select(Traces)).one().source_start_frame == 0
    finally:
        engine.dispose()


def test_interrupted_ddl_rolls_back_and_retries(tmp_path: Path) -> None:
    path = tmp_path / "legacy.cali"
    _legacy_database(path)
    engine = create_engine(f"sqlite:///{path}")
    original = _engine._startup_discard

    def interrupted(conn: Connection) -> None:
        original(conn)
        raise RuntimeError("simulated interruption")

    try:
        with patch.object(
            _engine, "_MIGRATIONS", (_engine._analysis_gates, interrupted)
        ):
            with pytest.raises(RuntimeError, match="interruption"):
                ensure_schema_current(engine)
        with engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 1
            columns = {
                row[1] for row in conn.exec_driver_sql("PRAGMA table_info(trace)")
            }
            assert "source_start_frame" not in columns
        ensure_schema_current(engine)
        with Session(engine) as session:
            assert session.exec(select(Traces)).one().inferred_spikes == [0.0, 0.5]
    finally:
        engine.dispose()


def test_new_database_stays_unversioned_until_tables_exist() -> None:
    engine = create_cali_engine("sqlite:///:memory:")
    try:
        with engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 0
        create_database_and_tables(engine)
        with engine.connect() as conn:
            assert (
                conn.exec_driver_sql("PRAGMA user_version").scalar_one()
                == SCHEMA_VERSION
            )
    finally:
        engine.dispose()


def test_newer_database_rejected_without_changes() -> None:
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        with pytest.raises(ValueError, match="newer"):
            create_database_and_tables(engine)
        with engine.connect() as conn:
            assert conn.exec_driver_sql("SELECT name FROM sqlite_master").all() == []
            assert (
                conn.exec_driver_sql("PRAGMA user_version").scalar_one()
                == SCHEMA_VERSION + 1
            )
    finally:
        engine.dispose()


def test_concurrent_open_migrates_once(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "legacy.cali"
    _legacy_database(path)

    def open_database() -> int:
        engine = create_cali_engine(f"sqlite:///{path}", connect_args={"timeout": 30})
        try:
            with engine.connect() as conn:
                return conn.exec_driver_sql("PRAGMA user_version").scalar_one()
        finally:
            engine.dispose()

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert (
            list(pool.map(lambda _: open_database(), range(8))) == [SCHEMA_VERSION] * 8
        )


def test_existing_session_keeps_uncommitted_work() -> None:
    engine = create_cali_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            settings = ExtractionSettings(frame_rate=17)
            session.add(settings)
            session.flush()
            loaded = ExtractionSettings.load_from_database(
                "", id=settings.id, session=session
            )
            assert loaded.frame_rate == 17
            session.commit()
        with Session(engine) as session:
            assert session.exec(select(ExtractionSettings)).one().frame_rate == 17
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "model", [Experiment, CaliResult, DetectionSettings, AnalysisSettings]
)
def test_direct_loaders_upgrade_legacy_database(tmp_path: Path, model: type) -> None:
    path = tmp_path / "legacy.cali"
    _legacy_database(path)
    assert model.load_from_database(path, id=1).id == 1
    engine = create_engine(f"sqlite:///{path}")
    try:
        with engine.connect() as conn:
            assert (
                conn.exec_driver_sql("PRAGMA user_version").scalar_one()
                == SCHEMA_VERSION
            )
    finally:
        engine.dispose()


def test_runs_panel_opens_legacy_database(tmp_path: Path, qtbot: QtBot) -> None:
    from cali.gui._runs_panel import _RunsPanel

    path = tmp_path / "legacy.cali"
    _legacy_database(path)
    panel = _RunsPanel()
    qtbot.addWidget(panel)
    panel.set_database_path(path)
    assert panel._runs_list.count() == 1


def test_schema_validation_does_not_advance_failed_version() -> None:
    engine = create_engine("sqlite:///:memory:")
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("CREATE TABLE analysis_settings (unexpected INTEGER)")
        with pytest.raises(ValueError, match="primary-key"):
            ensure_schema_current(engine)
        with engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 0
            assert [
                row[1]
                for row in conn.exec_driver_sql("PRAGMA table_info(analysis_settings)")
            ] == ["unexpected"]
    finally:
        engine.dispose()


def test_factory_does_not_create_a_missing_database(tmp_path: Path) -> None:
    path = tmp_path / "new.cali"
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        ensure_schema_current(engine)
        assert not path.exists()
        create_database_and_tables(engine)
        assert path.exists()
        with engine.connect() as conn:
            assert (
                conn.exec_driver_sql("PRAGMA user_version").scalar_one()
                == SCHEMA_VERSION
            )
    finally:
        engine.dispose()
