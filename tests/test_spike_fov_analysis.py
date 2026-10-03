"""Method-bound FOV storage, exact backfill, and source isolation."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection
from sqlmodel import Session, select

from cali.sqlmodel import (
    FOV,
    ROI,
    CaliResult,
    Experiment,
    FOVAnalysis,
    MigrationIssue,
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
from cali.sqlmodel._spike_fov_analysis_migration import (
    _METRIC_COLUMNS,
    migrate_spike_fov_analyses,
)

from ._legacy_spike_json import write_legacy_spike_json


def _metrics() -> dict:
    values = {}
    for name, kind in _METRIC_COLUMNS:
        if kind == "JSON":
            values[name] = (
                [[0, 1], [-1, 0]]
                if "values_matrix" in name
                else [[1, 0.25], [0.25, 1]]
                if "matrix" in name
                else [0, 2]
                if name.endswith("starts")
                else [1, 3]
                if name.endswith("ends")
                else [0.1, 0.8, 0.2]
            )
        else:
            values[name] = 2 if kind == "INTEGER" else 0.125
    return values


def _records(session: Session) -> tuple[FOVAnalysis, list[Traces]]:
    experiment = Experiment(name="FOV spike metrics")
    fov = FOV(name="A1", position_index=0)
    session.add_all([experiment, fov])
    session.flush()
    owner = CaliResult(
        experiment=experiment.id,
        positions_extracted=[0],
        legacy_trace_resolution="self",
    )
    traces = [
        Traces(
            roi=ROI(fov=fov, label_value=label),
            analysis_result=owner,
            dff=[0, 0.2, 0],
            den_dff=[0, 0.1, 0],
            inferred_spikes=[0, label * 0.5, 0],
        )
        for label in (1, 2)
    ]
    parent = FOVAnalysis(
        fov=fov,
        analysis_result=owner,
        calcium_active_roi_labels=[2, 1],
        calcium_dff_correlation_matrix=[[1, 0.4], [0.4, 1]],
        spike_analyses=[SpikeFOVAnalysis(active_roi_labels=[2, 1], **_metrics())],
    )
    session.add_all([*traces, parent])
    session.flush()
    return parent, traces


def _legacy(path: Path, unresolved: bool = False) -> None:
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        parent, _ = _records(session)
        if unresolved:
            parent.analysis_result.legacy_trace_resolution = "multiple_sources"
        session.commit()
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE spike_fov_analysis")
        connection.exec_driver_sql(
            "ALTER TABLE fov_analysis DROP COLUMN calcium_active_roi_labels"
        )
        connection.exec_driver_sql("UPDATE fov_analysis SET active_roi_labels='[2,1]'")
        for name, kind in _METRIC_COLUMNS:
            value = _metrics()[name]
            connection.exec_driver_sql(
                f"UPDATE fov_analysis SET {name}=?",
                (json.dumps(value) if kind == "JSON" else value,),
            )
        write_legacy_spike_json(connection)
        connection.exec_driver_sql("PRAGMA user_version=6")
    engine.dispose()


def test_every_fov_metric_and_ordering_backfills_exactly(tmp_path: Path) -> None:
    path = tmp_path / "legacy.cali"
    _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        parent = session.exec(select(FOVAnalysis)).one()
        child = parent.get_spike_analysis("oasis")
        assert child.active_roi_labels == parent.calcium_active_roi_labels == [2, 1]
        assert child.provenance_source == "legacy_import"
        assert child.spike_inference_run_id is not None
        assert child.analysis_result_id == parent.analysis_result_id
        assert child.fov_id == parent.fov_id
        for name, value in _metrics().items():
            assert parent.get_spike_metric("oasis", name) == value
        assert parent.calcium_dff_correlation_matrix == [[1, 0.4], [0.4, 1]]
        assert not session.exec(select(MigrationIssue)).all()
    ensure_schema_current(engine)
    engine.dispose()


def test_unresolved_fov_metrics_are_readable_audited_and_read_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unresolved.cali"
    _legacy(path, unresolved=True)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        parent = session.exec(select(FOVAnalysis)).one()
        child = parent.get_spike_analysis("oasis")
        assert child.spike_inference_run_id is None
        assert parent.spike_burst_count == 2
        assert child.provenance_source == "legacy_unresolved"
        issue = session.exec(select(MigrationIssue)).one()
        assert issue.code == "unresolved_spike_fov_analysis"
        assert issue.details["rows"][0]["fov_analysis_id"] == parent.id
        child.spike_burst_count = 8
        with pytest.raises(ValueError, match="read-only"):
            session.flush()
    engine.dispose()


def test_fov_migration_rolls_back_ddl_and_backfill_then_retries(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.cali"
    _legacy(path)
    engine = create_engine(f"sqlite:///{path}")

    def interrupt(connection: Connection) -> None:
        migrate_spike_fov_analyses(connection)
        raise RuntimeError("interrupted FOV migration")

    with patch.object(_engine, "_MIGRATIONS", (*_engine._MIGRATIONS[:6], interrupt)):
        with pytest.raises(RuntimeError, match="interrupted"):
            ensure_schema_current(engine)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 6
        assert not connection.exec_driver_sql(
            "PRAGMA table_info(spike_fov_analysis)"
        ).all()
        assert "calcium_active_roi_labels" not in {
            row[1]
            for row in connection.exec_driver_sql("PRAGMA table_info(fov_analysis)")
        }
        assert (
            connection.exec_driver_sql(
                "SELECT spike_burst_count FROM fov_analysis"
            ).scalar()
            == 2
        )
    ensure_schema_current(engine)
    with Session(engine) as session:
        assert (
            session.exec(select(SpikeFOVAnalysis)).one().spike_inference_run_id
            is not None
        )
    engine.dispose()


def test_dual_fov_results_keep_distinct_orderings_after_detach(tmp_path: Path) -> None:
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'dual.cali'}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        parent, traces = _records(session)
        run = SpikeInferenceRun(
            extraction_result=parent.analysis_result,
            method="cascade",
            units="spikes/frame",
            backend_package="synthetic-test",
        )
        for trace in traces:
            trace.spike_traces.append(SpikeTrace(inference_run=run, values=[0, 0.2, 0]))
        parent.spike_analyses.append(
            SpikeFOVAnalysis(
                method="cascade",
                units="spikes/frame",
                active_roi_labels=[1],
                spike_burst_count=9,
            )
        )
        session.commit()
        parent_id = parent.id
        with engine.connect() as connection:
            old_columns = ", ".join(SPIKE_FOV_METRICS)
            assert all(
                value is None
                for value in connection.exec_driver_sql(
                    f"SELECT {old_columns}, active_roi_labels FROM fov_analysis"
                ).one()
            )
    with Session(engine) as session:
        detached = session.get(FOVAnalysis, parent_id)
    assert detached.get_spike_roi_labels("oasis") == [2, 1]
    assert detached.get_spike_roi_labels("cascade") == [1]
    assert detached.calcium_active_roi_labels == [2, 1]
    assert detached.get_spike_metric("cascade", "spike_burst_count") == 9
    assert detached.get_spike_metric("oasis", "spike_burst_count") == 2
    assert detached.get_spike_analysis("cascade").inference_run.method == "cascade"
    with pytest.raises(ValueError, match="Multiple spike FOV results"):
        _ = detached.spike_burst_count
    engine.dispose()


def test_fov_metrics_cannot_bind_to_another_extraction_run() -> None:
    engine = create_cali_engine("sqlite://")
    create_database_and_tables(engine)
    with Session(engine) as session:
        parent, _ = _records(session)
        unrelated = SpikeInferenceRun()
        session.add(unrelated)
        session.flush()
        child = parent.get_spike_analysis("oasis")
        child.inference_run = unrelated
        with pytest.raises(ValueError, match="stored traces' inference run"):
            session.flush()
    engine.dispose()


def test_fov_spike_units_must_match_method() -> None:
    engine = create_cali_engine("sqlite://")
    create_database_and_tables(engine)
    with Session(engine) as session:
        parent, _ = _records(session)
        parent.get_spike_analysis("oasis").units = "spikes/frame"
        with pytest.raises(ValueError, match="units must match"):
            session.flush()
    engine.dispose()


def test_spike_csv_uses_method_ordering_and_sorts_matrix_with_labels(
    tmp_path: Path,
) -> None:
    import pandas as pd

    from cali.util._database_to_csv import (
        export_inferred_spikes_cross_correlation_to_csv,
    )

    engine = create_cali_engine("sqlite://")
    create_database_and_tables(engine)
    with Session(engine) as session:
        parent, _ = _records(session)
        parent.calcium_active_roi_labels = [1, 2]
        parent.get_spike_analysis("oasis").spike_max_lag_correlation_matrix = [
            [1, 0.3],
            [0.6, 1],
        ]
        session.commit()
        owner_id = parent.analysis_result_id
    export_inferred_spikes_cross_correlation_to_csv(
        engine, tmp_path / "cross.csv", run_id=owner_id
    )
    data = pd.read_csv(tmp_path / "A1_cross.csv", index_col=0)
    assert list(data.index) == list(data.columns) == ["ROI_1", "ROI_2"]
    assert data.loc["ROI_1", "ROI_2"] == 0.6
    assert data.loc["ROI_2", "ROI_1"] == 0.3
    engine.dispose()


def test_missing_ordering_is_quarantined_without_inventing_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_order.cali"
    _legacy(path)
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.exec_driver_sql("UPDATE fov_analysis SET active_roi_labels=NULL")
    ensure_schema_current(engine)
    with Session(engine) as session:
        child = session.exec(select(SpikeFOVAnalysis)).one()
        assert child.active_roi_labels is None
        assert child.spike_inference_run_id is None
        assert child.spike_burst_count == 2
        assert child.provenance_source == "legacy_unresolved"
    engine.dispose()


def test_conflicting_partial_fov_backfill_does_not_advance_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.cali"
    _legacy(path)
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        migrate_spike_fov_analyses(connection)
        connection.exec_driver_sql(
            "UPDATE spike_fov_analysis SET spike_inference_run_id=NULL"
        )
    with pytest.raises(ValueError, match="source verification"):
        ensure_schema_current(engine)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 6
        assert (
            connection.exec_driver_sql(
                "SELECT spike_inference_run_id FROM spike_fov_analysis"
            ).scalar()
            is None
        )
    engine.dispose()


def test_missing_fov_parent_is_audited_without_invalid_new_foreign_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orphan.cali"
    _legacy(path)
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.exec_driver_sql("UPDATE fov_analysis SET fov_id=999")
    ensure_schema_current(engine)
    with Session(engine) as session:
        child = session.exec(select(SpikeFOVAnalysis)).one()
        assert child.fov_id is None and child.spike_inference_run_id is None
        assert child.spike_burst_count == 2
        issue = session.exec(select(MigrationIssue)).one()
        assert issue.details["rows"][0]["legacy_fov_id"] == 999
    engine.dispose()


def test_fk_only_staged_copy_binds_fov_results_and_survives_source_deletion() -> None:
    engine = create_cali_engine("sqlite://")
    create_database_and_tables(engine)
    with Session(engine) as session:
        original, traces = _records(session)
        session.commit()
        source_id = original.analysis_result_id
        source_run_id = original.get_spike_analysis("oasis").spike_inference_run_id
        target = CaliResult(
            experiment=original.analysis_result.experiment,
            source_extraction_result_id=source_id,
        )
        session.add(target)
        session.flush()
        copied = Traces(
            roi_id=traces[0].roi_id,
            analysis_result_id=target.id,
            dff=traces[0].dff,
            den_dff=traces[0].den_dff,
            extraction_frame_window=traces[0].extraction_frame_window,
            spike_traces=[
                SpikeTrace(
                    inference_run=traces[0].get_spike_trace("oasis").inference_run,
                    values=[0, 0.5, 0],
                )
            ],
        )
        parent = FOVAnalysis(
            fov_id=original.fov_id,
            analysis_result_id=target.id,
            calcium_active_roi_labels=[1],
            spike_analyses=[
                SpikeFOVAnalysis(active_roi_labels=[1], spike_burst_count=1)
            ],
        )
        session.add_all([copied, parent])
        session.commit()
        child_id = parent.get_spike_analysis("oasis").id
        assert (
            parent.get_spike_analysis("oasis").spike_inference_run_id == source_run_id
        )
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "DELETE FROM analysis_result WHERE id=?", (source_id,)
        )
    with Session(engine) as session:
        child = session.get(SpikeFOVAnalysis, child_id)
        assert child.spike_burst_count == 1
        assert child.spike_inference_run_id == source_run_id
        assert child.inference_run.extraction_result_id is None
    engine.dispose()


def test_orm_run_deletion_removes_spike_metrics_without_sqlite_fk_listener(
    tmp_path: Path,
) -> None:
    path = tmp_path / "external_engine.cali"
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        parent, _ = _records(session)
        session.commit()
        owner_id = parent.analysis_result_id
    engine.dispose()
    external = create_engine(f"sqlite:///{path}")
    with Session(external) as session:
        assert session.connection().exec_driver_sql("PRAGMA foreign_keys").scalar() == 0
        session.delete(session.get(CaliResult, owner_id))
        session.commit()
        assert not session.exec(select(SpikeFOVAnalysis)).all()
    external.dispose()
