"""Method-bound ROI results and transactional, audited legacy imports."""

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection
from sqlmodel import Session, select

from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    Experiment,
    MigrationIssue,
    SpikeAnalysis,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    _engine,
    create_cali_engine,
    create_database_and_tables,
    ensure_schema_current,
)
from cali.sqlmodel._spike_analysis_migration import migrate_spike_analyses

from ._legacy_spike_json import write_legacy_spike_json


def _records(session: Session) -> tuple[CaliResult, ROI, Traces, DataAnalysis]:
    experiment = Experiment(name="spike metrics")
    settings = AnalysisSettings()
    fov = FOV(name="A1", position_index=0)
    session.add_all([experiment, settings, fov])
    session.flush()
    result = CaliResult(
        experiment=experiment.id,
        analysis_settings_id=settings.id,
        positions_extracted=[0],
        legacy_trace_resolution="self",
    )
    roi = ROI(fov=fov, label_value=1)
    trace = Traces(
        roi=roi,
        analysis_result=result,
        raw_trace=[1, 2, 1],
        den_dff=[0, 0.1, 0],
        inferred_spikes=[0, 0.75, 0],
    )
    analysis = DataAnalysis(
        roi=roi,
        analysis_result=result,
        den_dff_frequency=0.0,
        peaks_den_dff=[],
        calcium_active=False,
        spike_analyses=[
            SpikeAnalysis(
                spike_trace=trace.get_spike_trace("oasis"),
                threshold=0.25,
                threshold_mode="multiplier",
                suprathreshold_sample_rate_hz=0.5,
                suprathreshold_rising_edge_rate_hz=0.25,
                spike_active=True,
            )
        ],
    )
    session.add_all([trace, analysis])
    session.flush()
    return result, roi, trace, analysis


def _legacy(path: Path, unresolved: bool = False) -> None:
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        result, _, _, _ = _records(session)
        if unresolved:
            result.legacy_trace_resolution = "unresolved_payload_mismatch"
        session.commit()
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE spike_analysis")
        connection.exec_driver_sql(
            "ALTER TABLE data_analysis DROP COLUMN calcium_active"
        )
        connection.exec_driver_sql(
            "UPDATE data_analysis SET inferred_spikes_threshold = 0.25, "
            "inferred_spikes_frequency = 0.5, "
            "inferred_spikes_rising_edge_frequency = 0.25"
        )
        write_legacy_spike_json(connection)
        connection.exec_driver_sql("PRAGMA user_version = 5")
    engine.dispose()


def test_legacy_metrics_migrate_exactly_with_independent_activity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.cali"
    _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        parent = session.exec(select(DataAnalysis)).one()
        child = parent.get_spike_analysis("oasis")
        assert child is not None
        assert (
            child.threshold,
            child.suprathreshold_sample_rate_hz,
            child.suprathreshold_rising_edge_rate_hz,
        ) == (0.25, 0.5, 0.25)
        assert child.threshold_mode == "multiplier"
        assert child.spike_trace_id is not None
        assert child.analysis_result_id == parent.analysis_result_id
        assert child.provenance_source == "legacy_import"
        assert child.spike_active is True and parent.calcium_active is False
        assert parent.peaks_den_dff == [] and parent.den_dff_frequency == 0
        assert not session.exec(select(MigrationIssue)).all()
    ensure_schema_current(engine)
    engine.dispose()


def test_unresolved_metrics_are_audited_readable_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "quarantined.cali"
    _legacy(path, unresolved=True)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        parent = session.exec(select(DataAnalysis)).one()
        child = parent.get_spike_analysis("oasis")
        assert child.spike_trace_id is None
        assert child.provenance_source == "legacy_unresolved"
        assert parent.inferred_spikes_frequency == 0.5
        issue = session.exec(select(MigrationIssue)).one()
        assert issue.code == "unresolved_spike_analysis" and not issue.resolved
        assert issue.details["rows"][0]["data_analysis_id"] == parent.id
        child.threshold = 3
        with pytest.raises(ValueError, match="read-only"):
            session.flush()
        session.rollback()
    engine.dispose()


def test_roi_migration_rolls_back_ddl_and_backfill_then_retries(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.cali"
    _legacy(path)
    engine = create_engine(f"sqlite:///{path}")

    def interrupt(connection: Connection) -> None:
        migrate_spike_analyses(connection)
        raise RuntimeError("interrupted ROI migration")

    with patch.object(_engine, "_MIGRATIONS", (*_engine._MIGRATIONS[:5], interrupt)):
        with pytest.raises(RuntimeError, match="interrupted"):
            ensure_schema_current(engine)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 5
        assert not connection.exec_driver_sql("PRAGMA table_info(spike_analysis)").all()
        assert "calcium_active" not in {
            row[1]
            for row in connection.exec_driver_sql("PRAGMA table_info(data_analysis)")
        }
        assert (
            connection.exec_driver_sql(
                "SELECT inferred_spikes_frequency FROM data_analysis"
            ).scalar()
            == 0.5
        )
    ensure_schema_current(engine)
    with Session(engine) as session:
        assert session.exec(select(SpikeAnalysis)).one().spike_trace_id is not None
    engine.dispose()


def test_dual_metrics_reopen_detach_without_singular_alias(tmp_path: Path) -> None:
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'dual.cali'}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        result, _, trace, parent = _records(session)
        cascade_trace = SpikeTrace(
            trace=trace,
            inference_run=SpikeInferenceRun(
                extraction_result=result,
                method="cascade",
                units="spikes/frame",
                backend_package="synthetic-test",
            ),
            values=[0, 0.2, 0],
        )
        parent.spike_analyses.append(
            SpikeAnalysis(
                spike_trace=cascade_trace,
                method="cascade",
                units="spikes/frame",
                threshold=0.1,
                threshold_mode="cascade_ap",
                expected_spike_rate_hz=2,
                expected_spike_count=0.2,
                suprathreshold_excursion_rate_hz=1,
                spike_active=False,
            )
        )
        session.commit()
        parent_id = parent.id
        assert parent.get_spike_analysis("cascade").analysis_result_id == result.id
        with engine.connect() as connection:
            assert tuple(
                connection.exec_driver_sql(
                    "SELECT inferred_spikes_threshold, inferred_spikes_frequency, "
                    "inferred_spikes_rising_edge_frequency FROM data_analysis"
                ).one()
            ) == (None, None, None)
    with Session(engine) as session:
        detached = session.get(DataAnalysis, parent_id)
    assert detached.get_spike_analysis("oasis").spike_active is True
    assert detached.get_spike_analysis("cascade").spike_active is False
    assert detached.get_spike_metric("cascade", "expected_spike_rate_hz") == 2
    assert detached.get_spike_metric("oasis", "inferred_spikes_frequency") == 0.5
    assert detached.get_spike_analysis("cascade").spike_trace.values == [0, 0.2, 0]
    with pytest.raises(ValueError, match="Multiple spike results"):
        _ = detached.inferred_spikes_threshold
    engine.dispose()


@pytest.mark.parametrize(
    "change, message",
    [
        ({"units": "spikes/frame"}, "units must match"),
        ({"threshold_mode": "cascade_ap"}, "incompatible"),
        ({"expected_spike_count": 2}, "OASIS cannot"),
        ({"threshold": float("nan")}, "finite"),
        ({"suprathreshold_sample_rate_hz": -1}, "non-negative"),
    ],
)
def test_invalid_roi_metric_writes_fail(change: dict, message: str) -> None:
    engine = create_cali_engine("sqlite://")
    create_database_and_tables(engine)
    with Session(engine) as session:
        _, _, _, parent = _records(session)
        child = parent.get_spike_analysis("oasis")
        for name, value in change.items():
            setattr(child, name, value)
        with pytest.raises(ValueError, match=message):
            session.flush()
    engine.dispose()


def test_cross_roi_trace_is_rejected() -> None:
    engine = create_cali_engine("sqlite://")
    create_database_and_tables(engine)
    with Session(engine) as session:
        _, roi, _, parent = _records(session)
        other_roi = ROI(fov=roi.fov, label_value=2)
        session.add(other_roi)
        session.flush()
        parent.roi = other_roi
        parent.get_spike_analysis("oasis").threshold = 0.3
        with pytest.raises(ValueError, match="parent's ROI"):
            session.flush()
    engine.dispose()


def test_pending_result_ids_and_synthetic_api_binding() -> None:
    engine = create_cali_engine("sqlite://")
    create_database_and_tables(engine)
    with Session(engine) as session:
        _, roi, _, existing = _records(session)
        owner = CaliResult(experiment=existing.analysis_result.experiment)
        trace = Traces(roi=roi, analysis_result=owner, inferred_spikes=[0, 1])
        parent = DataAnalysis(
            roi=roi, analysis_result=owner, inferred_spikes_threshold=0.4
        )
        session.add_all([trace, parent])
        session.commit()
        child = parent.get_spike_analysis("oasis")
        assert child.analysis_result_id == owner.id
        assert child.spike_trace_id == trace.get_spike_trace("oasis").id
        assert child.provenance_source == "legacy_api_bound"
        stub = DataAnalysis(inferred_spikes_frequency=0.6)
        session.add(stub)
        session.commit()
        assert stub.get_spike_analysis("oasis").spike_trace_id is None
        assert (
            stub.get_spike_analysis("oasis").provenance_source == "synthetic_legacy_api"
        )
    engine.dispose()


def test_reanalysis_metrics_bind_to_stored_copy_and_survive_source_deletion() -> None:
    from cali.runner import CaliRunner

    engine = create_cali_engine("sqlite://")
    create_database_and_tables(engine)
    with Session(engine) as session:
        source, roi, trace, _ = _records(session)
        target = CaliResult(
            experiment=source.experiment,
            analysis_settings_id=source.analysis_settings_id,
        )
        session.add(target)
        session.flush()
        roi._analysis_source_trace = trace
        roi._new_data_analysis = [
            DataAnalysis(
                spike_analyses=[
                    SpikeAnalysis(
                        spike_trace=trace.get_spike_trace("oasis"),
                        threshold=0.4,
                        threshold_mode="global",
                        suprathreshold_sample_rate_hz=0.3,
                        spike_active=True,
                    )
                ]
            )
        ]
        with session.no_autoflush:
            CaliRunner()._process_fov_results(
                roi.fov, session, target.id, include_traces=False
            )
        session.commit()
        parent = session.exec(
            select(DataAnalysis).where(DataAnalysis.analysis_result_id == target.id)
        ).one()
        child = parent.get_spike_analysis("oasis")
        child_id = child.id
        assert child.spike_trace.trace.analysis_result_id == target.id
        assert child.spike_trace_id != trace.get_spike_trace("oasis").id
        assert child.spike_trace.inference_run.extraction_result_id == source.id
        source_id = source.id
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "DELETE FROM analysis_result WHERE id = ?", (source_id,)
        )
    with Session(engine) as session:
        child = session.get(SpikeAnalysis, child_id)
        assert child.threshold == 0.4
        assert child.spike_trace.values == [0, 0.75, 0]
        assert child.spike_trace.inference_run.extraction_result_id is None
    engine.dispose()


def test_quarantined_metrics_block_implicit_source_selection(tmp_path: Path) -> None:
    from cali.runner import CaliRunner

    path = tmp_path / "quarantined.cali"
    _legacy(path, unresolved=True)
    engine = create_cali_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        # Metric ownership remains unresolved even if the broader source pointer
        # has subsequently been repaired. A repair must also resolve the metrics.
        connection.exec_driver_sql(
            "UPDATE analysis_result SET legacy_trace_resolution='self'"
        )
    with Session(engine) as session:
        fov = session.exec(select(FOV)).one()
        with pytest.raises(ValueError, match="source is unresolved"):
            CaliRunner()._pin_analysis_source_traces(session, [fov], None, None)
    engine.dispose()


def test_conflicting_partial_roi_backfill_does_not_advance_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.cali"
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        _records(session)
        session.commit()
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE data_analysis SET inferred_spikes_threshold=0.25, "
            "inferred_spikes_frequency=0.5, inferred_spikes_rising_edge_frequency=0.25"
        )
        connection.exec_driver_sql("UPDATE spike_analysis SET spike_trace_id=NULL")
        write_legacy_spike_json(connection)
        connection.exec_driver_sql("PRAGMA user_version=5")
    with pytest.raises(ValueError, match="source verification"):
        ensure_schema_current(engine)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 5
        assert (
            connection.exec_driver_sql(
                "SELECT spike_trace_id FROM spike_analysis"
            ).scalar()
            is None
        )
    engine.dispose()
