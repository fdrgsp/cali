"""Ambiguous legacy extracted-stage flags never become invented source links."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection
from sqlmodel import Session, select

from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    CaliResult,
    DataAnalysis,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    FOVAnalysis,
    MigrationIssue,
    SpikeAnalysis,
    SpikeFOVAnalysis,
    Traces,
    _engine,
    create_cali_engine,
    create_database_and_tables,
    ensure_schema_current,
    select_legacy_result_source,
)
from cali.sqlmodel._source_audit_migration import audit_legacy_stage_sources
from cali.sqlmodel._trace_array_codec import decode_trace_array

from ._legacy_spike_json import write_legacy_spike_json


def _legacy(path: Path, variation: str = "identical") -> tuple[int, int, int]:
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        experiment = Experiment(name="legacy stage audit")
        detection, extraction = DetectionSettings(), ExtractionSettings()
        fov = FOV(name="A1", position_index=0)
        session.add_all([experiment, detection, extraction, fov])
        session.flush()
        roi = ROI(fov=fov, label_value=1, detection_settings_id=detection.id)
        traces = []
        owners = []
        for index in range(3):
            owner = CaliResult(
                experiment=experiment.id,
                detection_settings_id=detection.id,
                extraction_settings_id=extraction.id,
                positions_extracted=[0] if index < 2 else None,
                positions_analyzed=[0] if index else None,
                created_at=datetime(2020, 1, 1) + timedelta(days=index),
            )
            trace = Traces(
                analysis_result=owner,
                roi=roi,
                raw_trace=[1, 2, 1],
                dff=[0, 0.1, 0],
                den_dff=[0, 0.09, 0],
                inferred_spikes=[0, 0.5, 0],
                x_axis=[0, 100, 200],
                x_axis_units="ms",
            )
            if index == 2:
                trace.extraction_frame_window = traces[1].extraction_frame_window
                trace.get_spike_trace("oasis").inference_run = (
                    traces[1].get_spike_trace("oasis").inference_run
                )
            session.add(trace)
            session.flush()
            if index == 1 and variation == "partial":
                session.add(
                    Traces(
                        analysis_result=owner,
                        roi=ROI(fov=fov, label_value=2),
                        raw_trace=[9, 8, 7],
                        dff=[0, 0.8, 0],
                        den_dff=[0, 0.7, 0],
                        inferred_spikes=[0, 0.9, 0],
                        x_axis=[0, 100, 200],
                        x_axis_units="ms",
                    )
                )
                session.flush()
            owner.source_extraction_result_id = owner.id if index < 2 else owners[1].id
            owner.legacy_trace_resolution = (
                "legacy_stage_inferred" if index < 2 else "stored_provenance"
            )
            data = DataAnalysis(
                roi=roi,
                analysis_result=owner,
                inferred_spikes_threshold=0.25,
                inferred_spikes_frequency=3.3,
                inferred_spikes_rising_edge_frequency=3.3,
                den_dff_frequency=3.3,
                calcium_active=True,
            )
            fov_data = FOVAnalysis(
                fov=fov,
                analysis_result=owner,
                calcium_active_roi_labels=[1],
                spike_analyses=[
                    SpikeFOVAnalysis(
                        active_roi_labels=[1],
                        spike_burst_count=2,
                        spike_max_lag_correlation_matrix=[[1.0]],
                    )
                ],
            )
            session.add_all([data, fov_data])
            session.flush()
            traces.append(trace)
            owners.append(owner)
        ids = tuple(owner.id for owner in owners)
        session.commit()
    with engine.begin() as connection:
        for table in ("extraction_frame_window", "spike_inference_run"):
            connection.exec_driver_sql(
                f"UPDATE {table} SET provenance_source = 'legacy_import'"
            )
        if variation in {"raw_trace", "dff", "x_axis"}:
            connection.exec_driver_sql(
                f"UPDATE trace SET {variation} = '[9,8,7]' "
                "WHERE analysis_result_id = ?",
                (ids[1],),
            )
        elif variation == "spikes":
            connection.exec_driver_sql(
                "UPDATE spike_trace SET \"values\" = '[0,1,0]' WHERE trace_id IN "
                "(SELECT id FROM trace WHERE analysis_result_id = ?)",
                (ids[1],),
            )
        elif variation == "window":
            connection.exec_driver_sql(
                "UPDATE extraction_frame_window SET source_start_frame = 5 "
                "WHERE extraction_result_id = ?",
                (ids[1],),
            )
        elif variation == "period":
            connection.exec_driver_sql(
                "UPDATE analysis_result SET created_at = '2019-01-01' WHERE id = ?",
                (ids[1],),
            )
        elif variation == "tie":
            connection.exec_driver_sql(
                "UPDATE analysis_result SET created_at = '2020-01-01 00:00:00.000000'"
            )
        elif variation == "extraction_only":
            connection.exec_driver_sql(
                "UPDATE analysis_result SET positions_analyzed = NULL WHERE id = ?",
                (ids[1],),
            )
        elif variation == "explicit":
            connection.exec_driver_sql(
                "UPDATE analysis_result SET legacy_trace_resolution = 'self' "
                "WHERE id = ?",
                (ids[1],),
            )
        elif variation == "current":
            connection.exec_driver_sql(
                "UPDATE extraction_frame_window SET provenance_source = 'extraction'"
            )
        elif variation == "settings":
            connection.exec_driver_sql(
                "UPDATE analysis_result SET extraction_settings_id = NULL WHERE id = ?",
                (ids[1],),
            )
        elif variation == "position":
            connection.exec_driver_sql(
                "UPDATE analysis_result SET positions_extracted = '[1]' WHERE id = ?",
                (ids[0],),
            )
        if variation == "v4":
            connection.exec_driver_sql(
                "UPDATE analysis_result SET source_extraction_result_id = NULL, "
                "legacy_trace_resolution = NULL"
            )
        write_legacy_spike_json(connection)
        connection.exec_driver_sql(
            f"PRAGMA user_version = {4 if variation == 'v4' else 7}"
        )
    engine.dispose()
    return ids


@pytest.mark.parametrize("variation", ["identical", "tie", "partial", "v4"])
def test_ambiguous_flags_and_dependent_copies_are_audited(
    tmp_path: Path, variation: str
) -> None:
    path = tmp_path / "stage.cali"
    source_id, guessed_id, dependent_id = _legacy(path, variation)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        source = session.get(CaliResult, source_id)
        assert source.source_extraction_result_id == source_id
        for result_id in (guessed_id, dependent_id):
            result = session.get(CaliResult, result_id)
            assert result.source_extraction_result_id is None
            assert result.legacy_trace_resolution == "unresolved_stage_flags"
            issue = session.exec(
                select(MigrationIssue).where(
                    MigrationIssue.analysis_result_id == result_id,
                    MigrationIssue.code == "unresolved_stage_flags",
                )
            ).one()
            assert not issue.resolved
            evidence = issue.details["ambiguous_sources"][0]
            assert evidence["result_id"] == guessed_id
            assert (
                evidence["matching_traces"][0]["alternatives"][0]["result_id"]
                == source_id
            )
            trace = result.traces[0]
            assert trace.raw_trace == [1, 2, 1]
            assert trace.get_spike_values("oasis") == [0, 0.5, 0]
            for provenance in (
                trace.extraction_frame_window,
                trace.get_spike_trace("oasis").inference_run,
            ):
                assert provenance.extraction_result_id is None
                assert provenance.legacy_owner_result_id == guessed_id
                assert provenance.provenance_source == "legacy_stage_unresolved"
            roi_metric = result.data_analysis_results[0].get_spike_analysis("oasis")
            assert roi_metric.spike_trace_id is None
            assert roi_metric.threshold == 0.25
            assert roi_metric.suprathreshold_sample_rate_hz == 3.3
            assert roi_metric.provenance_source == "legacy_unresolved"
            fov_metric = result.fov_analysis_results[0].get_spike_analysis("oasis")
            assert fov_metric.spike_inference_run_id is None
            assert fov_metric.spike_burst_count == 2
            assert fov_metric.active_roi_labels == [1]
            assert fov_metric.provenance_source == "legacy_unresolved"
        assert len(session.exec(select(MigrationIssue)).all()) == 6
    ensure_schema_current(engine)
    with engine.begin() as connection:
        audit_legacy_stage_sources(connection)
        assert (
            connection.exec_driver_sql("SELECT COUNT(*) FROM migration_issue").scalar()
            == 6
        )
    engine.dispose()


def test_audit_changes_only_ownership_and_provenance_fields(tmp_path: Path) -> None:
    path = tmp_path / "exact.cali"
    _legacy(path)
    engine = create_engine(f"sqlite:///{path}")
    changes = {
        "trace": set(),
        "spike_trace": set(),
        "data_analysis": set(),
        "fov_analysis": set(),
        "extraction_frame_window": {
            "extraction_result_id",
            "legacy_owner_result_id",
            "provenance_source",
        },
        "spike_inference_run": {
            "extraction_result_id",
            "legacy_owner_result_id",
            "provenance_source",
        },
        "spike_analysis": {"spike_trace_id", "provenance_source"},
        "spike_fov_analysis": {"spike_inference_run_id", "provenance_source"},
        "analysis_result": {"source_extraction_result_id", "legacy_trace_resolution"},
    }

    def snapshot() -> dict:
        with engine.connect() as connection:
            return {
                table: [
                    {
                        key: decode_trace_array(value)
                        if table == "spike_trace" and key == "values"
                        else value
                        for key, value in row.items()
                        if key not in excluded
                    }
                    for row in connection.exec_driver_sql(
                        f"SELECT * FROM {table} ORDER BY id"
                    ).mappings()
                ]
                for table, excluded in changes.items()
            }

    before = snapshot()
    ensure_schema_current(engine)
    assert snapshot() == before
    engine.dispose()


@pytest.mark.parametrize(
    "variation",
    [
        "raw_trace",
        "dff",
        "x_axis",
        "spikes",
        "window",
        "period",
        "extraction_only",
        "explicit",
        "current",
        "settings",
        "position",
    ],
)
def test_distinct_or_explicit_extractions_keep_their_sources(
    tmp_path: Path, variation: str
) -> None:
    path = tmp_path / "distinct.cali"
    _, guessed_id, dependent_id = _legacy(path, variation)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        assert (
            session.get(CaliResult, guessed_id).source_extraction_result_id
            == guessed_id
        )
        assert (
            session.get(CaliResult, dependent_id).source_extraction_result_id
            == guessed_id
        )
        assert not session.exec(select(MigrationIssue)).all()
    engine.dispose()


def test_stage_audit_rolls_back_all_links_and_retries(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.cali"
    _, guessed_id, dependent_id = _legacy(path)
    engine = create_engine(f"sqlite:///{path}")

    def interrupt(connection: Connection) -> None:
        audit_legacy_stage_sources(connection)
        raise RuntimeError("interrupted stage audit")

    with patch.object(_engine, "_MIGRATIONS", (*_engine._MIGRATIONS[:7], interrupt)):
        with pytest.raises(RuntimeError, match="interrupted"):
            ensure_schema_current(engine)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 7
        assert (
            connection.exec_driver_sql("SELECT COUNT(*) FROM migration_issue").scalar()
            == 0
        )
        assert (
            connection.exec_driver_sql(
                "SELECT source_extraction_result_id FROM analysis_result WHERE id = ?",
                (dependent_id,),
            ).scalar()
            == guessed_id
        )
        assert (
            connection.exec_driver_sql(
                "SELECT COUNT(*) FROM spike_analysis WHERE spike_trace_id IS NULL"
            ).scalar()
            == 0
        )
    ensure_schema_current(engine)
    with Session(engine) as session:
        assert (
            session.get(CaliResult, guessed_id).legacy_trace_resolution
            == "unresolved_stage_flags"
        )
    engine.dispose()


def test_quarantined_metrics_are_read_only_and_clean_source_can_be_selected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reuse.cali"
    source_id, guessed_id, _ = _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
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
            source_id,
        )
        assert fov.rois[0]._analysis_source_trace.analysis_result_id == source_id
        for cls, field in (
            (SpikeAnalysis, "threshold"),
            (SpikeFOVAnalysis, "spike_burst_count"),
        ):
            child = session.exec(
                select(cls).where(cls.analysis_result_id == guessed_id)
            ).one()
            setattr(child, field, 8)
            with pytest.raises(ValueError, match="read-only"):
                session.flush()
            session.rollback()
    engine.dispose()


def test_explicit_source_repair_preserves_results_and_dependent_quarantine(
    tmp_path: Path,
) -> None:
    path = tmp_path / "repair.cali"
    source_id, guessed_id, dependent_id = _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        target = session.get(CaliResult, guessed_id)
        old_window = target.traces[0].extraction_frame_window
        old_run = target.traces[0].get_spike_trace("oasis").inference_run
        repaired = select_legacy_result_source(session, guessed_id, source_id)
        assert repaired is target
        session.commit()
        source = session.get(CaliResult, source_id)
        assert repaired.source_extraction_result_id == source_id
        assert repaired.legacy_trace_resolution == "source_selected"
        assert repaired.positions_extracted is None
        assert repaired.positions_analyzed == [0]
        trace = repaired.traces[0]
        original = source.traces[0]
        assert trace.raw_trace == [1, 2, 1]
        assert trace.extraction_frame_window is original.extraction_frame_window
        assert trace.get_spike_values("oasis") == [0, 0.5, 0]
        assert (
            trace.get_spike_trace("oasis").inference_run
            is original.get_spike_trace("oasis").inference_run
        )
        roi_metric = repaired.data_analysis_results[0].get_spike_analysis("oasis")
        assert roi_metric.spike_trace is trace.get_spike_trace("oasis")
        assert roi_metric.threshold == 0.25
        assert roi_metric.suprathreshold_sample_rate_hz == 3.3
        fov_metric = repaired.fov_analysis_results[0].get_spike_analysis("oasis")
        assert (
            fov_metric.inference_run is original.get_spike_trace("oasis").inference_run
        )
        assert fov_metric.spike_burst_count == 2
        assert fov_metric.active_roi_labels == [1]
        assert (
            roi_metric.provenance_source
            == fov_metric.provenance_source
            == "legacy_source_selected"
        )
        dependent = session.get(CaliResult, dependent_id)
        assert dependent.legacy_trace_resolution == "unresolved_stage_flags"
        assert dependent.traces[0].extraction_frame_window is old_window
        assert dependent.traces[0].get_spike_trace("oasis").inference_run is old_run
        assert old_window.extraction_result_id is None
        assert old_run.extraction_result_id is None
        issues = session.exec(
            select(MigrationIssue).where(
                MigrationIssue.analysis_result_id == guessed_id
            )
        ).all()
        assert len(issues) == 4 and all(issue.resolved for issue in issues)
        selection = next(
            issue for issue in issues if issue.code == "legacy_source_selected"
        )
        assert selection.details["previous_positions_extracted"] == [0]
        assert selection.details["selected_source_result_id"] == source_id
        assert (
            selection.details["trace_pairs"][0]["previous_window_id"] == old_window.id
        )
        with pytest.raises(ValueError, match="resolved extraction"):
            select_legacy_result_source(session, guessed_id, source_id)
        session.rollback()
    with Session(engine) as session:
        repaired = session.get(CaliResult, guessed_id)
        assert repaired.source_extraction_result_id == source_id
        assert (
            repaired.data_analysis_results[0].get_spike_analysis("oasis").spike_trace_id
            is not None
        )
        runner = CaliRunner()
        fov = session.exec(select(FOV)).one()
        runner._pin_analysis_source_traces(
            session,
            [fov],
            repaired.extraction_settings_id,
            repaired.detection_settings_id,
            source_id,
        )
        assert fov.rois[0]._analysis_source_trace.analysis_result_id == source_id
    engine.dispose()


@pytest.mark.parametrize(
    "variation", ["payload", "window", "ordering", "missing_roi", "threshold", "source"]
)
def test_incompatible_source_repair_changes_nothing(
    tmp_path: Path, variation: str
) -> None:
    path = tmp_path / "invalid_repair.cali"
    source_id, guessed_id, dependent_id = _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        if variation == "payload":
            connection.exec_driver_sql(
                "UPDATE trace SET dff = '[0,9,0]' WHERE analysis_result_id = ?",
                (guessed_id,),
            )
        elif variation == "window":
            connection.exec_driver_sql(
                "UPDATE extraction_frame_window SET source_start_frame = 5 "
                "WHERE legacy_owner_result_id = ?",
                (guessed_id,),
            )
        elif variation == "ordering":
            connection.exec_driver_sql(
                "UPDATE spike_fov_analysis SET active_roi_labels = '[9]' "
                "WHERE analysis_result_id = ?",
                (guessed_id,),
            )
        elif variation == "missing_roi":
            connection.exec_driver_sql(
                "UPDATE data_analysis SET roi_id = NULL WHERE analysis_result_id = ?",
                (guessed_id,),
            )
        elif variation == "threshold":
            connection.exec_driver_sql(
                "UPDATE spike_analysis SET threshold = -1 WHERE analysis_result_id = ?",
                (guessed_id,),
            )
    with Session(engine) as session:
        before = session.get(CaliResult, guessed_id)
        prior_window = before.traces[0].extraction_frame_window_id
        with pytest.raises(ValueError):
            select_legacy_result_source(
                session,
                guessed_id,
                dependent_id if variation == "source" else source_id,
            )
        assert not session.new and not session.dirty and not session.deleted
        assert before.legacy_trace_resolution == "unresolved_stage_flags"
        assert before.traces[0].extraction_frame_window_id == prior_window
        assert not session.exec(
            select(MigrationIssue).where(MigrationIssue.resolved)
        ).all()
    engine.dispose()


def test_source_repair_is_owned_by_callers_transaction(tmp_path: Path) -> None:
    path = tmp_path / "rollback_repair.cali"
    source_id, guessed_id, _ = _legacy(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        experiment = session.exec(select(Experiment)).one()
        experiment.name = "pending unrelated work"
        select_legacy_result_source(session, guessed_id, source_id)
        session.flush()
        session.rollback()
    with Session(engine) as session:
        target = session.get(CaliResult, guessed_id)
        assert target.legacy_trace_resolution == "unresolved_stage_flags"
        assert target.traces[0].extraction_frame_window.extraction_result_id is None
        assert (
            target.data_analysis_results[0].get_spike_analysis("oasis").spike_trace_id
            is None
        )
        assert session.exec(select(Experiment)).one().name == "legacy stage audit"
        assert not session.exec(
            select(MigrationIssue).where(MigrationIssue.resolved)
        ).all()
    engine.dispose()
