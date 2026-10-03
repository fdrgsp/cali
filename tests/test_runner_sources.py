"""Runner requests preserve extraction generations across batching and exports."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
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
    create_cali_engine,
    create_database_and_tables,
)


def _trace(label: int, amplitude: float = 1) -> Traces:
    values = [0.0] * 100
    for frame in (10, 30, 50, 70, 90):
        values[frame] = amplitude * label
    return Traces(
        raw_trace=[1 + value for value in values],
        dff=values,
        den_dff=values,
        inferred_spikes=values,
        x_axis=[frame * 100 for frame in range(100)],
        x_axis_units="ms",
    )


def _database(path: Path, threads: int = 1, extracted: bool = True) -> tuple:
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        experiment = Experiment(name="separate generations")
        detection, extraction = DetectionSettings(), ExtractionSettings(threads=threads)
        settings = AnalysisSettings(
            enable_calcium=False,
            enable_spikes=True,
            spike_threshold_mode="global",
            spike_threshold_value=0.1,
            threads=threads,
        )
        plate = Plate(name="plate", experiment=experiment)
        well = Well(name="A1", row=0, column=0, plate=plate)
        fovs = [
            FOV(name=f"A1_{position}", position_index=position, well=well)
            for position in range(3)
        ]
        session.add_all([detection, extraction, settings, *fovs])
        session.flush()
        sources = []
        for positions in ([0, 2], [1]) if extracted else ([0], [1]):
            source = CaliResult(
                experiment=experiment.id,
                detection_settings_id=detection.id,
                extraction_settings_id=extraction.id,
                positions_extracted=positions,
                positions_detected=positions,
                legacy_trace_resolution="self",
            )
            session.add(source)
            session.flush()
            source.source_extraction_result_id = source.id
            sources.append(source)
        for position, fov in enumerate(fovs):
            owner = sources[1 if position == 1 else 0]
            for label in (1, 2):
                roi = ROI(
                    fov=fov, label_value=label, detection_settings_id=detection.id
                )
                session.add(roi)
                if extracted or position != 2:
                    trace = _trace(label, 1 + position)
                    trace.roi = roi
                    trace.analysis_result = owner
                    session.add(trace)
        session.commit()
        source_ids = [source.id for source in sources]
        detection_id, extraction_id, settings_id = (
            detection.id,
            extraction.id,
            settings.id,
        )
        session.refresh(experiment)
        session.expunge(experiment)
    engine.dispose()
    return experiment, detection_id, extraction_id, settings_id, source_ids


def _run(
    path: Path, graph: tuple, runner: CaliRunner | None = None, **kwargs: object
) -> None:
    experiment, detection, extraction, settings, _ = graph
    positions = kwargs.pop("global_position_indices", [0, 1, 2])
    (runner or CaliRunner()).run(
        experiment=experiment,
        dataset_path=None,
        detection_settings=detection,
        extraction_settings=extraction,
        analysis_settings=settings,
        database_name=path.name,
        output_path=path.parent,
        global_position_indices=positions,
        **kwargs,
    )


def _verify_copy(session: Session, result: CaliResult) -> None:
    source = session.get(CaliResult, result.source_extraction_result_id)
    assert source is not None and source is not result
    assert result.positions_extracted is None
    assert result.legacy_trace_resolution == "source_selected"
    for trace in result.traces:
        original = next(
            value for value in source.traces if value.roi_id == trace.roi_id
        )
        assert trace.id != original.id
        assert trace.raw_trace == original.raw_trace
        assert trace.dff == original.dff
        assert trace.den_dff == original.den_dff
        assert trace.x_axis == original.x_axis
        assert trace.extraction_frame_window is original.extraction_frame_window
        assert trace.get_spike_values("oasis") == original.get_spike_values("oasis")
        assert (
            trace.get_spike_trace("oasis").inference_run
            is original.get_spike_trace("oasis").inference_run
        )
    for parent in result.data_analysis_results:
        metric = parent.get_spike_analysis("oasis")
        assert metric.spike_trace.trace.analysis_result_id == result.id
        assert metric.spike_trace.trace.roi_id == parent.roi_id
    for parent in result.fov_analysis_results:
        metric = parent.get_spike_analysis("oasis")
        assert metric.inference_run.extraction_result_id == source.id
        assert metric.active_roi_labels == [1, 2]
        assert np.asarray(metric.spike_max_lag_correlation_matrix).shape == (2, 2)


@pytest.mark.parametrize("threads", [1, 3])
def test_interleaved_positions_create_one_analysis_per_extraction_and_reuse_it(
    tmp_path: Path,
    threads: int,
) -> None:
    path = tmp_path / "interleaved.cali"
    graph = _database(path, threads)
    _run(path, graph)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        results = session.exec(select(CaliResult).order_by(CaliResult.id)).all()
        assert len(results) == 4
        assert [result.analysis_settings_id for result in results[:2]] == [None, None]
        assert [result.positions_analyzed for result in results[2:]] == [[0, 2], [1]]
        assert [result.source_extraction_result_id for result in results[2:]] == graph[
            -1
        ]
        for result in results[2:]:
            _verify_copy(session, result)
        assert not session.exec(select(MigrationIssue)).all()
        counts = (len(results), len(session.exec(select(Traces)).all()))
    _run(path, graph)
    with Session(engine) as session:
        assert (
            len(session.exec(select(CaliResult)).all()),
            len(session.exec(select(Traces)).all()),
        ) == counts
    engine.dispose()


def test_latest_extraction_requires_analysis_even_with_identical_settings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "latest.cali"
    graph = _database(path)
    _run(path, graph)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        owner = CaliResult(
            experiment=graph[0].id,
            detection_settings_id=graph[1],
            extraction_settings_id=graph[2],
            positions_extracted=[1],
            legacy_trace_resolution="self",
        )
        session.add(owner)
        session.flush()
        owner.source_extraction_result_id = owner.id
        fov = session.exec(select(FOV).where(FOV.position_index == 1)).one()
        for roi in fov.rois:
            trace = _trace(roi.label_value, 9)
            trace.roi, trace.analysis_result = roi, owner
            session.add(trace)
        session.commit()
        latest_id = owner.id
    _run(path, graph)
    with Session(engine) as session:
        results = session.exec(select(CaliResult).order_by(CaliResult.id)).all()
        assert len(results) == 6
        result = results[-1]
        assert result.source_extraction_result_id == latest_id
        assert result.positions_analyzed == [1]
        _verify_copy(session, result)
        assert result.traces[0].get_spike_values("oasis")[10] == 9
        assert results[3].source_extraction_result_id == graph[-1][1]
    engine.dispose()


def test_exports_are_separate_for_each_source(tmp_path: Path) -> None:
    path = tmp_path / "exports.cali"
    graph = _database(path, 3)
    with (
        patch("cali.util._database_to_csv.export_traces_to_csv") as traces,
        patch("cali.util._database_to_csv.export_correlations_to_csv") as correlations,
        patch("cali.util._database_to_csv.export_multi_well_to_csv") as aggregate,
    ):
        _run(
            path,
            graph,
            export_traces={"Raw Calcium Traces": True},
            export_correlations={"Multi-Well Aggregated Data": True},
        )
    assert traces.call_count == correlations.call_count == aggregate.call_count == 2
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        results = session.exec(
            select(CaliResult)
            .where(CaliResult.analysis_settings_id == graph[3])
            .order_by(CaliResult.id)
        ).all()
        for index, result in enumerate(results):
            assert traces.call_args_list[index].args[2] == result.id
            assert correlations.call_args_list[index].args[2] == result.id
            assert (
                correlations.call_args_list[index].kwargs["position_indices"]
                == result.positions_analyzed
            )
            assert aggregate.call_args_list[index].args[1] == result.id
    engine.dispose()


def test_mixed_rois_in_one_fov_fail_before_any_result_is_created(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed_roi.cali"
    graph = _database(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        fov = session.exec(select(FOV).where(FOV.position_index == 0)).one()
        owner = CaliResult(
            experiment=graph[0].id,
            detection_settings_id=graph[1],
            extraction_settings_id=graph[2],
            positions_extracted=[0],
        )
        trace = _trace(2, 5)
        trace.roi, trace.analysis_result = fov.rois[1], owner
        session.add(trace)
        session.commit()
    with pytest.raises(ValueError, match="ROIs span extraction generations"):
        _run(path, graph)
    with Session(engine) as session:
        assert len(session.exec(select(CaliResult)).all()) == 3
        assert not session.exec(
            select(CaliResult).where(CaliResult.analysis_settings_id == graph[3])
        ).all()
    engine.dispose()


def test_preflight_trace_ids_remain_pinned_when_newer_traces_appear(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pinned.cali"
    graph = _database(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    runner = CaliRunner()
    with Session(engine) as session:
        sources, trace_ids = runner._plan_analysis_sources(
            session, [0, 1, 2], graph[2], graph[1]
        )
        fov = session.exec(select(FOV).where(FOV.position_index == 0)).one()
        owner = CaliResult(
            experiment=graph[0].id,
            detection_settings_id=graph[1],
            extraction_settings_id=graph[2],
        )
        for roi in fov.rois:
            trace = _trace(roi.label_value, 10)
            trace.roi, trace.analysis_result = roi, owner
            session.add(trace)
        session.commit()
        runner._pin_analysis_source_traces(
            session,
            [fov],
            graph[2],
            graph[1],
            trace_ids_by_roi=trace_ids,
            source_ids_by_position=sources,
        )
        assert all(
            roi._analysis_source_trace.id == trace_ids[roi.id] for roi in fov.rois
        )
        assert fov.rois[0]._analysis_source_trace.get_spike_values("oasis")[10] == 1
    engine.dispose()


def test_cancelled_request_records_only_completed_source_positions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cancelled.cali"
    graph = _database(path)
    runner = CaliRunner()
    original = runner._run_analysis_only

    def cancelled(settings: AnalysisSettings, fovs: list[FOV]) -> Iterator[FOV]:
        for fov in original(settings, fovs):
            yield fov
            runner.cancel()
            return

    with patch.object(runner, "_run_analysis_only", cancelled):
        _run(path, graph, runner)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        results = session.exec(
            select(CaliResult)
            .where(CaliResult.analysis_settings_id == graph[3])
            .order_by(CaliResult.id)
        ).all()
        assert [result.positions_analyzed for result in results] == [[0], []]
        assert results[0].source_extraction_result_id == graph[-1][0]
        assert len(results[0].traces) == 2 and not results[1].traces
        assert all(result.positions_extracted is None for result in results)
    engine.dispose()


def test_new_extraction_and_reused_sources_are_separate(tmp_path: Path) -> None:
    path = tmp_path / "mixed_work.cali"
    graph = _database(path, extracted=False)
    runner = CaliRunner()

    def extracted(
        dataset: Any,
        extraction: ExtractionSettings,
        settings: AnalysisSettings,
        fovs: list[FOV],
    ) -> Iterator[FOV]:
        for fov in fovs:
            for roi in fov.rois:
                roi._new_traces = [_trace(roi.label_value, 3)]
            yield from AnalysisRunner().run([fov], settings, as_generator=True)

    with (
        patch("cali.runner._cali_runner.load_data_from_path", return_value=MagicMock()),
        patch.object(runner, "_run_extraction", extracted),
    ):
        runner.run(
            graph[0],
            tmp_path / "source.tensorstore.zarr",
            graph[1],
            extraction_settings=graph[2],
            analysis_settings=graph[3],
            database_name=path.name,
            output_path=tmp_path,
            global_position_indices=[0, 1, 2],
        )
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        results = session.exec(select(CaliResult).order_by(CaliResult.id)).all()
        assert len(results) == 5
        new = results[2]
        assert new.positions_extracted == new.positions_analyzed == [2]
        assert new.source_extraction_result_id == new.id
        for copy in results[3:]:
            _verify_copy(session, copy)
        assert not session.exec(select(MigrationIssue)).all()
    engine.dispose()


def test_forced_selected_source_preserves_other_generations(tmp_path: Path) -> None:
    path = tmp_path / "forced.cali"
    graph = _database(path, 3)
    _run(path, graph)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        results = session.exec(select(CaliResult).order_by(CaliResult.id)).all()
        original_ids = [trace.id for source in results[:2] for trace in source.traces]
        other_id = results[3].id
        other_trace_ids = [trace.id for trace in results[3].traces]
    _run(
        path,
        graph,
        source_extraction_result_id=graph[-1][0],
        global_position_indices=[0, 2],
        force=True,
    )
    with Session(engine) as session:
        results = session.exec(select(CaliResult).order_by(CaliResult.id)).all()
        assert len(results) == 4
        assert [
            trace.id for source in results[:2] for trace in source.traces
        ] == original_ids
        other = session.get(CaliResult, other_id)
        assert [trace.id for trace in other.traces] == other_trace_ids
        assert other.positions_analyzed == [1]
        for result in results[2:]:
            _verify_copy(session, result)
        assert len(results[2].traces) == 4
        assert len(results[2].data_analysis_results) == 4
        assert len(results[2].fov_analysis_results) == 2
    engine.dispose()


def test_preflight_rejects_rois_added_during_a_request(tmp_path: Path) -> None:
    path = tmp_path / "new_roi.cali"
    graph = _database(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    runner = CaliRunner()
    with Session(engine) as session:
        sources, trace_ids = runner._plan_analysis_sources(
            session, [0], graph[2], graph[1]
        )
        fov = session.exec(select(FOV).where(FOV.position_index == 0)).one()
        session.add(ROI(fov=fov, label_value=3, detection_settings_id=graph[1]))
        session.commit()
        with pytest.raises(
            ValueError, match="source selection changed after preflight"
        ):
            runner._pin_analysis_source_traces(
                session,
                [fov],
                graph[2],
                graph[1],
                trace_ids_by_roi=trace_ids,
                source_ids_by_position=sources,
            )
        assert not session.new and not session.dirty
    engine.dispose()


def test_completed_source_positions_survive_a_later_worker_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "worker_error.cali"
    graph = _database(path)
    runner = CaliRunner()
    original = runner._run_analysis_only

    def interrupted(settings: AnalysisSettings, fovs: list[FOV]) -> Iterator[FOV]:
        yield from original(settings, fovs)
        raise RuntimeError("worker stopped after a completed FOV")

    with (
        patch.object(runner, "_run_analysis_only", interrupted),
        pytest.raises(RuntimeError, match="worker stopped"),
    ):
        _run(path, graph, runner)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        results = session.exec(
            select(CaliResult)
            .where(CaliResult.analysis_settings_id == graph[3])
            .order_by(CaliResult.id)
        ).all()
        assert [result.positions_analyzed for result in results] == [[0], []]
        _verify_copy(session, results[0])
        assert not results[1].traces
    engine.dispose()


def test_incomplete_batch_rolls_back_source_products_and_stage_flags(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete_batch.cali"
    graph = _database(path, 3)
    runner = CaliRunner()
    original = runner._run_analysis_only

    def interrupted(settings: AnalysisSettings, fovs: list[FOV]) -> Iterator[FOV]:
        iterator = original(settings, fovs)
        yield next(iterator)
        raise RuntimeError("worker stopped before the batch commit")

    with (
        patch.object(runner, "_run_analysis_only", interrupted),
        pytest.raises(RuntimeError, match="worker stopped"),
    ):
        _run(path, graph, runner)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        results = session.exec(
            select(CaliResult).where(CaliResult.analysis_settings_id == graph[3])
        ).all()
        assert len(results) == 2
        for result in results:
            assert result.positions_analyzed == []
            assert result.positions_extracted is None
            assert not result.traces
            assert not result.data_analysis_results
            assert not result.fov_analysis_results
    engine.dispose()
