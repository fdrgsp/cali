"""Public runner parity, persistence, offline reuse and failed-FOV isolation."""

import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numba import njit
from sqlmodel import Session, select

from cali.analysis import AnalysisRunner
from cali.extraction._extraction_runner import ExtractionRunner
from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    FOVAnalysis,
    Plate,
    SpikeAnalysisSettings,
    Traces,
    Well,
    create_cali_engine,
    create_database_and_tables,
)

from .test_cascade_extraction import _dataset, _fov, _settings
from .test_cascade_reference import fake_reference as fake_reference
from .test_cascade_roi_analysis import _trace


@njit
def _seed_ccg() -> None:
    np.random.seed(42)


@pytest.fixture
def deterministic_ccg() -> object:
    """Compare real CCG calculations using identical draws in Numba's RNG."""
    from cali.analysis._fov_metrics import _compute_baseline_corrected_ccg_numba

    def compute(*args: object, **kwargs: object) -> object:
        _seed_ccg()
        return _compute_baseline_corrected_ccg_numba(*args, **kwargs)

    with patch(
        "cali.analysis._fov_metrics._compute_baseline_corrected_ccg_numba",
        side_effect=compute,
    ):
        yield


def _analysis(methods: tuple, rate: float = 10, **kwargs: object) -> AnalysisSettings:
    return AnalysisSettings(
        frame_rate=rate,
        threads=1,
        peaks_height_mode="global",
        peaks_height_value=0.005,
        spike_settings=[
            SpikeAnalysisSettings(
                method=method,
                threshold_mode="global",
                threshold_value=0.001 if method == "oasis" else 0.2,
                enable_rising_edge_analysis=True,
                ccg_n_shuffles=2,
            )
            for method in methods
        ],
        **kwargs,
    )


def _scalar_snapshot(model: object) -> str:
    excluded = {
        "id",
        "created_at",
        "roi_id",
        "fov_id",
        "analysis_result_id",
        "data_analysis_id",
        "spike_trace_id",
        "spike_inference_run_id",
        "fov_analysis_id",
    }
    return json.dumps(
        {
            name: getattr(model, name)
            for name in type(model).model_fields
            if name not in excluded
        },
        sort_keys=True,
    )


def _products(session: Session, run_id: int) -> tuple:
    traces = session.exec(
        select(Traces)
        .where(Traces.analysis_result_id == run_id)
        .order_by(Traces.roi_id)
    ).all()
    roi = session.exec(
        select(DataAnalysis)
        .where(DataAnalysis.analysis_result_id == run_id)
        .order_by(DataAnalysis.roi_id)
    ).all()
    fov = session.exec(
        select(FOVAnalysis).where(FOVAnalysis.analysis_result_id == run_id)
    ).one()
    return (
        [(trace.den_dff, trace.calcium_noise) for trace in traces],
        [_scalar_snapshot(parent) for parent in roi],
        _scalar_snapshot(fov),
        {
            method: [
                _scalar_snapshot(child)
                for parent in roi
                if (child := parent.get_spike_analysis(method)) is not None
            ]
            for method in ("oasis", "cascade")
        },
        {child.method: _scalar_snapshot(child) for child in fov.spike_analyses},
    )


def _seed(
    path: Path, methods: tuple, pretrained: bool = False, mask_path: Path | None = None
) -> tuple:
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        experiment = Experiment(name="full method analysis")
        if pretrained:
            metadata = json.loads(
                (
                    Path(__file__).parent / "fixtures/cascade_reference/manifest.json"
                ).read_text()
            )
            extraction = ExtractionSettings(
                spike_methods=methods,
                cascade_model=metadata["model_name"] if "cascade" in methods else None,
                cascade_device="cpu",
                frame_rate=30,
                dff_window=5,
                neuropil_inner_radius=0,
                discard_initial_value=11,
                threads=1,
            )
        else:
            extraction = _settings(methods, discard_initial_value=11, threads=1)
        detection, analysis = (
            DetectionSettings(),
            _analysis(
                methods,
                rate=30 if pretrained else 10,
                **(
                    {
                        "stimulation_mask_path": str(mask_path),
                        "experiment_type": "Evoked Activity",
                    }
                    if mask_path
                    else {}
                ),
            ),
        )
        well = Well(
            name="A1", row=0, column=0, plate=Plate(name="plate", experiment=experiment)
        )
        fov = _fov()
        fov.well = well
        session.add_all([detection, extraction, analysis, fov])
        session.flush()
        for roi in fov.rois:
            roi.detection_settings_id = detection.id
        owner = CaliResult(
            experiment=experiment.id,
            detection_settings_id=detection.id,
            positions_detected=[0],
        )
        session.add(owner)
        session.commit()
        graph = experiment, detection.id, extraction.id, analysis.id
        session.refresh(experiment)
        session.expunge(experiment)
    engine.dispose()
    return graph


def _run(path: Path, graph: tuple, **kwargs: object) -> None:
    experiment, detection, extraction, analysis = graph
    CaliRunner().run(
        experiment,
        kwargs.pop("dataset_path", None),
        detection,
        extraction_settings=extraction,
        analysis_settings=analysis,
        global_position_indices=[0],
        database_name=path.name,
        output_path=path.parent,
        **kwargs,
    )


@pytest.mark.parametrize("stimulated", [False, True])
def test_public_combined_and_offline_reanalysis_match_single_and_dual_outputs(
    fake_reference: tuple,
    deterministic_ccg: object,
    tmp_path: Path,
    stimulated: bool,
) -> None:
    _assert_public_parity(tmp_path, stimulated=stimulated)


@pytest.mark.skipif(
    os.environ.get("CALI_CASCADE_REFERENCE_TESTS") != "1",
    reason="Pretrained full-runner acceptance runs in the installed-wheel job",
)
def test_pretrained_public_combined_and_offline_reanalysis_match_all_modes(
    tmp_path: Path, deterministic_ccg: object
) -> None:
    _assert_public_parity(tmp_path, pretrained=True)


def _assert_public_parity(
    tmp_path: Path, pretrained: bool = False, stimulated: bool = False
) -> None:
    mask_path = None
    if stimulated:
        import tifffile

        mask_path = tmp_path / "stimulation.tif"
        tifffile.imwrite(mask_path, np.ones((4, 4), dtype=np.uint8))
    modes = []
    for methods in (("oasis",), ("cascade",), ("oasis", "cascade")):
        path = tmp_path / ("-".join(methods) + ".cali")
        graph = _seed(path, methods, pretrained, mask_path)
        dataset = _dataset(count=256, rate=30 if pretrained else 10)
        image, _ = dataset.isel.return_value
        image[:, 2:, 2:] = image[:, :2, :2]
        dataset.sequence.stage_positions = [object()]
        with patch(
            "cali.runner._cali_runner.load_data_from_path", return_value=dataset
        ):
            _run(path, graph, dataset_path=tmp_path / "source.zarr")
        engine = create_cali_engine(f"sqlite:///{path}")
        with Session(engine) as session:
            owner = session.exec(
                select(CaliResult).where(CaliResult.extraction_settings_id == graph[2])
            ).one()
            source_id = owner.id
            assert owner.positions_extracted == owner.positions_analyzed == [0]
            inline = _products(session, source_id)
            assert set(inline[4]) == set(methods)
            if "cascade" in methods and not pretrained:
                child = owner.data_analysis_results[0].get_spike_analysis("cascade")
                assert child.expected_spike_count == pytest.approx(45.25)
                assert child.expected_spike_rate_hz == pytest.approx(2.5)
                assert child.suprathreshold_sample_rate_hz is None
                assert (
                    owner.fov_analysis_results[0]
                    .get_spike_analysis("cascade")
                    .valid_start
                    == 32
                )
        # Re-analysis must not load models, rerun inference or access imaging data.
        with patch(
            "cali.extraction._spike_inference._cascade_reference.load_cascade_package",
            side_effect=AssertionError("offline analysis loaded inference"),
        ):
            _run(path, graph, source_extraction_result_id=source_id, force=True)
        with Session(engine) as session:
            copy = session.exec(
                select(CaliResult).where(
                    CaliResult.source_extraction_result_id == source_id,
                    CaliResult.id != source_id,
                )
            ).one()
            assert copy.positions_extracted is None and copy.positions_analyzed == [0]
            assert _products(session, copy.id) == inline
            for parent in copy.data_analysis_results:
                for child in parent.spike_analyses:
                    assert child.spike_trace.trace.analysis_result_id == copy.id
                    assert (
                        child.spike_trace.inference_run.extraction_result_id
                        == source_id
                    )
        engine.dispose()
        modes.append(inline)
    assert modes[0][:3] == modes[1][:3] == modes[2][:3]
    for method, single in (("oasis", modes[0]), ("cascade", modes[1])):
        assert single[3][method] == modes[2][3][method]
        assert single[4][method] == modes[2][4][method]


def _source_fov() -> object:
    fov = _fov()
    reference = _trace(("oasis", "cascade"))
    for roi in fov.rois:
        trace = _trace(("oasis", "cascade"))
        trace.extraction_frame_window = reference.extraction_frame_window
        for child in trace.spike_traces:
            method = child.inference_run.method
            child.inference_run = reference.get_spike_trace(method).inference_run
        roi.traces_history = [trace]
        roi.active, roi.stimulated = False, True
    return fov


@pytest.mark.parametrize("failure", ["roi", "fov", "cancel_roi", "cancel_fov"])
def test_failed_or_cancelled_analysis_restores_every_roi_and_can_retry(
    failure: str,
) -> None:
    fov = _source_fov()
    runner = AnalysisRunner()
    settings = _analysis(("oasis", "cascade"))
    original = runner._analyze_roi_traces

    def analyze(*args: object, **kwargs: object) -> object:
        if kwargs["roi"].label_value == 8:
            if failure == "roi":
                raise ValueError("ROI failed")
            if failure == "cancel_roi":
                runner.cancel()
        return original(*args, **kwargs)

    def compute(*args: object) -> object:
        if failure == "fov":
            raise ValueError("FOV failed")
        if failure == "cancel_fov":
            runner.cancel()
            return FOVAnalysis()
        from cali.analysis._fov_analysis import compute_fov_analysis

        return compute_fov_analysis(*args)

    with (
        patch.object(runner, "_analyze_roi_traces", side_effect=analyze),
        patch(
            "cali.analysis._fov_analysis_parallel.compute_fov_analysis_parallel",
            side_effect=compute,
        ),
    ):
        if "cancel" in failure:
            assert runner.run([fov], settings) == []
        else:
            with pytest.raises(ValueError, match="failed"):
                runner.run([fov], settings)
    for roi in fov.rois:
        assert roi.active is False and roi.stimulated is True
        assert not hasattr(roi, "_new_data_analysis")
        assert not roi.data_analysis_history
    assert not hasattr(fov, "_pending_analysis_settings")
    assert not hasattr(fov, "_new_fov_analysis")
    assert runner.run([fov], settings) == [fov]
    assert all(len(roi._new_data_analysis) == 1 for roi in fov.rois)


def test_analysis_worker_failure_cancels_and_joins_other_workers() -> None:
    runner = AnalysisRunner()
    started, finished = threading.Event(), threading.Event()
    quick, slow = _fov(), _fov(1)

    def analyze(_settings: object, fov: object) -> None:
        if fov is slow:
            started.set()
            assert runner._cancellation_event.wait(5)
            finished.set()
            return
        assert started.wait(5)
        raise ValueError("worker failed")

    with pytest.raises(ValueError, match="worker failed"):
        list(
            runner._exec_in_threadpool(
                analyze,
                runner._cancellation_event,
                [quick, slow],
                _analysis(("oasis",)),
                max_workers=2,
            )
        )
    assert runner._cancellation_event.is_set() and finished.is_set()


def test_cascade_missing_axis_is_a_failure_not_a_skipped_roi() -> None:
    fov = _source_fov()
    fov.rois[1].traces_history[0].x_axis = None
    with pytest.raises(ValueError, match="retained time axis"):
        AnalysisRunner().run([fov], _analysis(("oasis", "cascade")))
    assert all(not hasattr(roi, "_new_data_analysis") for roi in fov.rois)


def test_closing_analysis_generator_restores_unpublished_fovs() -> None:
    first, second = _source_fov(), _source_fov()
    settings = _analysis(("oasis", "cascade"))
    old = [DataAnalysis()]
    for roi in second.rois:
        roi._new_data_analysis = old
    generator = AnalysisRunner().run([first, second], settings, as_generator=True)
    published = next(generator)
    unpublished = second if published is first else first
    generator.close()
    assert hasattr(published, "_new_fov_analysis")
    assert not hasattr(unpublished, "_new_fov_analysis")
    for roi in unpublished.rois:
        assert roi.active is False and roi.stimulated is True
        if unpublished is second:
            assert roi._new_data_analysis is old
        else:
            assert not hasattr(roi, "_new_data_analysis")


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_or_cancelled_extraction_fov_analysis_restores_prior_stage(
    fake_reference: tuple, cancel: bool
) -> None:
    runner, fov = ExtractionRunner(), _fov()
    previous_traces, previous_analysis = [Traces()], [DataAnalysis()]
    for roi in fov.rois:
        roi._new_traces, roi._new_data_analysis = previous_traces, previous_analysis
        roi.active, roi.stimulated, roi.cell_size = False, True, 77

    def compute(*args: object) -> object:
        if cancel:
            runner.cancel()
            return FOVAnalysis()
        raise ValueError("FOV failed")

    with patch(
        "cali.analysis._fov_analysis_parallel.compute_fov_analysis_parallel",
        side_effect=compute,
    ):
        if cancel:
            assert (
                runner.run(
                    _dataset(),
                    _settings(("cascade",)),
                    [fov],
                    analysis_settings=_analysis(("cascade",)),
                )
                == []
            )
        else:
            with pytest.raises(ValueError, match="FOV failed"):
                runner.run(
                    _dataset(),
                    _settings(("cascade",)),
                    [fov],
                    analysis_settings=_analysis(("cascade",)),
                )
    for roi in fov.rois:
        assert roi._new_traces is previous_traces
        assert roi._new_data_analysis is previous_analysis
        assert roi.active is False and roi.stimulated is True and roi.cell_size == 77
    assert not hasattr(fov, "_new_fov_analysis")
    assert not hasattr(fov, "_pending_analysis_settings")


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("failure", ["roi", "fov"])
def test_public_failed_dual_run_saves_no_partial_products_and_preserves_source(
    fake_reference: tuple,
    deterministic_ccg: object,
    tmp_path: Path,
    offline: bool,
    failure: str,
) -> None:
    path = tmp_path / "failure.cali"
    graph = _seed(path, ("oasis", "cascade"))
    dataset = _dataset(count=256)
    dataset.sequence.stage_positions = [object()]
    engine = create_cali_engine(f"sqlite:///{path}")
    source_id, source_products = None, None
    if offline:
        with patch(
            "cali.runner._cali_runner.load_data_from_path", return_value=dataset
        ):
            _run(path, graph, dataset_path=tmp_path / "source.zarr")
        with Session(engine) as session:
            source_id = session.exec(select(CaliResult)).one().id
            source_products = _products(session, source_id)

    target = (
        "cali.analysis._fov_analysis_parallel.compute_fov_analysis_parallel"
        if failure == "fov"
        else (
            "cali.analysis._analysis_runner.AnalysisRunner._analyze_roi_traces"
            if offline
            else "cali.extraction._extraction_runner.ExtractionRunner._finalize_roi"
        )
    )
    with (
        patch("cali.runner._cali_runner.load_data_from_path", return_value=dataset),
        patch(target, side_effect=ValueError("selected analysis failed")),
        pytest.raises(ValueError, match="selected analysis failed"),
    ):
        _run(
            path,
            graph,
            dataset_path=None if offline else tmp_path / "source.zarr",
            source_extraction_result_id=source_id,
            force=offline,
        )
    with Session(engine) as session:
        owners = session.exec(select(CaliResult)).all()
        for owner in owners:
            if owner.id == source_id:
                assert owner.positions_extracted == owner.positions_analyzed == [0]
                assert _products(session, owner.id) == source_products
                continue
            assert not owner.positions_extracted and not owner.positions_analyzed
            assert not owner.traces and not owner.data_analysis_results
            assert not owner.fov_analysis_results
        for model in (Traces, DataAnalysis, FOVAnalysis):
            rows = session.exec(select(model)).all()
            assert all(row.analysis_result_id == source_id for row in rows)
    engine.dispose()
