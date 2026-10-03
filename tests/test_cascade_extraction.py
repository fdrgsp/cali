"""Selected batched outputs preserve calcium and publish complete FOVs only."""

import json
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from rich.console import Console
from rich.tree import Tree
from sqlmodel import Session, select

from cali._cascade_models import CascadeModelError
from cali._constants import CASCADE_EXPECTED_SPIKES_TRACES, INFERRED_SPIKES_TRACES
from cali.extraction._extraction_runner import ExtractionRunner
from cali.extraction._frame_window import StartupDiscardError
from cali.extraction._spike_inference import OasisBackend
from cali.extraction._spike_inference import _cascade_reference as reference
from cali.extraction._spike_inference._cascade_service import CascadeInferenceService
from cali.readers import TensorstoreZarrReader
from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    Experiment,
    ExtractionFrameWindow,
    ExtractionSettings,
    Mask,
    SpikeAnalysisSettings,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    create_cali_engine,
    create_database_and_tables,
    ensure_schema_current,
)
from cali.sqlmodel._visualize_experiment import _add_trace_provenance_to_tree
from cali.util._database_to_csv import export_traces_to_csv

# Reuse the pinned-package fakes used by the independent backend contract tests.
from .test_cascade_cached import toy_package as cached_fixture
from .test_cascade_reference import fake_reference as reference_fixture

toy_package = cached_fixture
fake_reference = reference_fixture


def _dataset(count: int = 128, rate: float = 10) -> MagicMock:
    rng = np.random.default_rng(42)
    data = rng.normal(100, 1, (count, 4, 4))
    for onset in (15, 50, 90):
        if onset < count:
            data[onset:, :2, :2] += (
                20 * np.exp(-np.arange(count - onset) / 8)[:, None, None]
            )
    dataset = MagicMock(spec=TensorstoreZarrReader)
    dataset.path = Path("source.zarr")
    dataset.isel.return_value = (
        data,
        [{"runner_time_ms": 500 + i * 1000 / rate} for i in range(count)],
    )
    return dataset


def _fov(position: int = 0) -> FOV:
    return FOV(
        name=f"A1_{position:04d}",
        position_index=position,
        rois=[
            ROI(
                label_value=7,
                roi_mask=Mask(
                    coords_y=[0, 0, 1, 1],
                    coords_x=[0, 1, 0, 1],
                    height=4,
                    width=4,
                    mask_type="roi",
                ),
            ),
            ROI(
                label_value=8,
                roi_mask=Mask(
                    coords_y=[2, 2, 3, 3],
                    coords_x=[2, 3, 2, 3],
                    height=4,
                    width=4,
                    mask_type="roi",
                ),
            ),
        ],
    )


def _settings(methods: tuple = ("cascade",), **kwargs: object) -> ExtractionSettings:
    return ExtractionSettings(
        spike_methods=methods,
        cascade_model="Test_10Hz" if "cascade" in methods else None,
        cascade_device="cpu",
        frame_rate=10,
        dff_window=5,
        neuropil_inner_radius=0,
        **kwargs,
    )


@pytest.mark.parametrize("discard", [0, 11])
@pytest.mark.parametrize("calcium_analysis", [False, True])
def test_all_output_modes_preserve_exact_oasis_calcium_and_noise(
    discard: int,
    calcium_analysis: bool,
    fake_reference: tuple,
) -> None:
    _, _, calls = fake_reference
    baselines = []
    for methods in (("oasis",), ("cascade",), ("oasis", "cascade")):
        fov = _fov()
        settings = _settings(methods, discard_initial_value=discard)
        analysis = (
            AnalysisSettings(
                frame_rate=10,
                enable_spikes=False,
                spike_settings=[SpikeAnalysisSettings(method=m) for m in methods],
            )
            if calcium_analysis
            else None
        )
        with patch.object(
            OasisBackend, "infer_all", wraps=OasisBackend().infer_all
        ) as oasis:
            result = ExtractionRunner().run(
                _dataset(), settings, [fov], analysis_settings=analysis
            )
        assert result == [fov]
        oasis.assert_called_once()
        assert oasis.call_args.args[0].shape == (2, 128 - discard)
        assert oasis.call_args.kwargs["frame_rate"] == (128 - discard) / (
            (127 - discard) / 10
        )
        rows = []
        for roi in fov.rois:
            trace = roi._new_traces[0]
            assert tuple(c.inference_run.method for c in trace.spike_traces) == methods
            window = trace.extraction_frame_window
            assert window.source_start_frame == discard
            assert window.retained_frame_count == 128 - discard
            assert window.acquisition_frame_rate_hz == 10
            assert trace.calcium_noise is not None
            rows.append(
                (
                    trace.den_dff,
                    trace.calcium_noise,
                    roi._new_data_analysis[0].peaks_den_dff if analysis else None,
                )
            )
            cascade = trace.get_spike_trace("cascade")
            if cascade:
                assert (cascade.valid_start, cascade.resolved_valid_stop) == (
                    32,
                    96 - discard,
                )
                np.testing.assert_array_equal(cascade.values[:32], 0)
                assert cascade.selected_noise_level in (2, 3, 4)
                assert cascade.inference_run.weights_manifest_sha256 == "manifest"
                assert cascade.inference_run.config_sha256 == "config-hash"
                assert cascade.inference_run.backend_revision == "package-pin"
            if len(methods) == 2:
                with pytest.raises(ValueError, match="Multiple spike"):
                    _ = trace.inferred_spikes
        baselines.append(rows)
    assert baselines[0] == baselines[1] == baselines[2]
    assert sum(c[0] == "predict" for c in calls) == 2


@pytest.mark.parametrize(
    "failure", ["cascade_error", "cascade_cancel", "finalize_error", "finalize_cancel"]
)
def test_dual_failure_or_cancel_leaves_every_roi_untouched(
    failure: str,
    fake_reference: tuple,
) -> None:
    _, package, _ = fake_reference
    runner = ExtractionRunner()
    fov = _fov()
    for roi in fov.rois:
        roi.cell_size = 77
        roi.active = True
    original_predict = package.cascade.predict
    original_finalize = runner._finalize_roi
    finalized = 0

    def predict(*args: object, **kwargs: object) -> np.ndarray:
        if failure == "cascade_error":
            raise ValueError("selected backend failed")
        result = original_predict(*args, **kwargs)
        if failure == "cascade_cancel":
            runner.cancel()
        return result

    def finalize(*args: object, **kwargs: object) -> object:
        nonlocal finalized
        result = original_finalize(*args, **kwargs)
        finalized += 1
        if finalized == 2:
            if failure == "finalize_error":
                raise ValueError("ROI finalization failed")
            if failure == "finalize_cancel":
                runner.cancel()
        return result

    with (
        patch.object(package.cascade, "predict", side_effect=predict),
        patch.object(runner, "_finalize_roi", side_effect=finalize),
    ):
        if failure.endswith("error"):
            with pytest.raises(ValueError, match="failed"):
                runner.run(_dataset(), _settings(("oasis", "cascade")), [fov])
        else:
            assert runner.run(_dataset(), _settings(("oasis", "cascade")), [fov]) == []
    assert all(not hasattr(roi, "_new_traces") for roi in fov.rois)
    assert all(roi.cell_size == 77 and roi.active for roi in fov.rois)


@pytest.mark.parametrize("failure", ["package", "model", "device", "settings_rate"])
def test_component_preflight_happens_before_reading_images_or_oasis(
    failure: str,
    fake_reference: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, package, _ = fake_reference
    settings = _settings(("oasis", "cascade"))
    if failure == "package":
        monkeypatch.setattr(
            reference,
            "load_cascade_package",
            MagicMock(side_effect=ImportError("missing package")),
        )
    elif failure == "model":
        monkeypatch.setattr(
            reference,
            "load_cascade_model",
            MagicMock(side_effect=CascadeModelError("missing model")),
        )
    elif failure == "device":
        settings.cascade_device = "cuda"
    else:
        settings.frame_rate = 30
    dataset = _dataset()
    with patch.object(OasisBackend, "infer_all") as oasis:
        with pytest.raises((ImportError, CascadeModelError)):
            ExtractionRunner().run(dataset, settings, [_fov()])
    dataset.isel.assert_not_called()
    oasis.assert_not_called()
    assert not package.torch.cuda.is_available()


@pytest.mark.parametrize(
    "failure", ["exposure", "irregular", "recording_rate", "short_crop"]
)
def test_model_timing_and_length_preflight_happens_before_roi_work(
    failure: str,
    fake_reference: tuple,
) -> None:
    dataset = _dataset(rate=20 if failure == "recording_rate" else 10)
    settings = _settings()
    data, meta = dataset.isel.return_value
    if failure == "exposure":
        meta = [{"exposure_ms": 100} for _ in meta]
    elif failure == "irregular":
        meta[40]["runner_time_ms"] += 20
    elif failure == "short_crop":
        settings.discard_initial_value = 64
    dataset.isel.return_value = (data, meta)
    runner = ExtractionRunner()
    with patch.object(runner, "_compute_roi_dff") as compute:
        with pytest.raises(StartupDiscardError, match="position 0, source"):
            runner.run(dataset, settings, [_fov()])
    compute.assert_not_called()


def test_cached_service_is_shared_across_fov_pool_and_closed(
    toy_package: tuple,
) -> None:
    model, _, state = toy_package
    services = []

    # Keep isinstance valid while recording construction through __init__.
    original = CascadeInferenceService.__init__

    def initialize(
        self: CascadeInferenceService, *args: object, **kwargs: object
    ) -> None:
        original(self, *args, **kwargs)
        services.append(self)

    with patch.object(CascadeInferenceService, "__init__", initialize):
        output = ExtractionRunner(experimental_cascade_cache=True).run(
            _dataset(), _settings(("oasis", "cascade"), threads=2), [_fov(), _fov(1)]
        )
    assert len(output) == 2
    assert len(services) == 1
    service = services[0]
    assert not service._worker.is_alive()
    assert service.stats.ensembles == 0
    assert service.stats.model_loads == 4
    assert {item[1] for item in state["loads"]} == {service._worker.ident}
    assert service._worker.ident != threading.get_ident()
    for fov in output:
        for roi in fov.rois:
            run = roi._new_traces[0].get_spike_trace("cascade").inference_run
            assert run.resolved_model == model.name
            assert run.provenance_source == "cascade_cached_inference"


def test_closing_generator_cancels_fov_pool_and_joins_service(
    toy_package: tuple,
) -> None:
    runner = ExtractionRunner(experimental_cascade_cache=True)
    original_analyze = runner._analyze_position
    original_initialize = CascadeInferenceService.__init__
    started = threading.Event()
    services = []

    def initialize(
        self: CascadeInferenceService, *args: object, **kwargs: object
    ) -> None:
        original_initialize(self, *args, **kwargs)
        services.append(self)

    def analyze(*args: object, **kwargs: object) -> FOV | None:
        fov = args[3]
        if fov.position_index == 1:
            started.set()
            assert runner._cancellation_event.wait(10)
            return None
        return original_analyze(*args, **kwargs)

    pending = _fov(1)
    with (
        patch.object(runner, "_analyze_position", side_effect=analyze),
        patch.object(CascadeInferenceService, "__init__", initialize),
    ):
        output = runner.run(
            _dataset(), _settings(threads=2), [_fov(), pending], as_generator=True
        )
        assert next(output).position_index == 0
        assert started.wait(5)
        output.close()
    assert runner._check_for_abort_requested()
    assert not services[0]._worker.is_alive()
    assert services[0].stats.ensembles == 0
    assert all(not hasattr(roi, "_new_traces") for roi in pending.rois)


def test_plate_session_reuses_models_across_separate_fov_batches(
    toy_package: tuple,
) -> None:
    _, _, state = toy_package
    runner = ExtractionRunner(experimental_cascade_cache=True)
    settings = _settings(("oasis", "cascade"))
    with runner.inference_session(settings):
        service = runner._active_cascade_backend
        assert isinstance(service, CascadeInferenceService)
        for position in range(3):
            assert runner.run(_dataset(), settings, [_fov(position)])
            assert service.stats.model_loads == 4
            assert service._worker.is_alive()
        assert len(state["loads"]) == 4
        with pytest.raises(RuntimeError, match="already active"):
            with runner.inference_session(settings):
                pass
        changed = _settings(("cascade",))
        with pytest.raises(RuntimeError, match="settings changed"):
            runner.run(_dataset(), changed, [_fov(3)])
    assert not service._worker.is_alive()
    assert runner._active_cascade_backend is None


@pytest.mark.parametrize("methods", [("cascade",), ("oasis", "cascade")])
def test_extraction_only_persists_and_exports_every_selected_method(
    methods: tuple,
    fake_reference: tuple,
    tmp_path: Path,
) -> None:
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'outputs.cali'}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            experiment = Experiment(name="actual extraction")
            fov = _fov()
            settings = _settings(methods, discard_initial_value=11)
            session.add_all([experiment, fov, settings])
            session.flush()
            result = CaliResult(
                experiment=experiment.id,
                extraction_settings_id=settings.id,
                positions_extracted=[0],
            )
            session.add(result)
            session.commit()
            run_id = result.id
            assert ExtractionRunner().run(_dataset(), settings, [fov]) == [fov]
            CaliRunner()._process_fov_results(fov, session, run_id, include_traces=True)
            session.commit()
            session.expunge_all()
            traces = session.exec(select(Traces)).all()
            assert len(traces) == 2
            assert len(session.exec(select(SpikeTrace)).all()) == 2 * len(methods)
            assert len(session.exec(select(SpikeInferenceRun)).all()) == len(methods)
            assert len(session.exec(select(ExtractionFrameWindow)).all()) == 1
            assert all(trace.calcium_noise is not None for trace in traces)
            assert all(
                trace.get_spike_trace("cascade").inference_run.extraction_result_id
                == run_id
                for trace in traces
            )
            tree = Tree("extraction")
            _add_trace_provenance_to_tree(tree, traces)
            console = Console(record=True, width=200)
            console.print(tree)
            text = console.export_text()
            for expected in (
                "cascade: spikes/frame",
                "model: Test_10Hz",
                "manifest: manifest",
                "device: cpu",
                "[32, 85)",
                "source start: 11",
            ):
                assert expected in text

        export_traces_to_csv(
            engine,
            {
                CASCADE_EXPECTED_SPIKES_TRACES: True,
                INFERRED_SPIKES_TRACES: "oasis" in methods,
            },
            run_id,
            tmp_path / "outputs.cali",
        )
        target = tmp_path / "outputs_exports" / f"run_{run_id}"
        frame = pd.read_csv(target / "cascade_expected_spikes.csv")
        assert frame.shape == (117, 2)
        assert frame.iloc[:32].isna().all().all()
        assert frame.iloc[85:].isna().all().all()
        np.testing.assert_array_equal(frame.iloc[32:85], 0.25)
        if "oasis" in methods:
            assert (target / "oasis_inferred_spikes_raw.csv").exists()
            assert not (target / "inferred_spikes_raw.csv").exists()
            # A stored dual result keeps method labels even for a single export.
            export_traces_to_csv(
                engine,
                {INFERRED_SPIKES_TRACES: True},
                run_id,
                tmp_path / "outputs.cali",
            )
            assert not (target / "inferred_spikes_raw.csv").exists()
        metadata = json.loads((target / "trace_metadata.json").read_text())
        first = metadata["traces"][0]
        assert first["frame_window"]["source_start_frame"] == 11
        assert first["frame_window"]["requested_discard_value"] == 11
        assert (
            tuple(child["provenance"]["method"] for child in first["spike_outputs"])
            == methods
        )
        assert first["spike_outputs"][-1]["valid_stop_frame_exclusive"] == 85
        assert (
            first["spike_outputs"][-1]["provenance"]["weights_manifest_sha256"]
            == "manifest"
        )
    finally:
        engine.dispose()


def test_detached_settings_expose_all_dispatch_and_timing_columns(
    fake_reference: tuple,
) -> None:
    engine = create_cali_engine("sqlite:///:memory:")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            session.add(
                _settings(
                    ("oasis", "cascade"),
                    frame_rate_verified=True,
                    discard_initial_value=0.5,
                    discard_initial_unit="seconds",
                )
            )
            session.commit()
            settings = session.exec(select(ExtractionSettings)).one()
            session.expunge(settings)
        assert (
            settings.spike_methods,
            settings.cascade_model,
            settings.cascade_device,
            settings.frame_rate,
            settings.frame_rate_verified,
            settings.discard_initial_value,
            settings.discard_initial_unit,
        ) == (("oasis", "cascade"), "Test_10Hz", "cpu", 10, True, 0.5, "seconds")
        assert ExtractionRunner().run(_dataset(), settings, [_fov()])
    finally:
        engine.dispose()


def test_calcium_noise_migration_leaves_historical_values_unknown(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.cali"
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        session.add(
            Traces(raw_trace=[1, 2], den_dff=[0.1, 0.2], inferred_spikes=[0, 1])
        )
        session.commit()
    with engine.begin() as connection:
        connection.exec_driver_sql("ALTER TABLE trace DROP COLUMN calcium_noise")
        connection.exec_driver_sql("PRAGMA user_version = 9")
    ensure_schema_current(engine)
    ensure_schema_current(engine)
    with Session(engine) as session:
        trace = session.exec(select(Traces)).one()
        assert trace.calcium_noise is None
        assert trace.raw_trace == [1, 2] and trace.inferred_spikes == [0, 1]
    engine.dispose()


@pytest.mark.skipif(
    os.environ.get("CALI_CASCADE_REFERENCE_TESTS") != "1",
    reason="Actual pretrained extraction runs in the optional installed-wheel job",
)
@pytest.mark.parametrize("experimental", [False, True])
def test_pretrained_extraction_matches_upstream_and_preserves_calcium(
    experimental: bool,
) -> None:
    metadata = json.loads(
        (Path(__file__).parent / "fixtures/cascade_reference/manifest.json").read_text()
    )
    package = reference.load_cascade_package()
    backend = reference.CascadeReferenceBackend(
        metadata["model_name"],
        expected_manifest=metadata["model_manifest_sha256"],
        device="cpu",
    )
    modes = []
    oasis_spikes = []
    for methods in (("oasis",), ("cascade",), ("oasis", "cascade")):
        settings = ExtractionSettings(
            spike_methods=methods,
            cascade_model=metadata["model_name"] if "cascade" in methods else None,
            cascade_device="cpu",
            frame_rate=30,
            dff_window=5,
            neuropil_inner_radius=0,
            discard_initial_value=7,
        )
        fov = _fov()
        output = ExtractionRunner(experimental_cascade_cache=experimental).run(
            _dataset(count=256, rate=30), settings, [fov]
        )
        assert output == [fov]
        traces = [roi._new_traces[0] for roi in fov.rois]
        modes.append([(trace.den_dff, trace.calcium_noise) for trace in traces])
        if "oasis" in methods:
            oasis_spikes.append([trace.get_spike_values("oasis") for trace in traces])
        if "cascade" in methods:
            dff = np.asarray([trace.dff for trace in traces])
            noise = package.utils.calculate_noise_levels(
                dff, backend.model.sampling_rate
            )
            expected = package.cascade.predict(
                backend.model.name,
                dff,
                model_folder=str(backend.model.directory.parent),
                threshold=0,
                padding=0,
                trace_noise_levels=noise,
                verbosity=0,
                device=package.torch.device("cpu"),
            )
            np.testing.assert_allclose(
                [trace.get_spike_values("cascade") for trace in traces],
                expected,
                rtol=1e-5,
                atol=1e-6,
            )
            for trace in traces:
                run = trace.get_spike_trace("cascade").inference_run
                assert run.weights_manifest_sha256 == metadata["model_manifest_sha256"]
                assert run.resolved_device == "cpu" and run.dtype == "float32"
    assert modes[0] == modes[1] == modes[2]
    assert oasis_spikes[0] == oasis_spikes[1]
