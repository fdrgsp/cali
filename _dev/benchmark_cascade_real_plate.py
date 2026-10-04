"""Preflight and benchmark an independently recorded plate with existing masks.

The input database is opened read-only and backed up before schema migration.
No detection is performed. Each measured phase writes a fresh database containing
only the selected masks and new products. Use --preflight-only before scheduling
the full seven-case CPU comparison. Input provenance is recorded, not inferred:
this command cannot certify that a recording is biologically representative.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, closing, contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch
from urllib.parse import quote

import numpy as np
from benchmark_cascade_extraction import (
    CASES,
    STOCHASTIC_FIELDS,
    BenchmarkRunner,
    Measurements,
    ProcessMemorySampler,
    peak_mib,
    scientific_fields,
)
from sqlmodel import Session, select

from cali._cascade_models import load_cascade_model
from cali._cascade_package import load_cascade_package
from cali.analysis import AnalysisRunner
from cali.extraction._frame_window import (
    build_timing_descriptor,
    preflight_retained_timing,
    resolve_initial_frame_window,
    validate_model_timing,
)
from cali.extraction._spike_inference import OasisBackend
from cali.extraction._spike_inference._cascade_reference import CascadeReferenceBackend
from cali.extraction._spike_inference._cascade_service import CascadeInferenceService
from cali.readers import TensorstoreZarrReader, TiffCollectionReader
from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    Mask,
    Plate,
    SpikeAnalysisSettings,
    Well,
    create_cali_engine,
    create_database_and_tables,
)
from cali.sqlmodel._trace_array_codec import decode_trace_array
from cali.util import commit_fov_result, export_noise_qc_to_csv, load_data_from_path

if TYPE_CHECKING:
    from collections.abc import Generator


def checksum(path: Path) -> str:
    """Hash a file without reading the entire database into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def readonly(path: Path) -> sqlite3.Connection:
    """Open an existing SQLite file without creating or migrating it."""
    return sqlite3.connect(
        "file:" + quote(str(path.resolve()), safe="/") + "?mode=ro", uri=True
    )


def snapshot_database(source: Path, target: Path) -> None:
    """Include committed WAL contents in a consistent, disposable backup."""
    if target.exists():
        raise FileExistsError(target)
    with (
        closing(readonly(source)) as original,
        closing(sqlite3.connect(target)) as copy,
    ):
        original.backup(copy)


def read_source(args: argparse.Namespace) -> dict:
    """Serialize selected masks/settings from a migrated temporary copy."""
    with tempfile.TemporaryDirectory(prefix="cali-real-plate-") as directory:
        copy = Path(directory) / "source.cali"
        snapshot_database(args.database, copy)
        snapshot_sha256 = checksum(copy)
        engine = create_cali_engine(f"sqlite:///{copy}")
        try:
            with Session(engine) as session:
                experiment = session.get(Experiment, args.experiment_id)
                extraction = session.get(
                    ExtractionSettings, args.extraction_settings_id
                )
                detection = session.get(DetectionSettings, args.detection_settings_id)
                if experiment is None or extraction is None or detection is None:
                    raise ValueError(
                        "Selected experiment or settings ID does not exist."
                    )
                if experiment.plate is None:
                    raise ValueError("Selected experiment has no plate.")
                fovs = []
                for well in experiment.plate.wells:
                    for fov in well.fovs:
                        if fov.position_index not in args.positions:
                            continue
                        rois = []
                        for roi in sorted(fov.rois, key=lambda item: item.label_value):
                            if roi.detection_settings_id != detection.id:
                                continue
                            if roi.roi_mask is None:
                                raise ValueError(
                                    f"Missing mask: {fov.name}/{roi.label_value}"
                                )
                            rois.append(
                                {
                                    "label": roi.label_value,
                                    "mask": roi.roi_mask.model_dump(
                                        mode="json", exclude={"id"}
                                    ),
                                }
                            )
                        if not rois:
                            raise ValueError(f"No selected detection masks: {fov.name}")
                        fovs.append(
                            {
                                "name": fov.name,
                                "position": fov.position_index,
                                "fov_number": fov.fov_number,
                                "well": [well.name, well.row, well.column],
                                "rois": rois,
                            }
                        )
                if sorted(fov["position"] for fov in fovs) != sorted(args.positions):
                    raise ValueError(
                        "Selected positions must resolve uniquely in the experiment."
                    )
                return {
                    "snapshot_sha256": snapshot_sha256,
                    "experiment": experiment.model_dump(mode="json", exclude={"id"}),
                    "extraction": extraction.model_dump(
                        mode="json", exclude={"id", "created_at"}
                    ),
                    "detection": detection.model_dump(
                        mode="json", exclude={"id", "created_at"}
                    ),
                    "fovs": sorted(fovs, key=lambda item: item["position"]),
                }
        finally:
            engine.dispose()


def open_reader(args: argparse.Namespace, source: dict) -> object:
    """Use the same TIFF/Zarr dispatch as the production runner."""
    experiment = Experiment.model_validate(source["experiment"])
    tiff = experiment.tiff_collection_settings(args.dataset)
    reader = TiffCollectionReader(tiff) if tiff else load_data_from_path(args.dataset)
    if reader is None or reader.sequence is None:
        raise ValueError(
            "Recording must use a supported reader with sequence metadata."
        )
    return reader


def image_identity(data: np.ndarray, metadata: list[dict]) -> dict:
    """Bind complete pixel values and acquisition metadata to each position."""
    if data.ndim != 3:
        raise ValueError(f"Expected time/y/x images, received shape {data.shape}.")
    digest = hashlib.sha256()
    # Avoid another complete image buffer when readers return strided arrays.
    for frame in data:
        digest.update(np.ascontiguousarray(frame).tobytes())
    return {
        "shape": list(data.shape),
        "dtype": data.dtype.str,
        "image_bytes": data.nbytes,
        "pixels_sha256": digest.hexdigest(),
        "metadata_sha256": hashlib.sha256(
            json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }


def validate_position(
    data: np.ndarray,
    metadata: list[dict],
    fov: dict,
    settings: ExtractionSettings,
    model: object,
) -> dict:
    """Reject incompatible masks/timing before any extraction or inference."""
    identity = image_identity(data, metadata)
    labels = [roi["label"] for roi in fov["rois"]]
    if len(set(labels)) != len(labels) or any(label < 1 for label in labels):
        raise ValueError("ROI labels must be unique positive integers within a FOV.")
    occupied = set()
    for roi in fov["rois"]:
        mask = roi["mask"]
        y, x = np.asarray(mask["coords_y"]), np.asarray(mask["coords_x"])
        if (
            (mask["height"], mask["width"]) != data.shape[-2:]
            or not len(y)
            or len(y) != len(x)
            or y.dtype.kind not in "iu"
            or x.dtype.kind not in "iu"
            or np.any(y < 0)
            or np.any(x < 0)
            or np.any(y >= data.shape[-2])
            or np.any(x >= data.shape[-1])
        ):
            raise ValueError(
                f"Mask does not match recording: {fov['name']}/{roi['label']}"
            )
        pixels = set(zip(y.tolist(), x.tolist(), strict=True))
        if len(pixels) != len(y) or pixels & occupied:
            raise ValueError(f"Duplicate or overlapping mask pixels: {fov['name']}")
        occupied.update(pixels)
    timing = build_timing_descriptor(
        metadata,
        len(data),
        frame_rate=settings.frame_rate,
        frame_rate_verified=settings.frame_rate_verified,
    )
    window = resolve_initial_frame_window(
        discard_value=settings.discard_initial_value,
        discard_unit=settings.discard_initial_unit,
        frame_rate=settings.frame_rate,
        frame_rate_verified=settings.frame_rate_verified,
        timing=timing,
    )
    retained = preflight_retained_timing(
        timing,
        window,
        frame_rate=settings.frame_rate,
        minimum_frames={
            "CASCADE model": model.minimum_frames,
            "OASIS": OasisBackend.minimum_frames,
        },
    )
    validate_model_timing(
        retained,
        settings_frame_rate=settings.frame_rate,
        model_frame_rate=model.sampling_rate,
    )
    return {
        **identity,
        "position": fov["position"],
        "roi_count": len(labels),
        "source_start_frame": window.source_start_frame,
        "retained_frames": window.retained_frame_count,
        "timing_source": retained.source,
        "observed_rate_hz": retained.frame_rate_hz,
        "jitter_fraction": retained.interval_jitter_fraction,
    }


def preflight(args: argparse.Namespace, source: dict) -> dict:
    """Validate every selected recording, leaving source files unchanged."""
    model = load_cascade_model(args.model, args.model_dir)
    settings = ExtractionSettings.model_validate(source["extraction"])
    settings.validate_output_settings()
    reader = open_reader(args, source)
    try:
        positions = []
        for fov in source["fovs"]:
            data, metadata = reader.isel(p=fov["position"], metadata=True)
            try:
                positions.append(
                    validate_position(data, metadata, fov, settings, model)
                )
            except ValueError as error:
                raise ValueError(
                    f"{fov['name']} (position {fov['position']}): {error}"
                ) from error
            del data, metadata
    finally:
        reader.close()
    return {
        "model": model.name,
        "manifest": model.manifest_sha256,
        "model_rate_hz": model.sampling_rate,
        "positions": positions,
        "source_database_snapshot_sha256": source["snapshot_sha256"],
        "dataset": str(args.dataset.resolve()),
        "database": str(args.database.resolve()),
        "recording_description": args.recording_description,
        "selection_sha256": hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()
        ).hexdigest(),
        "independent_real_plate_acceptance": (
            "pending biological review and measured comparisons"
        ),
    }


class VerifiedReader(TensorstoreZarrReader):
    """Recheck provenance on the actual measured reader calls."""

    def __init__(self, reader: object, expected: dict) -> None:
        self.reader, self.expected = reader, expected
        self._path = reader.path
        self.calls = 0

    def isel(self, **kwargs: object) -> tuple[np.ndarray, list[dict]]:
        """Verify the pixels and timing actually consumed by each FOV worker."""
        data, metadata = self.reader.isel(**kwargs)
        expected = self.expected[str(kwargs["p"])]
        actual = image_identity(data, metadata)
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError("Recording changed after preflight.")
        self.calls += 1
        return data, metadata


class RealPlateRunner(BenchmarkRunner):
    """Reuse measured upstream/cached paths with the explicitly selected model."""

    @contextmanager
    def _cascade_context(self, settings: ExtractionSettings) -> Generator:
        if "cascade" not in settings.spike_methods or self._active_cascade_backend:
            with super()._cascade_context(settings) as backend:
                yield backend
            return
        options = {
            "model_name": self.args.model,
            "model_dir": self.args.model_dir,
            "expected_manifest": self.args.manifest,
            "device": "cpu",
        }
        if self.args.backend == "reference":
            backend = CascadeReferenceBackend(**options)
        elif self.args.backend == "service":
            backend = CascadeInferenceService(**options, max_windows=self.args.chunk)
        else:
            from benchmark_cascade_extraction import LockedPredictor

            backend = LockedPredictor(**options, max_windows=self.args.chunk)
        self.backend = backend
        import threading

        lock = threading.Lock()
        infer = backend.infer_all

        def locked(*args: object, **kwargs: object) -> object:
            with lock:
                return infer(*args, **kwargs)

        backend.infer_all = self.measurements.wrap(
            "cascade_caller", locked if self.args.backend == "cached-lock" else infer
        )
        try:
            backend.prepare(settings.frame_rate)
            yield backend
        finally:
            if self.args.backend != "reference":
                backend.close()


def make_experiment(source: dict, templates: list[dict]) -> Experiment:
    """Preserve real labels/well positions without importing previous results."""
    wells = {}
    for template in templates:
        name, row, column = template["well"]
        well = wells.setdefault(name, Well(name=name, row=row, column=column))
        well.fovs.append(
            FOV(
                name=template["name"],
                position_index=template["position"],
                fov_number=template["fov_number"],
                rois=[
                    ROI(label_value=roi["label"], roi_mask=Mask(**roi["mask"]))
                    for roi in template["rois"]
                ],
            )
        )
    experiment = Experiment.model_validate(source["experiment"])
    experiment.plate = Plate(name="real-plate benchmark", wells=list(wells.values()))
    return experiment


def products(fov: FOV, *, staged: bool = True) -> dict:
    """Include inactive ROIs and absent population products without guessing zeros."""
    parents = [
        (roi._new_data_analysis if staged else roi.data_analysis_history)[-1]
        for roi in sorted(fov.rois, key=lambda item: item.label_value)
    ]
    history = (
        getattr(fov, "_new_fov_analysis", []) if staged else fov.fov_analysis_history
    )
    parent = history[-1] if history else None
    return {
        "calcium": {
            "roi": [scientific_fields(item) for item in parents],
            "fov": scientific_fields(parent) if parent else None,
        },
        "spikes": {
            method: {
                "roi": [
                    scientific_fields(child)
                    if (child := item.get_spike_analysis(method))
                    else None
                    for item in parents
                ],
                "fov": scientific_fields(child)
                if parent and (child := parent.get_spike_analysis(method))
                else None,
            }
            for method in ("oasis", "cascade")
            if any(item.get_spike_analysis(method) is not None for item in parents)
        },
    }


def save_arrays(completed: list[FOV], path: Path) -> None:
    """Keep independently captured arrays keyed by real position and ROI label."""
    arrays = {}
    for fov in completed:
        for roi in fov.rois:
            trace = roi._new_traces[-1]
            prefix = f"{fov.position_index}/{roi.label_value}/"
            for name in ("raw_trace", "dff", "den_dff", "x_axis", "calcium_noise"):
                arrays[prefix + name] = np.asarray(getattr(trace, name))
            for spike in trace.spike_traces:
                arrays[prefix + spike.inference_run.method] = np.asarray(spike.values)
    np.savez_compressed(path, **arrays)


def audit_database(database: Path, arrays_path: Path, methods: tuple[str, ...]) -> dict:
    """Independently decode SQLite payloads and audit every saved array/owner."""
    with (
        closing(readonly(database)) as connection,
        np.load(arrays_path, allow_pickle=False) as arrays,
    ):
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Broken foreign keys in benchmark output.")
        seen = set()
        payload = dict.fromkeys(methods, 0)
        legacy_bytes = 0
        traces = connection.execute(
            "SELECT t.id, f.position_index, r.label_value, t.raw_trace, t.dff, "
            "t.den_dff, t.x_axis, t.calcium_noise, t.analysis_result_id "
            "FROM trace t JOIN roi r ON r.id=t.roi_id JOIN fov f ON f.id=r.fov_id"
        )
        for trace_id, position, label, *values, owner in traces:
            prefix = f"{position}/{label}/"
            for name, value in zip(
                ("raw_trace", "dff", "den_dff", "x_axis", "calcium_noise"),
                values,
                strict=True,
            ):
                np.testing.assert_array_equal(
                    json.loads(value) if isinstance(value, str) else value,
                    arrays[prefix + name],
                )
                seen.add(prefix + name)
            spikes = connection.execute(
                'SELECT s."values", i.method, i.extraction_result_id '
                "FROM spike_trace s JOIN spike_inference_run i "
                "ON i.id=s.spike_inference_run_id WHERE s.trace_id=?",
                (trace_id,),
            ).fetchall()
            if sorted(method for _, method, _ in spikes) != sorted(methods):
                raise ValueError("Missing or duplicate method products.")
            for encoded, method, inference_owner in spikes:
                if owner is None or inference_owner != owner:
                    raise ValueError("Incorrect canonical inference owner.")
                decoded = decode_trace_array(encoded)
                np.testing.assert_array_equal(decoded, arrays[prefix + method])
                seen.add(prefix + method)
                payload[method] += len(encoded)
                if method == "oasis":
                    legacy_bytes += len(json.dumps(decoded).encode())
        if seen != set(arrays.files):
            raise ValueError("Missing or unexpected persisted arrays.")
        if connection.execute(
            "SELECT s.id FROM spike_fov_analysis s "
            "JOIN fov_analysis f ON f.id=s.fov_analysis_id "
            "JOIN spike_inference_run i ON i.id=s.spike_inference_run_id "
            "WHERE s.method != i.method OR f.analysis_result_id IS NULL "
            "OR i.extraction_result_id IS NULL "
            "OR f.analysis_result_id != i.extraction_result_id"
        ).fetchall():
            raise ValueError("Incorrect FOV inference ownership.")
    return {
        "array_count": len(seen),
        "all_samples_equal": True,
        "canonical_payload_bytes": payload,
        "projected_legacy_oasis_json_bytes": legacy_bytes,
        "database_sha256": checksum(database),
        "database_bytes": database.stat().st_size,
    }


def run_case(args: argparse.Namespace, source: dict, checked: dict) -> dict:
    """Measure complete extraction/persistence and inference-free offline reuse."""
    begin = time.perf_counter()
    package = load_cascade_package()
    package_initialization_s = time.perf_counter() - begin
    measurements = Measurements(package.torch)
    runner = RealPlateRunner(args, measurements)
    methods = ("oasis", "cascade") if args.mode == "dual" else (args.mode,)
    settings = ExtractionSettings.model_validate(
        {
            **source["extraction"],
            "spike_methods": methods,
            "cascade_model": args.model if "cascade" in methods else None,
            "cascade_device": "cpu",
            "threads": args.workers,
        }
    )
    settings_data = settings.model_dump(exclude={"id", "created_at"})
    analysis_data = {
        "frame_rate": settings.frame_rate,
        "enable_calcium": True,
        "enable_spikes": True,
        "n_processes": args.analysis_processes,
        "spike_settings": [
            SpikeAnalysisSettings(
                method=method,
                ccg_n_shuffles=args.ccg_shuffles,
                enable_rising_edge_analysis=False,
            ).model_dump(mode="json")
            for method in methods
        ],
    }
    reader = VerifiedReader(
        open_reader(args, source),
        {str(position["position"]): position for position in checked["positions"]},
    )
    baseline = peak_mib()
    memory = ProcessMemorySampler()
    phases = []
    memory.start()
    try:
        preparation_start = time.perf_counter()
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    package.torch,
                    "load",
                    measurements.wrap("model_load", package.torch.load),
                )
            )
            stack.enter_context(
                patch.object(
                    OasisBackend,
                    "infer_all",
                    measurements.wrap("oasis_inference", OasisBackend.infer_all),
                )
            )
            stack.enter_context(
                runner.inference_session(
                    settings, AnalysisSettings.model_validate(analysis_data)
                )
            )
            preparation_s = time.perf_counter() - preparation_start
            for phase, templates in (
                ("cold", source["fovs"][:1]),
                ("warm", source["fovs"]),
            ):
                stem = f"{args.mode}-{args.backend}-{phase}"
                database = args.output_dir / (stem + ".cali")
                if database.exists():
                    raise FileExistsError(database)
                engine = create_cali_engine(f"sqlite:///{database}")
                create_database_and_tables(engine)
                settings = ExtractionSettings.model_validate(settings_data)
                analysis = AnalysisSettings.model_validate(analysis_data)
                experiment = make_experiment(source, templates)
                try:
                    with Session(engine, expire_on_commit=False) as session:
                        detection = DetectionSettings.model_validate(
                            source["detection"]
                        )
                        session.add_all([experiment, settings, analysis, detection])
                        session.flush()
                        for well in experiment.plate.wells:
                            for fov in well.fovs:
                                for roi in fov.rois:
                                    roi.detection_settings_id = detection.id
                        result = CaliResult(
                            experiment=experiment.id,
                            detection_settings_id=detection.id,
                            extraction_settings_id=settings.id,
                            analysis_settings_id=analysis.id,
                        )
                        session.add(result)
                        session.commit()
                        result_id = result.id
                        fovs = CaliRunner()._load_fovs_from_db(
                            session,
                            detection.id,
                            [fov["position"] for fov in templates],
                        )
                        _ = settings.model_dump(), analysis.model_dump(), experiment.id
                        _ = analysis.stimulation_mask
                        session.expunge_all()
                        memory.stage = f"{phase}/extraction_analysis"
                        before_loads = measurements.calls["model_load"]
                        begin = time.perf_counter()
                        completed = runner.run(
                            reader, settings, fovs, analysis_settings=analysis
                        )
                        extraction_s = time.perf_counter() - begin
                        if sorted(fov.position_index for fov in completed) != sorted(
                            fov["position"] for fov in templates
                        ):
                            raise ValueError("Incomplete benchmark FOV batch.")
                        expected_products = {
                            str(fov.position_index): products(fov) for fov in completed
                        }
                        memory.stage = f"{phase}/validation"
                        arrays_path = args.output_dir / (stem + ".npz")
                        save_arrays(completed, arrays_path)
                        (args.output_dir / (stem + "-metrics.json")).write_text(
                            json.dumps(expected_products, sort_keys=True, indent=2)
                            + "\n"
                        )
                        memory.stage = f"{phase}/persistence"
                        begin = time.perf_counter()
                        for fov in completed:
                            CaliRunner()._process_fov_results(
                                fov, session, result_id, include_traces=True
                            )
                            commit_fov_result(session, experiment, fov)
                        persistence_s = time.perf_counter() - begin
                        stored = list(
                            session.exec(select(FOV).order_by(FOV.position_index)).all()
                        )
                        for fov in stored:
                            if (
                                products(fov, staged=False)
                                != expected_products[str(fov.position_index)]
                            ):
                                raise ValueError(
                                    "Scientific products changed during persistence."
                                )
                            for roi in fov.rois:
                                for trace in roi.traces_history:
                                    _ = trace.extraction_frame_window
                                    for spike in trace.spike_traces:
                                        _ = spike.inference_run
                        session.expunge_all()
                        memory.stage = f"{phase}/offline_analysis"
                        before_calls = (
                            reader.calls,
                            measurements.calls["cascade_caller"],
                            measurements.calls["oasis_inference"],
                            measurements.calls["model_load"],
                        )
                        begin = time.perf_counter()
                        offline = AnalysisRunner().run(
                            stored, AnalysisSettings.model_validate(analysis_data)
                        )
                        offline_s = time.perf_counter() - begin
                        if before_calls != (
                            reader.calls,
                            measurements.calls["cascade_caller"],
                            measurements.calls["oasis_inference"],
                            measurements.calls["model_load"],
                        ):
                            raise ValueError(
                                "Offline analysis read images, loaded models or "
                                "inferred spikes."
                            )
                        if len(offline) != len(completed) or any(
                            products(fov) != expected_products[str(fov.position_index)]
                            for fov in offline
                        ):
                            raise ValueError("Offline scientific products differ.")
                    memory.stage = f"{phase}/audit_export"
                    audit = audit_database(database, arrays_path, methods)
                    export_noise_qc_to_csv(
                        engine,
                        args.output_dir / (stem + "-noise-qc.csv"),
                        run_id=result_id,
                    )
                    phases.append(
                        {
                            "phase": phase,
                            "fovs": len(templates),
                            "extraction_with_analysis_s": extraction_s,
                            "persistence_s": persistence_s,
                            "complete_s": extraction_s
                            + persistence_s
                            + (preparation_s if phase == "cold" else 0),
                            "offline_reanalysis_s": offline_s,
                            "offline_image_inference_model_calls": 0,
                            "model_loads": measurements.calls["model_load"]
                            - before_loads,
                            "audit": audit,
                            "spike_storage_projection": {
                                "target_fovs": 96,
                                "scope": (
                                    "canonical spike payloads plus projected "
                                    "legacy OASIS JSON"
                                ),
                                "mib": (
                                    sum(audit["canonical_payload_bytes"].values())
                                    + audit["projected_legacy_oasis_json_bytes"]
                                )
                                / len(templates)
                                * 96
                                / 1024**2,
                                "budget_mib": 512,
                                "note": (
                                    "Arithmetic projection; does not measure "
                                    "a complete independent plate."
                                ),
                            },
                        }
                    )
                finally:
                    engine.dispose()
    finally:
        sampled_memory = memory.close()
        reader.reader.close()
    return {
        "mode": args.mode,
        "backend": args.backend,
        "phases": phases,
        "baseline_peak_rss_mib": baseline,
        "peak_rss_mib": peak_mib(),
        "memory": sampled_memory,
        "peak_cascade_callers": measurements.peak_callers,
        "package_initialization_s": package_initialization_s,
        "extraction_settings": settings_data,
        "analysis_settings": analysis_data,
        "inputs": checked,
        "release_gate": "pending; this individual case does not establish acceptance",
        "runtime": {
            "packages": {
                name: version(name)
                for name in (
                    "cali",
                    "CascadeTorch",
                    "torch",
                    "numpy",
                    "scipy",
                    "numba",
                )
            },
            "cascade_revision": package.package_revision,
            "cascade_source_manifest_sha256": package.source_manifest_sha256,
            "runtime_source_sha256": {
                cls.__name__: checksum(Path(inspect.getfile(cls)))
                for cls in (
                    CaliRunner,
                    AnalysisRunner,
                    CascadeReferenceBackend,
                    CascadeInferenceService,
                )
            },
        },
    }


def compare(directory: Path, cases: tuple = CASES) -> dict:
    """Compare every position/label across methods and backend implementations."""
    comparisons = {}
    for phase in ("cold", "warm"):
        reference = {
            method: json.loads(
                (directory / f"{method}-reference-{phase}-metrics.json").read_text()
            )
            for method in ("oasis", "cascade")
        }
        for mode, backend in cases:
            stem = f"{mode}-{backend}-{phase}"
            actual = json.loads((directory / (stem + "-metrics.json")).read_text())
            if set(actual) != set(reference["oasis"]) or set(actual) != set(
                reference["cascade"]
            ):
                raise ValueError(f"Position mismatch: {stem}")
            for position, product in actual.items():
                if product["calcium"] != reference["oasis"][position]["calcium"]:
                    raise ValueError(f"Calcium mismatch: {stem}/{position}")
                for method, value in product["spikes"].items():
                    if value != reference[method][position]["spikes"][method]:
                        raise ValueError(
                            f"Spike products mismatch: {stem}/{position}/{method}"
                        )
            with np.load(directory / (stem + ".npz"), allow_pickle=False) as arrays:
                expected_names = set()
                for method in ("oasis", "cascade"):
                    with np.load(
                        directory / f"{method}-reference-{phase}.npz",
                        allow_pickle=False,
                    ) as oracle:
                        expected_names.update(
                            name
                            for name in oracle.files
                            if name.rsplit("/", 1)[-1] not in {"oasis", "cascade"}
                            or (
                                name.rsplit("/", 1)[-1] == method
                                and (mode == "dual" or mode == method)
                            )
                        )
                        for name in arrays.files:
                            if name.rsplit("/", 1)[-1] == method or (
                                method == "oasis"
                                and name.rsplit("/", 1)[-1] not in {"oasis", "cascade"}
                            ):
                                np.testing.assert_array_equal(
                                    arrays[name], oracle[name]
                                )
                if set(arrays.files) != expected_names:
                    raise ValueError(f"Missing or unexpected array products: {stem}")
            comparisons[stem] = True
    return comparisons


def main() -> None:
    """Use a fresh process per case; never promote a gate from a smoke run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--detection-settings-id", type=int, required=True)
    parser.add_argument("--extraction-settings-id", type=int, required=True)
    parser.add_argument("--positions", type=int, nargs="+", required=True)
    parser.add_argument("--recording-description", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--analysis-processes", type=int, default=1)
    parser.add_argument("--ccg-shuffles", type=int, default=20)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--mode", choices=("all", "oasis", "cascade", "dual"), default="all"
    )
    parser.add_argument(
        "--backend",
        choices=("reference", "service", "cached-lock"),
        default="reference",
    )
    args = parser.parse_args()
    if min(
        args.workers, args.analysis_processes, args.ccg_shuffles, args.chunk
    ) < 1 or len(set(args.positions)) != len(args.positions):
        parser.error("Counts must be positive and positions must not repeat.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        source = read_source(args)
        checked = preflight(args, source)
    except (ValueError, OSError, sqlite3.Error) as error:
        (args.output_dir / "preflight-rejected.json").write_text(
            json.dumps(
                {
                    "status": "rejected",
                    "reason": str(error),
                    "dataset": str(args.dataset.resolve()),
                    "database": str(args.database.resolve()),
                    "release_gate": "pending; no benchmark was run",
                },
                indent=2,
            )
            + "\n"
        )
        raise
    (args.output_dir / "preflight.json").write_text(
        json.dumps(checked, indent=2) + "\n"
    )
    if args.preflight_only:
        print("Preflight passed; independent real-plate acceptance remains pending.")
        return
    args.manifest = checked["manifest"]
    if args.mode != "all":
        report = run_case(args, source, checked)
        (args.output_dir / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return
    reports = []
    original = sys.argv[1:]
    for mode, backend in CASES:
        case_dir = args.output_dir / f"{mode}-{backend}"
        # Required paths/mode are overridden by argparse's final occurrence.
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *original,
            "--mode",
            mode,
            "--backend",
            backend,
            "--output-dir",
            str(case_dir),
        ]
        with (args.output_dir / f"{mode}-{backend}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if json.loads((case_dir / "preflight.json").read_text()) != checked:
            raise ValueError("Source inputs changed between benchmark cases.")
        reports.append(json.loads((case_dir / "report.json").read_text()))
        for path in case_dir.iterdir():
            if path.name not in {"preflight.json", "report.json"}:
                path.rename(args.output_dir / path.name)
        print(f"Completed {mode}/{backend}", flush=True)
    parity = compare(args.output_dir)
    (args.output_dir / "report.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "script_sha256": checksum(Path(__file__)),
                "shared_benchmark_sha256": checksum(
                    Path(__file__).with_name("benchmark_cascade_extraction.py")
                ),
                "inputs": checked,
                "results": reports,
                "exact_parity": parity,
                "excluded_stochastic_fields": sorted(STOCHASTIC_FIELDS),
                "release_gate": (
                    "pending independent recording review, budget evaluation, "
                    "GPU and CI acceptance"
                ),
                "runtime": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "workers": args.workers,
                    "analysis_processes": args.analysis_processes,
                    "chunk": args.chunk,
                    "ccg_shuffles": args.ccg_shuffles,
                },
                "memory_note": (
                    "Sampled summed RSS can double-count shared pages and miss peaks; "
                    "not unique physical allocation."
                ),
                "provenance_hashing_note": (
                    "Complete timings include checking image/metadata hashes "
                    "on measured reader calls."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_RUNNING", "1")
    main()
