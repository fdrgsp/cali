"""Measure image extraction, method-bound analysis and persistence in fresh processes.

Run with cali[cascade], OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and
NUMBA_NUM_THREADS=1. This controlled
image workload exercises the actual FOV pool and database path, starting from
known masks. Select --analysis full to include spike analysis and offline reuse. It
does not substitute for the required independently recorded real-plate gate.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
from sqlalchemy import func
from sqlmodel import Session, select

from cali._cascade_package import load_cascade_package
from cali.analysis import (
    AnalysisRunner,
    _fov_analysis,
    _fov_analysis_parallel,
    _roi_analysis,
)
from cali.extraction._extraction_runner import ExtractionRunner
from cali.extraction._spike_inference import OasisBackend
from cali.extraction._spike_inference._cascade_cached import CachedCascadePredictor
from cali.extraction._spike_inference._cascade_reference import CascadeReferenceBackend
from cali.extraction._spike_inference._cascade_service import CascadeInferenceService
from cali.readers import TensorstoreZarrReader
from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    Experiment,
    ExtractionSettings,
    FOVAnalysis,
    Mask,
    Plate,
    SpikeAnalysis,
    SpikeAnalysisSettings,
    SpikeFOVAnalysis,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    Well,
    create_cali_engine,
    create_database_and_tables,
)
from cali.util import commit_fov_result

if TYPE_CHECKING:
    from collections.abc import Generator

MODEL = "Global_EXC_30Hz_smoothing25ms"
MANIFEST = "ac8954174ba0a01a2d929a7e8b3fc7e3a4365d5c01f5e262d2b597822fcae184"
CASES = (
    ("oasis", "reference"),
    ("cascade", "reference"),
    ("dual", "reference"),
    ("cascade", "service"),
    ("dual", "service"),
    ("cascade", "cached-lock"),
    ("dual", "cached-lock"),
)

# Identity differs between runs; shuffled CCG significance uses random surrogates.
# All deterministic scientific products, including raw CCG/lag/jitter, are checked.
IDENTITY_FIELDS = {
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
STOCHASTIC_FIELDS = {
    "spike_ccg_zscore_matrix",
    "fraction_significant_ccg_pairs",
    "spike_ccg_zscore_matrix_rising_edges",
    "fraction_significant_ccg_pairs_rising_edges",
}

# Only continuous ROI quantities derived from GPU predictions receive tolerance.
# Activity, excursions, labels, lag choices, burst counts/bounds, coordinates,
# model noise and applied thresholds must agree exactly.
GPU_CONTINUOUS_FIELDS = {
    "expected_spike_count",
    "expected_spike_rate_hz",
}
GPU_RTOL = 1e-5
GPU_ATOL = 1e-6


def compare_arrays(
    actual: np.ndarray, expected: np.ndarray, *, tolerant: bool
) -> float:
    """Reject broadcasting/nonfinite predictions before testing device parity."""
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    if tolerant:
        np.testing.assert_allclose(actual, expected, rtol=GPU_RTOL, atol=GPU_ATOL)
    else:
        np.testing.assert_array_equal(actual, expected)
    return float(np.max(np.abs(actual - expected))) if actual.size else 0.0


def compare_spike_products(actual: dict, expected: dict, *, tolerant: bool) -> dict:
    """Require exact discrete science even when continuous GPU values differ."""
    assert actual.keys() == expected.keys() == {"roi", "fov"}
    assert len(actual["roi"]) == len(expected["roi"])
    differences = {}
    pairs = [("fov", actual["fov"], expected["fov"])] + [
        (f"roi/{index}", row, oracle)
        for index, (row, oracle) in enumerate(
            zip(actual["roi"], expected["roi"], strict=True)
        )
    ]
    for owner, row, oracle in pairs:
        assert row.keys() == oracle.keys()
        for name, value in row.items():
            baseline = oracle[name]
            if tolerant and name in GPU_CONTINUOUS_FIELDS and value is not None:
                assert baseline is not None
                error = compare_arrays(
                    np.asarray(value), np.asarray(baseline), tolerant=True
                )
                if error:
                    differences[f"{owner}/{name}"] = error
            else:
                assert value == baseline, (
                    f"Discrete/exact metric changed: {owner}/{name}"
                )
    return differences


def scientific_fields(model: object) -> dict:
    """Compare scientific scalar fields independently of ORM identity and RNG."""
    return {
        name: getattr(model, name)
        for name in type(model).model_fields
        if name not in IDENTITY_FIELDS | STOCHASTIC_FIELDS
    }


def scientific_products(fov: FOV, *, staged: bool = True) -> dict:
    """Read every ROI and FOV pillar in stable label order."""
    rois = sorted(fov.rois, key=lambda roi: roi.label_value)
    analyses = [
        roi._new_data_analysis[-1] if staged else roi.data_analysis_history[-1]
        for roi in rois
    ]
    fov_analysis = fov._new_fov_analysis[-1] if staged else fov.fov_analysis_history[-1]
    return {
        "calcium": {
            "roi_labels": [roi.label_value for roi in rois],
            "roi": [scientific_fields(parent) for parent in analyses],
            "fov": scientific_fields(fov_analysis),
        },
        "spikes": {
            child.method: {
                "roi": [
                    scientific_fields(parent.get_spike_analysis(child.method))
                    for parent in analyses
                ],
                "fov": scientific_fields(child),
            }
            for child in fov_analysis.spike_analyses
        },
    }


def fingerprint(value: object) -> str:
    """Identify normalized products without retaining duplicate matrices."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def peak_mib() -> float:
    """Return process lifetime peak RSS with platform-correct units."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2 if sys.platform == "darwin" else 1024)


class ProcessMemorySampler:
    """Sample simultaneous RSS for the benchmark process and its descendants."""

    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.stage = "preparation"
        self.stages: dict[str, dict] = {}
        self.error: Exception | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)

    def sample(self) -> None:
        """Exclude the short-lived ps sampler from the descendant inventory."""
        stage = self.stage
        with subprocess.Popen(
            ["ps", "-axo", "pid=,ppid=,rss="],
            stdout=subprocess.PIPE,
            text=True,
        ) as process:
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
            if process.returncode:
                raise RuntimeError("Process-tree memory sampling failed.")
            sampler_pid = process.pid
        rows = [tuple(map(int, line.split())) for line in output.splitlines()]
        parent_pid = os.getpid()
        included = {parent_pid}
        while children := {
            pid
            for pid, owner, _ in rows
            if owner in included and pid not in included and pid != sampler_pid
        }:
            included.update(children)
        parent = sum(rss for pid, _, rss in rows if pid == parent_pid) / 1024
        descendants = {
            pid: rss / 1024
            for pid, _, rss in rows
            if pid in included and pid != parent_pid
        }
        combined = parent + sum(descendants.values())
        values = self.stages.setdefault(
            stage,
            {
                "samples": 0,
                "parent_peak_mib": 0.0,
                "descendants_peak_mib": 0.0,
                "combined_peak_mib": 0.0,
                "max_descendant_processes": 0,
                "combined_peak_snapshot": {},
            },
        )
        values["samples"] += 1
        values["parent_peak_mib"] = max(values["parent_peak_mib"], parent)
        values["descendants_peak_mib"] = max(
            values["descendants_peak_mib"], sum(descendants.values())
        )
        values["max_descendant_processes"] = max(
            values["max_descendant_processes"], len(descendants)
        )
        if combined > values["combined_peak_mib"]:
            values["combined_peak_mib"] = combined
            values["combined_peak_snapshot"] = {
                "parent_mib": parent,
                "descendants_mib": list(descendants.values()),
            }

    def _monitor(self) -> None:
        try:
            while not self.stop.wait(self.interval_s):
                self.sample()
        except Exception as error:
            self.error = error

    def start(self) -> None:
        """Fail immediately if the system ps command is unavailable."""
        self.sample()
        self.thread.start()

    def close(self) -> dict:
        """Return sampled peaks; never sum independent process high-water marks."""
        self.stop.set()
        self.thread.join()
        if self.error is not None:
            raise self.error
        self.sample()
        return {
            "interval_s": self.interval_s,
            "scope": (
                "simultaneous parent + descendant RSS, including resource trackers"
            ),
            "note": (
                "Sampling can miss short peaks; shared pages count in each process. "
                "This is a summed-RSS comparison, not unique physical memory. "
                "The ps sampler is excluded. Stage transitions are approximate."
            ),
            "samples": sum(value["samples"] for value in self.stages.values()),
            "combined_peak_mib": max(
                value["combined_peak_mib"] for value in self.stages.values()
            ),
            "max_descendant_processes": max(
                value["max_descendant_processes"] for value in self.stages.values()
            ),
            "stages": self.stages,
        }


class ControlledReader(TensorstoreZarrReader):
    """Deterministic synthetic images; each caller retains its own image buffer."""

    def __init__(self, rois: int, frames: int) -> None:
        self.rois, self.frames = rois, frames
        self.side = math.ceil(math.sqrt(rois)) * 4
        self._path = Path("controlled-images.zarr")

    def isel(self, **_: object) -> tuple[np.ndarray, list[dict]]:
        """Generate pixels and uniform acquisition timestamps."""
        rng = np.random.default_rng(9183)
        image = rng.normal(1000, 2, (self.frames, self.side, self.side))
        for index in range(self.rois):
            row, col = divmod(index, self.side // 4)
            signal = rng.normal(0, 120 + 280 * index / self.rois, self.frames)
            for onset in range(75, self.frames, 200):
                signal[onset:] += 350 * np.exp(-np.arange(self.frames - onset) / 12)
            image[:, row * 4 : row * 4 + 2, col * 4 : col * 4 + 2] += signal[
                :, None, None
            ]
        return image, [
            {"runner_time_ms": 500 + frame * 1000 / 30} for frame in range(self.frames)
        ]


def make_fov(position: int, rois: int) -> FOV:
    """Create nonoverlapping known masks without running detection."""
    side = math.ceil(math.sqrt(rois)) * 4
    cells = []
    for index in range(rois):
        row, col = divmod(index, side // 4)
        cells.append(
            ROI(
                label_value=index + 1,
                roi_mask=Mask(
                    coords_y=[row * 4, row * 4, row * 4 + 1, row * 4 + 1],
                    coords_x=[col * 4, col * 4 + 1, col * 4, col * 4 + 1],
                    height=side,
                    width=side,
                    mask_type="roi",
                ),
            )
        )
    return FOV(name=f"A1_{position:04d}", position_index=position, rois=cells)


class Measurements:
    """Thread-safe aggregate durations; overlapping caller times are not wall time."""

    def __init__(self, torch: object) -> None:
        self.torch = torch
        self.inference_thread_counts: set[int] = set()
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
        self.active = self.peak_callers = self.peak_queue = self.model_loads = 0
        self.loader_threads: set[int] = set()

    def wrap(self, name: str, operation: object) -> object:
        """Instrument one operation without changing its output or exceptions."""

        def measured(*args: object, **kwargs: object) -> object:
            begin = time.perf_counter()
            if name == "cascade_compute":
                with self.lock:
                    self.inference_thread_counts.add(self.torch.get_num_threads())
            if name == "cascade_caller":
                with self.lock:
                    self.active += 1
                    self.peak_callers = max(self.peak_callers, self.active)
            try:
                return operation(*args, **kwargs)
            finally:
                with self.lock:
                    self.seconds[name] += time.perf_counter() - begin
                    self.calls[name] += 1
                    if name == "cascade_caller":
                        self.active -= 1

        return measured

    def wrap_method(self, name: str, operation: object) -> object:
        """Time the actual selected method without pooling unlike populations."""

        def measured(product: object, *args: object, **kwargs: object) -> object:
            method = product.inference_run.method if name == "roi" else product.method
            return self.wrap(f"{name}_spike_analysis_{method}", operation)(
                product, *args, **kwargs
            )

        return measured


class LockedPredictor(CachedCascadePredictor):
    """Benchmark-only cross-thread ownership, guarded by one external caller lock."""

    def _claim_owner(self) -> None:
        pass


class BenchmarkRunner(ExtractionRunner):
    """Choose the experimental lock baseline without altering application dispatch."""

    def __init__(self, args: argparse.Namespace, measurements: Measurements) -> None:
        super().__init__()
        self.args, self.measurements = args, measurements
        self.backend = None

    @contextmanager
    def _cascade_context(self, settings: ExtractionSettings) -> Generator:
        if "cascade" not in settings.spike_methods or self._active_cascade_backend:
            with super()._cascade_context(settings) as backend:
                yield backend
            return
        options = {
            "model_name": MODEL,
            "model_dir": self.args.model_dir,
            "expected_manifest": MANIFEST,
            "device": self.args.device,
        }
        if self.args.backend == "reference":
            backend = CascadeReferenceBackend(**options)
        elif self.args.backend == "service":
            backend = CascadeInferenceService(**options, max_windows=self.args.chunk)
        else:
            backend = LockedPredictor(**options, max_windows=self.args.chunk)
        self.backend = backend
        caller = backend.infer_all
        caller_lock = threading.Lock()

        def locked(*args: object, **kwargs: object) -> object:
            with caller_lock:
                return caller(*args, **kwargs)

        backend.infer_all = self.measurements.wrap(
            "cascade_caller", locked if self.args.backend == "cached-lock" else caller
        )
        try:
            backend.prepare(settings.frame_rate)
            yield backend
        finally:
            if self.args.backend != "reference":
                backend.close()


def run_case(args: argparse.Namespace) -> dict:
    """Run one mode in an isolated process with cold then warm backend reuse."""
    begin = time.perf_counter()
    package = load_cascade_package()
    package_initialization_s = time.perf_counter() - begin
    baseline = peak_mib()
    measurements = Measurements(package.torch)
    original_load = package.torch.load

    def load(*args: object, **kwargs: object) -> object:
        with measurements.lock:
            measurements.model_loads += 1
            measurements.loader_threads.add(threading.get_ident())
        return original_load(*args, **kwargs)

    methods = ("oasis", "cascade") if args.mode == "dual" else (args.mode,)
    settings = ExtractionSettings(
        spike_methods=methods,
        cascade_model=MODEL if "cascade" in methods else None,
        cascade_device=args.device,
        frame_rate=30,
        dff_window=5,
        decay_constant=1.0,
        neuropil_inner_radius=0,
        threads=args.workers,
    )
    analysis = AnalysisSettings(
        frame_rate=30,
        enable_calcium=True,
        enable_spikes=args.analysis == "full",
        n_processes=args.analysis_processes,
        peaks_height_value=0.1,
        peaks_prominence_multiplier=0.5,
        spike_settings=[
            SpikeAnalysisSettings(
                method=method,
                ccg_n_shuffles=args.ccg_shuffles,
                enable_rising_edge_analysis=args.rising_edges,
            )
            for method in methods
        ],
    )
    settings_data = settings.model_dump(exclude={"id"})
    analysis_data = analysis.model_dump(mode="json", exclude={"id", "created_at"})
    reader = ControlledReader(args.rois, args.frames)
    runner = BenchmarkRunner(args, measurements)
    stop = threading.Event()
    memory = ProcessMemorySampler() if args.sample_memory else None
    if memory is not None:
        memory.start()

    def memory_stage(stage: str) -> None:
        if memory is not None:
            memory.stage = stage

    def monitor() -> None:
        while not stop.wait(0.005):
            if isinstance(runner.backend, CascadeInferenceService):
                measurements.peak_queue = max(
                    measurements.peak_queue, runner.backend.pending_count
                )

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    phases = []
    patches = [
        patch.object(package.torch, "load", load),
        patch.object(
            OasisBackend,
            "infer_all",
            measurements.wrap("oasis_inference", OasisBackend.infer_all),
        ),
        patch.object(
            CascadeReferenceBackend,
            "_predict",
            measurements.wrap("cascade_compute", CascadeReferenceBackend._predict),
        ),
        patch.object(
            CachedCascadePredictor,
            "_predict",
            measurements.wrap("cascade_compute", CachedCascadePredictor._predict),
        ),
        patch.object(
            runner,
            "_compute_roi_dff",
            measurements.wrap("image_roi_extraction", runner._compute_roi_dff),
        ),
        patch.object(
            runner,
            "_finalize_roi",
            measurements.wrap("trace_finalization", runner._finalize_roi),
        ),
        patch.object(
            _fov_analysis_parallel,
            "compute_fov_analysis_parallel",
            measurements.wrap(
                "fov_analysis_total",
                _fov_analysis_parallel.compute_fov_analysis_parallel,
            ),
        ),
        patch.object(
            _roi_analysis,
            "analyze_roi_calcium",
            measurements.wrap(
                "roi_calcium_analysis", _roi_analysis.analyze_roi_calcium
            ),
        ),
        patch.object(
            _roi_analysis,
            "analyze_spike_trace",
            measurements.wrap_method("roi", _roi_analysis.analyze_spike_trace),
        ),
        patch.object(
            _fov_analysis,
            "_compute_calcium_population",
            measurements.wrap(
                "fov_calcium_analysis", _fov_analysis._compute_calcium_population
            ),
        ),
        patch.object(
            _fov_analysis_parallel,
            "compute_spike_population",
            measurements.wrap_method(
                "fov", _fov_analysis_parallel.compute_spike_population
            ),
        ),
    ]
    from contextlib import ExitStack

    try:
        with ExitStack() as stack:
            for instrumentation in patches:
                stack.enter_context(instrumentation)
            begin = time.perf_counter()
            stack.enter_context(runner.inference_session(settings, analysis))
            preparation = time.perf_counter() - begin
            preparation_loads = measurements.model_loads
            for phase, count in (("cold", 1), ("warm", args.fovs)):
                database = args.output_dir / f"{args.mode}-{args.backend}-{phase}.cali"
                if database.exists():
                    raise FileExistsError(database)
                engine = create_cali_engine(f"sqlite:///{database}")
                create_database_and_tables(engine)
                fovs = [make_fov(index, args.rois) for index in range(count)]
                experiment = Experiment(
                    name="controlled benchmark",
                    plate=Plate(
                        name="controlled",
                        wells=[Well(name="A1", row=0, column=0, fovs=fovs)],
                    ),
                )
                before = dict(measurements.seconds)
                before_calls = dict(measurements.calls)
                before_loads = measurements.model_loads
                with Session(engine) as session:
                    session.add_all([experiment, settings, analysis])
                    session.flush()
                    result = CaliResult(
                        experiment=experiment.id,
                        extraction_settings_id=settings.id,
                        analysis_settings_id=analysis.id,
                    )
                    session.add(result)
                    session.commit()
                    # Eager-load all masks before entering the FOV thread pool.
                    for fov in fovs:
                        for roi in fov.rois:
                            _ = roi.roi_mask.coords_y, roi.roi_mask.coords_x
                            _ = roi.traces_history, roi.data_analysis_history
                    settings.validate_output_settings()
                    analysis.validate_spike_settings()
                    _ = analysis.model_dump(), experiment.id, result.id
                    # Workers receive detached FOVs in the public runner. Keeping
                    # them attached here lets one commit expire another FOV's
                    # ROI collection and lose its private staged products.
                    session.expunge_all()
                    memory_stage(f"{phase}/extraction_analysis")
                    extract_begin = time.perf_counter()
                    completed = runner.run(
                        reader, settings, fovs, analysis_settings=analysis
                    )
                    extraction_s = time.perf_counter() - extract_begin
                    memory_stage(f"{phase}/extraction_validation")
                    assert len(completed) == count
                    products = scientific_products(completed[0])
                    encoded_products = json.dumps(products, sort_keys=True)
                    for fov in completed:
                        assert (
                            json.dumps(scientific_products(fov), sort_keys=True)
                            == encoded_products
                        )
                    (
                        args.output_dir
                        / f"{args.mode}-{args.backend}-{phase}-metrics.json"
                    ).write_text(json.dumps(products, indent=2) + "\n")
                    stage_seconds = {
                        name: seconds - before.get(name, 0)
                        for name, seconds in measurements.seconds.items()
                    }
                    stage_calls = {
                        name: calls - before_calls.get(name, 0)
                        for name, calls in measurements.calls.items()
                    }
                    assert stage_calls["roi_calcium_analysis"] == count * args.rois
                    assert stage_calls["fov_calcium_analysis"] == count
                    for method in methods if args.analysis == "full" else ():
                        assert (
                            stage_calls[f"roi_spike_analysis_{method}"]
                            == count * args.rois
                        )
                        assert stage_calls[f"fov_spike_analysis_{method}"] == count
                    arrays = {
                        "den_dff": np.array(
                            [roi._new_traces[0].den_dff for roi in completed[0].rois]
                        ),
                        "calcium_noise": np.array(
                            [
                                roi._new_traces[0].calcium_noise
                                for roi in completed[0].rois
                            ]
                        ),
                    }
                    for method in methods:
                        arrays[method] = np.array(
                            [
                                roi._new_traces[0].get_spike_values(method)
                                for roi in completed[0].rois
                            ]
                        )
                    # Every position uses identical images. Check all pool outputs,
                    # including FOVs that finish after the first yielded result.
                    for fov in completed[1:]:
                        np.testing.assert_array_equal(
                            [roi._new_traces[0].den_dff for roi in fov.rois],
                            arrays["den_dff"],
                        )
                        np.testing.assert_array_equal(
                            [roi._new_traces[0].calcium_noise for roi in fov.rois],
                            arrays["calcium_noise"],
                        )
                        for method in methods:
                            np.testing.assert_array_equal(
                                [
                                    roi._new_traces[0].get_spike_values(method)
                                    for roi in fov.rois
                                ],
                                arrays[method],
                            )
                    np.savez_compressed(
                        args.output_dir / f"{args.mode}-{args.backend}-{phase}.npz",
                        **arrays,
                    )
                    memory_stage(f"{phase}/persistence")
                    persistence_begin = time.perf_counter()
                    for fov in completed:
                        CaliRunner()._process_fov_results(
                            fov, session, result.id, include_traces=True
                        )
                        commit_fov_result(session, experiment, fov)
                    persistence_s = time.perf_counter() - persistence_begin
                    memory_stage(f"{phase}/database_validation")
                    # Confirm stored products survived normal staging/commits.
                    stored_fovs = list(
                        session.exec(select(FOV).order_by(FOV.position_index)).all()
                    )
                    expected_rois = count * args.rois
                    rows = {
                        model.__tablename__: session.exec(
                            select(func.count()).select_from(model)
                        ).one()
                        for model in (
                            FOV,
                            ROI,
                            Traces,
                            DataAnalysis,
                            FOVAnalysis,
                            SpikeTrace,
                            SpikeAnalysis,
                            SpikeFOVAnalysis,
                            SpikeInferenceRun,
                        )
                    }
                    assert rows == {
                        "fov": count,
                        "roi": expected_rois,
                        "trace": expected_rois,
                        "data_analysis": expected_rois,
                        "fov_analysis": count,
                        "spike_trace": expected_rois * len(methods),
                        "spike_analysis": (
                            expected_rois * len(methods)
                            if args.analysis == "full"
                            else 0
                        ),
                        "spike_fov_analysis": (
                            count * len(methods) if args.analysis == "full" else 0
                        ),
                        "spike_inference_run": len(methods),
                    }
                    canonical_runs = {
                        run.method: run.id
                        for run in session.exec(select(SpikeInferenceRun)).all()
                    }
                    assert set(canonical_runs) == set(methods)
                    for inference in session.exec(select(SpikeInferenceRun)).all():
                        if inference.method == "cascade":
                            assert inference.resolved_device == args.device
                    assert len(stored_fovs) == count
                    assert (
                        sum(len(fov.rois) for fov in stored_fovs) == count * args.rois
                    )
                    for fov in stored_fovs:
                        assert (
                            json.dumps(
                                scientific_products(fov, staged=False), sort_keys=True
                            )
                            == encoded_products
                        )
                        for roi in fov.rois:
                            trace = roi.traces_history[-1]
                            _ = trace.x_axis, trace.den_dff, trace.calcium_noise
                            index = roi.label_value - 1
                            np.testing.assert_array_equal(
                                trace.den_dff, arrays["den_dff"][index]
                            )
                            np.testing.assert_array_equal(
                                trace.calcium_noise, arrays["calcium_noise"][index]
                            )
                            _ = trace.extraction_frame_window.acquisition_frame_rate_hz
                            for spike in trace.spike_traces:
                                _ = spike.values, spike.inference_run.semantic_key()
                                method = spike.inference_run.method
                                assert (
                                    spike.spike_inference_run_id
                                    == canonical_runs[method]
                                )
                                np.testing.assert_array_equal(
                                    spike.values, arrays[method][index]
                                )
                    # Detach fully loaded ORM inputs just as the public runner does.
                    session.expunge_all()
                    offline_settings = AnalysisSettings.model_validate(analysis_data)
                    offline_before = dict(measurements.seconds)
                    offline_before_calls = dict(measurements.calls)
                    offline_loads = measurements.model_loads
                    memory_stage(f"{phase}/offline_analysis")
                    offline_begin = time.perf_counter()
                    reanalyzed = AnalysisRunner().run(stored_fovs, offline_settings)
                    offline_s = time.perf_counter() - offline_begin
                    memory_stage(f"{phase}/offline_validation")
                    assert len(reanalyzed) == count
                    assert measurements.model_loads == offline_loads
                    for fov in reanalyzed:
                        assert (
                            json.dumps(scientific_products(fov), sort_keys=True)
                            == encoded_products
                        )
                    offline_seconds = {
                        name: seconds - offline_before.get(name, 0)
                        for name, seconds in measurements.seconds.items()
                    }
                    offline_calls = {
                        name: calls - offline_before_calls.get(name, 0)
                        for name, calls in measurements.calls.items()
                    }
                    assert not offline_calls.get("oasis_inference", 0)
                    assert not offline_calls.get("cascade_caller", 0)
                    assert offline_calls["roi_calcium_analysis"] == count * args.rois
                    assert offline_calls["fov_calcium_analysis"] == count
                    for method in methods if args.analysis == "full" else ():
                        assert (
                            offline_calls[f"roi_spike_analysis_{method}"]
                            == count * args.rois
                        )
                        assert offline_calls[f"fov_spike_analysis_{method}"] == count
                engine.dispose()
                phases.append(
                    {
                        "phase": phase,
                        "fovs": count,
                        "preparation_s": preparation if phase == "cold" else 0,
                        "preparation_model_loads": preparation_loads
                        if phase == "cold"
                        else 0,
                        "extraction_with_analysis_s": extraction_s,
                        "persistence_s": persistence_s,
                        "complete_s": extraction_s
                        + persistence_s
                        + (preparation if phase == "cold" else 0),
                        "stage_aggregate_s": stage_seconds,
                        "stage_calls": stage_calls,
                        "offline_reanalysis_s": offline_s,
                        "offline_stage_aggregate_s": offline_seconds,
                        "offline_stage_calls": offline_calls,
                        "calcium_products_sha256": fingerprint(products["calcium"]),
                        "method_products_sha256": {
                            method: fingerprint(product)
                            for method, product in products["spikes"].items()
                        },
                        "populations": {
                            method: {
                                "active_rois": len(
                                    product["fov"]["active_roi_labels"] or []
                                ),
                                "pairs": (
                                    n := len(product["fov"]["active_roi_labels"] or [])
                                )
                                * (n - 1)
                                // 2,
                                "valid_start": product["fov"]["valid_start"],
                                "valid_stop": product["fov"]["valid_stop"],
                            }
                            for method, product in products["spikes"].items()
                        },
                        "model_loads": measurements.model_loads - before_loads,
                        "database_bytes": database.stat().st_size,
                        "persistence_audit": {
                            "rows": rows,
                            "all_stored_arrays_equal_extraction": True,
                            "canonical_inference_ids": canonical_runs,
                        },
                    }
                )
                # Fresh settings objects avoid cross-database ORM identity reuse.
                settings = ExtractionSettings.model_validate(settings_data)
                analysis = AnalysisSettings.model_validate(analysis_data)
                memory_stage("between_phases")
            stats = (
                None
                if args.backend == "reference" or runner.backend is None
                else vars(runner.backend.stats)
            )
    finally:
        stop.set()
        monitor_thread.join()
        memory_report = None if memory is None else memory.close()
    return {
        "mode": args.mode,
        "backend": args.backend,
        "package_initialization_s": package_initialization_s,
        "scope": (
            "controlled images, known ROI masks, actual FOV pool, "
            f"{args.analysis} analysis, SQLite persistence and offline re-analysis"
        ),
        "excluded": [
            "detection",
            *(["spike analysis"] if args.analysis == "calcium" else []),
            "offline persistence and database-load time",
            "GUI rendering",
            "representative real plate",
        ],
        "stage_note": (
            "Aggregate stage times can overlap across FOV threads; "
            "cascade_caller includes queue/lock waiting; "
            "ROI helpers are subsets of trace_finalization; FOV population "
            "timers are subsets of fov_analysis_total."
        ),
        "rois_per_fov": args.rois,
        "frames": args.frames,
        "workers": args.workers,
        "analysis": args.analysis,
        "analysis_settings": analysis_data,
        "comparison_excludes": sorted(STOCHASTIC_FIELDS),
        "seed": 9183,
        "frame_rate_hz": 30,
        "model": MODEL if "cascade" in methods else None,
        "model_manifest_sha256": MANIFEST if "cascade" in methods else None,
        "package_revision": package.package_revision,
        "device": args.device if "cascade" in methods else "cpu",
        "torch": str(package.torch.__version__),
        "torch_threads": package.torch.get_num_threads(),
        "inference_torch_thread_counts": sorted(measurements.inference_thread_counts),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "oasis_decay_constant_s": 1.0,
        "retained_payload_bytes_per_cascade_caller_lower_bound": (
            args.frames * reader.side**2 * 8
            + args.rois * args.frames * 8 * 5
            + args.rois * reader.side**2
        ),
        "chunk_windows": args.chunk if args.backend != "reference" else None,
        "queue_capacity": 1 if args.backend == "service" else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "peak_active_cascade_callers": measurements.peak_callers,
        "peak_queue_depth": measurements.peak_queue,
        "model_loader_threads": len(measurements.loader_threads),
        "total_model_loads": measurements.model_loads,
        "cache_stats": stats,
        "baseline_peak_rss_mib": baseline,
        "peak_rss_mib": peak_mib(),
        "incremental_peak_rss_mib": peak_mib() - baseline,
        "process_tree_memory": memory_report,
        "phases": phases,
    }


def main() -> None:
    """Launch fresh processes and verify numerical equality across all output modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("all", "oasis", "cascade", "dual"), default="all"
    )
    parser.add_argument(
        "--backend",
        choices=("reference", "service", "cached-lock"),
        default="reference",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rois", type=int, default=32)
    parser.add_argument("--frames", type=int, default=2048)
    parser.add_argument("--fovs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--analysis", choices=("calcium", "full"), default="calcium")
    parser.add_argument("--analysis-processes", type=int, default=1)
    parser.add_argument("--ccg-shuffles", type=int, default=20)
    parser.add_argument("--rising-edges", action="store_true")
    parser.add_argument("--sample-memory", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.rois,
            args.fovs,
            args.workers,
            args.chunk,
            args.analysis_processes,
            args.ccg_shuffles,
        )
        < 1
        or args.frames < 65
    ):
        parser.error("Counts must be positive and traces need at least 65 frames.")
    if args.rois < 2:
        parser.error("Analysis comparisons need at least two ROIs.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode != "all":
        report = run_case(args)
        (args.output_dir / f"{args.mode}-{args.backend}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return
    reports = []
    for mode, backend in CASES:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            mode,
            "--backend",
            backend,
            "--model-dir",
            str(args.model_dir),
            "--device",
            args.device,
            "--output-dir",
            str(args.output_dir),
            "--rois",
            str(args.rois),
            "--frames",
            str(args.frames),
            "--fovs",
            str(args.fovs),
            "--workers",
            str(args.workers),
            "--chunk",
            str(args.chunk),
            "--analysis",
            args.analysis,
            "--analysis-processes",
            str(args.analysis_processes),
            "--ccg-shuffles",
            str(args.ccg_shuffles),
            *(["--rising-edges"] if args.rising_edges else []),
            *(["--sample-memory"] if args.sample_memory else []),
        ]
        with (args.output_dir / f"{mode}-{backend}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        reports.append(
            json.loads((args.output_dir / f"{mode}-{backend}.json").read_text())
        )
        print(f"Completed {mode}/{backend}", flush=True)
    differences = {}
    scientific_parity = {}
    scientific_differences = {}
    for phase in ("cold", "warm"):
        with np.load(args.output_dir / f"oasis-reference-{phase}.npz") as oasis:
            with np.load(args.output_dir / f"cascade-reference-{phase}.npz") as cascade:
                for mode, backend in CASES:
                    with np.load(
                        args.output_dir / f"{mode}-{backend}-{phase}.npz"
                    ) as data:
                        assert set(data.files) == {
                            "den_dff",
                            "calcium_noise",
                            *(("oasis", "cascade") if mode == "dual" else (mode,)),
                        }
                        errors = {}
                        for name in data.files:
                            expected = (
                                cascade[name] if name == "cascade" else oasis[name]
                            )
                            errors[name] = compare_arrays(
                                data[name],
                                expected,
                                tolerant=args.device != "cpu" and name == "cascade",
                            )
                        differences[f"{mode}/{backend}/{phase}"] = errors
        products = {
            (mode, backend): json.loads(
                (args.output_dir / f"{mode}-{backend}-{phase}-metrics.json").read_text()
            )
            for mode, backend in CASES
        }
        for case, value in products.items():
            assert value["calcium"] == products["oasis", "reference"]["calcium"]
            for method, product in value["spikes"].items():
                metric_errors = compare_spike_products(
                    product,
                    products[method, "reference"]["spikes"][method],
                    tolerant=args.device != "cpu" and method == "cascade",
                )
                scientific_differences[f"{case[0]}/{case[1]}/{phase}/{method}"] = (
                    metric_errors
                )
            scientific_parity[f"{case[0]}/{case[1]}/{phase}"] = True
    report = {
        "schema": 3,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_source_sha256": {
            name: hashlib.sha256(Path(inspect.getfile(value)).read_bytes()).hexdigest()
            for name, value in (
                ("extraction_runner", ExtractionRunner),
                ("fov_analysis_parallel", _fov_analysis_parallel),
                ("fov_analysis", _fov_analysis),
                ("roi_analysis", _roi_analysis),
                ("analysis_runner", AnalysisRunner),
            )
        },
        "results": reports,
        "max_abs_differences": differences,
        "deterministic_scientific_parity": scientific_parity,
        "continuous_scientific_max_abs_differences": scientific_differences,
        "comparison_policy": {
            "device": args.device,
            "rtol": GPU_RTOL if args.device != "cpu" else 0,
            "atol": GPU_ATOL if args.device != "cpu" else 0,
            "tolerant_cascade_fields": sorted(GPU_CONTINUOUS_FIELDS)
            if args.device != "cpu"
            else [],
            "calcium_oasis_discrete_metrics_persistence_offline": "exact",
        },
        "release_gate": (
            "pending: representative real plate, GPU acceptance, independent codec "
            "compressibility and memory acceptance; controlled workloads require review"
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_RUNNING", "1")
    main()
