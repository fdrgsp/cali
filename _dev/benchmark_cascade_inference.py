"""Measure CASCADE Phase-B inference, including retained FOV caller memory.

Run in cali[cascade] with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1. Each mode gets
a fresh process. This controlled workload does not claim complete extraction,
analysis, persistence, or real-plate performance; those gates follow runner wiring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from cali._cascade_package import load_cascade_package
from cali.extraction._extraction_runner import _RoiParts
from cali.extraction._frame_window import TimingDescriptor
from cali.extraction._spike_inference._cascade_cached import CachedCascadePredictor
from cali.extraction._spike_inference._cascade_reference import (
    CascadeReferenceBackend,
    resolve_cascade_device,
)
from cali.extraction._spike_inference._cascade_service import CascadeInferenceService

MODEL = "Global_EXC_30Hz_smoothing25ms"
MANIFEST = "ac8954174ba0a01a2d929a7e8b3fc7e3a4365d5c01f5e262d2b597822fcae184"
MODES = ("reference", "cached", "service", "cached-lock")
INCREMENTAL_RSS_BUDGET_MIB = 256
RTOL, ATOL = 1e-5, 1e-6


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def _device_memory(package: object, device: str) -> dict | None:
    """Report device bytes separately from host RSS, without inventing MPS peaks."""
    torch = package.torch
    if device == "mps":
        torch.mps.synchronize()
        return {
            "tensor_allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "recommended_max_bytes": torch.mps.recommended_max_memory(),
            "scope": "synchronized snapshots, not a peak-memory measurement",
        }
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        return {
            "tensor_allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_tensor_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "scope": "allocator bytes; excludes other processes and driver overhead",
        }
    return None


def _peak_mib() -> float:
    size = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return size / (1024**2 if sys.platform == "darwin" else 1024)


def _payload(template: np.ndarray) -> tuple[np.ndarray, list[_RoiParts], np.ndarray]:
    """Touch source images and all four actual Phase-A trace arrays per ROI."""
    image = np.ones((template.shape[1], 32, 32), dtype=np.uint16)
    parts = []
    for index, row in enumerate(template):
        mask = np.zeros((32, 32), dtype=bool)
        mask.flat[index % mask.size] = True
        parts.append(
            _RoiParts(index, mask, 100 + row, row.copy(), 20 + row, row.copy(), 1, "px")
        )
    return np.stack([part.dff for part in parts]), parts, image


class _LockedBenchmarkPredictor(CachedCascadePredictor):
    """Benchmark-only alternative: every operation is guarded by the caller lock.

    This deliberately transfers serialized Torch access across FOV threads to
    compare queue ownership with a global-lock baseline. The shipping predictor
    keeps its single-owner guard; this subclass is never used by the application.
    """

    def _claim_owner(self) -> None:
        pass


def _run_mode(args: argparse.Namespace) -> dict:
    begin = time.perf_counter()
    package = load_cascade_package()
    package_initialization = time.perf_counter() - begin
    baseline_peak = _peak_mib()
    rng = np.random.default_rng(9183)
    template = rng.normal(
        0, np.linspace(0.12, 0.4, args.rois)[:, None], (args.rois, args.frames)
    )
    for onset in range(75, args.frames, 200):
        template[:, onset:] += 0.35 * np.exp(-np.arange(args.frames - onset) / 12)
    timing = TimingDescriptor(
        (np.arange(args.frames) * 1000 / 30).tolist(), "runner_time", True
    )
    load_count = 0
    load_threads = set()
    original_load = package.torch.load

    def counted_load(*positional: object, **keywords: object) -> object:
        nonlocal load_count
        load_count += 1
        load_threads.add(threading.get_ident())
        return original_load(*positional, **keywords)

    package.torch.load = counted_load
    options = {
        "model_name": MODEL,
        "model_dir": args.model_dir,
        "expected_manifest": MANIFEST,
        "device": args.device,
    }
    if args.mode == "reference":
        backend = CascadeReferenceBackend(**options)
    elif args.mode == "service":
        backend = CascadeInferenceService(**options, max_windows=args.chunk)
    elif args.mode == "cached-lock":
        backend = _LockedBenchmarkPredictor(**options, max_windows=args.chunk)
    else:
        backend = CachedCascadePredictor(**options, max_windows=args.chunk)
    lock = threading.Lock()
    counters_lock = threading.Lock()
    active_callers = peak_callers = peak_queue = 0
    stop = threading.Event()

    def monitor() -> None:
        nonlocal peak_queue
        while not stop.wait(0.005):
            if args.mode == "service":
                peak_queue = max(peak_queue, backend.pending_count)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    cold = None
    device_memory = {}

    def infer() -> np.ndarray:
        nonlocal active_callers, peak_callers
        dff, parts, image = _payload(template)
        with counters_lock:
            active_callers += 1
            peak_callers = max(peak_callers, active_callers)
        try:
            if args.mode in {"reference", "cached-lock"}:
                with lock:
                    result = backend.infer_all(dff, 30, timing=timing)
            else:
                result = backend.infer_all(dff, 30, timing=timing)
            assert len(parts) == args.rois and image.shape[0] == args.frames
            if cold is not None:
                if args.device == "cpu":
                    np.testing.assert_array_equal(result.spikes, cold)
                else:
                    np.testing.assert_allclose(
                        result.spikes, cold, rtol=RTOL, atol=ATOL
                    )
            return result.spikes
        finally:
            with counters_lock:
                active_callers -= 1

    try:
        cold_begin = time.perf_counter()
        resolved_device = resolve_cascade_device(package, args.device)
        if resolved_device.startswith("cuda"):
            package.torch.cuda.reset_peak_memory_stats(resolved_device)
        device_memory["baseline"] = _device_memory(package, resolved_device)
        cold = infer()
        cold_s = time.perf_counter() - cold_begin
        device_memory["cold"] = _device_memory(package, resolved_device)
        cold_loads = load_count
        np.savez_compressed(
            args.output_dir / f"{args.mode}-prediction.npz", spikes=cold
        )
        warm_begin = time.perf_counter()
        if args.mode == "cached":
            totals = [float(infer().sum(dtype=np.float64)) for _ in range(args.fovs)]
        else:

            def run_fov(index: int) -> float:
                return float(infer().sum(dtype=np.float64))

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                totals = list(pool.map(run_fov, range(args.fovs)))
        warm_s = time.perf_counter() - warm_begin
        device_memory["warm"] = _device_memory(package, resolved_device)
        stats = None if args.mode == "reference" else vars(backend.stats)
    finally:
        stop.set()
        monitor_thread.join()
        if args.mode != "reference":
            backend.close()
        if cold is not None:
            device_memory["closed"] = _device_memory(package, resolved_device)
        package.torch.load = original_load
    peak = _peak_mib()
    incremental = max(peak - baseline_peak, 0)
    retained_bytes_per_fov = (
        args.rois * args.frames * 8 * 5
        + args.frames * 32 * 32 * 2
        + args.rois * 32 * 32
    )
    return {
        "mode": args.mode,
        "scope": (
            "controlled Phase-B inference with real _RoiParts "
            "and source-image retention"
        ),
        "model": MODEL,
        "model_manifest_sha256": MANIFEST,
        "package_revision": package.package_revision,
        "device": resolved_device,
        "device_memory": device_memory,
        "dtype": "float32",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": str(package.torch.__version__),
        "torch_threads": package.torch.get_num_threads(),
        "rois": args.rois,
        "frames": args.frames,
        "warm_fovs": args.fovs,
        "workers": 1 if args.mode == "cached" else args.workers,
        "chunk_windows": None if args.mode == "reference" else args.chunk,
        "queue_capacity": 1 if args.mode == "service" else None,
        "package_initialization_s": package_initialization,
        "model_cold_fov_s": cold_s,
        "warm_batch_s": warm_s,
        "cold_model_loads": cold_loads,
        "warm_model_loads": load_count - cold_loads,
        "model_loader_threads": len(load_threads),
        "cache_stats": stats,
        "peak_active_callers": peak_callers,
        "peak_queue_depth": peak_queue,
        "retained_payload_bytes_per_fov": retained_bytes_per_fov,
        "baseline_peak_rss_mib": baseline_peak,
        "peak_rss_mib": peak,
        "incremental_peak_rss_mib": incremental,
        "incremental_rss_budget_mib": INCREMENTAL_RSS_BUDGET_MIB,
        "within_incremental_budget": incremental <= INCREMENTAL_RSS_BUDGET_MIB,
        "warm_expected_spikes_totals": totals,
        "all_warm_fovs_match_cold": True,
        "input_dff_sha256": hashlib.sha256(template.tobytes()).hexdigest(),
        "prediction_sha256": _checksum(args.output_dir / f"{args.mode}-prediction.npz"),
        "package_source_manifest_sha256": package.source_manifest_sha256,
    }


def main() -> None:
    """Compare isolated devices/backends with explicitly bound CPU oracle inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rois", type=int, default=100)
    parser.add_argument("--frames", type=int, default=6000)
    parser.add_argument("--fovs", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--cpu-reference", type=Path)
    args = parser.parse_args()
    if min(args.rois, args.frames, args.fovs, args.workers, args.chunk) <= 0:
        parser.error("Counts and chunk size must be positive")
    if args.cpu_reference is not None and args.mode != "all":
        parser.error("--cpu-reference requires --mode all")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode != "all":
        for suffix in (".json", "-prediction.npz"):
            if (args.output_dir / (args.mode + suffix)).exists():
                raise FileExistsError(args.output_dir / (args.mode + suffix))
        result = _run_mode(args)
        (args.output_dir / f"{args.mode}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result, indent=2))
        return
    results = []
    if (args.output_dir / "report.json").exists():
        raise FileExistsError(args.output_dir / "report.json")
    for mode in MODES:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            mode,
            "--model-dir",
            str(args.model_dir),
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
            "--device",
            args.device,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        (args.output_dir / f"{mode}.log").write_text(
            completed.stdout + completed.stderr
        )
        if completed.returncode:
            raise RuntimeError(
                f"{mode} failed; see {args.output_dir / (mode + '.log')}"
            )
        results.append(json.loads((args.output_dir / f"{mode}.json").read_text()))
        print(
            f"{mode}: {results[-1]['warm_batch_s']:.3f}s warm, "
            f"{results[-1]['peak_rss_mib']:.1f} MiB peak",
            flush=True,
        )
    with np.load(args.output_dir / "reference-prediction.npz") as source:
        oracle = source["spikes"]
    max_errors = {}
    for mode in MODES[1:]:
        with np.load(args.output_dir / f"{mode}-prediction.npz") as saved:
            prediction = saved["spikes"]
        np.testing.assert_allclose(prediction, oracle, rtol=RTOL, atol=ATOL)
        max_errors[mode] = float(np.max(np.abs(prediction - oracle)))
    by_mode = {row["mode"]: row for row in results}
    report = {
        "schema_version": 1,
        "runs": results,
        "max_abs_errors_vs_reference": max_errors,
        "warm_service_speedup_vs_reference": by_mode["reference"]["warm_batch_s"]
        / by_mode["service"]["warm_batch_s"],
        "warm_service_speedup_vs_cached_lock": by_mode["cached-lock"]["warm_batch_s"]
        / by_mode["service"]["warm_batch_s"],
        "production_full_mode_gates": (
            "controlled inference only; independent real plate, representative "
            "memory and complete GPU workloads remain separate acceptance gates"
        ),
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "script_sha256": _checksum(Path(__file__)),
        "implementation_source_sha256": {
            name: hashlib.sha256(
                Path(
                    importlib.util.find_spec(
                        "cali.extraction._spike_inference." + name
                    ).origin
                ).read_bytes()
            ).hexdigest()
            for name in ("_cascade_reference", "_cascade_cached", "_cascade_service")
        },
    }
    if args.cpu_reference is not None:
        cpu = json.loads((args.cpu_reference / "reference.json").read_text())
        actual = by_mode["reference"]
        for field in (
            "model",
            "model_manifest_sha256",
            "package_revision",
            "package_source_manifest_sha256",
            "input_dff_sha256",
            "rois",
            "frames",
        ):
            if cpu[field] != actual[field]:
                raise ValueError(f"CPU oracle identity mismatch: {field}")
        cpu_path = args.cpu_reference / "reference-prediction.npz"
        if cpu["device"] != "cpu" or _checksum(cpu_path) != cpu["prediction_sha256"]:
            raise ValueError("CPU oracle device or prediction checksum mismatch.")
        with np.load(cpu_path, allow_pickle=False) as saved:
            cpu_prediction = saved["spikes"]
        cross_device_errors = {}
        for mode in MODES:
            with np.load(
                args.output_dir / f"{mode}-prediction.npz", allow_pickle=False
            ) as saved:
                prediction = saved["spikes"]
            np.testing.assert_allclose(prediction, cpu_prediction, rtol=RTOL, atol=ATOL)
            cross_device_errors[mode] = float(
                np.max(np.abs(prediction - cpu_prediction))
            )
        report["cross_device_cpu_oracle"] = {
            "prediction_sha256": cpu["prediction_sha256"],
            "max_abs_errors": cross_device_errors,
            "all_modes_within_tolerance": True,
        }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "runs"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
