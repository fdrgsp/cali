"""Measure current JSON storage and a lossless BLOB candidate, without migrating.

Short inputs are the real upstream golden excerpt, repeated to the requested ROI
count. Long inputs reuse the verified 100 x 6000 P3b predictions and reconstruct
their seeded DFF. Repeated FOVs are controlled storage workloads, not a real plate.
Each mode runs in a fresh process. Candidate BLOB files are benchmark artifacts;
the application cannot read them until a versioned codec migration is implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sqlite3
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from benchmark_cascade_extraction import MANIFEST, MODEL, make_fov, peak_mib
from sqlmodel import Session, select

from cali._constants import CASCADE_EXPECTED_SPIKES_TRACES, INFERRED_SPIKES_TRACES
from cali.extraction._spike_inference import OasisBackend
from cali.runner import CaliRunner
from cali.sqlmodel import (
    CaliResult,
    Experiment,
    ExtractionFrameWindow,
    ExtractionSettings,
    Plate,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    Well,
    create_cali_engine,
    create_database_and_tables,
)
from cali.util._database_to_csv import export_traces_to_csv

MODES = ("oasis", "cascade", "dual", "dual-legacy-duplication")
PLATE_FOVS = 96
SPIKE_PAYLOAD_BUDGET_BYTES = 512 * 1024**2


def inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bind the workload to recorded, independently checked predictions."""
    if args.workload == "short":
        fixture = Path(__file__).parents[1] / "tests/fixtures/cascade_reference"
        metadata = json.loads((fixture / "manifest.json").read_text())
        source = fixture / "real_excerpt.npz"
        if (
            hashlib.sha256(source.read_bytes()).hexdigest()
            != metadata["fixture_sha256"]
        ):
            raise ValueError("Golden fixture checksum changed.")
        with np.load(source) as data:
            indices = np.arange(args.rois) % len(data["dff"])
            dff = data["dff"][indices].copy()
            spikes = data["expected_spikes"][indices].astype(np.float32)
        return (
            dff,
            spikes,
            {
                "source": "real upstream golden excerpt; repeated ROI/FOV workload",
                "source_sha256": metadata["fixture_sha256"],
                "valid_interval": metadata["valid_interval"],
            },
        )
    if args.prediction is None or args.reference_prediction is None or args.rois != 100:
        raise ValueError(
            "Long workload needs the P3b cached and reference NPZ files and 100 ROIs."
        )
    with np.load(args.prediction) as data, np.load(args.reference_prediction) as oracle:
        spikes = data["spikes"].copy()
        if spikes.shape != (100, 6000) or spikes.dtype != np.float32:
            raise ValueError("Expected the recorded float32 100 x 6000 P3b workload.")
        np.testing.assert_array_equal(spikes, oracle["spikes"].astype(np.float32))
    rng = np.random.default_rng(9183)
    dff = rng.normal(0, np.linspace(0.12, 0.4, 100)[:, None], (100, 6000))
    for onset in range(75, 6000, 200):
        dff[:, onset:] += 0.35 * np.exp(-np.arange(6000 - onset) / 12)
    return (
        dff,
        spikes,
        {
            "source": (
                "recorded P3b synthetic DFF, "
                "cached predictions equal independent reference"
            ),
            "source_sha256": hashlib.sha256(args.prediction.read_bytes()).hexdigest(),
            "reference_sha256": hashlib.sha256(
                args.reference_prediction.read_bytes()
            ).hexdigest(),
            "valid_interval": [32, 5968],
        },
    )


def encode_candidate(values: list[float], method: str) -> tuple[bytes, str]:
    """Prototype lossless encoding; preserve OASIS float64 and CASCADE float32."""
    array = np.asarray(values, dtype="<f4" if method == "cascade" else "<f8")
    # CASCADE was already persisted from float32; never silently downcast others.
    np.testing.assert_array_equal(array.astype(np.float64), np.asarray(values))
    raw = array.tobytes()
    metadata = json.dumps(
        {
            "version": 1,
            "compression": "zlib",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        separators=(",", ":"),
    )
    return zlib.compress(raw, level=6), metadata


def decode_candidate(payload: bytes, encoded_metadata: str) -> np.ndarray:
    """Validate the candidate metadata, checksum, shape and exact byte length."""
    metadata = json.loads(encoded_metadata)
    if metadata["version"] != 1 or metadata["compression"] != "zlib":
        raise ValueError("Unsupported candidate codec.")
    raw = zlib.decompress(payload)
    dtype = np.dtype(metadata["dtype"])
    if (
        hashlib.sha256(raw).hexdigest() != metadata["sha256"]
        or len(raw) != math.prod(metadata["shape"]) * dtype.itemsize
    ):
        raise ValueError("Corrupted candidate trace.")
    return np.frombuffer(raw, dtype=dtype).reshape(metadata["shape"])


def rounding_errors(spikes: np.ndarray, start: int, stop: int) -> dict:
    """Bound six-decimal rounding, including Gaussian single-AP threshold crossings."""
    original = spikes[:, start:stop].astype(np.float64)
    rounded = np.round(original, 6)
    sum_error = np.abs(rounded.sum(axis=1) - original.sum(axis=1))
    duration = (stop - start) / 30
    ap_peak = 1 / (math.sqrt(2 * math.pi) * 0.025 * 30)
    eligible = (original != rounded) & (original > 0) & (original < ap_peak)
    candidates = np.where(eligible, np.abs(original - rounded), -1)
    index = np.unravel_index(np.argmax(candidates), original.shape)
    fraction = float((original[index] + rounded[index]) / 2 / ap_peak)
    return {
        "decimal_places": 6,
        "max_sample_abs_error": float(np.max(np.abs(rounded - original))),
        "max_roi_sum_abs_error": float(sum_error.max()),
        "max_roi_mean_rate_hz_abs_error": float(sum_error.max() / duration),
        "changed_ap_threshold_samples": {
            str(fraction): int(
                np.count_nonzero(
                    (original > fraction * ap_peak) != (rounded > fraction * ap_peak)
                )
            )
            for fraction in (0.1, 0.2, 0.5, 1.0)
        },
        "demonstrated_sensitive_ap_threshold": {
            "fraction": fraction,
            "changed_samples": int(
                np.count_nonzero(
                    (original > fraction * ap_peak) != (rounded > fraction * ap_peak)
                )
            ),
        }
        if np.any(eligible)
        else None,
        "decision": (
            "Do not quantize: candidate compression preserves all stored values "
            "exactly; tested thresholds do not bound arbitrary user thresholds."
        ),
    }


def run_case(args: argparse.Namespace) -> dict:
    """Write/read/export actual ORM JSON graphs; measure the candidate separately."""
    dff, cascade, source = inputs(args)
    count, frames = dff.shape
    start, stop = source["valid_interval"]
    oasis = OasisBackend().infer_all(
        dff, 30 * frames / (frames - 1), decay_constant=1.0
    )
    methods = ("oasis", "cascade") if args.mode.startswith("dual") else (args.mode,)
    database = args.output_dir / f"{args.workload}-{args.mode}.cali"
    if database.exists():
        raise FileExistsError(database)
    engine = create_cali_engine(f"sqlite:///{database}")
    create_database_and_tables(engine)
    settings = ExtractionSettings(
        spike_methods=methods,
        cascade_model=MODEL if "cascade" in methods else None,
        cascade_device="cpu",
        frame_rate=30,
    )
    fovs = [make_fov(index, count) for index in range(args.fovs)]
    experiment = Experiment(
        name="storage benchmark",
        plate=Plate(
            name="controlled", wells=[Well(name="A1", row=0, column=0, fovs=fovs)]
        ),
    )
    with Session(engine) as session:
        session.add_all([experiment, settings])
        session.flush()
        result = CaliResult(
            experiment=experiment.id, extraction_settings_id=settings.id
        )
        session.add(result)
        session.commit()
        run_id = result.id
        runs = {
            method: SpikeInferenceRun(
                method=method,
                units="spikes/frame" if method == "cascade" else "a.u.",
                backend_package="cascade2p" if method == "cascade" else "oasis-deconv",
                dtype="float32" if method == "cascade" else "float64",
                resolved_model=MODEL if method == "cascade" else None,
                weights_manifest_sha256=MANIFEST if method == "cascade" else None,
                provenance_source="benchmark_recorded_prediction",
            )
            for method in methods
        }
        for fov in fovs:
            window = ExtractionFrameWindow(
                original_frame_count=frames,
                retained_frame_count=frames,
                source_start_frame=0,
                source_start_time_ms=0,
                timing_source="runner_time",
                timing_trusted=True,
                acquisition_frame_rate_hz=30,
                schema_version=3,
            )
            for index, roi in enumerate(fov.rois):
                roi._new_traces = [
                    Traces(
                        raw_trace=(100 + dff[index]).tolist(),
                        dff=dff[index].tolist(),
                        den_dff=oasis.den_dff[index].tolist(),
                        calcium_noise=float(oasis.sn_by_roi[index]),
                        x_axis=(np.arange(frames) * 1000 / 30).tolist(),
                        x_axis_units="ms",
                        extraction_frame_window=window,
                        spike_traces=[
                            SpikeTrace(
                                values=(
                                    cascade if method == "cascade" else oasis.spikes
                                )[index].tolist(),
                                valid_start=start if method == "cascade" else 0,
                                valid_stop=stop if method == "cascade" else frames,
                                inference_run=runs[method],
                            )
                            for method in methods
                        ],
                    )
                ]
        begin = time.perf_counter()
        for fov in fovs:
            CaliRunner()._process_fov_results(fov, session, run_id, include_traces=True)
        session.commit()
        write_s = time.perf_counter() - begin
    engine.dispose()
    legacy_write_s = 0
    with sqlite3.connect(database) as connection:
        if args.mode == "dual-legacy-duplication":
            begin = time.perf_counter()
            connection.execute(
                'UPDATE trace SET inferred_spikes = (SELECT s."values" '
                "FROM spike_trace s JOIN spike_inference_run r "
                "ON r.id=s.spike_inference_run_id "
                "WHERE s.trace_id=trace.id AND r.method='oasis')"
            )
            connection.commit()
            legacy_write_s = time.perf_counter() - begin
        # Canonical file size after checkpoint/vacuum, with maintenance cost separate.
        begin = time.perf_counter()
        connection.execute("VACUUM")
        maintenance_s = time.perf_counter() - begin
        payload_bytes = dict(
            connection.execute(
                'SELECT r.method,SUM(length(CAST(s."values" AS BLOB))) '
                "FROM spike_trace s JOIN spike_inference_run r "
                "ON r.id=s.spike_inference_run_id GROUP BY r.method"
            )
        )
        legacy_bytes = connection.execute(
            "SELECT COALESCE(SUM(length(CAST(inferred_spikes AS BLOB))),0) "
            "FROM trace WHERE inferred_spikes != 'null'"
        ).fetchone()[0]
    engine = create_cali_engine(f"sqlite:///{database}")
    begin = time.perf_counter()
    with Session(engine) as session:
        rows = session.exec(select(SpikeTrace)).all()
        read_s = time.perf_counter() - begin
        begin = time.perf_counter()
        arrays = [
            np.asarray(row.values)[row.valid_start : row.resolved_valid_stop]
            for row in rows
        ]
        plot_preparation_s = time.perf_counter() - begin
        assert sum(len(array) for array in arrays) > 0
        methods_by_id = {row.id: row.inference_run.method for row in rows}
    begin = time.perf_counter()
    export_traces_to_csv(
        engine,
        {
            INFERRED_SPIKES_TRACES: "oasis" in methods,
            CASCADE_EXPECTED_SPIKES_TRACES: "cascade" in methods,
        },
        run_id,
        database,
    )
    export_s = time.perf_counter() - begin
    engine.dispose()
    candidate_path = database.with_suffix(".candidate.cali")
    with (
        sqlite3.connect(database) as original,
        sqlite3.connect(candidate_path) as candidate,
    ):
        original.backup(candidate)
        candidate.execute("ALTER TABLE spike_trace ADD COLUMN values_blob BLOB")
        candidate.execute("ALTER TABLE spike_trace ADD COLUMN values_metadata TEXT")
        begin = time.perf_counter()
        candidate_bytes = dict.fromkeys(methods, 0)
        for identifier, encoded in original.execute(
            'SELECT id,"values" FROM spike_trace'
        ):
            method = methods_by_id[identifier]
            payload, metadata = encode_candidate(json.loads(encoded), method)
            candidate_bytes[method] += len(payload) + len(metadata.encode())
            candidate.execute(
                "UPDATE spike_trace SET \"values\"='[]',values_blob=?,"
                "values_metadata=? WHERE id=?",
                (payload, metadata, identifier),
            )
        candidate.commit()
        candidate_encode_write_s = time.perf_counter() - begin
        begin = time.perf_counter()
        candidate.execute("VACUUM")
        candidate_maintenance_s = time.perf_counter() - begin
        begin = time.perf_counter()
        decoded_arrays = {}
        for identifier, payload, metadata in candidate.execute(
            "SELECT id,values_blob,values_metadata FROM spike_trace"
        ):
            decoded_arrays[identifier] = decode_candidate(payload, metadata)
        candidate_sql_read_decode_s = time.perf_counter() - begin
        begin = time.perf_counter()
        for identifier, decoded in decoded_arrays.items():
            original_values = json.loads(
                original.execute(
                    'SELECT "values" FROM spike_trace WHERE id=?', (identifier,)
                ).fetchone()[0]
            )
            np.testing.assert_array_equal(decoded.astype(np.float64), original_values)
        candidate_read_verify_s = time.perf_counter() - begin
    projection = sum(payload_bytes.values()) / args.fovs * PLATE_FOVS
    return {
        "mode": args.mode,
        "workload": args.workload,
        "source": source,
        "scope": (
            "actual ORM/SQLite JSON and CSV export; "
            "repeated input traces/FOVs, no inference timings"
        ),
        "rois_per_fov": count,
        "frames": frames,
        "fovs": args.fovs,
        "model": MODEL,
        "model_manifest_sha256": MANIFEST,
        "oasis_decay_constant_s": 1.0,
        "spike_payload_bytes": payload_bytes,
        "nonzero_sample_fraction": {
            "cascade": float(np.count_nonzero(cascade) / cascade.size),
            "oasis": float(np.count_nonzero(oasis.spikes) / oasis.spikes.size),
        },
        "legacy_duplicate_bytes": legacy_bytes,
        "database_bytes": database.stat().st_size,
        "write_s": write_s,
        "legacy_write_s": legacy_write_s,
        "maintenance_s": maintenance_s,
        "orm_read_s": read_s,
        "export_s": export_s,
        "plot_array_preparation_s": plot_preparation_s,
        "plot_note": (
            "ORM read plus NumPy conversion/valid slicing; GUI rendering and "
            "method-bound plot integration remain pending."
        ),
        "candidate": {
            "status": (
                "benchmark-only BLOB sidecar, not an application-readable database"
            ),
            "spike_payload_bytes_including_metadata": candidate_bytes,
            "database_bytes": candidate_path.stat().st_size,
            "encode_verify_write_s": candidate_encode_write_s,
            "maintenance_s": candidate_maintenance_s,
            "sql_read_decode_s": candidate_sql_read_decode_s,
            "compare_to_original_json_s": candidate_read_verify_s,
            "max_sample_error": 0,
            "max_sum_error": 0,
            "mean_rate_error": 0,
            "changed_threshold_crossings": 0,
        },
        "six_decimal_rounding": rounding_errors(cascade, start, stop),
        "projection_96_fov_spike_payload_bytes": projection,
        "projection_96_fov_candidate_payload_bytes": sum(candidate_bytes.values())
        / args.fovs
        * PLATE_FOVS,
        "projection_96_fov_candidate_with_legacy_bytes": (
            sum(candidate_bytes.values()) + legacy_bytes
        )
        / args.fovs
        * PLATE_FOVS,
        "spike_payload_budget_96_fovs_bytes": SPIKE_PAYLOAD_BUDGET_BYTES,
        "within_json_budget": projection + legacy_bytes / args.fovs * PLATE_FOVS
        <= SPIKE_PAYLOAD_BUDGET_BYTES,
        "peak_rss_mib": peak_mib(),
    }


def main() -> None:
    """Isolate modes and store the measured release decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--workload", choices=("short", "long"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--reference-prediction", type=Path)
    parser.add_argument("--rois", type=int, default=100)
    parser.add_argument("--fovs", type=int, default=4)
    args = parser.parse_args()
    if min(args.rois, args.fovs) < 1:
        parser.error("ROI/FOV counts must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode != "all":
        report = run_case(args)
        (args.output_dir / f"{args.workload}-{args.mode}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        return
    reports = []
    for mode in MODES:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--mode",
            mode,
            "--workload",
            args.workload,
            "--output-dir",
            str(args.output_dir),
            "--rois",
            str(args.rois),
            "--fovs",
            str(args.fovs),
        ]
        for flag, value in (
            ("--prediction", args.prediction),
            ("--reference-prediction", args.reference_prediction),
        ):
            if value is not None:
                command.extend([flag, str(value)])
        with (args.output_dir / f"{args.workload}-{mode}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        reports.append(
            json.loads((args.output_dir / f"{args.workload}-{mode}.json").read_text())
        )
        print(f"Completed {args.workload}/{mode}", flush=True)
    report = {
        "schema": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_script_sha256": hashlib.sha256(
            Path(__file__).with_name("benchmark_cascade_extraction.py").read_bytes()
        ).hexdigest(),
        "runtime_source_sha256": {
            name: hashlib.sha256(Path(inspect.getfile(value)).read_bytes()).hexdigest()
            for name, value in (
                ("spike_trace", SpikeTrace),
                ("cali_runner", CaliRunner),
                ("export_traces", export_traces_to_csv),
            )
        },
        "results": reports,
        "size_budget": (
            "512 MiB for spike arrays including compatibility duplication at "
            "96 FOVs x 100 ROIs x 6000 frames; "
            "base traces, indices and analyses are additional."
        ),
        "numerical_budget": (
            "Zero additional error relative to stored CASCADE float32 "
            "and OASIS float64; no quantization."
        ),
        "release_gate": (
            "blocked: implement a versioned lossless trace-array codec "
            "with legacy JSON reads"
        )
        if not all(row["within_json_budget"] for row in reports)
        else (
            "workload within size budget; "
            "long workload and real plate acceptance still required"
        ),
    }
    (args.output_dir / f"{args.workload}-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTEST_RUNNING", "1")
    main()
