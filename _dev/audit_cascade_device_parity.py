"""Compare completed GPU image workloads with a bound, read-only CPU oracle.

The CPU oracle may predate additive schema-14 noise summaries. Those fields are
excluded only from cross-version metrics; stored source/model noise is still exact.
This is controlled correctness evidence, not an independent real-recording gate.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

import numpy as np
from audit_cascade_extraction import audit, checksum
from audit_cascade_noise_qc import QC_FIELDS, without_qc
from benchmark_cascade_extraction import compare_arrays, compare_spike_products

from cali.sqlmodel._trace_array_codec import decode_trace_array

INPUT_FIELDS = (
    "mode",
    "rois_per_fov",
    "frames",
    "workers",
    "analysis",
    "analysis_settings",
    "seed",
    "frame_rate_hz",
    "model",
    "model_manifest_sha256",
    "package_revision",
    "torch",
    "torch_threads",
    "oasis_decay_constant_s",
)


def readonly(path: Path) -> sqlite3.Connection:
    """Never migrate or mutate either recorded workload."""
    uri = "file:" + quote(str(path.resolve()), safe="/") + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def compare_sources(cpu: Path, gpu: Path, *, device: str) -> dict:
    """Bind all inference inputs, noise, windows and threshold decisions exactly."""
    query = (
        "SELECT f.position_index,r.label_value,t.raw_trace,t.dff,t.den_dff,t.x_axis,"
        "t.calcium_noise,s.noise,s.selected_noise_level,s.valid_start,s.valid_stop,"
        "i.method,i.resolved_device,i.resolved_model,i.weights_manifest_sha256,"
        'a.threshold,a.spike_active,s."values" FROM trace t '
        "JOIN roi r ON r.id=t.roi_id "
        "JOIN fov f ON f.id=r.fov_id JOIN spike_trace s ON s.trace_id=t.id "
        "JOIN spike_inference_run i ON i.id=s.spike_inference_run_id "
        "JOIN spike_analysis a ON a.spike_trace_id=s.id "
        "ORDER BY f.position_index,r.label_value,i.method"
    )
    rows = 0
    with closing(readonly(cpu)) as oracle, closing(readonly(gpu)) as measured:
        assert not measured.execute("PRAGMA foreign_key_check").fetchall()
        for expected, actual in zip(
            oracle.execute(query), measured.execute(query), strict=True
        ):
            # Each source array is equal, not merely a matching sum/hash.
            assert expected[:2] == actual[:2]
            for index in range(2, 6):
                compare_arrays(
                    np.asarray(json.loads(actual[index])),
                    np.asarray(json.loads(expected[index])),
                    tolerant=False,
                )
            assert expected[6:12] == actual[6:12]
            assert expected[13:17] == actual[13:17]
            start, stop = actual[9:11]
            expected_values = np.asarray(decode_trace_array(expected[17]))
            actual_values = np.asarray(decode_trace_array(actual[17]))
            if actual[11] == "cascade":
                for values in (expected_values, actual_values):
                    assert np.isfinite(values).all() and (values >= 0).all()
                    assert not np.any(values[:start]) and not np.any(values[stop:])
            expected_values = expected_values[start:stop]
            actual_values = actual_values[start:stop]
            compare_arrays(
                actual_values, expected_values, tolerant=actual[11] == "cascade"
            )
            np.testing.assert_array_equal(
                actual_values > actual[15], expected_values > expected[15]
            )
            if actual[11] == "cascade":
                assert expected[12] == "cpu" and actual[12] == device
            else:
                assert expected[12] == actual[12] == "cpu"
            rows += 1
        # Source-frame windows also include timing verification and crop rules.
        columns = [
            row[1]
            for row in measured.execute("PRAGMA table_info(extraction_frame_window)")
            if row[1]
            not in {"id", "extraction_result_id", "fov_id", "legacy_owner_result_id"}
        ]
        window_query = (
            "SELECT f.position_index,"
            + ",".join(f'w."{name}"' for name in columns)
            + " FROM extraction_frame_window w JOIN fov f ON f.id=w.fov_id "
            "ORDER BY f.position_index"
        )
        assert (
            oracle.execute(window_query).fetchall()
            == measured.execute(window_query).fetchall()
        )
        provenance_columns = [
            row[1]
            for row in measured.execute("PRAGMA table_info(spike_inference_run)")
            if row[1]
            not in {
                "id",
                "extraction_result_id",
                "legacy_owner_result_id",
                "resolved_device",
            }
        ]
        provenance_query = (
            "SELECT "
            + ",".join(f'"{name}"' for name in provenance_columns)
            + " FROM spike_inference_run ORDER BY method"
        )
        assert (
            oracle.execute(provenance_query).fetchall()
            == measured.execute(provenance_query).fetchall()
        )
    assert rows > 0
    return {
        "trace_method_rows": rows,
        "source_arrays_noise_windows_thresholds_activity_exact": True,
        "all_stored_model_provenance_except_device_exact": True,
        "compared_source_arrays": ["raw_trace", "dff", "den_dff", "x_axis"],
        "cascade_finite_nonnegative_and_padding_exact": True,
    }


def compare(cpu: Path, gpu: Path) -> dict:
    """Use each output mode's CPU reference; preserve discrete population science."""
    baseline = json.loads((cpu / "report.json").read_text())
    report = json.loads((gpu / "report.json").read_text())
    device = report["comparison_policy"]["device"]
    assert device in {"mps", "cuda"}
    references = {
        case["mode"]: case
        for case in baseline["results"]
        if case["backend"] == "reference"
    }
    checked = {}
    for case in report["results"]:
        reference = references[case["mode"]]
        assert reference["device"] == "cpu"
        assert case["device"] == ("cpu" if case["mode"] == "oasis" else device)
        assert all(case[key] == reference[key] for key in INPUT_FIELDS)
        for phase, old_phase in zip(case["phases"], reference["phases"], strict=True):
            assert (phase["phase"], phase["fovs"]) == (
                old_phase["phase"],
                old_phase["fovs"],
            )
            stem = f"{case['mode']}-{case['backend']}-{phase['phase']}"
            source_stem = f"{case['mode']}-reference-{phase['phase']}"
            errors = {}
            with (
                np.load(cpu / f"{source_stem}.npz", allow_pickle=False) as expected,
                np.load(gpu / f"{stem}.npz", allow_pickle=False) as actual,
            ):
                assert set(actual.files) == set(expected.files)
                for name in actual.files:
                    errors[name] = compare_arrays(
                        actual[name], expected[name], tolerant=name == "cascade"
                    )
            original = without_qc(
                json.loads((cpu / f"{source_stem}-metrics.json").read_text())
            )
            measured = without_qc(
                json.loads((gpu / f"{stem}-metrics.json").read_text())
            )
            assert measured["calcium"] == original["calcium"]
            assert measured["spikes"].keys() == original["spikes"].keys()
            metrics = {}
            for method, products in measured["spikes"].items():
                oracle = original["spikes"][method]
                # Every FOV product derives from binary decisions: require exact
                # arrays, correlations, synchrony, lag choices and burst outputs.
                assert products["fov"] == oracle["fov"]
                metrics[method] = compare_spike_products(
                    products, oracle, tolerant=method == "cascade"
                )
            source_audit = compare_sources(
                cpu / f"{source_stem}.cali", gpu / f"{stem}.cali", device=case["device"]
            )
            checked[stem] = {
                "cpu_case_sha256": checksum(cpu / f"{case['mode']}-reference.json"),
                "cpu_arrays_sha256": checksum(cpu / f"{source_stem}.npz"),
                "cpu_database_sha256": checksum(cpu / f"{source_stem}.cali"),
                "gpu_arrays_sha256": checksum(gpu / f"{stem}.npz"),
                "max_abs_array_differences": errors,
                "continuous_roi_max_abs_differences": metrics,
                "calcium_oasis_and_all_deterministic_fov_products_exact": True,
                **source_audit,
            }
    return {
        "audit_script_sha256": checksum(Path(__file__)),
        "cpu_report_sha256": checksum(cpu / "report.json"),
        "gpu_report_sha256": checksum(gpu / "report.json"),
        "cross_version_additive_qc_fields_excluded": sorted(QC_FIELDS),
        "scope": (
            "controlled GPU complete workload vs stored CPU oracle; no CPU/GPU "
            "timing equivalence claim across schema versions"
        ),
        "comparisons": checked,
        "gpu_database_audit": audit(gpu),
    }


def main() -> None:
    """Write evidence only after every comparison and stored sample passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.cpu, args.gpu)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Checked {len(result['comparisons'])} CPU/GPU comparisons.")


if __name__ == "__main__":
    main()
