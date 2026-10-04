"""Audit stored noise QC and compare controlled runs with pre-QC science oracles.

Use completed all-case benchmark directories. SQLite inputs remain read-only;
summaries use every selected ROI, including inactive ones. This is correctness
evidence, not biological quality acceptance or a performance measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from urllib.parse import quote

import numpy as np
from audit_cascade_extraction import audit, checksum

QC_FIELDS = {
    "calcium_noise",
    "calcium_noise_median",
    "calcium_noise_iqr",
    "calcium_noise_roi_count",
    "model_noise_median",
    "model_noise_iqr",
    "model_noise_roi_count",
}
INPUT_FIELDS = (
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
    "device",
    "torch",
    "torch_threads",
    "oasis_decay_constant_s",
    "chunk_windows",
    "queue_capacity",
)


def without_qc(value: object) -> object:
    """Exclude only newly additive QC metadata; retain every original metric."""
    if isinstance(value, dict):
        return {
            key: without_qc(item) for key, item in value.items() if key not in QC_FIELDS
        }
    if isinstance(value, list):
        return [without_qc(item) for item in value]
    return value


def summary(values: list[float | None]) -> tuple:
    """Independently recompute the documented linear median/IQR and known count."""
    known = [
        value
        for value in values
        if value is not None and math.isfinite(value) and value >= 0
    ]
    if not known:
        return None, None, 0
    q1, median, q3 = np.percentile(known, [25, 50, 75], method="linear")
    return float(median), float(q3 - q1), len(known)


def compare(before: Path, after: Path) -> dict:
    """Require matching workload inputs, original scientific fields and stored QC."""
    old = json.loads((before / "report.json").read_text())
    new = json.loads((after / "report.json").read_text())
    assert all(new["deterministic_scientific_parity"].values())
    previous = {(case["mode"], case["backend"]): case for case in old["results"]}
    current = {(case["mode"], case["backend"]): case for case in new["results"]}
    assert current.keys() == previous.keys()
    comparisons = {}
    for identity, case in current.items():
        baseline = previous[identity]
        assert all(case[key] == baseline[key] for key in INPUT_FIELDS)
        assert len(case["phases"]) == len(baseline["phases"])
        for phase, old_phase in zip(case["phases"], baseline["phases"], strict=True):
            assert (
                phase["phase"] == old_phase["phase"]
                and phase["fovs"] == old_phase["fovs"]
            )
            stem = f"{identity[0]}-{identity[1]}-{phase['phase']}"
            original = json.loads((before / f"{stem}-metrics.json").read_text())
            measured = json.loads((after / f"{stem}-metrics.json").read_text())
            assert without_qc(measured) == without_qc(original), stem
            with (
                np.load(before / f"{stem}.npz", allow_pickle=False) as old_arrays,
                np.load(after / f"{stem}.npz", allow_pickle=False) as new_arrays,
            ):
                assert old_arrays.files == new_arrays.files
                for key in old_arrays.files:
                    np.testing.assert_array_equal(old_arrays[key], new_arrays[key])
            database = after / f"{stem}.cali"
            uri = "file:" + quote(str(database.resolve()), safe="/") + "?mode=ro"
            checked = []
            with sqlite3.connect(uri, uri=True) as connection:
                assert connection.execute("PRAGMA user_version").fetchone()[0] == 14
                for parent, fov, owner, median, iqr, count in connection.execute(
                    "SELECT id,fov_id,analysis_result_id,calcium_noise_median,"
                    "calcium_noise_iqr,calcium_noise_roi_count FROM fov_analysis"
                ):
                    calcium = connection.execute(
                        "SELECT a.calcium_noise,t.calcium_noise FROM data_analysis a "
                        "JOIN roi r ON r.id=a.roi_id JOIN trace t ON t.roi_id=a.roi_id "
                        "AND t.analysis_result_id=a.analysis_result_id "
                        "WHERE r.fov_id=? AND a.analysis_result_id=?",
                        (fov, owner),
                    ).fetchall()
                    assert len(calcium) == case["rois_per_fov"]
                    assert all(used == extracted for used, extracted in calcium)
                    assert (median, iqr, count) == summary([row[0] for row in calcium])
                    methods = {}
                    for method, run, median, iqr, count in connection.execute(
                        "SELECT method,spike_inference_run_id,model_noise_median,"
                        "model_noise_iqr,model_noise_roi_count FROM spike_fov_analysis "
                        "WHERE fov_analysis_id=?",
                        (parent,),
                    ):
                        if method == "oasis":
                            assert (median, iqr, count) == (None, None, None)
                        else:
                            assert method == "cascade"
                            noises = [
                                row[0]
                                for row in connection.execute(
                                    "SELECT s.noise FROM spike_trace s "
                                    "JOIN trace t ON t.id=s.trace_id "
                                    "JOIN roi r ON r.id=t.roi_id WHERE r.fov_id=? "
                                    "AND s.spike_inference_run_id=?",
                                    (fov, run),
                                )
                            ]
                            assert len(noises) == case["rois_per_fov"]
                            assert (median, iqr, count) == summary(noises)
                        methods[method] = {
                            "median": median,
                            "iqr": iqr,
                            "known_rois": count,
                        }
                    checked.append(
                        {
                            "fov_id": fov,
                            "calcium": summary([row[0] for row in calcium]),
                            "spikes": methods,
                        }
                    )
            assert len(checked) == phase["fovs"]
            comparisons[stem] = {
                "all_original_scientific_fields_exact": True,
                "all_source_arrays_exact": True,
                "all_stored_qc_matches_selected_inputs": True,
                "fovs": checked,
            }
    return {
        "script_sha256": checksum(Path(__file__)),
        "before_report_sha256": checksum(before / "report.json"),
        "after_report_sha256": checksum(after / "report.json"),
        "additive_fields_excluded_from_original_science_comparison": sorted(QC_FIELDS),
        "scope": (
            "Controlled pre/post QC input/science equality and independent "
            "read-only stored-noise summaries"
        ),
        "comparisons": comparisons,
        "independent_database_audit": audit(after),
    }


def main() -> None:
    """Write an audit only after every source, metric and QC comparison passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Checked {len(result['comparisons'])} original science/QC comparisons.")


if __name__ == "__main__":
    main()
