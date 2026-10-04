"""Independently audit completed controlled extraction benchmark databases.

Open SQLite read-only, check every requested row, and compare stored calcium and
decoded spike samples with the benchmark's independently saved NPZ arrays.
Accept a directory from either the all-case or single-case benchmark command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

import numpy as np

from cali.sqlmodel._trace_array_codec import decode_trace_array


def checksum(path: Path) -> str:
    """Hash complete files without retaining another database-sized buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def audit(directory: Path) -> dict:
    """Verify scientific arrays and canonical run ownership without migrating files."""
    report_path = directory / "report.json"
    if report_path.exists():
        cases = json.loads(report_path.read_text())["results"]
    else:
        cases = [
            value
            for path in sorted(directory.glob("*.json"))
            if "phases" in (value := json.loads(path.read_text()))
        ]
    if not cases:
        raise ValueError("No completed benchmark reports found.")
    audited = {}
    for case in cases:
        methods = ("oasis", "cascade") if case["mode"] == "dual" else (case["mode"],)
        for phase in case["phases"]:
            stem = f"{case['mode']}-{case['backend']}-{phase['phase']}"
            database = directory / f"{stem}.cali"
            expected_rois = phase["fovs"] * case["rois_per_fov"]
            expected_rows = {
                "fov": phase["fovs"],
                "roi": expected_rois,
                "trace": expected_rois,
                "data_analysis": expected_rois,
                "fov_analysis": phase["fovs"],
                "spike_trace": expected_rois * len(methods),
                "spike_inference_run": len(methods),
                "spike_analysis": (
                    expected_rois * len(methods) if case["analysis"] == "full" else 0
                ),
                "spike_fov_analysis": (
                    phase["fovs"] * len(methods) if case["analysis"] == "full" else 0
                ),
            }
            uri = "file:" + quote(str(database.resolve()), safe="/") + "?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                rows = {
                    table: connection.execute(
                        f"SELECT count(*) FROM {table}"
                    ).fetchone()[0]
                    for table in expected_rows
                }
                if rows != expected_rows:
                    raise ValueError(f"Incomplete {stem}: {rows} != {expected_rows}")
                runs = dict(
                    connection.execute("SELECT id, method FROM spike_inference_run")
                )
                if set(runs.values()) != set(methods):
                    raise ValueError(f"Incorrect inference methods in {stem}.")
                with np.load(directory / f"{stem}.npz", allow_pickle=False) as expected:
                    if set(expected.files) != {"den_dff", "calcium_noise", *methods}:
                        raise ValueError(f"Incorrect ground-truth array set in {stem}.")
                    for name in expected.files:
                        shape = (
                            (case["rois_per_fov"],)
                            if name == "calcium_noise"
                            else (case["rois_per_fov"], case["frames"])
                        )
                        if expected[name].shape != shape:
                            raise ValueError(f"Incorrect ground-truth shape in {stem}.")
                    traces = connection.execute(
                        "SELECT t.id, r.label_value, t.den_dff, t.calcium_noise, "
                        "t.analysis_result_id FROM trace t JOIN roi r ON r.id=t.roi_id"
                    )
                    for trace_id, label, den_dff, noise, owner in traces:
                        index = label - 1
                        np.testing.assert_array_equal(
                            json.loads(den_dff), expected["den_dff"][index]
                        )
                        np.testing.assert_array_equal(
                            noise, expected["calcium_noise"][index]
                        )
                        spikes = connection.execute(
                            'SELECT s."values", i.method, i.extraction_result_id '
                            "FROM spike_trace s JOIN spike_inference_run i "
                            "ON i.id=s.spike_inference_run_id WHERE s.trace_id=?",
                            (trace_id,),
                        ).fetchall()
                        if {method for _, method, _ in spikes} != set(methods):
                            raise ValueError(f"Missing trace method in {stem}/{label}.")
                        for values, method, inference_owner in spikes:
                            if inference_owner != owner:
                                raise ValueError(
                                    f"Incorrect inference owner in {stem}."
                                )
                            np.testing.assert_array_equal(
                                decode_trace_array(values), expected[method][index]
                            )
                fov_sources = connection.execute(
                    "SELECT s.spike_inference_run_id, i.extraction_result_id, "
                    "f.analysis_result_id FROM spike_fov_analysis s "
                    "JOIN spike_inference_run i ON i.id=s.spike_inference_run_id "
                    "JOIN fov_analysis f ON f.id=s.fov_analysis_id"
                )
                if any(
                    run not in runs or owner != parent
                    for run, owner, parent in fov_sources
                ):
                    raise ValueError(f"Incorrect FOV inference owner in {stem}.")
                schema = connection.execute("PRAGMA user_version").fetchone()[0]
            audited[stem] = {
                "rows": rows,
                "schema_version": schema,
                "database_bytes": database.stat().st_size,
                "database_sha256": checksum(database),
                "all_arrays_exact": True,
                "canonical_run_ownership": True,
            }
    return {
        "audit_script_sha256": checksum(Path(__file__)),
        "scope": "read-only SQLite row/ownership checks and every stored array sample",
        "databases": audited,
    }


def main() -> None:
    """Write an audit artifact only after every database passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Audited {len(result['databases'])} complete databases.")


if __name__ == "__main__":
    main()
