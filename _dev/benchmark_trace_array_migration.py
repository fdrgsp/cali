"""Verify and time production schema-10 to schema-11 upgrades on copied files.

Run after benchmark_cascade_storage.py, without competing measurement processes.
All original databases/exports remain untouched. Database fingerprints replace only
spike-array encoding with decoded float64 bytes; every other stored field is exact.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from sqlmodel import Session, select

from cali._constants import CASCADE_EXPECTED_SPIKES_TRACES, INFERRED_SPIKES_TRACES
from cali.sqlmodel import SpikeTrace, create_cali_engine
from cali.sqlmodel._trace_array_codec import decode_trace_array
from cali.sqlmodel._trace_array_migration import migrate_trace_arrays
from cali.util._database_to_csv import export_traces_to_csv


def fingerprint(path: Path) -> str:
    """Hash every stored field, normalizing only the canonical array encoding."""
    digest = hashlib.sha256()
    with sqlite3.connect(path) as connection:
        tables = sorted(
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        )
        for name in tables:
            quoted = '"' + name.replace('"', '""') + '"'
            cursor = connection.execute(f"SELECT * FROM {quoted} ORDER BY rowid")
            columns = [item[0] for item in cursor.description]
            digest.update(repr((name, columns)).encode())
            for row in cursor:
                values = list(row)
                if name == "spike_trace":
                    index = columns.index("values")
                    values[index] = np.asarray(
                        decode_trace_array(values[index]), dtype="<f8"
                    ).tobytes()
                digest.update(repr(tuple(values)).encode())
    return digest.hexdigest()


def export_hashes(directory: Path) -> dict[str, str]:
    """Hash exact CSV bytes and JSON content (object key order is immaterial)."""
    return {
        str(path.relative_to(directory)): hashlib.sha256(
            json.dumps(
                json.loads(path.read_text()), sort_keys=True, separators=(",", ":")
            ).encode()
            if path.suffix == ".json"
            else path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    """Upgrade a copy, compare every field/export and record maintenance separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / args.source.name
    if destination.exists():
        raise FileExistsError(destination)
    with sqlite3.connect(args.source) as original, sqlite3.connect(destination) as copy:
        version = original.execute("PRAGMA user_version").fetchone()[0]
        if version != 10:
            raise ValueError("This measurement requires an actual schema-10 file.")
        original.backup(copy)
        run_id = copy.execute("SELECT id FROM analysis_result").fetchone()[0]
    before = fingerprint(destination)
    begin = time.perf_counter()
    engine = create_cali_engine(f"sqlite:///{destination}")
    migration_s = time.perf_counter() - begin
    assert fingerprint(destination) == before
    begin = time.perf_counter()
    with Session(engine) as session:
        rows = session.exec(select(SpikeTrace)).all()
        read_s = time.perf_counter() - begin
        methods = {row.inference_run.method for row in rows}
        row_count = len(rows)
    begin = time.perf_counter()
    export_traces_to_csv(
        engine,
        {
            INFERRED_SPIKES_TRACES: "oasis" in methods,
            CASCADE_EXPECTED_SPIKES_TRACES: "cascade" in methods,
        },
        run_id,
        destination,
    )
    export_s = time.perf_counter() - begin
    engine.dispose()
    expected = export_hashes(args.source.with_name(args.source.stem + "_exports"))
    actual = export_hashes(destination.with_name(destination.stem + "_exports"))
    assert expected and actual == expected, "Migrated exports differ from original."
    before_vacuum_bytes = destination.stat().st_size
    with sqlite3.connect(destination) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 11
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM spike_trace WHERE typeof(\"values\")='blob'"
            ).fetchone()[0]
            == row_count
        )
        begin = time.perf_counter()
        connection.execute("VACUUM")
        maintenance_s = time.perf_counter() - begin
    report = {
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "migration_source_sha256": hashlib.sha256(
            Path(inspect.getfile(migrate_trace_arrays)).read_bytes()
        ).hexdigest(),
        "rows": row_count,
        "schema_before": version,
        "schema_after": 11,
        "database_fingerprint": before,
        "all_decoded_arrays_and_other_fields_exact": True,
        "all_csv_files_byte_identical": True,
        "all_json_sidecar_contents_identical": True,
        "exported_files": len(actual),
        "source_bytes": args.source.stat().st_size,
        "migrated_bytes_before_vacuum": before_vacuum_bytes,
        "migrated_bytes_after_vacuum": destination.stat().st_size,
        "migration_s": migration_s,
        "orm_read_s": read_s,
        "export_s": export_s,
        "vacuum_s": maintenance_s,
    }
    (args.output_dir / "migration-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
