"""Verify and time schema-10/12/13 upgrades on copied files.

Run after benchmark_cascade_storage.py, without competing measurement processes.
All original databases/exports remain untouched. Database fingerprints replace only
spike-array encoding with decoded float64 bytes; every original stored field is exact.
Schema-12's added coordinates are verified as NULL for schema-10 sources; existing
schema-12 coordinates remain part of the complete field fingerprint.
New schema-14 noise QC fields are separately verified as unknown, not backfilled.
Event exports may add threshold mode/units columns; all preexisting fields must match.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sqlite3
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path
from urllib.parse import quote

import numpy as np
from sqlmodel import Session, select

from cali._constants import CASCADE_EXPECTED_SPIKES_TRACES, INFERRED_SPIKES_TRACES
from cali.sqlmodel import SpikeTrace, create_cali_engine
from cali.sqlmodel._engine import SCHEMA_VERSION
from cali.sqlmodel._trace_array_codec import (
    _decode_blob,
    decode_trace_array,
    trace_array_storage_info,
)
from cali.sqlmodel._trace_array_migration import migrate_trace_array_shuffle
from cali.util._database_to_csv import export_traces_to_csv


def fingerprint(
    path: Path,
    *,
    ignore_population_coordinates: bool = False,
    ignore_noise_qc: bool = False,
) -> str:
    """Hash every stored field, normalizing only the canonical array encoding."""
    digest = hashlib.sha256()
    uri = "file:" + quote(str(path.resolve()), safe="/") + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        tables = sorted(
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        )
        for name in tables:
            quoted = '"' + name.replace('"', '""') + '"'
            columns = [
                row[1]
                for row in connection.execute(f"PRAGMA table_info({quoted})")
                if not (
                    ignore_population_coordinates
                    and name == "spike_fov_analysis"
                    and row[1] in {"valid_start", "valid_stop", "frame_rate_hz"}
                )
                and not (
                    ignore_noise_qc
                    and row[1]
                    in {
                        "data_analysis": {"calcium_noise"},
                        "fov_analysis": {
                            "calcium_noise_median",
                            "calcium_noise_iqr",
                            "calcium_noise_roi_count",
                        },
                        "spike_fov_analysis": {
                            "model_noise_median",
                            "model_noise_iqr",
                            "model_noise_roi_count",
                        },
                    }.get(name, set())
                )
            ]
            selected = ",".join('"' + col.replace('"', '""') + '"' for col in columns)
            cursor = connection.execute(
                f"SELECT {selected} FROM {quoted} ORDER BY rowid"
            )
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


def compare_exports(expected_dir: Path, actual_dir: Path) -> dict:
    """Check exact exports, permitting only the additive event threshold columns."""
    expected = export_hashes(expected_dir)
    actual = export_hashes(actual_dir)
    assert expected and actual.keys() == expected.keys(), "Exported files differ."
    added_columns: list[str] = []
    for name, digest in expected.items():
        if actual[name] == digest:
            continue
        assert Path(name).name == "events.csv", f"Migrated export differs: {name}"
        with (expected_dir / name).open(newline="") as original:
            old_rows = list(csv.reader(original))
        with (actual_dir / name).open(newline="") as migrated:
            new_rows = list(csv.reader(migrated))
        old_header, new_header = old_rows[0], new_rows[0]
        assert len(set(new_header)) == len(new_header), "Duplicate event columns."
        assert all(column in new_header for column in old_header)
        added = [column for column in new_header if column not in old_header]
        assert set(added) == {"threshold_mode", "threshold_units"}
        assert all(len(row) == len(new_header) for row in new_rows[1:])
        indices = [new_header.index(column) for column in old_header]
        assert [[row[index] for index in indices] for row in new_rows] == old_rows
        added_columns = added
    return {
        "all_csv_files_byte_identical": all(
            actual[name] == digest
            for name, digest in expected.items()
            if Path(name).suffix == ".csv"
        ),
        "all_existing_csv_fields_exact": True,
        "event_columns_added": added_columns,
        "all_json_sidecar_contents_identical": True,
        "exported_files": len(actual),
    }


def main() -> None:
    """Upgrade a copy, compare every field/export and record maintenance separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-exports", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / args.source.name
    if destination.exists():
        raise FileExistsError(destination)
    uri = "file:" + quote(str(args.source.resolve()), safe="/") + "?mode=ro"
    with (
        sqlite3.connect(uri, uri=True) as original,
        sqlite3.connect(destination) as copy,
    ):
        version = original.execute("PRAGMA user_version").fetchone()[0]
        if version not in (10, 12, 13):
            raise ValueError("This measurement requires a schema-10/12/13 file.")
        original.backup(copy)
        run_id = copy.execute("SELECT id FROM analysis_result").fetchone()[0]
    before = fingerprint(
        destination, ignore_population_coordinates=version < 12, ignore_noise_qc=True
    )
    begin = time.perf_counter()
    engine = create_cali_engine(f"sqlite:///{destination}")
    migration_s = time.perf_counter() - begin
    assert (
        fingerprint(
            destination,
            ignore_population_coordinates=version < 12,
            ignore_noise_qc=True,
        )
        == before
    )
    if version >= 12:
        with (
            sqlite3.connect(uri, uri=True) as original,
            sqlite3.connect(destination) as migrated,
        ):
            for identifier, payload in original.execute(
                'SELECT id,"values" FROM spike_trace'
            ):
                old_array, _ = _decode_blob(payload)
                stored = migrated.execute(
                    'SELECT "values" FROM spike_trace WHERE id=?', (identifier,)
                ).fetchone()[0]
                new_array, _ = _decode_blob(stored)
                assert old_array.dtype == new_array.dtype
                assert old_array.tobytes() == new_array.tobytes()
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
    export_comparison = compare_exports(
        args.expected_exports or args.source.with_name(args.source.stem + "_exports"),
        destination.with_name(destination.stem + "_exports"),
    )
    before_vacuum_bytes = destination.stat().st_size
    with sqlite3.connect(destination) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        for table, columns in (
            ("data_analysis", ("calcium_noise",)),
            (
                "fov_analysis",
                (
                    "calcium_noise_median",
                    "calcium_noise_iqr",
                    "calcium_noise_roi_count",
                ),
            ),
            (
                "spike_fov_analysis",
                ("model_noise_median", "model_noise_iqr", "model_noise_roi_count"),
            ),
        ):
            condition = " OR ".join(f"{column} IS NOT NULL" for column in columns)
            assert (
                connection.execute(
                    f"SELECT count(*) FROM {table} WHERE {condition}"
                ).fetchone()[0]
                == 0
            )
        assert version >= 12 or (
            connection.execute(
                "SELECT COUNT(*) FROM spike_fov_analysis WHERE valid_start IS NOT NULL "
                "OR valid_stop IS NOT NULL OR frame_rate_hz IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM spike_trace WHERE typeof(\"values\")='blob'"
            ).fetchone()[0]
            == row_count
        )
        payload_bytes: dict[str, int] = {}
        encodings: dict[str, int] = {}
        legacy_json_bytes = 0
        for method, payload in connection.execute(
            'SELECT i.method, s."values" FROM spike_trace s '
            "JOIN spike_inference_run i ON i.id=s.spike_inference_run_id"
        ):
            metadata = trace_array_storage_info(payload)
            assert metadata["version"] == 2
            payload_bytes[method] = payload_bytes.get(method, 0) + len(payload)
            key = f"{method}/{metadata['dtype']}/{metadata['compression']}"
            encodings[key] = encodings.get(key, 0) + 1
            if method == "oasis":
                legacy_json_bytes += len(
                    json.dumps(decode_trace_array(payload)).encode()
                )
        fov_count = connection.execute("SELECT count(*) FROM fov").fetchone()[0]
        begin = time.perf_counter()
        connection.execute("VACUUM")
        maintenance_s = time.perf_counter() - begin
    report = {
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "migration_source_sha256": hashlib.sha256(
            Path(inspect.getfile(migrate_trace_array_shuffle)).read_bytes()
        ).hexdigest(),
        "rows": row_count,
        "schema_before": version,
        "schema_after": SCHEMA_VERSION,
        "original_blob_dtype_and_bits_exact": version >= 12,
        "new_noise_qc_fields_unknown": True,
        "environment_versions": {
            "python": sys.version,
            "cali": package_version("cali"),
            "numpy": package_version("numpy"),
            "sqlalchemy": package_version("sqlalchemy"),
        },
        "database_fingerprint": before,
        "all_decoded_arrays_and_other_fields_exact": True,
        **export_comparison,
        "source_bytes": args.source.stat().st_size,
        "migrated_bytes_before_vacuum": before_vacuum_bytes,
        "migrated_bytes_after_vacuum": destination.stat().st_size,
        "migration_s": migration_s,
        "orm_read_s": read_s,
        "export_s": export_s,
        "vacuum_s": maintenance_s,
        "canonical_payload_bytes": payload_bytes,
        "encodings": encodings,
        "hypothetical_legacy_oasis_json_bytes": legacy_json_bytes,
        "projected_96_fov_with_legacy_oasis_mib": (
            (sum(payload_bytes.values()) + legacy_json_bytes) / fov_count * 96 / 1024**2
        ),
        "budget_mib": 512,
        "within_storage_budget": (
            (sum(payload_bytes.values()) + legacy_json_bytes) / fov_count * 96
            <= 512 * 1024**2
        ),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "codec_source_sha256": hashlib.sha256(
            Path(inspect.getfile(decode_trace_array)).read_bytes()
        ).hexdigest(),
    }
    (args.output_dir / "migration-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
