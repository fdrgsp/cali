"""Frozen SQLite v4 backfill; deliberately independent of ORM mappings."""

import json
from typing import Any

from sqlalchemy.engine import Connection


def migrate_trace_provenance(connection: Connection) -> None:
    """Copy legacy products exactly without inventing historical backend metadata."""
    connection.exec_driver_sql("""CREATE TABLE IF NOT EXISTS extraction_frame_window (
        id INTEGER NOT NULL PRIMARY KEY,
        extraction_result_id INTEGER REFERENCES analysis_result(id) ON DELETE SET NULL,
        fov_id INTEGER REFERENCES fov(id) ON DELETE SET NULL,
        legacy_owner_result_id INTEGER
            REFERENCES analysis_result(id) ON DELETE SET NULL,
        requested_discard_value FLOAT NOT NULL,
        requested_discard_unit VARCHAR NOT NULL,
        timing_source VARCHAR,
        conversion_rule VARCHAR NOT NULL,
        original_frame_count INTEGER,
        retained_frame_count INTEGER,
        source_start_frame INTEGER NOT NULL,
        source_start_time_ms FLOAT NOT NULL,
        source_time_origin_ms FLOAT,
        source_start_timestamp_ms FLOAT,
        discarded_duration_ms FLOAT NOT NULL,
        provenance_source VARCHAR NOT NULL,
        schema_version INTEGER NOT NULL,
        UNIQUE(extraction_result_id, fov_id)
    )""")
    connection.exec_driver_sql("""CREATE TABLE IF NOT EXISTS spike_inference_run (
        id INTEGER NOT NULL PRIMARY KEY,
        extraction_result_id INTEGER REFERENCES analysis_result(id) ON DELETE SET NULL,
        legacy_owner_result_id INTEGER
            REFERENCES analysis_result(id) ON DELETE SET NULL,
        method VARCHAR NOT NULL,
        units VARCHAR NOT NULL,
        backend_package VARCHAR NOT NULL,
        backend_version VARCHAR,
        backend_revision VARCHAR,
        resolved_model VARCHAR,
        catalogue_revision VARCHAR,
        config_sha256 VARCHAR,
        weights_manifest_sha256 VARCHAR,
        resolved_device VARCHAR,
        dtype VARCHAR,
        model_sampling_rate_hz FLOAT,
        smoothing_sigma FLOAT,
        kernel_type VARCHAR,
        provenance_source VARCHAR NOT NULL,
        schema_version INTEGER NOT NULL,
        UNIQUE(extraction_result_id, method)
    )""")
    connection.exec_driver_sql("""CREATE TABLE IF NOT EXISTS spike_trace (
        id INTEGER NOT NULL PRIMARY KEY,
        trace_id INTEGER NOT NULL REFERENCES trace(id) ON DELETE CASCADE,
        spike_inference_run_id INTEGER NOT NULL
            REFERENCES spike_inference_run(id) ON DELETE CASCADE,
        "values" JSON NOT NULL,
        valid_start INTEGER NOT NULL,
        valid_stop INTEGER,
        noise FLOAT,
        selected_noise_level FLOAT,
        ar_coefficients JSON,
        UNIQUE(trace_id, spike_inference_run_id)
    )""")
    columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(trace)")}
    if not columns:
        return
    if "id" not in columns:
        raise ValueError("Cannot migrate traces: missing primary-key column 'id'.")
    if "extraction_frame_window_id" not in columns:
        connection.exec_driver_sql(
            "ALTER TABLE trace ADD COLUMN extraction_frame_window_id "
            "INTEGER REFERENCES extraction_frame_window(id)"
        )
    connection.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_trace_extraction_frame_window_id "
        "ON trace(extraction_frame_window_id)"
    )

    def rows(table: str) -> dict[int | None, dict[str, Any]]:
        if not connection.exec_driver_sql(f"PRAGMA table_info({table})").all():
            return {}
        return {
            row["id"]: dict(row)
            for row in connection.exec_driver_sql(f"SELECT * FROM {table}").mappings()
        }

    results, settings, rois, fovs = (
        rows("analysis_result"),
        rows("extraction_settings"),
        rows("roi"),
        rows("fov"),
    )
    runs: dict[tuple, int] = {}
    windows: dict[tuple, tuple[int, dict]] = {}

    def ownership(owner_id: int | None) -> tuple[int | None, int | None, str]:
        owner = results.get(owner_id, {})
        if not owner:
            return None, None, "legacy_unlinked"
        extracted = json.loads(owner.get("positions_extracted") or "null")
        analyzed = json.loads(owner.get("positions_analyzed") or "null")
        if extracted:
            return owner_id, None, "legacy_import"
        if analyzed or owner.get("extraction_settings_id") is None:
            # Source resolution/audits are a later P2b migration. Preserve the
            # owner for that audit, but never mislabel a copied analysis as extraction.
            return None, owner_id, "legacy_analysis_copy_unresolved"
        return owner_id, None, "legacy_trace_owner"

    def inference_run(owner_id: int | None, trace_id: int | None = None) -> int:
        extraction_id, legacy_id, source = ownership(owner_id)
        key = (extraction_id, legacy_id, trace_id if owner_id not in results else None)
        if key in runs:
            return runs[key]
        existing = (
            connection.exec_driver_sql(
                "SELECT id FROM spike_inference_run WHERE extraction_result_id IS ? "
                "AND legacy_owner_result_id IS ? AND method = 'oasis'",
                (extraction_id, legacy_id),
            ).all()
            if extraction_id is not None or legacy_id is not None
            else []
        )
        if len(existing) > 1:
            raise ValueError("Duplicate legacy OASIS inference runs.")
        run_id: int | None
        if existing:
            run_id = existing[0][0]
        else:
            run_id = connection.exec_driver_sql(
                "INSERT INTO spike_inference_run (extraction_result_id, "
                "legacy_owner_result_id, method, units, backend_package, "
                "provenance_source, schema_version) VALUES (?, ?, 'oasis', "
                "'a.u.', 'oasis-deconv', ?, 1)",
                (extraction_id, legacy_id, source),
            ).lastrowid
        assert run_id is not None
        runs[key] = run_id
        return run_id

    # Even an extraction without valid ROI outputs owns its historical OASIS run.
    for owner_id, owner in results.items():
        if json.loads(owner.get("positions_extracted") or "null"):
            inference_run(owner_id)

    # Stream large legacy arrays instead of materializing the entire plate in memory.
    for trace in connection.exec_driver_sql("SELECT * FROM trace").mappings():
        trace_id, owner_id = trace["id"], trace.get("analysis_result_id")
        extraction_id, legacy_id, source = ownership(owner_id)
        roi = rois.get(trace.get("roi_id"), {})
        fov_id = roi.get("fov_id")
        if fov_id not in fovs:
            fov_id = None
        arrays = [
            json.loads(trace[name])
            for name in (
                "raw_trace",
                "dff",
                "den_dff",
                "dec_dff",
                "x_axis",
                "inferred_spikes",
            )
            if trace.get(name) is not None
        ]
        arrays = [array for array in arrays if array is not None]
        lengths = {len(array) for array in arrays}
        if len(lengths) > 1:
            raise ValueError(f"Trace {trace_id}: inconsistent legacy array lengths.")
        retained = next(iter(lengths), None)
        start = trace.get("source_start_frame", 0)
        original = trace.get("original_frame_count")
        if original is None and retained is not None:
            original = retained + start
        if (
            original is not None
            and retained is not None
            and original != retained + start
        ):
            raise ValueError(f"Trace {trace_id}: inconsistent source-frame counts.")
        request = settings.get(
            results.get(owner_id, {}).get("extraction_settings_id"), {}
        )
        axis = json.loads(trace.get("x_axis") or "null")
        origin = (
            axis[0]
            if not start and axis and trace.get("x_axis_units") == "ms"
            else None
        )
        window = {
            "extraction_result_id": extraction_id,
            "fov_id": fov_id,
            "legacy_owner_result_id": legacy_id,
            "requested_discard_value": request.get("discard_initial_value", 0.0),
            "requested_discard_unit": request.get("discard_initial_unit", "frames"),
            "timing_source": trace.get("discard_timing_source"),
            "conversion_rule": "legacy_source_fields"
            if start
            else "legacy_zero_discard",
            "original_frame_count": original,
            "retained_frame_count": retained,
            "source_start_frame": start,
            "source_start_time_ms": trace.get("source_start_time_ms", 0.0),
            "source_time_origin_ms": origin,
            "source_start_timestamp_ms": origin,
            "discarded_duration_ms": trace.get("discarded_duration_ms", 0.0),
            "provenance_source": source,
            "schema_version": 1,
        }
        key = (
            extraction_id,
            legacy_id,
            fov_id,
            trace_id if fov_id is None or owner_id not in results else None,
        )
        if key in windows:
            window_id, old = windows[key]
            if old != window:
                raise ValueError(
                    "Conflicting legacy frame windows for one extraction/FOV."
                )
        else:
            names = ", ".join(window)
            placeholders = ", ".join("?" for _ in window)
            window_id = connection.exec_driver_sql(
                f"INSERT INTO extraction_frame_window ({names}) "
                f"VALUES ({placeholders})",
                tuple(window.values()),
            ).lastrowid
            assert window_id is not None
            windows[key] = (window_id, window)
        connection.exec_driver_sql(
            "UPDATE trace SET extraction_frame_window_id = ? WHERE id = ?",
            (window_id, trace_id),
        )
        values = json.loads(trace.get("inferred_spikes") or "null")
        if values is not None:
            run_id = inference_run(owner_id, trace_id)
            existing = connection.exec_driver_sql(
                'SELECT "values" FROM spike_trace WHERE trace_id = ? '
                "AND spike_inference_run_id = ?",
                (trace_id, run_id),
            ).all()
            if existing:
                if len(existing) != 1 or json.loads(existing[0][0]) != values:
                    raise ValueError(
                        "Legacy spike backfill failed exact-value verification."
                    )
            else:
                connection.exec_driver_sql(
                    "INSERT INTO spike_trace "
                    '(trace_id, spike_inference_run_id, "values", '
                    "valid_start) VALUES (?, ?, ?, 0)",
                    (trace_id, run_id, trace["inferred_spikes"]),
                )
            copied = connection.exec_driver_sql(
                'SELECT "values" FROM spike_trace WHERE trace_id = ? '
                "AND spike_inference_run_id = ?",
                (trace_id, run_id),
            ).scalar_one()
            if json.loads(copied) != values:
                raise ValueError(
                    "Legacy spike backfill failed exact-value verification."
                )

    for table in ("extraction_frame_window", "spike_inference_run", "spike_trace"):
        if connection.exec_driver_sql(f"PRAGMA foreign_key_check({table})").all():
            raise ValueError(f"Invalid foreign keys in {table} backfill.")
    missing = connection.exec_driver_sql(
        "SELECT COUNT(*) FROM trace WHERE extraction_frame_window_id IS NULL"
    ).scalar_one()
    if missing:
        raise ValueError("Frame-window backfill missed a legacy trace.")
