"""Frozen SQLite v5 migration for historical source-extraction links."""

import json
from typing import Any

from sqlalchemy.engine import Connection


def migrate_source_links(connection: Connection) -> None:
    """Resolve unique preceding sources and audit missing/ambiguous histories."""
    connection.exec_driver_sql("""CREATE TABLE IF NOT EXISTS migration_issue (
        id INTEGER NOT NULL PRIMARY KEY,
        analysis_result_id INTEGER REFERENCES analysis_result(id) ON DELETE CASCADE,
        code VARCHAR NOT NULL,
        details JSON,
        resolved BOOLEAN NOT NULL DEFAULT 0,
        UNIQUE(analysis_result_id, code)
    )""")
    columns = {
        row[1]
        for row in connection.exec_driver_sql("PRAGMA table_info(analysis_result)")
    }
    if not columns:
        return
    for name, datatype in (
        (
            "source_extraction_result_id",
            "INTEGER REFERENCES analysis_result(id) ON DELETE SET NULL",
        ),
        ("legacy_trace_resolution", "VARCHAR"),
    ):
        if name not in columns:
            connection.exec_driver_sql(
                f"ALTER TABLE analysis_result ADD COLUMN {name} {datatype}"
            )
    connection.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_analysis_result_source_extraction_result_id "
        "ON analysis_result(source_extraction_result_id)"
    )
    connection.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_migration_issue_analysis_result_id "
        "ON migration_issue(analysis_result_id)"
    )
    results = {
        row["id"]: dict(row)
        for row in connection.exec_driver_sql(
            "SELECT * FROM analysis_result"
        ).mappings()
    }

    def positions(row: dict[str, Any], name: str) -> set[int]:
        return set(json.loads(row.get(name) or "null") or [])

    def audit(result_id: int, code: str, details: dict[str, Any]) -> None:
        connection.exec_driver_sql(
            "INSERT INTO migration_issue (analysis_result_id, code, details, resolved) "
            "VALUES (?, ?, ?, 0)",
            (result_id, code, json.dumps(details, sort_keys=True)),
        )

    def verify_copies(result_id: int, source_id: int) -> list[tuple[int, int]] | None:
        """Verify stored payloads before assigning historical extraction provenance."""
        pairs = []

        def spikes(trace_id: int) -> list[Any]:
            return [
                (json.loads(child[0]), child[1], child[2])
                for child in connection.exec_driver_sql(
                    'SELECT s."values", s.valid_start, s.valid_stop FROM spike_trace s '
                    "JOIN spike_inference_run r ON r.id = s.spike_inference_run_id "
                    "WHERE s.trace_id = ? AND r.method = 'oasis'",
                    (trace_id,),
                )
            ]

        traces = connection.exec_driver_sql(
            "SELECT * FROM trace WHERE analysis_result_id = ?", (result_id,)
        ).mappings()
        for trace in traces:
            originals = (
                connection.exec_driver_sql(
                    "SELECT * FROM trace WHERE analysis_result_id = ? AND roi_id IS ?",
                    (source_id, trace["roi_id"]),
                )
                .mappings()
                .all()
            )
            matches = []
            for original in originals:
                same = all(
                    json.loads(trace.get(name) or "null")
                    == json.loads(original.get(name) or "null")
                    for name in (
                        "raw_trace",
                        "corrected_trace",
                        "neuropil_trace",
                        "dff",
                        "den_dff",
                        "inferred_spikes",
                        "x_axis",
                    )
                )
                same = same and all(
                    trace.get(name) == original.get(name)
                    for name in (
                        "x_axis_units",
                        "source_start_frame",
                        "source_start_time_ms",
                        "original_frame_count",
                        "discarded_duration_ms",
                    )
                )
                same = same and spikes(trace["id"]) == spikes(original["id"])
                if same:
                    matches.append(original["id"])
            if len(matches) != 1:
                return None
            pairs.append((trace["id"], matches[0]))
        return pairs

    for result_id, row in results.items():
        if row.get("source_extraction_result_id") is not None:
            continue
        stored_sources = {
            item[0]
            for item in connection.exec_driver_sql(
                "SELECT DISTINCT w.extraction_result_id FROM trace t "
                "JOIN extraction_frame_window w ON w.id = t.extraction_frame_window_id "
                "WHERE t.analysis_result_id = ? AND w.extraction_result_id IS NOT NULL",
                (result_id,),
            )
        }
        if len(stored_sources) > 1:
            connection.exec_driver_sql(
                "UPDATE analysis_result SET "
                "legacy_trace_resolution = 'multiple_sources' "
                "WHERE id = ?",
                (result_id,),
            )
            audit(
                result_id,
                "multiple_extraction_sources",
                {"source_result_ids": sorted(stored_sources)},
            )
            continue
        if len(stored_sources) == 1 and stored_sources != {result_id}:
            source_id = next(iter(stored_sources))
            resolution = "stored_provenance"
            pairs: list[tuple[int, int]] = []
        elif positions(row, "positions_extracted"):
            source_id, resolution = result_id, "legacy_stage_inferred"
            pairs = []
        elif positions(row, "positions_analyzed") and row.get("extraction_settings_id"):
            required = positions(row, "positions_analyzed")
            candidates = [
                candidate
                for candidate in results.values()
                if candidate["id"] != result_id
                and candidate.get("created_at") is not None
                and row.get("created_at") is not None
                and candidate["created_at"] <= row["created_at"]
                and all(
                    candidate.get(name) == row.get(name)
                    for name in (
                        "experiment",
                        "detection_settings_id",
                        "extraction_settings_id",
                    )
                )
                and required <= positions(candidate, "positions_extracted")
            ]
            latest = max((r["created_at"] for r in candidates), default=None)
            candidates = [r for r in candidates if r["created_at"] == latest]
            if len(candidates) != 1:
                resolution = (
                    "unresolved_missing" if not candidates else "unresolved_ambiguous"
                )
                connection.exec_driver_sql(
                    "UPDATE analysis_result SET legacy_trace_resolution = ? "
                    "WHERE id = ?",
                    (resolution, result_id),
                )
                audit(
                    result_id,
                    resolution,
                    {
                        "candidate_result_ids": sorted(r["id"] for r in candidates),
                        "required_positions": sorted(required),
                    },
                )
                continue
            source_id = candidates[0]["id"]
            verified = verify_copies(result_id, source_id)
            if verified is None:
                connection.exec_driver_sql(
                    "UPDATE analysis_result SET legacy_trace_resolution = "
                    "'unresolved_payload_mismatch' WHERE id = ?",
                    (result_id,),
                )
                audit(
                    result_id,
                    "unresolved_payload_mismatch",
                    {"candidate_result_id": source_id},
                )
                continue
            pairs = verified
            resolution = "latest_preceding_inferred"
        else:
            continue
        connection.exec_driver_sql(
            "UPDATE analysis_result SET source_extraction_result_id = ?, "
            "legacy_trace_resolution = ? WHERE id = ?",
            (source_id, resolution, result_id),
        )
        for copied_id, original_id in pairs:
            connection.exec_driver_sql(
                "UPDATE trace SET extraction_frame_window_id = "
                "(SELECT extraction_frame_window_id FROM trace WHERE id = ?) "
                "WHERE id = ?",
                (original_id, copied_id),
            )
            connection.exec_driver_sql(
                "UPDATE spike_trace SET spike_inference_run_id = "
                "(SELECT spike_inference_run_id FROM spike_trace WHERE trace_id = ? "
                "AND spike_inference_run_id IN (SELECT id FROM spike_inference_run "
                "WHERE method = 'oasis')) WHERE trace_id = ? AND "
                "spike_inference_run_id IN (SELECT id FROM spike_inference_run "
                "WHERE method = 'oasis') AND EXISTS (SELECT 1 FROM spike_trace "
                "WHERE trace_id = ?)",
                (original_id, copied_id, original_id),
            )
    missing = connection.exec_driver_sql(
        "SELECT COUNT(*) FROM analysis_result a LEFT JOIN analysis_result s "
        "ON s.id = a.source_extraction_result_id "
        "WHERE a.source_extraction_result_id IS NOT NULL AND s.id IS NULL"
    ).scalar_one()
    if missing:
        raise ValueError("Invalid source-extraction references in backfill.")
    # Validate links this migration writes, without rejecting unrelated legacy
    # parent inconsistencies (which have their own audit/repair paths).
    for table in ("migration_issue", "spike_trace"):
        if connection.exec_driver_sql(f"PRAGMA foreign_key_check({table})").all():
            raise ValueError(f"Invalid foreign keys in {table} source-link backfill.")
