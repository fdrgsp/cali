"""Frozen, model-free SQLite v8 audit of ambiguous legacy extraction flags."""

import json
from typing import Any

from sqlalchemy.engine import Connection


def audit_legacy_stage_sources(connection: Connection) -> None:
    """Withdraw guessed ownership when an earlier run has identical ROI payloads.

    Historical analysis-only runs could claim positions_extracted after copying
    traces. Identical values also arise from deterministic re-extraction, so they
    establish ambiguity, never proof that a particular preceding run was used.
    Explicit production provenance is outside this legacy-only audit.
    """
    if not connection.exec_driver_sql("PRAGMA table_info(analysis_result)").all():
        return
    results = {
        row["id"]: dict(row)
        for row in connection.exec_driver_sql(
            "SELECT * FROM analysis_result"
        ).mappings()
    }

    def positions(row: dict[str, Any], name: str) -> set[int]:
        return set(json.loads(row.get(name) or "null") or [])

    def spike_payload(trace_id: int) -> list[tuple]:
        return [
            (
                row["method"],
                row["units"],
                json.loads(row["values"]),
                row["valid_start"],
                row["valid_stop"],
            )
            for row in connection.exec_driver_sql(
                'SELECT r.method, r.units, s."values", s.valid_start, s.valid_stop '
                "FROM spike_trace s JOIN spike_inference_run r "
                "ON r.id = s.spike_inference_run_id WHERE s.trace_id = ? "
                "ORDER BY r.method, s.id",
                (trace_id,),
            ).mappings()
        ]

    def matches(trace: Any, original: Any) -> bool:
        def window(trace_id: int) -> tuple | None:
            row = connection.exec_driver_sql(
                "SELECT w.requested_discard_value, w.requested_discard_unit, "
                "w.original_frame_count, w.retained_frame_count, w.source_start_frame, "
                "w.source_start_time_ms, w.source_time_origin_ms, "
                "w.source_start_timestamp_ms, w.discarded_duration_ms "
                "FROM trace t JOIN extraction_frame_window w "
                "ON w.id = t.extraction_frame_window_id WHERE t.id = ?",
                (trace_id,),
            ).first()
            return tuple(row) if row is not None else None

        return (
            all(
                json.loads(trace[name] or "null")
                == json.loads(original[name] or "null")
                for name in (
                    "raw_trace",
                    "corrected_trace",
                    "neuropil_trace",
                    "dff",
                    "den_dff",
                    "x_axis",
                )
            )
            and all(
                trace[name] == original[name]
                for name in (
                    "x_axis_units",
                    "source_start_frame",
                    "source_start_time_ms",
                    "original_frame_count",
                    "discarded_duration_ms",
                )
            )
            and window(trace["id"]) == window(original["id"])
            and spike_payload(trace["id"]) == spike_payload(original["id"])
        )

    ambiguous: dict[int, list[dict[str, Any]]] = {}
    for result_id, result in results.items():
        if (
            result.get("legacy_trace_resolution") != "legacy_stage_inferred"
            or result.get("source_extraction_result_id") != result_id
            or not positions(result, "positions_analyzed")
            or result.get("created_at") is None
        ):
            continue
        candidates = [
            candidate
            for candidate in results.values()
            if candidate["id"] != result_id
            and candidate.get("created_at") is not None
            and candidate["created_at"] <= result["created_at"]
            and all(
                candidate.get(name) == result.get(name)
                for name in (
                    "experiment",
                    "detection_settings_id",
                    "extraction_settings_id",
                )
            )
            and positions(candidate, "positions_extracted")
        ]
        if not candidates:
            continue
        # Stream arrays one ROI at a time; the legacy marker must originate from
        # the v4 backfill, rather than a caller's current explicit/synthetic graph.
        for trace in connection.exec_driver_sql(
            "SELECT t.*, f.position_index FROM trace t "
            "JOIN extraction_frame_window w ON w.id = t.extraction_frame_window_id "
            "JOIN roi ON roi.id = t.roi_id JOIN fov f ON f.id = roi.fov_id "
            "WHERE t.analysis_result_id = ? AND w.extraction_result_id = ? "
            "AND w.provenance_source = 'legacy_import'",
            (result_id, result_id),
        ).mappings():
            alternatives = []
            for candidate in candidates:
                if trace["position_index"] not in positions(
                    candidate, "positions_extracted"
                ):
                    continue
                for original in connection.exec_driver_sql(
                    "SELECT * FROM trace WHERE analysis_result_id = ? AND roi_id = ?",
                    (candidate["id"], trace["roi_id"]),
                ).mappings():
                    if matches(trace, original):
                        alternatives.append(
                            {
                                "result_id": candidate["id"],
                                "trace_id": original["id"],
                            }
                        )
            if alternatives:
                ambiguous.setdefault(result_id, []).append(
                    {"trace_id": trace["id"], "alternatives": alternatives}
                )

    if not ambiguous:
        return

    # Capture dependencies before withdrawing links. Shared inference/window
    # records can also be referenced by later, correctly pinned analysis copies.
    affected: dict[int, set[int]] = {}
    for source_id in ambiguous:
        dependent_ids = {source_id}
        dependent_ids.update(
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT id FROM analysis_result WHERE source_extraction_result_id = ? "
                "UNION SELECT t.analysis_result_id FROM trace t "
                "LEFT JOIN extraction_frame_window w "
                "ON w.id = t.extraction_frame_window_id "
                "LEFT JOIN spike_trace s ON s.trace_id = t.id "
                "LEFT JOIN spike_inference_run r ON r.id = s.spike_inference_run_id "
                "WHERE w.extraction_result_id = ? OR r.extraction_result_id = ? "
                "UNION SELECT a.analysis_result_id FROM spike_fov_analysis a "
                "JOIN spike_inference_run r ON r.id = a.spike_inference_run_id "
                "WHERE r.extraction_result_id = ? "
                "UNION SELECT a.analysis_result_id FROM spike_analysis a "
                "JOIN spike_trace s ON s.id = a.spike_trace_id "
                "JOIN spike_inference_run r ON r.id = s.spike_inference_run_id "
                "WHERE r.extraction_result_id = ?",
                (source_id,) * 5,
            )
            if row[0] in results
        )
        for result_id in dependent_ids:
            affected.setdefault(result_id, set()).add(source_id)

    def audit(result_id: int, code: str, details: dict[str, Any]) -> None:
        existing = connection.exec_driver_sql(
            "SELECT details FROM migration_issue WHERE analysis_result_id = ? "
            "AND code = ?",
            (result_id, code),
        ).first()
        if existing is not None:
            # Preserve any earlier quarantine evidence in the same audit record.
            details = {**json.loads(existing[0] or "{}"), **details}
        connection.exec_driver_sql(
            "INSERT INTO migration_issue (analysis_result_id, code, details, resolved) "
            "VALUES (?, ?, ?, 0) ON CONFLICT(analysis_result_id, code) DO UPDATE "
            "SET details = excluded.details, resolved = 0",
            (result_id, code, json.dumps(details, sort_keys=True)),
        )

    for result_id, source_ids in affected.items():
        audit(
            result_id,
            "unresolved_stage_flags",
            {
                "previous_source_result_id": results[result_id].get(
                    "source_extraction_result_id"
                ),
                "previous_resolution": results[result_id].get(
                    "legacy_trace_resolution"
                ),
                "ambiguous_sources": [
                    {"result_id": source_id, "matching_traces": ambiguous[source_id]}
                    for source_id in sorted(source_ids)
                ],
            },
        )
        connection.exec_driver_sql(
            "UPDATE analysis_result SET source_extraction_result_id = NULL, "
            "legacy_trace_resolution = 'unresolved_stage_flags' WHERE id = ?",
            (result_id,),
        )
        for table, fk, code in (
            ("spike_analysis", "spike_trace_id", "unresolved_spike_analysis"),
            (
                "spike_fov_analysis",
                "spike_inference_run_id",
                "unresolved_spike_fov_analysis",
            ),
        ):
            children = [
                {"id": row[0], "previous_source_id": row[1]}
                for row in connection.exec_driver_sql(
                    f"SELECT id, {fk} FROM {table} WHERE analysis_result_id = ? "
                    "AND provenance_source != 'legacy_unresolved'",
                    (result_id,),
                )
            ]
            if children:
                audit(result_id, code, {"stage_flag_quarantine": children})
                connection.exec_driver_sql(
                    f"UPDATE {table} SET {fk} = NULL, "
                    "provenance_source = 'legacy_unresolved' "
                    "WHERE analysis_result_id = ?",
                    (result_id,),
                )

    for source_id in ambiguous:
        for table in ("extraction_frame_window", "spike_inference_run"):
            connection.exec_driver_sql(
                f"UPDATE {table} SET legacy_owner_result_id = extraction_result_id, "
                "extraction_result_id = NULL, "
                "provenance_source = 'legacy_stage_unresolved' "
                "WHERE extraction_result_id = ?",
                (source_id,),
            )

    for table in (
        "migration_issue",
        "extraction_frame_window",
        "spike_inference_run",
        "spike_analysis",
        "spike_fov_analysis",
    ):
        if connection.exec_driver_sql(f"PRAGMA foreign_key_check({table})").all():
            raise ValueError(f"Invalid foreign keys in {table} legacy stage audit.")
