"""Frozen, model-free SQLite v6 ROI spike-result backfill."""

import json
from collections import defaultdict
from typing import Any

from sqlalchemy.engine import Connection


def migrate_spike_analyses(connection: Connection) -> None:
    """Preserve exact metrics and quarantine rows with unprovable trace ownership."""
    connection.exec_driver_sql("""CREATE TABLE IF NOT EXISTS spike_analysis (
        id INTEGER NOT NULL PRIMARY KEY,
        data_analysis_id INTEGER NOT NULL
            REFERENCES data_analysis(id) ON DELETE CASCADE,
        analysis_result_id INTEGER REFERENCES analysis_result(id) ON DELETE CASCADE,
        spike_trace_id INTEGER REFERENCES spike_trace(id) ON DELETE SET NULL,
        method VARCHAR NOT NULL,
        units VARCHAR NOT NULL,
        threshold FLOAT,
        threshold_mode VARCHAR,
        suprathreshold_sample_rate_hz FLOAT,
        suprathreshold_rising_edge_rate_hz FLOAT,
        expected_spike_rate_hz FLOAT,
        expected_spike_count FLOAT,
        suprathreshold_excursion_rate_hz FLOAT,
        spike_active BOOLEAN,
        provenance_source VARCHAR NOT NULL,
        UNIQUE(spike_trace_id, analysis_result_id),
        UNIQUE(data_analysis_id, method)
    )""")
    columns = {
        row[1] for row in connection.exec_driver_sql("PRAGMA table_info(data_analysis)")
    }
    if not columns:
        return
    if "calcium_active" not in columns:
        connection.exec_driver_sql(
            "ALTER TABLE data_analysis ADD COLUMN calcium_active BOOLEAN"
        )
    for name in ("analysis_result_id", "spike_trace_id"):
        connection.exec_driver_sql(
            f"CREATE INDEX IF NOT EXISTS ix_spike_analysis_{name} "
            f"ON spike_analysis({name})"
        )

    def rows(table: str) -> dict[int | None, dict[str, Any]]:
        if not connection.exec_driver_sql(f"PRAGMA table_info({table})").all():
            return {}
        return {
            row["id"]: dict(row)
            for row in connection.exec_driver_sql(f"SELECT * FROM {table}").mappings()
        }

    results, settings = rows("analysis_result"), rows("analysis_settings")
    modes = {
        row["analysis_settings_id"]: row["threshold_mode"]
        for row in connection.exec_driver_sql(
            "SELECT analysis_settings_id, threshold_mode FROM spike_analysis_settings "
            "WHERE method = 'oasis'"
        ).mappings()
    }
    duplicate_owners = {
        (row[0], row[1])
        for row in connection.exec_driver_sql(
            "SELECT roi_id, analysis_result_id FROM data_analysis "
            "GROUP BY roi_id, analysis_result_id HAVING COUNT(*) > 1"
        )
    }
    audits: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    pairs = (
        ("inferred_spikes_threshold", "threshold"),
        ("inferred_spikes_frequency", "suprathreshold_sample_rate_hz"),
        ("inferred_spikes_rising_edge_frequency", "suprathreshold_rising_edge_rate_hz"),
    )
    for parent in connection.exec_driver_sql("SELECT * FROM data_analysis").mappings():
        owner_id = parent.get("analysis_result_id")
        owner = results.get(owner_id, {})
        config = settings.get(owner.get("analysis_settings_id"), {})
        if parent.get("calcium_active") is None:
            peaks = json.loads(parent.get("peaks_den_dff") or "null")
            frequency = parent.get("den_dff_frequency")
            calcium_active = (
                bool(peaks)
                if peaks is not None
                else frequency > 0
                if frequency is not None
                else False
                if config
                else None
            )
            connection.exec_driver_sql(
                "UPDATE data_analysis SET calcium_active = ? WHERE id = ?",
                (calcium_active, parent["id"]),
            )
        values = {target: parent.get(source) for source, target in pairs}
        if all(value is None for value in values.values()):
            continue
        resolution = owner.get("legacy_trace_resolution") or ""
        candidates = connection.exec_driver_sql(
            'SELECT s.id, s."values" FROM spike_trace s '
            "JOIN spike_inference_run r ON r.id = s.spike_inference_run_id "
            "JOIN trace t ON t.id = s.trace_id "
            "WHERE t.roi_id IS ? AND t.analysis_result_id IS ? "
            "AND r.method = 'oasis' AND r.units = 'a.u.'",
            (parent.get("roi_id"), owner_id),
        ).all()
        linked = (
            bool(owner)
            and parent.get("roi_id") is not None
            and not resolution.startswith("unresolved")
            and resolution != "multiple_sources"
            and (parent.get("roi_id"), owner_id) not in duplicate_owners
            and len(candidates) == 1
        )
        spike_id = candidates[0][0] if linked else None
        rate = values["suprathreshold_sample_rate_hz"]
        active = rate > 0 if rate is not None else None
        if active is None and linked and values["threshold"] is not None:
            active = any(
                v > values["threshold"] and v > 0 for v in json.loads(candidates[0][1])
            )
        source = "legacy_import" if linked else "legacy_unresolved"
        if not linked:
            audits[owner_id if owner else None].append(
                {
                    "data_analysis_id": parent["id"],
                    "candidate_spike_trace_ids": [
                        candidate[0] for candidate in candidates
                    ],
                    "source_resolution": resolution or "missing_or_ambiguous_trace",
                }
            )
        existing = (
            connection.exec_driver_sql(
                "SELECT * FROM spike_analysis WHERE data_analysis_id = ? "
                "AND method = 'oasis'",
                (parent["id"],),
            )
            .mappings()
            .all()
        )
        if existing:
            if len(existing) != 1 or any(
                existing[0][name] != value for name, value in values.items()
            ):
                raise ValueError(
                    "Spike analysis backfill failed exact-value verification."
                )
            if (
                existing[0]["spike_trace_id"] != spike_id
                or existing[0]["analysis_result_id"] != (owner_id if owner else None)
                or existing[0]["units"] != "a.u."
                or existing[0]["provenance_source"] != source
            ):
                raise ValueError("Spike analysis backfill failed source verification.")
        else:
            connection.exec_driver_sql(
                "INSERT INTO spike_analysis (data_analysis_id, analysis_result_id, "
                "spike_trace_id, method, units, threshold, threshold_mode, "
                "suprathreshold_sample_rate_hz, suprathreshold_rising_edge_rate_hz, "
                "spike_active, provenance_source) VALUES (?, ?, ?, 'oasis', 'a.u.', "
                "?, ?, ?, ?, ?, ?)",
                (
                    parent["id"],
                    owner_id if owner else None,
                    spike_id,
                    values["threshold"],
                    modes.get(owner.get("analysis_settings_id")),
                    values["suprathreshold_sample_rate_hz"],
                    values["suprathreshold_rising_edge_rate_hz"],
                    active,
                    source,
                ),
            )
        copied = connection.exec_driver_sql(
            "SELECT threshold, suprathreshold_sample_rate_hz, "
            "suprathreshold_rising_edge_rate_hz FROM spike_analysis "
            "WHERE data_analysis_id = ? AND method = 'oasis'",
            (parent["id"],),
        ).one()
        if tuple(copied) != tuple(values.values()):
            raise ValueError("Spike analysis backfill failed exact-value verification.")
    for owner_id, details in audits.items():
        connection.exec_driver_sql(
            "INSERT INTO migration_issue (analysis_result_id, code, details, resolved) "
            "VALUES (?, 'unresolved_spike_analysis', ?, 0)",
            (owner_id, json.dumps({"rows": details}, sort_keys=True)),
        )
    if connection.exec_driver_sql("PRAGMA foreign_key_check(spike_analysis)").all():
        raise ValueError("Invalid spike-analysis references in backfill.")
