"""Frozen, model-free SQLite v7 FOV spike-result backfill."""

import json
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy.engine import Connection

# Keep migration columns frozen instead of deriving them from the current ORM.
_METRIC_COLUMNS = (
    ("spike_max_lag_correlation_matrix", "JSON"),
    ("global_spike_max_lag_correlation", "FLOAT"),
    ("spike_max_lag_values_matrix", "JSON"),
    ("spike_max_lag_correlation_matrix_rising_edges", "JSON"),
    ("global_spike_max_lag_correlation_rising_edges", "FLOAT"),
    ("spike_max_lag_values_matrix_rising_edges", "JSON"),
    ("spike_ccg_zscore_matrix", "JSON"),
    ("spike_ccg_zscore_matrix_rising_edges", "JSON"),
    ("fraction_significant_ccg_pairs", "FLOAT"),
    ("fraction_significant_ccg_pairs_rising_edges", "FLOAT"),
    ("spike_jitter_synchrony_matrix", "JSON"),
    ("global_spike_jitter_synchrony", "FLOAT"),
    ("spike_jitter_synchrony_matrix_rising_edges", "JSON"),
    ("global_spike_jitter_synchrony_rising_edges", "FLOAT"),
    ("spike_burst_count", "INTEGER"),
    ("spike_burst_avg_duration", "FLOAT"),
    ("spike_burst_avg_interval", "FLOAT"),
    ("spike_burst_starts", "JSON"),
    ("spike_burst_ends", "JSON"),
    ("spike_population_activity", "JSON"),
    ("spike_population_activity_raw", "JSON"),
)


def migrate_spike_fov_analyses(connection: Connection) -> None:
    """Copy exact legacy matrices and ordering, auditing uncertain ownership."""
    metrics_ddl = ", ".join(f"{name} {kind}" for name, kind in _METRIC_COLUMNS)
    connection.exec_driver_sql(f"""CREATE TABLE IF NOT EXISTS spike_fov_analysis (
        id INTEGER NOT NULL PRIMARY KEY,
        fov_analysis_id INTEGER NOT NULL REFERENCES fov_analysis(id) ON DELETE CASCADE,
        fov_id INTEGER REFERENCES fov(id) ON DELETE CASCADE,
        analysis_result_id INTEGER REFERENCES analysis_result(id) ON DELETE CASCADE,
        spike_inference_run_id INTEGER
            REFERENCES spike_inference_run(id) ON DELETE SET NULL,
        method VARCHAR NOT NULL,
        units VARCHAR NOT NULL,
        provenance_source VARCHAR NOT NULL,
        active_roi_labels JSON,
        {metrics_ddl},
        UNIQUE(fov_id, spike_inference_run_id, analysis_result_id),
        UNIQUE(fov_analysis_id, method)
    )""")
    columns = {
        row[1] for row in connection.exec_driver_sql("PRAGMA table_info(fov_analysis)")
    }
    if not columns:
        return
    if "calcium_active_roi_labels" not in columns:
        connection.exec_driver_sql(
            "ALTER TABLE fov_analysis ADD COLUMN calcium_active_roi_labels JSON"
        )
    for name in ("analysis_result_id", "spike_inference_run_id"):
        connection.exec_driver_sql(
            f"CREATE INDEX IF NOT EXISTS ix_spike_fov_analysis_{name} "
            f"ON spike_fov_analysis({name})"
        )
    owners = {
        row["id"]: dict(row)
        for row in connection.exec_driver_sql(
            "SELECT * FROM analysis_result"
        ).mappings()
    }
    fov_ids = {row[0] for row in connection.exec_driver_sql("SELECT id FROM fov")}
    duplicate_owners = {
        (row[0], row[1])
        for row in connection.exec_driver_sql(
            "SELECT fov_id, analysis_result_id FROM fov_analysis "
            "GROUP BY fov_id, analysis_result_id HAVING COUNT(*) > 1"
        )
    }
    audits: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    names = ("active_roi_labels", *(name for name, _ in _METRIC_COLUMNS))

    def decoded(name: str, value: Any) -> Any:
        return (
            json.loads(value)
            if value is not None
            and (name == "active_roi_labels" or dict(_METRIC_COLUMNS)[name] == "JSON")
            else value
        )

    for parent in connection.exec_driver_sql("SELECT * FROM fov_analysis").mappings():
        labels = parent.get("active_roi_labels")
        if parent.get("calcium_active_roi_labels") is None:
            connection.exec_driver_sql(
                "UPDATE fov_analysis SET calcium_active_roi_labels = ? WHERE id = ?",
                (labels, parent["id"]),
            )
        values = {name: parent.get(name) for name, _ in _METRIC_COLUMNS}
        if all(decoded(name, value) is None for name, value in values.items()):
            continue
        owner_id, fov_id = parent.get("analysis_result_id"), parent.get("fov_id")
        child_fov_id = fov_id if fov_id in fov_ids else None
        owner = owners.get(owner_id, {})
        resolution = owner.get("legacy_trace_resolution") or ""
        ordering = json.loads(labels) if labels else None
        candidates = connection.exec_driver_sql(
            "SELECT r.id, roi.label_value FROM spike_trace s "
            "JOIN spike_inference_run r ON r.id=s.spike_inference_run_id "
            "JOIN trace t ON t.id=s.trace_id JOIN roi ON roi.id=t.roi_id "
            "WHERE t.analysis_result_id IS ? AND roi.fov_id IS ? "
            "AND r.method='oasis' AND r.units='a.u.'",
            (owner_id, fov_id),
        ).all()
        selected = [row for row in candidates if ordering and row[1] in ordering]
        runs = {row[0] for row in selected}
        coverage = Counter(row[1] for row in selected)
        unresolved_roi_metrics = connection.exec_driver_sql(
            "SELECT 1 FROM spike_analysis s "
            "JOIN data_analysis d ON d.id=s.data_analysis_id "
            "JOIN roi ON roi.id=d.roi_id "
            "WHERE d.analysis_result_id IS ? AND roi.fov_id IS ? "
            "AND s.provenance_source='legacy_unresolved' LIMIT 1",
            (owner_id, fov_id),
        ).first()
        linked = (
            bool(owner)
            and child_fov_id is not None
            and ordering is not None
            and bool(ordering)
            and len(set(ordering)) == len(ordering)
            and coverage == Counter(ordering)
            and len(runs) == 1
            and not resolution.startswith("unresolved")
            and resolution != "multiple_sources"
            and not unresolved_roi_metrics
            and (fov_id, owner_id) not in duplicate_owners
        )
        run_id = next(iter(runs)) if linked else None
        source = "legacy_import" if linked else "legacy_unresolved"
        existing = (
            connection.exec_driver_sql(
                "SELECT * FROM spike_fov_analysis WHERE fov_analysis_id=? "
                "AND method='oasis'",
                (parent["id"],),
            )
            .mappings()
            .all()
        )
        if existing:
            if len(existing) != 1 or any(
                decoded(name, existing[0][name])
                != decoded(
                    name, labels if name == "active_roi_labels" else values[name]
                )
                for name in names
            ):
                raise ValueError("FOV spike backfill failed exact-value verification.")
            if (
                existing[0]["spike_inference_run_id"] != run_id
                or existing[0]["analysis_result_id"] != (owner_id if owner else None)
                or existing[0]["fov_id"] != child_fov_id
                or existing[0]["units"] != "a.u."
                or existing[0]["provenance_source"] != source
            ):
                raise ValueError("FOV spike backfill failed source verification.")
        else:
            insert_names = (
                "fov_analysis_id",
                "fov_id",
                "analysis_result_id",
                "spike_inference_run_id",
                "method",
                "units",
                "provenance_source",
                *names,
            )
            connection.exec_driver_sql(
                f"INSERT INTO spike_fov_analysis ({', '.join(insert_names)}) "
                f"VALUES ({', '.join('?' for _ in insert_names)})",
                (
                    parent["id"],
                    child_fov_id,
                    owner_id if owner else None,
                    run_id,
                    "oasis",
                    "a.u.",
                    source,
                    labels,
                    *values.values(),
                ),
            )
        copied = (
            connection.exec_driver_sql(
                f"SELECT {', '.join(names)} FROM spike_fov_analysis "
                "WHERE fov_analysis_id=? AND method='oasis'",
                (parent["id"],),
            )
            .mappings()
            .one()
        )
        if any(
            decoded(name, copied[name])
            != decoded(name, labels if name == "active_roi_labels" else values[name])
            for name in names
        ):
            raise ValueError("FOV spike backfill failed exact-value verification.")
        if not linked:
            audits[owner_id if owner else None].append(
                {
                    "fov_analysis_id": parent["id"],
                    "legacy_fov_id": fov_id,
                    "candidate_inference_run_ids": sorted(runs),
                    "source_resolution": resolution or "missing_or_ambiguous_trace",
                }
            )
    for owner_id, details in audits.items():
        connection.exec_driver_sql(
            "INSERT INTO migration_issue (analysis_result_id, code, details, resolved) "
            "VALUES (?, 'unresolved_spike_fov_analysis', ?, 0)",
            (owner_id, json.dumps({"rows": details}, sort_keys=True)),
        )
    if connection.exec_driver_sql("PRAGMA foreign_key_check(spike_fov_analysis)").all():
        raise ValueError("Invalid FOV spike references in backfill.")
