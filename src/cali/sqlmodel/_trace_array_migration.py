"""SQLite v11: compress canonical spike arrays without changing legacy copies."""

import numpy as np
from sqlalchemy.engine import Connection

from ._trace_array_codec import (
    TraceArrayError,
    decode_trace_array,
    encode_trace_array,
    trace_array_storage_info,
)


def migrate_trace_arrays(connection: Connection) -> None:
    """Stream and verify each array within the engine's versioned transaction.

    Historical migrations stay JSON-only and run before this step. Existing JSON
    column affinity accepts SQLite BLOBs; no destructive table rebuild is needed.
    Keep physical legacy inferred_spikes and all inference provenance unchanged.
    """
    columns = {
        row[1] for row in connection.exec_driver_sql("PRAGMA table_info(spike_trace)")
    }
    if not columns:
        return
    if not {"id", "values"} <= columns:
        raise ValueError("Cannot migrate spike arrays: missing id or values column.")
    last_id = None
    while True:
        rows = connection.exec_driver_sql(
            "SELECT id FROM spike_trace "
            + ("WHERE id > ? " if last_id is not None else "")
            + "ORDER BY id LIMIT 64",
            (last_id,) if last_id is not None else (),
        ).all()
        if not rows:
            return
        for (identifier,) in rows:
            try:
                payload = connection.exec_driver_sql(
                    'SELECT "values" FROM spike_trace WHERE id=?', (identifier,)
                ).scalar_one()
                original = decode_trace_array(payload)
                if trace_array_storage_info(payload)["version"] == 1:
                    last_id = identifier
                    continue
                encoded = encode_trace_array(original)
                connection.exec_driver_sql(
                    'UPDATE spike_trace SET "values"=? WHERE id=?',
                    (encoded, identifier),
                )
                stored = connection.exec_driver_sql(
                    'SELECT "values" FROM spike_trace WHERE id=?', (identifier,)
                ).scalar_one()
                if not np.array_equal(
                    original, decode_trace_array(stored), equal_nan=True
                ):
                    raise TraceArrayError(
                        "Migrated spike values failed exact verification."
                    )
            except (TypeError, TraceArrayError) as error:
                raise ValueError(f"Spike trace {identifier}: {error}") from error
            last_id = identifier
