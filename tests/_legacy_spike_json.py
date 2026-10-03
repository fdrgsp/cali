"""Restore historical storage when current ORM models construct old test files."""

import json

from sqlalchemy.engine import Connection

from cali.sqlmodel._trace_array_codec import decode_trace_array


def write_legacy_spike_json(connection: Connection) -> None:
    """Version <=10 fixtures must contain JSON before replaying frozen migrations."""
    columns = {
        row[1] for row in connection.exec_driver_sql("PRAGMA table_info(spike_trace)")
    }
    if "values" not in columns:
        return
    for identifier, payload in connection.exec_driver_sql(
        'SELECT id,"values" FROM spike_trace WHERE typeof("values")=\'blob\''
    ).all():
        connection.exec_driver_sql(
            'UPDATE spike_trace SET "values"=? WHERE id=?',
            (json.dumps(decode_trace_array(payload)), identifier),
        )
