"""Open cali databases only after ordered, transactional schema migrations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Connection

from ._source_migration import migrate_source_links
from ._spike_analysis_migration import migrate_spike_analyses
from ._spike_fov_analysis_migration import migrate_spike_fov_analyses
from ._trace_migration import migrate_trace_provenance

if TYPE_CHECKING:
    from sqlalchemy.engine import URL, Engine

SCHEMA_VERSION = 7


def _add_columns(
    connection: Connection, table: str, columns: tuple[tuple[str, str], ...]
) -> None:
    existing = {
        row[1] for row in connection.exec_driver_sql(f"PRAGMA table_info({table})")
    }
    if not existing:
        return
    if "id" not in existing:
        raise ValueError(f"Cannot migrate {table}: missing primary-key column 'id'.")
    for name, declaration in columns:
        if name not in existing:
            connection.exec_driver_sql(
                f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
            )


def _analysis_gates(connection: Connection) -> None:
    _add_columns(
        connection,
        "analysis_settings",
        (
            ("enable_calcium", "BOOLEAN DEFAULT 1 NOT NULL"),
            ("enable_spikes", "BOOLEAN DEFAULT 1 NOT NULL"),
        ),
    )


def _startup_discard(connection: Connection) -> None:
    _add_columns(
        connection,
        "extraction_settings",
        (
            ("frame_rate_verified", "BOOLEAN DEFAULT 0 NOT NULL"),
            ("discard_initial_value", "REAL DEFAULT 0.0 NOT NULL"),
            ("discard_initial_unit", "VARCHAR DEFAULT 'frames' NOT NULL"),
        ),
    )
    _add_columns(
        connection,
        "trace",
        (
            ("source_start_frame", "INTEGER DEFAULT 0 NOT NULL"),
            ("source_start_time_ms", "REAL DEFAULT 0.0 NOT NULL"),
            ("original_frame_count", "INTEGER"),
            ("discarded_duration_ms", "REAL DEFAULT 0.0 NOT NULL"),
            ("discard_timing_source", "VARCHAR"),
        ),
    )


def _method_settings(connection: Connection) -> None:
    """Normalize method selection and preserve legacy OASIS configuration exactly."""
    # Some historical files predate CCG/rising-edge settings. Known missing fields
    # receive the defaults used by the OASIS pipeline; existing values are untouched.
    _add_columns(
        connection,
        "analysis_settings",
        (
            ("spike_threshold_mode", "VARCHAR DEFAULT 'multiplier' NOT NULL"),
            ("spike_threshold_value", "FLOAT DEFAULT 3.0 NOT NULL"),
            ("burst_threshold", "FLOAT DEFAULT 65.0 NOT NULL"),
            ("burst_min_duration", "FLOAT DEFAULT 500.0 NOT NULL"),
            ("burst_gaussian_sigma", "FLOAT DEFAULT 0.3 NOT NULL"),
            ("spikes_sync_cross_corr_lag", "FLOAT DEFAULT 500.0 NOT NULL"),
            ("spikes_sync_jitter_window", "FLOAT DEFAULT 200.0 NOT NULL"),
            ("ccg_n_shuffles", "INTEGER DEFAULT 20 NOT NULL"),
            ("enable_rising_edge_analysis", "BOOLEAN DEFAULT 0 NOT NULL"),
        ),
    )
    _add_columns(
        connection,
        "extraction_settings",
        (
            ("spike_methods", "JSON DEFAULT '[\"oasis\"]' NOT NULL"),
            ("cascade_model", "VARCHAR"),
            ("cascade_device", "VARCHAR DEFAULT 'auto' NOT NULL"),
        ),
    )
    connection.exec_driver_sql(
        """CREATE TABLE IF NOT EXISTS spike_analysis_settings (
            id INTEGER NOT NULL PRIMARY KEY,
            analysis_settings_id INTEGER NOT NULL
                REFERENCES analysis_settings(id) ON DELETE CASCADE,
            method VARCHAR NOT NULL,
            threshold_mode VARCHAR NOT NULL,
            threshold_value FLOAT,
            cascade_ap_threshold_fraction FLOAT,
            burst_threshold FLOAT NOT NULL,
            burst_min_duration FLOAT NOT NULL,
            burst_gaussian_sigma FLOAT NOT NULL,
            spikes_sync_cross_corr_lag FLOAT NOT NULL,
            spikes_sync_jitter_window FLOAT NOT NULL,
            ccg_n_shuffles INTEGER NOT NULL,
            enable_rising_edge_analysis BOOLEAN NOT NULL,
            UNIQUE (analysis_settings_id, method)
        )"""
    )
    columns = {
        row[1]
        for row in connection.exec_driver_sql("PRAGMA table_info(analysis_settings)")
    }
    if not columns:
        return
    pairs = (
        ("spike_threshold_mode", "threshold_mode"),
        ("spike_threshold_value", "threshold_value"),
        ("burst_threshold", "burst_threshold"),
        ("burst_min_duration", "burst_min_duration"),
        ("burst_gaussian_sigma", "burst_gaussian_sigma"),
        ("spikes_sync_cross_corr_lag", "spikes_sync_cross_corr_lag"),
        ("spikes_sync_jitter_window", "spikes_sync_jitter_window"),
        ("ccg_n_shuffles", "ccg_n_shuffles"),
        ("enable_rising_edge_analysis", "enable_rising_edge_analysis"),
    )
    missing = {source for source, _ in pairs} - columns
    if missing:
        raise ValueError(
            f"Cannot migrate spike settings: missing columns {sorted(missing)}."
        )
    targets = ", ".join(target for _, target in pairs)
    sources = ", ".join(f"a.{source}" for source, _ in pairs)
    connection.exec_driver_sql(
        "INSERT INTO spike_analysis_settings "
        f"(analysis_settings_id, method, {targets}) "
        f"SELECT a.id, 'oasis', {sources} FROM analysis_settings a "
        "WHERE NOT EXISTS (SELECT 1 FROM spike_analysis_settings s "
        "WHERE s.analysis_settings_id = a.id AND s.method = 'oasis')"
    )
    # Exact SQL comparisons cover every legacy value, including non-default settings.
    different = " OR ".join(f"s.{target} IS NOT a.{source}" for source, target in pairs)
    mismatches = connection.exec_driver_sql(
        "SELECT COUNT(*) FROM analysis_settings a LEFT JOIN spike_analysis_settings s "
        "ON s.analysis_settings_id = a.id AND s.method = 'oasis' "
        f"WHERE s.id IS NULL OR {different}"
    ).scalar_one()
    if mismatches:
        raise ValueError("Spike settings backfill failed exact-value verification.")
    duplicates = connection.exec_driver_sql(
        "SELECT 1 FROM spike_analysis_settings "
        "GROUP BY analysis_settings_id, method HAVING COUNT(*) > 1 LIMIT 1"
    ).first()
    if duplicates:
        raise ValueError("Spike settings backfill contains duplicate method rows.")
    orphaned = connection.exec_driver_sql(
        "SELECT COUNT(*) FROM spike_analysis_settings s "
        "LEFT JOIN analysis_settings a ON a.id = s.analysis_settings_id "
        "WHERE s.analysis_settings_id IS NOT NULL AND a.id IS NULL"
    ).scalar_one()
    if orphaned:
        raise ValueError("Spike settings backfill contains orphaned parent references.")


_MIGRATIONS = (
    _analysis_gates,
    _startup_discard,
    _method_settings,
    migrate_trace_provenance,
    migrate_source_links,
    migrate_spike_analyses,
    migrate_spike_fov_analyses,
)


def migrate_database(engine: Engine) -> None:
    """Upgrade SQLite schemas before ORM access, preserving retryable versions.

    An empty database stays unversioned until tables are created. Each migration
    and its version update share an explicit SQLite transaction, including DDL.
    Databases from a newer cali release are rejected without modifying them.
    """
    if engine.dialect.name != "sqlite":
        raise ValueError("cali schema migrations require a SQLite database.")
    database = engine.url.database
    if (
        database
        and database != ":memory:"
        and not engine.url.query.get("uri")
        and not Path(database).exists()
    ):
        # Preserve SQLAlchemy's lazy creation: callers may check file existence
        # after preparing the engine (e.g. the GUI's overwrite decision).
        return
    with engine.connect() as connection:
        version = connection.exec_driver_sql("PRAGMA user_version").scalar_one()
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"Database schema version {version} is newer than supported "
                f"version {SCHEMA_VERSION}. Upgrade cali to open this database."
            )
        if version == SCHEMA_VERSION:
            return
        tables = connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%'"
        ).all()
        if not tables:
            return
        connection.commit()
        while version < SCHEMA_VERSION:
            # SQLite's deferred implicit transactions do not include DDL. Acquire
            # the writer lock explicitly and re-read the version for racing opens.
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                version = connection.exec_driver_sql("PRAGMA user_version").scalar_one()
                if version > SCHEMA_VERSION:
                    raise ValueError("Database was upgraded by a newer cali release.")
                if version < SCHEMA_VERSION:
                    _MIGRATIONS[version](connection)
                    version += 1
                    connection.exec_driver_sql(f"PRAGMA user_version = {version}")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise


def ensure_schema_current(engine: Engine | Connection) -> None:
    """Migrate an externally supplied engine/session bind before ORM queries."""
    if isinstance(engine, Connection):
        # Reading through a second engine connection can roll back a caller's
        # uncommitted work with SQLite's shared in-memory connection pool.
        active_transaction = engine.in_transaction()
        version = engine.exec_driver_sql("PRAGMA user_version").scalar_one()
        if version == SCHEMA_VERSION:
            return
        if active_transaction:
            raise ValueError(
                "Migrate the database with create_cali_engine before starting "
                "a transaction on an older schema."
            )
        engine.rollback()
        engine = engine.engine
    migrate_database(engine)


def create_cali_engine(url: str | URL, **kwargs: Any) -> Engine:
    """Create a SQLAlchemy engine and migrate existing cali tables before use."""
    engine = create_engine(url, **kwargs)
    if engine.dialect.name == "sqlite":
        # Install before migration opens the first pooled connection. A listener
        # added afterward misses that connection when the runner reuses it.
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    try:
        ensure_schema_current(engine)
    except BaseException:
        engine.dispose()
        raise
    return engine


def _enable_sqlite_foreign_keys(connection: Any, _: Any) -> None:
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
