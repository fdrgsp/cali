"""Open cali databases only after ordered, transactional schema migrations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

if TYPE_CHECKING:
    from sqlalchemy.engine import URL, Engine

SCHEMA_VERSION = 2


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


_MIGRATIONS = (_analysis_gates, _startup_discard)


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
    try:
        ensure_schema_current(engine)
    except BaseException:
        engine.dispose()
        raise
    return engine
