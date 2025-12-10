"""Pytest configuration for cali tests.

CRITICAL: This file is imported BEFORE pytest plugins initialize.
On Windows, we must import torch before PyQt6 to avoid DLL conflicts.
The pytest-qt plugin auto-imports PyQt6 during initialization, so we
preload torch here to ensure proper DLL loading order.
"""

import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from useq import register_well_plates

from cali.sqlmodel import Experiment
from cali.sqlmodel._util import create_database_and_tables

# Import torch before pytest-qt initializes on Windows
if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        # Torch/cellpose might not be installed, that's ok
        pass


TempDB = tuple[Engine, Path]


register_well_plates(
    {
        "dish-35mm-round": {
            "rows": 1,
            "columns": 1,
            "well_spacing": 0.0,
            "well_size": 35.0,
            "circular_wells": True,
            "name": "dish-35mm-round",
        },
    }
)


@pytest.fixture
def temp_db() -> Generator[TempDB, None, None]:
    """Create a temporary SQLite database for testing."""
    import gc

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    engine = create_engine(f"sqlite:///{db_path}")
    create_database_and_tables(engine)

    yield engine, db_path

    # Cleanup - dispose engine before deleting file
    # Dispose with close=True to close all checked-in connections (Python 3.13)
    engine.dispose(close=True)
    # Force garbage collection to ensure connections are closed
    gc.collect()
    # Delete the database file
    try:
        db_path.unlink(missing_ok=True)
    except PermissionError:
        # On Windows, file might still be locked
        pass


@pytest.fixture
def data_path() -> Path:
    """Return path to test data."""
    path = Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")
    if not path.exists():
        pytest.skip(f"Test data not found at {path}")
    return path


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Create a test database path."""
    return tmp_path / "test_runners.cali"


@pytest.fixture
def test_experiment(data_path: Path) -> Experiment:
    """Create a test experiment."""
    exp = Experiment.create_from_data(
        name="Test Runner Experiment",
        data_path=str(data_path),
    )
    return exp
