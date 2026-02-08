"""Pytest configuration for cali tests.

CRITICAL: This file is imported BEFORE pytest plugins initialize.
On Windows, we must import torch before PyQt6 to avoid DLL conflicts.
The pytest-qt plugin auto-imports PyQt6 during initialization, so we
preload torch here to ensure proper DLL loading order.
"""

import sys
import tempfile
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from useq import register_well_plates

from cali.sqlmodel import Experiment
from cali.sqlmodel._model import FOV, ROI, Mask
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


def _migrate_database_schema(engine: Engine) -> None:
    """Migrate database schema to add missing columns.

    This function ensures backward compatibility by adding new columns
    to existing test databases that were created with older schemas.
    """
    with engine.connect() as conn:
        # Add use_gpu column to detection_settings if missing
        try:
            conn.execute(text("SELECT use_gpu FROM detection_settings LIMIT 1"))
        except OperationalError:
            try:
                conn.execute(
                    text(
                        "ALTER TABLE detection_settings "
                        "ADD COLUMN use_gpu BOOLEAN DEFAULT 1"
                    )
                )
                conn.commit()
            except OperationalError:
                conn.rollback()


@pytest.fixture(scope="session", autouse=True)
def migrate_test_databases() -> None:
    """Automatically migrate test database schemas before running tests.

    This fixture runs once per session and migrates all test databases
    to include new schema columns added since the database was created.
    """
    # Migrate the shared test database if it exists
    test_db_path = Path("tests/test_data/data_and_db_for_tests/test_db.cali")
    if test_db_path.exists():
        engine = create_engine(f"sqlite:///{test_db_path}")
        _migrate_database_schema(engine)
        engine.dispose(close=True)


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


def create_mock_fov(position_index: int = 0, num_rois: int = 3) -> FOV:
    """Create a mock FOV with ROIs for testing without running cellpose."""
    fov = FOV(position_index=position_index, name=f"A1_{position_index:04d}")

    rois = []
    for i in range(1, num_rois + 1):
        # Create a simple circular mask matching dataset dims (256x256)
        mask_data = np.zeros((256, 256), dtype=np.uint8)
        cy, cx = 50 + i * 20, 50 + i * 20
        y, x = np.ogrid[:256, :256]
        mask_region = ((x - cx) ** 2 + (y - cy) ** 2) <= 100
        mask_data[mask_region] = 1

        # Get coordinates from mask
        coords = np.where(mask_data)
        coords_y = coords[0].tolist()
        coords_x = coords[1].tolist()

        mask = Mask(
            mask_type="roi",
            coords_y=coords_y,
            coords_x=coords_x,
            height=256,
            width=256,
        )

        roi = ROI(
            label_value=i,
            roi_mask=mask,
        )
        rois.append(roi)

    fov.rois = rois
    return fov


@pytest.fixture()
def mock_detection_runner() -> Iterator[MagicMock]:
    """Fixture that patches DetectionRunner to return mock FOVs quickly.

    This avoids slow Cellpose model loading and execution during tests.
    """
    with patch(
        "cali.detection._detection_runner.DetectionRunner._run_cellpose"
    ) as mock:

        def mock_detection(
            dataset: Any,
            detection_settings: Any,
            position_indices: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Iterator[FOV]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection
        yield mock
