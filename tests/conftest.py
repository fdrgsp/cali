"""Pytest configuration for cali tests.

CRITICAL: This file is imported BEFORE pytest plugins initialize.
On Windows, we must import torch before PyQt6 to avoid DLL conflicts.
The pytest-qt plugin auto-imports PyQt6 during initialization, so we
preload torch here to ensure proper DLL loading order.
"""

import shutil
import sys
import tempfile
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
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


_TEST_DB = Path("tests/test_data/data_and_db_for_tests/test_db.cali")


@pytest.fixture
def test_db_copy(tmp_path: Path) -> Path:
    """Return a disposable copy of test_db.cali so the original is never modified."""
    dest = tmp_path / "test_db.cali"
    shutil.copy2(_TEST_DB, dest)
    return dest


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


@pytest.fixture
def populated_db(
    tmp_path: Path,
    test_experiment: Any,
    data_path: Path,
    mock_detection_runner: MagicMock,
) -> Path:
    """Create a database with an Experiment and FOVs via a mocked detection run."""
    from cali.runner import CaliRunner
    from cali.sqlmodel._model import DetectionSettings

    db_path = tmp_path / "test_populated.cali"
    runner = CaliRunner(commit_batch_size=1)
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
        database_name=db_path.name,
        output_path=db_path.parent,
        global_position_indices=[0],
    )
    return db_path


@pytest.fixture
def label_tiff(tmp_path: Path) -> Path:
    """Create a simple 2D label TIFF with 3 labelled regions."""
    import tifffile

    arr = np.zeros((256, 256), dtype=np.uint16)
    arr[10:30, 10:30] = 1
    arr[50:70, 50:70] = 2
    arr[100:120, 100:120] = 3
    p = tmp_path / "A1_0000_labels.tif"
    tifffile.imwrite(p, arr)
    return p


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
