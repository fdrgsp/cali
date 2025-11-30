"""Pytest configuration for cali tests.

CRITICAL: This file is imported BEFORE pytest plugins initialize.
On Windows, we must import torch before PyQt6 to avoid DLL conflicts.
The pytest-qt plugin auto-imports PyQt6 during initialization, so we
preload torch here to ensure proper DLL loading order.
"""

import sys
from pathlib import Path

import pytest

from cali.sqlmodel import Experiment

# Import torch before pytest-qt initializes on Windows
if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        # Torch/cellpose might not be installed, that's ok
        pass


@pytest.fixture
def data_path() -> Path:
    """Return path to test data."""
    path = Path("tests/test_data/2pos/evk.tensorstore.zarr")
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
