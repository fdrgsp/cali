"""Tests for database_to_csv export functions edge cases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import create_engine

from cali._constants import (
    CALCIUM_DEC_DFF_CORRELATION,
    CALCIUM_DFF_CORRELATION,
    DFF_TRACES,
    NEUROPIL_CORRECTED_TRACES,
    RAW_CALCIUM_TRACES,
)
from cali.util._database_to_csv import (
    export_correlations_to_csv,
    export_traces_to_csv,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.engine import Engine


@pytest.fixture
def test_engine() -> Engine:
    """Create engine from existing test database."""
    db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
    engine = create_engine(f"sqlite:///{db_path}")
    yield engine
    engine.dispose(close=True)


def test_export_traces_creates_export_directory(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that export_traces_to_csv creates the export directory."""
    # Use tmp_path for database path to control output location
    db_path = tmp_path / "test.cali"

    export_traces = {
        RAW_CALCIUM_TRACES: True,
    }

    # The function should create the export directory
    export_traces_to_csv(test_engine, export_traces, run_id=1, db_path=db_path)

    # Verify export directory was created
    export_dir = tmp_path / "test_exports" / "run_1"
    assert export_dir.exists()
    assert export_dir.is_dir()


def test_export_correlations_creates_export_directory(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that export_correlations_to_csv creates the export directory."""
    db_path = tmp_path / "test.cali"

    export_correlations = {
        CALCIUM_DFF_CORRELATION: True,
    }

    export_correlations_to_csv(
        test_engine,
        export_correlations,
        run_id=1,
        db_path=db_path,
    )

    # Verify export directory was created
    export_dir = tmp_path / "test_exports" / "run_1"
    assert export_dir.exists()
    assert export_dir.is_dir()


def test_export_traces_respects_selection(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that export_traces_to_csv only exports selected types."""
    db_path = tmp_path / "test.cali"

    # Only export RAW_CALCIUM_TRACES
    export_traces = {
        RAW_CALCIUM_TRACES: True,
        DFF_TRACES: False,
        NEUROPIL_CORRECTED_TRACES: False,
    }

    export_traces_to_csv(test_engine, export_traces, run_id=1, db_path=db_path)

    export_dir = tmp_path / "test_exports" / "run_1"

    # Only raw_traces.csv should exist
    assert (export_dir / "raw_traces.csv").exists()
    assert not (export_dir / "dff_traces.csv").exists()
    assert not (export_dir / "neuropil_corrected_traces.csv").exists()


def test_export_correlations_respects_selection(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that export_correlations_to_csv only exports selected types."""
    db_path = tmp_path / "test.cali"

    # Only export one type
    export_correlations = {
        CALCIUM_DFF_CORRELATION: True,
        CALCIUM_DEC_DFF_CORRELATION: False,
    }

    export_correlations_to_csv(
        test_engine,
        export_correlations,
        run_id=1,
        db_path=db_path,
    )

    export_dir = tmp_path / "test_exports" / "run_1"

    # Check files exist correctly
    correlation_files = list(export_dir.glob("*_correlation_matrix.csv"))

    # Should have files (could be multiple FOVs)
    assert len(correlation_files) > 0

    # Check that all files are calcium_dff, not calcium_dec_dff
    for file in correlation_files:
        assert "calcium_dff_correlation" in file.name
        assert "calcium_dec_dff_correlation" not in file.name
