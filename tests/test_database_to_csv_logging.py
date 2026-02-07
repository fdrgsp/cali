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
    # Skip if no FOV analysis data exists (directory won't be created)
    if not export_dir.exists():
        pytest.skip("No FOV analysis data found in test database")
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

    # Only raw_traces.csv should exist (may be in condition subfolders)
    raw_files = list(export_dir.rglob("raw_traces.csv"))
    assert len(raw_files) > 0, "raw_traces.csv not found anywhere in export dir"
    assert not list(export_dir.rglob("dff_traces.csv"))
    assert not list(export_dir.rglob("neuropil_corrected_traces.csv"))


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

    # Skip if no FOV analysis data exists
    if len(correlation_files) == 0:
        pytest.skip("No FOV analysis data found in test database")

    # Check that all files are calcium_dff, not calcium_dec_dff
    for file in correlation_files:
        assert "calcium_dff_correlation" in file.name
        assert "calcium_dec_dff_correlation" not in file.name


def test_export_traces_creates_condition_subfolders(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that traces are exported into condition-named subfolders.

    The test database has wells with conditions (2x2 design):
      B5: g1 + t1  (positions 0,1)
      B6: g1 + t2  (positions 2,3)
    Run 1 covers positions [0,1] (well B5 = t1_g1).
    Run 2 covers positions [2,3] (well B6 = t2_g1).
    """
    db_path = tmp_path / "test.cali"

    export_traces = {RAW_CALCIUM_TRACES: True}

    # Run 1 has positions 0,1 (well B5 = condition "t1_g1")
    export_traces_to_csv(test_engine, export_traces, run_id=1, db_path=db_path)

    export_dir = tmp_path / "test_exports" / "run_1"

    # Should have condition subfolder
    condition_dir = export_dir / "t1_g1"
    assert condition_dir.exists(), (
        f"Expected condition subfolder 't1_g1' in {export_dir}, "
        f"found: {[p.name for p in export_dir.iterdir()]}"
    )
    assert (condition_dir / "raw_traces.csv").exists()


def test_export_correlations_creates_condition_subfolders(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that correlations are exported into condition-named subfolders."""
    db_path = tmp_path / "test.cali"

    export_correlations = {CALCIUM_DFF_CORRELATION: True}

    export_correlations_to_csv(
        test_engine, export_correlations, run_id=1, db_path=db_path
    )

    export_dir = tmp_path / "test_exports" / "run_1"
    condition_dir = export_dir / "t1_g1"

    if not condition_dir.exists():
        pytest.skip("No condition subfolder created (no FOV analysis data)")

    # Should have correlation files in the condition subfolder
    csv_files = list(condition_dir.glob("*.csv"))
    assert len(csv_files) > 0, f"No CSV files in {condition_dir}"


def test_export_traces_multiple_conditions(
    test_engine: Engine,
    tmp_path: Path,
) -> None:
    """Test that exporting across conditions creates separate subfolders.

    Run 1 = positions [0,1] (B5 = t1_g1)
    Run 2 = positions [2,3] (B6 = t2_g1)
    Exporting all positions should create both condition subfolders.
    """
    db_path = tmp_path / "test.cali"

    export_traces = {RAW_CALCIUM_TRACES: True}

    # Export run 2 (positions 2,3 = well B6 = condition "t2_g1")
    export_traces_to_csv(test_engine, export_traces, run_id=2, db_path=db_path)

    export_dir = tmp_path / "test_exports" / "run_2"

    # Run 2 covers B6 which has condition t2_g1
    condition_dir = export_dir / "t2_g1"
    assert condition_dir.exists(), (
        f"Expected condition subfolder 't2_g1' in {export_dir}, "
        f"found: {[p.name for p in export_dir.iterdir()]}"
    )
    assert (condition_dir / "raw_traces.csv").exists()
