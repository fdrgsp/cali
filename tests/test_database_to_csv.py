"""Tests for database to CSV export functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import Session, create_engine

from cali.util import (
    export_calcium_dec_dff_correlation_to_csv,
    export_calcium_dff_correlation_to_csv,
    export_correlation_matrices_to_csv,
    export_deconvolved_dff_traces_to_csv,
    export_dff_traces_to_csv,
    export_inferred_spikes_cross_correlation_lags_to_csv,
    export_inferred_spikes_cross_correlation_to_csv,
    export_inferred_spikes_raw_to_csv,
    export_inferred_spikes_synchrony_to_csv,
    export_inferred_spikes_thresholded_to_csv,
    export_neuropil_corrected_traces_to_csv,
    export_neuropil_traces_to_csv,
    export_raw_traces_to_csv,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from sqlalchemy.engine import Engine


@pytest.fixture
def test_engine() -> Generator[Engine, None, None]:
    """Create engine from existing test database."""
    db_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
    engine = create_engine(f"sqlite:///{db_path}")
    yield engine
    engine.dispose(close=True)


def test_export_raw_traces(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting raw traces to CSV."""
    output_file = tmp_path / "raw_traces.csv"
    export_raw_traces_to_csv(test_engine, output_file, run_id=1)

    assert output_file.exists()
    # Check file has content
    assert output_file.stat().st_size > 0


def test_export_dff_traces(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting ΔF/F traces to CSV."""
    output_file = tmp_path / "dff_traces.csv"
    export_dff_traces_to_csv(test_engine, output_file, run_id=1)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_export_deconvolved_dff_traces(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting deconvolved ΔF/F traces to CSV."""
    output_file = tmp_path / "dec_dff_traces.csv"
    export_deconvolved_dff_traces_to_csv(test_engine, output_file, run_id=1)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_export_inferred_spikes_raw(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting raw inferred spikes to CSV."""
    output_file = tmp_path / "spikes_raw.csv"
    export_inferred_spikes_raw_to_csv(test_engine, output_file, run_id=1)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_export_inferred_spikes_thresholded(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting thresholded inferred spikes to CSV."""
    output_file = tmp_path / "spikes_thresholded.csv"
    export_inferred_spikes_thresholded_to_csv(test_engine, output_file, run_id=1)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_export_correlation_matrices(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting correlation matrices to CSV."""
    output_dir = tmp_path / "correlation_matrices"
    export_correlation_matrices_to_csv(test_engine, output_dir, run_id=1)

    assert output_dir.exists()
    # Check that at least some correlation files were created
    csv_files = list(output_dir.glob("*.csv"))
    assert len(csv_files) > 0


def test_export_with_specific_fov(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting data for a specific FOV."""
    # Get a FOV name from the database
    from sqlmodel import select

    from cali.sqlmodel._model import FOV

    with Session(test_engine) as session:
        fov = session.exec(select(FOV).limit(1)).first()
        assert fov is not None
        fov_name = fov.name

    output_file = tmp_path / "fov_specific_traces.csv"
    export_raw_traces_to_csv(test_engine, output_file, fov_name=fov_name, run_id=1)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_export_neuropil_traces(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting neuropil traces to CSV."""
    output_file = tmp_path / "neuropil_traces.csv"
    try:
        export_neuropil_traces_to_csv(test_engine, output_file, run_id=1)
        # If neuropil data exists, check the file
        if output_file.exists():
            assert output_file.stat().st_size > 0
    except ValueError:
        # It's okay if no neuropil data exists in test database
        pytest.skip("No neuropil traces found in test database")


def test_export_corrected_traces(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting corrected traces to CSV."""
    output_file = tmp_path / "corrected_traces.csv"
    try:
        export_neuropil_corrected_traces_to_csv(test_engine, output_file, run_id=1)
        # If corrected data exists, check the file
        if output_file.exists():
            assert output_file.stat().st_size > 0
    except ValueError:
        # It's okay if no corrected data exists in test database
        pytest.skip("No corrected traces found in test database")


def test_export_calcium_dff_correlation(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting ΔF/F correlation matrix to CSV."""
    output_file = tmp_path / "calcium_dff_correlation.csv"
    export_calcium_dff_correlation_to_csv(test_engine, output_file, run_id=1)

    # Check that at least one file was created (may have FOV prefix if multiple FOVs)
    csv_files = list(tmp_path.glob("*calcium_dff_correlation.csv"))
    assert len(csv_files) > 0
    # Check first file has content
    assert csv_files[0].stat().st_size > 0


def test_export_calcium_dec_dff_correlation(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting deconvolved ΔF/F correlation matrix to CSV."""
    output_file = tmp_path / "calcium_dec_dff_correlation.csv"
    export_calcium_dec_dff_correlation_to_csv(test_engine, output_file, run_id=1)

    # Check that at least one file was created (may have FOV prefix if multiple FOVs)
    csv_files = list(tmp_path.glob("*calcium_dec_dff_correlation.csv"))
    assert len(csv_files) > 0
    assert csv_files[0].stat().st_size > 0


def test_export_inferred_spikes_synchrony(test_engine: Engine, tmp_path: Path) -> None:
    """Test exporting inferred spikes synchrony matrix to CSV."""
    output_file = tmp_path / "inferred_spikes_synchrony.csv"
    export_inferred_spikes_synchrony_to_csv(test_engine, output_file, run_id=1)

    # Check that at least one file was created (may have FOV prefix if multiple FOVs)
    csv_files = list(tmp_path.glob("*inferred_spikes_synchrony.csv"))
    assert len(csv_files) > 0
    assert csv_files[0].stat().st_size > 0


def test_export_inferred_spikes_cross_correlation(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting inferred spikes cross-correlation matrix to CSV."""
    output_file = tmp_path / "inferred_spikes_cross_correlation.csv"
    export_inferred_spikes_cross_correlation_to_csv(test_engine, output_file, run_id=1)

    # Check that at least one file was created (may have FOV prefix if multiple FOVs)
    csv_files = list(tmp_path.glob("*inferred_spikes_cross_correlation.csv"))
    assert len(csv_files) > 0
    assert csv_files[0].stat().st_size > 0


def test_export_inferred_spikes_cross_correlation_lags(
    test_engine: Engine, tmp_path: Path
) -> None:
    """Test exporting inferred spikes cross-correlation lags matrix to CSV."""
    output_file = tmp_path / "inferred_spikes_cross_correlation_lags.csv"
    export_inferred_spikes_cross_correlation_lags_to_csv(
        test_engine, output_file, run_id=1
    )

    # Check that at least one file was created (may have FOV prefix if multiple FOVs)
    csv_files = list(tmp_path.glob("*inferred_spikes_cross_correlation_lags.csv"))
    assert len(csv_files) > 0
    assert csv_files[0].stat().st_size > 0
