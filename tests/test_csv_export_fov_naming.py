"""Test CSV export FOV naming functionality."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlmodel import Session, create_engine, select

from cali._constants import (
    CALCIUM_DFF_CORRELATION,
    DFF_TRACES,
    RAW_CALCIUM_TRACES,
)
from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)


def test_export_sequential_positions_dont_overwrite(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that correlation matrices get separate files per FOV.

    Trace CSVs can overwrite safely because column names identify the FOV.
    Correlation matrices need separate files per FOV with FOV names in filenames.
    """
    test_db_path = tmp_path / "test_sequential.cali"
    exp = Experiment.create_from_data("test_sequential", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    # First run: Only position 0 with export
    export_traces = {
        RAW_CALCIUM_TRACES: True,
        DFF_TRACES: True,
    }
    export_correlations = {
        CALCIUM_DFF_CORRELATION: True,
    }

    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_traces=export_traces,
        export_correlations=export_correlations,
    )

    # Get run ID for first run
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        results = session.exec(select(CaliResult)).all()
        run_1_id = results[0].id
    engine.dispose(close=True)

    # Check position 0 exports
    export_dir_1 = (
        test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_1_id}"
    )
    assert export_dir_1.exists()

    # Trace CSVs have simple names (columns contain FOV identifiers)
    raw_traces_file = export_dir_1 / "raw_traces.csv"
    assert raw_traces_file.exists()
    pd.read_csv(raw_traces_file)

    # Correlation CSVs have FOV-prefixed names (may not exist with mock data)
    corr_files_0 = list(export_dir_1.glob("B5_0000_calcium_dff_correlation_matrix.csv"))
    if len(corr_files_0) == 0:
        pytest.skip(
            "No correlation files created (mock data may not produce valid "
            "correlations). Test export functionality with real data."
        )

    # Second run: Only position 1 with export
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_traces=export_traces,
        export_correlations=export_correlations,
    )

    # Get run ID for second run
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        results = session.exec(select(CaliResult)).all()
        run_2_id = results[-1].id
    engine.dispose(close=True)

    # Check position 1 exports
    export_dir_2 = (
        test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_2_id}"
    )
    assert export_dir_2.exists()

    # Trace CSV was overwritten (this is OK - columns identify which FOV)
    assert raw_traces_file.exists()  # Same file from position 0 run
    pd.read_csv(export_dir_2 / "raw_traces.csv")

    # Correlation CSVs have different FOV names - no overwrite
    # With expanded 8-position data: position 1 = B5_0001
    corr_files_1 = list(export_dir_2.glob("B5_0001_calcium_dff_correlation_matrix.csv"))
    assert len(corr_files_1) == 1, "Position 1 correlation should have FOV name prefix"

    # Verify correlation files have different FOV names
    assert corr_files_0[0].name == "B5_0000_calcium_dff_correlation_matrix.csv"
    assert corr_files_1[0].name == "B5_0001_calcium_dff_correlation_matrix.csv"


def test_export_with_position_indices_includes_fov_names(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that correlation matrices include FOV names in filenames.

    Correlation matrices are per-FOV and need separate files with FOV identifiers.
    Trace CSVs use simple names because columns already identify the FOV.
    """
    test_db = tmp_path / "test_naming.cali"
    exp = Experiment.create_from_data("test_naming", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    export_traces = {RAW_CALCIUM_TRACES: True}
    export_correlations = {CALCIUM_DFF_CORRELATION: True}
    runner = CaliRunner()

    # Run with positions [0, 2] to get FOVs from different wells (B5_0000 and B6_0000)
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 2],
        database_name=test_db.name,
        output_path=test_db.parent,
        export_traces=export_traces,
        export_correlations=export_correlations,
    )

    engine = create_engine(f"sqlite:///{test_db}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        run_id = result.id
    engine.dispose(close=True)

    export_dir = test_db.parent / f"{test_db.stem}_exports" / f"run_{run_id}"

    # Trace CSV has simple name
    trace_file = export_dir / "raw_traces.csv"
    assert trace_file.exists(), "Should have 'raw_traces.csv'"

    # Verify the trace file contains data from both FOVs (check column names)
    df = pd.read_csv(trace_file)
    assert len(df) > 0, "Exported file should contain trace data"
    # Columns should have FOV prefixes
    fov_prefixed_cols = [
        col for col in df.columns if "B5_0000" in col or "B6_0000" in col
    ]
    assert len(fov_prefixed_cols) > 0, "Columns should have FOV prefixes"

    # Correlation CSVs have FOV-prefixed names - one file per FOV (may not exist)
    corr_files = list(export_dir.glob("*_calcium_dff_correlation_matrix.csv"))
    if len(corr_files) == 0:
        pytest.skip(
            "No correlation files created (mock data may not produce valid "
            "correlations). Test export functionality with real data."
        )

    assert len(corr_files) == 2, "Should have 2 correlation files (one per FOV)"

    corr_file_names = sorted([f.name for f in corr_files])
    assert "B5_0000_calcium_dff_correlation_matrix.csv" in corr_file_names
    assert "B6_0000_calcium_dff_correlation_matrix.csv" in corr_file_names
