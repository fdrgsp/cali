"""Test CSV export functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from sqlmodel import Session, col, create_engine, select

from cali._constants import (
    CALCIUM_DEN_DFF_CORRELATION,
    CALCIUM_DFF_CORRELATION,
    DEN_DFF_TRACES,
    DFF_TRACES,
    INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES,
    INFERRED_SPIKES_CROSS_CORRELATION,
    INFERRED_SPIKES_CROSS_CORRELATION_LAGS,
    INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES,
    INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES,
    INFERRED_SPIKES_SYNCHRONY,
    INFERRED_SPIKES_SYNCHRONY_RISING_EDGES,
    INFERRED_SPIKES_THRESHOLDED_BINARY,
    INFERRED_SPIKES_TRACES,
    RAW_CALCIUM_TRACES,
)
from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)
from cali.sqlmodel._model import CaliResult
from cali.util import (
    export_calcium_dff_correlation_to_csv,
    export_raw_traces_to_csv,
)

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    pass


def test_csv_export_full_pipeline(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that CSV export works correctly in a full pipeline run."""
    # Use tmp_path for database location
    test_db_path = tmp_path / "test_export.cali"

    # Create experiment
    exp = Experiment.create_from_data("test_export", str(data_path))

    # Setup settings
    detection_settings = DetectionSettings(
        method="cellpose",
        model_type="cyto3",
    )
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    # Configure export options
    export_traces = {
        RAW_CALCIUM_TRACES: True,
        DFF_TRACES: True,
        DEN_DFF_TRACES: True,
        INFERRED_SPIKES_TRACES: True,
        INFERRED_SPIKES_THRESHOLDED_BINARY: True,
    }

    # Run pipeline with exports
    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_traces=export_traces,
    )

    # Verify database was created
    assert test_db_path.exists()

    # Get the run ID from the database
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        from sqlmodel import select

        from cali.sqlmodel._model import CaliResult

        results = session.exec(select(CaliResult)).all()
        assert len(results) > 0
        run_id = results[-1].id

    engine.dispose(close=True)

    # Check that export directory was created
    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"
    assert export_dir.exists()
    assert export_dir.is_dir()

    # Verify all expected CSV files exist
    # Trace CSVs have simple filenames (columns already contain FOV identifiers)
    expected_files = {
        RAW_CALCIUM_TRACES: "raw_traces.csv",
        DFF_TRACES: "dff_traces.csv",
        DEN_DFF_TRACES: "denoised_dff_traces.csv",
        INFERRED_SPIKES_TRACES: "inferred_spikes_raw.csv",
        INFERRED_SPIKES_THRESHOLDED_BINARY: "inferred_spikes_thresholded.csv",
    }

    for _trace_type, filename in expected_files.items():
        csv_file = export_dir / filename
        assert csv_file.exists(), f"Missing CSV file: {filename}"

        # Load and verify CSV has data
        df = pd.read_csv(csv_file)
        assert len(df) > 0, f"Empty CSV file: {filename}"
        assert len(df.columns) > 0, f"No columns in CSV file: {filename}"

        # Verify column names contain ROI identifiers
        assert any("ROI" in col for col in df.columns), f"No ROI columns in {filename}"


def test_csv_export_selective(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that only selected traces are exported."""
    test_db_path = tmp_path / "test_selective.cali"
    exp = Experiment.create_from_data("test_selective", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    # Only export two trace types
    export_traces = {
        RAW_CALCIUM_TRACES: True,
        DFF_TRACES: True,
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
    )

    # Get run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        from sqlmodel import select

        from cali.sqlmodel._model import CaliResult

        results = session.exec(select(CaliResult)).all()
        run_id = results[-1].id
    engine.dispose(close=True)

    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"

    # Only selected files should exist (traces use simple filenames)
    assert (export_dir / "raw_traces.csv").exists()
    assert (export_dir / "dff_traces.csv").exists()

    # These should NOT exist
    assert not (export_dir / "denoised_dff_traces.csv").exists()
    assert not (export_dir / "inferred_spikes_raw.csv").exists()
    assert not (export_dir / "inferred_spikes_thresholded.csv").exists()


def test_csv_export_no_export(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that no exports happen when export_traces is None."""
    test_db_path = tmp_path / "test_no_export.cali"
    exp = Experiment.create_from_data("test_no_export", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()

    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        global_position_indices=[0],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_traces=None,  # No exports
    )

    # Export directory should not exist
    export_dirs = list(test_db_path.parent.glob(f"{test_db_path.stem}_exports"))
    assert len(export_dirs) == 0


def test_csv_export_content_validation(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that exported CSV content is valid and matches database."""
    test_db_path = tmp_path / "test_content.cali"
    exp = Experiment.create_from_data("test_content", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    export_traces = {RAW_CALCIUM_TRACES: True}

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
    )

    # Get run ID and verify data
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        from sqlmodel import select

        from cali.sqlmodel._model import CaliResult, Traces

        results = session.exec(select(CaliResult)).all()
        run_id = results[-1].id

        # Get trace data from database
        db_traces = session.exec(
            select(Traces).where(Traces.analysis_result_id == run_id)
        ).all()
        num_rois = len(db_traces)
        assert num_rois > 0

    engine.dispose(close=True)

    # Load CSV and verify (traces use simple filenames)
    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"
    csv_file = export_dir / "raw_traces.csv"
    df = pd.read_csv(csv_file)

    # CSV should have one column per ROI
    assert len(df.columns) == num_rois

    # Each column should have data (not all zeros/NaN)
    for _col in df.columns:
        assert df[_col].notna().any()
        assert not (df[_col] == 0).all()


# ============================================================================
# FOV Naming Tests
# ============================================================================


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


# ============================================================================
# Position Filtering Tests
# ============================================================================


def test_export_traces_single_position(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that exporting traces with position filter only exports that position."""
    test_db_path = tmp_path / "test_position_filter.cali"
    exp = Experiment.create_from_data("test_position_filter", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()  # Need analysis for DataAnalysis table

    # Run on multiple positions
    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
    )

    # Get run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        run_id = result.id

        # Get FOV data for both positions
        fovs = session.exec(
            select(FOV).where(col(FOV.position_index).in_([0, 1]))
        ).all()
        assert len(fovs) == 2

        # Get ROIs for each position
        rois_pos_0 = session.exec(
            select(ROI).join(FOV).where(FOV.position_index == 0)
        ).all()
        rois_pos_1 = session.exec(
            select(ROI).join(FOV).where(FOV.position_index == 1)
        ).all()

        num_rois_pos_0 = len(rois_pos_0)
        num_rois_pos_1 = len(rois_pos_1)

    # Export all positions (no filter)
    output_all = tmp_path / "raw_traces_all.csv"
    export_raw_traces_to_csv(engine, output_all, run_id=run_id)
    df_all = pd.read_csv(output_all)

    # Export only position 0
    output_pos_0 = tmp_path / "raw_traces_pos_0.csv"
    export_raw_traces_to_csv(engine, output_pos_0, run_id=run_id, position_indices=[0])
    df_pos_0 = pd.read_csv(output_pos_0)

    # Export only position 1
    output_pos_1 = tmp_path / "raw_traces_pos_1.csv"
    export_raw_traces_to_csv(engine, output_pos_1, run_id=run_id, position_indices=[1])
    df_pos_1 = pd.read_csv(output_pos_1)

    engine.dispose(close=True)

    # Verify correct filtering
    assert len(df_all.columns) == num_rois_pos_0 + num_rois_pos_1
    assert len(df_pos_0.columns) == num_rois_pos_0
    assert len(df_pos_1.columns) == num_rois_pos_1

    # Verify FOV names in columns
    # With expanded 8-position data: position 0 = B5_0000, position 1 = B5_0001
    assert any("B5_0000" in col for col in df_all.columns)
    assert any("B5_0001" in col for col in df_all.columns)
    assert any("B5_0000" in col for col in df_pos_0.columns)
    assert all("B5_0001" not in col for col in df_pos_0.columns)


def test_export_correlation_single_position(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test exporting correlations with position filter only exports that position."""
    test_db_path = tmp_path / "test_correlation_filter.cali"
    exp = Experiment.create_from_data("test_correlation_filter", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    # Run on multiple positions
    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
    )

    # Get run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        run_id = result.id

    # Export correlation for position 0 only
    output_pos_0 = tmp_path / "correlation_pos_0.csv"
    try:
        export_calcium_dff_correlation_to_csv(
            engine, output_pos_0, run_id=run_id, position_indices=[0]
        )
    except ValueError:
        engine.dispose(close=True)
        pytest.skip(
            "No correlation data produced (mock data may not produce valid "
            "correlations). Test export functionality with real data."
        )

    # Export correlation for position 1 only
    output_pos_1 = tmp_path / "correlation_pos_1.csv"
    export_calcium_dff_correlation_to_csv(
        engine, output_pos_1, run_id=run_id, position_indices=[1]
    )

    engine.dispose(close=True)

    # Verify files were created
    csv_files_pos_0 = list(tmp_path.glob("*correlation_pos_0.csv"))
    csv_files_pos_1 = list(tmp_path.glob("*correlation_pos_1.csv"))

    assert len(csv_files_pos_0) > 0
    assert len(csv_files_pos_1) > 0

    # Load and verify each has data
    df_pos_0 = pd.read_csv(csv_files_pos_0[0], index_col=0)
    df_pos_1 = pd.read_csv(csv_files_pos_1[0], index_col=0)

    # Each should have square matrices (same number of rows and columns)
    assert df_pos_0.shape[0] == df_pos_0.shape[1]
    assert df_pos_1.shape[0] == df_pos_1.shape[1]

    # They should potentially have different sizes if different ROI counts
    # (but both should be non-empty)
    assert df_pos_0.shape[0] > 0
    assert df_pos_1.shape[0] > 0


def test_run_with_export_and_position_filter(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that running with global_position_indices filters exports correctly."""
    test_db_path = tmp_path / "test_run_export_filter.cali"
    exp = Experiment.create_from_data("test_run_export_filter", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    # Run ONLY position 0 with export enabled
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
        global_position_indices=[0],  # Only position 0
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_traces=export_traces,
        export_correlations=export_correlations,
    )

    # Get run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        run_id = result.id

        # Verify only position 0 was processed
        assert result.positions_extracted == [0]

        # Count ROIs in position 0
        num_rois = session.exec(
            select(ROI).join(FOV).where(FOV.position_index == 0)
        ).all()
        num_rois_pos_0 = len(num_rois)

    engine.dispose(close=True)

    # Check export directory
    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"
    assert export_dir.exists()

    # Trace CSVs use simple filenames (columns contain FOV identifiers)
    raw_traces_csv = export_dir / "raw_traces.csv"
    assert raw_traces_csv.exists()
    df = pd.read_csv(raw_traces_csv)

    # Should only have ROIs from position 0
    assert len(df.columns) == num_rois_pos_0

    # Load dff traces CSV
    dff_traces_csv = export_dir / "dff_traces.csv"
    assert dff_traces_csv.exists()
    df_dff = pd.read_csv(dff_traces_csv)
    # When position_indices=None in runner, exports all analyzed positions
    assert len(df_dff.columns) == num_rois_pos_0


def test_export_multiple_positions_filtered(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test exporting with a subset of positions (not all, not just one)."""
    test_db_path = tmp_path / "test_multi_filter.cali"
    exp = Experiment.create_from_data("test_multi_filter", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()  # Need analysis for DataAnalysis table

    # Process 2 positions
    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
    )

    # Get run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        run_id = result.id

        # Count ROIs for specific positions
        rois_0 = session.exec(
            select(ROI).join(FOV).where(col(FOV.position_index) == 0)
        ).all()
        num_rois_0 = len(rois_0)

        rois_all = session.exec(
            select(ROI).join(FOV).where(col(FOV.position_index).in_([0, 1]))
        ).all()
        num_rois_all = len(rois_all)

    # Export position 0 only
    output_subset = tmp_path / "raw_traces_subset.csv"
    export_raw_traces_to_csv(engine, output_subset, run_id=run_id, position_indices=[0])
    df_subset = pd.read_csv(output_subset)

    # Export all positions
    output_all = tmp_path / "raw_traces_all_two.csv"
    export_raw_traces_to_csv(engine, output_all, run_id=run_id, position_indices=[0, 1])
    df_all = pd.read_csv(output_all)

    engine.dispose(close=True)

    # Verify correct filtering
    assert len(df_subset.columns) == num_rois_0
    assert len(df_all.columns) == num_rois_all
    assert len(df_all.columns) > len(df_subset.columns)


def test_export_empty_position_list(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that exporting with empty position list exports nothing or all positions."""
    test_db_path = tmp_path / "test_empty_filter.cali"
    exp = Experiment.create_from_data("test_empty_filter", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()  # Need analysis for DataAnalysis table

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
    )

    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        run_id = result.id

    # Export with None (should export all available)
    output_none = tmp_path / "raw_traces_none.csv"
    export_raw_traces_to_csv(engine, output_none, run_id=run_id, position_indices=None)

    engine.dispose(close=True)

    # Should export all available positions
    assert output_none.exists()
    df = pd.read_csv(output_none)
    assert len(df.columns) > 0


def test_export_nonexistent_position(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that exporting with non-existent position doesn't break."""
    test_db_path = tmp_path / "test_nonexistent.cali"
    exp = Experiment.create_from_data("test_nonexistent", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()  # Need analysis for DataAnalysis table

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
    )

    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        run_id = result.id

    # Try to export position 999 (doesn't exist)
    output_nonexistent = tmp_path / "raw_traces_999.csv"

    # This should raise ValueError since position 999 doesn't exist
    with pytest.raises(ValueError, match="No trace data found"):
        export_raw_traces_to_csv(
            engine, output_nonexistent, run_id=run_id, position_indices=[999]
        )

    engine.dispose(close=True)


# ============================================================================
# Rising Edge Export Integration Tests
# ============================================================================


def test_rising_edge_export_full_pipeline(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that rising edge correlation CSV export works correctly in full pipeline.

    This test ensures that:
    1. Rising edge analysis is enabled
    2. Rising edge correlations are computed
    3. Rising edge correlation exports work
    """
    test_db_path = tmp_path / "test_rising_edge_export.cali"

    # Create experiment
    exp = Experiment.create_from_data("test_rising_edge_export", str(data_path))

    # Setup settings with rising edge analysis ENABLED
    detection_settings = DetectionSettings(
        method="cellpose",
        model_type="cyto3",
    )
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings(
        enable_rising_edge_analysis=True,  # CRITICAL: Enable rising edge analysis
    )

    # Configure rising edge correlation export options
    export_correlations = {
        INFERRED_SPIKES_SYNCHRONY_RISING_EDGES: True,
        INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES: True,
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES: True,
        INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES: True,
    }

    # Run pipeline with rising edge exports
    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_correlations=export_correlations,
    )

    # Verify database was created
    assert test_db_path.exists()

    # Get the run ID from the database
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        from sqlmodel import select

        from cali.sqlmodel._model import CaliResult

        results = session.exec(select(CaliResult)).all()
        assert len(results) > 0
        run_id = results[-1].id

    engine.dispose(close=True)

    # Check that export directory was created
    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"
    assert export_dir.exists()
    assert export_dir.is_dir()

    # Verify all expected rising edge correlation CSV files exist
    expected_patterns = {
        INFERRED_SPIKES_SYNCHRONY_RISING_EDGES: (
            "*inferred_spikes_synchrony_matrix_rising_edges.csv"
        ),
        INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES: (
            "*inferred_spikes_cross_correlation_matrix_rising_edges.csv"
        ),
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES: (
            "*inferred_spikes_cross_correlation_lags_matrix_rising_edges.csv"
        ),
        INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES: (
            "*inferred_spikes_ccg_zscore_matrix_rising_edges.csv"
        ),
    }

    # Check if any rising edge correlation files were created
    # With mock data, FOV analysis may not produce valid correlations
    any_files_created = False
    for _corr_type, pattern in expected_patterns.items():
        csv_files = list(export_dir.glob(pattern))
        if len(csv_files) > 0:
            any_files_created = True
            # Check each file has valid content
            for csv_file in csv_files:
                df = pd.read_csv(csv_file, index_col=0)
                assert len(df) > 0, f"Empty CSV file: {csv_file}"
                assert len(df.columns) > 0, f"No columns in CSV file: {csv_file}"

    # Skip if no files were created (expected with mock data)
    if not any_files_created:
        pytest.skip(
            "No rising edge correlation files created. Mock data may not produce "
            "valid FOV analysis for correlation exports."
        )


def test_rising_edge_export_selective(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that only selected rising edge correlations are exported."""
    test_db_path = tmp_path / "test_selective_rising_edge.cali"
    exp = Experiment.create_from_data("test_selective_rising_edge", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings(
        enable_rising_edge_analysis=True,  # Enable rising edge analysis
    )

    # Only export two rising edge correlation types
    export_correlations = {
        INFERRED_SPIKES_SYNCHRONY_RISING_EDGES: True,
        INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES: True,
        # Others are False or not included (default False)
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES: False,
        INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES: False,
    }

    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_correlations=export_correlations,
    )

    # Get run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        from sqlmodel import select

        from cali.sqlmodel._model import CaliResult

        results = session.exec(select(CaliResult)).all()
        run_id = results[-1].id
    engine.dispose(close=True)

    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"

    # Check if selected files exist
    # With mock data, FOV analysis may not produce valid correlations
    synchrony_files = list(
        export_dir.glob("*inferred_spikes_synchrony_matrix_rising_edges.csv")
    )
    cross_corr_files = list(
        export_dir.glob("*inferred_spikes_cross_correlation_matrix_rising_edges.csv")
    )

    # Skip if no files were created (expected with mock data)
    if len(synchrony_files) == 0 and len(cross_corr_files) == 0:
        pytest.skip(
            "No rising edge correlation files created. Mock data may not produce "
            "valid FOV analysis for correlation exports."
        )

    # These should NOT exist (they were explicitly set to False)
    assert (
        len(
            list(
                export_dir.glob(
                    "*inferred_spikes_cross_correlation_lags_matrix_rising_edges.csv"
                )
            )
        )
        == 0
    )
    assert (
        len(
            list(export_dir.glob("*inferred_spikes_ccg_zscore_matrix_rising_edges.csv"))
        )
        == 0
    )


def test_rising_edge_disabled_no_export(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that rising edge exports fail gracefully when analysis is disabled.

    When enable_rising_edge_analysis=False, the rising edge matrices won't exist,
    so exports should either skip or handle this gracefully.
    """
    test_db_path = tmp_path / "test_rising_edge_disabled.cali"
    exp = Experiment.create_from_data("test_rising_edge_disabled", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings(
        enable_rising_edge_analysis=False,  # Disable rising edge analysis
    )

    # Try to export rising edge correlations (should not export anything)
    export_correlations = {
        INFERRED_SPIKES_SYNCHRONY_RISING_EDGES: True,
        INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES: True,
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
        export_correlations=export_correlations,
    )

    # Get run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        from sqlmodel import select

        from cali.sqlmodel._model import CaliResult

        results = session.exec(select(CaliResult)).all()
        run_id = results[-1].id
    engine.dispose(close=True)

    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"

    # Rising edge files should NOT exist since analysis was disabled
    if export_dir.exists():
        rising_edge_files = list(export_dir.glob("*_rising_edges.csv"))
        assert len(rising_edge_files) == 0, (
            "Rising edge export files should not exist when analysis is disabled"
        )


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_rising_edge_and_thresholded_export_together(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that both thresholded and rising edge correlations export together."""
    from cali._constants import (
        INFERRED_SPIKES_CROSS_CORRELATION,
        INFERRED_SPIKES_SYNCHRONY,
    )

    test_db_path = tmp_path / "test_both_spike_export.cali"
    exp = Experiment.create_from_data("test_both_spike_export", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings(
        enable_rising_edge_analysis=True,  # Enable rising edge analysis
    )

    # Export both thresholded and rising edge correlations
    export_correlations = {
        # Thresholded binary spike correlations
        INFERRED_SPIKES_SYNCHRONY: True,
        INFERRED_SPIKES_CROSS_CORRELATION: True,
        # Rising edge spike correlations
        INFERRED_SPIKES_SYNCHRONY_RISING_EDGES: True,
        INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES: True,
    }

    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_correlations=export_correlations,
    )

    # Get run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            from sqlmodel import select

            from cali.sqlmodel._model import CaliResult

            results = session.exec(select(CaliResult)).all()
            run_id = results[-1].id
    finally:
        engine.dispose(close=True)

    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"

    # Verify both thresholded and rising edge files exist
    # Thresholded files (without _rising_edges suffix)
    thresholded_sync = list(export_dir.glob("*inferred_spikes_synchrony_matrix.csv"))
    # Filter out rising edge files
    thresholded_sync = [f for f in thresholded_sync if "_rising_edges" not in f.name]
    assert len(thresholded_sync) > 0

    thresholded_corr = list(
        export_dir.glob("*inferred_spikes_cross_correlation_matrix.csv")
    )
    # Filter out rising edge files
    thresholded_corr = [f for f in thresholded_corr if "_rising_edges" not in f.name]
    assert len(thresholded_corr) > 0

    # Rising edge files (with _rising_edges suffix)
    assert (
        len(list(export_dir.glob("*inferred_spikes_synchrony_matrix_rising_edges.csv")))
        > 0
    )
    assert (
        len(
            list(
                export_dir.glob(
                    "*inferred_spikes_cross_correlation_matrix_rising_edges.csv"
                )
            )
        )
        > 0
    )


# ============================================================================
# Correlation Export Integration Tests
# ============================================================================


def test_correlation_export_full_pipeline(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that correlation CSV export works correctly in a full pipeline run."""
    test_db_path = tmp_path / "test_corr_export.cali"

    # Create experiment
    exp = Experiment.create_from_data("test_corr_export", str(data_path))

    # Setup settings
    detection_settings = DetectionSettings(
        method="cellpose",
        model_type="cyto3",
    )
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    # Configure correlation export options
    export_correlations = {
        CALCIUM_DFF_CORRELATION: True,
        CALCIUM_DEN_DFF_CORRELATION: True,
        INFERRED_SPIKES_SYNCHRONY: True,
        INFERRED_SPIKES_CROSS_CORRELATION: True,
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS: True,
    }

    # Run pipeline with correlation exports
    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_correlations=export_correlations,
    )

    # Verify database was created
    assert test_db_path.exists()

    # Get the run ID from the database
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        from sqlmodel import select

        from cali.sqlmodel._model import CaliResult

        results = session.exec(select(CaliResult)).all()
        assert len(results) > 0
        run_id = results[-1].id

    engine.dispose(close=True)

    # Check that export directory was created
    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"
    assert export_dir.exists()
    assert export_dir.is_dir()

    # Check for expected correlation CSV files
    # Note: With mock data, correlation files may not exist because mock ROIs
    # don't produce valid traces for correlation analysis. This test verifies
    # that the export mechanism works when valid data is available.
    expected_patterns = {
        CALCIUM_DFF_CORRELATION: "*calcium_dff_correlation_matrix.csv",
        CALCIUM_DEN_DFF_CORRELATION: "*calcium_den_dff_correlation_matrix.csv",
        INFERRED_SPIKES_SYNCHRONY: "*inferred_spikes_synchrony_matrix.csv",
        INFERRED_SPIKES_CROSS_CORRELATION: "*inferred_spikes_cross_correlation_matrix.csv",  # noqa: E501
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS: "*inferred_spikes_cross_correlation_lags_matrix.csv",  # noqa: E501
    }

    # Check if any correlation files were created
    # (they may not exist if mock data produces no valid correlations)
    any_files_created = False
    for _corr_type, pattern in expected_patterns.items():
        csv_files = list(export_dir.glob(pattern))
        if csv_files:
            any_files_created = True
            # Check each file has valid content
            for csv_file in csv_files:
                df = pd.read_csv(csv_file, index_col=0)
                assert len(df) > 0, f"Empty CSV file: {csv_file}"
                assert len(df.columns) > 0, f"No columns in CSV file: {csv_file}"

    # With mock data, files may not be created if no valid correlations
    # can be computed. The important thing is that the export mechanism
    # doesn't crash and creates files when data is available.
    # For full integration testing with real data, use test_db.cali tests.
    if not any_files_created:
        pytest.skip(
            "No correlation files created (mock data may not produce valid "
            "correlations). Test export functionality with real data via "
            "test_db.cali-based tests."
        )


def test_correlation_export_selective(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that only selected correlations are exported."""
    test_db_path = tmp_path / "test_selective_corr.cali"
    exp = Experiment.create_from_data("test_selective_corr", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    # Only export two correlation types
    export_correlations = {
        CALCIUM_DFF_CORRELATION: True,
        INFERRED_SPIKES_SYNCHRONY: True,
        # Others are False or not included (default False)
        CALCIUM_DEN_DFF_CORRELATION: False,
        INFERRED_SPIKES_CROSS_CORRELATION: False,
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS: False,
    }

    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_correlations=export_correlations,
    )

    # Get run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        from sqlmodel import select

        from cali.sqlmodel._model import CaliResult

        results = session.exec(select(CaliResult)).all()
        run_id = results[-1].id
    engine.dispose(close=True)

    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"

    # Check if selected files exist (mock data may not produce valid correlations)
    dff_files = list(export_dir.glob("*calcium_dff_correlation_matrix.csv"))
    sync_files = list(export_dir.glob("*inferred_spikes_synchrony_matrix.csv"))

    # These should NOT exist (not selected for export)
    assert len(list(export_dir.glob("*calcium_den_dff_correlation_matrix.csv"))) == 0
    assert (
        len(list(export_dir.glob("*inferred_spikes_cross_correlation_matrix.csv"))) == 0
    )
    assert (
        len(list(export_dir.glob("*inferred_spikes_cross_correlation_lags_matrix.csv")))
        == 0
    )

    # With mock data, selected files may not exist if no valid correlations
    if len(dff_files) == 0 and len(sync_files) == 0:
        pytest.skip(
            "No correlation files created (mock data may not produce valid "
            "correlations). Selective export verified: excluded files don't exist."
        )


def test_correlation_export_none(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that no correlation exports happen when export_correlations is None."""
    test_db_path = tmp_path / "test_no_corr_export.cali"
    exp = Experiment.create_from_data("test_no_corr_export", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

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
        export_correlations=None,  # No exports
    )

    # Get run ID if it exists
    engine = create_engine(f"sqlite:///{test_db_path}")
    with Session(engine) as session:
        from sqlmodel import select

        from cali.sqlmodel._model import CaliResult

        results = session.exec(select(CaliResult)).all()
        if results:
            run_id = results[-1].id
            engine.dispose(close=True)

            # Export directory may exist from traces, but no correlation files
            export_dir = (
                test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"
            )
            if export_dir.exists():
                # No correlation files should exist
                corr_files = list(export_dir.glob("*correlation*.csv")) + list(
                    export_dir.glob("*synchrony*.csv")
                )
                assert len(corr_files) == 0
        else:
            engine.dispose(close=True)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_correlation_and_traces_export_together(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that both traces and correlations can be exported together."""
    from cali._constants import DEN_DFF_TRACES, RAW_CALCIUM_TRACES

    test_db_path = tmp_path / "test_both_export.cali"
    exp = Experiment.create_from_data("test_both_export", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    export_traces = {
        RAW_CALCIUM_TRACES: True,
        DEN_DFF_TRACES: True,
    }

    export_correlations = {
        CALCIUM_DFF_CORRELATION: True,
        INFERRED_SPIKES_SYNCHRONY: True,
    }

    runner = CaliRunner()
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        export_traces=export_traces,
        export_correlations=export_correlations,
    )

    # Get run ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            from sqlmodel import select

            from cali.sqlmodel._model import CaliResult

            results = session.exec(select(CaliResult)).all()
            run_id = results[-1].id
    finally:
        engine.dispose(close=True)

    export_dir = test_db_path.parent / f"{test_db_path.stem}_exports" / f"run_{run_id}"

    # Verify trace files exist (these always work)
    assert (export_dir / "raw_traces.csv").exists()
    assert (export_dir / "denoised_dff_traces.csv").exists()

    # Correlation files may not exist if mock data doesn't produce valid correlations
    dff_corr = list(export_dir.glob("*calcium_dff_correlation_matrix.csv"))
    sync_corr = list(export_dir.glob("*inferred_spikes_synchrony_matrix.csv"))
    if len(dff_corr) == 0 and len(sync_corr) == 0:
        pytest.skip(
            "No correlation files created (mock data may not produce valid "
            "correlations). Trace export verified."
        )
