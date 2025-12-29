"""Test CSV export position filtering functionality."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlmodel import Session, col, create_engine, select

from cali._constants import (
    CALCIUM_DFF_CORRELATION,
    DFF_TRACES,
    RAW_CALCIUM_TRACES,
)
from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)
from cali.util import (
    export_calcium_dff_correlation_to_csv,
    export_raw_traces_to_csv,
)


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
    assert any("B5_0000" in col for col in df_all.columns)
    assert any("B6_0000" in col for col in df_all.columns)
    assert any("B5_0000" in col for col in df_pos_0.columns)
    assert all("B6_0000" not in col for col in df_pos_0.columns)


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
    export_calcium_dff_correlation_to_csv(
        engine, output_pos_0, run_id=run_id, position_indices=[0]
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
