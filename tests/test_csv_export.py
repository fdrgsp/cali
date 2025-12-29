"""Test CSV export functionality."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
from sqlmodel import Session, create_engine

from cali._constants import (
    DEC_DFF_TRACES,
    DFF_TRACES,
    INFERRED_SPIKES_THRESHOLDED_BINARY,
    INFERRED_SPIKES_TRACES,
    RAW_CALCIUM_TRACES,
)
from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)


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
        DEC_DFF_TRACES: True,
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
        DEC_DFF_TRACES: "deconvolved_dff_traces.csv",
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
    assert not (export_dir / "deconvolved_dff_traces.csv").exists()
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
    for col in df.columns:
        assert df[col].notna().any()
        assert not (df[col] == 0).all()
