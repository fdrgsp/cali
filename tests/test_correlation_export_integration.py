"""Test correlation CSV export integration with CaliRunner."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlmodel import Session, create_engine

from cali._constants import (
    CALCIUM_DEC_DFF_CORRELATION,
    CALCIUM_DFF_CORRELATION,
    INFERRED_SPIKES_CROSS_CORRELATION,
    INFERRED_SPIKES_CROSS_CORRELATION_LAGS,
    INFERRED_SPIKES_SYNCHRONY,
)
from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)


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
        CALCIUM_DEC_DFF_CORRELATION: True,
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

    # Verify all expected correlation CSV files exist
    # Note: Correlation files have FOV prefixes since there are multiple FOVs
    expected_patterns = {
        CALCIUM_DFF_CORRELATION: "*calcium_dff_correlation_matrix.csv",
        CALCIUM_DEC_DFF_CORRELATION: "*calcium_dec_dff_correlation_matrix.csv",
        INFERRED_SPIKES_SYNCHRONY: "*inferred_spikes_synchrony_matrix.csv",
        INFERRED_SPIKES_CROSS_CORRELATION: "*inferred_spikes_cross_correlation_matrix.csv",  # noqa: E501
        INFERRED_SPIKES_CROSS_CORRELATION_LAGS: "*inferred_spikes_cross_correlation_lags_matrix.csv",  # noqa: E501
    }

    for corr_type, pattern in expected_patterns.items():
        csv_files = list(export_dir.glob(pattern))
        assert len(csv_files) > 0, (
            f"Missing correlation files for {corr_type}: {pattern}"
        )

        # Check each file has valid content
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=0)
            assert len(df) > 0, f"Empty CSV file: {csv_file}"
            assert len(df.columns) > 0, f"No columns in CSV file: {csv_file}"


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
        CALCIUM_DEC_DFF_CORRELATION: False,
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

    # Only selected files should exist
    assert len(list(export_dir.glob("*calcium_dff_correlation_matrix.csv"))) > 0
    assert len(list(export_dir.glob("*inferred_spikes_synchrony_matrix.csv"))) > 0

    # These should NOT exist
    assert len(list(export_dir.glob("*calcium_dec_dff_correlation_matrix.csv"))) == 0
    assert (
        len(list(export_dir.glob("*inferred_spikes_cross_correlation_matrix.csv"))) == 0
    )
    assert (
        len(list(export_dir.glob("*inferred_spikes_cross_correlation_lags_matrix.csv")))
        == 0
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
    from cali._constants import DEC_DFF_TRACES, RAW_CALCIUM_TRACES

    test_db_path = tmp_path / "test_both_export.cali"
    exp = Experiment.create_from_data("test_both_export", str(data_path))

    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    export_traces = {
        RAW_CALCIUM_TRACES: True,
        DEC_DFF_TRACES: True,
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

    # Verify both trace and correlation files exist
    assert (export_dir / "raw_traces.csv").exists()
    assert (export_dir / "deconvolved_dff_traces.csv").exists()
    assert len(list(export_dir.glob("*calcium_dff_correlation_matrix.csv"))) > 0
    assert len(list(export_dir.glob("*inferred_spikes_synchrony_matrix.csv"))) > 0
