"""Test rising edge correlation CSV export integration with CaliRunner."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlmodel import Session, create_engine

from cali._constants import (
    INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES,
    INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES,
    INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES,
    INFERRED_SPIKES_SYNCHRONY_RISING_EDGES,
)
from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)


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

    for corr_type, pattern in expected_patterns.items():
        csv_files = list(export_dir.glob(pattern))
        assert len(csv_files) > 0, (
            f"Missing rising edge correlation files for {corr_type}: {pattern}"
        )

        # Check each file has valid content
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=0)
            assert len(df) > 0, f"Empty CSV file: {csv_file}"
            assert len(df.columns) > 0, f"No columns in CSV file: {csv_file}"


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

    # Only selected files should exist
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

    # These should NOT exist
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
