"""Tests for manual pipeline execution."""

from pathlib import Path

import pytest
from sqlmodel import Session, create_engine, func, select

from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner
from cali.extraction import ExtractionRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    DataAnalysis,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    Traces,
    save_experiment_to_database,
)
from cali.util import load_fovs_from_database, update_fovs_in_database
from cali.util._util import load_data_from_path


def test_manual_pipeline_execution(tmp_path: Path) -> None:
    """Test manual execution of the pipeline components."""
    # Setup paths
    dataset_path = Path("tests/test_data/evoked/evk.tensorstore.zarr")
    if not dataset_path.exists():
        pytest.skip(f"Test data not found at {dataset_path}")

    db_path = tmp_path / "manual_run.cali"

    # Create and save experiment
    exp = Experiment.create_from_data("manual_exp", str(dataset_path))
    save_experiment_to_database(exp, db_path.parent, database_name=db_path.name)

    # Create engine
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"timeout": 30.0, "check_same_thread": False},
        pool_pre_ping=True,
    )

    try:
        # Load data
        data = load_data_from_path(dataset_path)
        assert data is not None
        assert data.sequence is not None

        # Use only position 0
        positions_to_process = [0]

        # 1. Detection
        detection_runner = DetectionRunner()
        # Use standard model for testing
        detection_settings = DetectionSettings(
            method="cellpose",
            model_type="cpsam",
            diameter=30.0,
        )

        for fov in detection_runner.run(
            dataset=data,
            detection_settings=detection_settings,
            global_position_indices=positions_to_process,
            as_generator=True,
        ):
            update_fovs_in_database(db_path, fov)

        # 2. Extraction
        extraction_runner = ExtractionRunner()
        extraction_settings = ExtractionSettings(dff_window=150, threads=1)

        # We need to load FOVs from DB to get the ROIs created in step 1
        fovs_for_extraction = load_fovs_from_database(engine, positions_to_process)
        assert len(fovs_for_extraction) > 0
        assert len(fovs_for_extraction[0].rois) > 0

        for fov in extraction_runner.run(
            dataset=data,
            extraction_settings=extraction_settings,
            fovs=fovs_for_extraction,
            as_generator=True,
        ):
            update_fovs_in_database(db_path, fov)

        # 3. Analysis
        analysis_runner = AnalysisRunner()
        analysis_settings = AnalysisSettings(peaks_height_value=2, threads=1)

        # Load FOVs again to get traces created in step 2
        fovs_for_analysis = load_fovs_from_database(engine, positions_to_process)
        # Verify traces exist
        assert len(fovs_for_analysis[0].rois[0].traces_history) > 0

        for fov in analysis_runner.run(
            fovs_for_analysis,
            analysis_settings=analysis_settings,
            as_generator=True,
        ):
            update_fovs_in_database(db_path, fov)

        # Verify results
        with Session(engine) as session:
            for pos in positions_to_process:
                fov = session.exec(select(FOV).where(FOV.position_index == pos)).first()
                assert fov is not None

                # Check ROIs
                assert len(fov.rois) > 0

                # Check Traces
                trace_count = session.exec(
                    select(func.count(Traces.id))  # type: ignore
                    .join(ROI)
                    .where(ROI.fov_id == fov.id)
                ).one()
                assert trace_count > 0

                # Check Analysis
                analysis_count = session.exec(
                    select(func.count(DataAnalysis.id))  # type: ignore
                    .join(ROI)
                    .where(ROI.fov_id == fov.id)
                ).one()
                assert analysis_count > 0

    finally:
        engine.dispose()
