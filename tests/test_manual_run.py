"""Tests for manual pipeline execution."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
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
    Mask,
    Traces,
    save_experiment_to_database,
)
from cali.util import load_fovs_from_database, update_fovs_in_database
from cali.util._util import load_data_from_path

THREADS = 1


def create_mock_fov(position_index: int = 0, num_rois: int = 3) -> FOV:
    """Create a mock FOV with ROIs for testing without running cellpose."""
    # Use A1_ naming convention for well parsing
    fov = FOV(position_index=position_index, name=f"A1_{position_index:04d}")

    rois = []
    for i in range(1, num_rois + 1):
        # Create a simple circular mask matching dataset dims (256x256)
        mask_data = np.zeros((256, 256), dtype=np.uint8)
        cy, cx = 50 + i * 20, 50 + i * 20
        y, x = np.ogrid[:256, :256]
        mask_region = ((x - cx) ** 2 + (y - cy) ** 2) <= 100
        mask_data[mask_region] = 1

        # Get coordinates from mask
        coords = np.where(mask_data)
        coords_y = coords[0].tolist()
        coords_x = coords[1].tolist()

        mask = Mask(
            mask_type="roi",
            coords_y=coords_y,
            coords_x=coords_x,
            height=256,
            width=256,
        )

        roi = ROI(
            label_value=i,
            roi_mask=mask,
            fov_id=0,  # Dummy ID, will be handled by relationship
        )
        rois.append(roi)

    fov.rois = rois
    return fov


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
            model_type="cyto3",
            diameter=30.0,
        )

        # Mock cellpose execution
        with patch(
            "cali.detection._detection_runner.DetectionRunner._run_cellpose"
        ) as mock_run:

            def mock_side_effect(
                dataset: Any,
                detection_settings: Any,
                position_indices: list[int],
                *args: Any,
                **kwargs: Any,
            ) -> Iterator[FOV]:
                for pos_idx in position_indices:
                    yield create_mock_fov(pos_idx)

            mock_run.side_effect = mock_side_effect

            # Run detection (mocked)
            fovs = list(
                detection_runner.run(
                    dataset=data,
                    detection_settings=detection_settings,
                    global_position_indices=positions_to_process,
                )
            )

        assert len(fovs) == 1
        assert len(fovs[0].rois) == 3

        # Save FOVs to database
        update_fovs_in_database(engine, fovs)

        # Verify saved
        with Session(engine) as session:
            saved_fovs = session.exec(select(FOV)).all()
            assert len(saved_fovs) == 1
            assert len(saved_fovs[0].rois) == 3

        # 2. Extraction
        extraction_runner = ExtractionRunner()
        extraction_settings = ExtractionSettings(dff_window=150, threads=THREADS)

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
        analysis_settings = AnalysisSettings(peaks_height_value=2, threads=THREADS)

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
