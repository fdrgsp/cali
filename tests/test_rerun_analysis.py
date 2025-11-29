"""Test re-running analysis (regression test for identity map issue)."""

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    Mask,
)

MODEL = "cpsam"


def create_mock_fov(position_index: int = 0, num_rois: int = 3) -> FOV:
    """Create a mock FOV with ROIs for testing without running cellpose."""
    fov = FOV(position_index=position_index, name=f"A1_{position_index:04d}")

    rois = []
    for i in range(1, num_rois + 1):
        # Create a simple circular mask matching dataset dims (256x256)
        mask_data = np.zeros((256, 256), dtype=np.uint8)
        cy, cx = 50 + i * 20, 50 + i * 20
        y, x = np.ogrid[:256, :256]
        circle_mask = ((y - cy) ** 2 + (x - cx) ** 2) <= 10**2
        mask_data[circle_mask] = i

        # Convert to sparse coordinates
        coords_y, coords_x = np.where(mask_data == i)

        roi_mask = Mask(
            coords_y=coords_y.tolist(),
            coords_x=coords_x.tolist(),
            height=256,
            width=256,
            mask_type="roi",
        )

        roi = ROI(
            label_value=i,
            active=True,
            stimulated=False,
            roi_mask=roi_mask,
        )
        rois.append(roi)

    fov.rois = rois
    return fov


@pytest.fixture
def mock_detection_runner() -> Generator[MagicMock, None, None]:
    """Fixture that patches DetectionRunner to return mock FOVs quickly."""
    with patch(
        "cali.detection._detection_runner.DetectionRunner._run_cellpose"
    ) as mock:

        def mock_detection(
            dataset: Any,
            detection_settings: Any,
            position_indices: list[int],
            *args: Any,
            **kwargs: Any,
        ) -> Generator[FOV, None, None]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection
        yield mock


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test_rerun.cali"


@pytest.fixture
def test_experiment() -> Experiment:
    """Create a test experiment."""
    return Experiment(name="Test Rerun Experiment")


@pytest.fixture
def data_path() -> Path:
    """Path to test data."""
    return Path("tests/test_data/spontaneous/spont.tensorstore.zarr")


def test_rerun_analysis_same_settings(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test re-running analysis with same settings doesn't cause identity map conflicts.

    This is a regression test for the error:
    sqlalchemy.exc.InvalidRequestError: Can't attach instance <Traces at 0x...>;
    another instance with key (..., (5,), None) is already present in this session.
    """
    runner = CaliRunner()

    ds = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)
    es = ExtractionSettings(neuropil_inner_radius=10)
    as_ = AnalysisSettings(peaks_height_value=10.0)

    # First run - full pipeline
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds,
        extraction_settings=es,
        analysis_settings=as_,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify first run created result
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            assert len(results) == 1
            result1_id = results[0].id
            ds_id = results[0].detection_settings
            es_id = results[0].extraction_settings_id
            assert ds_id is not None
            assert es_id is not None
    finally:
        engine.dispose()

    # Second run - new analysis settings on same detection/extraction (using IDs)
    # This should create a new CaliResult but reuse detection
    as2 = AnalysisSettings(peaks_height_value=15.0)  # Different threshold

    # This should NOT raise InvalidRequestError
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_id,
        extraction_settings=es_id,
        analysis_settings=as2,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify second run created new result
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            assert len(results) == 2
            # Should have two different results
            result_ids = {r.id for r in results}
            assert len(result_ids) == 2
            assert result1_id in result_ids
    finally:
        engine.dispose()


def test_rerun_extraction_on_existing_detection(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner: MagicMock,
) -> None:
    """Test re-running extraction+analysis on existing detection.

    This is the exact scenario from the user's error log.
    """
    runner = CaliRunner()

    ds = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)
    es1 = ExtractionSettings(neuropil_inner_radius=10)
    as1 = AnalysisSettings(peaks_height_value=10.0)

    # First run - full pipeline
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds,
        extraction_settings=es1,
        analysis_settings=as1,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Get the detection settings ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            result1 = session.exec(select(CaliResult)).first()
            assert result1 is not None
            ds_id = result1.detection_settings
            es_id = result1.extraction_settings_id
            assert ds_id is not None
            assert es_id is not None
    finally:
        engine.dispose()

    # Second run - new analysis on same detection/extraction (using IDs)
    # This matches the user's scenario: "Created new AnalysisSettings ID 3"
    # while reusing DetectionSettings and ExtractionSettings
    as2 = AnalysisSettings(peaks_height_value=15.0)

    # This should NOT raise InvalidRequestError
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_id,
        extraction_settings=es_id,
        analysis_settings=as2,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify the run completed successfully
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            # Should have multiple results (one for each analysis)
            assert len(results) >= 2
    finally:
        engine.dispose()
