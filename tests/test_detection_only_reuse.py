"""Test detection-only runs update existing results instead of creating new ones."""

from collections.abc import Generator, Iterator
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


def create_mock_fov(
    position_index: int = 0, num_rois: int = 3, name: str | None = None
) -> FOV:
    """Create a mock FOV with ROIs for testing without running cellpose."""
    if name is None:
        name = "B5_0000" if position_index == 0 else "B6_0000"
    fov = FOV(position_index=position_index, name=name)

    rois = []
    for i in range(1, num_rois + 1):
        mask_data = np.zeros((256, 256), dtype=np.uint8)
        cy, cx = 50 + i * 20, 50 + i * 20
        y, x = np.ogrid[:256, :256]
        mask_region = ((x - cx) ** 2 + (y - cy) ** 2) <= 100
        mask_data[mask_region] = 1

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

        # Use placeholder fov_id=0, will be set by commit_fov_result
        roi = ROI(label_value=i, roi_mask=mask, fov_id=0)
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
        ) -> Iterator[FOV]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection
        yield mock


@pytest.fixture
def experiment() -> Experiment:
    """Create a test experiment."""
    return Experiment(name="Test_Detection_Reuse")


@pytest.fixture
def detection_settings() -> DetectionSettings:
    """Create test detection settings."""
    return DetectionSettings(method="cellpose", model_type="cpsam")


@pytest.fixture
def extraction_settings() -> ExtractionSettings:
    """Create test extraction settings."""
    return ExtractionSettings(neuropil_inner_radius=10)


@pytest.fixture
def analysis_settings() -> AnalysisSettings:
    """Create test analysis settings."""
    return AnalysisSettings(peaks_height_value=2)


def test_detection_only_reuses_existing_result(
    tmp_path: Path,
    data_path: Path,
    experiment: Experiment,
    detection_settings: DetectionSettings,
    extraction_settings: ExtractionSettings,
    analysis_settings: AnalysisSettings,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that detection-only run updates existing result instead of creating new one.

    This tests the fix for the issue where:
    - Run 1: pos 0 with det+ext+an → creates CaliResult ID 1
    - Run 2: pos 0,1 with det-only → was creating separate result (BUG)
    - Run 3: pos 0,1 with det+ext+an → merges into ID 1, making ID 2 useless

    After fix:
    - Run 2 should UPDATE ID 1's positions_detected, not create new result
    """
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test_detection_reuse.cali"

    # Step 1: Run position 0 with detection+extraction+analysis
    # This creates the first CaliResult
    runner.run(
        experiment=experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Verify: 1 result exists for position 0
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1, f"Expected 1 result after run 1, got {len(results)}"

        result_1 = results[0]
        assert result_1.positions_detected == [0]
        # Note: extraction may or may not work depending on test data,
        # but that's not what we're testing here
        assert result_1.detection_settings_id is not None
        assert result_1.extraction_settings_id is not None
        assert result_1.analysis_settings_id is not None

        run_1_id = result_1.id

    # Step 2: Run positions 0,1 with detection-only (same detection settings)
    # **KEY TEST**: This should UPDATE result 1, not create a new result
    runner.run(
        experiment=experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        extraction_settings=None,
        analysis_settings=None,
        global_position_indices=[0, 1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Verify: Still only 1 result, with updated positions_detected
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1, (
            f"Expected 1 result after detection-only run, got {len(results)}. "
            "Detection-only should reuse existing result, not create new one."
        )

        result = results[0]
        assert result.id == run_1_id, "Result ID should remain the same"
        assert result.positions_detected == [0, 1], (
            f"Expected positions_detected=[0, 1], got {result.positions_detected}"
        )
        # Settings should remain unchanged
        assert result.detection_settings_id is not None
        assert result.extraction_settings_id is not None
        assert result.analysis_settings_id is not None

    engine.dispose()


def test_detection_only_creates_result_if_none_exists(
    tmp_path: Path,
    data_path: Path,
    experiment: Experiment,
    detection_settings: DetectionSettings,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that detection-only creates result if no existing result found.

    This is the normal case: first run is detection-only.
    """
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test_detection_only.cali"

    # Run detection-only on position 0
    runner.run(
        experiment=experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        extraction_settings=None,
        analysis_settings=None,
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Verify: 1 detection-only result exists
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1

        result = results[0]
        assert result.positions_detected == [0]
        assert result.positions_extracted is None
        assert result.positions_analyzed is None
        assert result.detection_settings_id is not None
        assert result.extraction_settings_id is None
        assert result.analysis_settings_id is None

    engine.dispose()


def test_subsequent_full_run_upgrades_detection_only_result(
    tmp_path: Path,
    data_path: Path,
    experiment: Experiment,
    detection_settings: DetectionSettings,
    extraction_settings: ExtractionSettings,
    analysis_settings: AnalysisSettings,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that full pipeline run on new positions works after detection-only.

    Scenario:
    1. Run pos 0 with det-only → creates detection-only result
    2. Run pos 0,1 with det+ext+an → should upgrade result, add pos 1
    3. Verify only 1 result exists with full settings
    """
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test_upgrade.cali"

    # Step 1: Detection-only on position 0
    runner.run(
        experiment=experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        extraction_settings=None,
        analysis_settings=None,
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")

    # Verify detection-only result
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        result_1 = results[0]
        assert result_1.extraction_settings_id is None
        assert result_1.analysis_settings_id is None

    # Step 2: Full pipeline on positions 0,1
    runner.run(
        experiment=experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Verify: Still 1 result, but upgraded with extraction/analysis settings
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        # The detection-only result should have been deleted and replaced
        assert len(results) == 1, (
            f"Expected 1 result after full run, got {len(results)}"
        )

        result = results[0]
        # Key test: result was upgraded from detection-only to full settings
        assert result.detection_settings_id is not None
        assert result.extraction_settings_id is not None
        assert result.analysis_settings_id is not None
        # Positions detected should include both 0 and 1
        assert result.positions_detected == [0, 1]

    engine.dispose()
