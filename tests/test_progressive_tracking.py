"""Tests for progressive pipeline tracking (detection, extraction, analysis).

This module tests the three-field tracking system in CaliResult:
- positions_detected: positions with ROIs
- positions_extracted: positions with traces
- positions_analyzed: positions with full analysis
"""

from collections.abc import Iterator
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

THREADS = 1


def create_mock_fov(position_index: int = 0, num_rois: int = 3) -> FOV:
    """Create a mock FOV with ROIs for testing without running cellpose."""
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
        )
        rois.append(roi)

    fov.rois = rois
    return fov


@pytest.fixture
def mock_detection_runner() -> Iterator[MagicMock]:
    """Fixture that patches DetectionRunner to return mock FOVs quickly."""
    with patch(
        "cali.detection._detection_runner.DetectionRunner._run_cellpose"
    ) as mock:

        def mock_detection(
            dataset: Any,
            detection_settings: Any,
            position_indices: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Iterator[FOV]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection
        yield mock


def test_detection_only_tracking(
    tmp_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
    sample_tensorstore_zarr: Path,
) -> None:
    """Test that detection-only run tracks positions_detected."""
    detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")
    db_path = tmp_path / "test.cali"

    runner = CaliRunner()
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        global_position_indices=[0, 1],
        database_name="test.cali",
        output_path=tmp_path,
    )

    # Check CaliResult
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        assert result.positions_detected == [0, 1]
        assert result.positions_extracted is None
        assert result.positions_analyzed is None
        assert result.extraction_settings_id is None
        assert result.analysis_settings_id is None

    engine.dispose(close=True)


def test_detection_plus_extraction_tracking(
    tmp_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
    sample_tensorstore_zarr: Path,
) -> None:
    """Test that detection+extraction run tracks both stages."""
    detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")
    extraction_settings = ExtractionSettings(neuropil_inner_radius=10)
    db_path = tmp_path / "test.cali"

    runner = CaliRunner()
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        global_position_indices=[0, 1],
        database_name="test.cali",
        output_path=tmp_path,
    )

    # Check CaliResult
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        # Should have extracted positions (extraction succeeded)
        assert result.positions_extracted == [0, 1]
        # Should not have analyzed (no analysis settings)
        assert result.positions_analyzed is None
        assert result.extraction_settings_id is not None
        assert result.analysis_settings_id is None

    engine.dispose(close=True)


def test_full_pipeline_tracking(
    tmp_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
    sample_tensorstore_zarr: Path,
) -> None:
    """Test that full pipeline run tracks all three stages."""
    detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")
    extraction_settings = ExtractionSettings(neuropil_inner_radius=10)
    analysis_settings = AnalysisSettings(peaks_height_value=2.0)
    db_path = tmp_path / "test.cali"

    runner = CaliRunner()
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1, 2],
        database_name="test.cali",
        output_path=tmp_path,
    )

    # Check CaliResult
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        # Full pipeline should track extraction and analysis
        assert result.positions_extracted == [0, 1, 2]
        assert result.positions_analyzed == [0, 1, 2]
        assert result.extraction_settings_id is not None
        assert result.analysis_settings_id is not None

    engine.dispose(close=True)


def test_progressive_runs_merge_positions(
    tmp_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
    sample_tensorstore_zarr: Path,
) -> None:
    """Test that running pipeline progressively merges positions correctly."""
    detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")
    extraction_settings = ExtractionSettings(neuropil_inner_radius=10)
    analysis_settings = AnalysisSettings(peaks_height_value=2.0)
    db_path = tmp_path / "test.cali"

    runner = CaliRunner()

    # Step 1: Detection only on [0]
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        global_position_indices=[0],
        database_name="test.cali",
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        assert result.positions_detected == [0]
        assert result.positions_extracted is None
        assert result.positions_analyzed is None
        result_id = result.id

    engine.dispose(close=True)

    # Step 2: Add detection on [1]
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        global_position_indices=[1],
        database_name="test.cali",
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        assert result.id == result_id  # Same result
        assert result.positions_detected == [0, 1]  # Merged
        assert result.positions_extracted is None
        assert result.positions_analyzed is None

    engine.dispose(close=True)

    # Step 3: Run extraction on [0, 1] - should upgrade existing result
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        global_position_indices=[0, 1],
        database_name="test.cali",
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        assert result.id == result_id  # Same result upgraded
        assert result.positions_detected == [0, 1]
        assert result.positions_extracted == [0, 1]
        assert result.positions_analyzed is None

    engine.dispose(close=True)

    # Step 4: Run full pipeline on [0, 1, 2] - should upgrade with analysis
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1, 2],
        database_name="test.cali",
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None
        assert result.id == result_id  # Same result upgraded
        assert result.positions_detected == [0, 1, 2]
        assert result.positions_extracted == [0, 1, 2]
        assert result.positions_analyzed == [0, 1, 2]

    engine.dispose(close=True)


def test_different_settings_create_separate_results(
    tmp_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
    sample_tensorstore_zarr: Path,
) -> None:
    """Test that different analysis settings create separate results."""
    detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")
    extraction_settings = ExtractionSettings(neuropil_inner_radius=10)
    analysis_settings = AnalysisSettings(peaks_height_value=2.0)
    db_path = tmp_path / "test.cali"

    runner = CaliRunner()

    # Run 1: Full pipeline with settings 1
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0, 1],
        database_name="test.cali",
        output_path=tmp_path,
    )

    # Run 2: Same extraction, different analysis settings
    analysis_settings2 = AnalysisSettings(peaks_height_value=3.0)
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings2,
        global_position_indices=[0, 1, 2],
        database_name="test.cali",
        output_path=tmp_path,
    )

    # Should have 2 CaliResults
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 2

        # First result
        result1 = results[0]
        assert result1.positions_extracted == [0, 1]
        assert result1.positions_analyzed == [0, 1]

        # Second result
        result2 = results[1]
        assert result2.positions_extracted == [0, 1, 2]
        assert result2.positions_analyzed == [0, 1, 2]
        assert result2.analysis_settings_id != result1.analysis_settings_id

    engine.dispose(close=True)


def test_detection_only_positions_query(
    tmp_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
    sample_tensorstore_zarr: Path,
) -> None:
    """Test querying positions that are detection-only."""
    detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")
    extraction_settings = ExtractionSettings(neuropil_inner_radius=10)
    db_path = tmp_path / "test.cali"

    runner = CaliRunner()

    # Run detection on [0, 1, 2, 3]
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        global_position_indices=[0, 1, 2, 3],
        database_name="test.cali",
        output_path=tmp_path,
    )

    # Run extraction on [0, 1] only
    runner.run(
        experiment=test_experiment,
        dataset_path=sample_tensorstore_zarr,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        global_position_indices=[0, 1],
        database_name="test.cali",
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        result = session.exec(select(CaliResult)).first()
        assert result is not None

        # Detection-only positions: detected but not extracted
        detected = set(result.positions_detected or [])
        extracted = set(result.positions_extracted or [])
        detection_only = detected - extracted

        assert detection_only == {2, 3}
        assert extracted == {0, 1}

    engine.dispose(close=True)


def test_equality_and_hash_include_new_fields(
    detection_settings: DetectionSettings,
) -> None:
    """Test that CaliResult equality and hash include new position fields."""
    result1 = CaliResult(
        experiment=1,
        detection_settings=1,
        extraction_settings_id=None,
        analysis_settings_id=None,
        positions_detected=[0, 1],
        positions_extracted=None,
        positions_analyzed=None,
    )

    result2 = CaliResult(
        experiment=1,
        detection_settings=1,
        extraction_settings_id=None,
        analysis_settings_id=None,
        positions_detected=[0, 1],
        positions_extracted=None,
        positions_analyzed=None,
    )

    result3 = CaliResult(
        experiment=1,
        detection_settings=1,
        extraction_settings_id=None,
        analysis_settings_id=None,
        positions_detected=[0, 1, 2],  # Different
        positions_extracted=None,
        positions_analyzed=None,
    )

    # Same positions should be equal
    assert result1 == result2
    assert hash(result1) == hash(result2)

    # Different positions should not be equal
    assert result1 != result3
    assert hash(result1) != hash(result3)
