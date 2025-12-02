"""Comprehensive tests for all CaliRunner combinations and workflows.

This test suite ensures CaliRunner handles all possible run scenarios correctly:
- All stage combinations (D, D+E, D+E+A)
- Sequential runs with different stages
- Different position sets
- Settings variations
- Edge cases
"""

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


def create_mock_fov(
    position_index: int = 0, num_rois: int = 3, name: str | None = None
) -> FOV:
    """Create a mock FOV with ROIs for testing without running cellpose."""
    if name is None:
        # Use well name format: A1_0000, B5_0001, etc.
        row_letter = chr(ord("A") + (position_index // 12))  # 12 columns per row
        col_number = (position_index % 12) + 1
        name = f"{row_letter}{col_number}_{position_index:04d}"
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
        ) -> Generator[FOV, None, None]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection
        yield mock


# ============================================================================
# BASIC WORKFLOWS
# ============================================================================


def test_detection_only(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test detection-only run."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        assert results[0].detection_settings_id is not None
        assert results[0].extraction_settings_id is None
        assert results[0].analysis_settings_id is None
        assert results[0].positions_detected == [0]
    engine.dispose()


def test_detection_extraction(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test detection + extraction run."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        assert results[0].detection_settings_id is not None
        assert results[0].extraction_settings_id is not None
        assert results[0].analysis_settings_id is None
        assert results[0].positions_detected == [0]
    engine.dispose()


def test_full_pipeline(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test full detection + extraction + analysis run."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(peaks_height_value=2),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        assert results[0].detection_settings_id is not None
        assert results[0].extraction_settings_id is not None
        assert results[0].analysis_settings_id is not None
        assert results[0].positions_detected == [0]
    engine.dispose()


# ============================================================================
# SEQUENTIAL RUNS - SAME POSITIONS
# ============================================================================


def test_detection_then_skip_detection(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test D → D (should skip second detection)."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: Detection
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Run 2: Same detection (should skip) - create new settings object
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1, "Should reuse same result"
    engine.dispose()


def test_detection_then_extraction(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test D → D+E (should upgrade detection-only result)."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: Detection only
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Run 2: Detection + Extraction (should upgrade)
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1, "Should upgrade, not create new result"
        result = results[0]
        assert result.extraction_settings_id is not None
    engine.dispose()


def test_detection_then_full_pipeline(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test D → D+E+A (should delete detection-only, create full result)."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: Detection only
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Run 2: Full pipeline
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(peaks_height_value=2),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1, "Should have one result (detection-only deleted)"
        assert results[0].analysis_settings_id is not None
    engine.dispose()


def test_extraction_then_detection_updates_positions(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test D+E → D (should update positions_detected on existing result)."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: Detection + Extraction on pos 0
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Run 2: Detection only on pos 0,1 (should update positions_detected)
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        global_position_indices=[0, 1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1, "Should reuse existing result"
        assert results[0].positions_detected == [0, 1]
    engine.dispose()


def test_full_pipeline_then_detection(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test D+E+A → D (should update positions_detected)."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: Full pipeline on pos 0
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(peaks_height_value=2),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Run 2: Detection only on pos 0,1
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        global_position_indices=[0, 1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        assert results[0].positions_detected == [0, 1]
        assert results[0].extraction_settings_id is not None
        assert results[0].analysis_settings_id is not None
    engine.dispose()


# ============================================================================
# DIFFERENT POSITION SETS
# ============================================================================


def test_different_positions_merge(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test running same settings on different positions merges into one result."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: Position 0
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Run 2: Position 1
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        global_position_indices=[1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1, "Should merge into single result"
        assert set(results[0].positions_detected or []) == {0, 1}
    engine.dispose()


def test_full_pipeline_different_positions(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test full pipeline on different position sets merges correctly."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: Position 0
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(peaks_height_value=2),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Run 2: Position 1
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(peaks_height_value=2),
        global_position_indices=[1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        assert results[0].positions_detected == [0, 1]
    engine.dispose()


# ============================================================================
# DIFFERENT SETTINGS
# ============================================================================


def test_different_detection_settings_separate_results(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test different detection settings create separate results."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: cellpose with diameter 30
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", diameter=30),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Run 2: cellpose with diameter 40
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", diameter=40),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 2, "Different detection settings = separate results"
    engine.dispose()


def test_different_extraction_settings_separate_results(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test different extraction settings create separate results."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: Extraction with neuropil_inner_radius=10
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(neuropil_inner_radius=10),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Run 2: Extraction with neuropil_inner_radius=20
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(neuropil_inner_radius=20),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 2, "Different extraction settings = separate results"
    engine.dispose()


def test_different_analysis_settings_separate_results(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test different analysis settings create separate results."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: Analysis with peaks_height_value=2
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(peaks_height_value=2),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    # Run 2: Analysis with peaks_height_value=3
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(peaks_height_value=3),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 2, "Different analysis settings = separate results"
    engine.dispose()


# ============================================================================
# EDGE CASES
# ============================================================================


def test_rerun_exact_same_configuration(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test rerunning exact same configuration skips appropriately."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(peaks_height_value=2),
        global_position_indices=[0, 1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        result_1 = session.exec(select(CaliResult)).first()
        assert result_1 is not None
        result_1_id = result_1.id

    # Run 2: Exact same (should skip everything)
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(peaks_height_value=2),
        global_position_indices=[0, 1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        assert results[0].id == result_1_id, "Should reuse same result"
    engine.dispose()


def test_complex_scenario_multiple_runs(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test complex scenario with multiple runs and settings combinations.

    Scenario:
    1. D on pos 0 with settings A
    2. D+E on pos 0 with settings A (should upgrade)
    3. D on pos 1 with settings A (should merge)
    4. D+E+A on pos 0,1 with settings A (should upgrade and merge)
    5. D on pos 0,1 with settings B (should create new result)
    """
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: D on pos 0 with settings A
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", diameter=30),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        assert results[0].extraction_settings_id is None

    # Run 2: D+E on pos 0 with settings A (should upgrade)
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", diameter=30),
        extraction_settings=ExtractionSettings(),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        assert results[0].extraction_settings_id is not None

    # Run 3: D on pos 1 with settings A (should merge)
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", diameter=30),
        global_position_indices=[1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        assert results[0].positions_detected == [0, 1]

    # Run 4: D+E+A on pos 0,1 with settings A (should upgrade and merge)
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", diameter=30),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(peaks_height_value=2),
        global_position_indices=[0, 1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 1
        assert results[0].analysis_settings_id is not None
        assert results[0].positions_detected == [0, 1]

    # Run 5: D on pos 0,1 with settings B (should create new result)
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", diameter=40),
        global_position_indices=[0, 1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    with Session(engine) as session:
        results = list(session.exec(select(CaliResult)).all())
        assert len(results) == 2, "Different detection settings = new result"
    engine.dispose()


def test_settings_reuse_by_id(
    tmp_path: Path,
    data_path: Path,
    test_experiment: Experiment,
    mock_detection_runner: MagicMock,
) -> None:
    """Test that settings can be reused by ID."""
    runner = CaliRunner(commit_batch_size=1)
    db_path = tmp_path / "test.cali"

    # Run 1: Create settings
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose"),
        extraction_settings=ExtractionSettings(),
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        det_settings = session.exec(select(DetectionSettings)).first()
        ext_settings = session.exec(select(ExtractionSettings)).first()
        assert det_settings is not None
        assert ext_settings is not None
        det_id = det_settings.id
        ext_id = ext_settings.id

    # Run 2: Reuse settings by ID
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=det_id,  # type: ignore
        extraction_settings=ext_id,  # type: ignore
        global_position_indices=[1],
        database_name=db_path.name,
        output_path=tmp_path,
    )

    with Session(engine) as session:
        # Should still only have 1 of each setting
        det_count = len(list(session.exec(select(DetectionSettings)).all()))
        ext_count = len(list(session.exec(select(ExtractionSettings)).all()))
        assert det_count == 1
        assert ext_count == 1
    engine.dispose()
