"""Tests for cancellation handling and position tracking in CaliRunner.

Verifies that CaliResult.positions_analyzed accurately reflects completed positions
when runs are cancelled mid-execution.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)
from tests.test_runners import create_mock_fov


def test_detection_cancel_after_partial_completion(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test that positions_analyzed reflects partial completion on cancellation."""
    runner = CaliRunner(commit_batch_size=1)

    # Mock cellpose to yield only 2 of 3 requested positions before cancelling
    positions_yielded = []

    def mock_detection_with_cancel(
        dataset: Any,
        detection_settings: Any,
        position_indices: list[int],
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[FOV]:
        for i, pos_idx in enumerate(position_indices):
            if i == 2:  # Cancel after yielding positions 0 and 1
                runner.cancel()
                return
            positions_yielded.append(pos_idx)
            yield create_mock_fov(pos_idx)

    with patch(
        "cali.detection._detection_runner.DetectionRunner._run_cellpose",
        side_effect=mock_detection_with_cancel,
    ):
        detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")

        # Run detection on positions [0, 1, 2]
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0, 1, 2],
        )

    # Verify database state
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            # Check CaliResult reflects only completed positions
            result = session.exec(select(CaliResult)).first()
            assert result is not None
            assert result.positions_analyzed == [0, 1]  # Only 2 completed
            assert len(positions_yielded) == 2

            # Verify ROIs exist only for completed positions
            # (The test_experiment fixture creates 3 FOVs, but ROIs are only
            # created by detection for the positions that completed)
            rois = list(session.exec(select(ROI)).all())
            # 3 ROIs per FOV * 2 FOVs = 6 ROIs total
            assert len(rois) == 6
            roi_fov_positions = {roi.fov.position_index for roi in rois}
            assert roi_fov_positions == {0, 1}  # Only positions 0 and 1 have ROIs
    finally:
        engine.dispose()


def test_detection_cancel_before_any_completion(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test cancellation before any positions complete."""

    def mock_detection_immediate_cancel(
        dataset: Any,
        detection_settings: Any,
        position_indices: list[int],
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[FOV]:
        # Cancel immediately without yielding anything
        return
        yield  # pragma: no cover - unreachable but keeps generator type

    runner = CaliRunner(commit_batch_size=1)

    with patch(
        "cali.detection._detection_runner.DetectionRunner._run_cellpose",
        side_effect=mock_detection_immediate_cancel,
    ):
        detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")

        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0, 1, 2],
        )

    # Verify database state
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            # CaliResult should exist but with empty positions_analyzed
            result = session.exec(select(CaliResult)).first()
            assert result is not None
            assert result.positions_analyzed == []

            # FOVs should exist (from experiment structure) but have no ROIs
            fovs = list(session.exec(select(FOV)).all())
            assert len(fovs) > 0  # FOVs created from experiment
            # But no ROIs should have been detected
            for fov in fovs:
                assert len(fov.rois) == 0
    finally:
        engine.dispose()


def test_extraction_cancel_after_partial_completion(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test extraction cancellation updates positions_analyzed correctly."""
    runner = CaliRunner(commit_batch_size=1)

    # First, run detection to completion
    with patch(
        "cali.detection._detection_runner.DetectionRunner._run_cellpose"
    ) as mock_det:

        def mock_detection(
            dataset: Any,
            detection_settings: Any,
            position_indices: list[int],
            *args: Any,
            **kwargs: Any,
        ) -> Iterator[FOV]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock_det.side_effect = mock_detection

        detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0, 1, 2],
        )

    # Now run extraction but cancel after processing 1 position
    runner2 = CaliRunner(commit_batch_size=1)
    positions_extracted = []

    original_run_extraction = runner2._run_extraction

    def mock_extraction_with_cancel(
        dataset: Any,
        extraction_settings: Any,
        analysis_settings: Any,
        fovs: list[FOV],
    ) -> Iterator[FOV]:
        count = 0
        for fov in original_run_extraction(
            dataset, extraction_settings, analysis_settings, fovs
        ):
            if count == 1:  # Cancel after first position
                runner2.cancel()
                return
            positions_extracted.append(fov.position_index)
            count += 1
            yield fov

    runner2._run_extraction = mock_extraction_with_cancel  # type: ignore

    extraction_settings = ExtractionSettings(dff_window=100, threads=1)

    # Create a fresh detection_settings for the second run
    # (the one from the first run is detached from session)
    detection_settings_2 = DetectionSettings(method="cellpose", model_type="cpsam")

    runner2.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings_2,
        extraction_settings=extraction_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0, 1, 2],
    )

    # Verify database state
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            # Should have 2 CaliResults:
            # detection-only (all 3 pos) + extraction (partial)
            results = list(session.exec(select(CaliResult)).all())

            # Find extraction result (has extraction_settings_id)
            extraction_result = next(
                (r for r in results if r.extraction_settings_id is not None), None
            )
            assert extraction_result is not None
            # Should only show the 1 position that was extracted
            assert len(extraction_result.positions_analyzed or []) == 1
            assert len(positions_extracted) == 1
    finally:
        engine.dispose()


def test_successful_run_all_positions_tracked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test that successful runs track all positions correctly."""
    runner = CaliRunner(commit_batch_size=1)

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

        detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")

        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0, 1, 2],
        )

    # Verify database state
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            result = session.exec(select(CaliResult)).first()
            assert result is not None
            assert result.positions_analyzed == [0, 1, 2]  # All positions completed

            # Check that we have ROIs for all positions
            # (Note: experiment creates FOVs too, so we filter by ROI existence)
            fovs_with_rois = [
                fov for fov in session.exec(select(FOV)).all() if len(fov.rois) > 0
            ]
            assert len(fovs_with_rois) == 3
    finally:
        engine.dispose()


def test_partial_positions_subset_requested(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test running on a subset of positions works correctly."""
    runner = CaliRunner(commit_batch_size=1)

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

        detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")

        # Only run on positions [1, 2] (skip 0)
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[1, 2],
        )

    # Verify database state
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            result = session.exec(select(CaliResult)).first()
            assert result is not None
            assert result.positions_analyzed == [1, 2]  # Only requested positions

            # Filter FOVs that have ROIs (from detection)
            fovs_with_rois = [
                fov for fov in session.exec(select(FOV)).all() if len(fov.rois) > 0
            ]
            assert len(fovs_with_rois) == 2
            fov_positions = {fov.position_index for fov in fovs_with_rois}
            assert fov_positions == {1, 2}
    finally:
        engine.dispose()
