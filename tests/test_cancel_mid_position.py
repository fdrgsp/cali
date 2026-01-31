"""Test for cancellation during mid-position extraction (partial ROI processing).

This test verifies that when cancellation happens while processing ROIs within
a single position, the position is NOT marked as complete in the database.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel import (
    ROI,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    Traces,
)
from tests.conftest import create_mock_fov


def test_cancel_mid_position_roi_processing(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test that cancelling mid-position does NOT mark position as complete.

    Scenario:
    1. Run detection to create a position with ROIs
    2. Start extraction on that position
    3. Cancel while processing ROIs (before all ROIs are done)
    4. Verify the position is NOT in positions_extracted/positions_analyzed
    """
    runner = CaliRunner(commit_batch_size=1)

    # Step 1: Run detection to completion
    with patch(
        "cali.detection._detection_runner.DetectionRunner._run_cellpose"
    ) as mock_det:

        def mock_detection(
            dataset: Any,
            detection_settings: Any,
            position_indices: list[int],
            *args: Any,
            **kwargs: Any,
        ) -> Iterator[Any]:
            # Create FOV with multiple ROIs to simulate your 118 ROI scenario
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock_det.side_effect = mock_detection

        detection_settings = DetectionSettings(
            method="cellpose", model_type="cpsam", batch_size=1
        )
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )

    # Verify detection completed
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            result = session.exec(select(CaliResult)).first()
            assert result is not None
            assert result.positions_detected == [0]

            # Get the number of ROIs created
            rois = list(session.exec(select(ROI)).all())
            num_rois = len(rois)
            assert num_rois > 0, "Detection should create ROIs"
    finally:
        engine.dispose()

    # Step 2: Run extraction but cancel mid-position
    runner2 = CaliRunner(commit_batch_size=1)

    # Track how many ROIs were processed before cancellation
    rois_processed = 0
    cancel_after_n_rois = 1  # Cancel after processing just 1 ROI

    # Patch the extraction method to cancel mid-position
    original_extract = runner2._extraction_runner._process_roi_trace

    def mock_extract_with_cancel(*args: Any, **kwargs: Any) -> Any:
        nonlocal rois_processed
        result = original_extract(*args, **kwargs)
        rois_processed += 1
        if rois_processed >= cancel_after_n_rois:
            # Cancel after processing N ROIs (simulating your 30/118 scenario)
            runner2.cancel()
        return result

    with patch.object(
        runner2._extraction_runner,
        "_process_roi_trace",
        side_effect=mock_extract_with_cancel,
    ):
        extraction_settings = ExtractionSettings(dff_window=100, threads=1)
        detection_settings_2 = DetectionSettings(
            method="cellpose", model_type="cpsam", batch_size=1
        )

        runner2.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings_2,
            extraction_settings=extraction_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )

    # Step 3: Verify the cancelled position is NOT marked as complete
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = list(session.exec(select(CaliResult)).all())

            # Find the extraction result
            extraction_result = None
            for r in results:
                if r.extraction_settings_id is not None:
                    extraction_result = r
                    break

            # The key assertion: position 0 should NOT be in the completed list
            # because we cancelled mid-position
            if extraction_result is not None:
                # If a result exists, it should have empty positions
                # (because we cancelled before completing the position)
                assert (
                    extraction_result.positions_extracted == []
                    or extraction_result.positions_extracted is None
                ), (
                    f"Position 0 should NOT be marked as extracted because we "
                    f"cancelled mid-position, but got: "
                    f"{extraction_result.positions_extracted}"
                )
                assert (
                    extraction_result.positions_analyzed == []
                    or extraction_result.positions_analyzed is None
                ), (
                    f"Position 0 should NOT be marked as analyzed because we "
                    f"cancelled mid-position, but got: "
                    f"{extraction_result.positions_analyzed}"
                )

            # Verify that the cancellation happened mid-position
            assert rois_processed < num_rois, (
                f"Should have cancelled before all {num_rois} ROIs were processed"
            )

            # Verify no traces were committed (because we returned None)
            traces = list(session.exec(select(Traces)).all())
            # Since we cancelled mid-position and return None, no traces
            # should be committed for this position
            assert len(traces) == 0, (
                f"No traces should be committed when cancelled mid-position, "
                f"got {len(traces)}"
            )
    finally:
        engine.dispose()
