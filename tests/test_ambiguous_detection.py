"""Tests for ambiguous detection scenarios.

When running detection-only on new positions, but there are multiple existing
runs with the same detection settings (but different extraction/analysis settings),
we need to either:
1. Require the user to specify which run to add to (by providing extraction/analysis)
2. Or raise an error indicating ambiguity
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    AnalysisSettings,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)
from tests.test_runners import create_mock_fov


def test_detection_only_with_single_existing_run(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test detection-only when there's only one existing run - should work fine."""
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
        ) -> Generator[FOV, None, None]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection

        detection_settings = DetectionSettings(
            method="cellpose", model_type="cpsam", batch_size=1
        )
        extraction_settings = ExtractionSettings(dff_window=100, threads=1)

        # Run 1: Detection + Extraction on position 0
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )

        # Run 2: Detection-only on position 1 (should add to existing run)
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[1],
        )

        # Verify: Should have 1 result with both positions detected
        engine = create_engine(f"sqlite:///{test_db_path}")
        try:
            with Session(engine) as session:
                results = list(session.exec(select(CaliResult)).all())
                assert len(results) == 1
                assert set(results[0].positions_detected or []) == {0, 1}
                assert results[0].extraction_settings_id is not None
        finally:
            engine.dispose()


def test_detection_only_with_multiple_runs_same_detection_different_extraction(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test detection-only with multiple runs.

    Same detection but different extraction should raise an error.
    """
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
        ) -> Generator[FOV, None, None]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection

        detection_settings = DetectionSettings(
            method="cellpose", model_type="cpsam", batch_size=1
        )

        # Run 1: Detection + Extraction (dff_window=100) on position 0
        extraction_settings_1 = ExtractionSettings(dff_window=100, threads=1)
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings_1,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )

        # Run 2: Detection + Extraction (dff_window=200) on position 1
        extraction_settings_2 = ExtractionSettings(dff_window=200, threads=1)
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings_2,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[1],
        )

        # Verify we have 2 runs
        engine = create_engine(f"sqlite:///{test_db_path}")
        try:
            with Session(engine) as session:
                results = list(session.exec(select(CaliResult)).all())
                assert len(results) == 2
        finally:
            engine.dispose()

        # Run 3: Detection-only on position 0 with force=True - should raise error
        # Force re-detection, and since there are now multiple extraction settings,
        # detection-only is ambiguous and should raise an error
        with pytest.raises(
            ValueError,
            match=(
                r"Multiple runs exist.*same detection.*different extraction.*"
                r"specify extraction_settings"
            ),
        ):
            runner.run(
                experiment=test_experiment,
                dataset_path=data_path,
                detection_settings=detection_settings,
                database_name=test_db_path.name,
                output_path=test_db_path.parent,
                global_position_indices=[0],
                force=True,  # Force re-detection to trigger ambiguity check
            )


def test_detection_only_with_multiple_runs_same_detection_extraction_different_analysis(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test detection-only with multiple runs.

    Same detection+extraction but different analysis should raise an error.
    """
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
        ) -> Generator[FOV, None, None]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection

        detection_settings = DetectionSettings(
            method="cellpose", model_type="cpsam", batch_size=1
        )
        extraction_settings = ExtractionSettings(dff_window=100, threads=1)

        # Run 1: Detection + Extraction + Analysis (height=1.0) on position 0
        analysis_settings_1 = AnalysisSettings(
            peaks_height_value=1.0, peaks_height_mode="std", threads=1
        )
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings,
            analysis_settings=analysis_settings_1,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )

        # Run 2: Detection + Extraction + Analysis (height=2.0) on position 1
        analysis_settings_2 = AnalysisSettings(
            peaks_height_value=2.0, peaks_height_mode="std", threads=1
        )
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings,
            analysis_settings=analysis_settings_2,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[1],
        )

        # Verify we have 2 runs
        engine = create_engine(f"sqlite:///{test_db_path}")
        try:
            with Session(engine) as session:
                results = list(session.exec(select(CaliResult)).all())
                assert len(results) == 2
        finally:
            engine.dispose()

        # Run 3: Detection-only on position 0 with force=True - should raise error
        # Force re-detection, and since there are now multiple analysis settings,
        # detection-only is ambiguous and should raise an error
        with pytest.raises(
            ValueError,
            match=(
                r"Multiple runs exist.*same detection.*different analysis.*"
                r"specify.*analysis_settings"
            ),
        ):
            runner.run(
                experiment=test_experiment,
                dataset_path=data_path,
                detection_settings=detection_settings,
                database_name=test_db_path.name,
                output_path=test_db_path.parent,
                global_position_indices=[0],
                force=True,  # Force re-detection to trigger ambiguity check
            )


def test_detection_only_disambiguated_by_extraction(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test extraction settings disambiguates multiple runs."""
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
        ) -> Generator[FOV, None, None]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection

        detection_settings = DetectionSettings(
            method="cellpose", model_type="cpsam", batch_size=1
        )

        # Run 1: Detection + Extraction (dff_window=100) on position 0
        extraction_settings_1 = ExtractionSettings(dff_window=100, threads=1)
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings_1,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )

        # Run 2: Detection + Extraction (dff_window=200) on position 1
        extraction_settings_2 = ExtractionSettings(dff_window=200, threads=1)
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings_2,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[1],
        )

        # Run 3: Detection + Extraction (dff_window=100) on position 1
        # This should add to Run 1 because extraction settings match
        extraction_settings_1_copy = ExtractionSettings(dff_window=100, threads=1)
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings_1_copy,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[1],
        )

        # Verify: Should still have 2 results, with Run 1 having positions [0, 1]
        engine = create_engine(f"sqlite:///{test_db_path}")
        try:
            with Session(engine) as session:
                results = list(session.exec(select(CaliResult)).all())
                assert len(results) == 2

                # Find the result with dff_window=100
                run1 = None
                for r in results:
                    if r.extraction_settings_id:
                        ext = session.get(ExtractionSettings, r.extraction_settings_id)
                        if ext and ext.dff_window == 100:
                            run1 = r
                            break

                assert run1 is not None
                assert set(run1.positions_detected or []) == {0, 1}
        finally:
            engine.dispose()


def test_detection_only_disambiguated_by_analysis(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
) -> None:
    """Test analysis settings disambiguates multiple runs."""
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
        ) -> Generator[FOV, None, None]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection

        detection_settings = DetectionSettings(
            method="cellpose", model_type="cpsam", batch_size=1
        )
        extraction_settings = ExtractionSettings(dff_window=100, threads=1)

        # Run 1: Full pipeline (height=1.0) on position 0
        analysis_settings_1 = AnalysisSettings(
            peaks_height_value=1.0, peaks_height_mode="std", threads=1
        )
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings,
            analysis_settings=analysis_settings_1,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )

        # Run 2: Full pipeline (height=2.0) on position 1
        analysis_settings_2 = AnalysisSettings(
            peaks_height_value=2.0, peaks_height_mode="std", threads=1
        )
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings,
            analysis_settings=analysis_settings_2,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[1],
        )

        # Run 3: Full pipeline (height=1.0) on position 1
        # Should add to Run 1 because analysis settings match
        analysis_settings_1_copy = AnalysisSettings(
            peaks_height_value=1.0, peaks_height_mode="std", threads=1
        )
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings,
            analysis_settings=analysis_settings_1_copy,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[1],
        )

        # Verify: Should still have 2 results, with Run 1 having positions [0, 1]
        engine = create_engine(f"sqlite:///{test_db_path}")
        try:
            with Session(engine) as session:
                results = list(session.exec(select(CaliResult)).all())
                assert len(results) == 2

                # Find the result with peaks_height_value=1.0
                run1 = None
                for r in results:
                    if r.analysis_settings_id:
                        ans = session.get(AnalysisSettings, r.analysis_settings_id)
                        if ans and ans.peaks_height_value == 1.0:
                            run1 = r
                            break

                assert run1 is not None
                assert set(run1.positions_detected or []) == {0, 1}
                # Position 0 extraction may fail with mock data, but position 1 should
                # succeed
                assert 1 in (run1.positions_analyzed or [])
        finally:
            engine.dispose()
