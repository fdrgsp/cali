"""Test that commit batch logging reports correct FOV counts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    import pytest


def test_commit_batch_logging_reports_correct_count(
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that commit batch logging shows correct FOV count per batch.

    This test verifies that when processing multiple batches of FOVs,
    the "Committing final batch of X FOVs" message shows the correct
    count for just that batch, not the accumulated count.

    Regression test for issue where fov_count accumulated across batches,
    causing misleading log messages like "Committing final batch of 4 FOVs"
    when only 2 FOVs were in that batch.
    """
    test_db_path = tmp_path / "test_commit_batch.cali"

    # Create experiment
    exp = Experiment.create_from_data("test_commit_batch", str(data_path))

    # Setup settings
    detection_settings = DetectionSettings(
        method="cellpose",
        model_type="cyto3",
    )
    extraction_settings = ExtractionSettings()
    analysis_settings = AnalysisSettings()

    # Use a large commit batch size to force "final batch" commits
    # Process 2 positions with commit_batch_size=10 and threads=1
    # This creates 2 outer batches, each processing 1 FOV
    # Since batch size (1) < commit_batch_size (10), we get "final batch" commits
    runner = CaliRunner(commit_batch_size=10)

    # Run pipeline with 2 positions
    with caplog.at_level("INFO", logger="cali_logger"):
        runner.run(
            exp,
            data_path,
            detection_settings,
            extraction_settings=extraction_settings,
            analysis_settings=analysis_settings,
            global_position_indices=[0, 1],
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
        )

    # Verify database was created
    assert test_db_path.exists()

    # Parse log messages for commit batch counts IN EXTRACTION/ANALYSIS
    # Detection phase doesn't have outer batch looping, so we ignore it
    commit_messages = []
    in_extraction_phase = False

    for record in caplog.records:
        # Mark when we enter extraction/analysis phase
        if "Running Extraction and" in record.message:
            in_extraction_phase = True

        # Collect "final batch" messages only from extraction/analysis
        if in_extraction_phase and "Committing final batch of" in record.message:
            commit_messages.append(record.message)

    # Should have "final batch" commit messages from both extraction batches
    # With commit_batch_size=10 (larger than batch size), each outer batch
    # will have uncommitted FOVs that trigger "final batch" commits
    assert len(commit_messages) >= 2, (
        "Expected at least 2 'final batch' messages from extraction, got "
        f"{len(commit_messages)}"
    )

    # Extract FOV counts from messages
    import re

    for msg in commit_messages:
        match = re.search(r"Committing final batch of (\d+) FOVs", msg)
        if match:
            count = int(match.group(1))
            # Each outer batch processes 1 FOV (threads=1), so final batch should be 1
            # Not accumulated counts like 1, 2, 3, etc.
            assert count == 1, (
                f"Expected final batch count of 1 (not accumulated), got {count}. "
                f"This suggests fov_count is accumulating across batches instead of "
                f"using batch_fov_count."
            )
