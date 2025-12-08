"""Tests for burst detection in individual ROI traces."""

from __future__ import annotations

import numpy as np
import pytest

from cali.analysis._trace_analysis import detect_bursts_in_trace


@pytest.mark.parametrize(
    (
        "dec_dff",
        "elapsed_time_ms",
        "burst_threshold",
        "min_duration_ms",
        "expected_count",
        "expected_duration_not_none",
        "expected_interval_not_none",
        "description",
    ),
    [
        # No bursts - below threshold
        (
            np.array([0.1, 0.2, 0.1, 0.2, 0.1]),
            [0.0, 100.0, 200.0, 300.0, 400.0],
            1.0,
            100.0,
            0,
            False,
            False,
            "no_bursts_below_threshold",
        ),
        # Single burst
        (
            np.array([0.1, 0.2, 2.0, 2.5, 2.0, 0.2, 0.1]),
            [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            1.0,
            100.0,
            1,
            True,
            False,
            "single_burst",
        ),
        # Multiple bursts with interval
        (
            np.array([0.1, 0.2, 2.0, 2.5, 2.0, 0.2, 0.1, 0.2, 2.0, 2.5, 2.0, 0.2, 0.1]),
            [
                0.0,
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                1000.0,
                1100.0,
                1200.0,
            ],
            1.0,
            100.0,
            2,
            True,
            True,
            "multiple_bursts",
        ),
        # Burst filtered by min duration
        (
            np.array([0.1, 2.0, 0.1, 0.2]),
            [0.0, 50.0, 100.0, 150.0],
            1.0,
            100.0,
            0,
            False,
            False,
            "filtered_by_min_duration",
        ),
        # Empty trace
        (
            np.array([]),
            [],
            1.0,
            100.0,
            0,
            False,
            False,
            "empty_trace",
        ),
    ],
)
def test_detect_bursts_scenarios(
    dec_dff: np.ndarray,
    elapsed_time_ms: list[float],
    burst_threshold: float,
    min_duration_ms: float,
    expected_count: int,
    expected_duration_not_none: bool,
    expected_interval_not_none: bool,
    description: str,
) -> None:
    """Test burst detection across various scenarios."""
    burst_count, avg_duration, avg_interval = detect_bursts_in_trace(
        dec_dff,
        elapsed_time_ms,
        burst_threshold=burst_threshold,
        min_duration_ms=min_duration_ms,
        gaussian_sigma=1.0,
    )

    assert burst_count == expected_count, f"Failed for {description}"

    if expected_duration_not_none:
        assert avg_duration is not None, f"Failed for {description}"
        assert avg_duration > 0.0, f"Failed for {description}"
    else:
        assert avg_duration is None, f"Failed for {description}"

    if expected_interval_not_none:
        assert avg_interval is not None, f"Failed for {description}"
        assert avg_interval > 0.0, f"Failed for {description}"
    else:
        assert avg_interval is None, f"Failed for {description}"


def test_detect_bursts_returns_seconds() -> None:
    """Test that durations and intervals are returned in seconds, not milliseconds."""
    # Create a trace with known burst duration
    dec_dff = np.array([0.1, 2.0, 2.0, 2.0, 0.1])
    elapsed_time_ms = [0.0, 100.0, 200.0, 300.0, 400.0]  # 300ms burst

    burst_count, avg_duration, _ = detect_bursts_in_trace(
        dec_dff,
        elapsed_time_ms,
        burst_threshold=1.0,
        min_duration_ms=100.0,
        gaussian_sigma=1.0,
    )

    assert burst_count == 1
    assert avg_duration is not None
    # Duration should be ~0.2-0.3 seconds (200-300ms), not 200-300
    assert 0.1 < avg_duration < 0.5
