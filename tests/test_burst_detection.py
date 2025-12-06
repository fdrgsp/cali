"""Tests for burst detection in individual ROI traces."""

import numpy as np

from cali.analysis._trace_analysis import detect_bursts_in_trace


def test_detect_bursts_no_bursts() -> None:
    """Test burst detection when there are no bursts (below threshold)."""
    # Low amplitude trace, won't exceed threshold
    dec_dff = np.array([0.1, 0.2, 0.1, 0.2, 0.1])
    elapsed_time_ms = [0.0, 100.0, 200.0, 300.0, 400.0]

    burst_count, avg_duration, avg_interval = detect_bursts_in_trace(
        dec_dff,
        elapsed_time_ms,
        burst_threshold=1.0,  # High threshold
        min_duration_ms=100.0,
        gaussian_sigma=1.0,
    )

    assert burst_count == 0
    assert avg_duration is None
    assert avg_interval is None


def test_detect_bursts_single_burst() -> None:
    """Test detection of a single burst."""
    # Create a trace with one clear burst
    dec_dff = np.array([0.1, 0.2, 2.0, 2.5, 2.0, 0.2, 0.1])
    elapsed_time_ms = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]

    burst_count, avg_duration, avg_interval = detect_bursts_in_trace(
        dec_dff,
        elapsed_time_ms,
        burst_threshold=1.0,
        min_duration_ms=100.0,
        gaussian_sigma=1.0,
    )

    assert burst_count == 1
    assert avg_duration is not None
    assert avg_duration > 0.0
    assert avg_interval is None  # Only 1 burst, no interval


def test_detect_bursts_multiple_bursts() -> None:
    """Test detection of multiple bursts and inter-burst interval."""
    # Create a trace with two clear bursts separated by baseline
    dec_dff = np.array(
        [
            0.1,
            0.2,  # Baseline
            2.0,
            2.5,
            2.0,  # First burst
            0.2,
            0.1,
            0.2,  # Inter-burst interval
            2.0,
            2.5,
            2.0,  # Second burst
            0.2,
            0.1,  # Baseline
        ]
    )
    elapsed_time_ms = [
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
    ]

    burst_count, avg_duration, avg_interval = detect_bursts_in_trace(
        dec_dff,
        elapsed_time_ms,
        burst_threshold=1.0,
        min_duration_ms=100.0,
        gaussian_sigma=1.0,
    )

    assert burst_count == 2
    assert avg_duration is not None
    assert avg_duration > 0.0
    assert avg_interval is not None
    assert avg_interval > 0.0


def test_detect_bursts_min_duration_filter() -> None:
    """Test that bursts shorter than min_duration are filtered out."""
    # Create a trace with short burst (won't meet min duration)
    dec_dff = np.array([0.1, 2.0, 0.1, 0.2])
    elapsed_time_ms = [0.0, 50.0, 100.0, 150.0]  # Burst only 50ms

    burst_count, avg_duration, avg_interval = detect_bursts_in_trace(
        dec_dff,
        elapsed_time_ms,
        burst_threshold=1.0,
        min_duration_ms=100.0,  # Require 100ms minimum
        gaussian_sigma=1.0,
    )

    assert burst_count == 0
    assert avg_duration is None
    assert avg_interval is None


def test_detect_bursts_empty_trace() -> None:
    """Test burst detection with empty trace."""
    dec_dff = np.array([])
    elapsed_time_ms: list[float] = []

    burst_count, avg_duration, avg_interval = detect_bursts_in_trace(
        dec_dff,
        elapsed_time_ms,
        burst_threshold=1.0,
        min_duration_ms=100.0,
        gaussian_sigma=1.0,
    )

    assert burst_count == 0
    assert avg_duration is None
    assert avg_interval is None


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
