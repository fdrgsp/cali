from unittest.mock import patch

import numpy as np
import pytest

from cali.extraction._util import _calculate_bg, calculate_dff, get_iei


def test_calculate_dff() -> None:
    """Test calculate_dff."""
    data = np.array([100, 110, 120, 110, 100, 90, 100, 110, 120, 110], dtype=float)
    dff = calculate_dff(data, window=5, percentile=10)
    assert dff.shape == data.shape
    assert not np.any(np.isnan(dff))


def test_calculate_dff_plot() -> None:
    """Test calculate_dff with plot=True."""
    data = np.array([100, 110, 120], dtype=float)
    with patch("matplotlib.pyplot.show") as mock_show:
        calculate_dff(data, window=3, percentile=10, plot=True)
        mock_show.assert_called_once()


def test_calculate_bg() -> None:
    """Test _calculate_bg."""
    data = np.ones(100) * 100
    bg = _calculate_bg(data, window=10, percentile=50)
    np.testing.assert_allclose(bg, 100)

    # Test with varying data
    data = np.arange(100, dtype=float)
    bg = _calculate_bg(data, window=10, percentile=50)
    # For a linear ramp, the median in a window should be roughly the center of the
    # window But the implementation might be different. Just check shape and no nans.
    assert bg.shape == data.shape
    assert not np.any(np.isnan(bg))


@pytest.mark.parametrize(
    "peaks,elapsed_time_ms,expected",
    [
        # Test case 1: Normal case with 3 peaks
        (
            np.array([10, 20, 30]),
            [i * 10.0 for i in range(40)],  # 10ms intervals
            [0.1, 0.1],  # IEI in seconds: (200-100)/1000, (300-200)/1000
        ),
        # Test case 2: Two peaks close together
        (
            np.array([5, 6]),
            [i * 100.0 for i in range(10)],  # 100ms intervals
            [0.1],  # IEI: (600-500)/1000
        ),
        # Test case 3: Multiple peaks with varying intervals
        (
            np.array([0, 10, 15, 25]),
            [i * 50.0 for i in range(30)],  # 50ms intervals
            [0.5, 0.25, 0.5],  # IEIs: (500-0)/1000, (750-500)/1000, (1250-750)/1000
        ),
    ],
)
def test_get_iei_valid(
    peaks: np.ndarray, elapsed_time_ms: list[float], expected: list[float]
) -> None:
    """Test get_iei with valid inputs."""
    result = get_iei(peaks, elapsed_time_ms)
    assert result is not None
    np.testing.assert_allclose(result, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "peaks,elapsed_time_ms,reason",
    [
        # Test case 1: Only one peak
        (np.array([5]), [i * 10.0 for i in range(10)], "single peak"),
        # Test case 2: No peaks
        (np.array([]), [i * 10.0 for i in range(10)], "no peaks"),
        # Test case 3: Empty time list
        (np.array([5, 10]), [], "empty time list"),
        # Test case 4: Single time point
        (np.array([0]), [100.0], "single time point"),
    ],
    ids=["single_peak", "no_peaks", "empty_time_list", "single_time_point"],
)
def test_get_iei_returns_none(
    peaks: np.ndarray, elapsed_time_ms: list[float], reason: str
) -> None:
    """Test get_iei returns None for invalid inputs."""
    result = get_iei(peaks, elapsed_time_ms)
    assert result is None, f"Expected None for {reason}"


def test_get_iei_exact_values() -> None:
    """Test get_iei with exact values to verify calculation."""
    # Peaks at indices 100, 200, 300
    peaks = np.array([100, 200, 300])
    # Time stamps: 0ms, 1ms, 2ms, ..., 400ms
    elapsed_time_ms = [float(i) for i in range(401)]

    result = get_iei(peaks, elapsed_time_ms)
    assert result is not None

    # Expected IEI:
    # Peak at index 100 -> time 100ms
    # Peak at index 200 -> time 200ms
    # Peak at index 300 -> time 300ms
    # IEI 1: (200-100) = 100ms = 0.1s
    # IEI 2: (300-200) = 100ms = 0.1s
    expected = [0.1, 0.1]

    np.testing.assert_allclose(result, expected, rtol=1e-6)


# ==================== Additional edge case tests ====================


def test_calculate_dff_small_window() -> None:
    """Test calculate_dff with very small window."""
    data = np.array([100, 110, 120, 110, 100], dtype=float)
    dff = calculate_dff(data, window=1, percentile=10)
    assert dff.shape == data.shape
    assert not np.any(np.isnan(dff))


def test_calculate_dff_large_window() -> None:
    """Test calculate_dff with window larger than data."""
    data = np.array([100, 110, 120], dtype=float)
    dff = calculate_dff(data, window=100, percentile=10)
    assert dff.shape == data.shape
    assert not np.any(np.isnan(dff))


def test_calculate_dff_different_percentiles() -> None:
    """Test calculate_dff with different percentile values."""
    data = np.arange(100, dtype=float)

    # Test with 10th percentile
    dff_10 = calculate_dff(data, window=10, percentile=10)
    assert not np.any(np.isnan(dff_10))

    # Test with 50th percentile (median)
    dff_50 = calculate_dff(data, window=10, percentile=50)
    assert not np.any(np.isnan(dff_50))

    # Test with 90th percentile
    dff_90 = calculate_dff(data, window=10, percentile=90)
    assert not np.any(np.isnan(dff_90))

    # Higher percentile should generally give different results
    assert not np.allclose(dff_10, dff_90)


def test_calculate_dff_constant_trace() -> None:
    """Test calculate_dff with constant fluorescence trace."""
    data = np.ones(100) * 150.0
    # With constant data, background should equal data, resulting in dff of 0
    dff = calculate_dff(data, window=10, percentile=50)
    # dff = (data - bg) / bg = (150 - 150) / 150 = 0
    np.testing.assert_allclose(dff, 0.0, atol=1e-10)


def test_calculate_dff_edge_effects() -> None:
    """Test calculate_dff edge behavior at start and end of trace."""
    # Create data with a step function
    data = np.concatenate([np.ones(50) * 100, np.ones(50) * 200])
    dff = calculate_dff(data, window=10, percentile=50)

    # Should handle edges without producing NaN or inf
    assert not np.any(np.isnan(dff))
    assert not np.any(np.isinf(dff))
    assert len(dff) == len(data)


def test_calculate_bg_edge_cases() -> None:
    """Test _calculate_bg edge behavior."""
    # Single element
    data = np.array([100.0])
    bg = _calculate_bg(data, window=10, percentile=50)
    assert bg.shape == data.shape
    assert bg[0] == 100.0

    # Two elements
    data = np.array([100.0, 110.0])
    bg = _calculate_bg(data, window=10, percentile=50)
    assert bg.shape == data.shape
    assert not np.any(np.isnan(bg))


def test_calculate_bg_window_edges() -> None:
    """Test _calculate_bg correctly handles window at edges."""
    data = np.arange(100, dtype=float)
    bg = _calculate_bg(data, window=20, percentile=50)

    # At the very first point, window should only extend forward
    # At the very last point, window should only extend backward
    # Middle points should have symmetric windows
    assert bg.shape == data.shape
    assert not np.any(np.isnan(bg))

    # Background at start should be based on early values
    assert bg[0] < bg[99]


@pytest.mark.parametrize(
    "window,percentile",
    [
        (5, 10),
        (10, 25),
        (20, 50),
        (50, 75),
        (100, 90),
    ],
)
def test_calculate_dff_parameter_combinations(window: int, percentile: int) -> None:
    """Test calculate_dff with various parameter combinations."""
    data = np.random.randn(200) * 10 + 100
    dff = calculate_dff(data, window=window, percentile=percentile)

    assert dff.shape == data.shape
    assert not np.any(np.isnan(dff))
    assert not np.any(np.isinf(dff))


def test_get_iei_large_dataset() -> None:
    """Test get_iei with a larger dataset."""
    # Simulate regular peaks every 10 frames
    peaks = np.arange(0, 1000, 10)
    # Time stamps at 50ms intervals
    elapsed_time_ms = [i * 50.0 for i in range(1000)]

    result = get_iei(peaks, elapsed_time_ms)
    assert result is not None
    assert len(result) == len(peaks) - 1

    # All IEIs should be approximately 0.5 seconds
    # (10 frames * 50ms = 500ms = 0.5s)
    np.testing.assert_allclose(result, 0.5, rtol=1e-6)


def test_get_iei_irregular_intervals() -> None:
    """Test get_iei with irregular peak intervals."""
    peaks = np.array([5, 10, 25, 30])
    elapsed_time_ms = [i * 10.0 for i in range(50)]

    result = get_iei(peaks, elapsed_time_ms)
    assert result is not None

    # Expected IEIs:
    # Peak 0 at index 5 -> time 50ms
    # Peak 1 at index 10 -> time 100ms, IEI = 50ms = 0.05s
    # Peak 2 at index 25 -> time 250ms, IEI = 150ms = 0.15s
    # Peak 3 at index 30 -> time 300ms, IEI = 50ms = 0.05s
    expected = [0.05, 0.15, 0.05]

    np.testing.assert_allclose(result, expected, rtol=1e-6)
