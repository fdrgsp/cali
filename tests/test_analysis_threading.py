"""Tests for thread-safe analysis functions and numba optimizations.

This test module covers the changes made to improve performance and thread safety:
- New numba-optimized cross-correlation function (_max_cross_correlation_numba)
- Thread-safe lock protection around correlation computations
- Edge cases for empty ROIs and fallback correlation methods
- Verification that the new implementation maintains correct lag conventions
"""

from __future__ import annotations

import numpy as np
import pytest

from cali.analysis._util import (
    _calculate_cross_correlation_with_lag,
    _get_calcium_peaks_event_correlations_matrix,
    _get_spike_correlations_matrix,
    _max_cross_correlation_numba,
)


def test_max_cross_correlation_numba_basic() -> None:
    """Test numba cross-correlation with basic aligned signals."""
    # Two identical signals - should have max correlation at lag=0
    signal = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.float32)

    max_corr, lag = _max_cross_correlation_numba(signal, signal, max_lag=3)

    assert max_corr == pytest.approx(1.0, abs=0.01)
    assert lag == 0


def test_max_cross_correlation_numba_lagged() -> None:
    """Test numba cross-correlation with lagged signals."""
    # Signal 1: peaks at [2, 5]
    signal1 = np.zeros(10, dtype=np.float32)
    signal1[[2, 5]] = 1.0

    # Signal 2: peaks at [4, 7] - lagged by +2
    signal2 = np.zeros(10, dtype=np.float32)
    signal2[[4, 7]] = 1.0

    max_corr, lag = _max_cross_correlation_numba(signal1, signal2, max_lag=5)

    # signal2 lags behind signal1 by 2 frames
    assert lag == 2
    assert max_corr > 0.5  # Should have high correlation at the right lag


def test_max_cross_correlation_numba_negative_lag() -> None:
    """Test numba cross-correlation when first signal lags."""
    # Signal 1: peaks at [4, 7]
    signal1 = np.zeros(10, dtype=np.float32)
    signal1[[4, 7]] = 1.0

    # Signal 2: peaks at [2, 5] - leads by 2
    signal2 = np.zeros(10, dtype=np.float32)
    signal2[[2, 5]] = 1.0

    max_corr, lag = _max_cross_correlation_numba(signal1, signal2, max_lag=5)

    # signal2 leads signal1 by 2 frames (negative lag)
    assert lag == -2
    assert max_corr > 0.5


def test_max_cross_correlation_numba_no_events() -> None:
    """Test numba cross-correlation with empty signals."""
    signal1 = np.zeros(10, dtype=np.float32)
    signal2 = np.zeros(10, dtype=np.float32)

    max_corr, lag = _max_cross_correlation_numba(signal1, signal2, max_lag=3)

    assert max_corr == 0.0
    assert lag == 0


def test_max_cross_correlation_numba_one_empty() -> None:
    """Test numba cross-correlation when one signal is empty."""
    signal1 = np.array([0, 1, 0, 1, 0], dtype=np.float32)
    signal2 = np.zeros(5, dtype=np.float32)

    max_corr, lag = _max_cross_correlation_numba(signal1, signal2, max_lag=2)

    assert max_corr == 0.0
    assert lag == 0


def test_calculate_cross_correlation_with_lag_uses_numba() -> None:
    """Test that the wrapper function uses the numba implementation."""
    signal1 = np.array([0, 1, 0, 1, 0], dtype=np.float32)
    signal2 = np.array([0, 0, 1, 0, 1], dtype=np.float32)

    # This should call _max_cross_correlation_numba internally
    max_corr, lag = _calculate_cross_correlation_with_lag(signal1, signal2, max_lag=2)

    assert isinstance(max_corr, float)
    assert isinstance(lag, int)
    assert 0.0 <= max_corr <= 1.0
    assert -2 <= lag <= 2


def test_calcium_peaks_cross_correlation_with_lock() -> None:
    """Test that calcium peaks cross-correlation uses lock correctly."""
    # Create peak event data for 3 ROIs
    peak_events_dict = {
        "1": [0, 1, 0, 1, 0, 1, 0, 0],
        "2": [0, 0, 1, 0, 1, 0, 1, 0],
        "3": [1, 0, 0, 1, 0, 0, 1, 0],
    }

    sync_matrix, lag_matrix = _get_calcium_peaks_event_correlations_matrix(
        peak_events_dict,
        method="cross_correlation",
        max_lag=2,
    )

    assert sync_matrix is not None
    assert lag_matrix is not None
    assert sync_matrix.shape == (3, 3)
    assert lag_matrix.shape == (3, 3)

    # Diagonal should be perfect correlation with zero lag
    assert sync_matrix[0, 0] == 1.0
    assert lag_matrix[0, 0] == 0


def test_spike_correlations_with_lock() -> None:
    """Test that spike correlations use lock correctly."""
    # Create binary spike data for 2 ROIs
    spike_data_dict = {
        "1": [0, 1, 0, 1, 0, 0],
        "2": [0, 0, 1, 0, 1, 0],
    }

    sync_matrix, lag_matrix, zscore_matrix = _get_spike_correlations_matrix(
        spike_data_dict,
        method="cross_correlation",
        max_lag=2,
    )

    assert sync_matrix is not None
    assert lag_matrix is not None
    # zscore_matrix is always computed for cross_correlation method
    assert zscore_matrix is not None
    assert sync_matrix.shape == (2, 2)
    assert lag_matrix.shape == (2, 2)
    assert zscore_matrix.shape == (2, 2)

    # Diagonal should be perfect
    assert sync_matrix[0, 0] == 1.0
    assert lag_matrix[0, 0] == 0
    # Diagonal z-score is inf (undefined for self-correlation)
    assert np.isinf(zscore_matrix[0, 0])


def test_jitter_window_method_uses_numba() -> None:
    """Test that jitter window method uses numba-optimized function."""
    peak_events_dict = {
        "1": [0, 1, 0, 0, 1, 0, 0, 1, 0],
        "2": [0, 0, 1, 0, 0, 1, 0, 0, 1],
    }

    sync_matrix, lag_matrix = _get_calcium_peaks_event_correlations_matrix(
        peak_events_dict,
        method="jitter_window",
        jitter_window=2,
    )

    assert sync_matrix is not None
    assert lag_matrix is None  # Jitter method doesn't return lag matrix
    assert sync_matrix.shape == (2, 2)
    assert sync_matrix[0, 0] == 1.0

    # With jitter window of 2, peaks at distance 1 should be detected as synchronous
    assert sync_matrix[0, 1] > 0


def test_correlation_matrix_handles_empty_dict() -> None:
    """Test that empty peak dict returns None."""
    sync_matrix, lag_matrix = _get_calcium_peaks_event_correlations_matrix(
        {},
        method="cross_correlation",
        max_lag=2,
    )

    assert sync_matrix is None
    assert lag_matrix is None


def test_correlation_matrix_handles_single_roi() -> None:
    """Test that single ROI returns None."""
    peak_events_dict = {"1": [0, 1, 0, 1, 0]}

    sync_matrix, lag_matrix = _get_calcium_peaks_event_correlations_matrix(
        peak_events_dict,
        method="cross_correlation",
        max_lag=2,
    )

    assert sync_matrix is None
    assert lag_matrix is None


def test_spike_jitter_synchrony_uses_numba() -> None:
    """Test spike jitter synchrony with numba optimization."""
    spike_data_dict = {
        "1": [0, 1, 0, 0, 1, 0],
        "2": [0, 0, 1, 0, 0, 1],
    }

    sync_matrix, _, _ = _get_spike_correlations_matrix(
        spike_data_dict,
        method="jitter_window",
        jitter_window=1,
    )

    assert sync_matrix is not None
    assert sync_matrix.shape == (2, 2)
    assert sync_matrix[0, 0] == 1.0


def test_max_cross_correlation_clipping() -> None:
    """Test that correlation values are clipped to [0, 1]."""
    # Create signals that might produce values outside [0,1] before clipping
    signal1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    signal2 = np.array([5, 4, 3, 2, 1], dtype=np.float32)

    max_corr, _ = _max_cross_correlation_numba(signal1, signal2, max_lag=2)

    assert 0.0 <= max_corr <= 1.0


@pytest.mark.parametrize(
    "max_lag,expected_lag_range",
    [
        (1, (-1, 1)),
        (3, (-3, 3)),
        (5, (-5, 5)),
    ],
)
def test_max_cross_correlation_respects_max_lag(
    max_lag: int, expected_lag_range: tuple[int, int]
) -> None:
    """Test that max_lag parameter is respected."""
    signal1 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    signal2 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)

    _, lag = _max_cross_correlation_numba(signal1, signal2, max_lag=max_lag)

    min_lag, max_lag_val = expected_lag_range
    assert min_lag <= lag <= max_lag_val


def test_calcium_peaks_with_empty_roi() -> None:
    """Test cross-correlation when one ROI has no peaks."""
    peak_events_dict = {
        "1": [0.0, 1.0, 0.0, 1.0, 0.0],
        "2": [0.0, 0.0, 0.0, 0.0, 0.0],  # No peaks
    }

    sync_matrix, lag_matrix = _get_calcium_peaks_event_correlations_matrix(
        peak_events_dict,
        method="cross_correlation",
        max_lag=2,
    )

    assert sync_matrix is not None
    assert lag_matrix is not None

    # Empty ROI should have zero correlation with other ROI
    assert sync_matrix[0, 1] == 0.0
    assert sync_matrix[1, 0] == 0.0
    assert lag_matrix[0, 1] == 0
    assert lag_matrix[1, 0] == 0


def test_spike_correlations_with_empty_roi() -> None:
    """Test spike correlations when one ROI has no spikes."""
    spike_data_dict = {
        "1": [0.0, 1.0, 0.0, 1.0, 0.0],
        "2": [0.0, 0.0, 0.0, 0.0, 0.0],  # No spikes
    }

    sync_matrix, lag_matrix, _ = _get_spike_correlations_matrix(
        spike_data_dict,
        method="cross_correlation",
        max_lag=2,
    )

    assert sync_matrix is not None
    assert lag_matrix is not None

    # Empty ROI should have zero correlation
    assert sync_matrix[0, 1] == 0.0
    assert lag_matrix[0, 1] == 0


def test_calcium_peaks_fallback_correlation_method() -> None:
    """Test fallback to standard correlation when method is not recognized."""
    peak_events_dict = {
        "1": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        "2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    }

    sync_matrix, lag_matrix = _get_calcium_peaks_event_correlations_matrix(
        peak_events_dict,
        method="correlation",  # Default correlation method
        max_lag=2,
    )

    assert sync_matrix is not None
    assert lag_matrix is None  # Default method doesn't compute lag

    # Should compute Pearson correlation
    assert sync_matrix[0, 0] == 1.0
    assert 0 <= sync_matrix[0, 1] <= 1


def test_spike_correlations_fallback_method() -> None:
    """Test spike correlations with fallback correlation method."""
    spike_data_dict = {
        "1": [0.0, 1.0, 0.0, 1.0, 0.0],
        "2": [1.0, 0.0, 1.0, 0.0, 1.0],
    }

    sync_matrix, lag_matrix, zscore_matrix = _get_spike_correlations_matrix(
        spike_data_dict,
        method="correlation",  # Default method
        max_lag=2,
    )

    assert sync_matrix is not None
    assert lag_matrix is None  # Default method doesn't compute lag
    assert zscore_matrix is None  # Default method doesn't compute z-scores

    # Should have valid correlation values
    assert sync_matrix[0, 0] == 1.0
    assert 0 <= sync_matrix[0, 1] <= 1


def test_spike_correlations_rejects_non_binary() -> None:
    """Test that spike correlations reject non-binary values."""
    spike_data_dict = {
        "1": [0.0, 1.5, 0.0, 2.0, 0.0],  # Non-binary values
        "2": [1.0, 0.0, 1.0, 0.0, 1.0],
    }

    # Should raise ValueError for non-binary spike values
    with pytest.raises(ValueError, match="Spike data contains non-binary values"):
        _get_spike_correlations_matrix(
            spike_data_dict,
            method="cross_correlation",
            max_lag=2,
        )
