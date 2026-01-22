"""Test improved CCG (cross-correlogram) functions.

These tests validate the standard CCG implementation based on
best practices from spike train analysis and calcium imaging literature.

The implementation uses:
- Per-trigger probability normalization (P(spike in j | spike in i))
- Border correction for unbiased estimates at large lags
- Shift predictor baseline correction for significance testing
"""

import numpy as np
import pytest

from cali.analysis._util import (
    _compute_baseline_corrected_ccg,
    _compute_ccg_vector,
    _summarize_ccg_near_zero,
)


def test_ccg_vector_computation_basic() -> None:
    """Test that CCG vector is computed correctly for simple case."""
    # Create two spike trains with a known relationship
    events_i = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    events_j = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)

    lags, ccg = _compute_ccg_vector(events_i, events_j, max_lag=3)

    # Check shape
    assert len(lags) == 7  # -3 to +3
    assert len(ccg) == 7
    assert lags[0] == -3
    assert lags[-1] == 3

    # At lag=1, events_j should align well with events_i
    # (i has spikes at 0,4,8 and j has spikes at 1,5,9)
    lag_1_idx = np.where(lags == 1)[0][0]
    assert ccg[lag_1_idx] > 0.5  # Should have high correlation at lag=1


def test_ccg_trigger_prob_normalization() -> None:
    """Test trigger_prob normalization (now the default and only option)."""
    # 2 spikes in reference, both align with target at lag=0
    events_i = np.array([1, 0, 1, 0, 0], dtype=np.float32)
    events_j = np.array([1, 0, 1, 0, 0], dtype=np.float32)

    lags, ccg = _compute_ccg_vector(events_i, events_j, max_lag=1)

    # 2 coincidences / 2 reference spikes = 1.0 probability
    # With border correction, the value may be slightly different
    lag_0_idx = np.where(lags == 0)[0][0]
    assert ccg[lag_0_idx] > 0.9  # Should be close to 1.0


def test_ccg_empty_reference_train() -> None:
    """Test handling of empty reference train."""
    events_i = np.zeros(10, dtype=np.float32)
    events_j = np.ones(10, dtype=np.float32)

    _lags, ccg = _compute_ccg_vector(events_i, events_j, max_lag=2)

    # Should return zeros
    assert np.all(ccg == 0.0)


def test_ccg_empty_target_train() -> None:
    """Test handling of empty target train."""
    events_i = np.ones(10, dtype=np.float32)
    events_j = np.zeros(10, dtype=np.float32)

    _lags, ccg = _compute_ccg_vector(events_i, events_j, max_lag=2)

    # Should return zeros (no coincidences possible)
    assert np.all(ccg == 0.0)


def test_baseline_corrected_ccg_structure() -> None:
    """Test that baseline correction returns proper structure."""
    np.random.seed(42)
    events_i = np.random.binomial(1, 0.1, 200).astype(np.float32)
    events_j = np.random.binomial(1, 0.1, 200).astype(np.float32)

    lags, ccg_raw, baseline_mean, baseline_std = _compute_baseline_corrected_ccg(
        events_i, events_j, max_lag=10, n_shuffles=50
    )

    # Check shapes
    assert len(lags) == 21  # -10 to +10
    assert len(ccg_raw) == 21
    assert len(baseline_mean) == 21
    assert len(baseline_std) == 21

    # Baseline std should be > 0 (there's variability in shuffles)
    assert np.all(baseline_std >= 0.0)
    assert np.any(baseline_std > 0.0)


def test_baseline_corrected_ccg_removes_slow_comodulation() -> None:
    """Test that baseline correction handles slow co-modulations."""
    # Create two trains with slow co-modulation (both increase activity over time)
    # but no fast coupling
    n = 200
    events_i = np.zeros(n, dtype=np.float32)
    events_j = np.zeros(n, dtype=np.float32)

    # Add increasing spike probability
    np.random.seed(123)
    for i in range(n):
        prob = i / n * 0.3  # Probability increases over time
        if np.random.random() < prob:
            events_i[i] = 1.0
        if np.random.random() < prob:
            events_j[i] = 1.0

    np.random.seed(42)
    lags, ccg_raw, baseline_mean, baseline_std = _compute_baseline_corrected_ccg(
        events_i, events_j, max_lag=5, n_shuffles=100
    )

    # The raw CCG might be elevated due to slow co-modulation
    # The baseline should capture this
    lag_0_idx = np.where(lags == 0)[0][0]

    # Baseline should be > 0 due to slow co-modulation
    assert baseline_mean[lag_0_idx] > 0.0

    # Z-score should be modest (no strong fast coupling)
    z_score = (ccg_raw[lag_0_idx] - baseline_mean[lag_0_idx]) / (
        baseline_std[lag_0_idx] + 1e-10
    )
    assert abs(z_score) < 3.0  # Not a strong outlier


def test_baseline_corrected_ccg_detects_true_coupling() -> None:
    """Test that baseline correction preserves true coupling signal."""
    # Create two trains with true coupling at lag=0
    n = 200
    events_i = np.zeros(n, dtype=np.float32)
    events_j = np.zeros(n, dtype=np.float32)

    # Add coupled spikes
    np.random.seed(42)
    for i in range(n):
        if np.random.random() < 0.1:
            events_i[i] = 1.0
            events_j[i] = 1.0  # Perfect coupling at lag=0

    lags, ccg_raw, baseline_mean, baseline_std = _compute_baseline_corrected_ccg(
        events_i, events_j, max_lag=5, n_shuffles=100
    )

    # Z-score at lag=0 should be high
    lag_0_idx = np.where(lags == 0)[0][0]
    z_score = (ccg_raw[lag_0_idx] - baseline_mean[lag_0_idx]) / (
        baseline_std[lag_0_idx] + 1e-10
    )

    assert z_score > 5.0  # Strong coupling signal


def test_summarize_ccg_near_zero() -> None:
    """Test CCG summary around zero lag."""
    # Create a CCG with peak at zero
    lags = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    ccg = np.array([0.1, 0.1, 0.2, 0.3, 0.5, 1.0, 0.5, 0.3, 0.2, 0.1, 0.1])

    # Window = 1 should capture [-1, 0, 1]
    summary = _summarize_ccg_near_zero(lags, ccg, window=1)
    assert summary == pytest.approx(1.0, abs=0.01)  # Peak at 0

    # Window = 2 should capture [-2, -1, 0, 1, 2]
    summary = _summarize_ccg_near_zero(lags, ccg, window=2)
    assert summary == pytest.approx(1.0, abs=0.01)  # Still peak at 0


def test_summarize_ccg_near_zero_with_offset_peak() -> None:
    """Test CCG summary when peak is offset from zero."""
    # Create a CCG with peak at lag=2
    lags = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    ccg = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 1.0, 0.3, 0.1, 0.1])

    # Small window should miss the peak
    summary = _summarize_ccg_near_zero(lags, ccg, window=1)
    assert summary < 0.5  # Doesn't capture the peak at lag=2

    # Larger window captures the peak
    summary = _summarize_ccg_near_zero(lags, ccg, window=3)
    assert summary == pytest.approx(1.0, abs=0.01)


def test_ccg_symmetry_property() -> None:
    """Test that CCG(i,j, lag) relates to CCG(j,i, -lag)."""
    np.random.seed(42)
    events_i = np.random.binomial(1, 0.1, 100).astype(np.float32)
    events_j = np.random.binomial(1, 0.1, 100).astype(np.float32)

    # Compute CCG(i,j)
    lags_ij, ccg_ij = _compute_ccg_vector(events_i, events_j, max_lag=5)

    # Compute CCG(j,i)
    lags_ji, ccg_ji = _compute_ccg_vector(events_j, events_i, max_lag=5)

    # With per-trigger normalization:
    # CCG_ij(tau) = count(tau) / N_i
    # CCG_ji(-tau) = count(-tau) / N_j = count(tau) / N_j
    # So CCG_ij(tau) * N_i = CCG_ji(-tau) * N_j = count(tau)

    n_i = np.sum(events_i)
    n_j = np.sum(events_j)

    # Check that raw counts match (within numerical precision)
    for lag in [0, 1, 2, -1, -2]:
        idx_ij = np.where(lags_ij == lag)[0][0]
        idx_ji = np.where(lags_ji == -lag)[0][0]

        # Raw count should be the same
        count_from_ij = ccg_ij[idx_ij] * n_i
        count_from_ji = ccg_ji[idx_ji] * n_j

        assert count_from_ij == pytest.approx(count_from_ji, rel=0.01)


def test_ccg_with_shifted_perfect_correlation() -> None:
    """Test CCG correctly identifies shifted correlation."""
    # events_j is events_i shifted by 2 frames
    n = 100
    events_i = np.zeros(n, dtype=np.float32)
    events_i[0::10] = 1.0  # Spikes every 10 frames

    # Shift by 2 frames
    events_j = np.zeros(n, dtype=np.float32)
    events_j[2::10] = 1.0  # Same pattern but shifted by 2

    lags, ccg = _compute_ccg_vector(events_i, events_j, max_lag=5)

    # Should have peak at lag=2 (j lags behind i by 2 frames)
    # Find peak in the range [-5, 5]
    peak_lag_idx = np.argmax(ccg)
    peak_lag = lags[peak_lag_idx]
    assert peak_lag == 2


def test_summarize_ccg_invalid_window() -> None:
    """Test that summarize_ccg_near_zero handles invalid windows gracefully."""
    lags = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], dtype=np.int32)
    ccg = np.array([0.1, 0.1, 0.2, 0.3, 0.5, 1.0, 0.5, 0.3, 0.2, 0.1, 0.1])

    # Window that's larger than the lag range should still work
    summary = _summarize_ccg_near_zero(lags, ccg, window=100)
    assert summary == pytest.approx(1.0, abs=0.01)  # Still finds the peak

    # Window size of 0 should just check lag=0
    summary = _summarize_ccg_near_zero(lags, ccg, window=0)
    assert summary == pytest.approx(1.0, abs=0.01)  # Peak is at 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
