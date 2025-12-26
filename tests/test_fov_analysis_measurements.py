"""Test FOV-level analysis measurements for correlation and synchrony.

This module tests the 6 different metrics computed in compute_fov_analysis:
- Calcium: zero-lag correlation, jitter synchrony, max lag correlation
- Spikes: zero-lag correlation, max lag correlation, jitter synchrony
"""

from __future__ import annotations

import numpy as np

from cali.analysis._fov_analysis import (
    _compute_zero_lag_corr_matrix,
    compute_fov_analysis,
)
from cali.sqlmodel import FOV, ROI, AnalysisSettings, DataAnalysis, Traces


def test_zero_lag_correlation_on_dff_traces() -> None:
    """Test zero-lag Pearson correlation on DF/F traces."""
    # Create perfectly correlated traces
    trace1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    trace2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    traces = [trace1, trace2]

    corr_matrix = _compute_zero_lag_corr_matrix(traces)

    assert corr_matrix is not None
    assert corr_matrix.shape == (2, 2)
    # Perfect correlation
    assert np.allclose(corr_matrix[0, 1], 1.0, atol=1e-10)
    assert np.allclose(corr_matrix[1, 0], 1.0, atol=1e-10)
    # Diagonal is always 1
    assert corr_matrix[0, 0] == 1.0
    assert corr_matrix[1, 1] == 1.0


def test_zero_lag_correlation_anticorrelated() -> None:
    """Test zero-lag correlation detects anticorrelation."""
    # Perfectly anticorrelated traces
    trace1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    trace2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    traces = [trace1, trace2]

    corr_matrix = _compute_zero_lag_corr_matrix(traces)

    assert corr_matrix is not None
    # Perfect anticorrelation
    assert np.allclose(corr_matrix[0, 1], -1.0, atol=1e-10)
    assert np.allclose(corr_matrix[1, 0], -1.0, atol=1e-10)


def test_zero_lag_correlation_phase_shifted() -> None:
    """Test zero-lag correlation is low for phase-shifted signals."""
    # Phase-shifted sine waves (90 degrees apart)
    t = np.linspace(0, 2 * np.pi, 100)
    trace1 = np.sin(t)
    trace2 = np.cos(t)  # 90 degree phase shift
    traces = [trace1, trace2]

    corr_matrix = _compute_zero_lag_corr_matrix(traces)

    assert corr_matrix is not None
    # Zero-lag correlation should be near zero for 90-degree phase shift
    assert abs(corr_matrix[0, 1]) < 0.1


def test_zero_lag_correlation_constant_traces() -> None:
    """Test zero-lag correlation handles constant traces."""
    # Constant traces should have zero correlation
    trace1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    trace2 = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Constant
    traces = [trace1, trace2]

    corr_matrix = _compute_zero_lag_corr_matrix(traces)

    assert corr_matrix is not None
    # Correlation with constant trace is 0
    assert np.allclose(corr_matrix[0, 1], 0.0, atol=1e-10)


def test_zero_lag_correlation_multiple_traces() -> None:
    """Test zero-lag correlation with multiple traces."""
    # Create 4 traces with known relationships
    trace1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    trace2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Same as trace1
    trace3 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Anticorrelated with trace1
    trace4 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # 2x trace1
    traces = [trace1, trace2, trace3, trace4]

    corr_matrix = _compute_zero_lag_corr_matrix(traces)

    assert corr_matrix is not None
    assert corr_matrix.shape == (4, 4)
    # trace1 == trace2: perfect correlation
    assert np.allclose(corr_matrix[0, 1], 1.0, atol=1e-10)
    # trace1 anticorrelated with trace3
    assert np.allclose(corr_matrix[0, 2], -1.0, atol=1e-10)
    # trace1 perfectly correlated with trace4 (scaled version)
    assert np.allclose(corr_matrix[0, 3], 1.0, atol=1e-10)


def test_compute_fov_analysis_with_calcium_measurements() -> None:
    """Test that compute_fov_analysis computes all calcium measurements."""
    # Create a minimal FOV with 2 ROIs
    fov = FOV(name="test_fov", position_index=1)

    # Create ROI 1 with spike data
    roi1 = ROI(label_value=1, active=True, fov_id=fov.id)
    # Create spike pattern: spikes at frames 5, 15, 25, 35, 45
    spike_pattern1 = np.zeros(50)
    spike_pattern1[[5, 15, 25, 35, 45]] = 2.0
    traces1 = Traces(
        dff=[1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        dec_dff=[1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        inferred_spikes=spike_pattern1.tolist(),
    )
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(
        peaks_dec_dff=[5, 15, 25, 35, 45], inferred_spikes_threshold=1.0
    )
    roi1._new_data_analysis = [data_analysis1]

    # Create ROI 2 with spike data (shifted by 1 frame)
    roi2 = ROI(label_value=2, active=True, fov_id=fov.id)
    spike_pattern2 = np.zeros(50)
    spike_pattern2[[6, 16, 26, 36, 46]] = 2.0
    traces2 = Traces(
        dff=[1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        dec_dff=[1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        inferred_spikes=spike_pattern2.tolist(),
    )
    roi2._new_traces = [traces2]
    data_analysis2 = DataAnalysis(
        peaks_dec_dff=[6, 16, 26, 36, 46], inferred_spikes_threshold=1.0
    )
    roi2._new_data_analysis = [data_analysis2]

    fov.rois = [roi1, roi2]

    # Create analysis settings (values in milliseconds)
    settings = AnalysisSettings(
        spikes_sync_jitter_window=200,  # 2 frames at 10fps = 200ms
        spikes_sync_cross_corr_lag=500,  # 5 frames at 10fps = 500ms
    )

    # Compute analysis
    fov_analysis = compute_fov_analysis(fov, settings)

    assert fov_analysis is not None

    # Check that all calcium and spike measurements are computed
    assert fov_analysis.calcium_dff_correlation_matrix is not None
    assert fov_analysis.spike_jitter_synchrony_matrix is not None
    assert fov_analysis.spike_max_lag_correlation_matrix is not None

    # Check global metrics
    assert fov_analysis.global_spike_jitter_synchrony is not None
    assert fov_analysis.global_spike_max_lag_correlation is not None

    # Check matrix shapes (2x2 for 2 ROIs)
    assert len(fov_analysis.calcium_dff_correlation_matrix) == 2
    assert len(fov_analysis.calcium_dff_correlation_matrix[0]) == 2


def test_compute_fov_analysis_with_spike_measurements() -> None:
    """Test that compute_fov_analysis computes all spike measurements."""
    # Create a minimal FOV with 2 ROIs with spike data
    fov = FOV(name="test_fov", position_index=1)

    # Create binary spike pattern
    spike_pattern = np.zeros(50)
    spike_pattern[[5, 15, 25, 35, 45]] = 2.0  # Spike amplitude = 2.0

    # Create ROI 1
    roi1 = ROI(label_value=1, active=True, fov_id=fov.id)
    traces1 = Traces(
        dff=[1.0] * 50,
        dec_dff=[1.0] * 50,
        inferred_spikes=spike_pattern.tolist(),
    )
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(
        peaks_dec_dff=[5, 15, 25],
        inferred_spikes_threshold=1.0,
    )
    roi1._new_data_analysis = [data_analysis1]

    # Create ROI 2 with slightly shifted spikes
    spike_pattern2 = np.zeros(50)
    spike_pattern2[[6, 16, 26, 36, 46]] = 2.0

    roi2 = ROI(label_value=2, active=True, fov_id=fov.id)
    traces2 = Traces(
        dff=[1.0] * 50,
        dec_dff=[1.0] * 50,
        inferred_spikes=spike_pattern2.tolist(),
    )
    roi2._new_traces = [traces2]
    data_analysis2 = DataAnalysis(
        peaks_dec_dff=[6, 16, 26],
        inferred_spikes_threshold=1.0,
    )
    roi2._new_data_analysis = [data_analysis2]

    fov.rois = [roi1, roi2]

    # Create analysis settings (values in milliseconds)
    settings = AnalysisSettings(
        spikes_sync_jitter_window=200,  # 2 frames at 10fps = 200ms
        spikes_sync_cross_corr_lag=500,  # 5 frames at 10fps = 500ms
    )

    # Compute FOV analysis
    fov_analysis = compute_fov_analysis(fov, settings)

    assert fov_analysis is not None

    # Check that spike measurements are computed
    assert fov_analysis.spike_max_lag_correlation_matrix is not None
    assert fov_analysis.spike_jitter_synchrony_matrix is not None

    # Check global metrics
    assert fov_analysis.global_spike_max_lag_correlation is not None
    assert fov_analysis.global_spike_jitter_synchrony is not None

    # Check matrix shapes (2x2 for 2 ROIs)
    assert len(fov_analysis.spike_max_lag_correlation_matrix) == 2
    assert len(fov_analysis.spike_max_lag_correlation_matrix[0]) == 2


def test_compute_fov_analysis_insufficient_rois() -> None:
    """Test that compute_fov_analysis returns None with < 2 ROIs."""
    fov = FOV(name="test_fov", position_index=1)

    # Create only 1 ROI
    roi1 = ROI(label_value=1, active=True, fov_id=fov.id)
    traces1 = Traces(dec_dff=[1.0, 2.0, 3.0])
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(peaks_dec_dff=[1])
    roi1._new_data_analysis = [data_analysis1]

    fov.rois = [roi1]

    settings = AnalysisSettings()

    fov_analysis = compute_fov_analysis(fov, settings)

    # Should return None with insufficient ROIs
    assert fov_analysis is None


def test_compute_fov_analysis_inactive_rois() -> None:
    """Test that inactive ROIs are excluded from analysis."""
    fov = FOV(name="test_fov", position_index=1)

    # Create 2 ROIs but mark one as inactive
    roi1 = ROI(label_value=1, active=True, fov_id=fov.id)
    traces1 = Traces(dec_dff=[1.0, 2.0, 3.0])
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(peaks_dec_dff=[1])
    roi1._new_data_analysis = [data_analysis1]

    roi2 = ROI(label_value=2, active=False, fov_id=fov.id)  # Inactive
    traces2 = Traces(dec_dff=[4.0, 5.0, 6.0])
    roi2._new_traces = [traces2]

    fov.rois = [roi1, roi2]

    settings = AnalysisSettings()

    fov_analysis = compute_fov_analysis(fov, settings)

    # Should return None because only 1 active ROI
    assert fov_analysis is None


def test_jitter_synchrony_vs_max_lag_correlation() -> None:
    """Test that jitter synchrony and max lag correlation give different results."""
    fov = FOV(name="test_fov", position_index=1)

    # Create ROI 1 with peaks at [10, 20, 30]
    roi1 = ROI(label_value=1, active=True, fov_id=fov.id)
    peak_array1 = np.zeros(50)
    peak_array1[[10, 20, 30]] = 1.0
    spike_pattern1 = np.zeros(50)
    spike_pattern1[[10, 20, 30]] = 2.0
    traces1 = Traces(
        dff=peak_array1.tolist(),
        dec_dff=peak_array1.tolist(),
        inferred_spikes=spike_pattern1.tolist(),
    )
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(
        peaks_dec_dff=[10, 20, 30], inferred_spikes_threshold=1.0
    )
    roi1._new_data_analysis = [data_analysis1]

    # Create ROI 2 with peaks at [15, 25, 35] - shifted by 5 frames
    roi2 = ROI(label_value=2, active=True, fov_id=fov.id)
    peak_array2 = np.zeros(50)
    peak_array2[[15, 25, 35]] = 1.0
    spike_pattern2 = np.zeros(50)
    spike_pattern2[[15, 25, 35]] = 2.0
    traces2 = Traces(
        dff=peak_array2.tolist(),
        dec_dff=peak_array2.tolist(),
        inferred_spikes=spike_pattern2.tolist(),
    )
    roi2._new_traces = [traces2]
    data_analysis2 = DataAnalysis(
        peaks_dec_dff=[15, 25, 35], inferred_spikes_threshold=1.0
    )
    roi2._new_data_analysis = [data_analysis2]

    fov.rois = [roi1, roi2]

    settings = AnalysisSettings(
        # Too small to capture 5-frame shift (2 frames at 10fps)
        spikes_sync_jitter_window=200,
        # Large enough to capture shift (10 frames at 10fps)
        spikes_sync_cross_corr_lag=1000,
    )

    fov_analysis = compute_fov_analysis(fov, settings)

    assert fov_analysis is not None

    jitter_sync = fov_analysis.spike_jitter_synchrony_matrix
    max_lag_corr = fov_analysis.spike_max_lag_correlation_matrix

    assert jitter_sync is not None
    assert max_lag_corr is not None

    # Jitter synchrony should be LOW (peaks are 5 frames apart, window is 2)
    # Max lag correlation should be HIGH (can find the 5-frame shift)
    # This demonstrates they measure different things
    assert jitter_sync[0][1] < max_lag_corr[0][1]


def test_spike_max_lag_values_matrix() -> None:
    """Test that spike max-lag values matrix correctly identifies lag."""
    fov = FOV(name="test_fov", position_index=1)

    # Create ROI 1 with spikes at [10, 20, 30]
    roi1 = ROI(label_value=1, active=True, fov_id=fov.id)
    spike_array1 = np.zeros(50)
    spike_array1[[10, 20, 30]] = 1.5  # Spike amplitudes
    traces1 = Traces(
        dff=spike_array1.tolist(),
        dec_dff=spike_array1.tolist(),
        inferred_spikes=spike_array1.tolist(),
    )
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(inferred_spikes_threshold=1.0)
    roi1._new_data_analysis = [data_analysis1]

    # Create ROI 2 with spikes at [13, 23, 33] - shifted by +3 frames
    roi2 = ROI(label_value=2, active=True, fov_id=fov.id)
    spike_array2 = np.zeros(50)
    spike_array2[[13, 23, 33]] = 1.5
    traces2 = Traces(
        dff=spike_array2.tolist(),
        dec_dff=spike_array2.tolist(),
        inferred_spikes=spike_array2.tolist(),
    )
    roi2._new_traces = [traces2]
    data_analysis2 = DataAnalysis(inferred_spikes_threshold=1.0)
    roi2._new_data_analysis = [data_analysis2]

    fov.rois = [roi1, roi2]

    settings = AnalysisSettings(
        spikes_sync_cross_corr_lag=500,  # 5 frames at 10fps = 500ms
    )

    fov_analysis = compute_fov_analysis(fov, settings)

    assert fov_analysis is not None
    assert fov_analysis.spike_max_lag_values_matrix is not None
    assert fov_analysis.spike_max_lag_correlation_matrix is not None

    lag_matrix = fov_analysis.spike_max_lag_values_matrix

    # Check matrix shape
    assert len(lag_matrix) == 2
    assert len(lag_matrix[0]) == 2

    # Diagonal should be zero (no lag with self)
    assert lag_matrix[0][0] == 0
    assert lag_matrix[1][1] == 0

    # roi2 lags behind roi1 by 3 frames
    # New numba implementation: positive lag means j lags behind i
    # Since roi1 spikes at [10,20,30] and roi2 at [13,23,33], roi2 comes after roi1
    # So from roi1's perspective (row 0), roi2 (column 1) has positive lag +3
    assert lag_matrix[0][1] == 3

    # From roi2's perspective (row 1), roi1 (column 0) leads, so negative lag
    assert lag_matrix[1][0] == -3

    # Correlation should be high since the spike patterns match well
    corr_matrix = fov_analysis.spike_max_lag_correlation_matrix
    assert corr_matrix[0][1] > 0.9  # High correlation after time shift
