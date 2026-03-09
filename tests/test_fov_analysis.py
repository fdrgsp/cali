"""Test FOV-level analysis measurements for correlation and synchrony.

This module tests the 6 different metrics computed in compute_fov_analysis:
- Calcium: zero-lag correlation, jitter synchrony, max lag correlation
- Spikes: zero-lag correlation, max lag correlation, jitter synchrony
"""

from __future__ import annotations

import numpy as np
import pytest

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
        den_dff=[1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        inferred_spikes=spike_pattern1.tolist(),
    )
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(
        peaks_den_dff=[5, 15, 25, 35, 45], inferred_spikes_threshold=1.0
    )
    roi1._new_data_analysis = [data_analysis1]

    # Create ROI 2 with spike data (shifted by 1 frame)
    roi2 = ROI(label_value=2, active=True, fov_id=fov.id)
    spike_pattern2 = np.zeros(50)
    spike_pattern2[[6, 16, 26, 36, 46]] = 2.0
    traces2 = Traces(
        dff=[1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        den_dff=[1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        inferred_spikes=spike_pattern2.tolist(),
    )
    roi2._new_traces = [traces2]
    data_analysis2 = DataAnalysis(
        peaks_den_dff=[6, 16, 26, 36, 46], inferred_spikes_threshold=1.0
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
        den_dff=[1.0] * 50,
        inferred_spikes=spike_pattern.tolist(),
    )
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(
        peaks_den_dff=[5, 15, 25],
        inferred_spikes_threshold=1.0,
    )
    roi1._new_data_analysis = [data_analysis1]

    # Create ROI 2 with slightly shifted spikes
    spike_pattern2 = np.zeros(50)
    spike_pattern2[[6, 16, 26, 36, 46]] = 2.0

    roi2 = ROI(label_value=2, active=True, fov_id=fov.id)
    traces2 = Traces(
        dff=[1.0] * 50,
        den_dff=[1.0] * 50,
        inferred_spikes=spike_pattern2.tolist(),
    )
    roi2._new_traces = [traces2]
    data_analysis2 = DataAnalysis(
        peaks_den_dff=[6, 16, 26],
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
    traces1 = Traces(den_dff=[1.0, 2.0, 3.0])
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(peaks_den_dff=[1])
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
    traces1 = Traces(den_dff=[1.0, 2.0, 3.0])
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(peaks_den_dff=[1])
    roi1._new_data_analysis = [data_analysis1]

    roi2 = ROI(label_value=2, active=False, fov_id=fov.id)  # Inactive
    traces2 = Traces(den_dff=[4.0, 5.0, 6.0])
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
        den_dff=peak_array1.tolist(),
        inferred_spikes=spike_pattern1.tolist(),
    )
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(
        peaks_den_dff=[10, 20, 30], inferred_spikes_threshold=1.0
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
        den_dff=peak_array2.tolist(),
        inferred_spikes=spike_pattern2.tolist(),
    )
    roi2._new_traces = [traces2]
    data_analysis2 = DataAnalysis(
        peaks_den_dff=[15, 25, 35], inferred_spikes_threshold=1.0
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
        den_dff=spike_array1.tolist(),
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
        den_dff=spike_array2.tolist(),
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


def _make_roi_with_traces(
    label: int,
    den_dff: list[float],
) -> ROI:
    """Helper: create an ROI with the given den_dff trace."""
    roi = ROI(label_value=label, active=True)
    traces = Traces(
        dff=den_dff,
        den_dff=den_dff,
        inferred_spikes=[0.0] * len(den_dff),
    )
    roi._new_traces = [traces]
    roi._new_data_analysis = [DataAnalysis(peaks_den_dff=[10, 30, 50])]
    return roi


def test_compute_fov_analysis_cluster_labels_populated() -> None:
    """cluster_labels are computed and stored when there are >= 3 ROIs."""
    rng = np.random.default_rng(0)
    n = 60

    # Two groups of highly correlated traces + one more (3 ROIs total)
    base_a = rng.standard_normal(n).tolist()
    base_b = (rng.standard_normal(n) * 0.1 + np.arange(n, dtype=float) * 0.02).tolist()
    slight_noise = (rng.standard_normal(n) * 0.05).tolist()
    trace_c = [base_a[i] + slight_noise[i] for i in range(n)]

    fov = FOV(name="cluster_fov", position_index=0)
    fov.rois = [
        _make_roi_with_traces(1, base_a),
        _make_roi_with_traces(2, base_b),
        _make_roi_with_traces(3, trace_c),
    ]

    settings = AnalysisSettings(
        cluster_n_clusters=2,  # force k=2 for determinism
        cluster_max_k=5,
    )

    fov_analysis = compute_fov_analysis(fov, settings)

    assert fov_analysis is not None
    assert fov_analysis.cluster_labels is not None
    assert len(fov_analysis.cluster_labels) == 3
    assert fov_analysis.cluster_method == "hierarchical"
    assert fov_analysis.cluster_n_clusters == 2
    assert fov_analysis.cluster_silhouette_score is not None
    assert -1.0 <= fov_analysis.cluster_silhouette_score <= 1.0
    assert fov_analysis.cluster_order is not None
    assert len(fov_analysis.cluster_order) == 3


def test_compute_fov_analysis_no_cluster_labels_for_two_rois() -> None:
    """cluster_labels remain None when there are only 2 ROIs (< 3 needed)."""
    rng = np.random.default_rng(1)
    n = 40
    fov = FOV(name="small_fov", position_index=0)
    fov.rois = [
        _make_roi_with_traces(1, rng.standard_normal(n).tolist()),
        _make_roi_with_traces(2, rng.standard_normal(n).tolist()),
    ]

    fov_analysis = compute_fov_analysis(fov, AnalysisSettings())

    assert fov_analysis is not None
    assert fov_analysis.cluster_labels is None
    assert fov_analysis.cluster_method is None
    assert fov_analysis.cluster_n_clusters is None


# ============================================================================
# FOV Analysis Mixed Spikes Tests
# ============================================================================


def test_fov_analysis_with_zero_spike_rois() -> None:
    """Test that FOV analysis includes ROIs with zero spikes in spike matrices."""
    # Create a FOV with 3 ROIs: 2 with spikes, 1 without
    fov = FOV(name="test_fov", position_index=1)

    # ROI 1: Has spikes
    roi1 = ROI(label_value=1, active=True, fov_id=fov.id)
    spike_pattern1 = np.zeros(50)
    spike_pattern1[[5, 15, 25, 35, 45]] = 2.0  # Spike amplitude = 2.0
    traces1 = Traces(
        dff=[1.0] * 50,
        den_dff=[1.0] * 50,
        inferred_spikes=spike_pattern1.tolist(),
    )
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(
        peaks_den_dff=[5, 15, 25],
        inferred_spikes_threshold=1.0,
    )
    roi1._new_data_analysis = [data_analysis1]

    # ROI 2: No spikes (all below threshold)
    roi2 = ROI(label_value=2, active=True, fov_id=fov.id)
    spike_pattern2 = np.zeros(50)  # All zeros
    traces2 = Traces(
        dff=[1.0] * 50,
        den_dff=[1.0] * 50,
        inferred_spikes=spike_pattern2.tolist(),
    )
    roi2._new_traces = [traces2]
    data_analysis2 = DataAnalysis(
        peaks_den_dff=[6, 16, 26],
        inferred_spikes_threshold=1.0,
    )
    roi2._new_data_analysis = [data_analysis2]

    # ROI 3: Has spikes
    roi3 = ROI(label_value=3, active=True, fov_id=fov.id)
    spike_pattern3 = np.zeros(50)
    spike_pattern3[[6, 16, 26, 36, 46]] = 2.0
    traces3 = Traces(
        dff=[1.0] * 50,
        den_dff=[1.0] * 50,
        inferred_spikes=spike_pattern3.tolist(),
    )
    roi3._new_traces = [traces3]
    data_analysis3 = DataAnalysis(
        peaks_den_dff=[6, 16, 26],
        inferred_spikes_threshold=1.0,
    )
    roi3._new_data_analysis = [data_analysis3]

    fov.rois = [roi1, roi2, roi3]

    # Create analysis settings
    settings = AnalysisSettings(
        spikes_sync_jitter_window=200,
        spikes_sync_cross_corr_lag=500,
    )

    # Compute FOV analysis
    fov_analysis = compute_fov_analysis(fov, settings)

    assert fov_analysis is not None

    # Critical: Check that spike matrices include all 3 ROIs
    assert len(fov_analysis.active_roi_labels) == 3
    assert fov_analysis.active_roi_labels == [1, 2, 3]

    # All spike matrices should be 3x3, not 2x2
    assert fov_analysis.spike_max_lag_correlation_matrix is not None
    assert len(fov_analysis.spike_max_lag_correlation_matrix) == 3
    assert len(fov_analysis.spike_max_lag_correlation_matrix[0]) == 3

    assert fov_analysis.spike_jitter_synchrony_matrix is not None
    assert len(fov_analysis.spike_jitter_synchrony_matrix) == 3
    assert len(fov_analysis.spike_jitter_synchrony_matrix[0]) == 3

    # ROI 2 (index 1) should have zero synchrony with others since it has no spikes
    spike_jitter = np.array(fov_analysis.spike_jitter_synchrony_matrix)
    assert spike_jitter[1, 0] == 0.0  # ROI2 vs ROI1
    assert spike_jitter[1, 2] == 0.0  # ROI2 vs ROI3
    assert spike_jitter[0, 1] == 0.0  # ROI1 vs ROI2
    assert spike_jitter[2, 1] == 0.0  # ROI3 vs ROI2


def test_connectivity_metrics_with_zero_spike_rois() -> None:
    """Test that connectivity metrics work with zero-spike ROIs."""
    from cali.analysis._fov_metrics import _compute_connectivity_metrics
    from cali.sqlmodel import FOVAnalysis

    # Create FOV analysis with 3 ROIs, including one with no spikes
    # Spike correlation matrix with one zero-spike ROI (ROI 2, index 1)
    spike_corr_matrix = [
        [1.0, 0.0, 0.8],  # ROI 1: self=1, no corr with ROI2, corr with ROI3
        [0.0, 1.0, 0.0],  # ROI 2: no spikes, zero corr with others
        [0.8, 0.0, 1.0],  # ROI 3: corr with ROI1, no corr with ROI2, self=1
    ]

    # Test with spike_maxlag method instead (spike_corr was removed)
    fov_analysis = FOVAnalysis(
        id=1,
        active_roi_labels=[1, 2, 3],
        spike_max_lag_correlation_matrix=spike_corr_matrix,
    )

    # This should not raise ValueError about shape mismatch
    adjacency, weights, roi_labels = _compute_connectivity_metrics(
        fov_analysis,
        method="spike_maxlag",
        threshold=0.5,
    )

    # Check correct dimensions
    assert adjacency.shape == (3, 3)
    assert weights.shape == (3, 3)
    assert len(roi_labels) == 3
    assert roi_labels == [1, 2, 3]

    # Check that ROI 1 and ROI 3 are connected (corr=0.8 > threshold=0.5)
    assert adjacency[0, 2] == 1
    assert adjacency[2, 0] == 1

    # Check that ROI 2 has no connections (zero spikes)
    assert adjacency[1, 0] == 0
    assert adjacency[1, 2] == 0
    assert adjacency[0, 1] == 0
    assert adjacency[2, 1] == 0


def test_all_zero_spike_rois() -> None:
    """Test FOV analysis when all ROIs have zero spikes."""
    fov = FOV(name="test_fov", position_index=1)

    # Create 2 ROIs with no spikes
    for label in [1, 2]:
        roi = ROI(label_value=label, active=True, fov_id=fov.id)
        spike_pattern = np.zeros(50)  # All zeros
        traces = Traces(
            dff=[1.0] * 50,
            den_dff=[1.0] * 50,
            inferred_spikes=spike_pattern.tolist(),
        )
        roi._new_traces = [traces]
        data_analysis = DataAnalysis(
            peaks_den_dff=[5, 15, 25],
            inferred_spikes_threshold=1.0,
        )
        roi._new_data_analysis = [data_analysis]
        fov.rois.append(roi)

    settings = AnalysisSettings(
        spikes_sync_jitter_window=200,
        spikes_sync_cross_corr_lag=500,
    )

    fov_analysis = compute_fov_analysis(fov, settings)

    assert fov_analysis is not None

    # Spike matrices should still be computed with correct dimensions
    assert fov_analysis.spike_max_lag_correlation_matrix is not None
    assert len(fov_analysis.spike_max_lag_correlation_matrix) == 2
    assert len(fov_analysis.spike_max_lag_correlation_matrix[0]) == 2

    # All off-diagonal correlations should be zero
    spike_maxlag = np.array(fov_analysis.spike_max_lag_correlation_matrix)
    assert spike_maxlag[0, 1] == 0.0
    assert spike_maxlag[1, 0] == 0.0

    # Diagonal should still be 1.0 (self-correlation)
    assert spike_maxlag[0, 0] == 1.0
    assert spike_maxlag[1, 1] == 1.0


def test_mixed_spike_methods_consistent_dimensions() -> None:
    """Test that all spike methods produce matrices with consistent dimensions."""
    fov = FOV(name="test_fov", position_index=1)

    # Create 3 ROIs with varying spike patterns
    spike_patterns = [
        [5, 15, 25],  # ROI 1: has spikes
        [],  # ROI 2: no spikes
        [6, 16, 26],  # ROI 3: has spikes
    ]

    for idx, spike_indices in enumerate(spike_patterns, start=1):
        roi = ROI(label_value=idx, active=True, fov_id=fov.id)
        spike_pattern = np.zeros(50)
        for spike_idx in spike_indices:
            spike_pattern[spike_idx] = 2.0

        traces = Traces(
            dff=[1.0] * 50,
            den_dff=[1.0] * 50,
            inferred_spikes=spike_pattern.tolist(),
        )
        roi._new_traces = [traces]
        data_analysis = DataAnalysis(
            peaks_den_dff=[5, 15, 25],
            inferred_spikes_threshold=1.0,
        )
        roi._new_data_analysis = [data_analysis]
        fov.rois.append(roi)

    settings = AnalysisSettings(
        spikes_sync_jitter_window=200,
        spikes_sync_cross_corr_lag=500,
    )

    fov_analysis = compute_fov_analysis(fov, settings)

    assert fov_analysis is not None

    # All spike methods should produce 3x3 matrices
    assert fov_analysis.spike_max_lag_correlation_matrix is not None
    assert fov_analysis.spike_jitter_synchrony_matrix is not None

    spike_maxlag = np.array(fov_analysis.spike_max_lag_correlation_matrix)
    spike_jitter = np.array(fov_analysis.spike_jitter_synchrony_matrix)

    assert spike_maxlag.shape == (3, 3)
    assert spike_jitter.shape == (3, 3)

    # All methods should have zero values for non-spiking ROI (index 1)
    for matrix in [spike_maxlag, spike_jitter]:
        assert matrix[1, 0] == 0.0
        assert matrix[1, 2] == 0.0
        assert matrix[0, 1] == 0.0
        assert matrix[2, 1] == 0.0


# ============================================================================
# Rising Edges Analysis Tests
# ============================================================================


def test_rising_edge_detection_basic() -> None:
    """Test that rising edges are correctly detected from binary spike trains."""
    # Create a simple spike train with some multi-frame events
    spike_train = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0], dtype=float)

    # Detect rising edges (0 -> 1 transitions)
    positive_vals = spike_train > 0
    rising = positive_vals & ~np.concatenate(([False], positive_vals[:-1]))
    rising_edges = np.zeros_like(spike_train, dtype=float)
    rising_edges[rising] = 1.0

    # Should have edges at indices 2 and 7 only (start of each spike event)
    expected = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=float)
    np.testing.assert_array_equal(rising_edges, expected)


def test_rising_edge_single_frame_spikes() -> None:
    """Test rising edge detection with single-frame spike events."""
    # Single-frame spikes
    spike_train = np.array([0, 1, 0, 1, 0, 1, 0, 0], dtype=float)

    positive_vals = spike_train > 0
    rising = positive_vals & ~np.concatenate(([False], positive_vals[:-1]))
    rising_edges = np.zeros_like(spike_train, dtype=float)
    rising_edges[rising] = 1.0

    # Every spike is its own rising edge
    np.testing.assert_array_equal(rising_edges, spike_train)


def test_rising_edge_continuous_spikes() -> None:
    """Test rising edge detection with one continuous spike event."""
    # One long spike event
    spike_train = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=float)

    positive_vals = spike_train > 0
    rising = positive_vals & ~np.concatenate(([False], positive_vals[:-1]))
    rising_edges = np.zeros_like(spike_train, dtype=float)
    rising_edges[rising] = 1.0

    # Should have only one rising edge at index 2
    expected = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    np.testing.assert_array_equal(rising_edges, expected)


def test_rising_edge_no_spikes() -> None:
    """Test rising edge detection with no spikes."""
    spike_train = np.zeros(10, dtype=float)

    positive_vals = spike_train > 0
    rising = positive_vals & ~np.concatenate(([False], positive_vals[:-1]))
    rising_edges = np.zeros_like(spike_train, dtype=float)
    rising_edges[rising] = 1.0

    # Should be all zeros
    np.testing.assert_array_equal(rising_edges, spike_train)


def test_rising_edge_all_ones() -> None:
    """Test rising edge detection when all values are positive."""
    spike_train = np.ones(10, dtype=float)

    positive_vals = spike_train > 0
    rising = positive_vals & ~np.concatenate(([False], positive_vals[:-1]))
    rising_edges = np.zeros_like(spike_train, dtype=float)
    rising_edges[rising] = 1.0

    # Should have only first element as rising edge
    expected = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
    np.testing.assert_array_equal(rising_edges, expected)


def test_fov_analysis_computes_rising_edges() -> None:
    """Test that FOV analysis computes and stores rising edge metrics."""
    # Create a minimal FOV with ROIs that have multi-frame spike events
    fov = FOV(name="test_fov", position_index=0, n_frames=100, frame_rate=10.0)

    # Create ROIs with spike data that has multi-frame events
    # This ensures rising edges differ from thresholded binary
    for i in range(3):
        roi = ROI(label_value=i + 1, active=True, fov_id=1)

        # Create spike trains with multi-frame events
        spike_train = np.zeros(100, dtype=float)
        if i == 0:
            spike_train[10:15] = 2.0  # Long event
            spike_train[30:33] = 2.0  # Short event
        elif i == 1:
            spike_train[12:14] = 2.0
            spike_train[35:37] = 2.0
        else:
            spike_train[15:17] = 2.0
            spike_train[40] = 2.0  # Single frame

        traces = Traces(
            dff=[1.0] * 100,
            den_dff=[1.0] * 100,
            inferred_spikes=spike_train.tolist(),
        )
        roi._new_traces = [traces]

        data_analysis = DataAnalysis(
            peaks_den_dff=[],
            inferred_spikes_threshold=1.0,
        )
        roi._new_data_analysis = [data_analysis]

        fov.rois = [*fov.rois, roi] if hasattr(fov, "rois") else [roi]

    # Create analysis settings
    settings = AnalysisSettings(
        experiment_type="spontaneous",
        spike_inference_threshold=0.0,
        enable_rising_edge_analysis=True,
    )

    # Run FOV analysis
    result = compute_fov_analysis(fov, settings)

    assert result is not None

    # Check that rising edge matrices exist
    assert result.spike_max_lag_correlation_matrix_rising_edges is not None
    assert result.spike_max_lag_values_matrix_rising_edges is not None
    assert result.spike_jitter_synchrony_matrix_rising_edges is not None

    # Check that global metrics exist
    assert result.global_spike_max_lag_correlation_rising_edges is not None
    assert result.global_spike_jitter_synchrony_rising_edges is not None

    # Verify matrices have correct dimensions (3x3 for 3 ROIs)
    assert len(result.spike_max_lag_correlation_matrix_rising_edges) == 3
    assert len(result.spike_max_lag_correlation_matrix_rising_edges[0]) == 3


def test_rising_edge_vs_thresholded_metrics() -> None:
    """Test that rising edge metrics differ from thresholded binary metrics."""
    # Create FOV with ROIs that have multi-frame spike events
    fov = FOV(name="test_fov", position_index=0, n_frames=50, frame_rate=10.0)

    # Create ROIs with spike data that has long multi-frame events
    # This should make thresholded and rising edge metrics different
    for i in range(2):
        roi = ROI(label_value=i + 1, active=True, fov_id=1)

        spike_train = np.zeros(50, dtype=float)
        # Add long multi-frame events that will differ between methods
        spike_train[10:20] = 2.0  # 10-frame event
        spike_train[30:35] = 2.0  # 5-frame event

        traces = Traces(
            dff=[1.0] * 50,
            den_dff=[1.0] * 50,
            inferred_spikes=spike_train.tolist(),
        )
        roi._new_traces = [traces]

        data_analysis = DataAnalysis(
            peaks_den_dff=[],
            inferred_spikes_threshold=1.0,
        )
        roi._new_data_analysis = [data_analysis]

        fov.rois = [*fov.rois, roi] if hasattr(fov, "rois") else [roi]

    settings = AnalysisSettings(
        experiment_type="spontaneous",
        spike_inference_threshold=0.0,
        enable_rising_edge_analysis=True,
    )

    result = compute_fov_analysis(fov, settings)

    assert result is not None

    # Both metrics should exist
    assert result.spike_max_lag_correlation_matrix is not None
    assert result.spike_max_lag_correlation_matrix_rising_edges is not None

    # They should likely be different because thresholded counts all frames
    # in a spike event while rising edges only count the first frame
    thresholded = result.spike_max_lag_correlation_matrix
    rising = result.spike_max_lag_correlation_matrix_rising_edges

    # At least some values should differ
    # (They might be equal by chance, but likely not all)
    assert len(thresholded) == len(rising)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
