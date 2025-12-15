"""Test FOV analysis with mixed spiking and non-spiking ROIs.

This tests the fix for the dimension mismatch issue where spike matrices
must include all active ROIs, even those with zero spikes.
"""

from __future__ import annotations

import numpy as np

from cali.analysis._fov_analysis import compute_fov_analysis
from cali.sqlmodel import FOV, ROI, AnalysisSettings, DataAnalysis, Traces


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
        dec_dff=[1.0] * 50,
        inferred_spikes=spike_pattern1.tolist(),
    )
    roi1._new_traces = [traces1]
    data_analysis1 = DataAnalysis(
        peaks_dec_dff=[5, 15, 25],
        inferred_spikes_threshold=1.0,
    )
    roi1._new_data_analysis = [data_analysis1]

    # ROI 2: No spikes (all below threshold)
    roi2 = ROI(label_value=2, active=True, fov_id=fov.id)
    spike_pattern2 = np.zeros(50)  # All zeros
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

    # ROI 3: Has spikes
    roi3 = ROI(label_value=3, active=True, fov_id=fov.id)
    spike_pattern3 = np.zeros(50)
    spike_pattern3[[6, 16, 26, 36, 46]] = 2.0
    traces3 = Traces(
        dff=[1.0] * 50,
        dec_dff=[1.0] * 50,
        inferred_spikes=spike_pattern3.tolist(),
    )
    roi3._new_traces = [traces3]
    data_analysis3 = DataAnalysis(
        peaks_dec_dff=[6, 16, 26],
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
    from cali.analysis._util import _compute_connectivity_metrics
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
            dec_dff=[1.0] * 50,
            inferred_spikes=spike_pattern.tolist(),
        )
        roi._new_traces = [traces]
        data_analysis = DataAnalysis(
            peaks_dec_dff=[5, 15, 25],
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
            dec_dff=[1.0] * 50,
            inferred_spikes=spike_pattern.tolist(),
        )
        roi._new_traces = [traces]
        data_analysis = DataAnalysis(
            peaks_dec_dff=[5, 15, 25],
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
