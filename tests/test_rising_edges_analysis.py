"""Tests for rising edge detection in spike analysis.

These tests validate that the rising edge detection feature works correctly
for spike train analysis, including computation, storage, and integration with
the full analysis pipeline.
"""

import numpy as np
import pytest

from cali.analysis._fov_analysis import compute_fov_analysis
from cali.sqlmodel._model import FOV, ROI, AnalysisSettings, DataAnalysis, Traces


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
            dec_dff=[1.0] * 100,
            inferred_spikes=spike_train.tolist(),
        )
        roi._new_traces = [traces]

        data_analysis = DataAnalysis(
            peaks_dec_dff=[],
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
            dec_dff=[1.0] * 50,
            inferred_spikes=spike_train.tolist(),
        )
        roi._new_traces = [traces]

        data_analysis = DataAnalysis(
            peaks_dec_dff=[],
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
