"""Test that numba parallel functions work correctly with multithreading.

This test verifies that the threading lock properly protects numba parallel=True
functions from concurrent access when called from multiple Python threads.

Background:
- Numba's workqueue threading layer is not threadsafe
- When multiple Python threads call numba parallel=True functions simultaneously,
  it causes: "Concurrent access has been detected" error
- Solution: Use a global threading.Lock() to serialize access to numba functions

See: https://numba.readthedocs.io/en/stable/user/threading-layer.html
"""

import threading

import numpy as np
import pytest

from cali.analysis._util import (
    _get_calcium_peaks_event_correlations_matrix,
    _get_spike_correlations_matrix,
)


def test_numba_synchrony_with_threading() -> None:
    """Test that synchrony computations work correctly when called from threads."""
    # Create test data
    n_rois = 10
    n_timepoints = 1000

    # Create random peak events for calcium
    peak_events_dict = {}
    for i in range(n_rois):
        events = np.random.rand(n_timepoints)
        events[events < 0.95] = 0.0  # Sparse events
        events[events >= 0.95] = 1.0
        peak_events_dict[str(i)] = events.tolist()

    # Create random spike data (binary)
    spike_data_dict = {}
    for i in range(n_rois):
        spikes = np.random.rand(n_timepoints)
        spikes[spikes < 0.98] = 0.0  # Very sparse spikes
        spikes[spikes >= 0.98] = 1.0  # Binary values
        spike_data_dict[str(i)] = spikes.tolist()

    # Function to run in thread - calls numba parallel function
    def compute_calcium_sync() -> None:
        matrix, _ = _get_calcium_peaks_event_correlations_matrix(
            peak_events_dict, method="jitter_window", jitter_window=5
        )
        assert matrix is not None
        assert matrix.shape == (n_rois, n_rois)

    def compute_spike_sync() -> None:
        matrix, _ = _get_spike_correlations_matrix(
            spike_data_dict, method="jitter_window", jitter_window=5
        )
        assert matrix is not None
        assert matrix.shape == (n_rois, n_rois)

    # Run computations in parallel threads (simulates extraction pipeline)
    threads = []
    n_threads = 4

    # Mix of calcium and spike synchrony computations
    for i in range(n_threads):
        if i % 2 == 0:
            t = threading.Thread(target=compute_calcium_sync)
        else:
            t = threading.Thread(target=compute_spike_sync)
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # If we get here without crashes, the lock is working correctly


def test_numba_synchrony_results_consistent() -> None:
    """Test that threading lock doesn't affect computation correctness."""
    # Create deterministic test data
    np.random.seed(42)
    n_rois = 5
    n_timepoints = 100

    peak_events_dict = {}
    for i in range(n_rois):
        events = np.zeros(n_timepoints)
        # Create specific peak pattern
        events[i * 10 : i * 10 + 5] = 1.0
        peak_events_dict[str(i)] = events.tolist()

    # Compute matrix multiple times
    results = []
    for _ in range(5):
        matrix, _ = _get_calcium_peaks_event_correlations_matrix(
            peak_events_dict, method="jitter_window", jitter_window=3
        )
        results.append(matrix)

    # All results should be identical (deterministic computation)
    for i in range(1, len(results)):
        np.testing.assert_array_equal(results[0], results[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
