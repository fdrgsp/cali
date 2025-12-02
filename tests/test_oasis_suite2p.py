import numpy as np

from cali.extraction._oasis_suite2p import (
    oasis,
    oasis_trace,
    preprocess,
)


def test_oasis_trace() -> None:
    """Test oasis_trace low-level function."""
    # Create a simple trace with a spike
    T = 100
    tau = 1.0
    fs = 30.0

    # Create synthetic calcium trace
    # Spike at t=20
    true_spikes = np.zeros(T, dtype=np.float32)
    true_spikes[20] = 1.0

    # Convolve with exponential decay
    decay = np.exp(-np.arange(T) / (tau * fs))
    calcium = np.convolve(true_spikes, decay)[:T].astype(np.float32)

    # Prepare output arrays
    v = np.zeros(T, dtype=np.float32)
    w = np.zeros(T, dtype=np.float32)
    t = np.zeros(T, dtype=np.int64)
    l_arr = np.zeros(T, dtype=np.float32)
    s = np.zeros(T, dtype=np.float32)

    oasis_trace(calcium, v, w, t, l_arr, s, tau, fs)

    # Check if spike is detected around t=20
    # s contains the deconvolved spikes
    assert np.sum(s) > 0
    assert s[20] > 0 or s[19] > 0 or s[21] > 0


def test_preprocess() -> None:
    """Test preprocess."""
    trace = np.random.randn(10, 100).astype(np.float32) + 100

    # Test constant_prctile
    processed = preprocess(
        trace,
        baseline="constant_prctile",
        win_baseline=10.0,
        sig_baseline=10.0,
        fs=30.0,
    )
    assert processed.shape == trace.shape

    # Test maximin
    processed = preprocess(
        trace,
        baseline="maximin",
        win_baseline=10.0,
        sig_baseline=10.0,
        fs=30.0,
    )
    assert processed.shape == trace.shape

    # Test constant
    processed = preprocess(
        trace,
        baseline="constant",
        win_baseline=10.0,
        sig_baseline=10.0,
        fs=30.0,
    )
    assert processed.shape == trace.shape

    # Test unknown baseline (should be 0 subtraction)
    processed = preprocess(
        trace,
        baseline="unknown",
        win_baseline=10.0,
        sig_baseline=10.0,
        fs=30.0,
    )
    assert np.allclose(processed, trace)


def test_oasis() -> None:
    """Test high-level oasis."""
    T = 200
    trace = np.zeros((1, T), dtype=np.float32)
    trace[0, 50:100] = 1.0  # Boxcar

    s = oasis(trace, batch_size=200, tau=1.0, fs=30.0)

    assert s.shape == trace.shape
