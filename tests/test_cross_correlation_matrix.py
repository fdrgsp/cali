"""Test the cross-correlation matrix computation.

Tests verify that:
1. Zero-lag Pearson correlation is computed correctly
2. Constant traces are handled (return 0 correlation)
3. Results match numpy's corrcoef
4. Different length traces raise ValueError
"""

import numpy as np
import pytest

from cali.analysis._fov_analysis import _compute_cross_correlation_matrix


def test_perfect_correlation() -> None:
    """Test that identical traces have correlation = 1.0."""
    trace = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    traces = [trace, trace.copy()]

    matrix = _compute_cross_correlation_matrix(traces)

    assert matrix is not None
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0
    assert np.isclose(matrix[0, 1], 1.0)
    assert np.isclose(matrix[1, 0], 1.0)


def test_perfect_anticorrelation() -> None:
    """Test that perfectly anti-correlated traces have correlation = -1.0."""
    trace1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    trace2 = -trace1
    traces = [trace1, trace2]

    matrix = _compute_cross_correlation_matrix(traces)

    assert matrix is not None
    assert np.isclose(matrix[0, 1], -1.0, atol=1e-10)


def test_zero_correlation() -> None:
    """Test uncorrelated traces have correlation near 0."""
    np.random.seed(42)
    trace1 = np.random.randn(1000)
    trace2 = np.random.randn(1000)
    traces = [trace1, trace2]

    matrix = _compute_cross_correlation_matrix(traces)

    assert matrix is not None
    # With enough samples, random traces should be uncorrelated
    assert abs(matrix[0, 1]) < 0.1


def test_constant_trace() -> None:
    """Test that constant traces are handled correctly (correlation = 0)."""
    trace1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    trace2 = np.array([3.0, 3.0, 3.0, 3.0, 3.0])  # Constant
    traces = [trace1, trace2]

    matrix = _compute_cross_correlation_matrix(traces)

    assert matrix is not None
    # Correlation with constant trace should be 0
    assert matrix[0, 1] == 0.0
    assert matrix[1, 0] == 0.0
    # Self-correlation should still be 1
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0


def test_matches_numpy_corrcoef() -> None:
    """Test that results match numpy's corrcoef."""
    np.random.seed(42)
    traces = [
        np.random.randn(100),
        np.random.randn(100),
        np.random.randn(100),
    ]

    our_matrix = _compute_cross_correlation_matrix(traces)
    numpy_matrix = np.corrcoef(traces)

    assert our_matrix is not None
    np.testing.assert_allclose(our_matrix, numpy_matrix, rtol=1e-10, atol=1e-10)


def test_different_length_traces_raises() -> None:
    """Test that different length traces raise ValueError."""
    trace1 = np.array([1.0, 2.0, 3.0])
    trace2 = np.array([1.0, 2.0, 3.0, 4.0])
    traces = [trace1, trace2]

    with pytest.raises(ValueError, match="All traces must have same length"):
        _compute_cross_correlation_matrix(traces)


def test_insufficient_traces() -> None:
    """Test that <2 traces returns None."""
    traces = [np.array([1.0, 2.0, 3.0])]

    matrix = _compute_cross_correlation_matrix(traces)

    assert matrix is None


def test_symmetric_matrix() -> None:
    """Test that correlation matrix is symmetric."""
    np.random.seed(42)
    n_rois = 5
    traces = [np.random.randn(100) for _ in range(n_rois)]

    matrix = _compute_cross_correlation_matrix(traces)

    assert matrix is not None
    # Check symmetry
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-10, atol=1e-10)


def test_correlation_bounds() -> None:
    """Test that all correlations are in [-1, 1]."""
    np.random.seed(42)
    traces = [np.random.randn(100) for _ in range(10)]

    matrix = _compute_cross_correlation_matrix(traces)

    assert matrix is not None
    assert np.all(matrix >= -1.0)
    assert np.all(matrix <= 1.0)


def test_multiple_constant_traces() -> None:
    """Test multiple constant traces."""
    trace1 = np.array([5.0, 5.0, 5.0, 5.0])
    trace2 = np.array([3.0, 3.0, 3.0, 3.0])
    trace3 = np.array([1.0, 2.0, 3.0, 4.0])
    traces = [trace1, trace2, trace3]

    matrix = _compute_cross_correlation_matrix(traces)

    assert matrix is not None
    # Constant traces should have 0 correlation with everything (except self)
    assert matrix[0, 1] == 0.0
    assert matrix[0, 2] == 0.0
    assert matrix[1, 2] == 0.0
    # Diagonal should still be 1
    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 1.0
    assert matrix[2, 2] == 1.0


def test_phase_shifted_traces() -> None:
    """Test that phase-shifted sinusoids have low correlation.

    This demonstrates the difference between zero-lag correlation and
    max cross-correlation: phase-shifted signals have low zero-lag corr
    but would have high max cross-corr.
    """
    t = np.linspace(0, 4 * np.pi, 100)
    trace1 = np.sin(t)
    trace2 = np.sin(t + np.pi / 2)  # 90 degree phase shift
    traces = [trace1, trace2]

    matrix = _compute_cross_correlation_matrix(traces)

    assert matrix is not None
    # Orthogonal sinusoids should have ~0 zero-lag correlation
    assert abs(matrix[0, 1]) < 0.1
    # But if we used max cross-correlation, they would have ~1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
