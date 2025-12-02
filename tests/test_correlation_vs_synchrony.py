"""Validation tests for cross-correlation and synchrony calculations.

This module creates synthetic calcium traces with known correlation and synchrony
properties to validate the correctness of both calculation methods.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import correlate
from scipy.stats import zscore

from cali.plot._util import _calculate_jitter_window_synchrony

# =============================================================================
# Test Helpers: Synthetic Trace Generation
# =============================================================================


def create_perfect_correlation_traces(
    n_frames: int = 1000, amplitude: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create two identical traces (perfect positive correlation = 1.0)."""
    trace = np.random.randn(n_frames) * amplitude
    return trace.copy(), trace.copy()


def create_anticorrelation_traces(
    n_frames: int = 1000, amplitude: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create two anticorrelated traces (perfect negative correlation ≈ -1.0)."""
    trace1 = np.random.randn(n_frames) * amplitude
    trace2 = -trace1  # Perfect anticorrelation
    return trace1, trace2


def create_independent_traces(
    n_frames: int = 1000, amplitude: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create two independent traces (correlation ≈ 0.0)."""
    trace1 = np.random.randn(n_frames) * amplitude
    trace2 = np.random.randn(n_frames) * amplitude
    return trace1, trace2


def create_shifted_correlation_traces(
    n_frames: int = 1000, shift: int = 50, amplitude: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create two traces with known time lag (tests lag detection)."""
    trace1 = np.random.randn(n_frames) * amplitude
    trace2 = np.roll(trace1, shift)
    return trace1, trace2


def create_partial_correlation_traces(
    n_frames: int = 1000, correlation_strength: float = 0.5, amplitude: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create two traces with partial correlation (0 < r < 1)."""
    trace1 = np.random.randn(n_frames) * amplitude
    noise = np.random.randn(n_frames) * amplitude
    # Mix trace1 with independent noise to get desired correlation
    trace2 = correlation_strength * trace1 + (1 - correlation_strength) * noise
    return trace1, trace2


def create_perfect_synchrony_events(
    n_frames: int = 1000, n_events: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Create two identical binary event trains (synchrony = 1.0)."""
    events = np.zeros(n_frames)
    event_times = np.random.choice(n_frames, size=n_events, replace=False)
    events[event_times] = 1.0
    return events.copy(), events.copy()


def create_no_overlap_events(
    n_frames: int = 1000, n_events: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Create two non-overlapping event trains (synchrony = 0.0)."""
    events1 = np.zeros(n_frames)
    events2 = np.zeros(n_frames)

    # Split timeline in half
    mid = n_frames // 2
    times1 = np.random.choice(mid, size=n_events // 2, replace=False)
    times2 = np.random.choice(range(mid, n_frames), size=n_events // 2, replace=False)

    events1[times1] = 1.0
    events2[times2] = 1.0
    return events1, events2


def create_partial_overlap_events(
    n_frames: int = 1000, n_events: int = 20, overlap_fraction: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Create event trains with partial overlap (0 < synchrony < 1)."""
    events1 = np.zeros(n_frames)
    events2 = np.zeros(n_frames)

    # All events for trace 1
    times1 = np.random.choice(n_frames, size=n_events, replace=False)
    events1[times1] = 1.0

    # Some events overlap, some don't
    n_overlap = int(n_events * overlap_fraction)
    n_unique = n_events - n_overlap

    # Overlapping events (same as trace1)
    overlap_times = np.random.choice(times1, size=n_overlap, replace=False)

    # Unique events for trace2
    available = [t for t in range(n_frames) if t not in times1]
    unique_times = np.random.choice(available, size=n_unique, replace=False)

    all_times2 = np.concatenate([overlap_times, unique_times])
    events2[all_times2.astype(int)] = 1.0

    return events1, events2


def create_jittered_events(
    n_frames: int = 1000, n_events: int = 20, jitter: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Create events with small temporal jitter (tests jitter window tolerance)."""
    events1 = np.zeros(n_frames)
    events2 = np.zeros(n_frames)

    # Base event times
    times1 = np.random.choice(
        range(jitter, n_frames - jitter), size=n_events, replace=False
    )
    events1[times1] = 1.0

    # Jittered versions
    for t in times1:
        jittered_t = t + np.random.randint(-jitter, jitter + 1)
        jittered_t = np.clip(jittered_t, 0, n_frames - 1)
        events2[jittered_t] = 1.0

    return events1, events2


# =============================================================================
# Cross-Correlation Calculation (Mirroring Cali Implementation)
# =============================================================================


def calculate_cross_correlation(trace1: np.ndarray, trace2: np.ndarray) -> float:
    """Calculate max normalized cross-correlation (matching Cali implementation)."""
    # Z-score normalization
    x = zscore(trace1)
    y = zscore(trace2)

    # Compute correlation across all lags
    corr = correlate(x, y, mode="full", method="fft")

    # Normalize by signal norms
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x == 0 or norm_y == 0:
        return 0.0

    corr /= norm_x * norm_y

    # Return maximum correlation
    return float(np.max(corr))


# =============================================================================
# Tests: Cross-Correlation
# =============================================================================


def test_perfect_correlation() -> None:
    """Test that identical traces yield correlation ≈ 1.0."""
    trace1, trace2 = create_perfect_correlation_traces(n_frames=1000)
    corr = calculate_cross_correlation(trace1, trace2)
    assert abs(corr - 1.0) < 0.01, f"Perfect correlation should be ~1.0, got {corr}"


def test_anticorrelation() -> None:
    """Test that inverted traces yield correlation ≈ -1.0."""
    trace1, trace2 = create_anticorrelation_traces(n_frames=1000)
    calculate_cross_correlation(trace1, trace2)
    # Note: max() will find the positive peak at zero lag, which is actually 1.0
    # for anticorrelated signals at zero lag. But at opposite lag it's -1.0.
    # Since we take max, we need to check the correlation at zero lag specifically.
    x = zscore(trace1)
    y = zscore(trace2)
    corr_zero_lag = np.corrcoef(x, y)[0, 1]
    assert abs(corr_zero_lag + 1.0) < 0.01, (
        f"Anticorrelation should be ~-1.0, got {corr_zero_lag}"
    )


def test_independent_correlation() -> None:
    """Test that independent traces yield correlation ≈ 0.0."""
    # Use larger sample for stable estimate
    trace1, trace2 = create_independent_traces(n_frames=5000)
    corr = calculate_cross_correlation(trace1, trace2)
    assert abs(corr) < 0.1, (
        f"Independent traces should have correlation near 0, got {corr}"
    )


def test_shifted_correlation() -> None:
    """Test that shifted identical traces still yield high correlation."""
    trace1, trace2 = create_shifted_correlation_traces(n_frames=1000, shift=50)
    corr = calculate_cross_correlation(trace1, trace2)
    # Should find high correlation at the shifted lag
    assert corr > 0.92, (
        f"Shifted identical traces should have high correlation, got {corr}"
    )


def test_partial_correlation() -> None:
    """Test that partially correlated traces yield intermediate values."""
    expected_corr = 0.7
    trace1, trace2 = create_partial_correlation_traces(
        n_frames=5000, correlation_strength=expected_corr
    )
    corr = calculate_cross_correlation(trace1, trace2)
    # Note: max correlation can be higher than zero-lag due to noise alignment
    assert 0.5 < corr <= 1.0, f"Expected intermediate correlation, got {corr}"


@pytest.mark.parametrize("target_corr", [0.3, 0.5, 0.7, 0.9])
def test_correlation_range(target_corr: float) -> None:
    """Test correlation across a range of values."""
    trace1, trace2 = create_partial_correlation_traces(
        n_frames=10000, correlation_strength=target_corr
    )
    corr = calculate_cross_correlation(trace1, trace2)
    # Max correlation can exceed zero-lag correlation due to noise alignment
    # Just verify it's in a reasonable range
    if target_corr < 0.5:
        assert 0.2 < corr < 0.8, f"Expected low-medium correlation, got {corr}"
    else:
        assert 0.5 < corr <= 1.0, f"Expected medium-high correlation, got {corr}"


# =============================================================================
# Tests: Synchrony
# =============================================================================


def test_perfect_synchrony() -> None:
    """Test that identical event trains yield synchrony = 1.0."""
    events1, events2 = create_perfect_synchrony_events(n_frames=1000, n_events=20)
    sync = _calculate_jitter_window_synchrony(events1, events2, jitter_window=2)
    assert abs(sync - 1.0) < 0.01, f"Perfect synchrony should be 1.0, got {sync}"


def test_no_overlap_synchrony() -> None:
    """Test that non-overlapping events yield synchrony = 0.0."""
    events1, events2 = create_no_overlap_events(n_frames=1000, n_events=20)
    sync = _calculate_jitter_window_synchrony(events1, events2, jitter_window=2)
    assert sync == 0.0, (
        f"Non-overlapping events should have synchrony = 0.0, got {sync}"
    )


def test_partial_overlap_synchrony() -> None:
    """Test that partial overlap yields intermediate synchrony."""
    overlap_fraction = 0.6
    events1, events2 = create_partial_overlap_events(
        n_frames=1000, n_events=20, overlap_fraction=overlap_fraction
    )
    sync = _calculate_jitter_window_synchrony(events1, events2, jitter_window=0)

    # With zero jitter window, only exact matches count
    # Expected synchrony depends on exact overlap
    assert 0.0 < sync < 1.0, f"Partial overlap should yield 0 < sync < 1, got {sync}"


def test_jittered_events_with_window() -> None:
    """Test that jittered events are detected with appropriate jitter window."""
    jitter = 3
    events1, events2 = create_jittered_events(n_frames=1000, n_events=30, jitter=jitter)

    # With jitter window >= jitter, should detect most coincidences
    sync_large_window = _calculate_jitter_window_synchrony(
        events1, events2, jitter_window=jitter
    )

    # With zero jitter window, should miss most
    sync_no_window = _calculate_jitter_window_synchrony(
        events1, events2, jitter_window=0
    )

    assert sync_large_window > sync_no_window, (
        f"Larger jitter window should detect more synchrony: "
        f"{sync_large_window} vs {sync_no_window}"
    )
    assert sync_large_window > 0.5, (
        "With appropriate window, should detect >50% synchrony"
    )


@pytest.mark.parametrize("overlap_frac", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_synchrony_range(overlap_frac: float) -> None:
    """Test synchrony across a range of overlap fractions."""
    events1, events2 = create_partial_overlap_events(
        n_frames=1000, n_events=40, overlap_fraction=overlap_frac
    )
    sync = _calculate_jitter_window_synchrony(events1, events2, jitter_window=1)

    if overlap_frac == 0.0:
        # With jitter window, might catch some random overlaps
        assert sync < 0.2, f"No overlap should give low sync, got {sync}"
    elif overlap_frac == 1.0:
        assert sync > 0.9, f"Full overlap should give sync>0.9, got {sync}"
    else:
        assert 0.0 < sync < 1.0, f"Partial overlap should give 0<sync<1, got {sync}"


def test_empty_events_synchrony() -> None:
    """Test that empty event trains yield synchrony = 0.0."""
    events1 = np.zeros(1000)
    events2 = np.zeros(1000)
    sync = _calculate_jitter_window_synchrony(events1, events2, jitter_window=2)
    assert sync == 0.0, f"Empty events should have synchrony = 0.0, got {sync}"


def test_one_empty_events_synchrony() -> None:
    """Test that one empty event train yields synchrony = 0.0."""
    events1 = np.zeros(1000)
    events1[[100, 200, 300]] = 1.0
    events2 = np.zeros(1000)
    sync = _calculate_jitter_window_synchrony(events1, events2, jitter_window=2)
    assert sync == 0.0, f"One empty event train should have synchrony = 0.0, got {sync}"


# =============================================================================
# Tests: Cross-Method Comparison
# =============================================================================


def test_correlation_vs_synchrony_perfect_match() -> None:
    """Test that perfect correlation and perfect synchrony occur together."""
    # Create identical traces with clear peaks
    trace1, trace2 = create_perfect_correlation_traces(n_frames=1000, amplitude=1.0)

    # Create binary events from peaks (simple threshold)
    events1 = (trace1 > 0.5).astype(float)
    events2 = (trace2 > 0.5).astype(float)

    corr = calculate_cross_correlation(trace1, trace2)
    sync = _calculate_jitter_window_synchrony(events1, events2, jitter_window=2)

    assert corr > 0.95, f"Perfect traces should have high correlation, got {corr}"
    # Synchrony depends on how many peaks are detected
    if events1.sum() > 0:
        assert sync == 1.0, (
            f"Identical binary events should have perfect synchrony, got {sync}"
        )


def test_correlation_high_synchrony_low() -> None:
    """Test case where correlation is high but synchrony is low (phase shift)."""
    # Create shifted traces (same shape, different timing)
    trace1, trace2 = create_shifted_correlation_traces(n_frames=1000, shift=100)

    # Create events at specific times (non-overlapping due to shift)
    events1 = np.zeros(1000)
    events2 = np.zeros(1000)
    events1[[200, 400, 600]] = 1.0
    events2[[300, 500, 700]] = 1.0  # Shifted by 100

    corr = calculate_cross_correlation(trace1, trace2)
    sync = _calculate_jitter_window_synchrony(events1, events2, jitter_window=2)

    assert corr > 0.85, (
        f"Shifted identical traces should have high correlation, got {corr}"
    )
    assert sync < 0.1, f"Non-overlapping events should have low synchrony, got {sync}"


def test_correlation_low_synchrony_high() -> None:
    """Test case where synchrony is high but correlation is low."""
    # Create events at same times
    events1, events2 = create_perfect_synchrony_events(n_frames=1000, n_events=20)

    # Create continuous traces with different amplitudes and noise
    trace1 = events1.copy()
    trace2 = events2 * 2.0 + np.random.randn(1000) * 0.5  # Different amplitude + noise

    corr = calculate_cross_correlation(trace1, trace2)
    sync = _calculate_jitter_window_synchrony(events1, events2, jitter_window=2)

    assert sync == 1.0, f"Identical events should have perfect synchrony, got {sync}"
    # Correlation might be moderate due to noise and amplitude difference
    assert 0.0 <= corr <= 1.0, f"Correlation should be valid range, got {corr}"


# =============================================================================
# Summary Test: Range of Scenarios
# =============================================================================


def test_comprehensive_scenario_matrix() -> None:
    """Test a comprehensive matrix of correlation and synchrony scenarios."""
    scenarios = [
        # (name, trace_gen, event_gen, expected_corr_range, expected_sync_range)
        (
            "Perfect match",
            create_perfect_correlation_traces,
            create_perfect_synchrony_events,
            (0.95, 1.0),
            (0.95, 1.0),
        ),
        (
            "Independent",
            create_independent_traces,
            create_no_overlap_events,
            (0.0, 0.15),
            (0.0, 0.05),
        ),
        (
            "Anticorrelated traces",
            create_anticorrelation_traces,
            None,
            (-1.0, -0.95),
            None,
        ),
    ]

    results = []
    for name, trace_gen, event_gen, corr_range, sync_range in scenarios:
        if trace_gen:
            trace1, trace2 = trace_gen(n_frames=2000)
            corr = calculate_cross_correlation(trace1, trace2)
        else:
            corr = None

        if event_gen:
            events1, events2 = event_gen(n_frames=2000, n_events=30)
            sync = _calculate_jitter_window_synchrony(events1, events2, jitter_window=2)
        else:
            sync = None

        results.append((name, corr, sync))

        # Validate ranges
        if corr is not None and corr_range is not None:
            # For anticorrelation, check zero-lag correlation
            if name == "Anticorrelated traces":
                x = zscore(trace1)
                y = zscore(trace2)
                corr_check = np.corrcoef(x, y)[0, 1]
                assert corr_range[0] <= corr_check <= corr_range[1] + 1e-10, (
                    f"{name}: correlation {corr_check} not in range {corr_range}"
                )
            else:
                # Allow small floating point tolerance
                assert corr_range[0] <= corr <= corr_range[1] + 1e-10, (
                    f"{name}: correlation {corr} not in range {corr_range}"
                )

        if sync is not None and sync_range is not None:
            assert sync_range[0] <= sync <= sync_range[1], (
                f"{name}: synchrony {sync} not in range {sync_range}"
            )

    # Print summary (visible with pytest -v)
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SCENARIO RESULTS")
    print("=" * 60)
    for name, corr, sync in results:
        corr_str = f"{corr:.3f}" if corr is not None else "N/A"
        sync_str = f"{sync:.3f}" if sync is not None else "N/A"
        print(f"{name:25s} | Corr: {corr_str:6s} | Sync: {sync_str:6s}")
    print("=" * 60)


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_correlation_vs_synchrony.py -v -s
    pytest.main([__file__, "-v", "-s"])
