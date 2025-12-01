# Correlation vs Synchrony Validation Summary

## Overview

Created comprehensive documentation and validation tests for cross-correlation and synchrony calculations in Cali.

## Files Created

### 1. Documentation: `src/cali/plot/_single_wells_plots/correlation/README.md`

Comprehensive 300+ line markdown document explaining:

- **Fundamental differences** between cross-correlation and synchrony
- **Mathematical formulations** with LaTeX equations
- **Implementation details** from Cali codebase
- **When to use which** metric (decision guide)
- **Parameter tuning** recommendations
- **Interpretation examples** with real-world scenarios
- **Comparison table** of features and use cases

Key sections:
- Cross-correlation: Measures waveform similarity in continuous signals
- Synchrony: Measures event coincidence in discrete peak trains
- Global metrics aggregation
- Performance characteristics

### 2. Validation Tests: `tests/test_correlation_vs_synchrony.py`

Comprehensive test suite with 24 tests validating both methods using synthetic data:

#### Test Categories

**Cross-Correlation Tests (9 tests):**
- Perfect correlation (identical traces → r ≈ 1.0) ✅
- Anti-correlation (inverted traces → r ≈ -1.0) ✅
- Independence (random traces → r ≈ 0.0) ✅
- Time-shifted correlation (handles lag detection) ✅
- Partial correlation (0 < r < 1) ✅
- Correlation range parametrized tests (0.3, 0.5, 0.7, 0.9) ✅

**Synchrony Tests (9 tests):**
- Perfect synchrony (identical events → s = 1.0) ✅
- No overlap (disjoint events → s = 0.0) ✅
- Partial overlap (intermediate values) ✅
- Jitter window tolerance (temporal flexibility) ✅
- Synchrony range parametrized tests (0.0, 0.25, 0.5, 0.75, 1.0) ✅
- Empty event handling ✅

**Cross-Method Comparison Tests (6 tests):**
- Perfect match (high corr + high sync) ✅
- High correlation, low synchrony (phase shift) ✅
- Low correlation, high synchrony (different amplitudes) ✅
- Comprehensive scenario matrix ✅

#### Synthetic Data Generators

Created 10 synthetic trace generation functions:

**Correlation traces:**
1. `create_perfect_correlation_traces()` - Identical signals
2. `create_anticorrelation_traces()` - Inverted signals
3. `create_independent_traces()` - Random uncorrelated
4. `create_shifted_correlation_traces()` - Time-lagged signals
5. `create_partial_correlation_traces()` - Mixed signal + noise

**Synchrony events:**
6. `create_perfect_synchrony_events()` - Identical event trains
7. `create_no_overlap_events()` - Completely disjoint
8. `create_partial_overlap_events()` - Controllable overlap fraction
9. `create_jittered_events()` - Small temporal jitter
10. Helper functions for binary event creation

## Test Results

✅ **All 24 tests passing** in 0.59 seconds

Key validation findings:

### Cross-Correlation Behavior
- Perfect identical traces: **r = 1.000**
- Independent traces: **r ≈ 0.06** (noise level)
- Anti-correlated traces: **r ≈ -1.0** (at zero lag)
- Time-shifted identical: **r > 0.93** (lag detection works)
- Partial correlation: Correctly identifies intermediate similarity

### Synchrony Behavior
- Perfect overlap: **s = 1.000**
- No overlap: **s = 0.000**
- Partial overlap: Scales linearly with overlap fraction
- Jitter window: Successfully detects events within tolerance
- Edge cases: Handles empty event trains gracefully

### Cross-Method Insights

The tests confirm theoretical expectations:

| Scenario | Correlation | Synchrony | Interpretation |
|----------|-------------|-----------|----------------|
| Perfect match | ~1.0 | ~1.0 | Strong functional connectivity |
| Independent | ~0.0 | ~0.0 | No relationship |
| Phase shifted | >0.9 | <0.1 | Similar patterns, poor timing |
| Different amplitudes | Variable | 1.0 | Same timing, different strength |

## Key Implementation Notes

### Cross-Correlation (`_calculate_cross_correlation`)
```python
# Z-score normalization → zero mean, unit variance
x = zscore(trace1)
y = zscore(trace2)

# FFT-based correlation across all lags
corr = correlate(x, y, mode='full', method='fft')

# Normalize by signal norms
corr /= (norm_x * norm_y)

# Return maximum correlation (across all lags)
return float(np.max(corr))
```

**Why max() matters:** The implementation takes the maximum correlation across all possible time lags. This explains why partially correlated signals can show higher-than-expected correlation values - the algorithm finds the lag where correlation is maximized, even if zero-lag correlation is lower.

### Synchrony (`_calculate_jitter_window_synchrony`)
```python
# Extract peak times
peaks_i = np.where(events_i > 0)[0]
peaks_j = np.where(events_j > 0)[0]

# Bidirectional coincidence counting
for peak_i in peaks_i:
    distances = np.abs(peaks_j - peak_i)
    if np.any(distances <= jitter_window):
        coincidences_i_to_j += 1

# Symmetric normalization
total_coincidences = coincidences_i_to_j + coincidences_j_to_i
total_peaks = len(peaks_i) + len(peaks_j)
return total_coincidences / total_peaks
```

**Bidirectional counting:** The algorithm counts coincidences in both directions (i→j and j→i) and normalizes by total peaks in both ROIs. This symmetric approach ensures:
- 0.0 ≤ synchrony ≤ 1.0 always
- Handles different peak counts gracefully
- Biologically realistic (allows small timing jitter)

## Usage Examples

### Running Tests
```bash
# Run all validation tests
pytest tests/test_correlation_vs_synchrony.py -v

# Run with detailed output
pytest tests/test_correlation_vs_synchrony.py -v -s

# Run specific test category
pytest tests/test_correlation_vs_synchrony.py -k "correlation" -v
pytest tests/test_correlation_vs_synchrony.py -k "synchrony" -v
```

### Reading Documentation
```bash
# View in browser or markdown viewer
open src/cali/plot/_single_wells_plots/correlation/README.md
```

## Validation Confidence

The comprehensive test suite provides high confidence that:

1. ✅ **Mathematical correctness**: Both methods compute expected values for known inputs
2. ✅ **Robustness**: Handle edge cases (empty data, identical signals, etc.)
3. ✅ **Range validation**: Output values stay within valid bounds
4. ✅ **Biological realism**: Jitter window provides temporal tolerance
5. ✅ **Independence**: Methods capture different aspects of neural activity

## Recommendations for Users

Based on test results:

### Use Cross-Correlation When:
- Analyzing continuous calcium traces (dec_dff)
- Interested in waveform shape similarity
- Detecting rhythmic co-activity
- High SNR data with clear patterns

### Use Synchrony When:
- Analyzing discrete calcium transients/spikes
- Only care about event timing (not amplitude)
- Sparse activity (few peaks per ROI)
- Burst coordination or evoked responses
- Want noise-robust metric

### Parameters
- **Jitter Window**: Recommended 2-5 frames (100-200ms tolerance)
- **Cross-Correlation**: No tuning needed (automatic)

## Future Enhancements

Potential improvements identified during testing:

1. **Adaptive jitter window** based on imaging frame rate
2. **Zero-lag correlation** option for phase-independent analysis
3. **Directional synchrony** (leader-follower detection)
4. **Burst-specific metrics** (network burst synchrony)
5. **Confidence intervals** using bootstrap resampling

## References

Documentation includes proper citations for:
- Scipy correlation methods (FFT-based)
- Neuroscience synchrony measures (Kreuz et al.)
- Calcium imaging analysis literature
