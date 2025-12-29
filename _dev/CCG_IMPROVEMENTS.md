# Cross-Correlogram (CCG) Analysis Improvements

## Overview

The cross-correlation analysis in CALI has been significantly improved to align with standard practices in spike train analysis and calcium imaging literature. These improvements make the CCG analysis more consistent with methods used in electrophysiology, neuroscience papers using OASIS-deconvolved spikes, and standard libraries like Elephant (Neo).

## Summary of Changes

Based on feedback from experts in the field, we've implemented the following improvements:

### 1. Full CCG Vector Computation

**Before:** Only the maximum correlation and its lag were returned.

**Now:** The complete CCG curve across all lags is computed and can be returned for further analysis.

**Why this matters:** Standard CCG analysis examines the entire curve to identify:
- Peak correlation and its lag
- Width of the correlation peak
- Asymmetry in the CCG (directional connectivity)
- Multiple peaks (polysynaptic connections)

**Functions:**
- `_compute_ccg_vector()` - Returns full CCG curve
- `_calculate_cross_correlation_with_lag()` - Legacy function (backward compatible)

### 2. Standard Normalization Methods

**Before:** Used cosine similarity normalization (dot product divided by signal norms).

**Now:** Three normalization options:

1. **`trigger_rate`** (default, standard CCG):
   - `CCG(τ) = (count at lag τ) / N_ref / Δt`
   - Gives units of Hz (spikes per second)
   - Interpretation: "conditional rate of target spikes at lag τ relative to reference spikes"

2. **`trigger_prob`**:
   - `CCG(τ) = (count at lag τ) / N_ref`
   - Unitless probability (0 to 1)
   - Interpretation: "probability of target spike at lag τ given reference spike"

3. **`cosine`** (legacy):
   - Original cosine similarity
   - Maintained for backward compatibility

**Why this matters:** Per-trigger normalization is the standard in spike train CCG analysis. It directly measures the conditional probability/rate of observing target events relative to reference events, which is the fundamental definition of a cross-correlogram.

### 3. Border/Overlap Correction

**Before:** No correction for reduced overlap at large lags.

**Now:** Optional border correction (enabled by default) that accounts for the fact that at large lags, the overlapping region between signals is smaller.

**Why this matters:** Without correction, CCG values at large lags are systematically biased downward simply because fewer samples overlap. This makes it hard to compare CCG values across different lags. Border correction normalizes by the actual overlap length at each lag.

**Example:**
```python
# At lag=0: full overlap (n samples)
# At lag=50 with n=100: only 50 samples overlap
# Border correction scales the count by n/overlap
```

### 4. Jitter/Shuffle Null Model

**Before:** No baseline correction for slow co-modulations.

**Now:** `_compute_baseline_corrected_ccg()` computes both:
- Raw CCG
- Baseline CCG from shuffled/shifted surrogates
- Mean and std of the baseline for z-score calculation

**Why this matters:** Calcium signals often have slow global modulations (e.g., network state changes, temperature drift, photobleaching). Two neurons might show correlation simply because they both increase activity over time, not because they're functionally connected. The jitter/shuffle baseline captures this "chance" correlation level, allowing you to identify true functional coupling.

**Method:**
- Circularly shift one signal by random amounts (avoiding small shifts)
- Compute CCG for each shifted version
- Average to get baseline CCG
- Compare raw CCG to baseline (residual or z-score)

### 5. Near-Zero Lag Synchrony Summary

**Before:** Reported global maximum correlation across all lags.

**Now:** `_summarize_ccg_near_zero()` extracts peak CCG value within a small window around lag=0.

**Why this matters:** For functional connectivity analysis, you typically care about near-synchronous coupling (zero or small lag). A large correlation at lag=500ms might reflect common input or sequential activation, but not direct coupling. Focusing on near-zero lag is more appropriate for "synchrony" measures.

## Usage Examples

### Basic CCG Computation

```python
from cali.analysis._util import _compute_ccg_vector

# Your binary event trains (from OASIS or peak detection)
events_i = np.array([1, 0, 0, 1, 0, 1, ...])  # Reference neuron
events_j = np.array([0, 1, 0, 0, 1, 1, ...])  # Target neuron

# Compute CCG with standard normalization
lags, ccg = _compute_ccg_vector(
    events_i, 
    events_j,
    max_lag=50,  # frames
    normalization="trigger_rate",  # Hz units
    border_correction=True,
    dt=0.1  # 10 Hz frame rate -> 0.1s per frame
)

# Plot the CCG
import matplotlib.pyplot as plt
plt.plot(lags, ccg)
plt.xlabel('Lag (frames)')
plt.ylabel('Rate (Hz)')
plt.title('Cross-Correlogram')
```

### Baseline-Corrected CCG

```python
from cali.analysis._util import _compute_baseline_corrected_ccg

# Compute CCG with shuffle baseline
lags, ccg_raw, baseline_mean, baseline_std = _compute_baseline_corrected_ccg(
    events_i,
    events_j,
    max_lag=50,
    n_shuffles=100,
    normalization="trigger_rate",
    dt=0.1
)

# Compute z-score
ccg_zscore = (ccg_raw - baseline_mean) / (baseline_std + 1e-10)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

axes[0].plot(lags, ccg_raw, label='Raw CCG')
axes[0].plot(lags, baseline_mean, label='Baseline', linestyle='--')
axes[0].fill_between(lags, 
                      baseline_mean - 2*baseline_std, 
                      baseline_mean + 2*baseline_std, 
                      alpha=0.3, label='±2 SD')
axes[0].legend()
axes[0].set_ylabel('Rate (Hz)')

axes[1].plot(lags, ccg_zscore)
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].axhline(3, color='r', linestyle='--', alpha=0.3, label='3σ')
axes[1].axhline(-3, color='r', linestyle='--', alpha=0.3)
axes[1].legend()
axes[1].set_xlabel('Lag (frames)')
axes[1].set_ylabel('Z-score')
```

### Near-Zero Synchrony Summary

```python
from cali.analysis._util import _summarize_ccg_near_zero

# After computing CCG...
synchrony = _summarize_ccg_near_zero(lags, ccg, window=2)
# Returns peak CCG value within [-2, +2] frames of lag=0

print(f"Near-zero synchrony: {synchrony:.3f} Hz")
```

### Backward Compatible Usage

The legacy `_calculate_cross_correlation_with_lag()` function still works exactly as before:

```python
from cali.analysis._util import _calculate_cross_correlation_with_lag

# Legacy API (returns only max and its lag)
max_corr, best_lag = _calculate_cross_correlation_with_lag(
    events_i, events_j, max_lag=50
)
```

## Implementation Details

### Numba Optimization

All CCG functions are implemented with Numba JIT compilation for performance:

- `_compute_ccg_vector_numba()` - Core CCG computation
- `_compute_baseline_corrected_ccg_numba()` - CCG with shuffles
- `_max_cross_correlation_numba()` - Legacy implementation

Expected speedup: 10-100x compared to pure Python, depending on signal length and number of ROIs.

### Thread Safety

CCG computations use the `_NUMBA_LOCK` to prevent thread serialization issues during Numba compilation in parallel contexts.

## Testing

Comprehensive tests are in `tests/test_ccg_improvements.py`:

- Basic CCG computation
- All normalization methods
- Border correction
- Baseline correction with shuffles
- Near-zero synchrony summary
- Edge cases (empty trains, etc.)
- Consistency across normalizations

All tests pass and maintain backward compatibility with existing code.

## References

### Standards and Best Practices

1. **Elephant Library (Neo project)**
   - [Elephant CCG documentation](https://elephant.readthedocs.io/en/latest/reference/spike_train_correlation.html)
   - Standard implementation of spike train cross-correlograms
   - Defines border correction and normalization approaches

2. **Calcium Imaging with Deconvolved Spikes**
   - Many papers compute CCG from OASIS-deconvolved spike probabilities
   - Common to threshold to binary events or detect rising edges
   - Baseline correction via shuffling is standard practice

3. **Electrophysiology Literature**
   - CCG is defined as conditional probability/rate
   - Per-trigger normalization is standard
   - Jitter correction for slow co-modulations

### Key Papers

- Papers using CCG with calcium imaging and deconvolved spikes typically:
  - Normalize by "total activity" or number of reference spikes
  - Use shuffled/jittered baselines to assess significance
  - Focus on near-zero lag for synchrony measures
  - Report both the CCG curve and summary statistics

## Migration Guide

### If you were using the old API

No changes needed! The `_calculate_cross_correlation_with_lag()` function still works with cosine similarity normalization.

### To use the new standard CCG

Replace:
```python
max_corr, best_lag = _calculate_cross_correlation_with_lag(events_i, events_j, max_lag=50)
```

With:
```python
lags, ccg = _compute_ccg_vector(
    events_i, events_j, 
    max_lag=50, 
    normalization="trigger_rate",  # or "trigger_prob"
    dt=0.1  # your frame rate
)
# Then analyze the full CCG curve...
max_idx = np.argmax(ccg)
max_corr = ccg[max_idx]
best_lag = lags[max_idx]
```

### To add baseline correction

Simply wrap with:
```python
lags, ccg_raw, baseline_mean, baseline_std = _compute_baseline_corrected_ccg(
    events_i, events_j, max_lag=50, n_shuffles=100
)

# Compute z-score for significance testing
z_score = (ccg_raw - baseline_mean) / (baseline_std + 1e-10)

# Consider significant if z > 3 at near-zero lag
```

## Future Enhancements

Possible future additions:

1. **Multiple comparison correction** for matrix of CCGs
2. **Confidence intervals** from bootstrapping
3. **Automatic lag detection** with adaptive window sizing
4. **Visualization utilities** for CCG matrices
5. **Integration with existing matrix functions** in the codebase

## Questions?

For questions about the implementation, see:
- Code: `src/cali/analysis/_util.py` (functions starting with `_compute_ccg_`)
- Tests: `tests/test_ccg_improvements.py`
- Original discussion: ChatGPT feedback on standard CCG practices
