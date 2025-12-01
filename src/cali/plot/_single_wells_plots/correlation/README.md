# Cross-Correlation vs Synchrony: Methods Comparison

This document explains the fundamental differences between **cross-correlation** and **synchrony** calculations used in Cali for analyzing calcium imaging data.

---

## Overview

Both metrics measure similarity between neural activity patterns, but they differ fundamentally in:
- **What they measure**: Linear relationship vs temporal coincidence
- **Data type**: Continuous signals vs discrete events
- **Sensitivity**: Amplitude/shape vs timing only
- **Output range**: [-1, 1] vs [0, 1]

---

## Cross-Correlation

### Definition
**Cross-correlation** measures the linear similarity between two continuous signals as a function of time lag. It finds the maximum normalized correlation across all possible time shifts.

### Mathematical Formulation

For two zero-mean signals $x(t)$ and $y(t)$:

$$
\text{CrossCorr}(x, y) = \max_{\tau} \frac{\sum_t x(t) \cdot y(t + \tau)}{\|x\| \cdot \|y\|}
$$

Where:
- $\tau$ is the time lag (shift)
- $\|x\|$ is the L2 norm of signal $x$
- The result is normalized to $[-1, 1]$

### Implementation in Cali

```python
# From _plot_calcium_peaks_correlation.py
def _calculate_cross_correlation(traces_array):
    """
    1. Z-score normalize all traces (zero mean, unit variance)
    2. For each pair of ROIs (i, j):
       - Compute FFT-based cross-correlation across all lags
       - Normalize by signal norms
       - Take maximum correlation value
    3. Build symmetric correlation matrix
    """
    dff_zero_mean = zscore(traces_array, axis=1)  # (n_rois, n_frames)
    norms = np.linalg.norm(dff_zero_mean, axis=1)
    
    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            corr = correlate(x, y, mode='full', method='fft')
            corr /= (norms[i] * norms[j])
            max_corr = float(np.max(corr))
```

### Key Properties

✅ **Advantages:**
- Detects similar **waveform shapes** even if amplitude differs
- Robust to **temporal shifts** (finds best alignment)
- Standard neuroscience metric for signal similarity
- Works well with **continuous calcium traces**

❌ **Limitations:**
- Requires **sufficient data** (many time points)
- Sensitive to **signal-to-noise ratio**
- Can be affected by **baseline drift**
- Computationally expensive for many ROIs

### Use Cases
- Comparing **calcium trace shapes** (dec_dff signals)
- Finding ROIs with **similar activity patterns**
- Network connectivity analysis
- Detecting **coordinated slow dynamics**

---

## Synchrony (Jitter Window Method)

### Definition
**Synchrony** measures the proportion of discrete events (peaks) that occur close together in time, allowing for small timing jitter. It's specifically designed for event-based analysis.

### Mathematical Formulation

For two binary event trains $e_i$ and $e_j$ with peak times $P_i = \{t_1^i, t_2^i, ...\}$ and $P_j = \{t_1^j, t_2^j, ...\}$:

$$
\text{Synchrony}(i, j) = \frac{C_{i \to j} + C_{j \to i}}{|P_i| + |P_j|}
$$

Where:
- $C_{i \to j}$ = number of peaks in $P_i$ that have a matching peak in $P_j$ within jitter window $w$
- $C_{j \to i}$ = number of peaks in $P_j$ that have a matching peak in $P_i$ within jitter window $w$
- $w$ is the jitter window (typically 2-5 frames)

A peak $t^i$ in $P_i$ matches if: $\exists t^j \in P_j : |t^i - t^j| \leq w$

### Implementation in Cali

```python
# From _util.py
def _calculate_jitter_window_synchrony(events_i, events_j, jitter_window):
    """
    1. Extract peak timing indices from binary event arrays
    2. For each peak in ROI i:
       - Check if any peak in ROI j is within ±jitter_window frames
       - Count as coincident if true
    3. Repeat symmetrically (j → i)
    4. Normalize by total number of peaks in both ROIs
    """
    peaks_i = np.where(events_i > 0)[0]
    peaks_j = np.where(events_j > 0)[0]
    
    # Bidirectional coincidence counting
    coincidences_i_to_j = 0
    for peak_i in peaks_i:
        distances = np.abs(peaks_j - peak_i)
        if np.any(distances <= jitter_window):
            coincidences_i_to_j += 1
    
    # Symmetric normalization
    total_coincidences = coincidences_i_to_j + coincidences_j_to_i
    total_peaks = len(peaks_i) + len(peaks_j)
    return total_coincidences / total_peaks
```

### Key Properties

✅ **Advantages:**
- **Timing-focused**: Only cares about when events occur, not amplitude
- **Robust to noise**: Binary events filter out baseline fluctuations
- Handles **sparse activity** (few peaks)
- **Computationally efficient** for event data
- Tolerates small **temporal jitter** (biological realism)

❌ **Limitations:**
- Requires **peak detection** first (threshold-dependent)
- Loses **amplitude information**
- Sensitive to **jitter window parameter**
- Not suitable for continuous oscillations

### Use Cases
- Measuring **network bursting** synchronization
- Analyzing **evoked responses** (did cells respond together?)
- Detecting **functional connectivity** in sparse spiking
- **Population-level coordination** metrics

---

## Global Metrics

Both methods compute pairwise matrices, then aggregate to a single global value:

### Global Correlation
Average or median of upper triangle (excluding diagonal):
$$
\text{Global Corr} = \text{median}(\{C_{ij} : i < j\})
$$

### Global Synchrony
Median of row-wise means (excluding diagonal):
$$
\text{Global Sync} = \text{median}\left(\left\{\frac{1}{N-1}\sum_{j \neq i} S_{ij}\right\}\right)
$$

This gives each ROI equal weight regardless of connectivity.

---

## Comparison Table

| Feature | Cross-Correlation | Synchrony (Jitter Window) |
|---------|-------------------|---------------------------|
| **Input Data** | Continuous traces (dec_dff) | Binary peak events |
| **Output Range** | [-1, 1] | [0, 1] |
| **Measures** | Shape similarity | Temporal coincidence |
| **Amplitude Sensitivity** | Yes (normalized) | No (binary events) |
| **Temporal Tolerance** | All lags tested | ±jitter_window frames |
| **Computational Cost** | O(N²M log M) | O(N²P²) |
| **Best For** | Slow oscillations, waveforms | Sparse bursts, evoked responses |
| **Noise Robustness** | Moderate (SNR dependent) | High (peak threshold acts as filter) |
| **Data Efficiency** | Needs many time points | Works with few events |

Where:
- N = number of ROIs
- M = number of time points (frames)
- P = average number of peaks per ROI

---

## When to Use Which?

### Use Cross-Correlation When:
- ✅ Analyzing **continuous calcium dynamics**
- ✅ Interested in **waveform similarity** (shape, not just timing)
- ✅ Detecting **rhythmic co-activity** (oscillations)
- ✅ You have **high SNR data** with clear signals
- ✅ Need to find **phase relationships** between ROIs

### Use Synchrony When:
- ✅ Analyzing **discrete calcium transients** or spikes
- ✅ Only care about **event timing** (when, not how much)
- ✅ Working with **sparse activity** (few peaks per ROI)
- ✅ Studying **burst coordination** or evoked responses
- ✅ Want **noise-robust** metric (peaks already filtered)
- ✅ Need **interpretable** metric (% of events that coincide)

### Use Both When:
- 🔬 **Comprehensive network analysis** - they capture different aspects
- 🔬 Comparing **spontaneous vs evoked** activity patterns
- 🔬 Validating findings (orthogonal measures should agree)

---

## Parameter Tuning

### Cross-Correlation
- **Minimal parameters** (automatic)
- Z-score normalization handles amplitude differences
- FFT method handles all lags efficiently

### Synchrony
- **Jitter Window** (critical parameter):
  - Too small (e.g., 0-1 frames): Miss biologically synchronous events
  - Too large (e.g., >10 frames): False positives from random overlap
  - **Recommended**: 2-5 frames for typical imaging rates (10-30 Hz)
  - **Rule of thumb**: ~100-200 ms temporal tolerance

#### How to Set Jitter Window:
```python
# Calculate from imaging frame rate
frame_rate = 20  # Hz (frames per second)
tolerance_ms = 150  # milliseconds
jitter_window = int(tolerance_ms / 1000 * frame_rate)  # = 3 frames
```

---

## Interpretation Examples

### High Correlation, Low Synchrony
- ROIs have **similar activity patterns** (waveform shape)
- But peaks don't **align precisely in time**
- Example: Two cells with similar slow oscillations but phase-shifted

### Low Correlation, High Synchrony
- ROIs fire **together** (synchronized bursts)
- But have **different amplitudes** or baseline dynamics
- Example: Weakly connected cell bursts with strongly connected network

### High Correlation, High Synchrony
- **Strong functional connectivity**
- Both shape and timing match
- Example: Electrically coupled cells or common input

### Low Correlation, Low Synchrony
- **Independent activity**
- Different patterns and timing
- Example: Functionally distinct cell populations

---

## Implementation Notes

### Scipy Correlation (Cross-Correlation)
```python
from scipy.signal import correlate
from scipy.stats import zscore

# Fast FFT-based correlation across all lags
corr = correlate(x, y, mode='full', method='fft')
```
- **Method**: FFT convolution theorem
- **Speed**: O(M log M) per pair
- **Memory**: O(M) temporary arrays

### Jitter Window (Synchrony)
```python
# Count coincident peaks within window
for peak_i in peaks_i:
    distances = np.abs(peaks_j - peak_i)
    if np.any(distances <= jitter_window):
        coincidences += 1
```
- **Method**: Vectorized distance calculation
- **Speed**: O(P²) per pair where P = number of peaks
- **Memory**: O(P) peak indices only

---

## Validation and Testing

See `test_correlation_vs_synchrony.py` for synthetic data tests that verify:
1. Perfect correlation → corr ≈ 1.0
2. Anti-correlation → corr ≈ -1.0  
3. Perfect synchrony → sync = 1.0
4. No overlap → sync = 0.0
5. Partial overlap → sync ∈ (0, 1)
6. Independence → both near 0

---

## References

### Cross-Correlation
- Numpy/Scipy documentation on correlation
- Neuroscience: Functional connectivity analysis (fMRI, calcium imaging)
- Signal processing: Time-domain correlation for alignment

### Synchrony
- Kreuz et al. (2007) "Measuring synchronization in coupled model systems"
- Li et al. (2017) "Measuring spike train synchrony" J Neurosci Methods
- Calcium imaging burst detection literature

---

## Summary

**Cross-correlation** answers: *"Do these signals have similar temporal patterns?"*  
**Synchrony** answers: *"Do these neurons fire at the same time?"*

Both are valuable, complementary measures. Cross-correlation captures **continuous dynamics** and **shape similarity**, while synchrony captures **discrete event coincidence** and is more robust to noise. Choose based on your biological question and data characteristics.
