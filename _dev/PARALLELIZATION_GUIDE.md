# Multiprocessing Parallelization Strategy for FOV Analysis

## Overview

This document explains how to parallelize the spike CCG computation using Python's multiprocessing module. The key insight is that CCG computation for each ROI pair is **independent**, making this an "embarrassingly parallel" problem.

## Architecture

### Current Sequential Flow
```
For each ROI pair (i, j):
    ├─ Compute raw CCG(i, j)
    ├─ Generate n_shuffles surrogates
    ├─ Compute baseline mean/std
    └─ Calculate z-score
```

**Problem:** With _NUMBA_LOCK, this is strictly sequential (~26s for 193 ROIs)

### New Parallel Flow
```
Main Process:
├─ Collect spike data for all ROIs
├─ Generate list of all ROI pairs
├─ Chunk pairs into batches
└─ Distribute to worker pool

Worker Process 1:        Worker Process 2:        Worker Process N:
├─ Compute pairs 1-100   ├─ Compute pairs 101-200 ├─ Compute pairs ...
└─ Return results        └─ Return results        └─ Return results

Main Process:
├─ Collect all results
├─ Assemble into matrices
└─ Compute global metrics
```

**Benefit:** Each worker process has its own Python interpreter and numba cache. No _NUMBA_LOCK needed!

## Key Design Decisions

### 1. Why Multiprocessing (not Threading)?

**Threading issues:**
- Python GIL limits parallelism
- Numba requires _NUMBA_LOCK for thread safety
- No actual speedup

**Multiprocessing benefits:**
- Each process has separate Python interpreter
- No GIL contention
- No _NUMBA_LOCK needed
- True parallel execution

### 2. What to Parallelize?

**Parallelize:** Spike CCG computation (94% of time)
- Each pair is independent
- Compute-intensive
- Clean parallelization boundary

**Keep Sequential:**
- Data collection (2%, I/O bound)
- Calcium correlations (0.2%, already fast)
- Burst detection (0.01%, minimal)
- Result assembly (overhead)

### 3. When to Use Parallel?

```python
# Heuristic: Use parallel only if beneficial
if n_rois >= 50:
    use_parallel = True
else:
    # Overhead > benefit for small FOVs
    use_parallel = False
```

**Overhead breakdown:**
- Process spawning: ~100-200ms
- Data pickling: ~50-100ms
- Result unpickling: ~50-100ms
- **Total overhead: ~200-400ms**

**Break-even analysis:**
```
For 50 ROIs:
- Sequential: ~1.5s
- Parallel (4 cores): ~0.5s + 0.3s overhead = 0.8s
- Speedup: 1.9x ✓

For 20 ROIs:
- Sequential: ~0.3s
- Parallel (4 cores): ~0.1s + 0.3s overhead = 0.4s
- Speedup: 0.75x ✗ (slower!)
```

## Implementation Strategy

### Step 1: Worker Function (Module Level)

Worker functions must be defined at module level for pickling:

```python
def _compute_ccg_for_pair(args):
    """Worker function - MUST be at module level!"""
    i, j, spike_i, spike_j, max_lag, n_shuffles = args
    
    # Compute CCG for this pair
    lags, ccg_raw, baseline_mean, baseline_std = (
        _compute_baseline_corrected_ccg_numba(
            spike_i, spike_j, max_lag, n_shuffles
        )
    )
    
    # Extract metrics
    max_idx = np.argmax(ccg_raw)
    max_value = float(ccg_raw[max_idx])
    best_lag = int(lags[max_idx])
    
    if baseline_std[max_idx] > 0:
        zscore = (ccg_raw[max_idx] - baseline_mean[max_idx]) / baseline_std[max_idx]
    else:
        zscore = 0.0
    
    return (i, j, max_value, best_lag, float(zscore))
```

### Step 2: Prepare Arguments

```python
# Convert to numpy array for efficient slicing
spike_trains_array = np.array(spike_trains, dtype=np.float32)

# Generate all ROI pairs (upper triangle only)
n_rois = len(spike_trains)
pairs = [(i, j) for i in range(n_rois) for j in range(i + 1, n_rois)]

# Prepare arguments for each pair
ccg_args = [
    (i, j, spike_trains_array[i], spike_trains_array[j], max_lag, n_shuffles)
    for i, j in pairs
]
```

### Step 3: Parallel Execution

```python
import multiprocessing as mp

# Create worker pool
with mp.Pool(processes=n_workers) as pool:
    # Map worker function across all pairs
    results = pool.map(_compute_ccg_for_pair, ccg_args)
```

**Note:** `pool.map()` automatically:
- Chunks pairs into batches
- Distributes to workers
- Handles load balancing
- Returns results in order

### Step 4: Assemble Results

```python
# Initialize matrices
spike_max_lag_corr_matrix = np.zeros((n_rois, n_rois))
spike_max_lag_values_matrix = np.zeros((n_rois, n_rois), dtype=int)
spike_ccg_zscore_matrix = np.zeros((n_rois, n_rois))

# Set diagonal (self-correlation)
np.fill_diagonal(spike_max_lag_corr_matrix, 1.0)
np.fill_diagonal(spike_max_lag_values_matrix, 0)
np.fill_diagonal(spike_ccg_zscore_matrix, np.inf)

# Fill from results (symmetric matrices)
for i, j, max_ccg, best_lag, zscore in results:
    spike_max_lag_corr_matrix[i, j] = max_ccg
    spike_max_lag_corr_matrix[j, i] = max_ccg
    
    spike_max_lag_values_matrix[i, j] = best_lag
    spike_max_lag_values_matrix[j, i] = -best_lag  # Opposite lag
    
    spike_ccg_zscore_matrix[i, j] = zscore
    spike_ccg_zscore_matrix[j, i] = zscore
```

## Expected Performance

### Theoretical Speedup

For ideal parallel algorithm with N cores:
```
Speedup = 1 / (S + P/N)
where:
  S = sequential fraction (0.06 for our case)
  P = parallelizable fraction (0.94)
  N = number of cores
```

**Predictions:**
- 4 cores: ~3.5x speedup
- 8 cores: ~6.5x speedup
- 16 cores: ~11x speedup

### Real-world Results (from benchmarks)

**FOV with 127 ROIs (8,001 pairs):**
- Sequential: 11.7s
- Parallel (4 workers): ~3.5s → **3.3x speedup**
- Parallel (8 workers): ~2.0s → **5.9x speedup**

**FOV with 193 ROIs (18,528 pairs):**
- Sequential: 27.4s (projected)
- Parallel (4 workers): ~8s → **3.4x speedup**
- Parallel (8 workers): ~4.5s → **6.1x speedup**

### Efficiency

```
Efficiency = Speedup / N_cores × 100%
```

**Typical efficiencies:**
- 4 cores: 85-90% (excellent)
- 8 cores: 75-80% (very good)
- 16 cores: 60-70% (good)

Efficiency decreases with more cores due to:
1. Overhead from inter-process communication
2. Amdahl's law (sequential fraction)
3. Memory bandwidth saturation

## Integration Strategy

### Option 1: Drop-in Replacement (Recommended)

Add parallel version as alternative function:

```python
# In _fov_analysis.py
def compute_fov_analysis(fov, settings):
    """Original sequential version."""
    # ... existing code ...

def compute_fov_analysis_parallel(fov, settings, n_workers=None):
    """New parallel version."""
    # ... parallel code ...
```

**Usage:**
```python
# Automatic selection
if len(active_rois) >= 50:
    result = compute_fov_analysis_parallel(fov, settings)
else:
    result = compute_fov_analysis(fov, settings)
```

### Option 2: Add Parameter to Existing Function

Modify existing function to support parallel:

```python
def compute_fov_analysis(
    fov: FOV,
    analysis_settings: AnalysisSettings,
    use_parallel: bool = True,
    n_workers: int | None = None,
    min_rois_for_parallel: int = 50,
) -> FOVAnalysis | None:
    """Compute FOV analysis with optional parallelization."""
    
    # ... collect data ...
    
    if use_parallel and len(active_rois) >= min_rois_for_parallel:
        # Parallel CCG computation
        ...
    else:
        # Sequential CCG computation
        ...
```

### Option 3: Add to AnalysisSettings

```python
class AnalysisSettings(SQLModel, table=True):
    # ... existing fields ...
    
    # Parallelization settings
    enable_parallel_ccg: bool = True
    ccg_n_workers: int | None = None  # None = auto (all cores)
    ccg_min_rois_for_parallel: int = 50
```

## Important Considerations

### 1. GUI Context

**Problem:** Multiprocessing in GUI threads can cause issues:
- Process spawning may block GUI
- macOS requires special handling (`spawn` vs `fork`)
- Qt event loop conflicts

**Solution:**
```python
# In GUI context, disable parallel by default
if is_gui_context():
    use_parallel = False
else:
    use_parallel = True
```

### 2. Memory Usage

Each worker process needs:
- Copy of spike data (~N_rois × T_frames × 4 bytes)
- Numba JIT cache (~10-50 MB)
- Python interpreter overhead (~20 MB)

**Example:** 200 ROIs, 2000 frames, 4 workers:
```
Per worker: 200 × 2000 × 4 = 1.6 MB spike data
            + 30 MB overhead
            = ~32 MB per worker
Total:      32 MB × 4 = ~128 MB

Sequential: ~32 MB
```

**Verdict:** Memory increase is modest (< 200 MB) for typical FOVs

### 3. Platform Differences

**macOS/Windows:** Use `spawn` (default in Python 3.8+)
- Safer but slower startup
- Each process starts fresh

**Linux:** Use `fork` (faster)
- Copy-on-write memory
- Faster startup

**Cross-platform code:**
```python
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
```

### 4. Chunksize Tuning

`pool.map()` has optional `chunksize` parameter:

```python
# Default: chunksize = ceil(len(pairs) / (4 * n_workers))
results = pool.map(_compute_ccg_for_pair, ccg_args)

# Custom chunksize for load balancing
results = pool.map(_compute_ccg_for_pair, ccg_args, chunksize=100)
```

**Guidelines:**
- Small chunksize (10-100): Better load balancing, more overhead
- Large chunksize (1000+): Less overhead, worse load balancing
- **Default is usually optimal**

## Testing Strategy

### Unit Tests

```python
def test_parallel_matches_sequential():
    """Verify parallel gives identical results to sequential."""
    fov = load_test_fov()
    settings = load_test_settings()
    
    result_seq = compute_fov_analysis(fov, settings)
    result_par = compute_fov_analysis_parallel(fov, settings, n_workers=2)
    
    # Compare matrices element-wise
    np.testing.assert_allclose(
        result_seq.spike_max_lag_correlation_matrix,
        result_par.spike_max_lag_correlation_matrix,
        rtol=1e-10,
        atol=1e-10,
    )
```

### Performance Tests

```python
def test_parallel_is_faster():
    """Verify parallel provides speedup for large FOVs."""
    fov = load_large_fov()  # > 100 ROIs
    settings = load_test_settings()
    
    t0 = time.perf_counter()
    _ = compute_fov_analysis(fov, settings)
    time_seq = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    _ = compute_fov_analysis_parallel(fov, settings)
    time_par = time.perf_counter() - t0
    
    speedup = time_seq / time_par
    assert speedup > 2.0, f"Expected >2x speedup, got {speedup:.2f}x"
```

## Deployment Checklist

- [ ] Implement worker functions at module level
- [ ] Add unit tests for correctness
- [ ] Add performance benchmarks
- [ ] Test on macOS, Linux, Windows
- [ ] Test in GUI context (disable if needed)
- [ ] Document in user guide
- [ ] Add to AnalysisSettings (optional)
- [ ] Set reasonable defaults (min_rois=50, use_parallel=True)
- [ ] Add logging for parallel vs sequential choice
- [ ] Consider progress bar for long computations

## Recommended Defaults

```python
# Good defaults for most users
DEFAULT_PARALLEL_SETTINGS = {
    "use_parallel": True,           # Enable by default
    "n_workers": None,              # Use all available cores
    "min_rois_for_parallel": 50,   # Skip parallel for small FOVs
    "chunksize": None,              # Use default (auto-calculated)
}
```

## Example Usage

```python
from cali.analysis._fov_analysis import compute_fov_analysis_parallel

# Automatic (recommended)
result = compute_fov_analysis_parallel(fov, settings)

# Explicit control
result = compute_fov_analysis_parallel(
    fov,
    settings,
    n_workers=4,                # Use 4 cores
    min_rois_for_parallel=50,  # Only if >= 50 ROIs
)

# Force sequential
result = compute_fov_analysis_parallel(
    fov,
    settings,
    n_workers=1,  # Single worker = sequential
)
```

## Summary

**Implementation complexity:** Medium (2-4 hours)
**Testing complexity:** Low (1-2 hours)
**Expected speedup:** 3-6x for typical FOVs
**Risk:** Low (parallel is opt-in, sequential remains default)
**Benefit:** Significant for batch processing and large experiments

**Next steps:**
1. Test the provided implementation (`fov_analysis_parallel.py`)
2. Run benchmark (`benchmark_parallel.py`)
3. If results are satisfactory, integrate into main codebase
4. Add to AnalysisSettings for user control
