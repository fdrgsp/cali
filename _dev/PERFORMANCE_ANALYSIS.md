"""Analysis of compute_fov_analysis performance bottleneck and optimization strategies.

EXECUTIVE SUMMARY
=================
The Spike CCG (Cross-Correlogram) computation dominates execution time,
accounting for 94-95% of total runtime. With 30 shuffles and 193 ROIs,
the operation takes ~26 seconds for one FOV.

TIMING BREAKDOWN
================
FOV B2_0000 (193 ROIs, 2000 frames):
- Total time: 27.4 seconds
- Spike CCG: 25.8 seconds (94.3%) ← BOTTLENECK
- Jitter synchrony: 0.8 seconds (3.1%)
- Data collection: 0.6 seconds (2.2%)
- Other operations: < 1% each

FOV B3_0000 (127 ROIs, 2000 frames):
- Total time: 11.6 seconds
- Spike CCG: 11.0 seconds (94.7%) ← BOTTLENECK
- Other operations: < 5% combined

MATHEMATICAL COMPLEXITY ANALYSIS
================================

Current Implementation:
-----------------------
For N ROIs with T timepoints:

1. Spike CCG computation:
   - Compute CCG for all ROI pairs: O(N²)
   - For each pair:
     * Compute raw CCG: O(T * max_lag)
     * Compute baseline with n_shuffles: O(n_shuffles * T * max_lag)
   
   Total: O(N² * n_shuffles * T * max_lag)

With N=193, T=2000, max_lag=5, n_shuffles=30:
- Number of pairs: 193 * 192 / 2 = 18,528 pairs
- Each pair requires: 30 shuffles * 2000 frames * 5 lags ≈ 300,000 operations
- Total: ~5.6 billion operations

2. Jitter synchrony: O(N² * n_peaks * jitter_window)
   - Much faster because it only checks peak coincidences
   - No shuffling required

WHY IS NUMBA LOCK NEEDED?
==========================
You are correct that multithreading must be stopped for numba-JIT functions.
This is because:

1. Numba's @njit decorator releases the GIL (Global Interpreter Lock)
2. Multiple threads calling numba functions simultaneously can cause:
   - Race conditions in numba's internal state
   - Potential crashes or incorrect results
3. The _NUMBA_LOCK ensures sequential execution of numba-compiled code

However, the parallelization opportunity exists WITHIN each numba function
(using parallel=True), not across ROI pairs in Python.

CURRENT PARALLELIZATION
=======================
The jitter synchrony already uses numba parallel=True:
  @njit(cache=True, parallel=True)
  def _compute_jitter_synchrony_matrix_numba(...)

This provides significant speedup for jitter window computation.

The CCG computation does NOT use parallel=True because:
- It processes ROI pairs sequentially in Python
- Each pair calls _compute_baseline_corrected_ccg_numba
- The numba functions are not parallelized internally

OPTIMIZATION STRATEGIES
========================

Strategy 1: Parallelize CCG at Python level (RECOMMENDED)
----------------------------------------------------------
Instead of sequential processing with _NUMBA_LOCK, use:
- multiprocessing.Pool to distribute ROI pairs across CPU cores
- Each worker processes a subset of ROI pairs
- No _NUMBA_LOCK needed (separate processes)

Estimated speedup: 4-8x (depending on CPU cores)
Implementation complexity: Medium
Risk: Low (well-tested pattern)

Pros:
+ Significant speedup without changing core algorithm
+ Works with existing numba code
+ Easy to parallelize at the pair level

Cons:
- Requires careful chunking of ROI pairs
- Overhead from inter-process communication
- May not work well in GUI context


Strategy 2: Reduce n_shuffles (QUICK WIN)
------------------------------------------
Current: n_shuffles = 30
Consider: n_shuffles = 10-20

From statistics literature, 10-20 shuffles often provide sufficient
baseline estimation for significance testing.

Estimated speedup: 1.5-3x (proportional to reduction)
Implementation: Change one parameter
Risk: Low (validate z-scores are still stable)

Analysis:
- 30 shuffles → 20 shuffles: 1.5x faster
- 30 shuffles → 10 shuffles: 3x faster
- Need to verify z-scores remain reliable


Strategy 3: Optimize numba shuffle generation (MINOR)
------------------------------------------------------
Current code pre-generates random shifts, which is good.
However, the circular shift in Python/numba could be optimized:

    # Current: Manual indexing in numba
    for i in range(n):
        events_j_shifted[i] = events_j[(i + shift) % n]
    
    # Better: Use numpy.roll (but numba doesn't support it well)

Estimated speedup: 5-10% (minor)
Implementation: Low
Risk: Low


Strategy 4: Early termination for non-connected pairs
-----------------------------------------------------
If most ROI pairs have no functional connectivity (z-score < threshold),
could terminate shuffle computation early.

Estimated speedup: Varies (10-50% if many non-connected pairs)
Implementation: Medium-High
Risk: Medium (may miss subtle connections)


Strategy 5: Use GPU acceleration (ADVANCED)
--------------------------------------------
Rewrite CCG computation using:
- CuPy (CUDA arrays, drop-in numpy replacement)
- Numba CUDA kernels
- PyTorch for tensor operations

Estimated speedup: 10-100x (depending on GPU)
Implementation: High (major rewrite)
Risk: High (requires GPU, complex debugging)


Strategy 6: Reduce max_lag parameter
-------------------------------------
Current: max_lag = 500ms → 5 frames (at 10Hz)

If 5 frames is sufficient for your calcium dynamics, this is already optimal.
Reducing further may lose biological signal.

No speedup recommended without domain justification.


Strategy 7: Cache CCG results
------------------------------
If the same FOVs are analyzed multiple times, cache CCG results keyed by:
- ROI spike trains (hash)
- Analysis parameters

Estimated speedup: Infinite for re-analysis
Implementation: Medium
Risk: Low (must handle cache invalidation)


Strategy 8: Adaptive shuffling
-------------------------------
Start with fewer shuffles (e.g., 10), then:
- If baseline std is unstable, add more shuffles
- If z-score is clearly significant/insignificant, stop early

Estimated speedup: 20-50% average
Implementation: High
Risk: Medium (complex stopping criteria)


RECOMMENDED IMMEDIATE ACTIONS
==============================

1. **Reduce n_shuffles** (30 → 15-20)
   - Immediate 1.5-2x speedup
   - Minimal code change
   - Validate z-scores remain stable
   - Time to implement: 5 minutes

2. **Add profiling flag** to disable shuffles entirely for quick analysis
   - Enable fast preview mode without z-scores
   - Keep full analysis for publication-quality results
   - Time to implement: 30 minutes

3. **Parallelize at Python level** (if not GUI context)
   - Use multiprocessing to distribute ROI pairs
   - Expected 4-6x speedup on modern CPUs
   - Time to implement: 2-4 hours

4. **Rising edge analysis** is currently DISABLED (good!)
   - Keep it disabled unless scientifically necessary
   - Would double CCG computation time if enabled


LONG-TERM OPTIMIZATIONS
========================

1. Investigate GPU acceleration for CCG (if processing many experiments)
2. Implement adaptive shuffling with early termination
3. Profile and optimize the numba shuffle loop
4. Consider caching for repeated analyses


CONCLUSION
==========
The current implementation is already well-optimized with numba JIT compilation.
The slowness is inherent to the O(N² * n_shuffles * T) complexity of CCG
with baseline correction.

Quick wins:
- Reduce n_shuffles: 1.5-3x speedup (change one parameter)
- Add fast preview mode: Skip shuffles entirely for quick checks

Significant speedup:
- Parallel processing: 4-8x speedup (requires refactoring)

The "it is what it is" answer: Sort of, but reducing shuffles to 15-20 is
reasonable and would provide significant speedup with minimal risk.

Taleb Ghrear, 2026-01-29
"""
