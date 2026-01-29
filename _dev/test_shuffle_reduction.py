"""Test the effect of reducing n_shuffles on z-score stability.

This script compares z-scores computed with different numbers of shuffles
to validate that reducing n_shuffles (e.g., from 30 to 15) doesn't
significantly impact the statistical reliability of CCG baseline correction.
"""

import time

import numpy as np
from sqlmodel import Session, create_engine, select

from cali.analysis._util import _compute_baseline_corrected_ccg_numba
from cali.sqlmodel._model import FOV

database_path = "/Users/fdrgsp/Desktop/cali/results_new.cali"
fov_name = "B2_0000"  # Use smaller dataset
max_lag = 5  # From settings

engine = create_engine(
    f"sqlite:///{database_path}",
    connect_args={"timeout": 30.0, "check_same_thread": False},
    pool_pre_ping=True,
)

print("Loading spike data...")
with Session(engine) as session:
    fov = session.exec(select(FOV).where(FOV.name == fov_name)).first()
    if fov is None:
        print(f"FOV {fov_name} not found")
        exit(1)

    active_rois = [roi for roi in fov.rois if roi.active]
    print(f"Active ROIs: {len(active_rois)}")

    # Get spike trains for first 20 ROIs (to speed up testing)
    spike_trains = []
    for roi in active_rois[:20]:
        if roi.label_value is None:
            continue

        traces = None
        if hasattr(roi, "_new_traces") and roi._new_traces:
            traces = roi._new_traces[-1]
        elif roi.traces_history:
            traces = roi.traces_history[-1]

        if traces is None or traces.inferred_spikes is None:
            continue

        data_analysis = None
        if hasattr(roi, "_new_data_analysis") and roi._new_data_analysis:
            data_analysis = roi._new_data_analysis[-1]
        elif roi.data_analysis_history:
            data_analysis = roi.data_analysis_history[-1]

        spikes = np.asarray(traces.inferred_spikes, dtype=float)
        spike_threshold = (
            data_analysis.inferred_spikes_threshold
            if data_analysis is not None
            else None
        )

        if spike_threshold is not None:
            spikes_binary = spikes.copy()
            spikes_binary[spikes_binary <= spike_threshold] = 0.0
            spike_train = (spikes_binary > 0.0).astype(float)
            spike_trains.append(spike_train)

print(f"Loaded {len(spike_trains)} spike trains\n")

if len(spike_trains) < 2:
    print("Not enough spike trains")
    exit(1)

# Test different n_shuffles values
shuffle_values = [5, 10, 15, 20, 25, 30, 50]
results = {}

print("Testing n_shuffles impact on z-scores...\n")
print(f"Using first 10 ROI pairs from {len(spike_trains)} ROIs")
print("=" * 70)

# Use first few pairs for testing
test_pairs = [(0, 1), (0, 2), (1, 2), (0, 5), (1, 5), (2, 5), (0, 10), (5, 10)]
test_pairs = [p for p in test_pairs if p[0] < len(spike_trains) and p[1] < len(spike_trains)][:10]

for n_shuffles in shuffle_values:
    print(f"\nTesting n_shuffles = {n_shuffles}")
    print("-" * 70)

    t0 = time.perf_counter()
    z_scores = []
    max_values = []

    for i, j in test_pairs:
        events_i = spike_trains[i]
        events_j = spike_trains[j]

        # Skip if either has no spikes
        if np.sum(events_i) == 0 or np.sum(events_j) == 0:
            continue

        lags, ccg_raw, baseline_mean, baseline_std = (
            _compute_baseline_corrected_ccg_numba(
                events_i, events_j, max_lag, n_shuffles
            )
        )

        # Get max CCG and z-score
        max_idx = np.argmax(ccg_raw)
        max_value = ccg_raw[max_idx]

        if baseline_std[max_idx] > 0:
            z = (ccg_raw[max_idx] - baseline_mean[max_idx]) / baseline_std[max_idx]
        else:
            z = 0.0

        z_scores.append(z)
        max_values.append(max_value)

    elapsed = time.perf_counter() - t0

    results[n_shuffles] = {
        "z_scores": np.array(z_scores),
        "max_values": np.array(max_values),
        "time": elapsed,
    }

    print(f"  Time: {elapsed:.3f} s")
    print(f"  Z-scores: mean={np.mean(z_scores):.3f}, std={np.std(z_scores):.3f}")
    print(
        f"  Z-scores: min={np.min(z_scores):.3f}, max={np.max(z_scores):.3f}, "
        f"median={np.median(z_scores):.3f}"
    )
    print(f"  Max values: mean={np.mean(max_values):.4f}")

# Analysis
print("\n" + "=" * 70)
print("ANALYSIS: Comparing z-scores across n_shuffles")
print("=" * 70)

baseline_n = 30  # Reference value
baseline_z = results[baseline_n]["z_scores"]

print(f"\nReference: n_shuffles = {baseline_n}")
print(f"  Mean z-score: {np.mean(baseline_z):.3f}")
print(f"  Std z-score: {np.std(baseline_z):.3f}")

print("\nComparison to reference:")
for n_shuffles in shuffle_values:
    if n_shuffles == baseline_n:
        continue

    test_z = results[n_shuffles]["z_scores"]

    # Compute correlation with baseline
    correlation = np.corrcoef(baseline_z, test_z)[0, 1]

    # Compute mean absolute difference
    mad = np.mean(np.abs(baseline_z - test_z))

    # Relative time
    time_ratio = results[baseline_n]["time"] / results[n_shuffles]["time"]

    print(f"\n  n_shuffles = {n_shuffles}:")
    print(f"    Correlation with n=30: {correlation:.4f}")
    print(f"    Mean abs difference: {mad:.4f}")
    print(f"    Speedup: {time_ratio:.2f}x")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

# Find the smallest n_shuffles with high correlation
for n_shuffles in [10, 15, 20]:
    if n_shuffles not in results:
        continue
    test_z = results[n_shuffles]["z_scores"]
    correlation = np.corrcoef(baseline_z, test_z)[0, 1]
    mad = np.mean(np.abs(baseline_z - test_z))
    time_ratio = results[baseline_n]["time"] / results[n_shuffles]["time"]

    if correlation > 0.95 and mad < 0.5:
        print(f"\n✓ n_shuffles = {n_shuffles} is ACCEPTABLE:")
        print(f"  - High correlation (r={correlation:.4f})")
        print(f"  - Low z-score error (MAD={mad:.4f})")
        print(f"  - Speedup: {time_ratio:.2f}x faster")
        print(f"  - Recommended for routine analysis")
    elif correlation > 0.90:
        print(f"\n~ n_shuffles = {n_shuffles} is MARGINAL:")
        print(f"  - Good correlation (r={correlation:.4f})")
        print(f"  - Moderate z-score error (MAD={mad:.4f})")
        print(f"  - Speedup: {time_ratio:.2f}x faster")
        print(f"  - Consider for exploratory analysis only")
    else:
        print(f"\n✗ n_shuffles = {n_shuffles} is NOT RECOMMENDED:")
        print(f"  - Low correlation (r={correlation:.4f})")
        print(f"  - High z-score error (MAD={mad:.4f})")
        print(f"  - Too unreliable for statistical inference")

print("\nConclusion:")
print("  Keep n_shuffles = 30 for publication-quality analysis")
print("  Use n_shuffles = 15-20 for routine/exploratory analysis")
print("  Use n_shuffles = 5-10 for rapid previews only")
