"""Generate visual examples for correlation metrics documentation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
output_dir = Path(__file__).parent / "correlation_figures"
output_dir.mkdir(exist_ok=True)

# Time parameters
fs = 10  # sampling rate (Hz)
duration = 30  # seconds
t = np.arange(0, duration, 1 / fs)


def create_dff_traces() -> None:
    """Create DF/F traces showing correlation."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # ROI 1: Base signal with slow fluctuations
    base_freq1 = 0.15
    base_signal1 = 0.5 * np.sin(2 * np.pi * base_freq1 * t)
    base_signal1 += 0.3 * np.sin(2 * np.pi * 0.25 * t + 0.5)
    noise1 = np.random.normal(0, 0.1, len(t))
    dff1 = base_signal1 + noise1

    # ROI 2: Correlated with ROI 1
    dff2 = 0.8 * dff1 + 0.2 * np.random.normal(0, 0.15, len(t))

    # ROI 3: Uncorrelated
    base_freq3 = 0.22
    base_signal3 = 0.4 * np.sin(2 * np.pi * base_freq3 * t + 1.5)
    noise3 = np.random.normal(0, 0.12, len(t))
    dff3 = base_signal3 + noise3

    # Plot
    axes[0].plot(t, dff1, "b-", linewidth=1.5, label="ROI 1")
    axes[0].set_ylabel("ΔF/F", fontsize=12)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        "DF/F Calcium Traces - Zero-lag Pearson Correlation",
        fontsize=14,
        fontweight="bold",
    )

    axes[1].plot(t, dff2, "r-", linewidth=1.5, label="ROI 2 (correlated with ROI 1)")
    axes[1].set_ylabel("ΔF/F", fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, dff3, "g-", linewidth=1.5, label="ROI 3 (independent)")
    axes[2].set_ylabel("ΔF/F", fontsize=12)
    axes[2].set_xlabel("Time (s)", fontsize=12)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Calculate correlations
    corr_12 = np.corrcoef(dff1, dff2)[0, 1]
    corr_13 = np.corrcoef(dff1, dff3)[0, 1]

    msg = (
        f"Pearson r (ROI 1 vs ROI 2): {corr_12:.3f}  |  "
        f"Pearson r (ROI 1 vs ROI 3): {corr_13:.3f}"
    )
    fig.text(
        0.5,
        0.02,
        msg,
        ha="center",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_dir / "dff_traces_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Created DF/F traces (r1-r2={corr_12:.3f}, r1-r3={corr_13:.3f})")


def create_deconvolved_traces() -> None:
    """Create deconvolved C(t) traces."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Simulate calcium events (sparse transients)
    events1 = np.zeros(len(t))
    event_times1 = [30, 80, 150, 200, 250]
    for ev in event_times1:
        if ev < len(t):
            # Fast rise, exponential decay
            tau = 20  # decay time constant
            events1[ev : min(ev + 100, len(t))] += np.exp(
                -np.arange(min(100, len(t) - ev)) / tau
            )

    # ROI 2: Similar events with slight shift
    events2 = np.zeros(len(t))
    event_times2 = [35, 85, 155, 205, 255]
    for ev in event_times2:
        if ev < len(t):
            tau = 18
            events2[ev : min(ev + 100, len(t))] += 0.9 * np.exp(
                -np.arange(min(100, len(t) - ev)) / tau
            )

    # ROI 3: Different pattern
    events3 = np.zeros(len(t))
    event_times3 = [50, 120, 180, 220, 270]
    for ev in event_times3:
        if ev < len(t):
            tau = 22
            events3[ev : min(ev + 100, len(t))] += 0.8 * np.exp(
                -np.arange(min(100, len(t) - ev)) / tau
            )

    # Add small noise
    c1 = events1 + np.random.normal(0, 0.02, len(t))
    c2 = events2 + np.random.normal(0, 0.02, len(t))
    c3 = events3 + np.random.normal(0, 0.02, len(t))

    # Plot
    axes[0].plot(t, c1, "b-", linewidth=1.5, label="ROI 1")
    axes[0].set_ylabel("C(t)", fontsize=12)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        "Deconvolved Traces C(t) - Zero-lag Pearson Correlation",
        fontsize=14,
        fontweight="bold",
    )

    axes[1].plot(t, c2, "r-", linewidth=1.5, label="ROI 2 (similar dynamics)")
    axes[1].set_ylabel("C(t)", fontsize=12)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, c3, "g-", linewidth=1.5, label="ROI 3 (different pattern)")
    axes[2].set_ylabel("C(t)", fontsize=12)
    axes[2].set_xlabel("Time (s)", fontsize=12)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Calculate correlations
    corr_12 = np.corrcoef(c1, c2)[0, 1]
    corr_13 = np.corrcoef(c1, c3)[0, 1]

    fig.text(
        0.5,
        0.02,
        f"Pearson r (ROI 1 vs ROI 2): {corr_12:.3f}  |  "
        f"Pearson r (ROI 1 vs ROI 3): {corr_13:.3f}",
        ha="center",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(
        output_dir / "deconvolved_traces_correlation.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"✓ Created deconvolved C(t) traces (r1-r2={corr_12:.3f}, r1-r3={corr_13:.3f})"
    )


def create_calcium_peaks() -> None:
    """Create calcium peaks with jitter synchrony."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Peak times (in seconds)
    peaks1 = np.array([2.5, 7.0, 12.3, 16.8, 22.1, 26.5])
    peaks2 = np.array([2.7, 7.2, 12.5, 17.0, 22.3, 26.7])  # Correlated with jitter
    peaks3 = np.array([4.0, 9.5, 14.2, 19.0, 24.5, 28.0])  # Less synchronized

    # Plot raster-style
    axes[0].eventplot(
        [peaks1], lineoffsets=0.5, linelengths=0.8, colors="b", linewidths=2
    )
    axes[0].set_ylabel("ROI 1", fontsize=12)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([])
    axes[0].grid(True, alpha=0.3, axis="x")
    axes[0].set_title(
        "Calcium Peaks - Jitter Synchrony (±200ms window)",
        fontsize=14,
        fontweight="bold",
    )

    axes[1].eventplot(
        [peaks2], lineoffsets=0.5, linelengths=0.8, colors="r", linewidths=2
    )
    axes[1].set_ylabel("ROI 2", fontsize=12)
    axes[1].set_ylim(0, 1)
    axes[1].set_yticks([])
    axes[1].grid(True, alpha=0.3, axis="x")

    # Highlight synchronous events
    jitter_window = 0.4  # ±200ms
    for p1 in peaks1:
        for p2 in peaks2:
            if abs(p1 - p2) <= jitter_window:
                axes[1].axvspan(
                    p1 - jitter_window,
                    p1 + jitter_window,
                    alpha=0.15,
                    color="yellow",
                )
                break

    axes[2].eventplot(
        [peaks3], lineoffsets=0.5, linelengths=0.8, colors="g", linewidths=2
    )
    axes[2].set_ylabel("ROI 3", fontsize=12)
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks([])
    axes[2].set_xlabel("Time (s)", fontsize=12)
    axes[2].grid(True, alpha=0.3, axis="x")

    # Cross-correlogram for ROI 1 vs ROI 2
    max_lag = 5  # in units
    lags = np.arange(-max_lag, max_lag + 1)
    xcorr = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        lag_sec = lag * 0.2  # 200ms bins
        for p1 in peaks1:
            for p2 in peaks2:
                if abs(p1 - p2 - lag_sec) < 0.1:
                    xcorr[i] += 1

    axes[3].bar(
        lags * 200, xcorr, width=150, color="purple", alpha=0.7, edgecolor="black"
    )
    axes[3].set_xlabel("Lag (ms)", fontsize=12)
    axes[3].set_ylabel("Coincidences", fontsize=12)
    axes[3].set_title("Event Cross-Correlation (ROI 1 vs ROI 2)", fontsize=11)
    axes[3].grid(True, alpha=0.3, axis="y")
    axes[3].axvline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    # Calculate synchrony metrics
    sync_count_12 = 0
    for p1 in peaks1:
        for p2 in peaks2:
            if abs(p1 - p2) <= jitter_window:
                sync_count_12 += 1
                break
    sync_12 = sync_count_12 / len(peaks1)

    sync_count_13 = 0
    for p1 in peaks1:
        for p3 in peaks3:
            if abs(p1 - p3) <= jitter_window:
                sync_count_13 += 1
                break
    sync_13 = sync_count_13 / len(peaks1)

    fig.text(
        0.5,
        0.02,
        f"Synchrony (ROI 1 vs ROI 2): {sync_12:.3f}  |  "
        f"Synchrony (ROI 1 vs ROI 3): {sync_13:.3f}  (±200ms window)",
        ha="center",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(
        output_dir / "calcium_peaks_synchrony.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"✓ Created calcium peaks (sync1-2={sync_12:.3f}, sync1-3={sync_13:.3f})")


def create_spike_trains() -> None:
    """Create inferred spike trains with CCG."""
    plt.figure(figsize=(12, 10))

    # Create spike trains
    spike_times1 = np.array(
        [1.2, 3.5, 5.1, 7.8, 10.2, 12.5, 15.0, 17.3, 20.1, 22.8, 25.5, 28.0]
    )
    spike_times2 = np.array(
        [1.25, 3.55, 5.15, 7.85, 10.25, 12.55, 15.05, 17.4, 20.2, 23.0, 25.7, 28.1]
    )  # Coupled
    spike_times3 = np.array(
        [2.0, 4.5, 8.0, 11.0, 14.5, 18.0, 21.5, 24.0, 27.0, 29.0]
    )  # Independent

    # Spike raster
    ax1 = plt.subplot(3, 2, (1, 2))
    ax1.eventplot(
        [spike_times1],
        lineoffsets=2,
        linelengths=0.8,
        colors="b",
        linewidths=2.5,
        label="ROI 1",
    )
    ax1.eventplot(
        [spike_times2],
        lineoffsets=1,
        linelengths=0.8,
        colors="r",
        linewidths=2.5,
        label="ROI 2",
    )
    ax1.eventplot(
        [spike_times3],
        lineoffsets=0,
        linelengths=0.8,
        colors="g",
        linewidths=2.5,
        label="ROI 3",
    )
    ax1.set_ylabel("ROI", fontsize=12)
    ax1.set_yticks([0, 1, 2])
    ax1.set_yticklabels(["ROI 3", "ROI 2", "ROI 1"])
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_xlim(0, 30)
    ax1.grid(True, alpha=0.3, axis="x")
    ax1.legend(loc="upper right")
    ax1.set_title("Inferred Spike Trains", fontsize=14, fontweight="bold")

    # Cross-correlogram ROI 1 vs ROI 2
    ax2 = plt.subplot(3, 2, 3)
    max_lag_ms = 500
    bin_size_ms = 20
    n_bins = int(2 * max_lag_ms / bin_size_ms) + 1
    ccg = np.zeros(n_bins)
    lags = np.linspace(-max_lag_ms, max_lag_ms, n_bins)

    for s1 in spike_times1:
        for s2 in spike_times2:
            lag_ms = (s2 - s1) * 1000
            if abs(lag_ms) <= max_lag_ms:
                bin_idx = int((lag_ms + max_lag_ms) / bin_size_ms)
                if 0 <= bin_idx < n_bins:
                    ccg[bin_idx] += 1

    ax2.bar(
        lags, ccg, width=bin_size_ms * 0.9, color="purple", alpha=0.7, edgecolor="black"
    )
    ax2.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax2.set_xlabel("Lag (ms)", fontsize=11)
    ax2.set_ylabel("Spike count", fontsize=11)
    ax2.set_title("CCG: ROI 1 vs ROI 2 (coupled)", fontsize=11, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Find peak
    peak_idx = np.argmax(ccg)
    peak_lag = lags[peak_idx]
    ax2.text(
        0.98,
        0.95,
        f"Peak at {peak_lag:.0f} ms",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.7},
    )

    # Cross-correlogram ROI 1 vs ROI 3
    ax3 = plt.subplot(3, 2, 4)
    ccg_13 = np.zeros(n_bins)

    for s1 in spike_times1:
        for s3 in spike_times3:
            lag_ms = (s3 - s1) * 1000
            if abs(lag_ms) <= max_lag_ms:
                bin_idx = int((lag_ms + max_lag_ms) / bin_size_ms)
                if 0 <= bin_idx < n_bins:
                    ccg_13[bin_idx] += 1

    ax3.bar(
        lags,
        ccg_13,
        width=bin_size_ms * 0.9,
        color="gray",
        alpha=0.7,
        edgecolor="black",
    )
    ax3.axvline(0, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax3.set_xlabel("Lag (ms)", fontsize=11)
    ax3.set_ylabel("Spike count", fontsize=11)
    ax3.set_title("CCG: ROI 1 vs ROI 3 (independent)", fontsize=11, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # Jitter synchrony visualization
    ax4 = plt.subplot(3, 2, (5, 6))
    jitter_window = 0.1  # ±50ms

    ax4.eventplot(
        [spike_times1],
        lineoffsets=1,
        linelengths=0.8,
        colors="b",
        linewidths=2.5,
        label="ROI 1",
    )
    ax4.eventplot(
        [spike_times2],
        lineoffsets=0,
        linelengths=0.8,
        colors="r",
        linewidths=2.5,
        label="ROI 2",
    )

    # Highlight synchronous spikes
    for s1 in spike_times1:
        for s2 in spike_times2:
            if abs(s1 - s2) <= jitter_window:
                ax4.axvspan(
                    s1 - jitter_window,
                    s1 + jitter_window,
                    alpha=0.2,
                    color="yellow",
                )
                break

    ax4.set_ylabel("ROI", fontsize=12)
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["ROI 2", "ROI 1"])
    ax4.set_xlabel("Time (s)", fontsize=12)
    ax4.set_xlim(0, 30)
    ax4.legend(loc="upper right")
    ax4.set_title("Jitter Synchrony (±50ms window)", fontsize=11, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="x")

    # Calculate synchrony
    sync_count = 0
    for s1 in spike_times1:
        for s2 in spike_times2:
            if abs(s1 - s2) <= jitter_window:
                sync_count += 1
                break
    sync_score = sync_count / len(spike_times1)

    ax4.text(
        0.98,
        0.95,
        f"Synchrony: {sync_score:.3f}",
        transform=ax4.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.7},
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "spike_trains_ccg_synchrony.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"✓ Created spike trains (peak lag={peak_lag:.0f}ms, sync={sync_score:.3f})")


if __name__ == "__main__":
    print("Generating correlation metric figures...")
    print(f"Output directory: {output_dir}")
    print()

    create_dff_traces()
    create_deconvolved_traces()
    create_calcium_peaks()
    create_spike_trains()

    print()
    print(f"✅ All figures saved to: {output_dir}")
