"""Core analysis functions for trace data.

This module contains pure functions for analyzing extracted traces:
- Peak detection in deconvolved traces
- Inter-event interval (IEI) calculation
- Frequency computation
- Amplitude extraction
- Burst detection in individual ROI traces
"""

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from cali._constants import GLOBAL_HEIGHT, GLOBAL_SPIKE_THRESHOLD

if TYPE_CHECKING:
    from cali.sqlmodel._model import AnalysisSettings


def compute_peak_detection_thresholds(
    dec_dff: np.ndarray,
    spikes: np.ndarray,
    settings: "AnalysisSettings",
) -> tuple[float, float, float]:
    """Compute thresholds for peak detection.

    Parameters
    ----------
    dec_dff : np.ndarray
        Deconvolved dF/F trace
    spikes : np.ndarray
        Inferred spikes from OASIS deconvolution
    settings : AnalysisSettings
        Analysis settings containing threshold parameters

    Returns
    -------
    tuple[float, float, float]
        - peaks_height_dec_dff: Height threshold for peak detection
        - peaks_prominence_dec_dff: Prominence threshold
        - spike_detection_threshold: Threshold for spike detection
    """
    # Compute spike detection threshold
    spike_threshold_value = settings.spike_threshold_value
    spike_threshold_mode = settings.spike_threshold_mode

    if spike_threshold_mode == GLOBAL_SPIKE_THRESHOLD:
        spike_detection_threshold = spike_threshold_value
    else:  # MULTIPLIER
        # for spike amp use percentile-based approach to determine noise level
        non_zero_spikes = spikes[spikes > 0]
        # need sufficient data for reliable percentile
        if len(non_zero_spikes) > 5:
            spike_noise_reference = float(np.percentile(non_zero_spikes, 5))
        else:
            spike_noise_reference = 0.01  # fallback value if not enough data
        spike_detection_threshold = spike_noise_reference * spike_threshold_value

    # Get noise level from the ΔF/F0 trace using Median Absolute Deviation (MAD)
    noise_level_dec_dff = float(
        np.median(np.abs(dec_dff - np.median(dec_dff))) / 0.6745
    )

    # Set prominence threshold (how much peaks must stand out from surroundings)
    # Use a fraction of noise level to be less restrictive than height threshold
    prom_multiplier = settings.peaks_prominence_multiplier
    peaks_prominence_dec_dff: float = noise_level_dec_dff * prom_multiplier

    # use the peaks height widget to get the height threshold
    # if the mode is GLOBAL_HEIGHT, use the value directly, otherwise
    # use the value as a multiplier of the noise level
    peaks_height_value = settings.peaks_height_value
    peaks_height_mode = settings.peaks_height_mode
    if peaks_height_mode == GLOBAL_HEIGHT:
        peaks_height_dec_dff = peaks_height_value
    else:  # MULTIPLIER
        peaks_height_dec_dff = noise_level_dec_dff * peaks_height_value

    return peaks_height_dec_dff, peaks_prominence_dec_dff, spike_detection_threshold


def detect_peaks_in_trace(
    dec_dff: np.ndarray,
    peaks_height: float,
    peaks_prominence: float,
    min_distance_frames: int,
) -> tuple[np.ndarray, list[float]]:
    """Detect peaks in deconvolved trace and extract amplitudes.

    Parameters
    ----------
    dec_dff : np.ndarray
        Deconvolved dF/F trace
    peaks_height : float
        Minimum peak height threshold
    peaks_prominence : float
        Minimum peak prominence threshold
    min_distance_frames : int
        Minimum distance between peaks in frames

    Returns
    -------
    tuple[np.ndarray, list[float]]
        - peaks_dec_dff: Array of peak indices
        - peaks_amplitudes_dec_dff: List of peak amplitudes
    """
    # find peaks in the deconvolved trace
    peaks_dec_dff, _ = find_peaks(
        dec_dff,
        prominence=peaks_prominence,
        height=peaks_height,
        distance=min_distance_frames,
    )
    peaks_dec_dff = cast("np.ndarray", peaks_dec_dff)

    # get the amplitudes of the peaks in the dec_dff trace
    peaks_amplitudes_dec_dff = [float(dec_dff[p]) for p in peaks_dec_dff]

    return peaks_dec_dff, peaks_amplitudes_dec_dff


def calculate_frequency(
    num_peaks: int,
    total_time_sec: float,
) -> float | None:
    """Calculate event frequency from peak count and recording duration.

    Parameters
    ----------
    num_peaks : int
        Number of detected peaks
    total_time_sec : float
        Total recording duration in seconds

    Returns
    -------
    float | None
        Frequency in Hz, or None if no peaks or invalid time
    """
    if total_time_sec and num_peaks > 0:
        return num_peaks / total_time_sec
    return None


def calculate_inter_event_intervals(
    peak_indices: np.ndarray,
    elapsed_time_list: list[float],
) -> list[float]:
    """Calculate inter-event intervals from peak indices.

    Parameters
    ----------
    peak_indices : np.ndarray
        Array of frame indices where peaks were detected
    elapsed_time_list : list[float]
        List of elapsed times for each frame (in ms or sec)

    Returns
    -------
    list[float]
        List of inter-event intervals in the same units as elapsed_time_list
    """
    if len(peak_indices) < 2:
        return []

    iei = []
    for i in range(1, len(peak_indices)):
        interval = (
            elapsed_time_list[peak_indices[i]] - elapsed_time_list[peak_indices[i - 1]]
        )
        iei.append(interval)

    return iei


def detect_bursts_in_trace(
    dec_dff: np.ndarray,
    elapsed_time_ms: list[float],
    burst_threshold: float,
    min_duration_ms: float,
    gaussian_sigma: float = 1.0,
) -> tuple[int, float | None, float | None]:
    """Detect bursts in a deconvolved calcium trace.

    A burst is defined as a continuous period where the smoothed trace
    exceeds the burst threshold for at least the minimum duration.

    Parameters
    ----------
    dec_dff : np.ndarray
        Deconvolved dF/F trace
    elapsed_time_ms : list[float]
        List of elapsed times for each frame (milliseconds)
    burst_threshold : float
        Threshold value for burst detection (% dF/F)
    min_duration_ms : float
        Minimum burst duration in milliseconds
    gaussian_sigma : float
        Sigma for Gaussian smoothing (default 1.0)

    Returns
    -------
    tuple[int, float | None, float | None]
        - burst_count: Number of bursts detected
        - burst_avg_duration: Average burst duration in seconds (None if no bursts)
        - burst_avg_interval: Average inter-burst interval in seconds
          (None if < 2 bursts)
    """
    if len(dec_dff) == 0 or len(elapsed_time_ms) == 0:
        return 0, None, None

    # Smooth the trace
    if gaussian_sigma > 0:
        smoothed = gaussian_filter1d(dec_dff, sigma=gaussian_sigma)
    else:
        smoothed = dec_dff

    # Find regions above threshold
    above_threshold = smoothed > burst_threshold
    if not np.any(above_threshold):
        return 0, None, None

    # Find burst start and end points
    above_int = above_threshold.astype(int)
    changes = np.diff(above_int)

    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases
    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(above_threshold))

    # Calculate burst durations and filter by minimum duration
    burst_starts = []
    burst_ends = []
    burst_durations_sec = []

    for start_idx, end_idx in zip(starts, ends):
        duration_ms = elapsed_time_ms[end_idx - 1] - elapsed_time_ms[start_idx]
        if duration_ms >= min_duration_ms:
            burst_starts.append(start_idx)
            burst_ends.append(end_idx)
            burst_durations_sec.append(duration_ms / 1000.0)  # Convert to seconds

    burst_count = len(burst_durations_sec)

    if burst_count == 0:
        return 0, None, None

    burst_avg_duration = float(np.mean(burst_durations_sec))

    # Calculate inter-burst intervals (time from end of one burst to start of next)
    if burst_count < 2:
        burst_avg_interval = None
    else:
        intervals_sec = []
        for i in range(1, burst_count):
            interval_ms = (
                elapsed_time_ms[burst_starts[i]]
                - elapsed_time_ms[burst_ends[i - 1] - 1]
            )
            intervals_sec.append(interval_ms / 1000.0)  # Convert to seconds
        burst_avg_interval = float(np.mean(intervals_sec))

    return burst_count, burst_avg_duration, burst_avg_interval
