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
from scipy.signal import find_peaks

from cali._constants import GLOBAL_HEIGHT, GLOBAL_SPIKE_THRESHOLD

if TYPE_CHECKING:
    from cali.sqlmodel._model import AnalysisSettings


def compute_inferred_spike_threshold(
    spikes: np.ndarray, settings: "AnalysisSettings"
) -> float:
    """Compute threshold for inferred spikes from OASIS deconvolution.

    Parameters
    ----------
    spikes : np.ndarray
        Inferred spikes from OASIS deconvolution (non-negative)
    settings : AnalysisSettings
        Analysis settings containing threshold parameters

    Returns
    -------
    float
        Threshold for spike detection
    """
    spike_threshold_value = settings.spike_threshold_value
    spike_threshold_mode = settings.spike_threshold_mode

    # User-provided global threshold (absolute units)
    if spike_threshold_mode == GLOBAL_SPIKE_THRESHOLD:
        return float(spike_threshold_value)

    # MULTIPLIER mode → estimate noise level from small positive spikes
    non_zero_spikes = spikes[spikes > 0]

    if non_zero_spikes.size < 5:
        # Very few spikes: be conservative, basically no detection
        return np.inf  # type: ignore

    # Use lower half of distribution as "noise-ish" region
    med_all = np.median(non_zero_spikes)
    lower = non_zero_spikes[non_zero_spikes <= med_all]
    if lower.size < 5:
        lower = non_zero_spikes

    # Robust noise estimate: MAD-based std
    med = np.median(lower)
    mad = np.median(np.abs(lower - med)) / 0.6745 if lower.size > 1 else 0.0

    if mad == 0.0:
        # All small spikes almost identical → threshold slightly above them
        return float(med * spike_threshold_value)

    # Interpret spike_threshold_value as "k" in med + k*noise
    k = float(spike_threshold_value)
    the = med + k * mad

    return float(the)


def compute_calcium_peak_detection_thresholds(
    dec_dff: np.ndarray,
    noise: float | None,
    settings: "AnalysisSettings",
) -> tuple[float, float]:
    """Compute thresholds for peak detection.

    Parameters
    ----------
    dec_dff : np.ndarray
        Deconvolved dF/F trace
    noise : float | None
        Estimated noise level; if None, it will be computed from dec_dff
        as Median Absolute Deviation (MAD)
    settings : AnalysisSettings
        Analysis settings containing threshold parameters

    Returns
    -------
    tuple[float, float]
        - peaks_height_dec_dff: Height threshold for peak detection
        - peaks_prominence_dec_dff: Prominence threshold
    """
    if noise is None:
        # Get noise level from the ΔF/F0 trace using Median Absolute Deviation (MAD)
        noise = float(np.median(np.abs(dec_dff - np.median(dec_dff))) / 0.6745)

    # Set prominence threshold (how much peaks must stand out from surroundings)
    # Use a fraction of noise level to be less restrictive than height threshold
    prom_multiplier = settings.peaks_prominence_multiplier
    peaks_prominence_dec_dff: float = noise * prom_multiplier

    # use the peaks height widget to get the height threshold
    # if the mode is GLOBAL_HEIGHT, use the value directly, otherwise
    # use the value as a multiplier of the noise level
    peaks_height_value = settings.peaks_height_value
    peaks_height_mode = settings.peaks_height_mode
    if peaks_height_mode == GLOBAL_HEIGHT:
        peaks_height_dec_dff = peaks_height_value
    else:  # MULTIPLIER
        peaks_height_dec_dff = noise * peaks_height_value

    return peaks_height_dec_dff, peaks_prominence_dec_dff


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


def threshold_spike_train(
    spikes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Create binary spike train by thresholding continuous spike trace.

    Parameters
    ----------
    spikes : np.ndarray
        Continuous spike trace (e.g., from OASIS deconvolution)
    threshold : float
        Spike detection threshold

    Returns
    -------
    np.ndarray
        Binary spike train (1.0 where spike > threshold, 0.0 elsewhere)
    """
    spikes_binary = spikes.copy()
    spikes_binary[spikes_binary <= threshold] = 0.0
    return (spikes_binary > 0.0).astype(float)


def compute_rising_edges(
    spike_train: np.ndarray,
) -> np.ndarray:
    """Compute rising edges from a binary spike train.

    Detects transitions from 0 to positive values (below to above threshold).

    Parameters
    ----------
    spike_train : np.ndarray
        Binary spike train (from threshold_spike_train)

    Returns
    -------
    np.ndarray
        Binary array with 1.0 at rising edges, 0.0 elsewhere
    """
    positive_vals = spike_train > 0
    rising = positive_vals & ~np.concatenate(([False], positive_vals[:-1]))
    spike_train_rising_edges = np.zeros_like(spike_train, dtype=float)
    spike_train_rising_edges[rising] = 1.0
    return spike_train_rising_edges


def count_thresholded_spike_events(
    spikes: np.ndarray,
    threshold: float,
) -> tuple[int, int]:
    """Count thresholded spike events and rising edges.

    This function counts both the number of frames where spike values exceed
    the threshold (thresholded spikes) and the number of rising edge events
    (transitions from below to above threshold).

    Parameters
    ----------
    spikes : np.ndarray
        Continuous spike trace (e.g., from OASIS deconvolution)
    threshold : float
        Spike detection threshold

    Returns
    -------
    tuple[int, int]
        (num_thresholded_frames, num_rising_edges)
        - num_thresholded_frames: Number of frames where spike > threshold
        - num_rising_edges: Number of transitions from below to above threshold
    """
    spike_train = threshold_spike_train(spikes, threshold)
    num_thresholded = int(np.sum(spike_train > 0))

    rising_edges = compute_rising_edges(spike_train)
    num_rising_edges = int(np.sum(rising_edges))

    return num_thresholded, num_rising_edges


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
