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


def compute_inferred_spike_threshold(
    spikes: np.ndarray, settings: "AnalysisSettings"
) -> float:
    """Compute threshold for inferred spikes from OASIS deconvolution.

    Parameters
    ----------
    spikes : np.ndarray
        Inferred spikes from OASIS deconvolution
    settings : AnalysisSettings
        Analysis settings containing threshold parameters

    Returns
    -------
    float
        Threshold for spike detection
    """
    spike_threshold_value = settings.spike_threshold_value
    spike_threshold_mode = settings.spike_threshold_mode

    if spike_threshold_mode == GLOBAL_SPIKE_THRESHOLD:
        spike_detection_threshold = spike_threshold_value
    else:  # MULTIPLIER
        # for spike amp use percentile-based approach to determine noise level
        non_zero_spikes = spikes[spikes > 0]
        # need sufficient data for reliable percentile
        if len(non_zero_spikes) > 5:
            spike_noise_reference = float(np.percentile(non_zero_spikes, 10))
        else:
            spike_noise_reference = 0.01  # fallback value if not enough data
        spike_detection_threshold = spike_noise_reference * spike_threshold_value

    return spike_detection_threshold


def compute_calcium_peak_detection_thresholds(
    dec_dff: np.ndarray,
    settings: "AnalysisSettings",
) -> tuple[float, float]:
    """Compute thresholds for peak detection.

    Parameters
    ----------
    dec_dff : np.ndarray
        Deconvolved dF/F trace
    settings : AnalysisSettings
        Analysis settings containing threshold parameters

    Returns
    -------
    tuple[float, float, float]
        - peaks_height_dec_dff: Height threshold for peak detection
        - peaks_prominence_dec_dff: Prominence threshold
    """
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
