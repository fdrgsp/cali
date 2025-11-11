import re
from typing import Callable

import numpy as np

from cali._constants import MAX_FRAMES_AFTER_STIMULATION, MWCM
from cali.logger import cali_logger
from cali.sqlmodel._util import ROIData


def equation_from_str(equation: str) -> Callable | None:
    """Parse various equation formats and return a callable function.

    Supported formats:
    - Linear: y = m*x + q  (e.g. "y = 2*x + 3")
    - Quadratic: y = a*x^2 + b*x + c  (e.g. "y = 0.5*x^2 + 2*x + 1")
    - Exponential: y = a*exp(b*x) + c  (e.g. "y = 2*exp(0.1*x) + 1")
    - Power: y = a*x^b + c  (e.g. "y = 2*x^0.5 + 1")
    - Logarithmic: y = a*log(x) + b  (e.g. "y = 2*log(x) + 1")
    """
    if not equation:
        return None

    # Remove all whitespace for easier parsing
    eq = equation.replace(" ", "").lower()

    try:
        if linear_match := re.match(r"y=([+-]?\d*\.?\d+)\*x([+-]\d*\.?\d+)", eq):
            m = float(linear_match[1])
            q = float(linear_match[2])
            return lambda x: m * x + q

        if quad_match := re.match(
            r"y=([+-]?\d*\.?\d+)\*x\^2([+-]\d*\.?\d+)\*x([+-]\d*\.?\d+)", eq
        ):
            a = float(quad_match[1])
            b = float(quad_match[2])
            c = float(quad_match[3])
            return lambda x: a * x**2 + b * x + c

        if exp_match := re.match(
            r"y=([+-]?\d*\.?\d+)\*exp\(([+-]?\d*\.?\d+)\*x\)([+-]\d*\.?\d+)",
            eq,
        ):
            a = float(exp_match[1])
            b = float(exp_match[2])
            c = float(exp_match[3])
            return lambda x: a * np.exp(b * x) + c

        if power_match := re.match(
            r"y=([+-]?\d*\.?\d+)\*x\^([+-]?\d*\.?\d+)([+-]\d*\.?\d+)", eq
        ):
            a = float(power_match[1])
            b = float(power_match[2])
            c = float(power_match[3])
            return lambda x: a * (x**b) + c

        if log_match := re.match(r"y=([+-]?\d*\.?\d+)\*log\(x\)([+-]\d*\.?\d+)", eq):
            a = float(log_match[1])
            b = float(log_match[2])
            return lambda x: a * np.log(x) + b

        # If no pattern matches, show error
        msg = (
            "Invalid equation format! Using values from the metadata.\n"
            "Only Linear, Quadratic, Exponential, Power, and Logarithmic equations "
            "are supported."
        )
        cali_logger.error(msg)
        return None

    except ValueError as e:
        msg = (
            f"Error parsing equation coefficients: {e}\nUsing values from the metadata."
        )
        cali_logger.error(msg)
        return None


def _get_calcium_peaks_event_synchrony(
    peak_event_synchrony_matrix: np.ndarray | None,
) -> float | None:
    """Calculate global peak event synchrony score from a peak event synchrony matrix.

    This function reuses the same approach as spike synchrony.
    """
    if peak_event_synchrony_matrix is None or peak_event_synchrony_matrix.size == 0:
        return None
    # Ensure the matrix is at least 2x2 and square
    if (
        peak_event_synchrony_matrix.shape[0] < 2
        or peak_event_synchrony_matrix.shape[0] != peak_event_synchrony_matrix.shape[1]
    ):
        return None

    # Calculate the sum of each row, excluding the diagonal
    n_rois = peak_event_synchrony_matrix.shape[0]
    off_diagonal_sum = np.sum(peak_event_synchrony_matrix, axis=1) - np.diag(
        peak_event_synchrony_matrix
    )

    # Normalize by the number of off-diagonal elements per row
    mean_synchrony_per_roi = off_diagonal_sum / (n_rois - 1)

    # Return the median synchrony across all ROIs
    return float(np.median(mean_synchrony_per_roi))


def _get_calcium_peaks_events_from_rois(
    roi_data_dict: dict[str, ROIData],
    rois: list[int] | None = None,
) -> dict[str, np.ndarray] | None:
    """Extract binary peak event trains from ROI data.

    Args:
        roi_data_dict: Dictionary of ROI data
        rois: List of ROI indices to include, None for all

    Returns
    -------
        Dictionary mapping ROI names to binary peak event arrays
    """
    peak_trains: dict[str, np.ndarray] = {}

    if rois is None:
        rois = [int(roi) for roi in roi_data_dict if roi.isdigit()]

    if len(rois) < 2:
        return None

    max_frames = 0
    for roi_key, roi_data in roi_data_dict.items():
        try:
            roi_id = int(roi_key)
            if roi_id not in rois or not roi_data.active:
                continue
        except ValueError:
            continue

        max_frames = len(roi_data.corrected_trace) if roi_data.corrected_trace else 0
        if max_frames == 0:
            return None

        if (
            roi_data.dec_dff
            and roi_data.peaks_dec_dff
            and len(roi_data.peaks_dec_dff) > 0
        ):
            # Create binary peak event train
            peak_train = np.zeros(max_frames, dtype=np.float32)
            for peak_frame in roi_data.peaks_dec_dff:
                if 0 <= int(peak_frame) < max_frames:
                    peak_train[int(peak_frame)] = 1.0

            if np.sum(peak_train) > 0:  # Only include ROIs with at least one peak
                peak_trains[roi_key] = peak_train

    return peak_trains if len(peak_trains) >= 2 else None


def _get_calcium_peaks_event_synchrony_matrix(
    peak_event_dict: dict[str, list[float]],
    method: str = "correlation",
    jitter_window: int = 2,
    max_lag: int = 5,
) -> np.ndarray | None:
    """Compute pairwise peak event synchrony using robust methods.

    Handles timing jitter better than simple correlation.

    Parameters
    ----------
    peak_event_dict : dict
        Dictionary mapping ROI names to binary peak event arrays
    method : str
        Method to use - "jitter_window", "cross_correlation", or "correlation"
    jitter_window : int
        Tolerance window for peak coincidence (frames)
    max_lag : int
        Maximum lag for cross-correlation method (frames)

    Returns
    -------
    np.ndarray or None
        Synchrony matrix robust to small temporal shifts
    """
    active_rois = list(peak_event_dict.keys())
    if len(active_rois) < 2:
        return None

    try:
        # Convert peak event data into a NumPy array of shape (#ROIs, #Timepoints)
        peak_array = np.array(
            [peak_event_dict[roi] for roi in active_rois], dtype=np.float32
        )
    except ValueError:
        return None

    if peak_array.shape[0] < 2:
        return None

    n_rois = peak_array.shape[0]
    synchrony_matrix = np.zeros((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(n_rois):
            if i == j:
                synchrony_matrix[i, j] = 1.0  # Perfect self-synchrony
            else:
                events_i = peak_array[i]
                events_j = peak_array[j]

                # Handle case where one or both ROIs have no peaks
                if np.sum(events_i) == 0 or np.sum(events_j) == 0:
                    synchrony_matrix[i, j] = 0.0
                else:
                    if method == "jitter_window":
                        sync_value = _calculate_jitter_window_synchrony(
                            events_i, events_j, jitter_window
                        )
                    elif method == "cross_correlation":
                        sync_value = _calculate_cross_correlation_synchrony(
                            events_i, events_j, max_lag
                        )
                    else:
                        # Fallback to original correlation method (default)
                        correlation = np.corrcoef(events_i, events_j)[0, 1]
                        sync_value = 0.0 if np.isnan(correlation) else abs(correlation)

                    synchrony_matrix[i, j] = sync_value

    return synchrony_matrix


def _calculate_jitter_window_synchrony(
    events_i: np.ndarray, events_j: np.ndarray, jitter_window: int
) -> float:
    """Calculate synchrony allowing for temporal jitter within a window.

    For each peak in ROI i, check if there's a peak in ROI j within ±jitter_window.
    """
    peaks_i = np.where(events_i > 0)[0]
    peaks_j = np.where(events_j > 0)[0]

    if len(peaks_i) == 0 or len(peaks_j) == 0:
        return 0.0

    # Count coincident peaks (bidirectional)
    coincidences_i_to_j = 0
    for peak_i in peaks_i:
        # Check if any peak in j is within jitter window of peak_i
        distances = np.abs(peaks_j - peak_i)
        if np.any(distances <= jitter_window):
            coincidences_i_to_j += 1

    coincidences_j_to_i = 0
    for peak_j in peaks_j:
        # Check if any peak in i is within jitter window of peak_j
        distances = np.abs(peaks_i - peak_j)
        if np.any(distances <= jitter_window):
            coincidences_j_to_i += 1

    # Calculate symmetric synchrony measure
    total_peaks = len(peaks_i) + len(peaks_j)
    total_coincidences = coincidences_i_to_j + coincidences_j_to_i

    return total_coincidences / total_peaks if total_peaks > 0 else 0.0


def get_stimulated_amplitudes_from_roi_data(
    roi_data: ROIData,
    led_power_equation: Callable | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Get stimulated and non-stimulated amplitudes from ROIData on-demand.

    Args:
        roi_data: ROIData object containing the necessary data
        led_power_equation: Optional function to convert power percentage to mW/cm²

    Returns
    -------
        Tuple of (amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks)
    """
    if (
        not roi_data.evoked_experiment
        or roi_data.dec_dff is None
        or roi_data.peaks_dec_dff is None
        or roi_data.stimulations_frames_and_powers is None
    ):
        return {}, {}

    return separate_stimulated_vs_non_stimulated_peaks(
        dec_dff=np.array(roi_data.dec_dff),
        peaks_dec_dff=np.array(roi_data.peaks_dec_dff),
        pulse_on_frames_and_powers=roi_data.stimulations_frames_and_powers,
        is_roi_stimulated=roi_data.stimulated,
        led_pulse_duration=roi_data.led_pulse_duration or "unknown",
        led_power_equation=led_power_equation,
    )


def separate_stimulated_vs_non_stimulated_peaks(
    dec_dff: np.ndarray,
    peaks_dec_dff: np.ndarray,
    pulse_on_frames_and_powers: dict[str, int],
    is_roi_stimulated: bool,
    led_pulse_duration: str = "unknown",
    led_power_equation: Callable | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Separate peak amplitudes into stimulated and non-stimulated categories.

    Args:
        dec_dff: Deconvolved dF/F signal
        peaks_dec_dff: Array of peak indices
        pulse_on_frames_and_powers: Dict mapping frame numbers to power values
        is_roi_stimulated: Whether this ROI is in a stimulated area
        led_pulse_duration: Duration of LED pulse (for labeling)
        led_power_equation: Optional function to convert power percentage to mW/cm²

    Returns
    -------
        Tuple of (amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks)
        Each is a dict mapping power_duration strings to lists of amplitudes
    """
    import bisect

    amplitudes_stimulated_peaks: dict[str, list[float]] = {}
    amplitudes_non_stimulated_peaks: dict[str, list[float]] = {}

    sorted_peaks_dec_dff = sorted(peaks_dec_dff)

    for frame, power in pulse_on_frames_and_powers.items():
        stim_frame = int(frame)
        # Find index of first peak >= stim_frame
        i = bisect.bisect_left(sorted_peaks_dec_dff, stim_frame)

        # Check if index is valid
        if i >= len(sorted_peaks_dec_dff):
            continue

        peak_idx = sorted_peaks_dec_dff[i]

        # Check if peak is within stimulation window
        if (
            peak_idx >= stim_frame
            and peak_idx <= stim_frame + MAX_FRAMES_AFTER_STIMULATION
        ):
            amplitude = float(dec_dff[peak_idx])

            # Format power value
            if led_power_equation is not None:
                power_val = led_power_equation(power)
                power_str = f"{power_val:.3f}{MWCM}"
            else:
                power_str = f"{power}%"

            # Create column key
            col = f"{power_str}_{led_pulse_duration}"

            # Categorize based on stimulation status
            if is_roi_stimulated:
                amplitudes_stimulated_peaks.setdefault(col, []).append(amplitude)
            else:
                amplitudes_non_stimulated_peaks.setdefault(col, []).append(amplitude)

    return amplitudes_stimulated_peaks, amplitudes_non_stimulated_peaks


def _get_spikes_over_threshold(
    roi_data: ROIData, raw: bool = False
) -> list[float] | None:
    """Get spikes over threshold from ROI data."""
    if not roi_data.inferred_spikes or roi_data.inferred_spikes_threshold is None:
        return None
    if raw:
        # Return raw inferred spikes
        return roi_data.inferred_spikes
    spikes_thresholded = []
    for spike in roi_data.inferred_spikes:
        if spike > roi_data.inferred_spikes_threshold:
            spikes_thresholded.append(spike)
        else:
            spikes_thresholded.append(0.0)
    return spikes_thresholded


def _get_spike_synchrony_matrix(
    spike_data_dict: dict[str, list[float]],
    method: str = "correlation",
    jitter_window: int = 2,
    max_lag: int = 5,
) -> np.ndarray | None:
    """Compute pairwise spike synchrony from spike amplitude data.

    Parameters
    ----------
    spike_data_dict : dict
        Dictionary mapping ROI names to spike amplitude arrays
    method : str
        Method to use - "jitter_window", "cross_correlation", or "correlation"
    jitter_window : int
        Tolerance window for spike coincidence (frames)
    max_lag : int
        Maximum lag for cross-correlation method (frames)

    Returns
    -------
    np.ndarray or None
        Synchrony matrix robust to small temporal shifts
    """
    active_rois = list(spike_data_dict.keys())
    if len(active_rois) < 2:
        return None

    try:
        # Convert spike data into a NumPy array of shape (#ROIs, #Timepoints)
        spike_array = np.array(
            [spike_data_dict[roi] for roi in active_rois], dtype=np.float32
        )
    except ValueError:
        return None

    if spike_array.shape[0] < 2:
        return None

    # Create binary spike matrices (1 where spike > 0, 0 otherwise)
    binary_spikes = (spike_array > 0).astype(np.float32)

    # Calculate pairwise synchrony using correlation of binary spike trains
    n_rois = binary_spikes.shape[0]
    synchrony_matrix = np.zeros((n_rois, n_rois))

    for i in range(n_rois):
        for j in range(n_rois):
            if i == j:
                synchrony_matrix[i, j] = 1.0  # Perfect self-synchrony
            else:
                # Calculate correlation between binary spike trains
                spikes_i = binary_spikes[i]
                spikes_j = binary_spikes[j]

                # Handle case where one or both ROIs have no spikes
                if np.sum(spikes_i) == 0 or np.sum(spikes_j) == 0:
                    synchrony_matrix[i, j] = 0.0
                else:
                    if method == "jitter_window":
                        sync_value = _calculate_jitter_window_synchrony(
                            spikes_i, spikes_j, jitter_window
                        )
                    elif method == "cross_correlation":
                        sync_value = _calculate_cross_correlation_synchrony(
                            spikes_i, spikes_j, max_lag
                        )
                    else:
                        # Fallback to original correlation method (default)
                        correlation = np.corrcoef(spikes_i, spikes_j)[0, 1]
                        sync_value = 0.0 if np.isnan(correlation) else abs(correlation)

                    synchrony_matrix[i, j] = sync_value

    return synchrony_matrix


def _get_spike_synchrony(spike_synchrony_matrix: np.ndarray | None) -> float | None:
    """Calculate global spike synchrony score from a spike synchrony matrix."""
    if spike_synchrony_matrix is None or spike_synchrony_matrix.size == 0:
        return None
    # Ensure the matrix is at least 2x2 and square
    if (
        spike_synchrony_matrix.shape[0] < 2
        or spike_synchrony_matrix.shape[0] != spike_synchrony_matrix.shape[1]
    ):
        return None

    # Calculate the sum of each row, excluding the diagonal
    n_rois = spike_synchrony_matrix.shape[0]
    off_diagonal_sum = np.sum(spike_synchrony_matrix, axis=1) - np.diag(
        spike_synchrony_matrix
    )

    # Normalize by the number of off-diagonal elements per row
    mean_synchrony_per_roi = off_diagonal_sum / (n_rois - 1)

    # Return the median synchrony across all ROIs
    return float(np.median(mean_synchrony_per_roi))


def _calculate_cross_correlation_synchrony(
    events_i: np.ndarray, events_j: np.ndarray, max_lag: int
) -> float:
    """Calculate synchrony using maximum cross-correlation within lag range."""
    from scipy.signal import correlate

    # Cross-correlation
    xcorr = correlate(events_i, events_j, mode="full")

    # Get the center (zero-lag) position
    center = len(events_i) - 1

    # Extract correlations within max_lag range
    start_idx = max(0, center - max_lag)
    end_idx = min(len(xcorr), center + max_lag + 1)

    local_xcorr = xcorr[start_idx:end_idx]

    # Normalize by the geometric mean of autocorrelations
    auto_i = np.sum(events_i * events_i)
    auto_j = np.sum(events_j * events_j)

    if auto_i > 0 and auto_j > 0:
        normalization = np.sqrt(auto_i * auto_j)
        max_correlation = np.max(local_xcorr) / normalization
        return float(np.clip(max_correlation, 0, 1))
    else:
        return 0.0
