from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter1d
from skimage import filters, morphology
from sqlmodel import Session, col, select

from cali.logger import cali_logger
from cali.sqlmodel import FOV, ROI, DataAnalysis, Traces
from cali.sqlmodel._model import FOVAnalysis

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from cali.sqlmodel._model import FOVAnalysis


def create_stimulation_mask(stimulation_file: str) -> np.ndarray:
    """Create a binary mask from an input image.

    We use this to create a mask of the stimulated area. If the input image is a
    mask image already, simply return it.

    Parameters
    ----------
    stimulation_file : str
        Path to the stimulation image.
    """
    # load grayscale image
    blue_img = tifffile.imread(stimulation_file)

    # check if the image is already a binary mask
    unique = np.unique(blue_img)
    # if only pne values which is 1 (full fov illumination)
    if unique.size == 1 and unique[0] == 1:
        return blue_img
    # if only two values which are 0 and 1 (binary mask)
    elif unique.size == 2:
        # if the image is already a binary mask, return it
        if unique[0] == 0 and unique[1] == 1:
            return blue_img
    # apply Gaussian Blur to reduce noise
    blur = filters.gaussian(blue_img, sigma=2)

    # set the threshold to otsu's threshold and apply thresholding
    th = blur > filters.threshold_otsu(blur)

    # morphological operations
    selem_small = morphology.disk(2)
    selem_large = morphology.disk(5)

    # closing operation (removes small holes)
    closed = morphology.closing(th, selem_small)

    # erosion (removes small noise)
    eroded = morphology.erosion(closed, selem_small)

    # final closing with a larger structuring element
    final_mask = morphology.closing(eroded, selem_large)

    return final_mask.astype(np.uint8)


def get_overlap_roi_with_stimulated_area(
    stimulation_mask: np.ndarray, roi_mask: np.ndarray
) -> float:
    """Compute the fraction of the ROI that overlaps with the stimulated area."""
    if roi_mask.shape != stimulation_mask.shape:
        raise ValueError("roi_mask and st_area must have the same dimensions.")

    # count nonzero pixels in the ROI mask
    cell_pixels = np.count_nonzero(roi_mask)

    # if the ROI mask has no pixels, return 0
    if cell_pixels == 0:
        return 0.0

    # count overlapping pixels (logical AND operation)
    overlapping_pixels = np.count_nonzero(roi_mask & stimulation_mask)

    return float(overlapping_pixels / cell_pixels)


# =============================================================================
# Correlation and Synchrony Functions
# =============================================================================


def _get_calcium_peaks_events_from_rois(
    engine: Engine,
    fov_name: str,
    rois: list[int] | None = None,
    run_id: int | None = None,
) -> dict[str, np.ndarray] | None:
    """Extract binary peak event trains from ROI data.

    Args:
        engine: Database engine
        fov_name: Name of the FOV
        rois: List of ROI indices to include, None for all
        run_id: The run ID to filter by, None for latest

    Returns
    -------
        Dictionary mapping ROI names to binary peak event arrays
    """
    with Session(engine) as session:
        roi_data = []  # List of (ROI, Traces, DataAnalysis)

        if run_id is None:
            cali_logger.warning("No run_id provided for peak event extraction.")

        # Optimized query
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, ROI.fov_id == FOV.id)
            .join(
                Traces,
                (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
            )
            .join(
                DataAnalysis,
                (DataAnalysis.roi_id == ROI.id)
                & (DataAnalysis.analysis_result_id == run_id),
            )
            .where(col(FOV.name) == fov_name)
            .where(col(ROI.active) == True)  # noqa: E712
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))

        results = session.exec(stmt).all()
        roi_data = results

    if len(roi_data) < 2:
        return None

    # First pass: determine max_frames from any trace that has data,
    # or from maximum peak frame number
    max_frames = 0
    for _, traces, data_analysis in roi_data:
        if traces and traces.corrected_trace is not None:
            max_frames = max(max_frames, len(traces.corrected_trace))
        if data_analysis and data_analysis.peaks_dec_dff:
            max_peak = max((int(p) for p in data_analysis.peaks_dec_dff), default=0)
            max_frames = max(max_frames, max_peak + 1)

    if max_frames == 0:
        cali_logger.warning(
            f"Cannot determine number of frames for FOV '{fov_name}'. "
            "No trace data or peaks found."
        )
        return None

    peak_trains: dict[str, np.ndarray] = {}

    for roi, traces, data_analysis in roi_data:
        if traces is None or data_analysis is None:
            continue

        peaks_dec_dff = data_analysis.peaks_dec_dff

        if peaks_dec_dff is None or len(peaks_dec_dff) == 0:
            continue

        # Create binary peak event train
        peak_train = np.zeros(max_frames, dtype=np.float32)
        for peak_frame in peaks_dec_dff:
            if 0 <= int(peak_frame) < max_frames:
                peak_train[int(peak_frame)] = 1.0

        if np.sum(peak_train) > 0:  # Only include ROIs with at least one peak
            peak_trains[str(roi.label_value)] = peak_train

    return peak_trains if len(peak_trains) >= 2 else None


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


def _get_calcium_peaks_event_correlations_matrix(
    peak_event_dict: dict[str, list[float]],
    method: str = "correlation",
    jitter_window: int = 2,
    max_lag: int = 5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute pairwise peak event similarity matrix.

    Parameters
    ----------
    peak_event_dict : dict
        Dictionary mapping ROI names to binary peak event arrays
    method : str
        Method to use:
        - "correlation": Zero-lag Pearson correlation
        - "jitter_window": Synchrony with temporal tolerance (±jitter_window)
        - "cross_correlation": Max cross-correlation within ±max_lag
    jitter_window : int
        Tolerance window for peak coincidence (frames),
        used with "jitter_window" method
    max_lag : int
        Maximum lag for cross-correlation method (frames),
        used with "cross_correlation" method

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        (synchrony_matrix, lag_matrix) where:
        - synchrony_matrix: NxN matrix of correlation values
        - lag_matrix: NxN matrix of lag values (only for cross_correlation method,
          otherwise None). Positive lag means ROI_j lags behind ROI_i.
    """
    from cali.util._util import _NUMBA_LOCK

    active_rois = list(peak_event_dict.keys())
    if len(active_rois) < 2:
        return None, None

    try:
        # Convert peak event data into a NumPy array of shape (#ROIs, #Timepoints)
        peak_array = np.array(
            [peak_event_dict[roi] for roi in active_rois], dtype=np.float32
        )
    except ValueError:
        return None, None

    if peak_array.shape[0] < 2:
        return None, None

    n_rois = peak_array.shape[0]
    lag_matrix = None  # Only computed for cross_correlation method

    # Use numba-optimized version for jitter_window method
    if method == "jitter_window":
        with _NUMBA_LOCK:
            synchrony_matrix = _compute_jitter_synchrony_matrix_numba(
                peak_array, jitter_window
            )
    else:
        # Protect cross-correlation with NUMBA_LOCK to prevent thread serialization
        # scipy.signal.correlate can trigger numba/BLAS operations that aren't
        # thread-safe during initial compilation/execution
        with _NUMBA_LOCK:
            # Standard numpy implementation for other methods
            synchrony_matrix = np.zeros((n_rois, n_rois))
            if method == "cross_correlation":
                lag_matrix = np.zeros((n_rois, n_rois), dtype=int)

            for i in range(n_rois):
                for j in range(n_rois):
                    if i == j:
                        synchrony_matrix[i, j] = 1.0  # Perfect self-synchrony
                        if lag_matrix is not None:
                            lag_matrix[i, j] = 0  # Zero lag with self
                    else:
                        events_i = peak_array[i]
                        events_j = peak_array[j]

                        # Handle case where one or both ROIs have no peaks
                        if np.sum(events_i) == 0 or np.sum(events_j) == 0:
                            synchrony_matrix[i, j] = 0.0
                            if lag_matrix is not None:
                                lag_matrix[i, j] = 0
                        else:
                            if method == "cross_correlation":
                                sync_value, lag = _calculate_cross_correlation_with_lag(
                                    events_i, events_j, max_lag
                                )
                                lag_matrix[i, j] = lag  # type: ignore
                            else:
                                # Fallback to original correlation method (default)
                                correlation = np.corrcoef(events_i, events_j)[0, 1]
                                sync_value = (
                                    0.0 if np.isnan(correlation) else abs(correlation)
                                )

                            synchrony_matrix[i, j] = sync_value

    return synchrony_matrix, lag_matrix


def _detect_spike_onsets(spike_trace: np.ndarray) -> np.ndarray:
    """Detect spike onset events from continuous inferred spike traces.

    Converts continuous spike probability/amplitude traces to binary event arrays
    by detecting rising edges (0 -> positive transitions). This matches the
    interpretation used in raster plots where each spike event is a discrete onset.

    Parameters
    ----------
    spike_trace : np.ndarray
        Continuous spike trace (e.g., from CASCADE, OASIS, etc.)

    Returns
    -------
    np.ndarray
        Binary array with 1 at spike onsets, 0 elsewhere
    """
    # Detect positive values
    positive_vals = spike_trace > 0
    # Detect rising edges: 0 -> positive transitions
    rising = positive_vals & ~np.concatenate(([False], positive_vals[:-1]))
    # Create binary array: 1 at rising edges, 0 elsewhere
    binary_events = np.zeros_like(spike_trace, dtype=np.float32)
    binary_events[rising] = 1.0
    return binary_events


def _get_spike_correlations_matrix(
    spike_data_dict: dict[str, list[float]],
    method: str = "correlation",
    jitter_window: int = 2,
    max_lag: int = 5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute pairwise spike similarity matrix.

    Converts continuous inferred spike traces to discrete spike onset events
    (rising edge detection) before computing synchrony, matching the interpretation
    used in raster plots and standard neuroscience practice.

    Parameters
    ----------
    spike_data_dict : dict
        Dictionary mapping ROI names to continuous spike amplitude/probability arrays
        (e.g., from CASCADE, OASIS, or other spike inference methods)
    method : str
        Method to use:
        - "correlation": Zero-lag Pearson correlation on spike onset events
        - "jitter_window": Synchrony with temporal tolerance (±jitter_window)
        - "cross_correlation": Max cross-correlation within ±max_lag
    jitter_window : int
        Tolerance window for spike coincidence (frames),
        used with "jitter_window" method
    max_lag : int
        Maximum lag for cross-correlation method (frames),
        used with "cross_correlation" method

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        (synchrony_matrix, lag_matrix) where:
        - synchrony_matrix: NxN matrix of correlation values between spike onsets
        - lag_matrix: NxN matrix of lag values (only for cross_correlation method,
          otherwise None). Positive lag means ROI_j lags behind ROI_i.

    Notes
    -----
    Spike onsets are detected by identifying rising edges (0 → positive transitions)
    in the continuous spike traces. This ensures that multi-frame spike events are
    counted as single discrete events, not multiple overlapping spikes.
    """
    from cali.util._util import _NUMBA_LOCK

    active_rois = list(spike_data_dict.keys())
    if len(active_rois) < 2:
        return None, None

    try:
        # Convert spike data into a NumPy array of shape (#ROIs, #Timepoints)
        spike_array = np.array(
            [spike_data_dict[roi] for roi in active_rois], dtype=np.float32
        )
    except ValueError:
        return None, None

    if spike_array.shape[0] < 2:
        return None, None

    # Convert continuous spike traces to discrete event arrays
    # using rising edge detection.
    # This matches raster plot interpretation: each spike event = one onset
    binary_spikes = np.zeros_like(spike_array, dtype=np.float32)
    for i in range(spike_array.shape[0]):
        binary_spikes[i] = _detect_spike_onsets(spike_array[i])

    n_rois = binary_spikes.shape[0]
    lag_matrix = None  # Only computed for cross_correlation method

    # Use numba-optimized version for jitter_window method
    if method == "jitter_window":
        with _NUMBA_LOCK:
            synchrony_matrix = _compute_jitter_synchrony_matrix_numba(
                binary_spikes, jitter_window
            )
    else:
        with _NUMBA_LOCK:
            # Standard numpy implementation for other methods
            synchrony_matrix = np.zeros((n_rois, n_rois))
            if method == "cross_correlation":
                lag_matrix = np.zeros((n_rois, n_rois), dtype=int)

            for i in range(n_rois):
                for j in range(n_rois):
                    if i == j:
                        synchrony_matrix[i, j] = 1.0  # Perfect self-synchrony
                        if lag_matrix is not None:
                            lag_matrix[i, j] = 0  # Zero lag with self
                    else:
                        # Calculate correlation between binary spike trains
                        spikes_i = binary_spikes[i]
                        spikes_j = binary_spikes[j]

                        # Handle case where one or both ROIs have no spikes
                        if np.sum(spikes_i) == 0 or np.sum(spikes_j) == 0:
                            synchrony_matrix[i, j] = 0.0
                            if lag_matrix is not None:
                                lag_matrix[i, j] = 0
                        else:
                            if method == "cross_correlation":
                                sync_value, lag = _calculate_cross_correlation_with_lag(
                                    spikes_i, spikes_j, max_lag
                                )
                                lag_matrix[i, j] = lag  # type: ignore
                            else:
                                # Fallback to original correlation method (default)
                                correlation = np.corrcoef(spikes_i, spikes_j)[0, 1]
                                sync_value = (
                                    0.0 if np.isnan(correlation) else abs(correlation)
                                )

                            synchrony_matrix[i, j] = sync_value

    return synchrony_matrix, lag_matrix


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


def _calculate_cross_correlation_with_lag(
    events_i: np.ndarray, events_j: np.ndarray, max_lag: int
) -> tuple[float, int]:
    """Calculate maximum cross-correlation within lag range.

    Computes normalized dot product (cross-correlogram style) at each lag and
    finds the lag with maximum correlation. For binary spike trains, this approach
    is preferred over Pearson correlation because zeros represent meaningful
    information (absence of spikes). Uses numba-optimized implementation for speed.

    Returns
    -------
    tuple[float, int]
        (max_correlation, lag_at_max) where:
        - max_correlation: normalized correlation value in [0, 1]
        - lag_at_max: lag in frames where max occurs.
          Positive means events_j lags behind events_i.
          Negative means events_j leads events_i.
    """
    # Use numba-optimized version for significant speedup
    return _max_cross_correlation_numba(events_i, events_j, max_lag)  # type: ignore


def _calculate_jitter_window_synchrony(
    events_i: np.ndarray, events_j: np.ndarray, jitter_window: int
) -> float:
    """Calculate synchrony allowing for temporal jitter within a window.

    For each peak in ROI i, check if there's a peak in ROI j within ±jitter_window.
    Uses numba JIT compilation for ~10-100x speedup.
    """
    return float(_jitter_window_synchrony_numba(events_i, events_j, jitter_window))


# =============================================================================
# Numba-optimized functions
# =============================================================================

from numba import njit  # noqa: E402


@njit(cache=True, parallel=True)  # type: ignore
def _compute_jitter_synchrony_matrix_numba(
    peak_array: np.ndarray, jitter_window: int
) -> np.ndarray:  # pragma: no cover
    """Numba-optimized computation of full synchrony matrix using jitter window.

    Uses parallel execution for dramatic speedup with many ROIs.
    Expected speedup: 10-100x for 100+ ROIs.
    """
    n_rois = peak_array.shape[0]
    synchrony_matrix = np.zeros((n_rois, n_rois), dtype=np.float64)

    # Parallel loop over ROI pairs
    for i in range(n_rois):
        synchrony_matrix[i, i] = 1.0  # Perfect self-synchrony

        for j in range(i + 1, n_rois):  # Only compute upper triangle
            events_i = peak_array[i]
            events_j = peak_array[j]

            sync_value = _jitter_window_synchrony_numba(
                events_i, events_j, jitter_window
            )

            # Symmetric matrix
            synchrony_matrix[i, j] = sync_value
            synchrony_matrix[j, i] = sync_value

    return synchrony_matrix


@njit(cache=True)  # type: ignore
def _jitter_window_synchrony_numba(
    events_i: np.ndarray, events_j: np.ndarray, jitter_window: int
) -> float:  # pragma: no cover
    """Numba-optimized jitter window synchrony calculation.

    For each peak in ROI i, check if there's a peak in ROI j within ±jitter_window.
    """
    # Extract peak indices
    peaks_i = np.where(events_i > 0)[0]
    peaks_j = np.where(events_j > 0)[0]

    n_peaks_i = len(peaks_i)
    n_peaks_j = len(peaks_j)

    if n_peaks_i == 0 or n_peaks_j == 0:
        return 0.0

    # Count coincident peaks (bidirectional)
    coincidences_i_to_j = 0
    for i in range(n_peaks_i):
        peak_i = peaks_i[i]
        # Check if any peak in j is within jitter window
        for j in range(n_peaks_j):
            if abs(peaks_j[j] - peak_i) <= jitter_window:
                coincidences_i_to_j += 1
                break  # Count each peak_i only once

    coincidences_j_to_i = 0
    for j in range(n_peaks_j):
        peak_j = peaks_j[j]
        # Check if any peak in i is within jitter window
        for i in range(n_peaks_i):
            if abs(peaks_i[i] - peak_j) <= jitter_window:
                coincidences_j_to_i += 1
                break  # Count each peak_j only once

    # Calculate symmetric synchrony measure
    total_peaks = n_peaks_i + n_peaks_j
    total_coincidences = coincidences_i_to_j + coincidences_j_to_i

    return total_coincidences / total_peaks if total_peaks > 0 else 0.0


@njit(cache=True)  # type: ignore
def _max_cross_correlation_numba(
    events_i: np.ndarray, events_j: np.ndarray, max_lag: int
) -> tuple[float, int]:  # pragma: no cover
    """Numba-optimized maximum cross-correlation within lag range.

    Computes normalized dot product (NOT Pearson correlation) at each lag,
    following standard spike train analysis methodology. For binary spike trains,
    zeros represent meaningful information (absence of spikes), so mean-centering
    is inappropriate. This matches cross-correlogram (CCG) analysis used in
    electrophysiology.

    Returns
    -------
    tuple[float, int]
        (max_correlation, lag_at_max) where:
        - max_correlation: normalized correlation value in [0, 1]
        - lag_at_max: lag relative to center where maximum occurs
        Positive lag means events_j lags behind events_i.
    """
    n = len(events_i)

    # Precompute normalizations (without mean centering)
    auto_i = 0.0
    auto_j = 0.0
    for k in range(n):
        auto_i += events_i[k] * events_i[k]
        auto_j += events_j[k] * events_j[k]

    if auto_i == 0.0 or auto_j == 0.0:
        return 0.0, 0

    normalization = np.sqrt(auto_i * auto_j)

    # Compute cross-correlation for each lag
    max_corr = 0.0
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        corr_sum = 0.0

        if lag >= 0:
            # j lags behind i: align i[0:n-lag] with j[lag:n]
            for k in range(n - lag):
                corr_sum += events_i[k] * events_j[k + lag]
        else:
            # j leads i: align i[-lag:n] with j[0:n+lag]
            for k in range(n + lag):
                corr_sum += events_i[k - lag] * events_j[k]

        corr_normalized = corr_sum / normalization

        if corr_normalized > max_corr:
            max_corr = corr_normalized
            best_lag = lag

    # Clip to [0, 1] range to handle numerical errors
    max_corr = min(max(max_corr, 0.0), 1.0)

    return max_corr, best_lag


def _compute_zero_lag_corr_matrix(
    traces: list[np.ndarray],
) -> np.ndarray | None:
    """Compute pairwise zero-lag Pearson correlation matrix for traces.

    Uses z-scored traces and computes standard Pearson correlation coefficient
    at zero lag, following the approach used in CaImAn and standard practice.

    Parameters
    ----------
    traces : list[np.ndarray]
        List of 1D trace arrays (must all be same length)

    Returns
    -------
    np.ndarray | None
        NxN correlation matrix with values in [-1, 1], or None if insufficient data

    Raises
    ------
    ValueError
        If traces have different lengths
    """
    if len(traces) < 2:
        return None

    # Verify all traces have same length
    lengths = [len(t) for t in traces]
    if len(set(lengths)) > 1:
        raise ValueError(
            f"All traces must have same length. Got lengths: {set(lengths)}"
        )

    # Stack traces
    traces_array = np.vstack(traces)  # (n_rois, n_frames)

    # Manually z-score to handle constant traces without warnings
    # Z-score: (x - mean(x)) / std(x)
    means = traces_array.mean(axis=1, keepdims=True)
    stds = traces_array.std(axis=1, keepdims=True, ddof=1)

    # Replace zero std (constant traces) with 1 to avoid division by zero
    # This will make constant traces have zero mean and all zeros after normalization
    stds[stds == 0] = 1.0

    dff_zero_mean = (traces_array - means) / stds

    n_rois = len(traces)
    correlation_matrix = np.zeros((n_rois, n_rois), dtype=float)

    # Compute norms for normalization
    norms = np.linalg.norm(dff_zero_mean, axis=1)
    # Avoid division by zero (constant trace after z-scoring)
    norms[norms == 0] = np.finfo(float).eps

    # Diagonal is always 1 (perfect self-correlation)
    np.fill_diagonal(correlation_matrix, 1.0)

    # Compute zero-lag Pearson correlation for all pairs
    for i in range(n_rois):
        x = dff_zero_mean[i]
        for j in range(i + 1, n_rois):
            y = dff_zero_mean[j]

            # Pearson correlation at zero lag: r = <x, y> / (||x|| ||y||)
            # After z-scoring, this is just the normalized dot product
            r0 = np.dot(x, y) / (norms[i] * norms[j])

            # Clamp to [-1, 1] to handle numerical errors
            r0 = np.clip(r0, -1.0, 1.0)

            correlation_matrix[i, j] = r0
            correlation_matrix[j, i] = r0

    return correlation_matrix


def _detect_spikes_population_bursts(
    spike_trains: list[np.ndarray],
    frame_rate: float,
    burst_threshold_percent: float,
    min_duration_ms: float,
    gaussian_sigma_sec: float,
) -> tuple[
    int,
    float | None,
    float | None,
    list[int],
    list[int],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Detect bursts in population spike activity.

    Computes mean population activity (fraction of active ROIs), smooths it,
    and detects periods above threshold that exceed minimum duration.

    Parameters
    ----------
    spike_trains : list[np.ndarray]
        List of binary spike trains for active ROIs (0/1 per frame)
    frame_rate : float
        Frame rate in Hz (frames per second)
    burst_threshold_percent : float
        Threshold as percentage of maximal activity (0-1 scale).
        For spikes, the population activity is already in [0,1], so a
        value of 65.0 corresponds to a threshold of 0.65 (65% of ROIs active).
    min_duration_ms : float
        Minimum burst duration in milliseconds
    gaussian_sigma_sec : float
        Gaussian smoothing sigma in seconds

    Returns
    -------
    tuple
        Seven-element tuple containing:
        - burst_count (int): Number of bursts detected
        - burst_avg_duration (float | None): Average burst duration in seconds
        - burst_avg_interval (float | None): Average inter-burst interval
        - burst_starts (list[int]): Frame indices where bursts start
        - burst_ends (list[int]): Frame indices where bursts end (exclusive)
        - population_activity (np.ndarray | None): Raw mean population activity
          (fraction of active ROIs, in [0,1])
        - smoothed_activity (np.ndarray | None): Smoothed population activity
    """
    if len(spike_trains) < 2:
        return 0, None, None, [], [], None, None

    # Stack spike trains and compute population activity (mean across ROIs)
    spike_array = np.vstack(spike_trains)  # (n_rois, n_frames), values 0 or 1
    population_activity = np.mean(spike_array, axis=0)  # (n_frames,), in [0,1]

    if population_activity.size == 0:
        return 0, None, None, [], [], None, None

    # Convert parameters to frame units
    min_duration_frames = max(1, int((min_duration_ms / 1000.0) * frame_rate))
    gaussian_sigma_frames = gaussian_sigma_sec * frame_rate

    # Smooth population activity
    if gaussian_sigma_frames > 0:
        smoothed_activity = gaussian_filter1d(
            population_activity, sigma=gaussian_sigma_frames, mode="nearest"
        )
    else:
        smoothed_activity = population_activity

    # If the signal is essentially flat, no bursts
    max_val = float(np.max(smoothed_activity))
    if max_val < np.finfo(float).eps:
        return 0, None, None, [], [], population_activity, smoothed_activity

    # Threshold in the same [0,1] units as smoothed_activity
    # e.g. burst_threshold_percent = 65 -> threshold_value = 0.65
    burst_threshold_value = burst_threshold_percent / 100.0

    # Detect regions above threshold
    above_threshold = smoothed_activity > burst_threshold_value
    if not np.any(above_threshold):
        return (
            0,
            None,
            None,
            [],
            [],
            population_activity,
            smoothed_activity,
        )

    # Find burst start and end points
    above_int = above_threshold.astype(int)
    changes = np.diff(above_int)

    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases (burst at beginning or end)
    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(above_threshold))

    # Filter bursts by minimum duration
    burst_starts_list: list[int] = []
    burst_ends_list: list[int] = []
    burst_durations_sec: list[float] = []

    for start_idx, end_idx in zip(starts, ends):
        duration_frames = end_idx - start_idx
        if duration_frames >= min_duration_frames:
            burst_starts_list.append(int(start_idx))
            burst_ends_list.append(int(end_idx))
            duration_sec = duration_frames / frame_rate
            burst_durations_sec.append(duration_sec)

    burst_count = len(burst_durations_sec)

    if burst_count == 0:
        return (
            0,
            None,
            None,
            [],
            [],
            population_activity,
            smoothed_activity,
        )

    # Average duration (seconds)
    burst_avg_duration = float(np.mean(burst_durations_sec))

    # Average inter-burst interval (seconds)
    burst_avg_interval: float | None = None
    if burst_count >= 2:
        intervals_sec: list[float] = []
        for i in range(1, burst_count):
            interval_frames = burst_starts_list[i] - burst_ends_list[i - 1]
            interval_sec = interval_frames / frame_rate
            intervals_sec.append(interval_sec)
        burst_avg_interval = float(np.mean(intervals_sec))

    return (
        burst_count,
        burst_avg_duration,
        burst_avg_interval,
        burst_starts_list,
        burst_ends_list,
        population_activity,  # raw fraction of active ROIs
        smoothed_activity,  # smoothed fraction trace
    )


def _detect_calcium_population_bursts(
    dec_dff_traces: list[np.ndarray],
    frame_rate: float,
    burst_threshold_percent: float,
    min_duration_ms: float,
    gaussian_sigma_sec: float,
) -> tuple[
    int,
    float | None,
    float | None,
    list[int],
    list[int],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Detect bursts in population calcium activity (deconvolved DF/F).

    Burst detection is done directly on the mean deconvolved DF/F trace
    (optionally smoothed), without explicit normalization. The threshold is
    interpreted as a percentage of the maximum smoothed population activity.

    Parameters
    ----------
    dec_dff_traces : list[np.ndarray]
        List of deconvolved DF/F traces for active ROIs
    frame_rate : float
        Frame rate in Hz (frames per second)
    burst_threshold_percent : float
        Threshold as percentage of the maximum smoothed population activity
        (e.g., 65.0 means 0.65 * max(smoothed_activity)).
    min_duration_ms : float
        Minimum burst duration in milliseconds
    gaussian_sigma_sec : float
        Gaussian smoothing sigma in seconds

    Returns
    -------
    tuple
        Seven-element tuple containing:
        - burst_count (int): Number of bursts detected
        - burst_avg_duration (float | None): Average burst duration in seconds
        - burst_avg_interval (float | None): Average inter-burst interval
        - burst_starts (list[int]): Frame indices where bursts start
        - burst_ends (list[int]): Frame indices where bursts end (exclusive)
        - population_activity (np.ndarray | None): Raw mean population activity
        - smoothed_activity (np.ndarray | None): Smoothed population activity
    """
    if len(dec_dff_traces) < 2:
        return 0, None, None, [], [], None, None

    # Stack traces and compute population activity (mean across ROIs)
    traces_array = np.vstack(dec_dff_traces)  # (n_rois, n_frames)
    population_activity = np.mean(traces_array, axis=0)  # raw mean (n_frames,)

    if population_activity.size == 0:
        return 0, None, None, [], [], None, None

    # Convert parameters to frame units
    min_duration_frames = max(1, int((min_duration_ms / 1000.0) * frame_rate))
    gaussian_sigma_frames = gaussian_sigma_sec * frame_rate

    # Smooth population activity
    if gaussian_sigma_frames > 0:
        smoothed_activity = gaussian_filter1d(
            population_activity, sigma=gaussian_sigma_frames, mode="nearest"
        )
    else:
        smoothed_activity = population_activity

    # If the signal is essentially flat, no bursts
    max_val = float(np.max(smoothed_activity))
    if max_val < np.finfo(float).eps:
        return 0, None, None, [], [], population_activity, smoothed_activity

    # Threshold in the SAME UNITS as smoothed_activity
    # e.g. burst_threshold_percent = 65 → 0.65 * max(smoothed_activity)
    burst_threshold_value = (burst_threshold_percent / 100.0) * max_val

    # Detect regions above threshold
    above_threshold = smoothed_activity > burst_threshold_value
    if not np.any(above_threshold):
        return (
            0,
            None,
            None,
            [],
            [],
            population_activity,
            smoothed_activity,
        )

    # Find burst start and end points
    above_int = above_threshold.astype(int)
    changes = np.diff(above_int)

    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # Handle edge cases (burst at beginning or end)
    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(above_threshold))

    # Filter bursts by minimum duration
    burst_starts_list: list[int] = []
    burst_ends_list: list[int] = []
    burst_durations_sec: list[float] = []

    for start_idx, end_idx in zip(starts, ends):
        duration_frames = end_idx - start_idx
        if duration_frames >= min_duration_frames:
            burst_starts_list.append(int(start_idx))
            burst_ends_list.append(int(end_idx))
            duration_sec = duration_frames / frame_rate
            burst_durations_sec.append(duration_sec)

    burst_count = len(burst_durations_sec)

    if burst_count == 0:
        return (
            0,
            None,
            None,
            [],
            [],
            population_activity,
            smoothed_activity,
        )

    # Average duration (seconds)
    burst_avg_duration = float(np.mean(burst_durations_sec))

    # Average inter-burst interval (seconds)
    burst_avg_interval: float | None = None
    if burst_count >= 2:
        intervals_sec: list[float] = []
        for i in range(1, burst_count):
            interval_frames = burst_starts_list[i] - burst_ends_list[i - 1]
            interval_sec = interval_frames / frame_rate
            intervals_sec.append(interval_sec)
        burst_avg_interval = float(np.mean(intervals_sec))

    return (
        burst_count,
        burst_avg_duration,
        burst_avg_interval,
        burst_starts_list,
        burst_ends_list,
        population_activity,  # raw mean ΔF/F
        smoothed_activity,  # smoothed ΔF/F used for detection
    )


ConnectivityMethod = Literal[
    # DF/F traces
    "calcium_dff_corr",  # 0. Zero-lag Pearson on DF/F
    "calcium_dec_dff_corr",  # 1. Zero-lag Pearson on deconvolved DF/F (default)
    # Calcium peaks
    "calcium_peaks_maxlag",  # 3. Max-lag correlation on calcium peaks
    "calcium_peaks_jitter",  # 4. Jitter synchrony on calcium peaks
    # Inferred spikes
    "spike_corr",  # 5. Zero-lag Pearson on spike trains
    "spike_maxlag",  # 6. Max-lag correlation (CCG) on spikes
    "spike_jitter",  # 7. Jitter synchrony on spikes
]


def _compute_connectivity_metrics(
    fov_analysis: FOVAnalysis,
    method: ConnectivityMethod = "calcium_dec_dff_corr",
    threshold: float = 0.9,
    use_absolute_for_corr: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Build a connectivity graph (adjacency + weights) from FOVAnalysis metrics.

    Parameters
    ----------
    fov_analysis : FOVAnalysis
        Object containing FOV-wide matrices and ROI labels.
        Uses:
        - active_roi_labels
        - calcium_dff_correlation_matrix
        - calcium_dec_dff_corr_matrix
        - spike_max_lag_correlation_matrix
        - calcium_peaks_jitter_synchrony_matrix
        - spike_correlation_matrix
        - spike_max_lag_correlation_matrix
        - spike_jitter_synchrony_matrix
    method : ConnectivityMethod, default "calcium_dec_dff_corr"
        Which metric to use:

        DF/F calcium traces
        -------------------
        - "calcium_dff_corr":
              zero-lag Pearson correlation on DF/F traces (metric 0)
        - "calcium_dec_dff_corr" (default):
              zero-lag Pearson on deconvolved DF/F traces (metric 1)

        Calcium peaks
        -------------
        - "calcium_peaks_maxlag":
              max-lag correlation on calcium peak events (metric 3)
        - "calcium_peaks_jitter":
              jitter synchrony on calcium peak events (metric 4)

        Inferred spikes
        ---------------
        - "spike_corr":
              zero-lag Pearson on spike trains (metric 5)
        - "spike_maxlag":
              max-lag CCG-like correlation on spike events (metric 6)
        - "spike_jitter":
              jitter synchrony on spike events (metric 7)

    threshold : float, default 0.9
        Threshold applied to the chosen metric to create edges.

        For correlation-like metrics (Pearson & max-lag):
            if use_absolute_for_corr is True:
                edge if |value_ij| >= threshold
            else:
                edge if value_ij >= threshold

        For jitter synchrony metrics (in [0, 1]):
            edge if sync_ij >= threshold

    use_absolute_for_corr : bool, default True
        If True, strong negative correlations also become edges (|r| >= threshold).
        If False, only strong positive correlations are kept (r >= threshold).

    Returns
    -------
    adjacency : np.ndarray
        Binary adjacency matrix of shape (N, N), 1 = connection, 0 = no connection.
        Diagonal is always 0 (no self-connections).
    weights : np.ndarray
        Underlying metric values (same shape as adjacency).
    roi_labels : list[int]
        ROI labels corresponding to rows/columns of adjacency/weights.

    Raises
    ------
    ValueError
        If the requested metric is not available or shapes are inconsistent.
    """
    roi_labels = fov_analysis.active_roi_labels or []
    if not roi_labels:
        raise ValueError("FOVAnalysis.active_roi_labels is empty or None.")

    n = len(roi_labels)

    # ------------------------------------------------------------------
    # Select metric matrix based on method
    # ------------------------------------------------------------------
    if method == "calcium_dff_corr":
        metric = fov_analysis.calcium_dff_correlation_matrix
        is_correlation = True

    elif method == "calcium_dec_dff_corr":
        metric = fov_analysis.calcium_dec_dff_corr_matrix
        is_correlation = True

    elif method == "calcium_peaks_maxlag":
        metric = fov_analysis.spike_max_lag_correlation_matrix
        is_correlation = True

    elif method == "calcium_peaks_jitter":
        metric = fov_analysis.calcium_peaks_jitter_synchrony_matrix
        is_correlation = False

    elif method == "spike_maxlag":
        metric = fov_analysis.spike_max_lag_correlation_matrix
        is_correlation = True

    elif method == "spike_jitter":
        metric = fov_analysis.spike_jitter_synchrony_matrix
        is_correlation = False

    else:
        raise ValueError(f"Unknown connectivity method: {method!r}")

    if metric is None:
        raise ValueError(
            f"Requested metric {method!r} is None on FOVAnalysis. "
            "Make sure FOV-level analysis was computed for this method."
        )

    weights = np.asarray(metric, dtype=float)
    if weights.shape != (n, n):
        raise ValueError(
            f"Metric matrix shape {weights.shape} does not match number of "
            f"ROI labels ({n})."
        )

    # ------------------------------------------------------------------
    # Threshold → adjacency
    # ------------------------------------------------------------------
    if is_correlation:
        if use_absolute_for_corr:
            values = np.abs(weights)
        else:
            values = weights
    else:
        # Jitter synchrony already in [0, 1]
        values = weights

    adjacency = (values >= threshold).astype(int)

    # Remove self-connections
    np.fill_diagonal(adjacency, 0)

    return adjacency, weights, list(roi_labels)
