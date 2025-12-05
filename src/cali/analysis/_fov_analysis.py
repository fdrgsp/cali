"""FOV-level analysis functions for computing correlation and synchrony matrices.

This module provides functions to compute pairwise correlation and synchrony
matrices across all active ROIs in a FOV. These metrics are computed once
during analysis and stored in the FOVAnalysis table for efficient retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cali.logger import cali_logger
from cali.plot._util import (
    _get_calcium_peaks_event_synchrony,
    _get_calcium_peaks_event_synchrony_matrix,
    _get_spike_synchrony,
    _get_spike_synchrony_matrix,
)
from cali.sqlmodel._model import FOVAnalysis

if TYPE_CHECKING:
    from cali.sqlmodel._model import FOV, ROI, AnalysisSettings


def compute_fov_analysis(
    fov: FOV,
    analysis_settings: AnalysisSettings,
) -> FOVAnalysis | None:
    """Compute FOV-level correlation and synchrony analysis.

    This function calculates 6 pairwise metrics for all active ROIs in a FOV:

    Calcium Peaks Metrics:
    1. Zero-lag Pearson correlation on DF/F traces
    2. Jitter synchrony on calcium peak events
    3. Max lag correlation on calcium peak events

    Spike Metrics:
    4. Zero-lag Pearson correlation on spike trains
    5. Max lag correlation on spike events
    6. Jitter synchrony on spike events

    It requires that ROIs have traces and data_analysis attached
    (either via history or _new_* attributes).

    Parameters
    ----------
    fov : FOV
        FOV containing ROIs with traces and analysis data
    analysis_settings : AnalysisSettings
        Settings containing jitter_window and max_lag parameters

    Returns
    -------
    FOVAnalysis | None
        FOVAnalysis object with computed matrices, or None if insufficient data
    """
    # Collect active ROIs with their traces and analysis data
    active_rois: list[ROI] = []
    for roi in fov.rois:
        if not roi.active:
            continue
        active_rois.append(roi)

    if len(active_rois) < 2:
        cali_logger.debug(
            f"FOV {fov.name}: Not enough active ROIs ({len(active_rois)}) "
            "for correlation/synchrony analysis. Need at least 2."
        )
        return None

    # Get traces and analysis data for active ROIs
    # Use _new_traces/_new_data_analysis if available (during extraction/analysis)
    # Otherwise fall back to traces_history/data_analysis_history
    roi_labels: list[int] = []
    dec_dff_traces: list[np.ndarray] = []
    spike_trains: list[np.ndarray] = []
    peak_events_dict: dict[str, list[float]] = {}
    spike_data_dict: dict[str, list[float]] = {}

    for roi in active_rois:
        if roi.label_value is None:
            continue

        # Get traces - prefer _new_traces if available
        traces = None
        if hasattr(roi, "_new_traces") and roi._new_traces:
            traces = roi._new_traces[-1]  # Most recent
        elif roi.traces_history:
            traces = roi.traces_history[-1]

        if traces is None or traces.dec_dff is None:
            continue

        # Get analysis data - prefer _new_data_analysis if available
        data_analysis = None
        if hasattr(roi, "_new_data_analysis") and roi._new_data_analysis:
            data_analysis = roi._new_data_analysis[-1]
        elif roi.data_analysis_history:
            data_analysis = roi.data_analysis_history[-1]

        dec_dff = np.asarray(traces.dec_dff, dtype=float)
        if dec_dff.ndim != 1 or dec_dff.size == 0:
            continue

        roi_labels.append(int(roi.label_value))
        dec_dff_traces.append(dec_dff)

        # Build peak event binary arrays
        if data_analysis is not None and data_analysis.peaks_dec_dff is not None:
            # Create binary peak event array
            peak_indices = [int(p) for p in data_analysis.peaks_dec_dff]
            peak_array = np.zeros(len(dec_dff), dtype=float)
            for idx in peak_indices:
                if 0 <= idx < len(peak_array):
                    peak_array[idx] = 1.0
            peak_events_dict[str(roi.label_value)] = peak_array.tolist()

        # Build spike trains for inferred spikes
        if traces.inferred_spikes is not None:
            spikes = np.asarray(traces.inferred_spikes, dtype=float)
            spike_threshold = (
                data_analysis.inferred_spikes_threshold
                if data_analysis is not None
                else None
            )

            if spike_threshold is not None:
                # Threshold and binarize
                spikes[spikes <= spike_threshold] = 0.0
                spike_train = (spikes > 0.0).astype(float)
                if spike_train.sum() > 0:
                    spike_trains.append(spike_train)
                    spike_data_dict[str(roi.label_value)] = spike_train.tolist()

    if len(roi_labels) < 2:
        cali_logger.debug(
            f"FOV {fov.name}: Not enough ROIs with valid traces "
            f"({len(roi_labels)}) for correlation analysis."
        )
        return None

    # Compute calcium peaks metrics (3 measurements):
    # 1. Zero-lag correlation on DF/F traces
    calcium_dff_corr_matrix = _compute_cross_correlation_matrix(dec_dff_traces)

    # 2. Jitter synchrony on calcium peaks
    calcium_peaks_jitter_sync_matrix = None
    global_calcium_peaks_jitter_sync = None
    if len(peak_events_dict) >= 2:
        jitter_window = analysis_settings.calcium_sync_jitter_window
        calcium_peaks_jitter_sync_matrix = _get_calcium_peaks_event_synchrony_matrix(
            peak_events_dict,
            method="jitter_window",
            jitter_window=jitter_window,
        )
        if calcium_peaks_jitter_sync_matrix is not None:
            global_calcium_peaks_jitter_sync = _get_calcium_peaks_event_synchrony(
                calcium_peaks_jitter_sync_matrix
            )

    # 3. Max lag correlation on calcium peaks
    calcium_peaks_max_lag_corr_matrix = None
    global_calcium_peaks_max_lag_corr = None
    if len(peak_events_dict) >= 2:
        max_lag = analysis_settings.calcium_peaks_max_lag
        calcium_peaks_max_lag_corr_matrix = _get_calcium_peaks_event_synchrony_matrix(
            peak_events_dict,
            method="cross_correlation",
            max_lag=max_lag,
        )
        if calcium_peaks_max_lag_corr_matrix is not None:
            global_calcium_peaks_max_lag_corr = _get_calcium_peaks_event_synchrony(
                calcium_peaks_max_lag_corr_matrix
            )

    # Compute spike metrics (3 measurements):
    # 1. Zero-lag correlation on spike trains
    spike_corr_matrix = None
    # 2. Max lag correlation on spikes
    spike_max_lag_corr_matrix = None
    global_spike_max_lag_corr = None
    # 3. Jitter synchrony on spikes
    spike_jitter_sync_matrix = None
    global_spike_jitter_sync = None

    if len(spike_data_dict) >= 2:
        # 1. Zero-lag Pearson correlation on spike trains
        spike_corr_matrix = _compute_cross_correlation_matrix(spike_trains)

        # 2. Max lag correlation on spikes
        max_lag = analysis_settings.spikes_sync_cross_corr_lag
        spike_max_lag_corr_matrix = _get_spike_synchrony_matrix(
            spike_data_dict,
            method="cross_correlation",
            max_lag=max_lag,
        )
        if spike_max_lag_corr_matrix is not None:
            global_spike_max_lag_corr = _get_spike_synchrony(spike_max_lag_corr_matrix)

        # 3. Jitter synchrony on spikes
        jitter_window = analysis_settings.calcium_sync_jitter_window
        spike_jitter_sync_matrix = _get_spike_synchrony_matrix(
            spike_data_dict,
            method="jitter_window",
            jitter_window=jitter_window,
        )
        if spike_jitter_sync_matrix is not None:
            global_spike_jitter_sync = _get_spike_synchrony(spike_jitter_sync_matrix)

    # Create FOVAnalysis object with all 6 measurements
    fov_analysis = FOVAnalysis(
        active_roi_labels=roi_labels,
        # Calcium metrics
        calcium_dff_correlation_matrix=(
            calcium_dff_corr_matrix.tolist()
            if calcium_dff_corr_matrix is not None
            else None
        ),
        calcium_peaks_jitter_synchrony_matrix=(
            calcium_peaks_jitter_sync_matrix.tolist()
            if calcium_peaks_jitter_sync_matrix is not None
            else None
        ),
        global_calcium_peaks_jitter_synchrony=global_calcium_peaks_jitter_sync,
        calcium_peaks_max_lag_correlation_matrix=(
            calcium_peaks_max_lag_corr_matrix.tolist()
            if calcium_peaks_max_lag_corr_matrix is not None
            else None
        ),
        global_calcium_peaks_max_lag_correlation=global_calcium_peaks_max_lag_corr,
        # Spike metrics
        spike_correlation_matrix=(
            spike_corr_matrix.tolist() if spike_corr_matrix is not None else None
        ),
        spike_max_lag_correlation_matrix=(
            spike_max_lag_corr_matrix.tolist()
            if spike_max_lag_corr_matrix is not None
            else None
        ),
        global_spike_max_lag_correlation=global_spike_max_lag_corr,
        spike_jitter_synchrony_matrix=(
            spike_jitter_sync_matrix.tolist()
            if spike_jitter_sync_matrix is not None
            else None
        ),
        global_spike_jitter_synchrony=global_spike_jitter_sync,
    )

    return fov_analysis


def _compute_cross_correlation_matrix(
    traces: list[np.ndarray],
) -> np.ndarray | None:
    """Compute pairwise Pearson correlation matrix for traces (zero-lag).

    Uses z-scored traces and computes standard Pearson correlation coefficient
    at zero lag, following the approach used in CaImAn and standard practice.

    Note: This computes zero-lag correlation (standard Pearson R), not max
    cross-correlation across lags. For lag-invariant correlation, use synchrony
    metrics instead.

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
