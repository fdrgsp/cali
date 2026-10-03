"""FOV spike metrics with one process pool per method population."""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING

import numpy as np

from cali.analysis._fov_metrics import (
    _compute_baseline_corrected_ccg_numba,
    _detect_spikes_population_bursts,
    _get_fraction_significant_pairs,
    _get_global_pairwise_score,
    _jitter_window_synchrony_numba,
)
from cali.logger import cali_logger
from cali.sqlmodel._spike_fov_analysis import SpikeFOVAnalysis

if TYPE_CHECKING:
    from cali.sqlmodel import FOV, AnalysisSettings, FOVAnalysis, SpikeAnalysisSettings

    from ._fov_inputs import SpikePopulation


def _compute_ccg_for_pair(args: tuple) -> tuple[int, int, float, int, float]:
    """Worker function to compute CCG for a single ROI pair.

    Parameters
    ----------
    args : tuple
        (i, j, spike_i, spike_j, max_lag, n_shuffles)

    Returns
    -------
    tuple
        (i, j, max_ccg, best_lag, zscore)
    """
    i, j, spike_i, spike_j, max_lag, n_shuffles = args

    if np.sum(spike_i) == 0 or np.sum(spike_j) == 0:
        return (i, j, 0.0, 0, 0.0)

    lags, ccg_raw, baseline_mean, baseline_std = _compute_baseline_corrected_ccg_numba(
        spike_i, spike_j, max_lag, n_shuffles
    )

    max_idx = np.argmax(ccg_raw)
    max_value = float(ccg_raw[max_idx])
    best_lag = int(lags[max_idx])

    if baseline_std[max_idx] > 0:
        zscore = (ccg_raw[max_idx] - baseline_mean[max_idx]) / baseline_std[max_idx]
    else:
        zscore = 0.0

    return (i, j, max_value, best_lag, float(zscore))


def _compute_jitter_for_pair(args: tuple) -> tuple[int, int, float]:
    """Worker function to compute jitter synchrony for a single ROI pair.

    Parameters
    ----------
    args : tuple
        (i, j, spike_i, spike_j, jitter_window)

    Returns
    -------
    tuple
        (i, j, synchrony_value)
    """
    i, j, spike_i, spike_j, jitter_window = args

    if np.sum(spike_i) == 0 or np.sum(spike_j) == 0:
        return (i, j, 0.0)

    sync_value = float(_jitter_window_synchrony_numba(spike_i, spike_j, jitter_window))
    return (i, j, sync_value)


def compute_spike_population(
    population: SpikePopulation,
    settings: SpikeAnalysisSettings,
    *,
    name: str,
    parallel: bool,
    n_workers: int,
) -> SpikeFOVAnalysis:
    """Compute metrics only within one method's common valid interval.

    Burst bounds use retained-relative coordinates. Population arrays cover
    [valid_start, valid_stop), whose persisted metadata supplies their origin.
    """
    settings.validate_parameters()
    if settings.method != population.method:
        raise ValueError("FOV spike settings must match the selected population.")
    n_rois = len(population.labels)
    n_pairs = n_rois * (n_rois - 1) // 2
    use_parallel = parallel and n_rois >= 10
    spike_trains = population.trains
    spike_data_dict = population.binary
    spike_data_dict_rising_edges = population.onsets

    def ms_to_frames(ms: float) -> int:
        return max(0, int((ms / 1000) * population.frame_rate))

    # Initialize matrices
    spike_max_lag_corr_matrix = None
    spike_max_lag_values_matrix = None
    global_spike_max_lag_corr = None
    spike_ccg_zscore_matrix = None
    frac_sig_ccg_pairs = None
    spike_jitter_sync_matrix = None
    global_spike_jitter_sync = None
    # Rising edges
    spike_max_lag_corr_matrix_rising_edges = None
    spike_max_lag_values_matrix_rising_edges = None
    global_spike_max_lag_corr_rising_edges = None
    spike_ccg_zscore_matrix_rising_edges = None
    frac_sig_ccg_pairs_rising_edges = None
    spike_jitter_sync_matrix_rising_edges = None
    global_spike_jitter_sync_rising_edges = None

    if len(spike_trains) >= 2:
        max_lag_frames = ms_to_frames(settings.spikes_sync_cross_corr_lag)
        jitter_window_frames = ms_to_frames(settings.spikes_sync_jitter_window)
        n_shuffles = settings.ccg_n_shuffles
        spike_trains_array = np.array(spike_trains, dtype=np.float32)
        pairs = [(i, j) for i in range(n_rois) for j in range(i + 1, n_rois)]

        if use_parallel:
            cali_logger.info(
                f"🖥️ FOV {name} ({population.method}): Parallel CCG "
                f"({n_rois} ROIs, {n_pairs} pairs, "
                f"{n_workers} workers)"
            )

            # Prepare arguments
            ccg_args = [
                (
                    i,
                    j,
                    spike_trains_array[i],
                    spike_trains_array[j],
                    max_lag_frames,
                    n_shuffles,
                )
                for i, j in pairs
            ]

            jitter_args = [
                (
                    i,
                    j,
                    spike_trains_array[i],
                    spike_trains_array[j],
                    jitter_window_frames,
                )
                for i, j in pairs
            ]

            # Use a single pool for all computations
            # 'spawn' is safer for numba but has overhead;
            # 'fork' is faster but can cause issues
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                # CCG computation
                ccg_results = pool.map(_compute_ccg_for_pair, ccg_args)

                # Jitter computation (reuse same pool)
                jitter_results = pool.map(_compute_jitter_for_pair, jitter_args)

            # Assemble CCG results
            spike_max_lag_corr_matrix = np.zeros((n_rois, n_rois))
            spike_max_lag_values_matrix = np.zeros((n_rois, n_rois), dtype=int)
            spike_ccg_zscore_matrix = np.zeros((n_rois, n_rois))

            np.fill_diagonal(spike_max_lag_corr_matrix, 1.0)
            np.fill_diagonal(spike_max_lag_values_matrix, 0)
            np.fill_diagonal(spike_ccg_zscore_matrix, np.inf)

            for i, j, max_ccg, best_lag, zscore in ccg_results:
                spike_max_lag_corr_matrix[i, j] = max_ccg
                spike_max_lag_corr_matrix[j, i] = max_ccg
                spike_max_lag_values_matrix[i, j] = best_lag
                spike_max_lag_values_matrix[j, i] = -best_lag
                spike_ccg_zscore_matrix[i, j] = zscore
                spike_ccg_zscore_matrix[j, i] = zscore

            global_spike_max_lag_corr = _get_global_pairwise_score(
                spike_max_lag_corr_matrix
            )
            frac_sig_ccg_pairs = _get_fraction_significant_pairs(
                spike_ccg_zscore_matrix
            )

            # Assemble jitter results
            spike_jitter_sync_matrix = np.zeros((n_rois, n_rois))
            np.fill_diagonal(spike_jitter_sync_matrix, 1.0)

            for i, j, sync_value in jitter_results:
                spike_jitter_sync_matrix[i, j] = sync_value
                spike_jitter_sync_matrix[j, i] = sync_value

            global_spike_jitter_sync = _get_global_pairwise_score(
                spike_jitter_sync_matrix
            )

        else:
            # Sequential computation for small FOVs
            from cali.analysis._fov_metrics import _get_spike_correlations_matrix

            (
                spike_max_lag_corr_matrix,
                spike_max_lag_values_matrix,
                spike_ccg_zscore_matrix,
            ) = _get_spike_correlations_matrix(
                spike_data_dict,
                method="cross_correlation",
                max_lag=max_lag_frames,
                n_shuffles=n_shuffles,
            )
            if spike_max_lag_corr_matrix is not None:
                global_spike_max_lag_corr = _get_global_pairwise_score(
                    spike_max_lag_corr_matrix
                )
            if spike_ccg_zscore_matrix is not None:
                frac_sig_ccg_pairs = _get_fraction_significant_pairs(
                    spike_ccg_zscore_matrix
                )

            spike_jitter_sync_matrix, _, _ = _get_spike_correlations_matrix(
                spike_data_dict,
                method="jitter_window",
                jitter_window=jitter_window_frames,
            )
            if spike_jitter_sync_matrix is not None:
                global_spike_jitter_sync = _get_global_pairwise_score(
                    spike_jitter_sync_matrix
                )

        # Rising edge analysis (if enabled)
        if (
            settings.enable_rising_edge_analysis
            and len(spike_data_dict_rising_edges) >= 2
        ):
            from cali.analysis._fov_metrics import _get_spike_correlations_matrix

            (
                spike_max_lag_corr_matrix_rising_edges,
                spike_max_lag_values_matrix_rising_edges,
                spike_ccg_zscore_matrix_rising_edges,
            ) = _get_spike_correlations_matrix(
                spike_data_dict_rising_edges,
                method="cross_correlation",
                max_lag=max_lag_frames,
                n_shuffles=n_shuffles,
            )
            if spike_max_lag_corr_matrix_rising_edges is not None:
                global_spike_max_lag_corr_rising_edges = _get_global_pairwise_score(
                    spike_max_lag_corr_matrix_rising_edges
                )
            if spike_ccg_zscore_matrix_rising_edges is not None:
                frac_sig_ccg_pairs_rising_edges = _get_fraction_significant_pairs(
                    spike_ccg_zscore_matrix_rising_edges
                )

            (
                spike_jitter_sync_matrix_rising_edges,
                _,
                _,
            ) = _get_spike_correlations_matrix(
                spike_data_dict_rising_edges,
                method="jitter_window",
                jitter_window=jitter_window_frames,
            )
            if spike_jitter_sync_matrix_rising_edges is not None:
                global_spike_jitter_sync_rising_edges = _get_global_pairwise_score(
                    spike_jitter_sync_matrix_rising_edges
                )

    # Burst detection (fast, no parallelization needed)
    spike_burst_count: int | None = None
    spike_burst_avg_duration: float | None = None
    spike_burst_avg_interval: float | None = None
    spike_burst_starts: list[int] = []
    spike_burst_ends: list[int] = []
    spike_population_activity: np.ndarray | None = None
    spike_population_activity_raw: np.ndarray | None = None

    if len(spike_trains) >= 2:
        (
            spike_burst_count,
            spike_burst_avg_duration,
            spike_burst_avg_interval,
            spike_burst_starts,
            spike_burst_ends,
            spike_population_activity_raw,
            spike_population_activity,
        ) = _detect_spikes_population_bursts(
            spike_trains=spike_trains,
            frame_rate=population.frame_rate,
            burst_threshold_percent=settings.burst_threshold,
            min_duration_ms=settings.burst_min_duration,
            gaussian_sigma_sec=settings.burst_gaussian_sigma,
        )

    return SpikeFOVAnalysis(
        method=population.method,
        units=population.units,
        active_roi_labels=population.labels,
        valid_start=population.valid_start,
        valid_stop=population.valid_stop,
        frame_rate_hz=population.frame_rate
        if population.valid_start is not None
        else None,
        inference_run=population.inference_run,
        spike_max_lag_correlation_matrix=spike_max_lag_corr_matrix.tolist()
        if spike_max_lag_corr_matrix is not None
        else None,
        global_spike_max_lag_correlation=global_spike_max_lag_corr,
        spike_max_lag_values_matrix=spike_max_lag_values_matrix.tolist()
        if spike_max_lag_values_matrix is not None
        else None,
        spike_ccg_zscore_matrix=spike_ccg_zscore_matrix.tolist()
        if spike_ccg_zscore_matrix is not None
        else None,
        spike_jitter_synchrony_matrix=spike_jitter_sync_matrix.tolist()
        if spike_jitter_sync_matrix is not None
        else None,
        global_spike_jitter_synchrony=global_spike_jitter_sync,
        spike_max_lag_correlation_matrix_rising_edges=spike_max_lag_corr_matrix_rising_edges.tolist()
        if spike_max_lag_corr_matrix_rising_edges is not None
        else None,
        global_spike_max_lag_correlation_rising_edges=global_spike_max_lag_corr_rising_edges,
        spike_max_lag_values_matrix_rising_edges=spike_max_lag_values_matrix_rising_edges.tolist()
        if spike_max_lag_values_matrix_rising_edges is not None
        else None,
        spike_ccg_zscore_matrix_rising_edges=spike_ccg_zscore_matrix_rising_edges.tolist()
        if spike_ccg_zscore_matrix_rising_edges is not None
        else None,
        fraction_significant_ccg_pairs=frac_sig_ccg_pairs,
        fraction_significant_ccg_pairs_rising_edges=frac_sig_ccg_pairs_rising_edges,
        spike_jitter_synchrony_matrix_rising_edges=spike_jitter_sync_matrix_rising_edges.tolist()
        if spike_jitter_sync_matrix_rising_edges is not None
        else None,
        global_spike_jitter_synchrony_rising_edges=global_spike_jitter_sync_rising_edges,
        spike_burst_count=spike_burst_count,
        spike_burst_avg_duration=spike_burst_avg_duration,
        spike_burst_avg_interval=spike_burst_avg_interval,
        spike_burst_starts=[
            x + (population.valid_start or 0) for x in spike_burst_starts
        ]
        or None,
        spike_burst_ends=[x + (population.valid_start or 0) for x in spike_burst_ends]
        or None,
        spike_population_activity=spike_population_activity.tolist()
        if spike_population_activity is not None
        else None,
        spike_population_activity_raw=spike_population_activity_raw.tolist()
        if spike_population_activity_raw is not None
        else None,
    )


def compute_fov_analysis_parallel(
    fov: FOV, analysis_settings: AnalysisSettings
) -> FOVAnalysis | None:
    """Compute independent FOV populations, parallelizing large spike collections."""
    from ._fov_analysis import _compute_fov_analysis

    return _compute_fov_analysis(fov, analysis_settings, parallel=True)
