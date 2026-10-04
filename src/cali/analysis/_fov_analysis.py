"""Independent calcium and per-method FOV correlation, synchrony and bursts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cali.analysis._cluster_analysis import compute_cluster_analysis
from cali.analysis._fov_metrics import (
    _compute_zero_lag_corr_matrix,
    _detect_calcium_population_bursts,
    _get_global_pairwise_score,
)
from cali.sqlmodel import FOVAnalysis
from cali.sqlmodel._spike_settings import canonical_spike_methods

from ._fov_inputs import CalciumPopulation, collect_calcium, collect_spikes
from ._noise_qc import summarize_noise

if TYPE_CHECKING:
    import numpy as np

    from cali.sqlmodel import FOV, AnalysisSettings


def _compute_calcium_population(
    population: CalciumPopulation, analysis_settings: AnalysisSettings
) -> FOVAnalysis:
    # Calcium trace metrics (gated by enable_calcium)
    calcium_dff_corr_matrix = None
    calcium_den_dff_corr_matrix = None
    global_calcium_dff_corr = None
    global_calcium_den_dff_corr = None

    if analysis_settings.enable_calcium:
        calcium_dff_corr_matrix = _compute_zero_lag_corr_matrix(population.dff)
        calcium_den_dff_corr_matrix = _compute_zero_lag_corr_matrix(population.den_dff)

        global_calcium_dff_corr = (
            _get_global_pairwise_score(calcium_dff_corr_matrix)
            if calcium_dff_corr_matrix is not None
            else None
        )
        global_calcium_den_dff_corr = (
            _get_global_pairwise_score(calcium_den_dff_corr_matrix)
            if calcium_den_dff_corr_matrix is not None
            else None
        )

    # Cluster analysis on denoised ΔF/F correlation matrix
    cluster_labels = None
    cluster_method_used = None
    cluster_n = None
    cluster_silhouette = None
    cluster_order = None

    if calcium_den_dff_corr_matrix is not None and len(population.labels) >= 3:
        cluster_result = compute_cluster_analysis(
            corr_matrix=calcium_den_dff_corr_matrix,
            method=analysis_settings.cluster_method,
            n_clusters=analysis_settings.cluster_n_clusters,
            max_k=analysis_settings.cluster_max_k,
        )
        if cluster_result is not None:
            cluster_labels = cluster_result.labels
            cluster_method_used = analysis_settings.cluster_method
            cluster_n = cluster_result.n_clusters
            cluster_silhouette = cluster_result.silhouette_score
            cluster_order = cluster_result.order

    calcium_burst_count: int | None = None
    calcium_burst_avg_duration: float | None = None
    calcium_burst_avg_interval: float | None = None
    calcium_burst_starts: list[int] = []
    calcium_burst_ends: list[int] = []
    calcium_population_activity: np.ndarray | None = None
    calcium_population_activity_raw: np.ndarray | None = None

    if analysis_settings.enable_calcium and len(population.peaks) >= 2:
        (
            calcium_burst_count,
            calcium_burst_avg_duration,
            calcium_burst_avg_interval,
            calcium_burst_starts,
            calcium_burst_ends,
            calcium_population_activity_raw,
            calcium_population_activity,
        ) = _detect_calcium_population_bursts(
            peak_events=population.peaks,
            frame_rate=analysis_settings.frame_rate,
            burst_threshold_percent=analysis_settings.calcium_burst_threshold,
            min_duration_ms=analysis_settings.calcium_burst_min_duration,
            gaussian_sigma_sec=analysis_settings.calcium_burst_gaussian_sigma,
        )

    # Create FOVAnalysis object with all measurements
    return FOVAnalysis(
        calcium_active_roi_labels=population.labels,
        calcium_dff_correlation_matrix=calcium_dff_corr_matrix.tolist()
        if calcium_dff_corr_matrix is not None
        else None,
        calcium_den_dff_corr_matrix=calcium_den_dff_corr_matrix.tolist()
        if calcium_den_dff_corr_matrix is not None
        else None,
        global_calcium_dff_correlation=global_calcium_dff_corr,
        global_calcium_den_dff_correlation=global_calcium_den_dff_corr,
        calcium_burst_count=calcium_burst_count,
        calcium_burst_avg_duration=calcium_burst_avg_duration,
        calcium_burst_avg_interval=calcium_burst_avg_interval,
        calcium_burst_starts=calcium_burst_starts if calcium_burst_starts else None,
        calcium_burst_ends=calcium_burst_ends if calcium_burst_ends else None,
        calcium_population_activity=calcium_population_activity.tolist()
        if calcium_population_activity is not None
        else None,
        calcium_population_activity_raw=calcium_population_activity_raw.tolist()
        if calcium_population_activity_raw is not None
        else None,
        cluster_labels=cluster_labels,
        cluster_method=cluster_method_used,
        cluster_n_clusters=cluster_n,
        cluster_silhouette_score=cluster_silhouette,
        cluster_order=cluster_order,
    )


def _compute_fov_analysis(
    fov: FOV, analysis_settings: AnalysisSettings, *, parallel: bool
) -> FOVAnalysis | None:
    from ._fov_analysis_parallel import compute_spike_population

    analysis_settings.validate_spike_settings()
    calcium = (
        collect_calcium(fov)
        if analysis_settings.enable_calcium
        else CalciumPopulation()
    )
    spikes = (
        [
            collect_spikes(fov, analysis_settings, method)
            for method in canonical_spike_methods(
                [child.method for child in analysis_settings.spike_settings]
            )
        ]
        if analysis_settings.enable_spikes
        else []
    )
    calcium_median, calcium_iqr, calcium_count = summarize_noise(calcium.noise)
    if (
        len(calcium.labels) < 2
        and not any(len(child.labels) >= 2 for child in spikes)
        and not calcium_count
        and not any(summarize_noise(child.noise)[2] for child in spikes)
    ):
        return None
    parent = (
        _compute_calcium_population(calcium, analysis_settings)
        if len(calcium.labels) >= 2
        else FOVAnalysis(calcium_active_roi_labels=calcium.labels)
    )
    parent.spike_analyses = [
        compute_spike_population(
            population,
            analysis_settings.get_spike_settings(population.method),
            name=fov.name,
            parallel=parallel,
            n_workers=max(1, analysis_settings.n_processes),
        )
        for population in spikes
    ]
    if analysis_settings.enable_calcium:
        parent.calcium_noise_median = calcium_median
        parent.calcium_noise_iqr = calcium_iqr
        parent.calcium_noise_roi_count = calcium_count
    return parent


def compute_fov_analysis(
    fov: FOV, analysis_settings: AnalysisSettings
) -> FOVAnalysis | None:
    """Compute independent populations using sequential pairwise spike metrics."""
    return _compute_fov_analysis(fov, analysis_settings, parallel=False)
