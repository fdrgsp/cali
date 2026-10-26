"""Population calculations that never correlate or smooth across a pause."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from cali.sqlmodel import SpikeFOVAnalysis

from ._fov_metrics import (
    _compute_ccg_vector_numba,
    _get_fraction_significant_pairs,
    _get_global_pairwise_score,
    _jitter_window_synchrony_numba,
)
from ._noise_qc import summarize_noise

if TYPE_CHECKING:
    from cali.sqlmodel import SpikeAnalysisSettings

    from ._fov_inputs import SpikePopulation


def _pooled_pair(
    chunks: list[tuple[np.ndarray, np.ndarray]], lag: int, shuffles: int, jitter: int
) -> tuple[float, int, float, float]:
    """Pool per-trigger CCGs, independently shifting each chunk for the null."""
    lag = min(lag, max(0, (min(len(a) for a, _ in chunks) - 2) // 2))
    weights = np.array([a.sum() for a, _ in chunks], dtype=float)
    if not weights.sum():
        return 0.0, 0, 0.0, 0.0
    weights /= weights.sum()
    raw = sum(
        weight * _compute_ccg_vector_numba(a, b, lag)[1]
        for weight, (a, b) in zip(weights, chunks)
    )
    rng = np.random.default_rng()
    null = np.array(
        [
            sum(
                weight
                * _compute_ccg_vector_numba(
                    a,
                    np.roll(
                        b,
                        int(rng.integers(lag + 1, len(b) - lag))
                        if len(b) > 2 * lag + 1
                        else 0,
                    ),
                    lag,
                )[1]
                for weight, (a, b) in zip(weights, chunks)
            )
            for _ in range(shuffles)
        ]
    )
    peak = int(np.argmax(raw))
    sd = float(null[:, peak].std())
    z = (float(raw[peak]) - float(null[:, peak].mean())) / sd if sd else 0.0
    lengths = np.array([len(a) for a, _ in chunks], dtype=float)
    sync = np.average(
        [
            float(_jitter_window_synchrony_numba(a, b, jitter))
            if a.sum() and b.sum()
            else 0.0
            for a, b in chunks
        ],
        weights=lengths,
    )
    return float(raw[peak]), peak - lag, z, float(sync)


def compute_segmented_population(
    population: SpikePopulation, settings: SpikeAnalysisSettings, *, name: str
) -> SpikeFOVAnalysis:
    from ._fov_analysis_parallel import compute_spike_population

    assert population.valid_intervals and population.valid_start is not None
    origin = population.valid_start
    intervals = population.valid_intervals
    parts = []
    for start, stop in intervals:
        a, b = start - origin, stop - origin
        chunk = replace(
            population,
            valid_start=start,
            valid_stop=stop,
            valid_intervals=None,
            trains=[v[a:b] for v in population.trains],
            binary={k: v[a:b] for k, v in population.binary.items()},
            onsets={k: v[a:b] for k, v in population.onsets.items()},
        )
        parts.append(
            compute_spike_population(
                chunk,
                settings,
                name=name,
                parallel=False,
                n_workers=1,
                _skip_pairwise=True,
            )
        )
    count = sum(p.spike_burst_count or 0 for p in parts)
    interval_count = sum(max(0, (p.spike_burst_count or 0) - 1) for p in parts)
    median, iqr, noise_count = summarize_noise(population.noise)
    result = SpikeFOVAnalysis(
        method=population.method,
        units=population.units,
        inference_run=population.inference_run,
        active_roi_labels=population.labels,
        valid_start=origin,
        valid_stop=population.valid_stop,
        valid_intervals=[list(v) for v in intervals],
        frame_rate_hz=population.frame_rate,
        model_noise_median=median,
        model_noise_iqr=iqr,
        model_noise_roi_count=noise_count,
        spike_burst_count=count if len(population.labels) >= 2 else None,
        spike_burst_avg_duration=sum(
            (p.spike_burst_avg_duration or 0) * (p.spike_burst_count or 0)
            for p in parts
        )
        / count
        if count
        else None,
        spike_burst_avg_interval=sum(
            (p.spike_burst_avg_interval or 0) * max(0, (p.spike_burst_count or 0) - 1)
            for p in parts
        )
        / interval_count
        if interval_count
        else None,
        spike_burst_starts=[v for p in parts for v in (p.spike_burst_starts or [])]
        or None,
        spike_burst_ends=[v for p in parts for v in (p.spike_burst_ends or [])] or None,
    )
    assert population.valid_stop is not None
    for field in ("spike_population_activity", "spike_population_activity_raw"):
        if any(getattr(p, field) is not None for p in parts):
            values = np.zeros(population.valid_stop - origin)
            for (a, b), part in zip(intervals, parts):
                values[a - origin : b - origin] = getattr(part, field)
            setattr(result, field, values.tolist())
    if len(population.labels) < 2:
        return result
    lag = max(
        0, int(settings.spikes_sync_cross_corr_lag / 1000 * population.frame_rate)
    )
    jitter = max(
        0, int(settings.spikes_sync_jitter_window / 1000 * population.frame_rate)
    )
    for rising in (False, True):
        if rising and not settings.enable_rising_edge_analysis:
            continue
        data = population.onsets if rising else population.binary
        n = len(population.labels)
        corr, lags, zs, sync = (
            np.eye(n),
            np.zeros((n, n), int),
            np.zeros((n, n)),
            np.eye(n),
        )
        np.fill_diagonal(zs, np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                a = np.asarray(data[str(population.labels[i])], dtype=np.float64)
                b = np.asarray(data[str(population.labels[j])], dtype=np.float64)
                chunks = [
                    (a[s - origin : e - origin], b[s - origin : e - origin])
                    for s, e in intervals
                ]
                score, best, z, jitter_score = _pooled_pair(
                    chunks, lag, settings.ccg_n_shuffles, jitter
                )
                corr[i, j] = corr[j, i] = score
                lags[i, j], lags[j, i] = best, -best
                zs[i, j] = zs[j, i] = z
                sync[i, j] = sync[j, i] = jitter_score
        suffix = "_rising_edges" if rising else ""
        for field, value in (
            ("spike_max_lag_correlation_matrix", corr.tolist()),
            ("spike_max_lag_values_matrix", lags.tolist()),
            ("spike_ccg_zscore_matrix", zs.tolist()),
            ("spike_jitter_synchrony_matrix", sync.tolist()),
            ("global_spike_max_lag_correlation", _get_global_pairwise_score(corr)),
            ("fraction_significant_ccg_pairs", _get_fraction_significant_pairs(zs)),
            ("global_spike_jitter_synchrony", _get_global_pairwise_score(sync)),
        ):
            setattr(result, field + suffix, value)
    return result
