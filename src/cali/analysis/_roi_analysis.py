"""Shared calcium and method-bound spike calculations for extraction/re-analysis."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from oasis.functions import GetSn  # type: ignore[import-untyped, unused-ignore]
from scipy import ndimage  # type: ignore[import-untyped, unused-ignore]

from cali.sqlmodel import DataAnalysis, SpikeAnalysis

from ._trace_analysis import (
    calculate_frequency,
    calculate_inter_event_intervals,
    compute_calcium_peak_detection_thresholds,
    compute_inferred_spike_threshold,
    compute_rising_edges,
    detect_peaks_in_trace,
    threshold_spike_train,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from cali.sqlmodel import (
        AnalysisSettings,
        SpikeAnalysisSettings,
        SpikeTrace,
        Traces,
    )


class AnalysisCancelled(Exception):
    """A caller cancelled shared ROI calculations before complete products exist."""


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise AnalysisCancelled


def valid_spike_values(spike: SpikeTrace) -> np.ndarray:
    """Select an explicit, nonempty valid interval without consulting padded edges."""
    stop = spike.resolved_valid_stop
    if not 0 <= spike.valid_start < stop <= len(spike.values):
        raise ValueError("Spike analysis requires a nonempty valid interval.")
    values = np.asarray(spike.values[spike.valid_start : stop], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Valid spike samples must be finite.")
    # OASIS can retain tiny negative numerical residuals; preserve legacy inputs.
    if spike.inference_run.method == "cascade" and np.any(values < 0):
        raise ValueError("Valid CASCADE samples must be non-negative.")
    return values


def valid_spike_events(
    spike: SpikeTrace, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid-only binary samples/onsets, suppressing an artificial crop edge.

    CASCADE's first valid sample is left-censored even when valid_start is zero.
    Full-length OASIS keeps its historical first-positive-sample convention.
    """
    binary = threshold_spike_train(valid_spike_values(spike), threshold)
    onsets = compute_rising_edges(binary)
    if spike.valid_start > 0 or spike.inference_run.method == "cascade":
        onsets[0] = 0
    return binary, onsets


def cascade_acquisition_rate(
    spike: SpikeTrace, source_trace: Traces | None = None
) -> float:
    """Use persisted recording/model timing; re-analysis never opens a model file."""
    run = spike.inference_run
    trace = source_trace if source_trace is not None else spike.trace
    window = trace.extraction_frame_window if trace is not None else None
    rate = window.acquisition_frame_rate_hz if window is not None else None
    model_rate = run.model_sampling_rate_hz
    if (
        rate is None
        or model_rate is None
        or not math.isfinite(rate)
        or rate <= 0
        or not math.isfinite(model_rate)
        or model_rate <= 0
        or abs(rate / model_rate - 1) > 0.01
    ):
        raise ValueError(
            "CASCADE analysis requires matching persisted acquisition/model rates."
        )
    return rate


def analyze_spike_trace(
    spike: SpikeTrace,
    settings: SpikeAnalysisSettings,
    *,
    legacy_duration_s: float,
    source_trace: Traces | None = None,
) -> SpikeAnalysis:
    """Compute one method's metrics with its own settings, units and valid samples."""
    settings.validate_parameters()
    run = spike.inference_run
    if run is None or settings.method != run.method:
        raise ValueError("Spike settings must match the stored inference method.")
    if (
        run.method not in {"oasis", "cascade"}
        or run.units != {"oasis": "a.u.", "cascade": "spikes/frame"}[run.method]
    ):
        raise ValueError("Spike inference method and units must match.")
    if source_trace is not None and not any(
        child is spike for child in source_trace.spike_traces
    ):
        raise ValueError("Spike analysis source must own the selected spike trace.")
    values = valid_spike_values(spike)
    if run.method == "oasis":
        threshold = compute_inferred_spike_threshold(values, settings)
        binary, onsets = valid_spike_events(spike, threshold)
        # Preserve T-1 timestamp duration for legacy full-length OASIS calculations.
        duration = legacy_duration_s
        if spike.valid_start or spike.resolved_valid_stop != len(spike.values):
            trace = source_trace if source_trace is not None else spike.trace
            if trace is None or trace.x_axis is None:
                raise ValueError("Cropped OASIS analysis requires stored timestamps.")
            times = trace.x_axis[spike.valid_start : spike.resolved_valid_stop]
            duration = (times[-1] - times[0]) / 1000
        return SpikeAnalysis(
            spike_trace=spike,
            method=run.method,
            units=run.units,
            threshold=threshold,
            threshold_mode=settings.threshold_mode,
            suprathreshold_sample_rate_hz=calculate_frequency(
                int(binary.sum()), duration
            ),
            suprathreshold_rising_edge_rate_hz=calculate_frequency(
                int(onsets.sum()), duration
            )
            if settings.enable_rising_edge_analysis
            else None,
            spike_active=bool(np.any(binary)),
        )
    rate = cascade_acquisition_rate(spike, source_trace)
    if settings.threshold_mode == "cascade_ap":
        smoothing = run.smoothing_sigma
        if smoothing is None or not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError(
                "CASCADE AP threshold requires persisted positive smoothing."
            )
        single = np.zeros(1001)
        single[500] = 1.0
        # Plain cutoff of the unmasked expected-rate trace, without upstream dilation.
        assert run.model_sampling_rate_hz is not None
        peak = float(
            ndimage.gaussian_filter1d(
                single, sigma=smoothing * run.model_sampling_rate_hz
            ).max()
        )
        assert settings.cascade_ap_threshold_fraction is not None
        threshold = settings.cascade_ap_threshold_fraction * peak
    else:
        assert settings.threshold_value is not None
        threshold = settings.threshold_value
    binary, onsets = valid_spike_events(spike, threshold)
    return SpikeAnalysis(
        spike_trace=spike,
        method=run.method,
        units=run.units,
        threshold=threshold,
        threshold_mode=settings.threshold_mode,
        expected_spike_count=float(values.sum()),
        expected_spike_rate_hz=float(values.mean()) * rate,
        suprathreshold_excursion_rate_hz=float(onsets.sum()) / len(values) * rate
        if settings.enable_rising_edge_analysis
        else None,
        spike_active=bool(np.any(binary)),
    )


def analyze_roi_calcium(
    traces: Traces,
    settings: AnalysisSettings,
    *,
    duration_s: float,
    cancel: Callable[[], bool] | None = None,
) -> DataAnalysis:
    """Compute the shared calcium pillar with legacy thresholds and duration."""
    _check_cancel(cancel)
    result = DataAnalysis(total_recording_time_sec=duration_s, calcium_active=False)
    if not settings.enable_calcium:
        return result
    den_dff = np.asarray(traces.den_dff)
    noise = traces.calcium_noise
    if noise is None:
        noise = GetSn(np.asarray(traces.dff), range_ff=[0.25, 0.5], method="median")
    height, prominence = compute_calcium_peak_detection_thresholds(
        den_dff, noise, settings
    )
    _check_cancel(cancel)
    distance = max(1, int(settings.peaks_distance / 1000 * settings.frame_rate))
    peaks, amplitudes = detect_peaks_in_trace(den_dff, height, prominence, distance)
    _check_cancel(cancel)
    assert traces.x_axis is not None
    intervals = calculate_inter_event_intervals(peaks, traces.x_axis)
    result.den_dff_frequency = calculate_frequency(len(peaks), duration_s)
    result.peaks_den_dff = peaks.tolist() if len(peaks) else None
    result.peaks_amplitudes_den_dff = amplitudes or None
    result.iei = [value / 1000 for value in intervals] or None
    result.peaks_prominence_den_dff = prominence
    result.peaks_height_den_dff = height
    result.calcium_active = bool(len(peaks))
    return result


def analyze_roi_traces(
    traces: Traces,
    settings: AnalysisSettings,
    *,
    duration_s: float,
    cancel: Callable[[], bool] | None = None,
) -> DataAnalysis:
    """Stage shared calcium and every selected spike product in canonical order."""
    if settings.enable_spikes:
        methods = tuple(child.inference_run.method for child in traces.spike_traces)
        settings.validate_spike_settings(methods)
        if len(methods) != len(set(methods)):
            raise ValueError("Duplicate stored spike methods cannot be analyzed.")
    result = analyze_roi_calcium(traces, settings, duration_s=duration_s, cancel=cancel)
    if settings.enable_spikes:
        for method in ("oasis", "cascade"):
            spike = traces.get_spike_trace(method)
            if spike is not None:
                _check_cancel(cancel)
                result.spike_analyses.append(
                    analyze_spike_trace(
                        spike,
                        settings.get_spike_settings(method),
                        legacy_duration_s=duration_s,
                        source_trace=traces,
                    )
                )
    _check_cancel(cancel)
    return result
