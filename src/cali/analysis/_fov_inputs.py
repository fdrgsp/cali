"""Independent calcium and method-bound spike populations for FOV analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ._roi_analysis import cascade_acquisition_rate, valid_spike_values
from ._trace_analysis import compute_rising_edges, threshold_spike_train

if TYPE_CHECKING:
    from cali.sqlmodel import (
        FOV,
        ROI,
        AnalysisSettings,
        DataAnalysis,
        SpikeInferenceRun,
        SpikeTrace,
        Traces,
    )
    from cali.sqlmodel._spike_settings import SpikeMethod


@dataclass
class CalciumPopulation:
    labels: list[int] = field(default_factory=list)
    dff: list[np.ndarray] = field(default_factory=list)
    den_dff: list[np.ndarray] = field(default_factory=list)
    peaks: list[np.ndarray] = field(default_factory=list)
    noise: list[float] = field(default_factory=list)


@dataclass
class SpikePopulation:
    method: SpikeMethod
    units: str
    frame_rate: float
    labels: list[int] = field(default_factory=list)
    trains: list[np.ndarray] = field(default_factory=list)
    binary: dict[str, list[float]] = field(default_factory=dict)
    onsets: dict[str, list[float]] = field(default_factory=dict)
    valid_start: int | None = None
    valid_stop: int | None = None
    inference_run: SpikeInferenceRun | None = None
    noise: list[float] = field(default_factory=list)


def selected_roi_products(roi: ROI) -> tuple[Traces | None, DataAnalysis | None]:
    """Use the pinned extraction and staged analysis, rather than unrelated history."""
    staged_traces = getattr(roi, "_new_traces", None)
    trace = (
        staged_traces[-1]
        if staged_traces
        else getattr(roi, "_analysis_source_trace", None)
    )
    if trace is None and roi.traces_history:
        trace = roi.traces_history[-1]
    staged_analysis = getattr(roi, "_new_data_analysis", None)
    analysis = (
        staged_analysis[-1]
        if staged_analysis
        else (roi.data_analysis_history[-1] if roi.data_analysis_history else None)
    )
    return trace, analysis


def _active(flag: bool | None, roi: ROI) -> bool:
    # Historical rows and the legacy constructor predate per-pillar flags.
    # An explicit False always wins over the union summary.
    return bool(roi.active) if flag is None else flag


def collect_calcium(fov: FOV) -> CalciumPopulation:
    result = CalciumPopulation()
    for roi in fov.rois:
        trace, analysis = selected_roi_products(roi)
        if roi.label_value is None or trace is None:
            continue
        noise = trace.calcium_noise
        # A staged result records the exact noise used for this selected trace,
        # including GetSn fallback for historical extractions. Unrelated stored
        # analysis history must not replace the selected extraction's estimate.
        if getattr(roi, "_new_data_analysis", None) and analysis is not None:
            noise = (
                analysis.calcium_noise if analysis.calcium_noise is not None else noise
            )
        if noise is not None:
            result.noise.append(noise)
        if not _active(analysis.calcium_active if analysis else None, roi):
            continue
        dff = np.asarray(trace.dff, dtype=float)
        den = np.asarray(trace.den_dff, dtype=float)
        if dff.ndim != 1 or not dff.size or den.ndim != 1 or not den.size:
            continue
        if dff.shape != den.shape or (result.dff and dff.shape != result.dff[0].shape):
            raise ValueError("Calcium FOV traces must have matching lengths.")
        result.labels.append(int(roi.label_value))
        result.dff.append(dff)
        result.den_dff.append(den)
        if analysis is not None and analysis.peaks_den_dff is not None:
            peaks = np.zeros(len(den), dtype=float)
            for index in analysis.peaks_den_dff:
                if 0 <= int(index) < len(peaks):
                    peaks[int(index)] = 1
            result.peaks.append(peaks)
    return result


def _check_alignment(reference: Traces, trace: Traces, size: int) -> None:
    if reference.x_axis is not None or trace.x_axis is not None:
        if (
            reference.x_axis is None
            or trace.x_axis is None
            or len(trace.x_axis) != size
            or len(reference.x_axis) != size
            or reference.x_axis_units != trace.x_axis_units
            or not np.allclose(reference.x_axis, trace.x_axis, rtol=0, atol=1e-6)
        ):
            raise ValueError("Spike FOV traces must share their retained time axis.")
    a, b = reference.extraction_frame_window, trace.extraction_frame_window
    if (a is None) != (b is None) or (
        a is not None
        and b is not None
        and any(
            getattr(a, key) != getattr(b, key)
            for key in (
                "source_start_frame",
                "source_start_time_ms",
                "retained_frame_count",
            )
        )
    ):
        raise ValueError("Spike FOV traces must share their extraction frame window.")


def collect_spikes(
    fov: FOV, settings: AnalysisSettings, method: SpikeMethod
) -> SpikePopulation:
    result = SpikePopulation(
        method=method,
        units="a.u." if method == "oasis" else "spikes/frame",
        frame_rate=settings.frame_rate,
    )
    selected: list[tuple[int, SpikeTrace, float]] = []
    reference = None
    size = None
    for roi in fov.rois:
        trace, analysis = selected_roi_products(roi)
        if roi.label_value is None or trace is None or analysis is None:
            continue
        spike = trace.get_spike_trace(method)
        metric = analysis.get_spike_analysis(method)
        if spike is None or metric is None:
            continue
        run = spike.inference_run
        if (
            run.method != method
            or run.units != result.units
            or metric.units != run.units
        ):
            raise ValueError(
                "Spike FOV inputs must match the selected method and units."
            )
        if (
            metric.spike_trace is not None
            and metric.spike_trace is not spike
            and (spike.id is None or metric.spike_trace_id != spike.id)
        ):
            raise ValueError("Spike FOV analysis must use its selected stored trace.")
        previous = result.inference_run
        if previous is not None and run is not previous:
            same_id = run.id is not None and run.id == previous.id
            synthetic = (
                method == "oasis"
                and run.provenance_source
                == previous.provenance_source
                == "synthetic_legacy_api"
                and run.id is None
                and previous.id is None
                and run.semantic_key() == previous.semantic_key()
            )
            if not (same_id or synthetic):
                raise ValueError(
                    "Spike FOV inputs must share one inference run per method."
                )
        result.inference_run = run
        if method == "cascade" and spike.noise is not None:
            result.noise.append(spike.noise)
        if not _active(metric.spike_active, roi):
            continue
        if metric.threshold is None:
            raise ValueError("Active spike FOV inputs require an applied threshold.")
        valid_spike_values(spike)  # validate before intersecting individual intervals
        if size is not None and len(spike.values) != size:
            raise ValueError("Spike FOV traces must have matching lengths.")
        _check_alignment(reference or trace, trace, len(spike.values))
        reference = trace
        size = len(spike.values)
        if method == "cascade":
            rate = cascade_acquisition_rate(spike, trace)
            if selected and not np.isclose(rate, result.frame_rate, rtol=1e-9):
                raise ValueError("CASCADE FOV traces must share an acquisition rate.")
            result.frame_rate = rate
        selected.append((int(roi.label_value), spike, metric.threshold))
    if not selected:
        return result
    start = max(spike.valid_start for _, spike, _ in selected)
    stop = min(spike.resolved_valid_stop for _, spike, _ in selected)
    if start >= stop:
        raise ValueError("Spike FOV inputs have no common valid interval.")
    result.valid_start, result.valid_stop = start, stop
    for label, spike, threshold in selected:
        if label in result.labels:
            raise ValueError("Spike FOV ROI labels must be unique.")
        binary = threshold_spike_train(
            np.asarray(spike.values[start:stop], dtype=float), threshold
        )
        onsets = compute_rising_edges(binary)
        if start > 0 or method == "cascade":
            onsets[0] = 0
        result.labels.append(label)
        result.trains.append(binary)
        result.binary[str(label)] = binary.tolist()
        result.onsets[str(label)] = onsets.tolist()
    return result
