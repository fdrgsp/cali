"""Stored-method selection shared by plots and headless metric consumers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import func
from sqlmodel import Session, col, select

from cali.analysis._roi_analysis import valid_spike_events, valid_spike_values
from cali.sqlmodel import (
    FOV,
    ROI,
    DataAnalysis,
    FOVAnalysis,
    SpikeAnalysis,
    SpikeFOVAnalysis,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
)
from cali.sqlmodel._engine import ensure_schema_current
from cali.sqlmodel._spike_fov_analysis import SPIKE_FOV_METRICS
from cali.sqlmodel._spike_settings import canonical_spike_methods

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.sqlmodel._spike_settings import SpikeMethod


SPIKE_METRIC_METHODS = {
    "threshold": ("oasis", "cascade"),
    "suprathreshold_sample_rate_hz": ("oasis",),
    "suprathreshold_rising_edge_rate_hz": ("oasis",),
    "expected_spike_rate_hz": ("cascade",),
    "expected_spike_count": ("cascade",),
    "suprathreshold_excursion_rate_hz": ("cascade",),
}
SPIKE_METRIC_ALIASES = {
    "inferred_spikes_threshold": "threshold",
    "inferred_spikes_frequency": "suprathreshold_sample_rate_hz",
    "inferred_spikes_rising_edge_frequency": "suprathreshold_rising_edge_rate_hz",
}


def get_stored_spike_capabilities(
    engine: Engine, run_id: int, *, fov_name: str | None = None
) -> dict[str, set[str]]:
    """Discover stored methods/non-NULL fields without loading any trace arrays."""
    ensure_schema_current(engine)
    capabilities: dict[str, set[str]] = {}
    with Session(engine) as session:
        stmt = (
            select(SpikeInferenceRun.method)
            .join(
                SpikeTrace,
                col(SpikeTrace.spike_inference_run_id) == col(SpikeInferenceRun.id),
            )
            .join(Traces, col(SpikeTrace.trace_id) == col(Traces.id))
            .join(ROI, col(Traces.roi_id) == col(ROI.id))
            .join(FOV, col(ROI.fov_id) == col(FOV.id))
            .where(Traces.analysis_result_id == run_id)
            .distinct()
        )
        if fov_name is not None:
            stmt = stmt.where(FOV.name == fov_name)
        for method in session.exec(stmt):
            canonical_spike_methods((method,))
            capabilities.setdefault(method, set()).add("spike_trace")
        for model, parent, fields, roi_level in (
            (SpikeAnalysis, DataAnalysis, tuple(SPIKE_METRIC_METHODS), True),
            (SpikeFOVAnalysis, FOVAnalysis, SPIKE_FOV_METRICS, False),
        ):
            metrics = select(
                col(model.method),
                *(func.max(col(getattr(model, name)).is_not(None)) for name in fields),
            ).join(parent)
            if roi_level:
                metrics = metrics.join(
                    ROI, col(DataAnalysis.roi_id) == col(ROI.id)
                ).join(FOV, col(ROI.fov_id) == col(FOV.id))
            else:
                metrics = metrics.join(FOV, col(FOVAnalysis.fov_id) == col(FOV.id))
            metrics = metrics.where(parent.analysis_result_id == run_id)
            if fov_name is not None:
                metrics = metrics.where(FOV.name == fov_name)
            for row in session.exec(metrics.group_by(col(model.method))):
                canonical_spike_methods((row[0],))
                capabilities.setdefault(row[0], set()).update(
                    name for name, present in zip(fields, row[1:]) if present
                )
    return capabilities


def validate_spike_metric(method: SpikeMethod, metric: str) -> str:
    """Reject metric meanings that cannot belong to the selected method."""
    canonical_spike_methods((method,))
    metric = SPIKE_METRIC_ALIASES.get(metric, metric)
    if metric not in SPIKE_METRIC_METHODS or method not in SPIKE_METRIC_METHODS[metric]:
        raise ValueError(f"Spike metric {metric!r} is unavailable for {method}.")
    return metric


def roi_is_active(
    roi: ROI, analysis: DataAnalysis | None, method: SpikeMethod | None = None
) -> bool:
    """Use the selected pillar; only unknown historical flags use the summary."""
    if analysis is None:
        return False
    if method is None:
        flag = analysis.calcium_active
    else:
        canonical_spike_methods((method,))
        child = analysis.get_spike_analysis(method)
        if child is None:
            return False
        flag = child.spike_active
    return bool(roi.active) if flag is None else flag


@dataclass
class SpikePlotData:
    """One source-bound result with padding masked rather than treated as silence."""

    spike: SpikeTrace
    metric: SpikeAnalysis | None
    values: np.ndarray

    def event_frames(self, *, onsets: bool = False) -> np.ndarray:
        """Return retained-frame indices using the shared censored-onset rule."""
        if self.metric is None or self.metric.threshold is None:
            raise ValueError("Spike event plots require an applied threshold.")
        binary, edges = valid_spike_events(self.spike, self.metric.threshold)
        return np.flatnonzero(edges if onsets else binary) + self.spike.valid_start


def spike_plot_data(
    trace: Traces,
    analysis: DataAnalysis | None,
    method: SpikeMethod,
    *,
    require_threshold: bool = False,
) -> SpikePlotData | None:
    """Validate stored provenance and mask only the selected method's padding."""
    canonical_spike_methods((method,))
    spike = trace.get_spike_trace(method)
    if spike is None:
        return None
    run = spike.inference_run
    units = "a.u." if method == "oasis" else "spikes/frame"
    if run.method != method or run.units != units:
        raise ValueError("Plotted spike method and inference units must match.")
    metric = analysis.get_spike_analysis(method) if analysis else None
    if metric is not None:
        if metric.units != units:
            raise ValueError("Plotted spike metric units must match its method.")
        if metric.provenance_source == "legacy_unresolved":
            if require_threshold:
                return None
            metric = None
        elif metric.spike_trace_id is not None and metric.spike_trace_id != spike.id:
            raise ValueError("Spike plots must use the analyzed source trace.")
    if require_threshold and (metric is None or metric.threshold is None):
        return None
    if metric is not None and metric.threshold is not None:
        cutoff = metric.threshold
        if not math.isfinite(cutoff) and not (
            cutoff == math.inf
            and method == "oasis"
            and metric.threshold_mode == "multiplier"
        ):
            raise ValueError("Plotted spike thresholds must be finite.")
    values = np.full(len(spike.values), np.nan)
    values[spike.valid_start : spike.resolved_valid_stop] = valid_spike_values(spike)
    return SpikePlotData(spike, metric, values)
