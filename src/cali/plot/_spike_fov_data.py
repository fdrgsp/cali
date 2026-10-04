"""Validated method-bound FOV results and explicit population coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cali.sqlmodel._spike_fov_analysis import SPIKE_FOV_METRICS
from cali.sqlmodel._spike_settings import canonical_spike_methods

if TYPE_CHECKING:
    from cali.sqlmodel import AnalysisSettings, FOVAnalysis, SpikeFOVAnalysis
    from cali.sqlmodel._spike_settings import SpikeMethod


def selected_spike_fov(
    parent: FOVAnalysis, method: SpikeMethod
) -> SpikeFOVAnalysis | None:
    """Select exactly one stored population without substituting another method."""
    canonical_spike_methods((method,))
    child = parent.get_spike_analysis(method)
    if child is None:
        return None
    units = "a.u." if method == "oasis" else "spikes/frame"
    if child.method != method or child.units != units:
        raise ValueError("Plotted spike FOV method and units must match.")
    run = child.inference_run
    if run is not None and (run.method != method or run.units != units):
        raise ValueError("Plotted spike FOV provenance must match its method.")
    labels = child.active_roi_labels or []
    if len(labels) != len(set(labels)):
        raise ValueError("Spike FOV ROI labels must be unique.")
    child.validate_population_coordinates()
    return child


def spike_fov_matrix(
    parent: FOVAnalysis, method: SpikeMethod, field: str
) -> tuple[np.ndarray | None, list[int] | None]:
    """Keep both matrix axes in the selected method's stored label order."""
    if field not in SPIKE_FOV_METRICS or "matrix" not in field:
        raise ValueError(f"Unknown spike matrix {field!r}.")
    child = selected_spike_fov(parent, method)
    if child is None:
        return None, None
    values, labels = getattr(child, field), child.active_roi_labels
    if values is None or not labels:
        return None, None
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Spike matrix dimensions must match its active ROI labels.")
    return matrix, list(labels)


def spike_population_duration(
    child: SpikeFOVAnalysis, settings: AnalysisSettings | None = None
) -> float | None:
    """Use the analyzed interval/rate; preserve only OASIS's historical fallback."""
    child.validate_population_coordinates()
    if child.valid_start is not None:
        assert child.valid_stop is not None and child.frame_rate_hz is not None
        return (child.valid_stop - child.valid_start) / child.frame_rate_hz
    activity = child.spike_population_activity
    if (
        child.method == "oasis"
        and activity
        and settings is not None
        and settings.frame_rate > 0
    ):
        return len(activity) / settings.frame_rate
    return None


def spike_burst_bounds(child: SpikeFOVAnalysis) -> list[tuple[int, int]]:
    """Return stored half-open retained bounds, rejecting mismatched intervals."""
    child.validate_population_coordinates()
    starts, stops = child.spike_burst_starts or [], child.spike_burst_ends or []
    if len(starts) != len(stops):
        raise ValueError("Spike bursts require matching start/stop bounds.")
    bounds = list(zip(starts, stops))
    if any(start < 0 or stop <= start for start, stop in bounds):
        raise ValueError("Spike burst bounds must be nonempty half-open intervals.")
    if child.valid_start is not None:
        assert child.valid_stop is not None
        if any(
            start < child.valid_start or stop > child.valid_stop
            for start, stop in bounds
        ):
            raise ValueError(
                "Spike bursts must lie within the stored population interval."
            )
    return bounds
