"""Paired spike outputs on a verified, common retained-frame interval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sqlmodel import Session, col, select

from cali.analysis._fov_inputs import _check_alignment
from cali.analysis._roi_analysis import cascade_acquisition_rate
from cali.sqlmodel import FOV, ROI, DataAnalysis, Traces
from cali.sqlmodel._engine import ensure_schema_current

from ._spike_data import SpikePlotData, spike_plot_data

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.sqlmodel._spike_settings import SpikeMethod


@dataclass
class SpikeComparison:
    """One ROI's paired outputs; amplitudes retain different scientific units."""

    roi_label: int
    source_trace: Traces
    oasis: SpikePlotData
    cascade: SpikePlotData
    valid_start: int
    valid_stop: int
    frame_rate_hz: float

    def values(self, method: SpikeMethod) -> np.ndarray:
        """Return only the common interval, preserving the original amplitudes."""
        return self._data(method).values[self.valid_start : self.valid_stop]

    def normalized_values(self, method: SpikeMethod) -> np.ndarray:
        """Scale independently by peak absolute amplitude in the common interval."""
        values = self.values(method)
        scale = float(np.max(np.abs(values)))
        return values / scale if scale else values.copy()

    def event_frames(self, method: SpikeMethod) -> np.ndarray:
        """Keep observed onsets, censoring the comparison interval's left boundary."""
        events = self._data(method).event_frames(onsets=True)
        return events[(events > self.valid_start) & (events < self.valid_stop)]

    def _data(self, method: SpikeMethod) -> SpikePlotData:
        if method not in ("oasis", "cascade"):
            raise ValueError("Comparisons support only OASIS and CASCADE.")
        return self.oasis if method == "oasis" else self.cascade


def paired_spike_comparison(
    trace: Traces,
    roi_label: int,
    analysis: DataAnalysis | None = None,
    *,
    require_threshold: bool = False,
) -> SpikeComparison | None:
    """Validate both outputs of one trace; never pair unrelated run histories."""
    if analysis is not None:
        for child in analysis.spike_analyses:
            if child.provenance_source == "legacy_unresolved":
                raise ValueError("Spike comparisons require resolved analysis sources.")
    oasis = spike_plot_data(trace, analysis, "oasis")
    cascade = spike_plot_data(trace, analysis, "cascade")
    if oasis is None or cascade is None:
        return None
    for data in (oasis, cascade):
        metric = data.metric
        if metric is not None and metric.spike_trace_id is None:
            if metric.spike_trace is not data.spike:
                raise ValueError(
                    "Spike comparisons must use the analyzed source trace."
                )
        if require_threshold and (metric is None or metric.threshold is None):
            return None
    if len(oasis.values) != len(cascade.values):
        raise ValueError("Compared spike outputs must have matching retained lengths.")
    window = trace.extraction_frame_window
    if window is None or window.retained_frame_count != len(oasis.values):
        raise ValueError(
            "Spike comparisons require a matching extraction frame window."
        )
    source_ids = {
        data.spike.inference_run.extraction_result_id for data in (oasis, cascade)
    }
    if len(source_ids) > 1:
        raise ValueError("Compared spike outputs must share their extraction source.")
    if window.extraction_result_id is not None and source_ids != {
        window.extraction_result_id
    }:
        raise ValueError(
            "Compared outputs must match the extraction frame window source."
        )
    frame_rate = cascade_acquisition_rate(cascade.spike, trace)
    _check_alignment(trace, trace, len(oasis.values))
    if trace.x_axis is not None:
        axis = np.asarray(trace.x_axis, dtype=float)
        if trace.x_axis_units not in {"ms", "frames"}:
            raise ValueError("Spike comparisons require known time-axis units.")
        interval = 1000 / frame_rate if trace.x_axis_units == "ms" else 1
        if not np.all(np.isfinite(axis)) or not np.allclose(
            np.diff(axis), interval, rtol=0.01, atol=1e-6
        ):
            raise ValueError("Compared time axes must match the acquisition rate.")
    start = max(oasis.spike.valid_start, cascade.spike.valid_start)
    stop = min(oasis.spike.resolved_valid_stop, cascade.spike.resolved_valid_stop)
    if start >= stop:
        return None
    return SpikeComparison(roi_label, trace, oasis, cascade, start, stop, frame_rate)


def get_spike_comparisons(
    engine: Engine,
    fov_name: str,
    run_id: int,
    rois: list[int] | None = None,
    *,
    require_threshold: bool = False,
) -> list[SpikeComparison]:
    """Read exact-run ROI pairs, including neurons inactive for either method."""
    if run_id is None:
        raise ValueError("Spike comparisons require one selected analysis run.")
    ensure_schema_current(engine)
    pairs: list[SpikeComparison] = []
    labels: set[int] = set()
    with Session(engine) as session:
        stmt = (
            select(ROI, Traces, DataAnalysis)
            .join(FOV, col(ROI.fov_id) == col(FOV.id))
            .join(
                Traces,
                (col(Traces.roi_id) == col(ROI.id))
                & (col(Traces.analysis_result_id) == run_id),
            )
            .outerjoin(
                DataAnalysis,
                (col(DataAnalysis.roi_id) == col(ROI.id))
                & (col(DataAnalysis.analysis_result_id) == run_id),
            )
            .where(FOV.name == fov_name)
            .order_by(col(ROI.label_value))
        )
        if rois is not None:
            stmt = stmt.where(col(ROI.label_value).in_(rois))
        for roi, trace, analysis in session.exec(stmt).all():
            pair = paired_spike_comparison(
                trace, roi.label_value, analysis, require_threshold=require_threshold
            )
            if pair is None:
                continue
            if pair.roi_label in labels:
                raise ValueError("Compared ROI labels must be unique within the FOV.")
            labels.add(pair.roi_label)
            if pairs:
                _check_alignment(pairs[0].source_trace, trace, len(pair.oasis.values))
                if not np.isclose(
                    pairs[0].frame_rate_hz, pair.frame_rate_hz, rtol=1e-9
                ):
                    raise ValueError(
                        "Compared ROI rows must share an acquisition rate."
                    )
            pairs.append(pair)
    return pairs
