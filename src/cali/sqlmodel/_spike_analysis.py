"""Method-bound ROI spike results, independent of shared calcium analysis."""

import math
from typing import TYPE_CHECKING, Any, Optional, cast

from sqlalchemy import UniqueConstraint, inspect
from sqlmodel import Field, Relationship, select

from ._result_json import ResultJSON
from ._trace_provenance import SpikeTrace

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ._model import CaliResult, DataAnalysis
    from ._spike_settings import SpikeMethod


LEGACY_SPIKE_METRICS = {
    "inferred_spikes_threshold": "threshold",
    "inferred_spikes_frequency": "suprathreshold_sample_rate_hz",
    "inferred_spikes_rising_edge_frequency": "suprathreshold_rising_edge_rate_hz",
}


class SpikeAnalysis(ResultJSON, table=True):  # type: ignore[call-arg, unused-ignore]
    """An applied threshold and rates for one spike trace and analysis run.

    A nullable trace is reserved for documented synthetic API inputs or quarantined
    legacy metrics. Production analysis always supplies an explicit spike trace.
    Method/units remain recorded if the referenced trace is later deleted.
    """

    __tablename__ = "spike_analysis"
    __table_args__ = (
        UniqueConstraint("spike_trace_id", "analysis_result_id"),
        UniqueConstraint("data_analysis_id", "method"),
    )

    id: int | None = Field(default=None, primary_key=True)
    data_analysis_id: int | None = Field(
        default=None, foreign_key="data_analysis.id", nullable=False, ondelete="CASCADE"
    )
    analysis_result_id: int | None = Field(
        default=None, foreign_key="analysis_result.id", ondelete="CASCADE", index=True
    )
    spike_trace_id: int | None = Field(
        default=None, foreign_key="spike_trace.id", ondelete="SET NULL", index=True
    )
    method: str = "oasis"
    units: str = "a.u."
    threshold: float | None = None
    threshold_mode: str | None = None
    suprathreshold_sample_rate_hz: float | None = None
    suprathreshold_rising_edge_rate_hz: float | None = None
    expected_spike_rate_hz: float | None = None
    expected_spike_count: float | None = None
    suprathreshold_excursion_rate_hz: float | None = None
    spike_active: bool | None = None
    provenance_source: str = "analysis"

    data_analysis: "DataAnalysis" = Relationship(back_populates="spike_analyses")
    analysis_result: Optional["CaliResult"] = Relationship(
        back_populates="spike_analysis_results"
    )
    spike_trace: Optional["SpikeTrace"] = Relationship(
        sa_relationship_kwargs={"lazy": "selectin"}
    )


def normalize_spike_analyses(session: "Session", *_: Any) -> None:
    """Bind legacy API inputs where unambiguous and validate normalized writes."""
    from ._model import ROI, CaliResult, DataAnalysis, Traces

    new_objects = list(session.new)
    parents = [
        obj
        for obj in new_objects + list(session.dirty)
        if isinstance(obj, DataAnalysis)
    ]
    for parent in parents:
        seen: set[str] = set()
        for child in parent.spike_analyses:
            if child.method in seen:
                raise ValueError(f"Duplicate spike analyses for {child.method}.")
            seen.add(child.method)
            if (
                child.analysis_result_id is not None
                and parent.analysis_result_id is not None
                and child.analysis_result_id != parent.analysis_result_id
            ):
                raise ValueError(
                    "Spike analysis and calcium parent must use the same run."
                )
            if parent.analysis_result is not None:
                child.analysis_result = parent.analysis_result
            else:
                child.analysis_result_id = parent.analysis_result_id
            if child.spike_trace is None and child.spike_trace_id is not None:
                child.spike_trace = session.get(SpikeTrace, child.spike_trace_id)
                if child.spike_trace is None:
                    raise ValueError("The analyzed spike trace does not exist.")
            if (
                child.spike_trace is None
                and child.provenance_source == "synthetic_legacy_api"
            ):
                # Bind only a unique trace for this ROI and run. Plot/test stubs with
                # no trace retain explicit synthetic provenance, without fake arrays.
                roi = parent.roi
                if roi is None and parent.roi_id is not None:
                    roi = session.get(ROI, parent.roi_id)
                candidates = [obj for obj in new_objects if isinstance(obj, Traces)]
                if roi is not None:
                    candidates += list(roi.traces_history)
                if parent.roi_id is not None and parent.analysis_result_id is not None:
                    candidates += list(
                        session.scalars(
                            select(Traces).where(
                                Traces.roi_id == parent.roi_id,
                                Traces.analysis_result_id == parent.analysis_result_id,
                            )
                        )
                    )
                matches: dict[int, SpikeTrace] = {}
                for trace in candidates:
                    same_roi = (
                        trace.roi is roi
                        if roi is not None
                        else (
                            parent.roi_id is not None and trace.roi_id == parent.roi_id
                        )
                    )
                    same_run = (
                        parent.analysis_result_id is not None
                        and trace.analysis_result_id == parent.analysis_result_id
                    ) or (
                        parent.analysis_result is not None
                        and trace.analysis_result is parent.analysis_result
                    )
                    spike = trace.get_spike_trace(cast("SpikeMethod", child.method))
                    if same_roi and same_run and spike is not None:
                        matches[id(spike)] = spike
                if len(matches) == 1:
                    child.spike_trace = next(iter(matches.values()))
                    child.provenance_source = "legacy_api_bound"
                elif len(matches) > 1:
                    raise ValueError(
                        "Ambiguous spike analysis source: select a spike trace."
                    )
            if child.threshold_mode is None and child.spike_trace is not None:
                owner = parent.analysis_result
                if owner is None and parent.analysis_result_id is not None:
                    owner = session.get(CaliResult, parent.analysis_result_id)
                if owner is not None and owner.analysis_settings_id is not None:
                    from ._model import AnalysisSettings

                    settings = session.get(AnalysisSettings, owner.analysis_settings_id)
                    if settings is not None:
                        child.threshold_mode = settings.get_spike_settings(
                            cast("SpikeMethod", child.method)
                        ).threshold_mode

    children = {
        id(obj): obj
        for obj in list(session.new)
        + list(session.dirty)
        + [child for parent in parents for child in parent.spike_analyses]
        if isinstance(obj, SpikeAnalysis)
    }
    for child in children.values():
        if not isinstance(child, SpikeAnalysis):
            continue
        if child.provenance_source == "legacy_unresolved":
            state = inspect(child)
            assert state is not None
            if child in session.new or any(
                state.attrs[name].history.has_changes()
                for name in type(child).model_fields
            ):
                raise ValueError("Unresolved legacy spike metrics are read-only.")
            continue
        expected_units = {"oasis": "a.u.", "cascade": "spikes/frame"}
        modes = {"oasis": {"global", "multiplier"}, "cascade": {"global", "cascade_ap"}}
        if (
            child.method not in expected_units
            or child.units != expected_units[child.method]
        ):
            raise ValueError("Spike analysis method and units must match.")
        if (
            child.threshold_mode is not None
            and child.threshold_mode not in modes[child.method]
        ):
            raise ValueError("Threshold mode is incompatible with the spike method.")
        spike = child.spike_trace
        if spike is not None:
            run = spike.inference_run
            if run.method != child.method or run.units != child.units:
                raise ValueError("Spike analysis must match its trace provenance.")
            parent = child.data_analysis
            if parent is not None:
                if (
                    parent.analysis_result_id is not None
                    and child.analysis_result_id is not None
                    and parent.analysis_result_id != child.analysis_result_id
                ):
                    raise ValueError(
                        "Spike analysis and calcium parent must use the same run."
                    )
                trace = spike.trace
                if trace is not None:
                    if (
                        parent.roi_id is not None
                        and trace.roi_id is not None
                        and parent.roi_id != trace.roi_id
                    ) or (
                        parent.roi is not None
                        and trace.roi is not None
                        and parent.roi is not trace.roi
                    ):
                        raise ValueError(
                            "Spike analysis must use its parent's ROI trace."
                        )
                    if (
                        parent.analysis_result_id is not None
                        and trace.analysis_result_id is not None
                        and parent.analysis_result_id != trace.analysis_result_id
                    ) or (
                        parent.analysis_result is not None
                        and trace.analysis_result is not None
                        and parent.analysis_result is not trace.analysis_result
                    ):
                        raise ValueError(
                            "Spike analysis must use the trace stored for its run."
                        )
        elif child.provenance_source not in {"synthetic_legacy_api", "legacy_import"}:
            raise ValueError("A spike analysis requires an explicit spike trace.")
        if child.method == "cascade" and (
            child.suprathreshold_sample_rate_hz is not None
            or child.suprathreshold_rising_edge_rate_hz is not None
        ):
            raise ValueError("CASCADE cannot store OASIS sample/rising-edge rates.")
        if child.method == "oasis" and any(
            value is not None
            for value in (
                child.expected_spike_rate_hz,
                child.expected_spike_count,
                child.suprathreshold_excursion_rate_hz,
            )
        ):
            raise ValueError("OASIS cannot store CASCADE expected-rate metrics.")
        for name in (
            "threshold",
            "suprathreshold_sample_rate_hz",
            "suprathreshold_rising_edge_rate_hz",
            "expected_spike_rate_hz",
            "expected_spike_count",
            "suprathreshold_excursion_rate_hz",
        ):
            value = getattr(child, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Spike metrics must be finite and non-negative.")
