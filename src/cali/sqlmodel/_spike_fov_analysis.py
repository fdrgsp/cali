"""Method-bound FOV spike results and their input ordering."""

from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import UniqueConstraint, inspect
from sqlmodel import JSON, Column, Field, Relationship, select

from ._result_json import ResultJSON
from ._trace_provenance import SpikeInferenceRun

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ._model import FOV, CaliResult, FOVAnalysis


SPIKE_FOV_METRICS = (
    "spike_max_lag_correlation_matrix",
    "global_spike_max_lag_correlation",
    "spike_max_lag_values_matrix",
    "spike_max_lag_correlation_matrix_rising_edges",
    "global_spike_max_lag_correlation_rising_edges",
    "spike_max_lag_values_matrix_rising_edges",
    "spike_ccg_zscore_matrix",
    "spike_ccg_zscore_matrix_rising_edges",
    "fraction_significant_ccg_pairs",
    "fraction_significant_ccg_pairs_rising_edges",
    "spike_jitter_synchrony_matrix",
    "global_spike_jitter_synchrony",
    "spike_jitter_synchrony_matrix_rising_edges",
    "global_spike_jitter_synchrony_rising_edges",
    "spike_burst_count",
    "spike_burst_avg_duration",
    "spike_burst_avg_interval",
    "spike_burst_starts",
    "spike_burst_ends",
    "spike_population_activity",
    "spike_population_activity_raw",
)


class SpikeFOVAnalysis(ResultJSON, table=True):  # type: ignore[call-arg, unused-ignore]
    """Spike matrices, synchrony, and bursts for one method, FOV, and run."""

    __tablename__ = "spike_fov_analysis"
    __table_args__ = (
        UniqueConstraint("fov_id", "spike_inference_run_id", "analysis_result_id"),
        UniqueConstraint("fov_analysis_id", "method"),
    )

    id: int | None = Field(default=None, primary_key=True)
    fov_analysis_id: int | None = Field(
        default=None, foreign_key="fov_analysis.id", nullable=False, ondelete="CASCADE"
    )
    fov_id: int | None = Field(default=None, foreign_key="fov.id", ondelete="CASCADE")
    analysis_result_id: int | None = Field(
        default=None, foreign_key="analysis_result.id", ondelete="CASCADE", index=True
    )
    spike_inference_run_id: int | None = Field(
        default=None,
        foreign_key="spike_inference_run.id",
        ondelete="SET NULL",
        index=True,
    )
    method: str = "oasis"
    units: str = "a.u."
    provenance_source: str = "analysis"
    active_roi_labels: list[int] | None = Field(default=None, sa_column=Column(JSON))

    spike_max_lag_correlation_matrix: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    global_spike_max_lag_correlation: float | None = None
    spike_max_lag_values_matrix: list[list[int]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    spike_max_lag_correlation_matrix_rising_edges: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    global_spike_max_lag_correlation_rising_edges: float | None = None
    spike_max_lag_values_matrix_rising_edges: list[list[int]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    spike_ccg_zscore_matrix: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    spike_ccg_zscore_matrix_rising_edges: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    fraction_significant_ccg_pairs: float | None = None
    fraction_significant_ccg_pairs_rising_edges: float | None = None
    spike_jitter_synchrony_matrix: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    global_spike_jitter_synchrony: float | None = None
    spike_jitter_synchrony_matrix_rising_edges: list[list[float]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    global_spike_jitter_synchrony_rising_edges: float | None = None
    spike_burst_count: int | None = None
    spike_burst_avg_duration: float | None = None
    spike_burst_avg_interval: float | None = None
    spike_burst_starts: list[int] | None = Field(default=None, sa_column=Column(JSON))
    spike_burst_ends: list[int] | None = Field(default=None, sa_column=Column(JSON))
    spike_population_activity: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    spike_population_activity_raw: list[float] | None = Field(
        default=None, sa_column=Column(JSON)
    )

    fov_analysis: "FOVAnalysis" = Relationship(back_populates="spike_analyses")
    analysis_result: Optional["CaliResult"] = Relationship(
        back_populates="spike_fov_analysis_results"
    )
    fov: Optional["FOV"] = Relationship()
    inference_run: Optional["SpikeInferenceRun"] = Relationship(
        sa_relationship_kwargs={"lazy": "selectin"}
    )


def normalize_spike_fov_analyses(session: "Session", *_: Any) -> None:
    """Bind canonical inference provenance after staged traces are normalized."""
    from ._model import FOV, ROI, FOVAnalysis, Traces

    objects = list(session.new) + list(session.dirty)
    parents = list(
        {
            id(parent): parent
            for parent in [obj for obj in objects if isinstance(obj, FOVAnalysis)]
            + [obj.fov_analysis for obj in objects if isinstance(obj, SpikeFOVAnalysis)]
            if parent is not None
        }.values()
    )
    for parent in parents:
        seen: set[str] = set()
        for child in parent.spike_analyses:
            if child.method in seen:
                raise ValueError(f"Duplicate FOV spike analyses for {child.method}.")
            seen.add(child.method)
            if child.provenance_source == "legacy_unresolved":
                continue
            if parent.fov is None and parent.fov_id is not None:
                parent.fov = session.get(FOV, parent.fov_id)
            if (
                child.analysis_result_id is not None
                and parent.analysis_result_id is not None
                and child.analysis_result_id != parent.analysis_result_id
            ):
                raise ValueError(
                    "FOV spike analysis and calcium parent must use the same run."
                )
            if (
                child.fov_id is not None
                and parent.fov_id is not None
                and child.fov_id != parent.fov_id
            ):
                raise ValueError(
                    "FOV spike analysis and calcium parent must use the same FOV."
                )
            if parent.analysis_result is not None:
                child.analysis_result = parent.analysis_result
            else:
                child.analysis_result_id = parent.analysis_result_id
            if parent.fov is not None:
                child.fov = parent.fov
            else:
                child.fov_id = parent.fov_id
            if child.inference_run is None and child.spike_inference_run_id is not None:
                child.inference_run = session.get(
                    SpikeInferenceRun, child.spike_inference_run_id
                )
                if child.inference_run is None:
                    raise ValueError("The FOV spike inference run does not exist.")
            candidates = [obj for obj in session.new if isinstance(obj, Traces)]
            if parent.fov is not None:
                candidates += [
                    trace for roi in parent.fov.rois for trace in roi.traces_history
                ]
            if parent.fov_id is not None and parent.analysis_result_id is not None:
                candidates += list(
                    session.scalars(
                        select(Traces)
                        .join(ROI)
                        .where(
                            ROI.fov_id == parent.fov_id,
                            Traces.analysis_result_id == parent.analysis_result_id,
                        )
                    )
                )
            runs: dict[int, SpikeInferenceRun] = {}
            for trace in candidates:
                trace_roi = trace.roi
                if trace_roi is None and trace.roi_id is not None:
                    trace_roi = session.get(ROI, trace.roi_id)
                trace_fov_id = None
                if trace_roi is not None:
                    trace_fov_id = trace_roi.fov_id
                    if trace_fov_id is None and trace_roi.fov is not None:
                        trace_fov_id = trace_roi.fov.id
                same_fov = trace_roi is not None and (
                    (parent.fov is not None and trace_roi.fov is parent.fov)
                    or (parent.fov_id is not None and trace_fov_id == parent.fov_id)
                )
                same_run = (
                    (
                        parent.analysis_result_id is not None
                        and trace.analysis_result_id == parent.analysis_result_id
                    )
                    or (
                        parent.analysis_result is not None
                        and trace.analysis_result is parent.analysis_result
                    )
                    or (
                        parent.analysis_result_id is None
                        and parent.analysis_result is None
                        and trace.analysis_result_id is None
                        and trace.analysis_result is None
                        and (
                            child.inference_run is None
                            or any(
                                spike.inference_run is child.inference_run
                                for spike in trace.spike_traces
                            )
                        )
                    )
                    or (
                        parent.analysis_result_id is None
                        and parent.analysis_result is None
                        and child.inference_run is not None
                        and any(
                            spike.inference_run is child.inference_run
                            for spike in trace.spike_traces
                        )
                    )
                )
                if same_fov and same_run:
                    for spike in trace.spike_traces:
                        if spike.inference_run.method == child.method:
                            runs[id(spike.inference_run)] = spike.inference_run
            if len(runs) > 1:
                raise ValueError("A FOV spike result cannot combine inference runs.")
            if runs:
                canonical_run = next(iter(runs.values()))
                if (
                    child.inference_run is not None
                    and child.inference_run is not canonical_run
                ):
                    raise ValueError(
                        "FOV spike analysis must use its stored traces' inference run."
                    )
                child.inference_run = canonical_run
                if child.provenance_source == "synthetic_legacy_api":
                    child.provenance_source = "legacy_api_bound"
            elif child.provenance_source == "analysis":
                raise ValueError("A FOV spike analysis requires stored source traces.")
    children = {
        id(obj): obj
        for obj in objects
        + [child for parent in parents for child in parent.spike_analyses]
        if isinstance(obj, SpikeFOVAnalysis)
    }
    for child in children.values():
        if child.provenance_source == "legacy_unresolved":
            state = inspect(child)
            assert state is not None
            if child in session.new or any(
                state.attrs[name].history.has_changes()
                for name in type(child).model_fields
            ):
                raise ValueError("Unresolved legacy FOV spike metrics are read-only.")
            continue
        if (
            child.method not in {"oasis", "cascade"}
            or child.units != {"oasis": "a.u.", "cascade": "spikes/frame"}[child.method]
        ):
            raise ValueError("FOV spike analysis method and units must match.")
        if child.inference_run is not None:
            if (
                child.inference_run.method != child.method
                or child.inference_run.units != child.units
            ):
                raise ValueError(
                    "FOV spike analysis must match its inference provenance."
                )
        elif child.provenance_source not in {"synthetic_legacy_api", "legacy_import"}:
            raise ValueError("A FOV spike analysis requires an explicit inference run.")


def bind_fov_spike_source(parent: "FOVAnalysis", fov: "FOV") -> None:
    """Record the actually selected source for standalone analysis without a run ID."""
    for child in parent.spike_analyses:
        runs: dict[int, SpikeInferenceRun] = {}
        for roi in fov.rois:
            if roi.label_value not in (child.active_roi_labels or []):
                continue
            traces = getattr(roi, "_new_traces", None)
            trace = (
                traces[-1] if traces else getattr(roi, "_analysis_source_trace", None)
            )
            if trace is None and roi.traces_history:
                trace = roi.traces_history[-1]
            if trace is not None:
                for spike in trace.spike_traces:
                    run = spike.inference_run
                    if run.method == child.method:
                        runs[run.id if run.id is not None else id(run)] = run
        if len(runs) == 1:
            run = next(iter(runs.values()))
            if run.id is not None:
                child.spike_inference_run_id = run.id
