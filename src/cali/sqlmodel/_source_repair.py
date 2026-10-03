"""Explicit source selection for verified, quarantined legacy result snapshots."""

from collections import Counter
from dataclasses import dataclass

from sqlmodel import Session, select

from ._engine import ensure_schema_current
from ._model import (
    CaliResult,
    DataAnalysis,
    FOVAnalysis,
    SpikeAnalysis,
    SpikeFOVAnalysis,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
)
from ._source_provenance import MigrationIssue


@dataclass(frozen=True)
class SourceRepairPreview:
    """Verified source comparison without changes to a stored legacy result."""

    result_id: int
    source_result_id: int
    positions: tuple[int, ...]
    trace_count: int
    roi_metric_count: int
    fov_metric_count: int
    spike_outputs: tuple[tuple[str, str], ...]
    retained_frame_counts: tuple[int, ...]
    source_start_frames: tuple[int, ...]


@dataclass
class _SourceRepairPlan:
    result: CaliResult
    source: CaliResult
    pairs: list[tuple[Traces, Traces]]
    roi_links: list[tuple[SpikeAnalysis, SpikeTrace]]
    fov_links: list[tuple[SpikeFOVAnalysis, SpikeInferenceRun]]
    issues: list[MigrationIssue]
    evidence: dict


def _spike_for_method(trace: Traces, method: str) -> SpikeTrace | None:
    if method == "oasis":
        return trace.get_spike_trace("oasis")
    if method == "cascade":
        return trace.get_spike_trace("cascade")
    raise ValueError(f"Unknown spike method: {method}.")


def _has_unresolved_metrics(result: CaliResult) -> bool:
    parents: list[DataAnalysis | FOVAnalysis] = [
        *result.data_analysis_results,
        *result.fov_analysis_results,
    ]
    return any(
        child.provenance_source == "legacy_unresolved"
        for parent in parents
        for child in parent.spike_analyses
    )


def _same_payload(copied: Traces, source: Traces) -> bool:
    arrays = (
        "raw_trace",
        "corrected_trace",
        "neuropil_trace",
        "dff",
        "den_dff",
        "x_axis",
        "x_axis_units",
    )
    if any(getattr(copied, name) != getattr(source, name) for name in arrays):
        return False
    first, second = copied.extraction_frame_window, source.extraction_frame_window
    if first is None or second is None:
        return False
    coordinates = (
        "requested_discard_value",
        "requested_discard_unit",
        "original_frame_count",
        "retained_frame_count",
        "source_start_frame",
        "source_start_time_ms",
        "source_time_origin_ms",
        "source_start_timestamp_ms",
        "discarded_duration_ms",
    )
    if any(getattr(first, name) != getattr(second, name) for name in coordinates):
        return False
    methods = [
        child.inference_run.method
        for child in copied.spike_traces
        if child.inference_run is not None
    ]
    if len(methods) != len(set(methods)):
        return False
    for child in copied.spike_traces:
        run = child.inference_run
        if run is None:
            return False
        original = _spike_for_method(source, run.method)
        if (
            original is None
            or original.inference_run is None
            or original.inference_run.units != run.units
            or original.values != child.values
            or original.valid_start != child.valid_start
            or original.resolved_valid_stop != child.resolved_valid_stop
        ):
            return False
    return True


def _prepare_source_repair(
    session: Session, result_id: int, source_extraction_result_id: int
) -> _SourceRepairPlan:
    """Verify the entire snapshot before previewing or applying source links."""
    ensure_schema_current(session.get_bind())
    with session.no_autoflush:
        result: CaliResult | None = session.get(CaliResult, result_id)
        source: CaliResult | None = session.get(CaliResult, source_extraction_result_id)
        if (
            result is None
            or source is None
            or result is source
            or not (
                (result.legacy_trace_resolution or "").startswith("unresolved")
                or result.legacy_trace_resolution == "multiple_sources"
            )
            or not source.positions_extracted
            or source.source_extraction_result_id != source.id
            or (source.legacy_trace_resolution or "").startswith("unresolved")
            or _has_unresolved_metrics(source)
            or any(
                getattr(result, field) != getattr(source, field)
                for field in (
                    "experiment",
                    "detection_settings_id",
                    "extraction_settings_id",
                )
            )
        ):
            raise ValueError(
                "Choose a resolved extraction with matching experiment, detection, "
                "and extraction settings for an unresolved legacy result."
            )
        issues = list(
            session.exec(
                select(MigrationIssue).where(
                    MigrationIssue.analysis_result_id == result_id
                )
            ).all()
        )
        if not any(not issue.resolved for issue in issues):
            raise ValueError("The legacy result has no unresolved migration audit.")

        targets = list(result.traces)
        originals = list(source.traces)
        if not targets or any(trace.roi_id is None for trace in targets):
            raise ValueError(
                "Source repair requires stored ROI traces; re-analyze instead."
            )
        if len({trace.roi_id for trace in targets}) != len(targets):
            raise ValueError("Duplicate legacy ROI traces require a fresh analysis.")
        pairs = []
        by_roi = {trace.roi_id: trace for trace in targets}
        for trace in targets:
            candidates = [
                original for original in originals if original.roi_id == trace.roi_id
            ]
            if len(candidates) != 1 or not _same_payload(trace, candidates[0]):
                raise ValueError(
                    f"ROI {trace.roi_id}: selected source does not match the stored "
                    "legacy payload and coordinates. Run a fresh analysis."
                )
            original = candidates[0]
            window = original.extraction_frame_window
            if (
                window is None
                or window.extraction_result_id != source.id
                or any(
                    child.inference_run is None
                    or child.inference_run.extraction_result_id != source.id
                    for child in original.spike_traces
                )
            ):
                raise ValueError(
                    "The selected extraction has unresolved trace provenance."
                )
            pairs.append((trace, original))

        roi_links: list[tuple[SpikeAnalysis, SpikeTrace]] = []
        roi_parents = list(result.data_analysis_results)
        if any(
            count > 1
            for count in Counter(parent.roi_id for parent in roi_parents).values()
        ):
            raise ValueError("Duplicate legacy ROI metrics require a fresh analysis.")
        for roi_parent in roi_parents:
            roi_trace = by_roi.get(roi_parent.roi_id)
            for roi_child in roi_parent.spike_analyses:
                roi_spike = (
                    _spike_for_method(roi_trace, roi_child.method)
                    if roi_trace is not None
                    else None
                )
                if (
                    roi_spike is None
                    or roi_spike.inference_run.units != roi_child.units
                ):
                    raise ValueError(
                        "Legacy ROI metrics have no matching stored spike trace."
                    )
                # Validate cached semantics before restoring a formerly read-only row.
                SpikeAnalysis.model_validate(
                    roi_child,
                    update={
                        "provenance_source": "legacy_source_selected",
                        "spike_trace": roi_spike,
                        "spike_trace_id": roi_spike.id,
                    },
                )
                roi_links.append((roi_child, roi_spike))

        fov_links: list[tuple[SpikeFOVAnalysis, SpikeInferenceRun]] = []
        fov_parents = list(result.fov_analysis_results)
        if any(
            count > 1
            for count in Counter(parent.fov_id for parent in fov_parents).values()
        ):
            raise ValueError("Duplicate legacy FOV metrics require a fresh analysis.")
        for fov_parent in fov_parents:
            for fov_child in fov_parent.spike_analyses:
                labels = fov_child.active_roi_labels
                selected = [
                    original
                    for _, original in pairs
                    if original.roi is not None
                    and original.roi.fov_id == fov_parent.fov_id
                    and labels
                    and original.roi.label_value in labels
                ]
                if not labels or Counter(
                    trace.roi.label_value for trace in selected
                ) != Counter(labels):
                    raise ValueError(
                        "Legacy FOV ordering has no unique source coverage."
                    )
                runs = {
                    spike.inference_run.id: spike.inference_run
                    for trace in selected
                    if (spike := _spike_for_method(trace, fov_child.method)) is not None
                    and spike.inference_run is not None
                    and spike.inference_run.units == fov_child.units
                }
                if len(runs) != 1 or any(
                    _spike_for_method(trace, fov_child.method) is None
                    for trace in selected
                ):
                    raise ValueError(
                        "Legacy FOV metrics require one matching inference run."
                    )
                run = next(iter(runs.values()))
                SpikeFOVAnalysis.model_validate(
                    fov_child,
                    update={
                        "provenance_source": "legacy_source_selected",
                        "inference_run": run,
                        "spike_inference_run_id": run.id,
                    },
                )
                fov_links.append((fov_child, run))

        evidence = {
            "selected_source_result_id": source.id,
            "previous_resolution": result.legacy_trace_resolution,
            "previous_positions_extracted": result.positions_extracted,
            "trace_pairs": [
                {
                    "trace_id": trace.id,
                    "source_trace_id": original.id,
                    "previous_window_id": trace.extraction_frame_window_id,
                    "previous_inference_run_ids": [
                        child.spike_inference_run_id for child in trace.spike_traces
                    ],
                }
                for trace, original in pairs
            ],
        }
        return _SourceRepairPlan(
            result, source, pairs, roi_links, fov_links, issues, evidence
        )


def preview_legacy_result_source(
    session: Session, result_id: int, source_extraction_result_id: int
) -> SourceRepairPreview:
    """Compare a selected source using the same checks as repair, without writing.

    Incompatible sources raise ValueError. Pending caller work is neither flushed
    nor changed, and no trace, metric, or audit link is modified.
    """
    plan = _prepare_source_repair(session, result_id, source_extraction_result_id)
    with session.no_autoflush:
        originals = [original for _, original in plan.pairs]
        return SourceRepairPreview(
            result_id=result_id,
            source_result_id=source_extraction_result_id,
            positions=tuple(
                sorted(
                    {
                        trace.roi.fov.position_index
                        for trace in originals
                        if trace.roi is not None
                        and trace.roi.fov is not None
                        and trace.roi.fov.position_index is not None
                    }
                )
            ),
            trace_count=len(plan.pairs),
            roi_metric_count=len(plan.roi_links),
            fov_metric_count=len(plan.fov_links),
            spike_outputs=tuple(
                sorted(
                    {
                        (child.inference_run.method, child.inference_run.units)
                        for trace, _ in plan.pairs
                        for child in trace.spike_traces
                    }
                )
            ),
            retained_frame_counts=tuple(
                sorted(
                    {
                        trace.extraction_frame_window.retained_frame_count
                        for trace in originals
                        if trace.extraction_frame_window is not None
                        and trace.extraction_frame_window.retained_frame_count
                        is not None
                    }
                )
            ),
            source_start_frames=tuple(
                sorted(
                    {
                        trace.extraction_frame_window.source_start_frame
                        for trace in originals
                        if trace.extraction_frame_window is not None
                    }
                )
            ),
        )


def select_legacy_result_source(
    session: Session, result_id: int, source_extraction_result_id: int
) -> CaliResult:
    """Stage an audited repair after the caller explicitly chooses an extraction.

    Verify the entire result before changing any links. Arrays, metrics, activity,
    ordering, and existing provenance records are preserved. Incompatible sources
    require a fresh analysis instead. Only this result is repaired; dependent
    histories keep their own audits until selected separately. The caller owns
    the transaction and must commit (or roll back) the staged changes.
    """
    plan = _prepare_source_repair(session, result_id, source_extraction_result_id)
    result, source = plan.result, plan.source
    with session.no_autoflush:
        # All validation has completed. Reuse original metadata while keeping the
        # result's own trace/metric rows and the old shared audited records intact.
        for trace, original in plan.pairs:
            trace.extraction_frame_window = original.extraction_frame_window
            for copied_spike in trace.spike_traces:
                matching_spike = _spike_for_method(
                    original, copied_spike.inference_run.method
                )
                assert matching_spike is not None  # Verified before mutation.
                copied_spike.inference_run = matching_spike.inference_run
        for roi_child, roi_spike in plan.roi_links:
            roi_child.spike_trace = roi_spike
            roi_child.provenance_source = "legacy_source_selected"
        for fov_child, run in plan.fov_links:
            fov_child.inference_run = run
            fov_child.provenance_source = "legacy_source_selected"
        result.source_extraction_result_id = source.id
        result.legacy_trace_resolution = "source_selected"
        result.positions_extracted = None
        for issue in plan.issues:
            if (
                issue.code.startswith("unresolved")
                or issue.code == "multiple_extraction_sources"
            ):
                issue.resolved = True
                issue.details = {
                    **issue.details,
                    "selected_source_result_id": source.id,
                }
        session.add(
            MigrationIssue(
                analysis_result_id=result_id,
                code="legacy_source_selected",
                details=plan.evidence,
                resolved=True,
            )
        )
        return result
