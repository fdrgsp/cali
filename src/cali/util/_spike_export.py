"""Method-qualified spike results with explicit threshold and population coordinates."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from sqlmodel import Session, col, select

from cali.sqlmodel import (
    FOV,
    ROI,
    DataAnalysis,
    ExtractionFrameWindow,
    FOVAnalysis,
    SpikeAnalysis,
    SpikeFOVAnalysis,
)
from cali.sqlmodel._engine import ensure_schema_current
from cali.sqlmodel._spike_fov_analysis import SPIKE_FOV_METRICS
from cali.sqlmodel._spike_settings import canonical_spike_methods

_ROI_METRICS = (
    "threshold",
    "suprathreshold_sample_rate_hz",
    "suprathreshold_rising_edge_rate_hz",
    "expected_spike_rate_hz",
    "expected_spike_count",
    "suprathreshold_excursion_rate_hz",
)
_ROI_METRIC_UNITS = {
    "threshold": "threshold_units",
    "suprathreshold_sample_rate_hz": "Hz",
    "suprathreshold_rising_edge_rate_hz": "Hz",
    "expected_spike_rate_hz": "Hz",
    "expected_spike_count": "spikes",
    "suprathreshold_excursion_rate_hz": "Hz",
}


def _fov_metric_units(name: str) -> str:
    if "values_matrix" in name or name in {"spike_burst_starts", "spike_burst_ends"}:
        return "frames"
    if "zscore" in name:
        return "standard deviations"
    if "correlation" in name:
        return "probability per trigger"
    if "jitter" in name:
        return "score"
    if "significant" in name:
        return "fraction of pairs"
    if "population_activity" in name:
        return "fraction of active ROIs"
    if name == "spike_burst_count":
        return "bursts"
    return "seconds"


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.engine import Engine

    from cali.sqlmodel._spike_settings import SpikeMethod


def _selected_methods(
    available: set[str], requested: Sequence[str] | None
) -> tuple[SpikeMethod, ...]:
    methods = canonical_spike_methods(list(available)) if available else ()
    if requested is not None:
        methods = canonical_spike_methods(tuple(requested))
        if missing := set(methods) - available:
            raise ValueError(f"No stored spike analysis results for {sorted(missing)}.")
    return methods


def _validate_method(method: str, units: str) -> None:
    if (
        method not in {"oasis", "cascade"}
        or units != {"oasis": "a.u.", "cascade": "spikes/frame"}[method]
    ):
        raise ValueError("Exported spike method and units must match.")


def _threshold_record(child: SpikeAnalysis, label: int | None) -> dict:
    _validate_method(child.method, child.units)
    threshold = child.threshold
    exported_threshold: float | str | None = threshold
    if threshold is not None and not math.isfinite(threshold):
        if (
            child.method != "oasis"
            or child.threshold_mode != "multiplier"
            or threshold != math.inf
        ):
            raise ValueError(
                "Only OASIS's disabled multiplier threshold may be infinite."
            )
        exported_threshold = "+infinity"
    return {
        "roi_label": label,
        "method": child.method,
        "threshold": exported_threshold,
        "threshold_mode": child.threshold_mode,
        "threshold_units": child.units,
    }


def fov_spike_metadata(
    session: Session, parent: FOVAnalysis, fov: FOV, child: SpikeFOVAnalysis
) -> dict:
    """Describe the exact input population, including every applied ROI threshold."""
    _validate_method(child.method, child.units)
    child.validate_population_coordinates()
    run = child.inference_run
    if run is not None and (run.method != child.method or run.units != child.units):
        raise ValueError("Exported FOV result must match its inference provenance.")
    thresholds = []
    stmt = (
        select(ROI, DataAnalysis)
        .join(DataAnalysis, col(DataAnalysis.roi_id) == col(ROI.id))
        .where(
            ROI.fov_id == fov.id,
            DataAnalysis.analysis_result_id == parent.analysis_result_id,
        )
        .order_by(col(ROI.label_value))
    )
    for roi, analysis in session.exec(stmt):
        if roi.label_value not in (child.active_roi_labels or []):
            continue
        metric = next(
            (item for item in analysis.spike_analyses if item.method == child.method),
            None,
        )
        if metric is not None:
            thresholds.append(_threshold_record(metric, roi.label_value))
    window = None
    if run is not None and run.extraction_result_id is not None:
        window = session.exec(
            select(ExtractionFrameWindow).where(
                ExtractionFrameWindow.fov_id == fov.id,
                ExtractionFrameWindow.extraction_result_id == run.extraction_result_id,
            )
        ).one_or_none()
    return {
        "schema_version": 1,
        "run_id": parent.analysis_result_id,
        "fov_name": fov.name,
        "position_index": fov.position_index,
        "method": child.method,
        "units": child.units,
        "threshold_units": child.units,
        "thresholds": thresholds,
        "metric_units": {name: _fov_metric_units(name) for name in SPIKE_FOV_METRICS},
        "active_roi_labels": child.active_roi_labels,
        "valid_start_frame_0based": child.valid_start,
        "valid_stop_frame_exclusive": child.valid_stop,
        "frame_rate_hz": child.frame_rate_hz,
        "population_coordinate_basis": "retained_relative"
        if child.valid_start is not None
        else "unknown",
        "provenance_source": child.provenance_source,
        "inference_run": run.model_dump(mode="json") if run else None,
        "frame_window": window.model_dump(mode="json") if window else None,
    }


def write_spike_metadata(path: Path, metadata: dict) -> None:
    """Write standard JSON; disabled OASIS thresholds use an explicit string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")


def available_spike_result_methods(
    engine: Engine,
    *,
    run_id: int,
    fov_name: str | None = None,
    position_indices: list[int] | None = None,
    fov_only: bool = False,
) -> set[str]:
    """Find methods from stored analysis rows, rather than extraction/UI defaults."""
    ensure_schema_current(engine)
    available: set[str] = set()
    with Session(engine) as session:
        for model, parent, roi_level in (
            (SpikeAnalysis, DataAnalysis, True),
            (SpikeFOVAnalysis, FOVAnalysis, False),
        ):
            if fov_only and roi_level:
                continue
            stmt = select(model.method).join(parent)
            if roi_level:
                stmt = stmt.join(ROI, col(DataAnalysis.roi_id) == col(ROI.id)).join(
                    FOV, col(ROI.fov_id) == col(FOV.id)
                )
            else:
                stmt = stmt.join(FOV, col(FOVAnalysis.fov_id) == col(FOV.id))
            stmt = stmt.where(parent.analysis_result_id == run_id)
            if fov_name is not None:
                stmt = stmt.where(FOV.name == fov_name)
            if position_indices is not None:
                stmt = stmt.where(col(FOV.position_index).in_(position_indices))
            available.update(session.exec(stmt).all())
    return available


def export_spike_results_to_csv(
    engine: Engine,
    output_dir: str | Path,
    *,
    run_id: int,
    spike_methods: Sequence[str] | None = None,
    fov_name: str | None = None,
    position_indices: list[int] | None = None,
    include_matrices: bool = True,
) -> tuple[SpikeMethod, ...]:
    """Export all stored methods separately, without pooling or guessing coordinates.

    ROI/FOV metrics, population samples and burst bounds use long-form tables with
    a method column. Matrices have method-qualified filenames and exact input labels.
    The units column records input spike units; metric units are explicit in metadata.
    Unknown historical population coordinates remain empty. Files are staged until
    every selected method has been validated, so a failed dual export replaces none.
    """
    from ._database_to_csv import _export_ordered_fov_matrix

    methods = _selected_methods(
        available_spike_result_methods(
            engine, run_id=run_id, fov_name=fov_name, position_indices=position_indices
        ),
        spike_methods,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    common = ["run_id", "fov_name", "position_index", "method", "units"]
    roi_columns = [
        *common,
        "roi_label",
        "spike_analysis_id",
        "spike_trace_id",
        "spike_inference_run_id",
        "provenance_source",
        "threshold_mode",
        "threshold_units",
        "spike_active",
        "valid_start_frame_0based",
        "valid_stop_frame_exclusive",
        *_ROI_METRICS,
    ]
    fov_metrics = [
        name
        for name in SPIKE_FOV_METRICS
        if "matrix" not in name
        and name
        not in {
            "spike_burst_starts",
            "spike_burst_ends",
            "spike_population_activity",
            "spike_population_activity_raw",
        }
    ]
    fov_columns = [
        *common,
        "spike_fov_analysis_id",
        "spike_inference_run_id",
        "provenance_source",
        "active_roi_labels",
        "threshold_units",
        "thresholds",
        "valid_start_frame_0based",
        "valid_stop_frame_exclusive",
        "frame_rate_hz",
        *fov_metrics,
    ]
    sample_columns = [
        *common,
        "array_sample_index",
        "retained_frame_0based",
        "valid_start_frame_0based",
        "valid_stop_frame_exclusive",
        "frame_rate_hz",
        "population_activity_raw",
        "population_activity_smoothed",
        "population_activity_units",
    ]
    burst_columns = [
        *common,
        "burst_index",
        "start_retained_frame_0based",
        "stop_retained_frame_exclusive",
        "valid_start_frame_0based",
        "valid_stop_frame_exclusive",
        "frame_rate_hz",
    ]
    with (
        TemporaryDirectory(prefix=".spike-export-", dir=destination) as temporary,
        Session(engine) as session,
    ):
        stage = Path(temporary)
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "methods": list(methods),
            "roi_metric_units": _ROI_METRIC_UNITS,
            "fovs": [],
        }
        roi_stmt = (
            select(ROI, DataAnalysis, FOV)
            .join(DataAnalysis, col(DataAnalysis.roi_id) == col(ROI.id))
            .join(FOV, col(ROI.fov_id) == col(FOV.id))
            .where(DataAnalysis.analysis_result_id == run_id)
            .order_by(col(FOV.position_index), col(ROI.label_value))
        )
        fov_stmt = (
            select(FOVAnalysis, FOV)
            .join(FOV, col(FOVAnalysis.fov_id) == col(FOV.id))
            .where(FOVAnalysis.analysis_result_id == run_id)
            .order_by(col(FOV.position_index), col(FOV.name))
        )
        if fov_name is not None:
            roi_stmt, fov_stmt = (
                roi_stmt.where(FOV.name == fov_name),
                fov_stmt.where(FOV.name == fov_name),
            )
        if position_indices is not None:
            roi_stmt, fov_stmt = (
                roi_stmt.where(col(FOV.position_index).in_(position_indices)),
                fov_stmt.where(col(FOV.position_index).in_(position_indices)),
            )
        with (stage / "spike_roi_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=roi_columns)
            writer.writeheader()
            for roi, analysis, fov in session.exec(roi_stmt).yield_per(32):
                for method in methods:
                    child = analysis.get_spike_analysis(method)
                    if child is None:
                        continue
                    _threshold_record(child, roi.label_value)
                    spike = child.spike_trace
                    run = spike.inference_run if spike else None
                    if run is not None and (
                        run.method != child.method or run.units != child.units
                    ):
                        raise ValueError(
                            "Exported ROI result must match its inference provenance."
                        )
                    row = {
                        "run_id": run_id,
                        "fov_name": fov.name,
                        "position_index": fov.position_index,
                        "method": method,
                        "units": child.units,
                        "roi_label": roi.label_value,
                        "spike_analysis_id": child.id,
                        "spike_trace_id": child.spike_trace_id,
                        "spike_inference_run_id": run.id if run else None,
                        "provenance_source": child.provenance_source,
                        "threshold_mode": child.threshold_mode,
                        "threshold_units": child.units,
                        "spike_active": child.spike_active,
                        "valid_start_frame_0based": (
                            spike.valid_start if spike else None
                        ),
                        "valid_stop_frame_exclusive": spike.resolved_valid_stop
                        if spike
                        else None,
                    }
                    row.update({name: getattr(child, name) for name in _ROI_METRICS})
                    writer.writerow(row)
        with (
            (stage / "spike_fov_metrics.csv").open("w", newline="") as fov_handle,
            (stage / "spike_population_activity.csv").open(
                "w", newline=""
            ) as sample_handle,
            (stage / "spike_bursts.csv").open("w", newline="") as burst_handle,
        ):
            fov_writer, sample_writer, burst_writer = (
                csv.DictWriter(handle, fieldnames=columns)
                for handle, columns in (
                    (fov_handle, fov_columns),
                    (sample_handle, sample_columns),
                    (burst_handle, burst_columns),
                )
            )
            for writer in (fov_writer, sample_writer, burst_writer):
                writer.writeheader()
            for parent, fov in session.exec(fov_stmt):
                for method in methods:
                    fov_child = parent.get_spike_analysis(method)
                    if fov_child is None:
                        continue
                    record = fov_spike_metadata(session, parent, fov, fov_child)
                    metadata["fovs"].append(record)
                    base = {name: record[name] for name in common}
                    coordinates = {
                        name: record[name]
                        for name in (
                            "valid_start_frame_0based",
                            "valid_stop_frame_exclusive",
                            "frame_rate_hz",
                        )
                    }
                    fov_writer.writerow(
                        {
                            **base,
                            **coordinates,
                            "spike_fov_analysis_id": fov_child.id,
                            "spike_inference_run_id": fov_child.spike_inference_run_id,
                            "provenance_source": fov_child.provenance_source,
                            "active_roi_labels": json.dumps(
                                fov_child.active_roi_labels
                            ),
                            "threshold_units": fov_child.units,
                            "thresholds": json.dumps(
                                record["thresholds"], allow_nan=False
                            ),
                            **{name: getattr(fov_child, name) for name in fov_metrics},
                        }
                    )
                    raw, smooth = (
                        fov_child.spike_population_activity_raw,
                        fov_child.spike_population_activity,
                    )
                    if (
                        raw is not None
                        and smooth is not None
                        and len(raw) != len(smooth)
                    ):
                        raise ValueError(
                            "Population exports require aligned raw/smoothed arrays."
                        )
                    size = (
                        len(raw)
                        if raw is not None
                        else len(smooth)
                        if smooth is not None
                        else 0
                    )
                    for index in range(size):
                        sample_writer.writerow(
                            {
                                **base,
                                **coordinates,
                                "array_sample_index": index,
                                "retained_frame_0based": fov_child.valid_start + index
                                if fov_child.valid_start is not None
                                else None,
                                "population_activity_raw": raw[index]
                                if raw is not None
                                else None,
                                "population_activity_smoothed": smooth[index]
                                if smooth is not None
                                else None,
                                "population_activity_units": "fraction of active ROIs",
                            }
                        )
                    starts, stops = (
                        fov_child.spike_burst_starts or [],
                        fov_child.spike_burst_ends or [],
                    )
                    if len(starts) != len(stops):
                        raise ValueError(
                            "Burst exports require matching start/stop bounds."
                        )
                    for index, (start, stop) in enumerate(zip(starts, stops)):
                        burst_writer.writerow(
                            {
                                **base,
                                **coordinates,
                                "burst_index": index,
                                "start_retained_frame_0based": start,
                                "stop_retained_frame_exclusive": stop,
                            }
                        )
                    if include_matrices:
                        record["matrices"] = []
                        for name in SPIKE_FOV_METRICS:
                            if "matrix" in name:
                                filename = f"{fov.name}_{method}_{name}.csv"
                                labels = _export_ordered_fov_matrix(
                                    session,
                                    fov,
                                    getattr(fov_child, name),
                                    fov_child.active_roi_labels,
                                    stage / filename,
                                )
                                if labels:
                                    record["matrices"].append(
                                        {
                                            "filename": filename,
                                            "metric": name,
                                            "exported_roi_labels": labels,
                                        }
                                    )
        metadata["files"] = sorted(
            [path.name for path in stage.iterdir()] + ["spike_results.metadata.json"]
        )
        write_spike_metadata(stage / "spike_results.metadata.json", metadata)
        for path in sorted(stage.iterdir()):
            path.replace(destination / path.name)
    return methods
