"""Common-interval comparison CSVs preserve distinct units and source provenance."""

from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
from sqlmodel import Session, col, select

from cali.plot._spike_comparison import get_spike_comparisons
from cali.sqlmodel import FOV, ROI, Traces
from cali.sqlmodel._engine import ensure_schema_current

from ._spike_export import _threshold_record, write_spike_metadata

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.engine import Engine

    from cali.sqlmodel._spike_settings import SpikeMethod


def export_spike_comparison_to_csv(
    engine: Engine,
    destination: Path,
    *,
    run_id: int,
    fov_names: Sequence[str] | None = None,
    include_onsets: bool = False,
) -> int:
    """Stage aligned dual-output samples and metadata before replacing any files.

    Original amplitudes and independently peak-scaled amplitudes are exported in
    long form. Onset export requires both applied thresholds and censors the
    common interval's left boundary. Return the number of paired ROIs exported.
    """
    ensure_schema_current(engine)
    with Session(engine) as session:
        stmt = (
            select(FOV.name)
            .join(ROI, col(ROI.fov_id) == col(FOV.id))
            .join(Traces, col(Traces.roi_id) == col(ROI.id))
            .where(Traces.analysis_result_id == run_id)
            .distinct()
            .order_by(FOV.name)
        )
        if fov_names is not None:
            stmt = stmt.where(col(FOV.name).in_(fov_names))
        names = session.exec(stmt).all()
    methods: tuple[SpikeMethod, ...] = ("oasis", "cascade")
    fields = (
        "run_id",
        "fov_name",
        "roi_label",
        "method",
        "units",
        "common_valid_start",
        "common_valid_stop",
        "frame_rate_hz",
        "retained_frame_0based",
        "source_frame_0based",
        "retained_time_ms",
        "source_time_ms",
        "amplitude",
        "normalized_amplitude",
        "normalization_scale",
        "threshold",
        "threshold_mode",
        "threshold_units",
        "threshold_onset",
    )
    metadata: dict = {
        "schema_version": 1,
        "run_id": run_id,
        "methods": list(methods),
        "interval": (
            "per-ROI intersection of both valid intervals; retained half-open bounds"
        ),
        "normalization": (
            "independent peak absolute amplitude within the common interval"
        ),
        "population": (
            "all paired ROIs, including inactive neurons; no ROI union filtering"
        ),
        "onsets": "left boundary censored" if include_onsets else "not requested",
        "rois": [],
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="cali-spike-comparison-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary)
        with (stage / "spike_comparison.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for fov_name in names:
                pairs = get_spike_comparisons(
                    engine, fov_name, run_id, require_threshold=include_onsets
                )
                for pair in pairs:
                    trace = pair.source_trace
                    transform = trace.source_frame_transform()
                    assert trace.extraction_frame_window is not None
                    record: dict = {
                        "fov_name": fov_name,
                        "roi_label": pair.roi_label,
                        "common_valid_start": pair.valid_start,
                        "common_valid_stop": pair.valid_stop,
                        "frame_rate_hz": pair.frame_rate_hz,
                        "frame_window": trace.extraction_frame_window.model_dump(
                            mode="json"
                        ),
                        "methods": {},
                    }
                    for method in methods:
                        data = pair.oasis if method == "oasis" else pair.cascade
                        cutoff = (
                            _threshold_record(data.metric, pair.roi_label)
                            if data.metric is not None
                            else {
                                "threshold": None,
                                "threshold_mode": None,
                                "threshold_units": data.spike.inference_run.units,
                            }
                        )
                        values = pair.values(method)
                        normalized = pair.normalized_values(method)
                        scale = float(np.max(np.abs(values)))
                        events = (
                            set(pair.event_frames(method)) if include_onsets else set()
                        )
                        record["methods"][method] = {
                            "inference_run": data.spike.inference_run.model_dump(
                                mode="json"
                            ),
                            "valid_start": data.spike.valid_start,
                            "valid_stop": data.spike.resolved_valid_stop,
                            "normalization_scale": scale,
                            "threshold": cutoff,
                            "spike_active": data.metric.spike_active
                            if data.metric
                            else None,
                        }
                        for index, frame in enumerate(
                            range(pair.valid_start, pair.valid_stop)
                        ):
                            if trace.x_axis is not None and trace.x_axis_units == "ms":
                                axis_time = trace.x_axis[frame]
                                retained_time = axis_time - trace.x_axis[0]
                                source_time = transform.retained_time_to_source(
                                    axis_time
                                )
                            else:
                                retained_time = frame / pair.frame_rate_hz * 1000
                                source_time = (
                                    retained_time + transform.source_start_time_ms
                                )
                            writer.writerow(
                                {
                                    "run_id": run_id,
                                    "fov_name": fov_name,
                                    "roi_label": pair.roi_label,
                                    "method": method,
                                    "units": data.spike.inference_run.units,
                                    "common_valid_start": pair.valid_start,
                                    "common_valid_stop": pair.valid_stop,
                                    "frame_rate_hz": pair.frame_rate_hz,
                                    "retained_frame_0based": frame,
                                    "source_frame_0based": transform.to_source(
                                        frame, one_based=False
                                    ),
                                    "retained_time_ms": retained_time,
                                    "source_time_ms": source_time,
                                    "amplitude": values[index],
                                    "normalized_amplitude": normalized[index],
                                    "normalization_scale": scale,
                                    "threshold": cutoff["threshold"],
                                    "threshold_mode": cutoff["threshold_mode"],
                                    "threshold_units": cutoff["threshold_units"],
                                    "threshold_onset": int(frame in events)
                                    if include_onsets
                                    else None,
                                }
                            )
                    metadata["rois"].append(record)
        if not metadata["rois"]:
            raise ValueError("No aligned dual-method spike data for the selected run.")
        write_spike_metadata(stage / "spike_comparison.metadata.json", metadata)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(stage.iterdir()):
            path.replace(destination / path.name)
    return len(metadata["rois"])
