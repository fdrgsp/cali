"""Normalized extraction timing and method-bound spike products."""

import math
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import UniqueConstraint
from sqlmodel import JSON, Column, Field, Relationship, SQLModel, select

from ._result_json import ResultJSON

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ._model import FOV, CaliResult, Traces


class ExtractionFrameWindow(SQLModel, table=True):  # type: ignore[call-arg, unused-ignore]
    """A shared, immutable source-to-retained mapping for an extraction/FOV."""

    __tablename__ = "extraction_frame_window"
    __table_args__ = (UniqueConstraint("extraction_result_id", "fov_id"),)

    id: int | None = Field(default=None, primary_key=True)
    extraction_result_id: int | None = Field(
        default=None, foreign_key="analysis_result.id", ondelete="SET NULL"
    )
    fov_id: int | None = Field(default=None, foreign_key="fov.id", ondelete="SET NULL")
    legacy_owner_result_id: int | None = Field(
        default=None, foreign_key="analysis_result.id", ondelete="SET NULL"
    )
    requested_discard_value: float = 0.0
    requested_discard_unit: str = "frames"
    timing_source: str | None = None
    conversion_rule: str = "legacy_unavailable"
    original_frame_count: int | None = None
    retained_frame_count: int | None = None
    source_start_frame: int = 0
    source_start_time_ms: float = 0.0
    source_time_origin_ms: float | None = None
    source_start_timestamp_ms: float | None = None
    discarded_duration_ms: float = 0.0
    provenance_source: str = "synthetic_legacy_api"
    schema_version: int = 1

    extraction_result: Optional["CaliResult"] = Relationship(
        sa_relationship_kwargs={
            "foreign_keys": "[ExtractionFrameWindow.extraction_result_id]"
        }
    )
    fov: Optional["FOV"] = Relationship()

    def semantic_key(self) -> tuple:
        return tuple(
            getattr(self, name)
            for name in type(self).model_fields
            if name
            not in {"id", "extraction_result_id", "fov_id", "legacy_owner_result_id"}
        )


class SpikeInferenceRun(SQLModel, table=True):  # type: ignore[call-arg, unused-ignore]
    """Run-constant backend provenance shared by all ROIs of one extraction."""

    __tablename__ = "spike_inference_run"
    __table_args__ = (UniqueConstraint("extraction_result_id", "method"),)

    id: int | None = Field(default=None, primary_key=True)
    extraction_result_id: int | None = Field(
        default=None, foreign_key="analysis_result.id", ondelete="SET NULL"
    )
    legacy_owner_result_id: int | None = Field(
        default=None, foreign_key="analysis_result.id", ondelete="SET NULL"
    )
    method: str = "oasis"
    units: str = "a.u."
    backend_package: str = "oasis-deconv"
    backend_version: str | None = None
    backend_revision: str | None = None
    resolved_model: str | None = None
    catalogue_revision: str | None = None
    config_sha256: str | None = None
    weights_manifest_sha256: str | None = None
    resolved_device: str | None = None
    dtype: str | None = None
    model_sampling_rate_hz: float | None = None
    smoothing_sigma: float | None = None
    kernel_type: str | None = None
    provenance_source: str = "synthetic_legacy_api"
    schema_version: int = 1

    extraction_result: Optional["CaliResult"] = Relationship(
        back_populates="spike_inference_runs",
        sa_relationship_kwargs={
            "foreign_keys": "[SpikeInferenceRun.extraction_result_id]"
        },
    )
    spike_traces: list["SpikeTrace"] = Relationship(
        back_populates="inference_run",
        sa_relationship_kwargs={"cascade": "delete"},
    )

    def semantic_key(self) -> tuple:
        return tuple(
            getattr(self, name)
            for name in type(self).model_fields
            if name not in {"id", "extraction_result_id", "legacy_owner_result_id"}
        )


class SpikeTrace(ResultJSON, table=True):  # type: ignore[call-arg, unused-ignore]
    """One method's spike array and valid interval for one base trace."""

    __tablename__ = "spike_trace"
    __table_args__ = (UniqueConstraint("trace_id", "spike_inference_run_id"),)

    id: int | None = Field(default=None, primary_key=True)
    trace_id: int | None = Field(
        default=None, foreign_key="trace.id", nullable=False, ondelete="CASCADE"
    )
    spike_inference_run_id: int | None = Field(
        default=None,
        foreign_key="spike_inference_run.id",
        nullable=False,
        ondelete="CASCADE",
    )
    values: list[float] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    valid_start: int = 0
    valid_stop: int | None = None
    noise: float | None = None
    selected_noise_level: float | None = None
    ar_coefficients: list[float] | None = Field(default=None, sa_column=Column(JSON))

    trace: "Traces" = Relationship(back_populates="spike_traces")
    inference_run: SpikeInferenceRun = Relationship(
        back_populates="spike_traces", sa_relationship_kwargs={"lazy": "selectin"}
    )

    @property
    def resolved_valid_stop(self) -> int:
        """A nullable stop in imported data means the full stored length."""
        return len(self.values) if self.valid_stop is None else self.valid_stop


def normalize_trace_provenance(session: "Session", *_: Any) -> None:
    """Bind/deduplicate provenance before insertion, including headless ORM writes."""
    from ._model import ROI, CaliResult, Traces

    new_objects = list(session.new)
    runs: dict[tuple, SpikeInferenceRun] = {}
    windows: dict[tuple, ExtractionFrameWindow] = {}
    replaced: set[int] = set()
    for trace in (obj for obj in new_objects if isinstance(obj, Traces)):
        owner = trace.analysis_result
        if owner is None and trace.analysis_result_id is not None:
            owner = session.get(CaliResult, trace.analysis_result_id)
        roi = trace.roi
        if roi is None and trace.roi_id is not None:
            roi = session.get(ROI, trace.roi_id)
        window = trace.extraction_frame_window
        if window is None and trace.extraction_frame_window_id is not None:
            window = session.get(
                ExtractionFrameWindow, trace.extraction_frame_window_id
            )
            if window is None:
                raise ValueError("The extraction frame window does not exist.")
            trace.extraction_frame_window = window
        if window is not None and window.id is None:
            if (
                window.extraction_result_id is None
                and window.legacy_owner_result_id is None
            ):
                window.extraction_result = owner
                window.extraction_result_id = owner.id if owner is not None else None
            if (
                roi is not None
                and window.fov_id is None
                and window.retained_frame_count not in (None, 0)
            ):
                window.fov = roi.fov
                window.fov_id = roi.fov_id
            result_key = window.extraction_result_id or (
                id(window.extraction_result)
                if window.extraction_result is not None
                else None
            )
            fov_key = window.fov_id or (
                id(window.fov) if window.fov is not None else None
            )
            if result_key is not None and fov_key is not None:
                key = (result_key, fov_key)
                existing_window = windows.get(key)
                if (
                    existing_window is None
                    and window.extraction_result_id is not None
                    and window.fov_id is not None
                ):
                    existing_window = session.scalar(
                        select(ExtractionFrameWindow).where(
                            ExtractionFrameWindow.extraction_result_id
                            == window.extraction_result_id,
                            ExtractionFrameWindow.fov_id == window.fov_id,
                        )
                    )
                if existing_window is not None and existing_window is not window:
                    if existing_window.semantic_key() != window.semantic_key():
                        raise ValueError(
                            "Conflicting frame windows for one extraction/FOV."
                        )
                    trace.extraction_frame_window = existing_window
                    replaced.add(id(window))
                else:
                    existing_window = window
                windows[key] = existing_window

        methods = set()
        for child in trace.spike_traces:
            run = child.inference_run
            if run is None and child.spike_inference_run_id is not None:
                run = session.get(SpikeInferenceRun, child.spike_inference_run_id)
                if run is not None:
                    child.inference_run = run
            if run is None:
                raise ValueError("A spike trace requires inference provenance.")
            if run.method in methods:
                raise ValueError(f"Duplicate spike traces for {run.method}.")
            methods.add(run.method)
            if run.id is not None:
                continue
            if run.extraction_result_id is None and run.legacy_owner_result_id is None:
                run.extraction_result = owner
                run.extraction_result_id = owner.id if owner is not None else None
            result_key = run.extraction_result_id or (
                id(run.extraction_result) if run.extraction_result is not None else None
            )
            if result_key is None:
                continue
            run_key = (result_key, run.method)
            existing_run = runs.get(run_key)
            if existing_run is None and run.extraction_result_id is not None:
                existing_run = session.scalar(
                    select(SpikeInferenceRun).where(
                        SpikeInferenceRun.extraction_result_id
                        == run.extraction_result_id,
                        SpikeInferenceRun.method == run.method,
                    )
                )
            if existing_run is not None and existing_run is not run:
                # Never ascribe today's version/device to imported historical data.
                # An existing unknown field remains unknown when extending a legacy run.
                excluded = {
                    "id",
                    "extraction_result_id",
                    "legacy_owner_result_id",
                    "provenance_source",
                }
                for name in type(run).model_fields:
                    old, new = getattr(existing_run, name), getattr(run, name)
                    if name not in excluded and old is not None and old != new:
                        raise ValueError(
                            "Conflicting inference provenance for one "
                            "extraction/method."
                        )
                child.inference_run = existing_run
                replaced.add(id(run))
            else:
                existing_run = run
            runs[run_key] = existing_run

    # Relationship reassignment leaves unused transient rows in session.new.
    for obj in new_objects:
        if id(obj) in replaced and obj in session.new:
            if isinstance(obj, SpikeInferenceRun):
                unused_owner = obj.extraction_result
                if unused_owner is not None:
                    # Equal pending SQLModel rows compare by values. Remove by
                    # identity so list.remove cannot detach the canonical run.
                    unused_owner.spike_inference_runs = [
                        run
                        for run in unused_owner.spike_inference_runs
                        if run is not obj
                    ]
                obj.extraction_result = None
            session.expunge(obj)
    for obj in list(session.new) + list(session.dirty):
        if isinstance(obj, SpikeTrace):
            if not 0 <= obj.valid_start <= obj.resolved_valid_stop <= len(obj.values):
                raise ValueError(
                    "Spike valid interval must lie within the stored array."
                )
        elif isinstance(obj, SpikeInferenceRun):
            expected_units = {"oasis": "a.u.", "cascade": "spikes/frame"}
            if (
                obj.method not in expected_units
                or obj.units != expected_units[obj.method]
            ):
                raise ValueError("Inference method and units must match.")
            if obj.method == "cascade" and obj.backend_package == "oasis-deconv":
                raise ValueError("CASCADE provenance cannot name the OASIS backend.")
        elif isinstance(obj, ExtractionFrameWindow):
            if obj.source_start_frame < 0:
                raise ValueError("A frame window cannot start before the source.")
            if (
                not math.isfinite(obj.requested_discard_value)
                or obj.requested_discard_value < 0
            ):
                raise ValueError("Requested discard must be finite and non-negative.")
            if obj.requested_discard_unit not in {"frames", "seconds"}:
                raise ValueError("Discard units must be frames or seconds.")
            if (
                obj.original_frame_count is not None
                and obj.retained_frame_count is not None
            ):
                if (
                    obj.retained_frame_count < 0
                    or obj.original_frame_count
                    != obj.retained_frame_count + obj.source_start_frame
                ):
                    raise ValueError(
                        "Frame window counts disagree with the source offset."
                    )
