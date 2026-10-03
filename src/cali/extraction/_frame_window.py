"""Resolve and transform the initial frame window used for trace extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from cali._constants import RUNNER_TIME_KEY

DiscardInitialUnit = Literal["frames", "seconds"]
TimingSource = Literal[
    "runner_time", "metadata_frame_period", "exposure", "user_verified"
]


class StartupDiscardError(ValueError):
    """Raised when a requested startup discard cannot be applied safely."""


@dataclass(frozen=True)
class TimingDescriptor:
    """Timing information for one uncropped source sequence."""

    timestamps_ms: list[float]
    source: TimingSource
    trusted: bool
    frame_rate_hz: float | None = None
    interval_jitter_fraction: float | None = None
    validation: str = "legacy_descriptor"


@dataclass(frozen=True)
class ExtractionFrameWindow:
    """Resolved retained window for one source sequence."""

    source_start_frame: int
    source_start_time_ms: float
    original_frame_count: int
    retained_frame_count: int
    discarded_duration_ms: float
    timing_source: TimingSource


def _describe_timing(
    timestamps_ms: list[float], source: TimingSource, trusted: bool, validation: str
) -> TimingDescriptor:
    rate = None
    jitter = None
    if trusted and len(timestamps_ms) >= 2:
        intervals = np.diff(timestamps_ms)
        period = float(np.median(intervals))
        if period > 0:
            rate = 1000.0 / period
            if source == "runner_time":
                jitter = float(np.max(np.abs(intervals - period)) / period)
    return TimingDescriptor(timestamps_ms, source, trusted, rate, jitter, validation)


def build_timing_descriptor(
    meta: list[dict],
    num_timepoints: int,
    *,
    frame_rate: float | None = None,
    frame_rate_verified: bool = False,
) -> TimingDescriptor:
    """Resolve acquisition timing without treating exposure as frame period.

    Complete acquisition timestamps take priority over explicit ``frame_period_ms``
    metadata, then user-verified settings. Exposure-only axes preserve legacy OASIS
    behavior but never authorize seconds-mode discard or CASCADE model selection.
    """
    if num_timepoints < 0:
        raise ValueError("Number of timepoints cannot be negative.")

    if (
        meta
        and len(meta) == num_timepoints
        and all(item.get(RUNNER_TIME_KEY) is not None for item in meta)
    ):
        try:
            runner_times = [float(item[RUNNER_TIME_KEY]) for item in meta]
        except (TypeError, ValueError) as error:
            raise StartupDiscardError(
                "Acquisition timestamps must be numeric."
            ) from error
        times = np.asarray(runner_times, dtype=float)
        if not np.all(np.isfinite(times)) or (
            len(times) >= 2 and not np.all(np.diff(times) > 0)
        ):
            raise StartupDiscardError(
                "Acquisition timestamps must be finite and strictly increasing."
            )
        return _describe_timing(runner_times, "runner_time", True, "trusted_timestamps")

    periods = [
        item["frame_period_ms"]
        for item in meta
        if item.get("frame_period_ms") is not None
    ]
    if periods:
        try:
            values = np.asarray(periods, dtype=float)
        except (TypeError, ValueError) as error:
            raise StartupDiscardError(
                "Frame-period metadata must be numeric."
            ) from error
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise StartupDiscardError(
                "Frame-period metadata must be finite and positive."
            )
        if not np.all(values == values[0]):
            raise StartupDiscardError(
                "Conflicting frame-period metadata requires per-frame timestamps."
            )
        timestamps = [frame * float(values[0]) for frame in range(num_timepoints)]
        return _describe_timing(
            timestamps, "metadata_frame_period", True, "trusted_frame_period"
        )

    if frame_rate_verified:
        if frame_rate is None or not math.isfinite(frame_rate) or frame_rate <= 0:
            raise StartupDiscardError("A positive verified frame rate is required.")
        timestamps = [frame * 1000.0 / frame_rate for frame in range(num_timepoints)]
        return _describe_timing(timestamps, "user_verified", True, "user_verified")

    raw_exposure = meta[0].get("exposure_ms", 0.0) if meta else 0.0
    try:
        exposure_ms = float(raw_exposure) if raw_exposure is not None else 0.0
    except (TypeError, ValueError) as error:
        raise StartupDiscardError("Exposure metadata must be numeric.") from error
    timestamps_ms = [frame * exposure_ms for frame in range(num_timepoints)]
    return _describe_timing(timestamps_ms, "exposure", False, "unverified_exposure")


def resolve_initial_frame_window(
    *,
    discard_value: float,
    discard_unit: str,
    frame_rate: float,
    frame_rate_verified: bool,
    timing: TimingDescriptor,
) -> ExtractionFrameWindow:
    """Resolve the requested initial discard to an exact source-frame window."""
    if not math.isfinite(discard_value) or discard_value < 0:
        raise StartupDiscardError(
            "Discard at Start must be a finite, non-negative value."
        )
    if discard_unit not in {"frames", "seconds"}:
        raise StartupDiscardError(
            "Discard at Start unit must be 'frames' or 'seconds'."
        )

    original_count = len(timing.timestamps_ms)
    timing_source: TimingSource = timing.source

    if discard_value == 0:
        source_start_frame = 0
    elif discard_unit == "frames":
        if not float(discard_value).is_integer():
            raise StartupDiscardError(
                "Discard at Start must be a whole number in Frames mode."
            )
        source_start_frame = int(discard_value)
    elif timing.trusted and timing.source == "runner_time":
        timestamps = np.asarray(timing.timestamps_ms, dtype=float)
        relative_ms = timestamps - timestamps[0]
        source_start_frame = int(
            np.searchsorted(relative_ms, discard_value * 1000.0, side="left")
        )
    elif timing.trusted and timing.source in {"metadata_frame_period", "user_verified"}:
        period = (
            timing.timestamps_ms[1] - timing.timestamps_ms[0]
            if original_count > 1
            else None
        )
        if period is None or not math.isfinite(period) or period <= 0:
            raise StartupDiscardError("At least two timed samples are required.")
        source_start_frame = math.ceil(discard_value * 1000.0 / period)
    else:
        if not frame_rate_verified:
            raise StartupDiscardError(
                "Discarding seconds requires valid per-frame timestamps or a "
                "user-verified acquisition frame rate."
            )
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise StartupDiscardError("A positive verified frame rate is required.")
        source_start_frame = math.ceil(discard_value * frame_rate)
        timing_source = "user_verified"

    retained_count = original_count - source_start_frame
    if source_start_frame >= original_count:
        raise StartupDiscardError(
            "Discard at Start removes the entire recording "
            f"({source_start_frame} of {original_count} frames)."
        )
    if source_start_frame == 0:
        source_start_time_ms = 0.0
    elif timing_source == "user_verified" and timing.source != "user_verified":
        source_start_time_ms = source_start_frame * 1000.0 / frame_rate
    else:
        source_start_time_ms = (
            timing.timestamps_ms[source_start_frame] - timing.timestamps_ms[0]
        )

    return ExtractionFrameWindow(
        source_start_frame=source_start_frame,
        source_start_time_ms=float(source_start_time_ms),
        original_frame_count=original_count,
        retained_frame_count=retained_count,
        discarded_duration_ms=float(source_start_time_ms),
        timing_source=timing_source,
    )


def retained_time_axis(
    timing: TimingDescriptor,
    window: ExtractionFrameWindow,
    *,
    frame_rate: float,
) -> list[float]:
    """Return the retained time axis, rebased to the first retained frame."""
    start = window.source_start_frame
    if window.timing_source == "user_verified" and timing.source != "user_verified":
        interval_ms = 1000.0 / frame_rate
        return [frame * interval_ms for frame in range(window.retained_frame_count)]

    if start == 0:
        # Preserve the exact legacy axis for the default/no-op configuration.
        return timing.timestamps_ms

    retained = timing.timestamps_ms[start:]
    first = retained[0]
    return [timestamp - first for timestamp in retained]


def preflight_retained_timing(
    timing: TimingDescriptor,
    window: ExtractionFrameWindow,
    *,
    frame_rate: float,
    minimum_frames: dict[str, int],
) -> TimingDescriptor:
    """Validate retained duration and every enabled consumer before ROI work."""
    if not minimum_frames or any(count < 1 for count in minimum_frames.values()):
        raise ValueError("Consumers must declare positive minimum frame counts.")
    limiting, required = max(minimum_frames.items(), key=lambda item: item[1])
    if window.retained_frame_count < required:
        raise StartupDiscardError(
            f"{window.original_frame_count} source frames minus "
            f"{window.source_start_frame} discarded leaves "
            f"{window.retained_frame_count} retained; {limiting} requires at least "
            f"{required} frames. Reduce the discard or use a longer recording."
        )
    timestamps = retained_time_axis(timing, window, frame_rate=frame_rate)
    times = np.asarray(timestamps, dtype=float)
    if (
        len(times) != window.retained_frame_count
        or len(times) < 2
        or (not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0))
    ):
        raise StartupDiscardError(
            "Retained timing must have one finite, strictly increasing timestamp "
            "per frame and positive duration. Supply acquisition timestamps, "
            "frame_period_ms metadata, or a verified acquisition frame rate."
        )
    return _describe_timing(
        timestamps,
        window.timing_source,
        timing.trusted or window.timing_source == "user_verified",
        timing.validation if timing.source == window.timing_source else "user_verified",
    )


def validate_model_timing(
    timing: TimingDescriptor,
    *,
    settings_frame_rate: float,
    model_frame_rate: float,
    relative_tolerance: float = 0.01,
) -> float:
    """Require trusted, uniform retained timing matching settings and model rate.

    The default maximum interval deviation and rate mismatch are both 1%.
    User-verified and metadata-period rates have unknown measured jitter.
    """
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("Timing tolerance must be finite and between zero and one.")
    if not timing.trusted or timing.source == "exposure":
        raise StartupDiscardError(
            "CASCADE requires acquisition timestamps, explicit frame-period metadata, "
            "or a user-verified acquisition frame rate; exposure alone is insufficient."
        )
    times = np.asarray(timing.timestamps_ms, dtype=float)
    if (
        len(times) < 2
        or not np.all(np.isfinite(times))
        or not np.all(np.diff(times) > 0)
    ):
        raise StartupDiscardError(
            "CASCADE requires finite, strictly increasing timing."
        )
    intervals = np.diff(times)
    period = float(np.median(intervals))
    observed_rate = 1000.0 / period
    if timing.source == "runner_time":
        jitter = float(np.max(np.abs(intervals - period)) / period)
        if jitter > relative_tolerance:
            raise StartupDiscardError(
                f"CASCADE requires uniform intervals; maximum interval deviation "
                f"is {jitter:.2%} (allowed {relative_tolerance:.2%})."
            )
    for label, rate in (
        ("extraction settings", settings_frame_rate),
        ("CASCADE model", model_frame_rate),
    ):
        if not math.isfinite(rate) or rate <= 0:
            raise StartupDiscardError(f"{label} require a finite, positive frame rate.")
        if abs(observed_rate - rate) / rate > relative_tolerance:
            raise StartupDiscardError(
                f"Acquisition rate {observed_rate:.6g} Hz does not match {label} "
                f"rate {rate:.6g} Hz (allowed {relative_tolerance:.2%})."
            )
    return observed_rate


@dataclass(frozen=True)
class SourceFrameTransform:
    """Map one retained trace to its immutable source coordinates.

    Frame indices inside cali are zero-based. User stimulation inputs and the
    explicit ``source_frame_1based`` export use one-based source frames. Times
    are relative to the source's first sample unless an absolute timestamp is
    requested; an unknown historical timestamp origin stays unknown.
    """

    source_start_frame: int
    retained_frame_count: int
    source_start_time_ms: float = 0.0
    source_time_origin_ms: float | None = None
    retained_timestamps_ms: tuple[float, ...] = ()
    source_one_based: bool = True

    def to_retained(self, source_frame: float) -> float:
        """Convert a source input frame to an unclipped retained index."""
        return source_frame - int(self.source_one_based) - self.source_start_frame

    def to_source(self, retained_frame: float, *, one_based: bool = True) -> float:
        """Convert a retained index to an explicit source frame convention."""
        return retained_frame + self.source_start_frame + int(one_based)

    def clip_interval(
        self, source_frame: float, duration_frames: float
    ) -> tuple[float, float] | None:
        """Intersect a half-open source interval with the retained recording."""
        start = self.to_retained(source_frame)
        stop = start + duration_frames
        clipped_start = max(0.0, start)
        clipped_stop = min(float(self.retained_frame_count), stop)
        return (clipped_start, clipped_stop) if clipped_stop > clipped_start else None

    def retained_time_to_source(self, time_ms: float) -> float:
        """Convert a stored trace time to time relative to the source origin."""
        axis_origin = (
            self.retained_timestamps_ms[0] if self.retained_timestamps_ms else 0
        )
        return time_ms - axis_origin + self.source_start_time_ms

    def source_time_to_retained(self, time_ms: float) -> float:
        """Convert source-relative time to the stored trace's time axis."""
        axis_origin = (
            self.retained_timestamps_ms[0] if self.retained_timestamps_ms else 0
        )
        return time_ms - self.source_start_time_ms + axis_origin

    def frame_times(
        self, frame: int
    ) -> tuple[float | None, float | None, float | None]:
        """Return retained, source-relative, and absolute time for a sample."""
        if not 0 <= frame < self.retained_frame_count:
            raise ValueError("Event frame lies outside the retained recording.")
        if not self.retained_timestamps_ms:
            return None, None, None
        stored_time = self.retained_timestamps_ms[frame]
        retained_time = stored_time - self.retained_timestamps_ms[0]
        source_time = self.retained_time_to_source(stored_time)
        timestamp = (
            source_time + self.source_time_origin_ms
            if self.source_time_origin_ms is not None
            else None
        )
        return retained_time, source_time, timestamp


def source_frame_to_retained(
    source_frame: float,
    source_start_frame: int,
    *,
    one_based: bool = True,
) -> float:
    """Translate a source-file frame coordinate into a retained zero-based index."""
    return SourceFrameTransform(
        source_start_frame, 0, source_one_based=one_based
    ).to_retained(source_frame)


def source_interval_to_retained(
    source_start_frame: float,
    duration_frames: float,
    retained_start_frame: int,
    retained_frame_count: int,
    *,
    one_based: bool = True,
) -> tuple[float, float] | None:
    """Translate, clip, or omit a source interval for a retained trace."""
    return SourceFrameTransform(
        retained_start_frame, retained_frame_count, source_one_based=one_based
    ).clip_interval(source_start_frame, duration_frames)
