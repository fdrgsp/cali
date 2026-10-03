"""Resolve and transform the initial frame window used for trace extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from cali._constants import RUNNER_TIME_KEY

DiscardInitialUnit = Literal["frames", "seconds"]
TimingSource = Literal["runner_time", "exposure", "user_verified"]


class StartupDiscardError(ValueError):
    """Raised when a requested startup discard cannot be applied safely."""


@dataclass(frozen=True)
class TimingDescriptor:
    """Timing information for one uncropped source sequence."""

    timestamps_ms: list[float]
    source: Literal["runner_time", "exposure"]
    trusted: bool


@dataclass(frozen=True)
class ExtractionFrameWindow:
    """Resolved retained window for one source sequence."""

    source_start_frame: int
    source_start_time_ms: float
    original_frame_count: int
    retained_frame_count: int
    discarded_duration_ms: float
    timing_source: TimingSource


def build_timing_descriptor(meta: list[dict], num_timepoints: int) -> TimingDescriptor:
    """Build the legacy time axis and record whether timestamps are trustworthy."""
    if num_timepoints < 0:
        raise ValueError("Number of timepoints cannot be negative.")

    exposure_ms = 0.0
    if meta:
        raw_exposure = meta[0].get("exposure_ms", 0.0)
        if raw_exposure is not None:
            exposure_ms = float(raw_exposure)

    runner_times: list[float] = []
    if meta and RUNNER_TIME_KEY in meta[0]:
        for frame_meta in meta:
            value = frame_meta.get(RUNNER_TIME_KEY)
            if value is not None:
                runner_times.append(float(value))

    if len(runner_times) == num_timepoints:
        times = np.asarray(runner_times, dtype=float)
        trusted = bool(
            np.all(np.isfinite(times))
            and (len(times) < 2 or np.all(np.diff(times) > 0))
        )
        return TimingDescriptor(runner_times, "runner_time", trusted)

    timestamps_ms = [frame * exposure_ms for frame in range(num_timepoints)]
    return TimingDescriptor(timestamps_ms, "exposure", False)


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
    elif timing.trusted:
        timestamps = np.asarray(timing.timestamps_ms, dtype=float)
        relative_ms = timestamps - timestamps[0]
        source_start_frame = int(
            np.searchsorted(relative_ms, discard_value * 1000.0, side="left")
        )
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
    if discard_value > 0 and retained_count < 2:
        raise StartupDiscardError(
            "Discard at Start must leave at least two frames for extraction "
            f"({retained_count} would remain)."
        )

    if source_start_frame == 0:
        source_start_time_ms = 0.0
    elif timing_source == "user_verified":
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
    if start == 0:
        # Preserve the exact legacy axis for the default/no-op configuration.
        return timing.timestamps_ms

    if window.timing_source == "user_verified":
        interval_ms = 1000.0 / frame_rate
        return [frame * interval_ms for frame in range(window.retained_frame_count)]

    retained = timing.timestamps_ms[start:]
    first = retained[0]
    return [timestamp - first for timestamp in retained]


def source_frame_to_retained(
    source_frame: float,
    source_start_frame: int,
    *,
    one_based: bool = True,
) -> float:
    """Translate a source-file frame coordinate into a retained zero-based index."""
    zero_based_source = source_frame - 1 if one_based else source_frame
    return zero_based_source - source_start_frame


def source_interval_to_retained(
    source_start_frame: float,
    duration_frames: float,
    retained_start_frame: int,
    retained_frame_count: int,
    *,
    one_based: bool = True,
) -> tuple[float, float] | None:
    """Translate, clip, or omit a source interval for a retained trace."""
    start = source_frame_to_retained(
        source_start_frame, retained_start_frame, one_based=one_based
    )
    stop = start + duration_frames
    clipped_start = max(0.0, start)
    clipped_stop = min(float(retained_frame_count), stop)
    if clipped_stop <= clipped_start:
        return None
    return clipped_start, clipped_stop
