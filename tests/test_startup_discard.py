"""Tests for excluding acquisition startup frames before trace extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import text
from sqlmodel import create_engine

from cali._constants import RUNNER_TIME_KEY
from cali.extraction._extraction_runner import ExtractionRunner, _RoiParts
from cali.extraction._frame_window import (
    StartupDiscardError,
    TimingDescriptor,
    build_timing_descriptor,
    resolve_initial_frame_window,
    retained_time_axis,
    source_frame_to_retained,
    source_interval_to_retained,
)
from cali.extraction._spike_inference import OasisResult
from cali.gui._extraction_gui import (
    NeuropilData,
    TraceExtractionData,
    _TraceExtractionWidget,
)
from cali.sqlmodel import FOV, ROI, ExtractionSettings
from cali.sqlmodel._util import migrate_startup_discard

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot


def test_resolve_exact_frame_discard_and_rebase() -> None:
    timing = TimingDescriptor([0.0, 100.0, 200.0, 300.0, 400.0], "runner_time", True)

    window = resolve_initial_frame_window(
        discard_value=2,
        discard_unit="frames",
        frame_rate=10.0,
        frame_rate_verified=False,
        timing=timing,
    )

    assert window.source_start_frame == 2
    assert window.source_start_time_ms == 200.0
    assert window.original_frame_count == 5
    assert window.retained_frame_count == 3
    assert retained_time_axis(timing, window, frame_rate=10.0) == [0.0, 100.0, 200.0]


def test_zero_discard_preserves_legacy_time_axis() -> None:
    timestamps = [12.5, 112.5, 212.5]
    timing = TimingDescriptor(timestamps, "runner_time", True)
    window = resolve_initial_frame_window(
        discard_value=0,
        discard_unit="frames",
        frame_rate=10.0,
        frame_rate_verified=False,
        timing=timing,
    )

    assert retained_time_axis(timing, window, frame_rate=10.0) is timestamps


def test_seconds_uses_timestamp_searchsorted_for_irregular_data() -> None:
    timing = TimingDescriptor(
        [1000.0, 1080.0, 1190.0, 1310.0, 1450.0], "runner_time", True
    )
    window = resolve_initial_frame_window(
        discard_value=0.2,
        discard_unit="seconds",
        frame_rate=10.0,
        frame_rate_verified=False,
        timing=timing,
    )

    # Relative timestamps are [0, 80, 190, 310, 450], so the first retained
    # sample at or beyond 200 ms is source frame 3.
    assert window.source_start_frame == 3
    assert window.discarded_duration_ms == 310.0
    assert retained_time_axis(timing, window, frame_rate=10.0) == [0.0, 140.0]


def test_seconds_fallback_requires_and_uses_verified_rate() -> None:
    timing = TimingDescriptor([0.0] * 10, "exposure", False)

    with pytest.raises(StartupDiscardError, match="user-verified"):
        resolve_initial_frame_window(
            discard_value=0.21,
            discard_unit="seconds",
            frame_rate=10.0,
            frame_rate_verified=False,
            timing=timing,
        )

    window = resolve_initial_frame_window(
        discard_value=0.21,
        discard_unit="seconds",
        frame_rate=10.0,
        frame_rate_verified=True,
        timing=timing,
    )
    assert window.source_start_frame == 3
    assert window.timing_source == "user_verified"
    assert retained_time_axis(timing, window, frame_rate=10.0) == [
        0.0,
        100.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
    ]


@pytest.mark.parametrize(
    ("value", "unit", "match"),
    [
        (-1.0, "frames", "non-negative"),
        (1.5, "frames", "whole number"),
        (1.0, "minutes", "unit"),
        (9.0, "frames", "entire recording"),
    ],
)
def test_invalid_discard_requests_fail_clearly(
    value: float, unit: str, match: str
) -> None:
    timing = TimingDescriptor([0.0, 100.0, 200.0, 300.0], "runner_time", True)
    with pytest.raises(StartupDiscardError, match=match):
        resolve_initial_frame_window(
            discard_value=value,
            discard_unit=unit,
            frame_rate=10.0,
            frame_rate_verified=False,
            timing=timing,
        )


def test_build_timing_descriptor_marks_only_valid_runner_times_trusted() -> None:
    meta = [
        {RUNNER_TIME_KEY: value, "exposure_ms": 50.0} for value in (20.0, 120.0, 250.0)
    ]
    descriptor = build_timing_descriptor(meta, 3)
    assert descriptor.timestamps_ms == [20.0, 120.0, 250.0]
    assert descriptor.source == "runner_time"
    assert descriptor.trusted is True

    descriptor = build_timing_descriptor(meta[:1], 3)
    assert descriptor.timestamps_ms == [0.0, 50.0, 100.0]
    assert descriptor.source == "exposure"
    assert descriptor.trusted is False


def test_source_frame_and_interval_transform() -> None:
    assert source_frame_to_retained(6, 5) == 0
    assert source_frame_to_retained(6, 5, one_based=False) == 1
    assert source_interval_to_retained(4, 4, 5, 10) == (0.0, 2)
    assert source_interval_to_retained(1, 2, 5, 10) is None
    assert source_interval_to_retained(14, 4, 5, 10) == (8, 10.0)


def test_extraction_crops_once_before_roi_processing() -> None:
    runner = ExtractionRunner()
    data = np.arange(6 * 2 * 2, dtype=float).reshape(6, 2, 2)
    meta = [
        {
            RUNNER_TIME_KEY: frame * 100.0,
            "exposure_ms": 100.0,
            "event": {"pos_name": "A1_0000"},
        }
        for frame in range(6)
    ]
    dataset = MagicMock()
    dataset.isel.return_value = (data, meta)

    fov = FOV(
        name="A1_0000",
        position_index=0,
        fov_number=0,
        well_id=1,
        rois=[ROI(label_value=1, fov_id=1)],
    )
    settings = ExtractionSettings(
        discard_initial_value=2,
        discard_initial_unit="frames",
        threads=1,
    )
    observed: dict[str, Any] = {}

    def fake_process(*args: Any, **kwargs: Any) -> _RoiParts:
        observed["data"] = args[0]
        observed["meta"] = args[1]
        return _RoiParts(
            1, args[5], np.zeros(4), None, None, np.zeros(4), 1.0, "pixels"
        )

    mask = np.ones((2, 2), dtype=bool)
    with (
        patch.object(runner, "_get_label_mask", return_value={1: mask}),
        patch.object(
            runner,
            "_prepare_neuropil_masks",
            return_value=({1: mask}, {}),
        ),
        patch.object(runner, "_compute_roi_dff", side_effect=fake_process),
        patch(
            "cali.extraction._extraction_runner.OasisBackend.infer_all",
            return_value=OasisResult(
                np.zeros((1, 4)), np.zeros((1, 4)), np.zeros(1), np.zeros((1, 1))
            ),
        ),
    ):
        result = runner._extract_trace_data_per_position(dataset, settings, None, fov)

    assert result is fov
    np.testing.assert_array_equal(observed["data"], data[2:])
    assert observed["meta"][0][RUNNER_TIME_KEY] == 200.0
    trace = fov.rois[0]._new_traces[0]
    assert trace.x_axis == [0.0, 100.0, 200.0, 300.0]
    assert trace.source_start_frame == 2


def test_startup_discard_gui_round_trip(qtbot: QtBot) -> None:
    widget = _TraceExtractionWidget()
    qtbot.addWidget(widget)

    defaults = widget.value(NeuropilData(), 10.0)
    assert defaults.discard_initial_value == 0
    assert defaults.discard_initial_unit == "frames"
    assert defaults.frame_rate_verified is False

    value = TraceExtractionData(
        discard_initial_value=1.25,
        discard_initial_unit="seconds",
        frame_rate_verified=True,
    )
    widget.setValue(value)
    assert widget.value(NeuropilData(), 10.0) == value

    widget.reset()
    assert widget.value(NeuropilData(), 10.0).discard_initial_unit == "frames"


def test_startup_discard_migration_is_idempotent() -> None:
    engine = create_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        conn.execute(
            text(
                "CREATE TABLE extraction_settings ("
                "id INTEGER PRIMARY KEY, frame_rate REAL DEFAULT 10.0)"
            )
        )
        conn.execute(
            text("CREATE TABLE trace (id INTEGER PRIMARY KEY, raw_trace JSON)")
        )
        conn.execute(text("INSERT INTO extraction_settings (id) VALUES (1)"))
        conn.execute(text("INSERT INTO trace (id, raw_trace) VALUES (1, '[1, 2]')"))
        conn.commit()

    migrate_startup_discard(engine)
    migrate_startup_discard(engine)

    with engine.connect() as conn:
        extraction_columns = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(extraction_settings)"))
        }
        trace_columns = {
            row[1] for row in conn.execute(text("PRAGMA table_info(trace)"))
        }
        row = conn.execute(
            text(
                "SELECT discard_initial_value, discard_initial_unit, "
                "frame_rate_verified FROM extraction_settings WHERE id = 1"
            )
        ).one()

    assert {
        "discard_initial_value",
        "discard_initial_unit",
        "frame_rate_verified",
    } <= extraction_columns
    assert {
        "source_start_frame",
        "source_start_time_ms",
        "original_frame_count",
        "discarded_duration_ms",
        "discard_timing_source",
    } <= trace_columns
    assert tuple(row) == (0.0, "frames", 0)
    engine.dispose(close=True)
