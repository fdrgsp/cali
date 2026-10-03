"""Acquisition timing, backend length limits, and non-invented persisted provenance."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

from cali._constants import EVENT_KEY, RUNNER_TIME_KEY
from cali.extraction._extraction_runner import ExtractionRunner
from cali.extraction._frame_window import (
    StartupDiscardError,
    build_timing_descriptor,
    preflight_retained_timing,
    resolve_initial_frame_window,
    validate_model_timing,
)
from cali.extraction._spike_inference import OasisBackend
from cali.sqlmodel import (
    FOV,
    ROI,
    ExtractionSettings,
    _engine,
    create_cali_engine,
    ensure_schema_current,
)


def test_explicit_frame_period_is_trusted_without_treating_exposure_as_interval() -> (
    None
):
    timing = build_timing_descriptor([{"exposure_ms": 20, "frame_period_ms": 100}], 10)
    assert timing.timestamps_ms == [frame * 100 for frame in range(10)]
    assert timing.source == "metadata_frame_period"
    assert timing.trusted
    assert timing.frame_rate_hz == 10
    assert timing.interval_jitter_fraction is None
    window = resolve_initial_frame_window(
        discard_value=0.21,
        discard_unit="seconds",
        frame_rate=50,
        frame_rate_verified=False,
        timing=timing,
    )
    assert window.source_start_frame == 3
    assert window.source_start_time_ms == 300
    assert window.timing_source == "metadata_frame_period"


def test_actual_timestamps_take_priority_over_period_exposure_and_settings() -> None:
    metadata = [
        {
            RUNNER_TIME_KEY: 1000 + frame * 100,
            "frame_period_ms": 25,
            "exposure_ms": "invalid",
        }
        for frame in range(6)
    ]
    timing = build_timing_descriptor(
        metadata, 6, frame_rate=40, frame_rate_verified=True
    )
    assert timing.source == "runner_time"
    assert timing.timestamps_ms[0] == 1000
    assert timing.frame_rate_hz == 10
    assert timing.interval_jitter_fraction == 0
    with pytest.raises(StartupDiscardError, match="extraction settings"):
        validate_model_timing(timing, settings_frame_rate=40, model_frame_rate=10)


@pytest.mark.parametrize("period", [0, -10, float("nan"), float("inf"), "invalid"])
def test_invalid_declared_period_is_not_silently_replaced_by_exposure(
    period: object,
) -> None:
    with pytest.raises(StartupDiscardError, match="Frame-period metadata"):
        build_timing_descriptor([{"exposure_ms": 100, "frame_period_ms": period}], 10)


def test_conflicting_metadata_periods_require_actual_timestamps() -> None:
    with pytest.raises(StartupDiscardError, match="Conflicting frame-period"):
        build_timing_descriptor([{"frame_period_ms": 100}, {"frame_period_ms": 110}], 2)


@pytest.mark.parametrize(
    "times", [[0, 100, 100], [100, 50, 0], [0, float("nan"), 200], [0, "bad", 200]]
)
def test_invalid_complete_timestamps_fail_even_with_verified_settings(
    times: list,
) -> None:
    with pytest.raises(StartupDiscardError, match="Acquisition timestamps"):
        build_timing_descriptor(
            [{RUNNER_TIME_KEY: value} for value in times],
            3,
            frame_rate=10,
            frame_rate_verified=True,
        )


def test_exposure_axis_preserves_legacy_values_but_cannot_select_cascade_model() -> (
    None
):
    timing = build_timing_descriptor([{"exposure_ms": 20}], 6)
    assert timing.timestamps_ms == [0, 20, 40, 60, 80, 100]
    assert timing.frame_rate_hz is None
    assert timing.interval_jitter_fraction is None
    assert timing.validation == "unverified_exposure"
    with pytest.raises(StartupDiscardError, match="exposure alone"):
        validate_model_timing(timing, settings_frame_rate=50, model_frame_rate=50)


def test_verified_rate_provides_timing_when_exposure_is_zero_even_without_discard() -> (
    None
):
    timing = build_timing_descriptor(
        [{"exposure_ms": 0}], 6, frame_rate=10, frame_rate_verified=True
    )
    assert timing.source == "user_verified"
    assert timing.timestamps_ms == [0, 100, 200, 300, 400, 500]
    assert timing.frame_rate_hz == 10
    assert timing.interval_jitter_fraction is None
    assert (
        validate_model_timing(timing, settings_frame_rate=10, model_frame_rate=10) == 10
    )


@pytest.mark.parametrize("rate", [None, 0, -10, float("inf"), float("nan")])
def test_invalid_user_verified_rate_fails_early(rate: float | None) -> None:
    with pytest.raises(StartupDiscardError, match="positive verified frame rate"):
        build_timing_descriptor([], 10, frame_rate=rate, frame_rate_verified=True)


def test_irregular_timestamps_can_crop_seconds_but_cannot_authorize_cascade() -> None:
    timing = build_timing_descriptor(
        [{RUNNER_TIME_KEY: value} for value in (1000, 1080, 1190, 1310, 1450, 1600)], 6
    )
    window = resolve_initial_frame_window(
        discard_value=0.2,
        discard_unit="seconds",
        frame_rate=10,
        frame_rate_verified=False,
        timing=timing,
    )
    assert window.source_start_frame == 3
    with pytest.raises(StartupDiscardError, match="uniform intervals"):
        validate_model_timing(timing, settings_frame_rate=10, model_frame_rate=10)


def test_model_timing_uses_sample_intervals_and_validates_both_rates() -> None:
    timing = build_timing_descriptor(
        [{RUNNER_TIME_KEY: frame * 100} for frame in range(6)], 6
    )
    assert (
        validate_model_timing(timing, settings_frame_rate=10, model_frame_rate=10) == 10
    )
    assert timing.frame_rate_hz != 6 / 0.5
    with pytest.raises(StartupDiscardError, match=r"CASCADE model.*30"):
        validate_model_timing(timing, settings_frame_rate=10, model_frame_rate=30)
    with pytest.raises(StartupDiscardError, match="extraction settings"):
        validate_model_timing(timing, settings_frame_rate=12, model_frame_rate=10)


@pytest.mark.parametrize(("deviation_ms", "valid"), [(0.9, True), (2.0, False)])
def test_model_timing_enforces_documented_one_percent_interval_tolerance(
    deviation_ms: float, valid: bool
) -> None:
    values = [0, 100, 200 + deviation_ms, 300 + deviation_ms, 400 + deviation_ms]
    timing = build_timing_descriptor([{RUNNER_TIME_KEY: value} for value in values], 5)
    if valid:
        assert (
            validate_model_timing(timing, settings_frame_rate=10, model_frame_rate=10)
            == 10
        )
    else:
        with pytest.raises(StartupDiscardError, match="uniform intervals"):
            validate_model_timing(timing, settings_frame_rate=10, model_frame_rate=10)


def test_cascade_jitter_check_uses_retained_window_after_startup_discard() -> None:
    values = [0, 250, 350, 450, 550, 650, 750]
    timing = build_timing_descriptor([{RUNNER_TIME_KEY: value} for value in values], 7)
    assert timing.interval_jitter_fraction == 1.5
    window = resolve_initial_frame_window(
        discard_value=2,
        discard_unit="frames",
        frame_rate=10,
        frame_rate_verified=False,
        timing=timing,
    )
    retained = preflight_retained_timing(
        timing, window, frame_rate=10, minimum_frames={"OASIS": 5}
    )
    assert retained.timestamps_ms == [0, 100, 200, 300, 400]
    assert retained.frame_rate_hz == 10
    assert retained.interval_jitter_fraction == 0
    assert (
        validate_model_timing(retained, settings_frame_rate=10, model_frame_rate=10)
        == 10
    )


@pytest.mark.parametrize("retained_count", [1, 2, 3, 4])
def test_retained_preflight_names_the_limiting_consumer_and_counts(
    retained_count: int,
) -> None:
    timing = build_timing_descriptor([{"exposure_ms": 100}], 10)
    window = resolve_initial_frame_window(
        discard_value=10 - retained_count,
        discard_unit="frames",
        frame_rate=10,
        frame_rate_verified=False,
        timing=timing,
    )
    with pytest.raises(
        StartupDiscardError, match=f"leaves {retained_count} retained; OASIS"
    ):
        preflight_retained_timing(
            timing, window, frame_rate=10, minimum_frames={"ΔF/F": 1, "OASIS": 5}
        )


def test_selected_model_can_declare_a_stricter_retained_length_requirement() -> None:
    timing = build_timing_descriptor([{"frame_period_ms": 100}], 64)
    window = resolve_initial_frame_window(
        discard_value=0,
        discard_unit="frames",
        frame_rate=10,
        frame_rate_verified=False,
        timing=timing,
    )
    with pytest.raises(
        StartupDiscardError, match="CASCADE model test requires at least 65"
    ):
        preflight_retained_timing(
            timing,
            window,
            frame_rate=10,
            minimum_frames={"OASIS": 5, "CASCADE model test": 65},
        )


@pytest.mark.parametrize("exposure", [0, -1, float("nan"), float("inf")])
def test_invalid_fallback_timing_is_rejected_before_roi_calculation(
    exposure: float,
) -> None:
    runner = ExtractionRunner()
    dataset = MagicMock(path=Path("acquisition.tif"))
    data = np.ones((10, 2, 2))
    dataset.isel.return_value = (
        data,
        [{"exposure_ms": exposure, EVENT_KEY: {"pos_name": "A1_0"}}],
    )
    fov = FOV(name="A1_0", position_index=0, rois=[ROI(label_value=1)])
    with (
        patch.object(runner, "_get_label_mask") as masks,
        patch.object(runner, "_compute_roi_dff") as dff,
        patch("cali.extraction._extraction_runner.OasisBackend.infer_all") as inference,
        pytest.raises(
            StartupDiscardError, match=r"A1_0.*acquisition\.tif.*Retained timing"
        ),
    ):
        runner._extract_trace_data_per_position(
            dataset, ExtractionSettings(), None, fov
        )
    masks.assert_not_called()
    dff.assert_not_called()
    inference.assert_not_called()
    assert not hasattr(fov.rois[0], "_new_traces")
    np.testing.assert_array_equal(data, np.ones((10, 2, 2)))


def test_short_recording_error_includes_fov_file_and_limiting_counts() -> None:
    runner = ExtractionRunner()
    dataset = MagicMock(path=Path("acquisition.tif"))
    dataset.isel.return_value = (
        np.ones((8, 2, 2)),
        [{"exposure_ms": 100, "file_path": "A1.tif", EVENT_KEY: {"pos_name": "A1_0"}}],
    )
    fov = FOV(name="A1_0", position_index=0, rois=[ROI(label_value=1)])
    with (
        patch.object(runner, "_get_label_mask") as masks,
        pytest.raises(
            StartupDiscardError,
            match=(
                r"A1_0.*A1\.tif.*8 source frames minus 4 discarded "
                r"leaves 4 retained; OASIS"
            ),
        ),
    ):
        runner._extract_trace_data_per_position(
            dataset, ExtractionSettings(discard_initial_value=4), None, fov
        )
    masks.assert_not_called()


def test_oasis_minimum_guards_the_actual_welch_noise_band() -> None:
    backend = OasisBackend()
    with pytest.raises(ValueError, match="at least 5 frames"):
        backend.infer_all(np.ones((1, 4)), 10)
    for count in range(backend.minimum_frames, 14):
        result = backend.infer_all(
            np.linspace(0, 1, count)[None, :], 10, decay_constant=0.5
        )
        assert np.isfinite(result.sn_by_roi).all()
        assert np.isfinite(result.den_dff).all()
    with pytest.raises(ValueError, match="finite DFF"):
        backend.infer_all(np.full((1, 10), float("nan")), 10)


def _version_eight_window(path: Path) -> None:
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE extraction_frame_window (id INTEGER PRIMARY KEY, "
            "source_start_frame INTEGER, timing_source VARCHAR)"
        )
        connection.exec_driver_sql(
            "INSERT INTO extraction_frame_window VALUES (1, 2, 'exposure')"
        )
        connection.exec_driver_sql("PRAGMA user_version = 8")
    engine.dispose()


def test_timing_migration_preserves_unknown_historical_sampling_and_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy_timing.cali"
    _version_eight_window(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    ensure_schema_current(engine)
    with engine.connect() as connection:
        assert tuple(
            connection.exec_driver_sql(
                "SELECT source_start_frame, timing_source, acquisition_frame_rate_hz, "
                "interval_jitter_fraction, timing_validation "
                "FROM extraction_frame_window"
            ).one()
        ) == (2, "exposure", None, None, None)
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 9
    engine.dispose()


def test_timing_migration_rolls_back_columns_and_version_then_retries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "timing_retry.cali"
    _version_eight_window(path)
    engine = create_engine(f"sqlite:///{path}")

    def interrupted(connection: Connection) -> None:
        _engine._acquisition_timing(connection)
        raise RuntimeError("timing migration interrupted")

    with patch.object(_engine, "_MIGRATIONS", (*_engine._MIGRATIONS[:8], interrupted)):
        with pytest.raises(RuntimeError, match="interrupted"):
            ensure_schema_current(engine)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 8
        assert "acquisition_frame_rate_hz" not in {
            row[1]
            for row in connection.exec_driver_sql(
                "PRAGMA table_info(extraction_frame_window)"
            )
        }
    ensure_schema_current(engine)
    engine.dispose()
