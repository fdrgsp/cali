"""Acquisition pauses are boundaries, never synthetic model samples or silence."""

import json
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection

from cali.analysis._fov_analysis_parallel import compute_spike_population
from cali.analysis._fov_inputs import SpikePopulation
from cali.analysis._roi_analysis import analyze_spike_trace, valid_spike_events
from cali.extraction._frame_window import (
    StartupDiscardError,
    TimingDescriptor,
    build_timing_descriptor,
    cascade_timing_chunks,
)
from cali.plot._spike_data import spike_plot_data
from cali.plot._spike_fov_data import spike_burst_bounds, spike_population_duration
from cali.sqlmodel import (
    DataAnalysis,
    ExtractionFrameWindow,
    SpikeAnalysisSettings,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    _engine,
    create_cali_engine,
)

from .test_cascade_reference import fake_reference as fake_reference


def _paused_timing() -> TimingDescriptor:
    return TimingDescriptor(
        np.r_[np.arange(96) * 100, 10000 + np.arange(96) * 100].tolist(),
        "runner_time",
        True,
    )


def test_evoked_recording_splits_at_its_three_recorded_stimulation_pauses() -> None:
    path = (
        Path(__file__).parent
        / "test_data/data_and_db_for_tests/evk.tensorstore.zarr/.zattrs"
    )
    rows = json.loads(path.read_text())["frame_metadatas"]
    rows = [row for row in rows if row["mda_event"]["index"]["p"] == 0]
    timing = build_timing_descriptor(rows, 153)
    assert cascade_timing_chunks(
        timing, settings_frame_rate=10, model_frame_rate=10, minimum_frames=33
    ) == [(0, 3), (3, 53), (53, 103), (103, 153)]


def test_pauses_split_bursts_but_jitter_and_rate_errors_still_fail() -> None:
    timing = _paused_timing()
    assert cascade_timing_chunks(
        timing, settings_frame_rate=10, model_frame_rate=10, minimum_frames=65
    ) == [(0, 96), (96, 192)]
    jittered = replace(timing, timestamps_ms=timing.timestamps_ms.copy())
    jittered.timestamps_ms[20] += 2
    with pytest.raises(StartupDiscardError, match="uniform intervals"):
        cascade_timing_chunks(jittered, settings_frame_rate=10, model_frame_rate=10)
    with pytest.raises(StartupDiscardError, match="extraction settings"):
        cascade_timing_chunks(timing, settings_frame_rate=20, model_frame_rate=10)
    with pytest.raises(StartupDiscardError, match="No acquisition chunk"):
        cascade_timing_chunks(
            timing, settings_frame_rate=10, model_frame_rate=10, minimum_frames=100
        )


def test_independent_predictions_mask_edges_count_only_valid_time_and_plot_gaps(
    fake_reference: tuple,
) -> None:
    backend, _, calls = fake_reference
    result = backend.infer_all(
        np.random.default_rng(22).normal(0, 0.1, (2, 192)), 10, timing=_paused_timing()
    )
    assert sum(call[0] == "predict" for call in calls) == 2
    assert [(s["start"], s["stop"]) for s in result.segments] == [(0, 96), (96, 192)]
    spike = SpikeTrace(
        values=result.spikes[0].tolist(),
        valid_start=result.valid_start,
        valid_stop=result.valid_stop,
        segments=list(result.segments),
        inference_run=SpikeInferenceRun(
            method="cascade",
            units="spikes/frame",
            backend_package="CascadeTorch",
            model_sampling_rate_hz=10,
            smoothing_sigma=0.2,
        ),
    )
    trace = Traces(
        raw_trace=[1.0] * 192,
        dff=[0.1] * 192,
        den_dff=[0.1] * 192,
        spike_traces=[spike],
        x_axis=_paused_timing().timestamps_ms,
        extraction_frame_window=ExtractionFrameWindow(
            retained_frame_count=192, acquisition_frame_rate_hz=10
        ),
    )
    assert spike.valid_intervals == [(32, 64), (128, 160)]
    metric = analyze_spike_trace(
        spike,
        SpikeAnalysisSettings(
            method="cascade", threshold_mode="global", threshold_value=0.1
        ),
        legacy_duration_s=100,
        source_trace=trace,
    )
    assert metric.expected_spike_count == 16
    assert metric.expected_spike_rate_hz == 2.5
    assert valid_spike_events(spike, 0.1)[1].sum() == 0
    data = spike_plot_data(trace, DataAnalysis(spike_analyses=[metric]), "cascade")
    assert data is not None
    np.testing.assert_array_equal(
        np.flatnonzero(np.isfinite(data.values)), spike.valid_frame_indices()
    )
    assert data.event_frames().tolist() == spike.valid_frame_indices()
    snapshot = spike.model_dump_json()
    restored = SpikeTrace.model_validate_json(snapshot)
    assert restored.valid_intervals == spike.valid_intervals


def test_short_initial_chunk_is_unavailable_and_no_sample_is_interpolated(
    fake_reference: tuple,
) -> None:
    backend, _, calls = fake_reference
    timing = TimingDescriptor(
        np.r_[[0, 100, 200], 600 + np.arange(96) * 100].tolist(), "runner_time", True
    )
    data = np.random.default_rng(3).normal(0, 0.1, (1, 99))
    original = data.copy()
    result = backend.infer_all(data, 10, timing=timing)
    assert result.spikes.shape == data.shape
    assert result.segments[0]["valid_start"] is None
    assert result.segments[1]["valid_start"] == 35
    assert sum(call[0] == "predict" for call in calls) == 1
    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(result.spikes[:, :35], 0)


def test_population_bursts_and_ccg_never_bridge_internal_invalid_regions() -> None:
    values = np.r_[np.ones(8), np.zeros(12), np.ones(8)]
    population = SpikePopulation(
        method="cascade",
        units="spikes/frame",
        frame_rate=10,
        labels=[1, 2],
        trains=[values.copy(), values.copy()],
        binary={"1": values.tolist(), "2": values.tolist()},
        onsets={"1": np.zeros(28).tolist(), "2": np.zeros(28).tolist()},
        valid_start=10,
        valid_stop=38,
        valid_intervals=[(10, 18), (30, 38)],
    )
    child = compute_spike_population(
        population,
        SpikeAnalysisSettings(
            method="cascade",
            threshold_mode="global",
            threshold_value=0.1,
            ccg_n_shuffles=2,
        ),
        name="paused",
        parallel=False,
        n_workers=1,
    )
    assert child.valid_intervals == [[10, 18], [30, 38]]
    assert spike_population_duration(child) == 1.6
    assert all(
        any(a <= s < e <= b for a, b in child.valid_intervals)
        for s, e in spike_burst_bounds(child)
    )
    assert child.spike_max_lag_correlation_matrix is not None
    assert child.spike_population_activity is not None
    assert len(child.spike_population_activity) == 28


def test_schema_fifteen_preserves_arrays_and_rolls_back_before_retry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "before-chunks.cali"
    old = create_engine(f"sqlite:///{path}")
    with old.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE spike_trace (id INTEGER PRIMARY KEY, value BLOB)"
        )
        connection.exec_driver_sql("INSERT INTO spike_trace VALUES (1, X'01020304')")
        connection.exec_driver_sql(
            "CREATE TABLE spike_fov_analysis (id INTEGER PRIMARY KEY)"
        )
        connection.exec_driver_sql("PRAGMA user_version=14")
    original = _engine._add_columns

    def failing(
        connection: Connection, table: str, columns: tuple[tuple[str, str], ...]
    ) -> None:
        original(connection, table, columns)
        if table == "spike_fov_analysis":
            raise RuntimeError("interrupted chunk migration")

    with patch.object(_engine, "_add_columns", failing):
        with pytest.raises(RuntimeError, match="interrupted chunk migration"):
            create_cali_engine(f"sqlite:///{path}")
    with old.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 14
        assert "segments" not in {
            row[1]
            for row in connection.exec_driver_sql("PRAGMA table_info(spike_trace)")
        }
    old.dispose()
    upgraded = create_cali_engine(f"sqlite:///{path}")
    with upgraded.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 15
        assert connection.exec_driver_sql(
            "SELECT value, segments FROM spike_trace"
        ).one() == (b"\x01\x02\x03\x04", None)
    upgraded.dispose()


@pytest.mark.skipif(
    os.environ.get("CALI_CASCADE_REFERENCE_TESTS") != "1",
    reason="Optional pretrained backend comparison",
)
def test_pretrained_paused_reference_and_cached_service_match() -> None:
    from cali.extraction._spike_inference._cascade_reference import (
        CascadeReferenceBackend,
    )
    from cali.extraction._spike_inference._cascade_service import (
        CascadeInferenceService,
    )

    metadata = json.loads(
        (Path(__file__).parent / "fixtures/cascade_reference/manifest.json").read_text()
    )
    name = metadata["model_name"]
    inputs = np.random.default_rng(20).normal(0, 0.1, (2, 192))
    timing = TimingDescriptor(
        np.r_[np.arange(96) * 1000 / 30, 3600 + np.arange(96) * 1000 / 30].tolist(),
        "runner_time",
        True,
    )
    reference = CascadeReferenceBackend(name, device="cpu").infer_all(
        inputs, 30, timing=timing
    )
    with CascadeInferenceService(name, device="cpu", max_windows=17) as service:
        cached = service.infer_all(inputs, 30, timing=timing)
    np.testing.assert_allclose(reference.spikes, cached.spikes, rtol=1e-5, atol=1e-6)
    assert reference.segments == cached.segments
