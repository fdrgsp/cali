"""Source/retained coordinate boundaries and persisted plot/export integration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from pytestqt.qtbot import QtBot
from sqlmodel import Session, select

from cali._constants import RAW_CALCIUM_TRACES
from cali.extraction._frame_window import SourceFrameTransform
from cali.plot._main_plot import _source_coordinate_tooltip
from cali.plot._multi_wells_plots._evoked_activity import (
    _query_evoked_amplitudes_by_condition,
)
from cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (  # noqa: E501
    _outside_intervals,
    _pulse_intervals,
)
from cali.plot._single_wells_plots.evoked._plot_evoked_experiment_data_plots import (
    _add_led_stimulation_bands,
)
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    DetectionSettings,
    Experiment,
    ExtractionFrameWindow,
    ExtractionSettings,
    Plate,
    SpikeAnalysis,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    Well,
    create_cali_engine,
    create_database_and_tables,
)
from cali.util import (
    export_events_to_csv,
    export_frame_coordinates_to_csv,
    export_raw_traces_to_csv,
)
from cali.util._database_to_csv import export_traces_to_csv
from tests.test_runner_sources import _database, _run


def test_irregular_sample_times_and_both_frame_bases_round_trip() -> None:
    transform = SourceFrameTransform(
        3,
        3,
        250,
        1000,
        (0, 70, 180),
    )
    for retained in range(3):
        assert transform.to_retained(transform.to_source(retained)) == retained
        assert transform.to_source(retained, one_based=False) == retained + 3
    assert transform.frame_times(2) == (180, 430, 1430)
    assert transform.source_time_to_retained(430) == 180
    assert transform.retained_time_to_source(180) == 430
    with pytest.raises(ValueError, match="outside"):
        transform.frame_times(3)


def test_uncropped_absolute_legacy_axis_keeps_relative_export_times() -> None:
    transform = SourceFrameTransform(0, 3, 0, 1200, (1200, 1300, 1410))
    assert transform.frame_times(1) == (100, 100, 1300)
    assert transform.source_time_to_retained(100) == 1300
    assert transform.retained_time_to_source(1300) == 100
    unknown = SourceFrameTransform(0, 3, retained_timestamps_ms=(0, 100, 200))
    assert unknown.frame_times(1) == (100, 100, None)


@pytest.mark.parametrize(
    "source", ["exposure", "user_verified", "metadata_frame_period"]
)
def test_synthetic_timing_does_not_invent_an_absolute_timestamp(source: str) -> None:
    trace = Traces(
        raw_trace=[1.0] * 5,
        x_axis=[0, 100, 200, 300, 400],
        x_axis_units="ms",
        extraction_frame_window=ExtractionFrameWindow(
            timing_source=source,
            source_time_origin_ms=0,
            source_start_time_ms=200,
            source_start_frame=2,
            original_frame_count=7,
            retained_frame_count=5,
            schema_version=3,
        ),
    )
    assert trace.source_frame_transform().frame_times(1) == (100, 300, None)


@pytest.mark.parametrize(
    "source,duration,expected",
    [
        (1, 2, None),
        (3, 3, (0, 2)),
        (4, 1, (0, 1)),
        (7, 4, (3, 5)),
        (9, 1, None),
    ],
)
def test_source_intervals_are_intersected_at_both_boundaries(
    source: int,
    duration: int,
    expected: tuple[int, int] | None,
) -> None:
    assert SourceFrameTransform(3, 5).clip_interval(source, duration) == expected


def test_response_windows_crossing_cutoff_and_overlaps_have_exact_complement() -> None:
    transform = SourceFrameTransform(5, 10)
    # Source frame 5 is discarded, but its +/-2 response window overlaps retained data.
    intervals = _pulse_intervals([1, 5, 7, 7, 15, 30], transform, 2)
    assert intervals == [(0, 4), (7, 10)]
    assert _outside_intervals(intervals, 10) == [(4, 7)]
    assert _outside_intervals([], 10) == [(0, 10)]
    assert _outside_intervals([(0, 10)], 10) == []
    covered = np.zeros(10, dtype=int)
    for start, stop in intervals + _outside_intervals(intervals, 10):
        covered[start:stop] += 1
    assert covered.tolist() == [1] * 10


def test_historical_window_sampling_keeps_legacy_onsets_and_overlap_weight() -> None:
    transform = SourceFrameTransform(5, 10, source_one_based=False)
    assert _pulse_intervals([4, 6, 7], transform, 2) == [(0, 4), (0, 5)]


def _coordinate_database(path: Path) -> tuple:
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        experiment = Experiment(name="source coordinates")
        plate = Plate(name="plate", experiment=experiment)
        well = Well(name="A1", row=0, column=0, plate=plate)
        detection, extraction = DetectionSettings(), ExtractionSettings()
        settings = AnalysisSettings(
            experiment_type="evoked",
            frame_rate=10,
            led_pulse_on_frames=[3],
            led_pulse_powers=[10],
            led_pulse_duration=300,
        )
        session.add_all([well, detection, extraction, settings])
        session.flush()
        result = CaliResult(
            experiment=experiment.id,
            detection_settings_id=detection.id,
            extraction_settings_id=extraction.id,
            analysis_settings_id=settings.id,
            positions_extracted=[0, 1],
            legacy_trace_resolution="self",
        )
        session.add(result)
        session.flush()
        result.source_extraction_result_id = result.id
        inference = SpikeInferenceRun(backend_version="test", resolved_device="cpu")
        for position, count in enumerate((5, 6)):
            fov = FOV(name=f"A1_{position}", position_index=position, well=well)
            roi = ROI(
                fov=fov,
                label_value=1,
                detection_settings_id=detection.id,
                active=True,
                stimulated=True,
            )
            spike = SpikeTrace(
                values=[2, 3, 0, 4, 6, 0][:count],
                valid_start=1,
                valid_stop=4,
                inference_run=inference,
            )
            trace = Traces(
                roi=roi,
                analysis_result=result,
                raw_trace=[1.0] * count,
                dff=[0.0] * count,
                den_dff=list(range(2, 2 + count)),
                x_axis=[frame * 100.0 for frame in range(count)],
                x_axis_units="ms",
                spike_traces=[spike],
                extraction_frame_window=ExtractionFrameWindow(
                    source_start_frame=2,
                    original_frame_count=count + 2,
                    retained_frame_count=count,
                    source_start_time_ms=200,
                    source_time_origin_ms=1000,
                    timing_source="runner_time",
                    schema_version=3,
                    provenance_source="extraction",
                ),
            )
            analysis = DataAnalysis(
                roi=roi,
                analysis_result=result,
                peaks_den_dff=[0, 3],
                peaks_height_den_dff=1,
                spike_analyses=[
                    SpikeAnalysis(
                        spike_trace=spike, threshold=1, threshold_mode="global"
                    )
                ],
            )
            session.add_all([trace, analysis])
        session.commit()
        return engine, result.id


def test_exports_preserve_ragged_traces_and_exact_source_coordinates(
    tmp_path: Path,
) -> None:
    engine, result_id = _coordinate_database(tmp_path / "events.cali")
    export_raw_traces_to_csv(engine, tmp_path / "raw.csv", run_id=result_id)
    raw = pd.read_csv(tmp_path / "raw.csv")
    assert raw.shape == (6, 2)
    assert pd.isna(raw.iloc[5, 0])
    assert raw.iloc[5, 1] == 1
    export_frame_coordinates_to_csv(
        engine, tmp_path / "coordinates.csv", run_id=result_id
    )
    coordinates = pd.read_csv(tmp_path / "coordinates.csv")
    assert len(coordinates) == 11
    first = coordinates.iloc[0]
    assert first.retained_frame_0based == 0
    assert first.source_frame_0based == 2
    assert first.source_frame_1based == 3
    assert first.retained_time_ms == 0
    assert first.source_time_ms == 200
    assert first.source_timestamp_ms == 1200
    export_events_to_csv(
        engine, tmp_path / "events.csv", run_id=result_id, position_indices=[0]
    )
    events = pd.read_csv(tmp_path / "events.csv")
    assert events.retained_frame_0based.tolist() == [0, 3, 3]
    assert events.source_frame_1based.tolist() == [3, 6, 6]
    assert events.source_timestamp_ms.tolist() == [1200, 1500, 1500]
    assert events.event_type.tolist() == [
        "calcium_peak",
        "calcium_peak",
        "threshold_excursion_start",
    ]
    spike = events.iloc[2]
    assert spike.method == "oasis" and spike.units == "a.u."
    assert spike.valid_start_frame_0based == 1
    assert spike.valid_stop_frame_0based == 4
    engine.dispose()


def test_selected_trace_exports_include_coordinate_and_event_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selected.cali"
    engine, result_id = _coordinate_database(path)
    export_traces_to_csv(
        engine, {RAW_CALCIUM_TRACES: True}, result_id, path, position_indices=[1]
    )
    folder = tmp_path / "selected_exports" / f"run_{result_id}"
    for name in ("frame_coordinates.csv", "events.csv"):
        data = pd.read_csv(folder / name)
        assert set(data.position_index) == {1}
    engine.dispose()


def test_evoked_peak_classification_uses_versioned_source_frame_base(
    tmp_path: Path,
) -> None:
    engine, result_id = _coordinate_database(tmp_path / "classification.cali")
    with Session(engine) as session:
        settings = session.exec(select(AnalysisSettings)).one()
        original_pulses = list(settings.led_pulse_on_frames)
    native = _query_evoked_amplitudes_by_condition(engine, run_id=result_id)
    # The source input frame 3 is the first retained sample after discarding 2.
    amplitudes = [
        amp
        for wells in native.values()
        for fovs in wells.values()
        for powers in fovs.values()
        for amp in powers.values()
    ]
    assert amplitudes == [[2.0], [2.0]]
    with Session(engine) as session:
        for window in session.exec(select(ExtractionFrameWindow)).all():
            window.schema_version = 2
        session.commit()
    historical = _query_evoked_amplitudes_by_condition(engine, run_id=result_id)
    amplitudes = [
        amp
        for wells in historical.values()
        for fovs in wells.values()
        for powers in fovs.values()
        for amp in powers.values()
    ]
    assert amplitudes == [[5.0], [5.0]]
    with Session(engine) as session:
        assert (
            session.exec(select(AnalysisSettings)).one().led_pulse_on_frames
            == original_pulses
        )
    engine.dispose()


def test_plot_bands_mark_clipped_source_pulses_and_offsets(
    tmp_path: Path, qtbot: QtBot
) -> None:
    engine, result_id = _coordinate_database(tmp_path / "bands.cali")
    with Session(engine) as session:
        settings = session.exec(select(AnalysisSettings)).one()
        settings.led_pulse_on_frames = [2]
        session.commit()
    widget = pg.PlotWidget()
    qtbot.addWidget(widget)
    plot = widget.getPlotItem()
    _add_led_stimulation_bands(plot, engine, result_id, "A1_0", stride=2)
    bands = [item for item in plot.items if isinstance(item, pg.LinearRegionItem)]
    assert len(bands) == 1
    assert bands[0].getRegion() == (0, 1)
    assert "clipped" in bands[0].toolTip()
    assert "2 frames (200 ms)" in bands[0].toolTip()
    metadata = _source_coordinate_tooltip(engine, "A1_0", result_id, None)
    assert "2 frames (200 ms)" in metadata
    assert "one-based" in metadata
    engine.dispose()


def test_analysis_only_keeps_exact_generation_transform_and_export_indices(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reanalysis.cali"
    graph = _database(path)
    engine = create_cali_engine(f"sqlite:///{path}")
    with Session(engine) as session:
        for window in session.exec(select(ExtractionFrameWindow)).all():
            window.source_start_frame = 4
            window.original_frame_count = 104
            window.source_start_time_ms = 400
            window.schema_version = 3
        session.commit()
    _run(path, graph, export_traces={RAW_CALCIUM_TRACES: True})
    with Session(engine) as session:
        results = session.exec(
            select(CaliResult).where(CaliResult.analysis_settings_id == graph[3])
        ).all()
        assert len(results) == 2
        for result in results:
            assert result.source_extraction_result_id in graph[4]
            for trace in result.traces:
                transform = trace.source_frame_transform()
                assert transform.source_one_based
                assert transform.to_source(10) == 15
                assert transform.frame_times(10) == (1000, 1400, None)
            folder = tmp_path / "reanalysis_exports" / f"run_{result.id}"
            events = pd.read_csv(folder / "events.csv")
            assert set(events.source_frame_1based - events.retained_frame_0based) == {5}
    engine.dispose()
