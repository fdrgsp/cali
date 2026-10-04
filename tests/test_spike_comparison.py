"""Comparisons preserve paired membership, units, valid bounds and source timing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from sqlalchemy import update
from sqlmodel import Session, select

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
from cali.plot import (
    AnalysisGroup,
    get_available_plots,
    get_stored_spike_capabilities,
    plot_single_well_data,
)
from cali.plot._spike_comparison import (
    get_spike_comparisons,
    paired_spike_comparison,
)
from cali.sqlmodel import DataAnalysis, SpikeAnalysis, SpikeInferenceRun, Traces
from cali.util import export_spike_comparison_to_csv

from .test_cascade_fov_analysis import _fov
from .test_spike_export import spike_database as spike_database
from .test_spike_fov_plot_selection import plot_widget as plot_widget


def test_common_interval_uses_original_units_and_independent_scaling() -> None:
    fov = _fov()
    trace = fov.rois[2]._new_traces[0]
    trace.get_spike_trace("oasis").values[5] = 1000  # outside common interval
    pair = paired_spike_comparison(trace, 3)
    assert (pair.valid_start, pair.valid_stop, pair.frame_rate_hz) == (20, 85, 10)
    for method in ("oasis", "cascade"):
        assert len(pair.values(method)) == 65
        assert max(pair.values(method)) == 0.5
        assert max(pair.normalized_values(method)) == 1
    assert pair.oasis.spike.inference_run.units == "a.u."
    assert pair.cascade.spike.inference_run.units == "spikes/frame"
    assert trace.get_spike_trace("oasis").values[5] == 1000
    with pytest.raises(ValueError, match="only OASIS"):
        pair.values("invalid")


def test_onsets_do_not_invent_common_boundary_events() -> None:
    roi = _fov().rois[2]
    trace, analysis = roi._new_traces[0], roi._new_data_analysis[0]
    oasis = trace.get_spike_trace("oasis")
    oasis.values[20] = 0.5
    pair = paired_spike_comparison(trace, 3, analysis, require_threshold=True)
    assert pair.event_frames("oasis").tolist() == [30, 70]
    assert pair.event_frames("cascade").tolist() == [30, 70]
    assert pair.valid_start == 20
    for method in ("oasis", "cascade"):
        data = pair.oasis if method == "oasis" else pair.cascade
        data.spike.values[84] = 0.5
        data.spike.values[85] = 0.5
    pair = paired_spike_comparison(trace, 3, analysis, require_threshold=True)
    assert pair.event_frames("oasis").tolist() == [30, 70, 84]
    assert pair.event_frames("cascade").tolist() == [30, 70, 84]


@pytest.mark.parametrize("methods", [("oasis",), ("cascade",)])
def test_single_outputs_never_substitute_for_a_pair(methods: tuple) -> None:
    assert paired_spike_comparison(_fov(methods).rois[0]._new_traces[0], 1) is None


@pytest.mark.parametrize(
    "fault,match",
    [
        ("length", "matching retained lengths"),
        ("window", "matching extraction frame window"),
        ("source", "share their extraction source"),
        ("window_source", "frame window source"),
        ("rate", "acquisition/model rates"),
        ("units", "inference units"),
        ("nonfinite", "must be finite"),
        ("timestamps", "retained time axis"),
        ("axis_rate", "match the acquisition rate"),
        ("axis_units", "known time-axis units"),
    ],
)
def test_malformed_pair_is_rejected(fault: str, match: str) -> None:
    trace = _fov().rois[2]._new_traces[0]
    cascade = trace.get_spike_trace("cascade")
    if fault == "length":
        cascade.values.append(0)
    elif fault == "window":
        trace.extraction_frame_window.retained_frame_count = 99
    elif fault == "source":
        cascade.inference_run.extraction_result_id = 100
    elif fault == "window_source":
        trace.extraction_frame_window.extraction_result_id = 100
    elif fault == "rate":
        cascade.inference_run.model_sampling_rate_hz = 30
    elif fault == "units":
        cascade.inference_run.units = "a.u."
    elif fault == "nonfinite":
        cascade.values[30] = float("nan")
    elif fault == "timestamps":
        trace.x_axis.pop()
    elif fault == "axis_rate":
        trace.x_axis[20] += 20
    else:
        trace.x_axis_units = "unknown"
    with pytest.raises(ValueError, match=match):
        paired_spike_comparison(trace, 3)


def test_disjoint_intervals_and_zero_traces() -> None:
    fov = _fov()
    trace = fov.rois[0]._new_traces[0]
    pair = paired_spike_comparison(trace, 1)
    assert np.isfinite(pair.normalized_values("cascade")).all()
    assert not pair.normalized_values("cascade").any()
    trace.get_spike_trace("oasis").valid_stop = 10
    assert paired_spike_comparison(trace, 1) is None


def test_unstored_comparison_metrics_must_bind_the_same_source_trace() -> None:
    roi = _fov().rois[2]
    trace, analysis = roi._new_traces[0], roi._new_data_analysis[0]
    analysis.get_spike_analysis("cascade").spike_trace = (
        _fov().rois[2]._new_traces[0].get_spike_trace("cascade")
    )
    with pytest.raises(ValueError, match="analyzed source trace"):
        paired_spike_comparison(trace, 3, analysis)


def test_missing_thresholds_leave_raw_comparison_available() -> None:
    roi = _fov().rois[2]
    trace, analysis = roi._new_traces[0], roi._new_data_analysis[0]
    analysis.get_spike_analysis("cascade").threshold = None
    assert paired_spike_comparison(trace, 3, analysis) is not None
    assert paired_spike_comparison(trace, 3, analysis, require_threshold=True) is None


def test_reader_requires_exact_run_and_keeps_all_paired_rois(
    spike_database: tuple,
) -> None:
    engine, _, run_id, methods = spike_database
    pairs = get_spike_comparisons(engine, "A1_0", run_id)
    if len(methods) == 2:
        assert [pair.roi_label for pair in pairs] == [1, 2, 3]
        assert [pair.valid_start for pair in pairs] == [10, 10, 20]
        assert get_spike_comparisons(engine, "A1_0", run_id, [3])[0].roi_label == 3
    else:
        assert pairs == []
    assert get_spike_comparisons(engine, "A1_0", run_id + 100) == []
    assert get_spike_comparisons(engine, "A1_0", run_id, []) == []
    with pytest.raises(ValueError, match="one selected"):
        get_spike_comparisons(engine, "A1_0", None)


def test_comparison_availability_requires_both_methods_metrics_and_stages(
    spike_database: tuple,
) -> None:
    engine, _, run_id, methods = spike_database
    capabilities = get_stored_spike_capabilities(engine, run_id)

    def names(**kwargs: object) -> list[str]:
        plots = get_available_plots(
            AnalysisGroup.SINGLE_WELL,
            stored_spike_methods=methods,
            **kwargs,
        )
        return [name for group in plots.values() for name in group]

    normalized = "OASIS / CASCADE Normalized Trace Comparison"
    onsets = "OASIS / CASCADE Threshold Onset Comparison"
    for active in methods:
        available = names(
            has_extraction=True,
            has_analysis=True,
            spike_method=active,
            stored_spike_capabilities=capabilities,
        )
        assert (normalized in available) == (len(methods) == 2)
        assert (onsets in available) == (len(methods) == 2)
    assert normalized not in names(has_extraction=True)
    assert onsets not in names(has_analysis=True)
    capabilities = {key: value - {"threshold"} for key, value in capabilities.items()}
    available = names(
        has_extraction=True, has_analysis=True, stored_spike_capabilities=capabilities
    )
    assert onsets not in available
    assert (normalized in available) == (len(methods) == 2)


@pytest.mark.parametrize("onsets", [False, True])
def test_comparison_rendering_uses_common_frames_and_separate_method_rows(
    spike_database: tuple, plot_widget: _SingleWellGraphWidget, onsets: bool
) -> None:
    engine, _, run_id, methods = spike_database
    product = "single_well.spike_comparison_" + ("onsets" if onsets else "normalized")
    plot_single_well_data(plot_widget, engine, "A1_0", product, run_id)
    items = [
        item
        for item in plot_widget.plot_item.items
        if item.property("roi_label") is not None
    ]
    if len(methods) == 1:
        assert items == []
        assert "No aligned dual-method" in plot_widget.plot_item.titleLabel.text
        return
    assert len(items) == 6
    for item in items:
        roi, method = int(item.property("roi_label")), item.property("spike_method")
        x, y = item.getData()
        if onsets:
            active = roi in (1, 3) if method == "oasis" else roi in (2, 3)
            assert x.tolist() == ([30, 70] if active else [])
            assert set(y) <= {(roi - 1) * 2 + int(method == "cascade")}
        else:
            assert x.tolist() == list(range(20, 85) if roi == 3 else range(10, 90))
            assert np.isfinite(y).all()
            assert np.max(y - (roi - 1) * 1.2) <= 1 + 1e-12
        assert "common retained interval" in item.toolTip()
    assert "Common Valid Interval" in plot_widget.plot_item.titleLabel.text
    plot_single_well_data(plot_widget, engine, "A1_0", product, run_id + 100)
    assert not plot_widget.plot_item.listDataItems()
    assert not plot_widget.legend.isVisible()


def test_comparison_clears_previous_heatmap_state(
    spike_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, methods = spike_database
    plot_single_well_data(
        plot_widget,
        engine,
        "A1_0",
        "single_well.sorted_inferred_spikes_thresholded_max_lag_values",
        run_id,
        spike_method=methods[0],
    )
    assert plot_widget.colorbar is not None
    assert callable(plot_widget.plot_item.property("sorted_lag_click_handler"))
    plot_single_well_data(
        plot_widget, engine, "A1_0", "single_well.spike_comparison_normalized", run_id
    )
    assert plot_widget.colorbar is None
    assert plot_widget.plot_item.property("sorted_lag_click_handler") is None
    assert not any(
        isinstance(item, pg.ImageItem) for item in plot_widget.plot_item.items
    )


@pytest.mark.parametrize("onsets", [False, True])
def test_comparison_export_preserves_intersection_units_and_coordinates(
    spike_database: tuple, tmp_path: Path, onsets: bool
) -> None:
    engine, _, run_id, methods = spike_database
    target = tmp_path / "comparison"
    if len(methods) == 1:
        with pytest.raises(ValueError, match="No aligned dual-method"):
            export_spike_comparison_to_csv(
                engine, target, run_id=run_id, include_onsets=onsets
            )
        assert not target.exists()
        return
    assert (
        export_spike_comparison_to_csv(
            engine, target, run_id=run_id, include_onsets=onsets
        )
        == 6
    )
    table = pd.read_csv(target / "spike_comparison.csv")
    assert set(table.method) == {"oasis", "cascade"}
    for (_, label, method), row in table.groupby(["fov_name", "roi_label", "method"]):
        frames = list(range(20, 85) if label == 3 else range(10, 90))
        assert row.retained_frame_0based.tolist() == frames
        assert row.source_frame_0based.tolist() == [frame + 10 for frame in frames]
        assert row.retained_time_ms.tolist() == [frame * 100 for frame in frames]
        assert row.source_time_ms.tolist() == [frame * 100 + 1000 for frame in frames]
        assert row.units.eq("a.u." if method == "oasis" else "spikes/frame").all()
        assert row.amplitude.max() <= 0.5
        assert row.normalized_amplitude.max() <= 1
        if onsets:
            active = label in (1, 3) if method == "oasis" else label in (2, 3)
            assert row.loc[
                row.threshold_onset == 1, "retained_frame_0based"
            ].tolist() == ([30, 70] if active else [])
        else:
            assert row.threshold_onset.isna().all()
    metadata = json.loads((target / "spike_comparison.metadata.json").read_text())
    assert len(metadata["rois"]) == 6
    assert "independent peak" in metadata["normalization"]
    assert metadata["rois"][2]["methods"]["cascade"]["spike_active"] is True


def test_comparison_export_rejects_bad_data_before_replacing_files(
    spike_database: tuple, tmp_path: Path
) -> None:
    engine, _, run_id, methods = spike_database
    if len(methods) < 2:
        return
    target = tmp_path / "comparison"
    export_spike_comparison_to_csv(engine, target, run_id=run_id)
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    with engine.begin() as connection:
        connection.execute(
            update(SpikeInferenceRun)
            .where(SpikeInferenceRun.method == "cascade")
            .values(units="invalid")
        )
    with pytest.raises(ValueError, match="inference units"):
        export_spike_comparison_to_csv(engine, target, run_id=run_id)
    assert before == {p.name: p.read_bytes() for p in target.iterdir()}


@pytest.mark.parametrize("require_threshold", [False, True])
def test_reader_rejects_unresolved_analysis_sources(
    spike_database: tuple, require_threshold: bool
) -> None:
    engine, _, run_id, methods = spike_database
    if len(methods) < 2:
        return
    with engine.begin() as connection:
        connection.execute(
            update(SpikeAnalysis)
            .where(SpikeAnalysis.method == "oasis")
            .values(provenance_source="legacy_unresolved")
        )
    with pytest.raises(ValueError, match="resolved analysis sources"):
        get_spike_comparisons(
            engine, "A1_0", run_id, require_threshold=require_threshold
        )


def test_reader_rejects_misaligned_roi_windows(spike_database: tuple) -> None:
    engine, _, run_id, methods = spike_database
    if len(methods) < 2:
        return
    with Session(engine) as session:
        trace = session.exec(
            select(Traces).join(DataAnalysis, DataAnalysis.roi_id == Traces.roi_id)
        ).first()
        # Alter this trace's stored time axis while other ROI rows remain aligned.
        trace.x_axis = [value + 100 for value in trace.x_axis]
        session.commit()
    with pytest.raises(ValueError, match="retained time axis"):
        get_spike_comparisons(engine, "A1_0", run_id)


def test_comparison_export_rebases_historical_timestamp_origin(
    spike_database: tuple, tmp_path: Path
) -> None:
    engine, _, run_id, methods = spike_database
    if len(methods) < 2:
        return
    with Session(engine) as session:
        for trace in session.exec(select(Traces)):
            trace.x_axis = [value + 500 for value in trace.x_axis]
        session.commit()
    target = tmp_path / "comparison"
    export_spike_comparison_to_csv(engine, target, run_id=run_id)
    table = pd.read_csv(target / "spike_comparison.csv")
    np.testing.assert_allclose(
        table.retained_time_ms, table.retained_frame_0based * 100
    )
    np.testing.assert_allclose(table.source_time_ms, table.retained_time_ms + 1000)
