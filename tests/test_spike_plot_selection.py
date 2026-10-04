"""Plot/compute selection preserves stored method meaning and valid samples."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from pytestqt.qtbot import QtBot
from sqlmodel import Session, select

from cali.gui._pygraph_plot_widgets import _SingleWellGraphWidget
from cali.plot import (
    ANALYSIS_PRODUCTS,
    AnalysisGroup,
    get_available_plots,
    get_stored_spike_capabilities,
    plot_single_well_data,
)
from cali.plot._multi_wells_plots._util import (
    _query_roi_parameter_by_condition,
    make_parameter_compute_fn,
)
from cali.plot._single_wells_plots.calcium_traces._plot_calcium_traces_data import (
    _get_traces_and_metadata,
)
from cali.plot._single_wells_plots.metrics._plot_inferred_spikes_frequency_data import (
    _plot_inferred_spikes_frequency_data,
)
from cali.plot._single_wells_plots.raster._plot_inferred_spike_raster_plots import (
    _generate_spike_intensity_heatmap,
    _generate_spike_intensity_heatmap_thresholded,
    _generate_spike_raster_plot,
)
from cali.plot._single_wells_plots.spikes._plot_inferred_spikes import (
    _plot_inferred_spikes,
)
from cali.plot._spike_data import roi_is_active, spike_plot_data
from cali.sqlmodel import FOV, CaliResult, DataAnalysis, Plate, Traces, Well
from cali.util._database_to_csv import export_multi_well_to_csv

from .test_cascade_fov_analysis import _fov
from .test_spike_export import spike_database as spike_database


@pytest.fixture
def plot_widget(qtbot: QtBot) -> _SingleWellGraphWidget:
    widget = _SingleWellGraphWidget(parent=None)
    qtbot.addWidget(widget)
    return widget


def _flatten(data: dict) -> list[float]:
    return [
        value
        for wells in data.values()
        for fovs in wells.values()
        for values in fovs.values()
        for value in values
    ]


def test_registry_ids_are_unique_and_display_name_is_an_alias(
    spike_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, methods = spike_database
    ids = [product.product_id for product in ANALYSIS_PRODUCTS]
    assert len(ids) == len(set(ids))
    for method in methods:
        for text in ("single_well.inferred_spikes", "Inferred Spikes"):
            plot_single_well_data(
                plot_widget, engine, "A1_0", text, run_id, spike_method=method
            )
            assert method.upper() in plot_widget.plot_item.titleLabel.text
            assert len(plot_widget.plot_item.listDataItems()) == 3


def test_availability_uses_stored_methods_and_metrics(spike_database: tuple) -> None:
    engine, _, run_id, methods = spike_database
    capabilities = get_stored_spike_capabilities(engine, run_id, fov_name="A1_0")
    assert set(capabilities) == set(methods)
    assert get_stored_spike_capabilities(engine, run_id, fov_name="absent") == {}
    for method in methods:
        assert "spike_trace" in capabilities[method]
        assert "threshold" in capabilities[method]
        plots = get_available_plots(
            AnalysisGroup.SINGLE_WELL,
            has_analysis=True,
            has_extraction=True,
            stored_spike_methods=methods,
            spike_method=method,
            available_metrics=capabilities[method],
        )
        names = [name for category in plots.values() for name in category]
        assert "Inferred Spikes" in names
        assert "Calcium Peaks Raster" in names
        if method == "cascade":
            assert "CASCADE Expected Spike Rate" in names
            assert ("CASCADE Threshold Excursion Rate" in names) == (
                "suprathreshold_excursion_rate_hz" in capabilities[method]
            )
            assert "Inferred Spikes Thresholded Frequency" not in names
            assert "Inferred Spikes Thresholded Burst Activity Analysis" in names
        else:
            assert "Inferred Spikes Thresholded Frequency" in names
            assert "CASCADE Expected Spike Rate" not in names
    plots = get_available_plots(
        AnalysisGroup.SINGLE_WELL,
        has_analysis=True,
        stored_spike_methods=(),
        available_metrics=set(),
    )
    assert all("Spike" not in name for category in plots.values() for name in category)
    plots = get_available_plots(
        AnalysisGroup.SINGLE_WELL,
        has_analysis=True,
        stored_spike_methods=("cascade",),
        spike_method="cascade",
        available_metrics={"spike_trace", "expected_spike_rate_hz"},
    )
    names = [name for category in plots.values() for name in category]
    assert "CASCADE Expected Spike Rate" in names
    assert "CASCADE Threshold Excursion Rate" not in names
    assert "Inferred Spikes Thresholded" not in names


def test_active_trace_membership_and_padding(
    spike_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, methods = spike_database
    with Session(engine) as session:
        for analysis in session.exec(select(DataAnalysis)):
            analysis.total_recording_time_sec = 9.9
        session.commit()
    for method in methods:
        _plot_inferred_spikes(
            plot_widget,
            engine,
            "A1_0",
            run_id=run_id,
            spike_method=method,
            active_only=True,
            normalize=True,
        )
        if method == "cascade":
            assert plot_widget.plot_item.getAxis("bottom").labelText == "Frames"
        curves = plot_widget.plot_item.listDataItems()
        assert {int(curve.property("roi_label")) for curve in curves} == (
            {1, 3} if method == "oasis" else {2, 3}
        )
        for curve in curves:
            x, y = curve.getData()
            assert x.tolist() == list(range(100))
            if method == "cascade":
                start, stop = (
                    (20, 85) if curve.property("roi_label") == "3" else (10, 90)
                )
                assert np.isnan(y[:start]).all()
                assert np.isnan(y[stop:]).all()
                assert np.isfinite(y[start:stop]).all()
                # Padding is 100, valid peaks .5: normalization must still reach 1.
                assert np.max(y[start:stop]) - np.min(y[start:stop]) == 1


def test_raster_uses_valid_events_and_censors_left_edge(
    spike_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, methods = spike_database
    for method in methods:
        _generate_spike_raster_plot(
            plot_widget, engine, "A1_0", run_id=run_id, spike_method=method, edges=True
        )
        scatters = [
            item
            for item in plot_widget.plot_item.items
            if isinstance(item, pg.ScatterPlotItem)
        ]
        assert {int(item.property("roi_label")) for item in scatters} == (
            {1, 3} if method == "oasis" else {2, 3}
        )
        for item in scatters:
            x, _ = item.getData()
            assert x.tolist() == ([0, 30, 70] if method == "oasis" else [30, 70])


@pytest.mark.parametrize("thresholded", [False, True])
def test_heatmap_padding_and_method_units(
    spike_database: tuple, plot_widget: _SingleWellGraphWidget, thresholded: bool
) -> None:
    engine, _, run_id, methods = spike_database
    render = (
        _generate_spike_intensity_heatmap_thresholded
        if thresholded
        else _generate_spike_intensity_heatmap
    )
    for method in methods:
        render(plot_widget, engine, "A1_0", run_id=run_id, spike_method=method)
        image = next(
            item
            for item in plot_widget.plot_item.items
            if isinstance(item, pg.ImageItem)
        )
        assert image.image.shape == ((2 if thresholded else 3), 100)
        if method == "cascade":
            assert np.isnan(image.image[:, :10]).all()
            assert np.isnan(image.image[:, 90:]).all()
            assert np.nanmax(image.image) == 0.5
        assert method.upper() in plot_widget.plot_item.titleLabel.text


def test_frequency_scalar_semantics_and_active_flags(
    spike_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, methods = spike_database
    for method in methods:
        _plot_inferred_spikes_frequency_data(
            plot_widget, engine, "A1_0", run_id=run_id, spike_method=method
        )
        scatter = next(
            item
            for item in plot_widget.plot_item.items
            if isinstance(item, pg.ScatterPlotItem)
        )
        assert {point.data() for point in scatter.points()} == (
            {"1", "3"} if method == "oasis" else {"2", "3"}
        )
        if method == "cascade":
            _, values = scatter.getData()
            np.testing.assert_allclose(values, [1.5 / 80 * 10, 1.5 / 65 * 10])
            assert "Expected Spike Rate" in plot_widget.plot_item.titleLabel.text
        assert "Hz" in plot_widget.plot_item.getAxis("left").labelText


def test_no_headless_cross_method_metric_substitution(
    spike_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, _ = spike_database
    with pytest.raises(ValueError, match="does not support"):
        plot_single_well_data(
            plot_widget,
            engine,
            "A1_0",
            "Inferred Spikes Thresholded Frequency",
            run_id,
            spike_method="cascade",
        )
    with pytest.raises(ValueError, match="unavailable for cascade"):
        _plot_inferred_spikes_frequency_data(
            plot_widget,
            engine,
            "A1_0",
            run_id=run_id,
            spike_method="cascade",
            metric="suprathreshold_sample_rate_hz",
        )
    with pytest.raises(ValueError, match="unavailable for oasis"):
        make_parameter_compute_fn("expected_spike_count", "spikes", "Count")
    with pytest.raises(ValueError, match="separate amplitude units"):
        _plot_inferred_spikes(
            plot_widget,
            engine,
            "A1_0",
            run_id=run_id,
            spike_method="cascade",
            den_dff=True,
        )
    product = next(
        p
        for p in ANALYSIS_PRODUCTS
        if p.product_id == "multi_well.cascade_expected_spike_count"
    )
    with pytest.raises(ValueError, match="does not support"):
        product.compute_data(engine, run_id, spike_method="oasis")


def test_scalar_aggregation_uses_method_flags_and_qualified_export(
    spike_database: tuple, tmp_path: Path
) -> None:
    engine, path, run_id, methods = spike_database
    with Session(engine) as session:
        well = Well(
            name="A1",
            row=0,
            column=0,
            plate=Plate(
                name="test plate",
                experiment_id=session.get(CaliResult, run_id).experiment,
            ),
        )
        for fov in session.exec(select(FOV)):
            fov.well = well
        session.add(well)
        session.commit()
    for method in methods:
        parameter = (
            "expected_spike_count"
            if method == "cascade"
            else "inferred_spikes_frequency"
        )
        data = _query_roi_parameter_by_condition(
            engine, parameter, run_id, spike_method=method
        )
        values = _flatten(data)
        assert len(values) == 4
        if method == "cascade":
            assert values == [1.5] * 4
    export_multi_well_to_csv(engine, run_id, path)
    target = path.with_name(path.stem + "_exports") / f"run_{run_id}" / "multi_well"
    csv_path = target / "cascade_expected_spike_count_bar_plot.csv"
    if "cascade" in methods:
        assert pd.read_csv(csv_path)["mean"].tolist() == [1.5]
        import json

        metadata = json.loads(csv_path.with_suffix(".metadata.json").read_text())
        assert metadata["method"] == "cascade"
        assert metadata["metric_units"] == "spikes"
    else:
        assert not csv_path.exists()


def test_calcium_activity_remains_independent(spike_database: tuple) -> None:
    engine, _, run_id, _ = spike_database
    result = _get_traces_and_metadata(
        engine,
        "A1_0",
        run_id=run_id,
        rois=None,
        raw=False,
        dff=False,
        dec=True,
        active_only=True,
    )
    assert result is not None
    assert result[1] == ["1", "2"]


def test_raw_trace_needs_no_analysis(
    spike_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, methods = spike_database
    with Session(engine) as session:
        extracted = CaliResult(experiment=session.get(CaliResult, run_id).experiment)
        session.add(extracted)
        session.flush()
        for trace in session.exec(
            select(Traces).where(Traces.analysis_result_id == run_id)
        ):
            trace.analysis_result = extracted
        session.commit()
        extracted_id = extracted.id
    for method in methods:
        capabilities = get_stored_spike_capabilities(engine, extracted_id)
        assert capabilities[method] == {"spike_trace"}
        _plot_inferred_spikes(
            plot_widget,
            engine,
            "A1_0",
            run_id=extracted_id,
            raw=True,
            spike_method=method,
        )
        assert len(plot_widget.plot_item.listDataItems()) == 3


def test_selection_rejects_a_metric_bound_to_another_trace() -> None:
    fov = _fov()
    roi = fov.rois[0]
    trace, analysis = roi._new_traces[0], roi._new_data_analysis[0]
    spike = trace.get_spike_trace("oasis")
    metric = analysis.get_spike_analysis("oasis")
    spike.id, metric.spike_trace_id = 10, 11
    with pytest.raises(ValueError, match="analyzed source trace"):
        spike_plot_data(trace, analysis, "oasis", require_threshold=True)
    metric.spike_trace_id = 10
    metric.spike_active = False
    roi.active = True
    assert not roi_is_active(roi, analysis, "oasis")
    metric.spike_active = None
    assert roi_is_active(roi, analysis, "oasis")
    metric.threshold = 0
    assert spike_plot_data(trace, analysis, "oasis", require_threshold=True) is not None
