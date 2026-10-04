"""Population consumers preserve method membership, timing, and missing values."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from pytestqt.qtbot import QtBot
from sqlalchemy import update
from sqlmodel import Session, select

from cali.gui._pygraph_plot_widgets import (
    _MultilWellGraphWidget,
    _SingleWellGraphWidget,
)
from cali.plot import ANALYSIS_PRODUCTS, plot_multi_well_data, plot_single_well_data
from cali.plot._multi_wells_plots._inferred_spikes import (
    _query_burst_metrics_by_condition,
    _query_fov_scalar_by_condition,
    compute_burst_rate_data,
)
from cali.plot._single_wells_plots.correlation._plot_inferred_spike_synchrony import (
    _get_spike_synchrony_matrix_from_db,
)
from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_correlation import (
    _get_ccg_zscore_matrix_from_db,
    _get_spike_max_lag_correlation_matrix_from_db,
)
from cali.plot._single_wells_plots.correlation._plot_spike_max_lag_values import (
    _get_spike_max_lag_values_matrix_from_db,
)
from cali.sqlmodel import FOV, CaliResult, FOVAnalysis, Plate, SpikeFOVAnalysis, Well
from cali.util._database_to_csv import export_multi_well_to_csv

from .test_cascade_fov_analysis import _settings
from .test_spike_export import spike_database as spike_database


@pytest.fixture
def population_database(spike_database: tuple) -> tuple:
    engine, path, run_id, methods = spike_database
    with Session(engine) as session:
        owner = session.get(CaliResult, run_id)
        settings = _settings(methods)
        session.add(settings)
        session.flush()
        owner.analysis_settings_id = settings.id
        well = Well(
            name="A1",
            row=0,
            column=0,
            plate=Plate(name="population plate", experiment_id=owner.experiment),
        )
        for fov in session.exec(select(FOV)):
            fov.well = well
        session.add(well)
        for child in session.exec(select(SpikeFOVAnalysis)):
            child.spike_burst_avg_interval = 1.23
            child.global_spike_jitter_synchrony = (
                0.12 if child.method == "oasis" else 0.78
            )
        session.commit()
    return engine, path, run_id, methods


@pytest.fixture
def plot_widget(qtbot: QtBot) -> _SingleWellGraphWidget:
    widget = _SingleWellGraphWidget(parent=None)
    qtbot.addWidget(widget)
    return widget


def _regions(widget: _SingleWellGraphWidget) -> list[tuple[float, float]]:
    return [
        item.getRegion()
        for item in widget.plot_item.items
        if isinstance(item, pg.LinearRegionItem)
    ]


def test_population_uses_stored_retained_interval_and_method_settings(
    population_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, methods = population_database
    for method in methods:
        plot_single_well_data(
            plot_widget,
            engine,
            "A1_0",
            "single_well.inferred_spikes_thresholded_burst_activity_analysis",
            run_id,
            rois=[1],
            spike_method=method,
        )
        curves = plot_widget.plot_item.listDataItems()
        frames = list(range(100)) if method == "oasis" else list(range(20, 85))
        assert len(curves) == 2
        assert all(curve.getData()[0].tolist() == frames for curve in curves)
        assert _regions(plot_widget) == [(30, 40)]
        assert plot_widget.plot_item.getAxis("bottom").labelText == "Retained Frames"
        threshold = next(
            item.value()
            for item in plot_widget.plot_item.items
            if isinstance(item, pg.InfiniteLine)
        )
        assert threshold == (0.25 if method == "oasis" else 0.5)
        title = plot_widget.plot_item.titleLabel.text
        assert method.upper() in title
        assert ("15.00/min" if method == "oasis" else "9.23/min") in title
        assert ("0.40 s" if method == "oasis" else "1.00 s") in title
        assert "Mean Interval: 1.23 s" in title


@pytest.mark.parametrize(
    "product_id",
    [
        "single_well.inferred_spikes_thresholded_normalized_with_network_bursts",
        "single_well.inferred_spike_raster_with_network_bursts",
    ],
)
def test_overlays_select_the_method_population_and_half_open_bounds(
    population_database: tuple, plot_widget: _SingleWellGraphWidget, product_id: str
) -> None:
    engine, _, run_id, methods = population_database
    for method in methods:
        plot_single_well_data(
            plot_widget, engine, "A1_0", product_id, run_id, spike_method=method
        )
        assert _regions(plot_widget) == [(30, 40)]
        items = [
            item
            for item in plot_widget.plot_item.items
            if item.property("roi_label") is not None
        ]
        assert {int(item.property("roi_label")) for item in items} == (
            {1, 3} if method == "oasis" else {2, 3}
        )
        assert method.upper() in plot_widget.plot_item.titleLabel.text
        if method == "cascade" and "raster" in product_id:
            assert "Threshold Excursion Starts" in plot_widget.plot_item.titleLabel.text
            assert all(item.getData()[0].tolist() == [30, 70] for item in items)


@pytest.mark.parametrize("rising_edges", [False, True])
def test_matrix_readers_preserve_selected_label_order_and_parameters(
    population_database: tuple, rising_edges: bool
) -> None:
    engine, _, run_id, methods = population_database
    suffix = "_rising_edges" if rising_edges else ""
    with Session(engine) as session:
        for child in session.exec(select(SpikeFOVAnalysis)):
            child.active_roi_labels = [3, 1] if child.method == "oasis" else [3, 2]
            for field in (
                "spike_max_lag_correlation_matrix",
                "spike_ccg_zscore_matrix",
                "spike_max_lag_values_matrix",
                "spike_jitter_synchrony_matrix",
            ):
                setattr(child, field + suffix, [[1, 0.1], [0.8, 1]])
        session.commit()
    for method in methods:
        labels = [3, 1] if method == "oasis" else [3, 2]
        for getter in (
            _get_spike_synchrony_matrix_from_db,
            _get_ccg_zscore_matrix_from_db,
            _get_spike_max_lag_correlation_matrix_from_db,
            _get_spike_max_lag_values_matrix_from_db,
        ):
            result = getter(
                engine, "A1_0", run_id, rising_edges=rising_edges, spike_method=method
            )
            np.testing.assert_allclose(result[0], [[1, 0.1], [0.8, 1]])
            assert result[1] == labels
        sync = _get_spike_synchrony_matrix_from_db(
            engine, "A1_0", run_id, spike_method=method
        )
        assert sync[2:] == ((0.12, 40) if method == "oasis" else (0.78, 200))
        lag = _get_spike_max_lag_values_matrix_from_db(
            engine, "A1_0", run_id, spike_method=method
        )
        assert lag[2] == (5 if method == "oasis" else 3)


def test_missing_method_and_malformed_matrix_never_substitute(
    population_database: tuple,
) -> None:
    engine, _, run_id, methods = population_database
    absent = "cascade" if methods == ("oasis",) else "oasis"
    if len(methods) == 1:
        assert (
            _get_spike_synchrony_matrix_from_db(
                engine, "A1_0", run_id, spike_method=absent
            )
            == (None,) * 4
        )
    with Session(engine) as session:
        child = session.exec(select(SpikeFOVAnalysis)).first()
        method = child.method
        child.spike_jitter_synchrony_matrix = [[1]]
        session.commit()
    with pytest.raises(ValueError, match="dimensions must match"):
        _get_spike_synchrony_matrix_from_db(engine, "A1_0", run_id, spike_method=method)


def test_historical_coordinates_are_explicit_and_cascade_rate_is_unknown(
    population_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, methods = population_database
    with Session(engine) as session:
        for child in session.exec(select(SpikeFOVAnalysis)):
            child.valid_start = child.valid_stop = child.frame_rate_hz = None
        session.commit()
    for method in methods:
        plot_single_well_data(
            plot_widget,
            engine,
            "A1_0",
            "single_well.inferred_spikes_thresholded_burst_activity_analysis",
            run_id,
            spike_method=method,
        )
        assert (
            "Retained Origin Unknown"
            in plot_widget.plot_item.getAxis("bottom").labelText
        )
        assert _regions(plot_widget) == []
        rate = compute_burst_rate_data(engine, run_id, spike_method=method)
        if method == "cascade":
            assert rate is None
            assert "Rate: unknown" in plot_widget.plot_item.titleLabel.text
        else:
            assert rate[0]["means"] == [15]


def test_aggregation_uses_method_rate_and_preserves_zero_fovs(
    population_database: tuple,
) -> None:
    engine, _, run_id, methods = population_database
    with Session(engine) as session:
        for child in session.exec(
            select(SpikeFOVAnalysis)
            .join(FOVAnalysis)
            .join(FOV)
            .where(FOV.name == "A1_1")
        ):
            child.spike_burst_count = 0
            child.spike_burst_avg_duration = child.spike_burst_avg_interval = None
        session.commit()
    for method in methods:
        query = _query_burst_metrics_by_condition(engine, run_id, spike_method=method)
        fovs = next(iter(next(iter(query.values())).values()))
        assert fovs["A1_1"]["count"] == 0
        assert fovs["A1_1"]["rate_per_min"] == 0
        assert "avg_duration_sec" not in fovs["A1_1"]
        assert fovs["A1_0"]["rate_per_min"] == pytest.approx(
            15 if method == "oasis" else 60 / 6.5
        )
        product = next(
            p
            for p in ANALYSIS_PRODUCTS
            if p.product_id == "multi_well.inferred_spikes_burst_count_bar_plot"
        )
        assert product.compute_data(engine, run_id, spike_method=method)[0][
            "means"
        ] == [0.5]
        scalar = _query_fov_scalar_by_condition(
            engine, run_id, "global_spike_jitter_synchrony", spike_method=method
        )
        values = [
            value
            for wells in scalar.values()
            for fovs in wells.values()
            for value in fovs.values()
        ]
        assert values == [(0.12 if method == "oasis" else 0.78, 1)] * 2


def test_multiwell_dispatch_and_csv_keep_methods_separate(
    population_database: tuple, qtbot: QtBot
) -> None:
    engine, path, run_id, methods = population_database
    widget = _MultilWellGraphWidget(parent=None)
    qtbot.addWidget(widget)
    for method in methods:
        plot_multi_well_data(
            widget,
            "multi_well.spike_jitter_synchrony_bar_plot",
            engine,
            run_id,
            spike_method=method,
        )
        assert method.upper() in widget.plot_item.titleLabel.text
    export_multi_well_to_csv(engine, run_id, path)
    target = path.with_name(path.stem + "_exports") / f"run_{run_id}" / "multi_well"
    for method in methods:
        prefix = method + "_" if method == "cascade" or len(methods) > 1 else ""
        csv_path = target / (prefix + "inferred_spikes_burst_rate_bar_plot.csv")
        assert pd.read_csv(csv_path)["mean"].tolist() == pytest.approx(
            [15 if method == "oasis" else 60 / 6.5]
        )
        metadata = json.loads(csv_path.with_suffix(".metadata.json").read_text())
        assert metadata["method"] == method
        assert metadata["metric_units"] == "bursts/min"
        assert (
            metadata["product_id"] == "multi_well.inferred_spikes_burst_rate_bar_plot"
        )
    if len(methods) > 1:
        assert not (target / "inferred_spikes_burst_rate_bar_plot.csv").exists()


def test_export_rejects_unavailable_methods_before_writing(
    population_database: tuple, tmp_path: Path
) -> None:
    engine, _, run_id, methods = population_database
    if len(methods) > 1:
        return
    other = "cascade" if methods == ("oasis",) else "oasis"
    path = tmp_path / "invalid.cali"
    with pytest.raises(ValueError, match="No stored spike analysis"):
        export_multi_well_to_csv(engine, run_id, path, spike_methods=(other,))
    assert not path.with_name("invalid_exports").exists()


@pytest.mark.parametrize(
    "product_id,field",
    [
        (
            "inferred_spikes_thresholded_max_lag_correlation",
            "spike_max_lag_correlation_matrix",
        ),
        ("inferred_spikes_thresholded_max_lag_values", "spike_max_lag_values_matrix"),
        ("inferred_spikes_thresholded_ccg_z_score", "spike_ccg_zscore_matrix"),
        (
            "inferred_spikes_thresholded_global_synchrony",
            "spike_jitter_synchrony_matrix",
        ),
    ],
)
@pytest.mark.parametrize("rising_edges", [False, True])
def test_matrix_dispatch_uses_selected_values_and_clears_missing_method(
    population_database: tuple,
    plot_widget: _SingleWellGraphWidget,
    product_id: str,
    field: str,
    rising_edges: bool,
) -> None:
    engine, _, run_id, methods = population_database
    suffix = "_rising_edges" if rising_edges else ""
    with Session(engine) as session:
        for child in session.exec(select(SpikeFOVAnalysis)):
            value = 0.2 if child.method == "oasis" else 0.7
            setattr(child, field + suffix, [[1, value], [value, 1]])
        session.commit()
    for method in methods:
        plot_single_well_data(
            plot_widget,
            engine,
            "A1_0",
            "single_well." + product_id + suffix,
            run_id,
            spike_method=method,
        )
        image = next(
            item
            for item in plot_widget.plot_item.items
            if isinstance(item, pg.ImageItem)
        )
        value = 0.2 if method == "oasis" else 0.7
        np.testing.assert_allclose(image.image, [[1, value], [value, 1]])
        assert method.upper() in plot_widget.plot_item.titleLabel.text
        if method == "cascade" and rising_edges:
            assert "Threshold Excursion Starts" in plot_widget.plot_item.titleLabel.text
    if len(methods) == 1:
        absent = "cascade" if methods == ("oasis",) else "oasis"
        plot_single_well_data(
            plot_widget,
            engine,
            "A1_0",
            "single_well." + product_id + suffix,
            run_id,
            spike_method=absent,
        )
        assert not any(
            isinstance(item, pg.ImageItem) for item in plot_widget.plot_item.items
        )
        assert plot_widget.colorbar is None
        assert absent.upper() in plot_widget.plot_item.titleLabel.text


@pytest.mark.parametrize("starts,stops", [([30], []), ([30], [30]), ([30], [101])])
def test_population_rejects_invalid_burst_bounds(
    population_database: tuple,
    plot_widget: _SingleWellGraphWidget,
    starts: list[int],
    stops: list[int],
) -> None:
    engine, _, run_id, _methods = population_database
    with Session(engine) as session:
        child = session.exec(select(SpikeFOVAnalysis)).first()
        method = child.method
        # Simulate a malformed on-disk row through SQL rather than the validating ORM.
        session.execute(
            update(SpikeFOVAnalysis)
            .where(SpikeFOVAnalysis.id == child.id)
            .values(spike_burst_starts=starts, spike_burst_ends=stops)
            .execution_options(synchronize_session=False)
        )
        session.commit()
    with pytest.raises(ValueError, match=r"[Ss]pike burst"):
        plot_single_well_data(
            plot_widget,
            engine,
            "A1_0",
            "single_well.inferred_spikes_thresholded_burst_activity_analysis",
            run_id,
            spike_method=method,
        )


def test_invalid_selected_population_does_not_replace_exports(
    population_database: tuple,
) -> None:
    engine, path, run_id, _methods = population_database
    export_multi_well_to_csv(engine, run_id, path)
    target = path.with_name(path.stem + "_exports") / f"run_{run_id}" / "multi_well"
    before = {file.name: file.read_bytes() for file in target.iterdir()}
    with Session(engine) as session:
        child = session.exec(select(SpikeFOVAnalysis)).first()
        session.execute(
            update(SpikeFOVAnalysis)
            .where(SpikeFOVAnalysis.id == child.id)
            .values(units="invalid")
            .execution_options(synchronize_session=False)
        )
        session.commit()
    with pytest.raises(ValueError, match="method and units"):
        export_multi_well_to_csv(engine, run_id, path)
    assert {file.name: file.read_bytes() for file in target.iterdir()} == before
