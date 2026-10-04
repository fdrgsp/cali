"""Evoked sorting, onsets, and PCA retain the selected method's scientific meaning."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from pytestqt.qtbot import QtBot
from sqlalchemy import update
from sqlmodel import Session, select

from cali._constants import EVOKED
from cali.gui._pygraph_plot_widgets import (
    _MultilWellGraphWidget,
    _PCAFeaturesDialog,
    _SingleWellGraphWidget,
)
from cali.plot import (
    ANALYSIS_PRODUCTS,
    AnalysisGroup,
    get_available_plots,
    get_stored_spike_capabilities,
    plot_single_well_data,
)
from cali.plot._multi_wells_plots._dimensionality_reduction import (
    CASCADE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    build_fov_feature_matrix,
    compute_pca,
    plot_pca_loadings,
    plot_pca_scatter,
    plot_pca_scree,
)
from cali.plot._multi_wells_plots._evoked_activity import (
    _query_evoked_amplitudes_by_condition,
)
from cali.plot._single_wells_plots.correlation._plot_evoked_correlation_synchrony import (  # noqa: E501
    _get_sorted_rois_by_stimulation,
)
from cali.plot._single_wells_plots.evoked._plot_evoked_experiment_data_plots import (
    _get_data_analysis_for_run,
    _plot_stim_and_non_stim_peaks_amplitude,
    _plot_stimulated_vs_non_stimulated_calcium_peaks_raster,
    _plot_stimulated_vs_non_stimulated_roi_traces,
)
from cali.sqlmodel import (
    FOV,
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    FOVAnalysis,
    SpikeAnalysis,
    SpikeFOVAnalysis,
    Traces,
)
from cali.util._database_to_csv import export_multi_well_pca_to_csv

from .test_spike_export import spike_database as spike_database
from .test_spike_fov_plot_selection import (
    plot_widget as plot_widget,
)
from .test_spike_fov_plot_selection import (
    population_database as population_database,
)


@pytest.fixture
def evoked_database(population_database: tuple) -> tuple:
    engine, path, run_id, methods = population_database
    with Session(engine) as session:
        owner = session.get(CaliResult, run_id)
        settings = session.get(AnalysisSettings, owner.analysis_settings_id)
        settings.experiment_type = EVOKED
        settings.led_pulse_on_frames = [41, 81]
        settings.led_pulse_duration = 100
        for trace in session.exec(select(Traces)):
            trace.extraction_frame_window.schema_version = 3
        for analysis in session.exec(select(DataAnalysis)):
            label = analysis.roi.label_value
            analysis.peaks_amplitudes_den_dff = [label * 10.0]
            analysis.den_dff_frequency = float(label)
            analysis.roi.cell_size = label * 100.0
            analysis.total_recording_time_sec = 9.9
            if analysis.roi.fov.name == "A1_1":
                for child in analysis.spike_analyses:
                    if child.expected_spike_count is not None:
                        child.expected_spike_count *= 2
                        child.expected_spike_rate_hz *= 2
                    if child.suprathreshold_sample_rate_hz is not None:
                        child.suprathreshold_sample_rate_hz *= 2
        session.commit()
    return engine, path, run_id, methods


@pytest.mark.parametrize(
    "product_id",
    [
        "single_well.stimulated_vs_non_stimulated_spike_traces",
        "single_well.stimulated_vs_non_stimulated_raster_inferred_spikes_thresholded_rising_edges",
    ],
)
def test_evoked_spikes_use_method_flags_valid_samples_and_source_pulses(
    evoked_database: tuple, plot_widget: _SingleWellGraphWidget, product_id: str
) -> None:
    engine, _, run_id, methods = evoked_database
    for method in methods:
        plot_single_well_data(
            plot_widget, engine, "A1_0", product_id, run_id, spike_method=method
        )
        items = [
            item
            for item in plot_widget.plot_item.items
            if item.property("roi_label") is not None
        ]
        assert {int(item.property("roi_label")) for item in items} == (
            {1, 3} if method == "oasis" else {2, 3}
        )
        by_label = {int(item.property("roi_label")): item for item in items}
        assert method.upper() in plot_widget.plot_item.titleLabel.text
        if "raster" in product_id:
            for item in items:
                assert item.getData()[0].tolist() == (
                    [0, 30, 70] if method == "oasis" else [30, 70]
                )
            assert set(by_label[3].getData()[1]) == {0}  # stimulated first
        else:
            assert by_label[3].property("roi_index") == 0
            if method == "cascade":
                y = by_label[3].getData()[1]
                assert np.isnan(y[:20]).all() and np.isnan(y[85:]).all()
                assert np.nanmax(y) == 0.5  # padding is 100, valid peaks .5
            assert (
                "a.u." if method == "oasis" else "spikes/frame"
            ) in plot_widget.plot_item.getAxis("left").labelText
        regions = [
            item.getRegion()
            for item in plot_widget.plot_item.items
            if isinstance(item, pg.LinearRegionItem)
        ]
        assert [region[0] for region in regions] == [30, 70]
        if method == "cascade":
            assert regions == [(30, 31), (70, 71)]
            assert plot_widget.plot_item.getAxis("bottom").labelText == "Frames"
    if len(methods) == 1:
        other = "cascade" if methods == ("oasis",) else "oasis"
        plot_single_well_data(
            plot_widget, engine, "A1_0", product_id, run_id, spike_method=other
        )
        assert not plot_widget.plot_item.listDataItems()
        assert other.upper() in plot_widget.plot_item.titleLabel.text


@pytest.mark.parametrize(
    "base,field",
    [
        (
            "sorted_inferred_spikes_thresholded_global_synchrony",
            "spike_jitter_synchrony_matrix",
        ),
        (
            "sorted_inferred_spikes_thresholded_max_lag_correlation",
            "spike_max_lag_correlation_matrix",
        ),
        (
            "sorted_inferred_spikes_thresholded_max_lag_values",
            "spike_max_lag_values_matrix",
        ),
        ("sorted_inferred_spikes_thresholded_ccg_z_score", "spike_ccg_zscore_matrix"),
    ],
)
@pytest.mark.parametrize("rising_edges", [False, True])
def test_sorted_matrices_use_method_membership_and_both_axes(
    evoked_database: tuple,
    plot_widget: _SingleWellGraphWidget,
    base: str,
    field: str,
    rising_edges: bool,
) -> None:
    engine, _, run_id, methods = evoked_database
    suffix = "_rising_edges" if rising_edges else ""
    with Session(engine) as session:
        for child in session.exec(select(SpikeFOVAnalysis)):
            setattr(child, field + suffix, [[1, 0.1], [0.8, 1]])
        session.commit()
    for method in methods:
        plot_single_well_data(
            plot_widget,
            engine,
            "A1_0",
            "single_well." + base + suffix,
            run_id,
            spike_method=method,
        )
        image = next(
            item
            for item in plot_widget.plot_item.items
            if isinstance(item, pg.ImageItem)
        )
        np.testing.assert_allclose(
            image.image, [[1, 0.8], [0.1, 1]]
        )  # both axes reorder to [3, 1/2]
        assert method.upper() in plot_widget.plot_item.titleLabel.text
        prefix = (
            "sorted_lag"
            if "lag_values" in base
            else "sorted_zscore"
            if "ccg_z_score" in base
            else "evoked"
        )
        handler = plot_widget.plot_item.property(f"{prefix}_click_handler")
        assert callable(handler)
        if method == "cascade" and rising_edges:
            assert "Threshold Excursion Starts" in plot_widget.plot_item.titleLabel.text
    if len(methods) == 1:
        other = "cascade" if methods == ("oasis",) else "oasis"
        plot_single_well_data(
            plot_widget,
            engine,
            "A1_0",
            "single_well." + base + suffix,
            run_id,
            spike_method=other,
        )
        assert plot_widget.colorbar is None
        for prefix in ("evoked", "sorted_lag", "sorted_zscore"):
            assert plot_widget.plot_item.property(f"{prefix}_click_handler") is None
            assert plot_widget.plot_item.property(f"{prefix}_hover_handler") is None


def test_calcium_sorted_population_does_not_use_union_flag(
    evoked_database: tuple,
) -> None:
    engine, _, run_id, _methods = evoked_database
    assert _get_sorted_rois_by_stimulation(engine, "A1_0", None, run_id=run_id) == (
        [1, 2],
        [],
        [1, 2],
    )


def test_evoked_calcium_consumers_use_calcium_flags(
    evoked_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, _ = evoked_database
    with Session(engine) as session:
        owner = session.get(CaliResult, run_id)
        settings = session.get(AnalysisSettings, owner.analysis_settings_id)
        settings.led_pulse_powers = [10, 10]
        for analysis in session.exec(select(DataAnalysis)):
            label = analysis.roi.label_value
            analysis.roi.active = label == 3  # opposite of calcium population
            analysis.peaks_den_dff = [30, 70]
            trace = analysis.roi.traces_history[0]
            trace.den_dff = [float(label)] * 100
        session.commit()
        roi = session.exec(select(DataAnalysis)).first().roi
        assert _get_data_analysis_for_run(roi, run_id + 100) is None
    for render in (
        _plot_stimulated_vs_non_stimulated_calcium_peaks_raster,
        _plot_stimulated_vs_non_stimulated_roi_traces,
    ):
        render(plot_widget, engine, "A1_0", run_id=run_id)
        labels = {
            int(item.property("roi_label"))
            for item in plot_widget.plot_item.items
            if item.property("roi_label") is not None
        }
        assert labels == {1, 2}
    _plot_stim_and_non_stim_peaks_amplitude(plot_widget, engine, "A1_0", run_id=run_id)
    assert "Stimulated: 0 ROIs, Non-Stimulated: 2 ROIs" in (
        plot_widget.plot_item.titleLabel.text
    )
    data = _query_evoked_amplitudes_by_condition(
        engine, stimulated=False, run_id=run_id
    )
    for wells in data.values():
        for fovs in wells.values():
            for powers in fovs.values():
                assert list(powers.values()) == [[1, 1, 2, 2]]
    assert data
    assert (
        _query_evoked_amplitudes_by_condition(engine, stimulated=True, run_id=run_id)
        == {}
    )


def test_switching_from_sorted_lags_to_evoked_clears_callbacks(
    evoked_database: tuple, plot_widget: _SingleWellGraphWidget
) -> None:
    engine, _, run_id, methods = evoked_database
    method = methods[0]
    plot_single_well_data(
        plot_widget,
        engine,
        "A1_0",
        "single_well.sorted_inferred_spikes_thresholded_max_lag_values",
        run_id,
        spike_method=method,
    )
    assert callable(plot_widget.plot_item.property("sorted_lag_click_handler"))
    plot_single_well_data(
        plot_widget,
        engine,
        "A1_0",
        "single_well.stimulated_vs_non_stimulated_spike_traces",
        run_id,
        spike_method=method,
    )
    assert plot_widget.plot_item.property("sorted_lag_click_handler") is None
    assert plot_widget.plot_item.property("sorted_lag_hover_handler") is None


def test_registry_exposes_only_stored_evoked_method_products(
    evoked_database: tuple,
) -> None:
    engine, _, run_id, methods = evoked_database
    for method in methods:
        capabilities = get_stored_spike_capabilities(engine, run_id)[method]
        plots = get_available_plots(
            AnalysisGroup.SINGLE_WELL,
            has_analysis=True,
            experiment_type=EVOKED,
            stored_spike_methods=methods,
            spike_method=method,
            available_metrics=capabilities,
        )
        names = [name for group in plots.values() for name in group]
        assert "Stimulated vs Non-Stimulated Spike Traces" in names
        assert "Sorted Inferred Spikes Thresholded Global Synchrony" in names
    assert all(
        product.supported_spike_methods == ("oasis", "cascade")
        for product in ANALYSIS_PRODUCTS
        if product.product_id.startswith("single_well.sorted_inferred_spikes")
    )


def test_pca_features_keep_independent_populations_and_metric_meanings(
    evoked_database: tuple,
) -> None:
    engine, _, run_id, methods = evoked_database
    for method in methods:
        frame = build_fov_feature_matrix(engine, run_id, spike_method=method)
        assert list(frame.columns) == [
            "fov_name",
            "condition",
            *(FEATURE_COLUMNS if method == "oasis" else CASCADE_FEATURE_COLUMNS),
        ]
        assert frame.attrs["spike_method"] == method
        assert frame["mean_amplitude"].tolist() == [15, 15]
        assert frame["mean_frequency"].tolist() == [1.5, 1.5]
        assert frame["mean_cell_size"].tolist() == [150, 150]
        np.testing.assert_allclose(frame["pct_active"], [200 / 3] * 2)
        np.testing.assert_allclose(frame["pct_spike_active"], [200 / 3] * 2)
        assert frame["burst_avg_duration_s"].tolist() == (
            [0.4] * 2 if method == "oasis" else [1] * 2
        )
        if method == "cascade":
            assert "mean_spike_freq" not in frame
            assert frame["mean_expected_spike_count"].tolist() == [1.5, 3]
            np.testing.assert_allclose(
                frame["mean_expected_spike_rate_hz"],
                [(1.5 / 80 * 10 + 1.5 / 65 * 10) / 2, (3 / 80 * 10 + 3 / 65 * 10) / 2],
            )
            assert frame["mean_suprathreshold_excursion_rate_hz"].isna().all()
        coords, _, used = compute_pca(frame)
        assert coords.shape[0] == 2
        assert set(used) <= set(frame.attrs["feature_columns"])
        split = build_fov_feature_matrix(
            engine, run_id, include_stim_status=True, spike_method=method
        )
        assert len(split) == 4
        assert split["burst_count"].isna().all()
        assert (
            split.loc[~split.fov_name.str.endswith("_non_stim"), "mean_amplitude"]
            .isna()
            .all()
        )
        assert split.loc[
            ~split.fov_name.str.endswith("_non_stim"), "pct_spike_active"
        ].tolist() == [100, 100]


def test_pca_rejects_incompatible_features_and_mixed_methods(
    evoked_database: tuple,
) -> None:
    engine, _, run_id, methods = evoked_database
    for method in methods:
        frame = build_fov_feature_matrix(engine, run_id, spike_method=method)
        other = "mean_expected_spike_count" if method == "oasis" else "mean_spike_freq"
        with pytest.raises(ValueError, match="selected spike method"):
            compute_pca(frame, feature_cols=[other])
        frame[other] = [1, 2]
        with pytest.raises(ValueError, match="cannot pool"):
            compute_pca(frame)
    mixed = pd.DataFrame(
        {"mean_spike_freq": [1, 2], "mean_expected_spike_count": [1, 3]}
    )
    with pytest.raises(ValueError, match="cannot pool"):
        compute_pca(mixed)
    if len(methods) == 1:
        other = "cascade" if methods == ("oasis",) else "oasis"
        with pytest.raises(ValueError, match="No stored"):
            build_fov_feature_matrix(engine, run_id, spike_method=other)


def test_pca_rejects_implicit_multi_run_pooling(evoked_database: tuple) -> None:
    engine, _, run_id, methods = evoked_database
    with Session(engine) as session:
        owner = session.get(CaliResult, run_id)
        other = CaliResult(experiment=owner.experiment)
        session.add(other)
        session.flush()
        session.add(
            FOVAnalysis(
                fov_id=session.exec(select(FOV)).first().id, analysis_result_id=other.id
            )
        )
        session.commit()
    with pytest.raises(ValueError, match="one analysis run"):
        build_fov_feature_matrix(engine, spike_method=methods[0])
    assert len(build_fov_feature_matrix(engine, run_id, spike_method=methods[0])) == 2


def test_pca_plots_and_dialog_bind_method_features(
    evoked_database: tuple, qtbot: QtBot
) -> None:
    engine, _, run_id, methods = evoked_database
    widget = _MultilWellGraphWidget(parent=None)
    qtbot.addWidget(widget)
    for method in methods:
        for render in (plot_pca_scatter, plot_pca_loadings, plot_pca_scree):
            render(widget, "PCA", engine, run_id, spike_method=method)
            assert method.upper() in widget.plot_item.titleLabel.text
        dialog = _PCAFeaturesDialog(None, EVOKED, False, spike_method=method)
        qtbot.addWidget(dialog)
        assert set(dialog._checkboxes) == set(
            FEATURE_COLUMNS if method == "oasis" else CASCADE_FEATURE_COLUMNS
        )
        if method == "cascade":
            assert (
                "Expected Spike Count"
                in dialog._checkboxes["mean_expected_spike_count"].text()
            )
            assert not dialog._checkboxes[
                "mean_suprathreshold_excursion_rate_hz"
            ].isEnabled()
        else:
            assert not dialog._checkboxes["mean_spike_freq_edges"].isEnabled()


def test_pca_exports_are_separate_and_describe_feature_meanings(
    evoked_database: tuple,
) -> None:
    engine, path, run_id, methods = evoked_database
    export_multi_well_pca_to_csv(engine, run_id, path)
    target = path.with_name(path.stem + "_exports") / f"run_{run_id}" / "multi_well"
    for method in methods:
        prefix = method + "_" if method == "cascade" or len(methods) > 1 else ""
        frame = pd.read_csv(target / f"{prefix}pca_feature_matrix.csv")
        coordinates = pd.read_csv(target / f"{prefix}pca_coordinates.csv")
        assert (
            frame.spike_method.eq(method).all()
            and coordinates.spike_method.eq(method).all()
        )
        metadata = json.loads((target / f"{prefix}pca.metadata.json").read_text())
        assert metadata["method"] == method
        assert metadata["calcium_activity_field"] == "calcium_active"
        assert metadata["spike_activity_field"] == "spike_active"
        assert ("mean_expected_spike_count" in metadata["spike_metric_fields"]) == (
            method == "cascade"
        )
        if method == "cascade":
            assert frame["mean_expected_spike_count"].tolist() == [1.5, 3]
    if len(methods) > 1:
        assert not (target / "pca_feature_matrix.csv").exists()


def test_pca_export_rejects_missing_method_before_writing(
    evoked_database: tuple, tmp_path: Path
) -> None:
    engine, _, run_id, methods = evoked_database
    if len(methods) > 1:
        return
    other = "cascade" if methods == ("oasis",) else "oasis"
    path = tmp_path / "unavailable.cali"
    with pytest.raises(ValueError, match="No stored spike analysis"):
        export_multi_well_pca_to_csv(engine, run_id, path, spike_methods=(other,))
    assert not path.with_name("unavailable_exports").exists()


def test_pca_export_rejects_bad_selected_units_before_replacing_files(
    evoked_database: tuple,
) -> None:
    engine, path, run_id, methods = evoked_database
    export_multi_well_pca_to_csv(engine, run_id, path)
    target = path.with_name(path.stem + "_exports")
    before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    with engine.begin() as connection:
        connection.execute(
            update(SpikeAnalysis)
            .where(SpikeAnalysis.method == methods[-1])
            .values(units="invalid")
        )
    with pytest.raises(ValueError, match="metric units"):
        export_multi_well_pca_to_csv(engine, run_id, path)
    assert before == {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
