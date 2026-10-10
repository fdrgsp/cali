"""Exports preserve method meaning, population membership and valid coordinates."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session, select

from cali._constants import (
    CASCADE_EXPECTED_SPIKES_TRACES,
    INFERRED_SPIKES_SYNCHRONY,
    INFERRED_SPIKES_TRACES,
)
from cali.analysis._fov_analysis import compute_fov_analysis
from cali.analysis._roi_analysis import analyze_spike_trace
from cali.gui._extraction_gui import _ExtractionGUI
from cali.sqlmodel import (
    CaliResult,
    Experiment,
    FOVAnalysis,
    create_cali_engine,
    create_database_and_tables,
)
from cali.util import (
    export_correlation_matrices_to_csv,
    export_events_to_csv,
    export_inferred_spikes_synchrony_to_csv,
    export_inferred_spikes_thresholded_to_csv,
    export_spike_results_to_csv,
)
from cali.util._database_to_csv import export_correlations_to_csv, export_traces_to_csv

from .test_cascade_fov_analysis import _fov, _settings

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pytestqt.qtbot import QtBot
    from sqlalchemy.engine import Engine


def test_gui_raw_export_selection_writes_only_retained_methods(
    qtbot: QtBot, spike_database: tuple
) -> None:
    engine, path, run_id, methods = spike_database
    widget = _ExtractionGUI(cascade_enabled=True)
    qtbot.addWidget(widget)
    widget._spike_outputs.setValue(
        methods, "explicit-model" if "cascade" in methods else None, "cpu"
    )
    keys = {"oasis": INFERRED_SPIKES_TRACES, "cascade": CASCADE_EXPECTED_SPIKES_TRACES}
    for key, (checkbox, _, _) in widget._export_group._checkboxes.items():
        checkbox.setChecked(key in keys.values())
    options = widget.get_export_options()
    assert options == {keys[method]: True for method in methods}
    export_traces_to_csv(engine, options, run_id, path)
    target = path.parent / f"{path.stem}_exports" / f"run_{run_id}"
    assert bool(list(target.rglob("cascade_expected_spikes.csv"))) == (
        "cascade" in methods
    )
    oasis_files = list(target.rglob("*inferred_spikes_raw.csv"))
    assert bool(oasis_files) == ("oasis" in methods)


@pytest.fixture(params=[("oasis",), ("cascade",), ("oasis", "cascade")])
def spike_database(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[tuple[Engine, Path, int, tuple]]:
    methods = request.param
    path = tmp_path / "export.cali"
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            experiment = Experiment(name="method-qualified exports")
            session.add(experiment)
            session.flush()
            owner = CaliResult(experiment=experiment.id, positions_extracted=[0, 1])
            session.add(owner)
            session.flush()
            settings = _settings(methods)
            for position in (0, 1):
                fov = _fov(methods)
                fov.name, fov.position_index = f"A1_{position}", position
                for roi in fov.rois:
                    roi.stimulated = roi.label_value == 3
                    trace = roi._new_traces[-1]
                    trace.analysis_result = owner
                    analysis = roi._new_data_analysis[-1]
                    analysis.analysis_result = owner
                    analysis.spike_analyses.clear()
                    analysis.spike_analyses = [
                        analyze_spike_trace(
                            spike,
                            settings.get_spike_settings(spike.inference_run.method),
                            legacy_duration_s=9.9,
                            source_trace=trace,
                        )
                        for spike in trace.spike_traces
                    ]
                parent = compute_fov_analysis(fov, settings)
                parent.fov, parent.analysis_result = fov, owner
                for child in parent.spike_analyses:
                    child.spike_jitter_synchrony_matrix = [[1, 0.1], [0.8, 1]]
                    child.spike_burst_starts, child.spike_burst_ends = [30], [40]
                    child.spike_burst_count = 1
                    child.spike_burst_avg_duration = 10 / child.frame_rate_hz
                session.add_all([fov, parent])
            session.commit()
            result_id = owner.id
        yield engine, path, result_id, methods
    finally:
        engine.dispose()


def test_bundle_keeps_roi_metrics_membership_units_and_population_coordinates(
    spike_database: tuple, tmp_path: Path
) -> None:
    engine, _, run_id, methods = spike_database
    target = tmp_path / "bundle"
    assert export_spike_results_to_csv(engine, target, run_id=run_id) == methods
    roi = pd.read_csv(target / "spike_roi_metrics.csv")
    assert len(roi) == 6 * len(methods)
    assert set(roi.method) == set(methods)
    assert roi.threshold_mode.eq("global").all()
    assert roi.threshold.eq(0.2).all()
    for method in methods:
        rows = roi[roi.method == method]
        assert rows.units.eq("a.u." if method == "oasis" else "spikes/frame").all()
        assert rows.threshold_units.eq(rows.units).all()
        assert rows[
            rows.roi_label == 3
        ].spike_active.all()  # union flag deliberately False
        if method == "cascade":
            assert rows.expected_spike_count.tolist() == [0, 1.5, 1.5] * 2
            assert rows.suprathreshold_sample_rate_hz.isna().all()
            assert rows.suprathreshold_rising_edge_rate_hz.isna().all()
        else:
            assert rows.expected_spike_count.isna().all()
            assert rows.expected_spike_rate_hz.isna().all()
    fovs = pd.read_csv(target / "spike_fov_metrics.csv")
    populations = pd.read_csv(target / "spike_population_activity.csv")
    bursts = pd.read_csv(target / "spike_bursts.csv")
    metadata = json.loads((target / "spike_results.metadata.json").read_text())
    assert metadata["methods"] == list(methods)
    assert sorted(metadata["files"]) == sorted(path.name for path in target.iterdir())
    for method in methods:
        active = [1, 3] if method == "oasis" else [2, 3]
        start, stop, rate = (0, 100, 25) if method == "oasis" else (20, 85, 10)
        row = fovs[fovs.method == method].iloc[0]
        assert json.loads(row.active_roi_labels) == active
        thresholds = json.loads(row.thresholds)
        assert [item["roi_label"] for item in thresholds] == active
        assert all(
            item["method"] == method and item["threshold_mode"] == "global"
            for item in thresholds
        )
        samples = populations[
            (populations.method == method) & (populations.position_index == 0)
        ]
        assert samples.retained_frame_0based.tolist() == list(range(start, stop))
        assert samples.frame_rate_hz.eq(rate).all()
        assert len(samples) == stop - start
        assert bursts[bursts.method == method].start_retained_frame_0based.tolist() == [
            30,
            30,
        ]
        record = next(item for item in metadata["fovs"] if item["method"] == method)
        assert record["frame_window"]["source_start_frame"] == 10
        assert record["inference_run"]["method"] == method
        matrix_metadata = next(
            item
            for item in record["matrices"]
            if item["metric"] == "spike_jitter_synchrony_matrix"
        )
        assert matrix_metadata["exported_roi_labels"] == [3, active[0]]
        assert record["metric_units"]["spike_burst_avg_duration"] == "seconds"
        matrix = pd.read_csv(
            target / f"A1_0_{method}_spike_jitter_synchrony_matrix.csv", index_col=0
        )
        assert matrix.columns.tolist() == ["ROI_3", f"ROI_{active[0]}"]
        np.testing.assert_array_equal(matrix.to_numpy(), [[1, 0.8], [0.1, 1]])


def test_high_level_correlations_export_each_method_without_overwriting(
    spike_database: tuple,
) -> None:
    engine, path, run_id, methods = spike_database
    export_correlations_to_csv(
        engine, {INFERRED_SPIKES_SYNCHRONY: True}, run_id, path, position_indices=[1]
    )
    target = path.with_name("export_exports") / f"run_{run_id}"
    for method in methods:
        qualified = f"{method}_" if methods != ("oasis",) else ""
        matrix_path = target / f"A1_1_{qualified}inferred_spikes_synchrony_matrix.csv"
        assert matrix_path.exists()
        assert not (
            target / f"A1_0_{qualified}inferred_spikes_synchrony_matrix.csv"
        ).exists()
        metadata = json.loads(matrix_path.with_suffix(".metadata.json").read_text())
        assert metadata["method"] == method
        assert metadata["threshold_units"] == (
            "a.u." if method == "oasis" else "spikes/frame"
        )
    assert pd.read_csv(target / "spike_roi_metrics.csv").position_index.eq(1).all()


def test_all_matrix_and_single_matrix_exports_select_the_stored_method(
    spike_database: tuple, tmp_path: Path
) -> None:
    engine, _, run_id, methods = spike_database
    method = methods[-1]
    target = tmp_path / "matrices"
    export_correlation_matrices_to_csv(
        engine, target, run_id=run_id, fov_name="A1_0", spike_methods=(method,)
    )
    qualifier = f"{method}_" if methods != ("oasis",) else ""
    matrix = pd.read_csv(target / f"{qualifier}spike_jitter_synchrony.csv", index_col=0)
    assert matrix.columns.tolist() == [
        "ROI_3",
        "ROI_1" if method == "oasis" else "ROI_2",
    ]
    export_inferred_spikes_synchrony_to_csv(
        engine,
        tmp_path / "single.csv",
        run_id=run_id,
        fov_name="A1_0",
        spike_method=method,
    )
    np.testing.assert_array_equal(
        pd.read_csv(tmp_path / "A1_0_single.csv", index_col=0).to_numpy(),
        matrix.to_numpy(),
    )


def test_binary_and_event_exports_use_strict_cutoffs_and_censored_onsets(
    spike_database: tuple, tmp_path: Path
) -> None:
    engine, _, run_id, methods = spike_database
    method = methods[-1]
    with engine.begin() as connection:
        # Exactly at the cutoff is inactive, matching analysis's strict > rule.
        from cali.sqlmodel._trace_array_codec import (
            decode_trace_array,
            encode_trace_array,
        )

        row = connection.exec_driver_sql(
            'SELECT id,"values" FROM spike_trace WHERE valid_start=? '
            "ORDER BY id LIMIT 1",
            (0 if method == "oasis" else 10,),
        ).one()
        values = decode_trace_array(row[1])
        values[35] = 0.2
        connection.exec_driver_sql(
            'UPDATE spike_trace SET "values"=? WHERE id=?',
            (encode_trace_array(values), row[0]),
        )
    export_inferred_spikes_thresholded_to_csv(
        engine,
        tmp_path / "binary.csv",
        run_id=run_id,
        fov_name="A1_0",
        spike_method=method,
    )
    binary = pd.read_csv(tmp_path / "binary.csv")
    assert binary.iloc[35].eq(0).all()
    if method == "cascade":
        assert binary.iloc[:10].isna().all().all()
        assert binary.iloc[90:].isna().all().all()
    metadata = json.loads((tmp_path / "binary.metadata.json").read_text())
    assert metadata["method"] == method
    assert all(
        item["threshold_mode"] == "global" and item["method"] == method
        for item in metadata["thresholds"]
    )
    export_events_to_csv(
        engine, tmp_path / "events.csv", run_id=run_id, fov_name="A1_0"
    )
    events = pd.read_csv(tmp_path / "events.csv")
    spike_events = events[events.event_type == "threshold_excursion_start"]
    assert set(spike_events.method) == set(methods)
    assert spike_events.threshold_mode.eq("global").all()
    for method in methods:
        frames = spike_events[
            spike_events.method == method
        ].retained_frame_0based.tolist()
        assert frames == ([0, 30, 70] * 2 if method == "oasis" else [30, 70] * 2)


def test_unknown_historical_population_origin_is_exported_as_unknown(
    spike_database: tuple, tmp_path: Path
) -> None:
    engine, _, run_id, _ = spike_database
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE spike_fov_analysis SET valid_start=NULL, valid_stop=NULL, "
            "frame_rate_hz=NULL"
        )
    export_spike_results_to_csv(engine, tmp_path / "historical", run_id=run_id)
    samples = pd.read_csv(tmp_path / "historical/spike_population_activity.csv")
    assert samples.retained_frame_0based.isna().all()
    assert samples.valid_start_frame_0based.isna().all()
    assert samples.frame_rate_hz.isna().all()
    metadata = json.loads(
        (tmp_path / "historical/spike_results.metadata.json").read_text()
    )
    assert all(
        item["population_coordinate_basis"] == "unknown" for item in metadata["fovs"]
    )


@pytest.mark.parametrize("malformed", ["array", "matrix"])
def test_failed_bundle_replaces_no_existing_output(
    spike_database: tuple,
    tmp_path: Path,
    malformed: str,
) -> None:
    engine, _, run_id, methods = spike_database
    target = tmp_path / "atomic"
    export_spike_results_to_csv(engine, target, run_id=run_id)
    original = {path.name: path.read_bytes() for path in target.iterdir()}
    with engine.begin() as connection:
        column, value = (
            ("spike_population_activity_raw", "[1]")
            if malformed == "array"
            else ("spike_jitter_synchrony_matrix", "[[1]]")
        )
        connection.exec_driver_sql(
            f"UPDATE spike_fov_analysis SET {column}=? WHERE method=?",
            (value, methods[-1]),
        )
    with pytest.raises(ValueError, match=r"valid interval|matching dimensions"):
        export_spike_results_to_csv(engine, target, run_id=run_id)
    assert {path.name: path.read_bytes() for path in target.iterdir()} == original


def test_explicit_unavailable_or_invalid_methods_fail_before_export(
    spike_database: tuple, tmp_path: Path
) -> None:
    engine, path, run_id, methods = spike_database
    requested = (
        ("wrong",)
        if len(methods) == 2
        else (("cascade",) if methods == ("oasis",) else ("oasis",))
    )
    with pytest.raises(ValueError):
        export_spike_results_to_csv(
            engine, tmp_path / "missing", run_id=run_id, spike_methods=requested
        )
    assert not (tmp_path / "missing").exists()
    with pytest.raises(ValueError):
        export_correlations_to_csv(
            engine,
            {INFERRED_SPIKES_SYNCHRONY: True},
            run_id,
            path,
            spike_methods=requested,
        )


def test_sparse_oasis_disable_threshold_remains_explicit_standard_json(
    tmp_path: Path,
) -> None:
    # Reuse the persisted fixture's helper, with a sparse zero-signal result.
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'sparse.cali'}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            fov = _fov(("oasis",))
            for roi in fov.rois:
                metric = roi._new_data_analysis[-1].get_spike_analysis("oasis")
                metric.threshold, metric.threshold_mode = np.inf, "multiplier"
            parent = compute_fov_analysis(fov, _settings(("oasis",)))
            parent.fov = fov
            session.add_all([fov, parent])
            session.commit()
            parent = session.exec(select(FOVAnalysis)).one()
            from cali.util._spike_export import fov_spike_metadata

            metadata = fov_spike_metadata(
                session, parent, fov, parent.get_spike_analysis("oasis")
            )
            assert all(
                item["threshold"] == "+infinity" for item in metadata["thresholds"]
            )
            json.dumps(metadata, allow_nan=False)
    finally:
        engine.dispose()
