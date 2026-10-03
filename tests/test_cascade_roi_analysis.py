"""Scientific CASCADE ROI semantics and shared extraction/re-analysis parity."""

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.ndimage import binary_dilation, gaussian_filter1d
from sqlmodel import Session, select

from cali.analysis import AnalysisRunner
from cali.analysis._roi_analysis import (
    AnalysisCancelled,
    analyze_roi_traces,
    analyze_spike_trace,
    valid_spike_events,
)
from cali.extraction._extraction_runner import ExtractionRunner, _RoiParts
from cali.extraction._frame_window import ExtractionFrameWindow as ResolvedWindow
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    Experiment,
    ExtractionFrameWindow,
    SpikeAnalysis,
    SpikeAnalysisSettings,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    create_cali_engine,
    create_database_and_tables,
)


def _trace(methods: tuple = ("cascade",)) -> Traces:
    calcium = np.zeros(100)
    calcium[[30, 70]] = 2
    oasis = np.zeros(100)
    oasis[[20, 60]] = 0.75
    oasis[0] = -1e-15  # OASIS numerical residuals stay unchanged.
    cascade = np.zeros(100)
    cascade[10] = 0.5  # left-censored excursion, still included in expected count
    cascade[20:23] = 0.25
    cascade[60] = 0.5
    cascade[:10] = 100
    cascade[90:] = 100
    return Traces(
        raw_trace=calcium.tolist(),
        dff=calcium.tolist(),
        den_dff=calcium.tolist(),
        calcium_noise=0.01,
        x_axis=(np.arange(100) * 100.0).tolist(),
        x_axis_units="ms",
        extraction_frame_window=ExtractionFrameWindow(
            original_frame_count=110,
            retained_frame_count=100,
            source_start_frame=10,
            source_start_time_ms=1000,
            acquisition_frame_rate_hz=10,
            timing_source="runner_time",
        ),
        spike_traces=[
            SpikeTrace(
                values=(oasis if method == "oasis" else cascade).tolist(),
                valid_start=0 if method == "oasis" else 10,
                valid_stop=100 if method == "oasis" else 90,
                inference_run=SpikeInferenceRun(
                    method=method,
                    units="a.u." if method == "oasis" else "spikes/frame",
                    backend_package="oasis-deconv"
                    if method == "oasis"
                    else "CascadeTorch",
                    model_sampling_rate_hz=10 if method == "cascade" else None,
                    smoothing_sigma=0.2 if method == "cascade" else None,
                ),
            )
            for method in methods
        ],
    )


def _settings(methods: tuple = ("cascade",), **kwargs: object) -> AnalysisSettings:
    return AnalysisSettings(
        frame_rate=10,
        spike_settings=[
            SpikeAnalysisSettings(
                method=method,
                threshold_mode="global",
                threshold_value=0.2,
                enable_rising_edge_analysis=True,
            )
            for method in methods
        ],
        **kwargs,
    )


def _calcium(result: DataAnalysis) -> dict:
    return result.model_dump(
        exclude={"id", "created_at", "roi_id", "analysis_result_id", "spike_analyses"}
    )


def _spikes(result: DataAnalysis) -> list[dict]:
    return [
        child.model_dump(
            exclude={
                "id",
                "data_analysis_id",
                "analysis_result_id",
                "spike_trace_id",
                "spike_trace",
            }
        )
        for child in sorted(
            result.spike_analyses,
            key=lambda child: (child.method != "oasis", child.method),
        )
    ]


def test_expected_counts_rates_and_excursions_use_only_valid_samples() -> None:
    trace = _trace()
    spike = trace.get_spike_trace("cascade")
    result = analyze_spike_trace(
        spike, _settings().get_spike_settings("cascade"), legacy_duration_s=10000
    )
    assert result.expected_spike_count == 1.75
    assert result.expected_spike_rate_hz == 1.75 / 80 * 10
    assert result.suprathreshold_excursion_rate_hz == 2 / 80 * 10
    assert result.suprathreshold_sample_rate_hz is None
    assert result.suprathreshold_rising_edge_rate_hz is None
    assert result.spike_active is True
    assert result.threshold_mode == "global" and result.units == "spikes/frame"
    assert result.spike_trace is spike
    binary, onsets = valid_spike_events(spike, 0.2)
    assert len(binary) == len(onsets) == 80
    assert binary[0] == 1 and onsets[0] == 0
    assert np.flatnonzero(onsets).tolist() == [10, 50]


def test_zero_signal_has_zero_expected_rate_and_no_excursions() -> None:
    trace = _trace()
    spike = trace.get_spike_trace("cascade")
    spike.values = [0.0] * 100
    settings = SpikeAnalysisSettings(method="cascade", enable_rising_edge_analysis=True)
    result = analyze_spike_trace(spike, settings, legacy_duration_s=9.9)
    assert result.expected_spike_count == result.expected_spike_rate_hz == 0
    assert result.suprathreshold_excursion_rate_hz == 0
    assert result.spike_active is False
    settings.enable_rising_edge_analysis = False
    result = analyze_spike_trace(spike, settings, legacy_duration_s=9.9)
    assert result.expected_spike_count == result.expected_spike_rate_hz == 0
    assert result.suprathreshold_excursion_rate_hz is None


def test_ap_threshold_is_exact_model_kernel_cutoff_without_dilation() -> None:
    trace = _trace()
    spike = trace.get_spike_trace("cascade")
    impulse = np.zeros(1001)
    impulse[500] = 1
    peak = gaussian_filter1d(impulse, sigma=2).max()
    fraction = 1 / math.e
    spike.values = [0.0] * 100
    spike.values[40:45] = [0.01, 0.01, float(peak), 0.01, 0.01]
    settings = SpikeAnalysisSettings(method="cascade", enable_rising_edge_analysis=True)
    result = analyze_spike_trace(spike, settings, legacy_duration_s=9.9)
    assert result.threshold == fraction * peak
    binary, _ = valid_spike_events(spike, result.threshold)
    upstream_mask = binary_dilation(binary.astype(bool), iterations=2)
    assert binary.sum() == 1 and upstream_mask.sum() == 5
    assert result.expected_spike_count == pytest.approx(peak + 0.04)
    assert result.suprathreshold_excursion_rate_hz == 10 / 80


def test_two_smoothed_action_potentials_can_form_one_excursion() -> None:
    trace = _trace()
    spike = trace.get_spike_trace("cascade")
    impulses = np.zeros(100)
    impulses[[40, 43]] = 1
    spike.values = gaussian_filter1d(impulses, sigma=2).tolist()
    result = analyze_spike_trace(
        spike,
        SpikeAnalysisSettings(method="cascade", enable_rising_edge_analysis=True),
        legacy_duration_s=9.9,
    )
    assert result.expected_spike_count == pytest.approx(2, abs=1e-12)
    assert result.suprathreshold_excursion_rate_hz == 10 / 80


def test_stored_rates_drive_analysis_even_when_current_settings_differ() -> None:
    trace = _trace()
    spike = trace.get_spike_trace("cascade")
    trace.extraction_frame_window.acquisition_frame_rate_hz = 30
    spike.inference_run.model_sampling_rate_hz = 30
    result = analyze_roi_traces(trace, _settings(), duration_s=9.9)
    assert result.spike_analyses[0].expected_spike_rate_hz == 1.75 / 80 * 30


@pytest.mark.parametrize(
    "rate,model_rate",
    [
        (None, 10),
        (0, 10),
        (-1, 10),
        (math.nan, 10),
        (math.inf, 10),
        (10, None),
        (10, 0),
        (10, 30),
    ],
)
def test_missing_or_incompatible_persisted_rates_fail(
    rate: float | None, model_rate: float | None
) -> None:
    trace = _trace()
    spike = trace.get_spike_trace("cascade")
    trace.extraction_frame_window.acquisition_frame_rate_hz = rate
    spike.inference_run.model_sampling_rate_hz = model_rate
    with pytest.raises(ValueError, match="acquisition/model rates"):
        analyze_spike_trace(
            spike,
            SpikeAnalysisSettings(method="cascade", enable_rising_edge_analysis=True),
            legacy_duration_s=9.9,
        )


@pytest.mark.parametrize("smoothing", [None, 0, -1, math.nan, math.inf])
def test_ap_threshold_requires_stored_smoothing_but_global_does_not(
    smoothing: float | None,
) -> None:
    trace = _trace()
    spike = trace.get_spike_trace("cascade")
    spike.inference_run.smoothing_sigma = smoothing
    with pytest.raises(ValueError, match="persisted positive smoothing"):
        analyze_spike_trace(
            spike,
            SpikeAnalysisSettings(method="cascade", enable_rising_edge_analysis=True),
            legacy_duration_s=9.9,
        )
    assert (
        analyze_spike_trace(
            spike, _settings().get_spike_settings("cascade"), legacy_duration_s=9.9
        ).threshold
        == 0.2
    )


@pytest.mark.parametrize(
    "change,message",
    [
        ({"valid_start": 90}, "nonempty"),
        ({"valid_stop": 101}, "nonempty"),
        ({"values": [math.nan] * 100}, "finite"),
        ({"values": [-0.01] * 100}, "non-negative"),
    ],
)
def test_invalid_cascade_interval_or_values_fail(change: dict, message: str) -> None:
    trace = _trace()
    spike = trace.get_spike_trace("cascade")
    for name, value in change.items():
        setattr(spike, name, value)
    with pytest.raises(ValueError, match=message):
        analyze_spike_trace(
            spike,
            SpikeAnalysisSettings(method="cascade", enable_rising_edge_analysis=True),
            legacy_duration_s=9.9,
        )


def test_illegal_method_units_and_mutated_threshold_modes_fail() -> None:
    trace = _trace()
    spike = trace.get_spike_trace("cascade")
    with pytest.raises(ValueError, match="stored inference method"):
        analyze_spike_trace(spike, SpikeAnalysisSettings(), legacy_duration_s=9.9)
    settings = SpikeAnalysisSettings(method="cascade", enable_rising_edge_analysis=True)
    settings.threshold_mode = "multiplier"
    with pytest.raises(ValueError, match="Invalid threshold mode"):
        analyze_spike_trace(spike, settings, legacy_duration_s=9.9)
    spike.inference_run.units = "a.u."
    with pytest.raises(ValueError, match="units must match"):
        analyze_spike_trace(
            spike,
            SpikeAnalysisSettings(method="cascade", enable_rising_edge_analysis=True),
            legacy_duration_s=9.9,
        )


@pytest.mark.parametrize(
    "calcium,spikes", [(True, True), (True, False), (False, True), (False, False)]
)
def test_shared_roi_results_are_independent_of_retained_spike_outputs(
    calcium: bool, spikes: bool
) -> None:
    results = {}
    for methods in (("oasis",), ("cascade",), ("oasis", "cascade")):
        result = analyze_roi_traces(
            _trace(methods),
            _settings(methods, enable_calcium=calcium, enable_spikes=spikes),
            duration_s=9.9,
        )
        results[methods] = result
    assert (
        _calcium(results[("oasis",)])
        == _calcium(results[("cascade",)])
        == _calcium(results[("oasis", "cascade")])
    )
    if spikes:
        assert _spikes(results[("oasis", "cascade")]) == [
            *_spikes(results[("oasis",)]),
            *_spikes(results[("cascade",)]),
        ]
    else:
        assert all(not result.spike_analyses for result in results.values())


@pytest.mark.parametrize("methods", [("oasis",), ("cascade",), ("oasis", "cascade")])
def test_inline_roi_finalization_matches_analysis_only_calculations(
    methods: tuple,
) -> None:
    source = _trace(methods)
    window = source.extraction_frame_window
    parts = _RoiParts(
        1,
        np.ones((2, 2), bool),
        np.asarray(source.raw_trace),
        None,
        None,
        np.asarray(source.dff),
        4,
        "pixels",
    )
    settings = _settings(methods)
    finalized = ExtractionRunner()._finalize_roi(
        parts,
        np.asarray(source.den_dff),
        np.zeros(100),
        0.01,
        settings,
        9.9,
        source.x_axis,
        "ms",
        ResolvedWindow(10, 1000, 110, 100, 1000, "runner_time"),
        stored_window=window,
        spike_traces=source.spike_traces,
    )
    assert finalized is not None
    trace, inline, _, _, _, _ = finalized
    reanalysis, _, _ = AnalysisRunner()._analyze_roi_traces(
        trace, settings, ROI(label_value=1)
    )
    assert _calcium(inline) == _calcium(reanalysis)
    assert _spikes(inline) == _spikes(reanalysis)


def test_calcium_only_reanalysis_accepts_cascade_settings() -> None:
    roi = ROI(label_value=1, traces_history=[_trace()])
    fov = FOV(name="A1", position_index=0, rois=[roi])
    result = AnalysisRunner().run([fov], _settings(enable_spikes=False))
    assert result == [fov]
    assert roi._new_data_analysis[0].calcium_active is True
    assert roi._new_data_analysis[0].spike_analyses == []


def test_missing_method_settings_and_cancellation_do_not_publish_partial_rows() -> None:
    trace = _trace(("oasis", "cascade"))
    with pytest.raises(ValueError, match="must match"):
        analyze_roi_traces(trace, _settings(("oasis",)), duration_s=9.9)
    methods_seen = []
    original = analyze_spike_trace

    def cancel_after_oasis(*args: object, **kwargs: object) -> SpikeAnalysis:
        result = original(*args, **kwargs)
        methods_seen.append(result.method)
        return result

    with (
        patch(
            "cali.analysis._roi_analysis.analyze_spike_trace",
            side_effect=cancel_after_oasis,
        ),
        pytest.raises(AnalysisCancelled),
    ):
        analyze_roi_traces(
            trace,
            _settings(("oasis", "cascade")),
            duration_s=9.9,
            cancel=lambda: bool(methods_seen),
        )
    assert methods_seen == ["oasis"]
    assert (
        trace.get_spike_trace("cascade").values
        == _trace().get_spike_trace("cascade").values
    )


def test_oasis_initial_sample_and_sparse_disable_threshold_remain_exact() -> None:
    trace = _trace(("oasis",))
    spike = trace.get_spike_trace("oasis")
    spike.values = [1.0] + [0.0] * 99
    settings = SpikeAnalysisSettings(
        threshold_mode="global", threshold_value=0.5, enable_rising_edge_analysis=True
    )
    result = analyze_spike_trace(spike, settings, legacy_duration_s=9.9)
    assert (
        result.suprathreshold_sample_rate_hz
        == result.suprathreshold_rising_edge_rate_hz
        == 1 / 9.9
    )
    disabled = analyze_spike_trace(
        spike, SpikeAnalysisSettings(), legacy_duration_s=9.9
    )
    assert disabled.threshold == math.inf and disabled.spike_active is False
    assert disabled.suprathreshold_sample_rate_hz is None
    assert disabled.suprathreshold_rising_edge_rate_hz is None


def test_persisted_dual_roi_results_snapshots_and_offline_reanalysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CALI_CASCADE_MODELS", str(tmp_path / "absent-models"))
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'roi.cali'}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            experiment = Experiment(name="ROI method semantics")
            fov = FOV(name="A1", position_index=0, rois=[ROI(label_value=1)])
            settings = _settings(("oasis", "cascade"))
            session.add_all([experiment, fov, settings])
            session.flush()
            result = CaliResult(
                experiment=experiment.id, analysis_settings_id=settings.id
            )
            source = _trace(("oasis", "cascade"))
            source.roi = fov.rois[0]
            source.analysis_result = result
            analysis = analyze_roi_traces(source, settings, duration_s=9.9)
            analysis.roi = source.roi
            analysis.analysis_result = result
            session.add_all([source, analysis])
            session.commit()
            session.expunge_all()
            reloaded = session.exec(select(Traces)).one()
            persisted = session.exec(select(DataAnalysis)).one()
            settings = session.exec(select(AnalysisSettings)).one()
            session.expunge_all()
        offline = analyze_roi_traces(reloaded, settings, duration_s=9.9)
        assert _calcium(offline) == _calcium(persisted)
        assert _spikes(offline) == _spikes(persisted)
        snapshot = DataAnalysis.model_validate_json(persisted.model_dump_json())
        assert _spikes(snapshot) == _spikes(persisted)
    finally:
        engine.dispose()


def test_pretrained_golden_expected_counts_and_model_ap_threshold() -> None:
    fixture = Path(__file__).parent / "fixtures/cascade_reference/real_excerpt.npz"
    impulse = np.zeros(1001)
    impulse[500] = 1
    ap_threshold = gaussian_filter1d(impulse, sigma=0.025 * 30).max() / math.e
    with np.load(fixture) as data:
        for values in data["expected_spikes"].astype(np.float32):
            trace = Traces(
                extraction_frame_window=ExtractionFrameWindow(
                    acquisition_frame_rate_hz=30
                ),
                spike_traces=[
                    SpikeTrace(
                        values=values.tolist(),
                        valid_start=32,
                        valid_stop=224,
                        inference_run=SpikeInferenceRun(
                            method="cascade",
                            units="spikes/frame",
                            backend_package="CascadeTorch",
                            model_sampling_rate_hz=30,
                            smoothing_sigma=0.025,
                        ),
                    )
                ],
            )
            result = analyze_spike_trace(
                trace.spike_traces[0],
                SpikeAnalysisSettings(method="cascade"),
                legacy_duration_s=255 / 30,
            )
            valid = values[32:224].astype(np.float64)
            assert result.expected_spike_count == valid.sum()
            assert result.expected_spike_rate_hz == valid.mean() * 30
            assert result.threshold == ap_threshold


def test_oasis_cropped_interval_does_not_create_an_onset_at_its_boundary() -> None:
    trace = _trace(("oasis",))
    spike = trace.get_spike_trace("oasis")
    spike.values = [0.0] * 100
    spike.values[10:14] = [1, 1, 0, 1]
    spike.valid_start, spike.valid_stop = 10, 14
    settings = SpikeAnalysisSettings(
        threshold_mode="global", threshold_value=0.5, enable_rising_edge_analysis=True
    )
    result = analyze_spike_trace(spike, settings, legacy_duration_s=9.9)
    assert result.suprathreshold_sample_rate_hz == 3 / 0.3
    assert result.suprathreshold_rising_edge_rate_hz == 1 / 0.3
    _, onsets = valid_spike_events(spike, 0.5)
    assert onsets.tolist() == [0, 0, 0, 1]


def test_sparse_oasis_disable_threshold_survives_database_and_json(
    tmp_path: Path,
) -> None:
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'silent.cali'}")
    create_database_and_tables(engine)
    try:
        with Session(engine) as session:
            experiment = Experiment(name="silent OASIS")
            roi = ROI(label_value=1, fov=FOV(name="A1", position_index=0))
            settings = AnalysisSettings(enable_calcium=False)
            session.add_all([experiment, roi, settings])
            session.flush()
            owner = CaliResult(
                experiment=experiment.id, analysis_settings_id=settings.id
            )
            trace = _trace(("oasis",))
            trace.spike_traces[0].values = [0.0] * 100
            trace.roi, trace.analysis_result = roi, owner
            result = analyze_roi_traces(trace, settings, duration_s=9.9)
            result.roi, result.analysis_result = roi, owner
            session.add_all([trace, result])
            session.commit()
            session.expunge_all()
            result = session.exec(select(DataAnalysis)).one()
            assert result.get_spike_analysis("oasis").threshold == math.inf
            snapshot = DataAnalysis.model_validate_json(result.model_dump_json())
            assert snapshot.get_spike_analysis("oasis").threshold == math.inf
            assert snapshot.get_spike_analysis("oasis").spike_active is False
    finally:
        engine.dispose()
