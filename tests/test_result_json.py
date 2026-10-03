"""Normalized product snapshots retain provenance without ORM graph cycles."""

import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from cali.sqlmodel import (
    DataAnalysis,
    ExtractionFrameWindow,
    FOVAnalysis,
    SpikeAnalysis,
    SpikeFOVAnalysis,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    create_cali_engine,
    create_database_and_tables,
)


def _products() -> tuple[Traces, DataAnalysis, FOVAnalysis]:
    oasis = SpikeInferenceRun(
        id=4,
        extraction_result_id=9,
        backend_version="1.2.3",
        resolved_device="cpu",
        dtype="float64",
    )
    cascade = SpikeInferenceRun(
        id=6,
        extraction_result_id=9,
        method="cascade",
        units="spikes/frame",
        backend_package="synthetic-test",
        resolved_model="verified-model",
        config_sha256="ab" * 32,
        weights_manifest_sha256="cd" * 32,
        resolved_device="cpu",
        dtype="float32",
        model_sampling_rate_hz=10,
    )
    trace = Traces(
        id=10,
        roi_id=5,
        analysis_result_id=31,
        raw_trace=[1, 2, 1],
        dff=[0, 0.1, 0],
        den_dff=[0, 0.09, 0],
        x_axis=[0, 100, 200],
        x_axis_units="ms",
        extraction_frame_window_id=3,
        extraction_frame_window=ExtractionFrameWindow(
            id=3,
            extraction_result_id=9,
            fov_id=7,
            requested_discard_value=1,
            original_frame_count=4,
            retained_frame_count=3,
            source_start_frame=1,
            source_start_time_ms=100,
            source_time_origin_ms=2000,
            source_start_timestamp_ms=2100,
            discarded_duration_ms=100,
            timing_source="timestamps",
            provenance_source="extraction",
        ),
        spike_traces=[
            SpikeTrace(
                id=12,
                trace_id=10,
                spike_inference_run_id=4,
                inference_run=oasis,
                values=[0, 0.75, 0],
                noise=0.25,
                ar_coefficients=[0.8],
                valid_stop=3,
            ),
            SpikeTrace(
                id=13,
                trace_id=10,
                spike_inference_run_id=6,
                inference_run=cascade,
                values=[0, 0.2, 0],
                valid_start=1,
                valid_stop=2,
                selected_noise_level=2,
            ),
        ],
    )
    analysis = DataAnalysis(
        id=30,
        roi_id=5,
        analysis_result_id=31,
        calcium_active=False,
        den_dff_frequency=0,
        spike_analyses=[
            SpikeAnalysis(
                id=32,
                data_analysis_id=30,
                analysis_result_id=31,
                spike_trace_id=12,
                spike_trace=trace.spike_traces[0],
                threshold=0.5,
                threshold_mode="global",
                suprathreshold_sample_rate_hz=0.1,
                spike_active=True,
            ),
            SpikeAnalysis(
                id=33,
                data_analysis_id=30,
                analysis_result_id=31,
                spike_trace_id=13,
                spike_trace=trace.spike_traces[1],
                method="cascade",
                units="spikes/frame",
                threshold=0.2,
                threshold_mode="cascade_ap",
                expected_spike_count=0.2,
                spike_active=False,
            ),
        ],
    )
    fov = FOVAnalysis(
        id=40,
        fov_id=7,
        analysis_result_id=31,
        calcium_active_roi_labels=[2, 1],
        calcium_dff_correlation_matrix=[[1, 0.4], [0.4, 1]],
        spike_analyses=[
            SpikeFOVAnalysis(
                id=41,
                fov_analysis_id=40,
                fov_id=7,
                analysis_result_id=31,
                spike_inference_run_id=4,
                inference_run=oasis,
                active_roi_labels=[2, 1],
                spike_ccg_zscore_matrix=[[float("inf"), 2.5], [3.1, float("inf")]],
                spike_burst_count=2,
            ),
            SpikeFOVAnalysis(
                id=42,
                fov_analysis_id=40,
                fov_id=7,
                analysis_result_id=31,
                spike_inference_run_id=6,
                inference_run=cascade,
                method="cascade",
                units="spikes/frame",
                active_roi_labels=[3],
                spike_burst_count=5,
            ),
        ],
    )
    return trace, analysis, fov


@pytest.mark.parametrize("index", [0, 1, 2])
def test_dual_product_json_round_trip(index: int) -> None:
    original = _products()[index]
    payload = original.model_dump_json()
    restored = type(original).model_validate_json(payload)
    assert restored.model_dump() == original.model_dump()
    assert json.loads(restored.model_dump_json()) == json.loads(payload)
    assert "roi" not in json.loads(payload) and "analysis_result" not in json.loads(
        payload
    )
    with pytest.raises(ValueError, match="Multiple spike"):
        _ = (
            restored.inferred_spikes
            if isinstance(restored, Traces)
            else restored.inferred_spikes_threshold
            if isinstance(restored, DataAnalysis)
            else restored.spike_burst_count
        )


def test_trace_json_preserves_window_valid_interval_and_model_identity() -> None:
    original, _, _ = _products()
    restored = Traces.model_validate_json(original.model_dump_json())
    assert restored.source_start_frame == 1
    assert restored.extraction_frame_window.source_time_origin_ms == 2000
    assert restored.extraction_frame_window.source_start_timestamp_ms == 2100
    child = restored.get_spike_trace("cascade")
    assert child.values == [0, 0.2, 0] and (child.valid_start, child.valid_stop) == (
        1,
        2,
    )
    assert child.inference_run.resolved_model == "verified-model"
    assert child.inference_run.weights_manifest_sha256 == "cd" * 32
    assert restored.get_spike_trace("oasis").ar_coefficients == [0.8]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_object_validation_clones_owned_relations_without_reparenting(
    index: int,
) -> None:
    products = _products()
    original = products[index]
    before = original.model_dump()
    clone = type(original).model_validate(original)
    assert clone.model_dump() == before
    assert original.model_dump() == before
    if isinstance(original, Traces):
        assert original.spike_traces[0].trace is original
        assert clone.spike_traces[0].trace is clone
        clone.spike_traces[0].values[1] = 9
        assert original.spike_traces[0].values[1] == 0.75
        assert (
            clone.spike_traces[0].inference_run
            is not original.spike_traces[0].inference_run
        )
        assert products[1].spike_analyses[0].spike_trace is original.spike_traces[0]
    else:
        assert clone.spike_analyses[0] is not original.spike_analyses[0]
        if isinstance(original, DataAnalysis):
            assert original.spike_analyses[0].data_analysis is original
            clone.spike_analyses[0].spike_trace.values[1] = 9
            assert products[0].spike_traces[0].values[1] == 0.75
        else:
            assert original.spike_analyses[0].fov_analysis is original
            clone.spike_analyses[0].inference_run.backend_version = "changed"
            assert products[0].spike_traces[0].inference_run.backend_version == "1.2.3"


@pytest.mark.parametrize(
    "model, legacy",
    [
        (Traces, {"inferred_spikes": [0, 0.75, 0], "source_start_frame": 2}),
        (
            DataAnalysis,
            {"inferred_spikes_threshold": 0.5, "inferred_spikes_frequency": 0.3},
        ),
        (FOVAnalysis, {"active_roi_labels": [1, 2], "spike_burst_count": 3}),
    ],
)
def test_legacy_json_becomes_explicit_synthetic_normalized_data(
    model: type, legacy: dict
) -> None:
    restored = model.model_validate_json(json.dumps(legacy))
    second = model.model_validate_json(restored.model_dump_json())
    assert second.model_dump() == restored.model_dump()
    if isinstance(restored, Traces):
        assert restored.source_start_frame == 2
        assert (
            restored.get_spike_trace("oasis").inference_run.provenance_source
            == "synthetic_legacy_api"
        )
    else:
        assert (
            restored.get_spike_analysis("oasis").provenance_source
            == "synthetic_legacy_api"
        )


def test_explicit_missing_provenance_and_quarantined_metrics_stay_missing() -> None:
    trace = Traces.model_validate(
        {"raw_trace": [1, 2], "extraction_frame_window": None}
    )
    assert trace.extraction_frame_window is None
    assert (
        Traces.model_validate_json(trace.model_dump_json()).extraction_frame_window
        is None
    )
    parent = DataAnalysis(
        spike_analyses=[
            SpikeAnalysis(threshold=0.4, provenance_source="legacy_unresolved")
        ]
    )
    restored = DataAnalysis.model_validate_json(parent.model_dump_json())
    assert restored.get_spike_analysis("oasis").spike_trace is None
    assert restored.get_spike_analysis("oasis").provenance_source == "legacy_unresolved"


@pytest.mark.parametrize("index", [0, 1, 2])
def test_duplicate_json_methods_are_rejected(index: int) -> None:
    original = _products()[index]
    payload = original.model_dump()
    key = "spike_traces" if index == 0 else "spike_analyses"
    payload[key].append(dict(payload[key][0]))
    with pytest.raises(ValueError, match="Duplicate spike methods"):
        type(original).model_validate(payload)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("units", "a.u.", "units must match"),
        ("threshold_mode", "multiplier", "incompatible"),
        ("suprathreshold_sample_rate_hz", 2, "metrics are incompatible"),
        ("threshold", -1, "non-negative"),
    ],
)
def test_json_cannot_relabel_cascade_metrics(
    field: str, value: object, message: str
) -> None:
    payload = _products()[1].model_dump()
    # Unlink just the optional cached source so the method-specific validation
    # itself is exercised, independently of matching inference provenance.
    payload["spike_analyses"][1]["spike_trace"] = None
    payload["spike_analyses"][1][field] = value
    with pytest.raises(ValueError, match=message):
        DataAnalysis.model_validate(payload)


def test_json_rejects_conflicting_ids_and_invalid_valid_interval() -> None:
    payload = _products()[0].model_dump()
    payload["spike_traces"][0]["spike_inference_run_id"] = 99
    with pytest.raises(ValueError, match="foreign-key IDs"):
        Traces.model_validate(payload)
    payload = _products()[0].model_dump()
    payload["spike_traces"][1]["valid_stop"] = 10
    with pytest.raises(ValueError, match="valid interval"):
        Traces.model_validate(payload)
    payload = _products()[1].model_dump()
    payload["spike_analyses"][0]["data_analysis_id"] = 99
    with pytest.raises(ValueError, match="child and parent IDs"):
        DataAnalysis.model_validate(payload)


def test_persisted_detached_trace_json_can_merge_back_without_duplicates(
    tmp_path: Path,
) -> None:
    engine = create_cali_engine(f"sqlite:///{tmp_path / 'roundtrip.cali'}")
    create_database_and_tables(engine)
    with Session(engine) as session:
        trace = Traces(raw_trace=[1, 2, 1], inferred_spikes=[0, 0.5, 0])
        session.add(trace)
        session.commit()
        trace_id = trace.id
    with Session(engine) as session:
        original = session.get(Traces, trace_id)
    restored = Traces.model_validate_json(original.model_dump_json())
    with Session(engine) as session:
        session.merge(restored)
        session.commit()
        assert len(session.exec(select(Traces)).all()) == 1
        assert len(session.exec(select(SpikeTrace)).all()) == 1
        assert len(session.exec(select(SpikeInferenceRun)).all()) == 1
        assert session.get(Traces, trace_id).get_spike_values("oasis") == [0, 0.5, 0]
    engine.dispose()


def test_json_include_exclude_options_apply_to_normalized_relationships() -> None:
    trace, _, _ = _products()
    assert "spike_traces" not in trace.model_dump(exclude={"spike_traces"})
    selected = trace.model_dump(include={"spike_traces": {"__all__": {"values"}}})
    assert selected == {
        "spike_traces": [{"values": [0, 0.75, 0]}, {"values": [0, 0.2, 0]}]
    }
    filtered = trace.model_dump(exclude={"spike_traces": {0: {"noise"}}})
    assert "noise" not in filtered["spike_traces"][0]
    assert "noise" in filtered["spike_traces"][1]
