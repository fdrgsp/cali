"""Developer GPU acceptance must reject changed decisions and corrupted inputs."""

from __future__ import annotations

import copy
import importlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Developer benchmark uses Unix ps/resource APIs."
)


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "_dev"))
    return importlib.import_module("benchmark_cascade_extraction")


def products() -> dict:
    return {
        "roi": [
            {
                "expected_spike_count": 100.0,
                "expected_spike_rate_hz": 1.0,
                "threshold": 0.1,
                "spike_active": True,
                "suprathreshold_excursion_rate_hz": 0.5,
            }
        ],
        "fov": {
            "active_roi_labels": [2, 11],
            "valid_start": 32,
            "valid_stop": 96,
            "spike_burst_starts": [45],
            "spike_max_lag_values_matrix": [[0, 1], [-1, 0]],
            "spike_population_activity_raw": [0.0, 50.0],
        },
    }


def test_continuous_gpu_rounding_requires_explicit_tolerance(benchmark: object) -> None:
    expected = products()
    actual = copy.deepcopy(expected)
    actual["roi"][0]["expected_spike_count"] += 1e-6
    differences = benchmark.compare_spike_products(actual, expected, tolerant=True)
    assert 0 < differences["roi/0/expected_spike_count"] < 1.1e-6
    with pytest.raises(AssertionError):
        benchmark.compare_spike_products(actual, expected, tolerant=False)
    actual["roi"][0]["expected_spike_count"] += 0.1
    with pytest.raises(AssertionError):
        benchmark.compare_spike_products(actual, expected, tolerant=True)


@pytest.mark.parametrize(
    ("owner", "key", "replacement"),
    [
        ("roi", "threshold", 0.10000001),
        ("roi", "spike_active", False),
        ("roi", "suprathreshold_excursion_rate_hz", 0.50000001),
        ("fov", "active_roi_labels", [11, 2]),
        ("fov", "valid_stop", 97),
        ("fov", "spike_burst_starts", [46]),
        ("fov", "spike_max_lag_values_matrix", [[0, 2], [-2, 0]]),
        ("fov", "spike_population_activity_raw", [0.0, 50.00000001]),
    ],
)
def test_gpu_tolerance_cannot_hide_changed_decisions(
    benchmark: object, owner: str, key: str, replacement: object
) -> None:
    expected = products()
    actual = copy.deepcopy(expected)
    (actual["roi"][0] if owner == "roi" else actual["fov"])[key] = replacement
    with pytest.raises(AssertionError):
        benchmark.compare_spike_products(actual, expected, tolerant=True)


@pytest.mark.parametrize(
    "actual", [np.ones((2, 1)), np.array([np.nan, 1]), np.array([np.inf, 1])]
)
def test_gpu_array_parity_rejects_broadcasting_and_nonfinite_values(
    benchmark: object, actual: np.ndarray
) -> None:
    with pytest.raises(AssertionError):
        benchmark.compare_arrays(actual, np.ones(2), tolerant=True)


def test_gpu_parity_rejects_missing_metrics_and_rois(benchmark: object) -> None:
    expected = products()
    actual = copy.deepcopy(expected)
    del actual["roi"][0]["spike_active"]
    with pytest.raises(AssertionError):
        benchmark.compare_spike_products(actual, expected, tolerant=True)
    with pytest.raises(AssertionError):
        benchmark.compare_spike_products(
            {"roi": [], "fov": expected["fov"]}, expected, tolerant=True
        )


@pytest.fixture
def stored_sources(benchmark: object, tmp_path: Path) -> tuple:
    """Keep a sample exactly on the threshold to catch tolerance masking events."""
    from cali.sqlmodel._trace_array_codec import encode_trace_array

    module = importlib.import_module("audit_cascade_device_parity")
    cpu, gpu = tmp_path / "cpu.cali", tmp_path / "gpu.cali"
    with closing(sqlite3.connect(cpu)) as connection:
        connection.executescript(
            "CREATE TABLE fov(id,position_index);"
            "CREATE TABLE roi(id,fov_id,label_value);"
            "CREATE TABLE trace(id,roi_id,raw_trace,dff,den_dff,x_axis,calcium_noise);"
            "CREATE TABLE spike_trace(id,trace_id,spike_inference_run_id,noise,"
            'selected_noise_level,valid_start,valid_stop,"values");'
            "CREATE TABLE spike_inference_run(id,method,resolved_device,"
            "resolved_model,weights_manifest_sha256);"
            "CREATE TABLE spike_analysis(spike_trace_id,threshold,spike_active);"
            "CREATE TABLE extraction_frame_window(id,extraction_result_id,fov_id,"
            "legacy_owner_result_id,source_start_frame);"
            "INSERT INTO fov VALUES(1,7);"
            "INSERT INTO roi VALUES(1,1,11);"
            "INSERT INTO spike_inference_run "
            "VALUES(1,'cascade','cpu','model','manifest');"
            "INSERT INTO spike_analysis VALUES(1,0.15,1);"
            "INSERT INTO extraction_frame_window VALUES(1,1,1,NULL,0);"
        )
        array = json.dumps([0.0, 1.0, 2.0, 3.0])
        connection.execute("INSERT INTO trace VALUES(1,1,?,?,?,?,0.1)", (array,) * 4)
        connection.execute(
            "INSERT INTO spike_trace VALUES(1,1,1,2,2,1,3,?)",
            (encode_trace_array([0.0, 0.15, 0.16, 0.0]),),
        )
        connection.commit()
        with closing(sqlite3.connect(gpu)) as target:
            connection.backup(target)
            target.execute("UPDATE spike_inference_run SET resolved_device='mps'")
            target.commit()
    return module, cpu, gpu


def test_source_audit_accepts_rounding_without_changed_events(
    stored_sources: tuple,
) -> None:
    from cali.sqlmodel._trace_array_codec import encode_trace_array

    module, cpu, gpu = stored_sources
    with closing(sqlite3.connect(gpu)) as connection:
        connection.execute(
            'UPDATE spike_trace SET "values"=?',
            (encode_trace_array([0.0, 0.15, 0.16000001, 0.0]),),
        )
        connection.commit()
    assert module.compare_sources(cpu, gpu, device="mps")["trace_method_rows"] == 1


def test_source_audit_rejects_rounding_that_changes_threshold_crossings(
    stored_sources: tuple,
) -> None:
    from cali.sqlmodel._trace_array_codec import encode_trace_array

    module, cpu, gpu = stored_sources
    with closing(sqlite3.connect(gpu)) as connection:
        connection.execute(
            'UPDATE spike_trace SET "values"=?',
            (encode_trace_array([0.0, 0.15000001, 0.16, 0.0]),),
        )
        connection.commit()
    with pytest.raises(AssertionError):
        module.compare_sources(cpu, gpu, device="mps")


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE trace SET dff='[0,1,2,4]'",
        "UPDATE spike_trace SET noise=2.01",
        "UPDATE spike_inference_run SET resolved_device='cpu'",
        "UPDATE extraction_frame_window SET source_start_frame=1",
    ],
)
def test_source_audit_rejects_wrong_inputs_or_provenance(
    stored_sources: tuple, statement: str
) -> None:
    module, cpu, gpu = stored_sources
    with closing(sqlite3.connect(gpu)) as connection:
        connection.execute(statement)
        connection.commit()
    with pytest.raises(AssertionError):
        module.compare_sources(cpu, gpu, device="mps")
