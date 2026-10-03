"""Lossless storage, checked metadata, transactional migration and consumer reads."""

import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, event
from sqlmodel import Session, select

from cali._constants import CASCADE_EXPECTED_SPIKES_TRACES, INFERRED_SPIKES_TRACES
from cali.sqlmodel import (
    FOV,
    ROI,
    CaliResult,
    Experiment,
    SpikeInferenceRun,
    SpikeTrace,
    Traces,
    _engine,
    create_cali_engine,
    create_database_and_tables,
    ensure_schema_current,
)
from cali.sqlmodel import _trace_array_codec as codec
from cali.sqlmodel._trace_array_codec import (
    TraceArrayError,
    decode_trace_array,
    encode_trace_array,
    trace_array_storage_info,
)
from cali.util._database_to_csv import export_traces_to_csv


def _parts(payload: bytes) -> tuple[dict, bytes]:
    length = struct.unpack(">I", payload[8:12])[0]
    return json.loads(payload[12 : 12 + length]), payload[12 + length :]


def _replace_header(payload: bytes, **updates: object) -> bytes:
    header, compressed = _parts(payload)
    header.update(updates)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return payload[:8] + struct.pack(">I", len(encoded)) + encoded + compressed


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0.0, -0.0, 0.5],
        [0, 0.1, np.nextafter(0.1, 1)],
        [np.finfo(float).max, np.nextafter(0.0, 1.0)],
    ],
)
def test_roundtrip_preserves_all_double_bits(values: list) -> None:
    encoded = encode_trace_array(values)
    decoded = decode_trace_array(encoded)
    expected = np.asarray(values, dtype="<f8")
    np.testing.assert_array_equal(
        np.asarray(decoded, dtype="<f8").view("<u8"), expected.view("<u8")
    )
    assert decode_trace_array(memoryview(encoded)) == decoded
    assert isinstance(decoded, list)


def test_float32_golden_bytes_and_metrics_remain_exact() -> None:
    with np.load(
        Path(__file__).parent / "fixtures/cascade_reference/real_excerpt.npz"
    ) as fixture:
        expected = fixture["expected_spikes"].astype("<f4")
    for row in expected:
        payload = encode_trace_array(row.tolist())
        metadata, compressed = _parts(payload)
        assert metadata["dtype"] == "<f4" and metadata["shape"] == [256]
        raw = zlib.decompress(compressed)
        assert raw == row.tobytes()
        identity = {key: value for key, value in metadata.items() if key != "sha256"}
        digest = hashlib.sha256(
            b"CALIARR\x00"
            + json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
            + raw
        ).hexdigest()
        assert metadata["sha256"] == digest
        decoded = np.asarray(decode_trace_array(payload))
        np.testing.assert_array_equal(decoded, row.astype(float))
        assert decoded[32:224].sum() == row[32:224].sum(dtype=float)
        for fraction in (0.1, 0.2, 0.5, 1):
            threshold = fraction / (math.sqrt(2 * math.pi) * 0.025 * 30)
            np.testing.assert_array_equal(
                decoded > threshold, row.astype(float) > threshold
            )


def test_imported_double_precision_is_not_quantized() -> None:
    values = [0.1, np.nextafter(0.5, 1), -np.nextafter(0, 1)]
    payload = encode_trace_array(values)
    assert trace_array_storage_info(payload)["dtype"] == "<f8"
    assert decode_trace_array(payload) == values
    # A caller can supply float64 CASCADE imports: preserve them exactly, too.
    assert decode_trace_array(json.dumps(values)) == values


def test_historical_nonfinite_json_numbers_are_preserved() -> None:
    values = decode_trace_array("[0.0, -0.0, NaN, Infinity, -Infinity]")
    payload = encode_trace_array(values)
    assert trace_array_storage_info(payload)["dtype"] == "<f8"
    decoded = decode_trace_array(payload)
    assert math.copysign(1, decoded[1]) == -1
    assert math.isnan(decoded[2]) and decoded[3:] == [math.inf, -math.inf]


@pytest.mark.parametrize("values", [[True], [[1, 2]], ["1"], [None], [2**60 + 1]])
def test_non_numeric_or_lossy_input_is_rejected(values: list) -> None:
    with pytest.raises(TraceArrayError):
        encode_trace_array(values)
    with pytest.raises(TraceArrayError):
        decode_trace_array(json.dumps(values))


@pytest.mark.parametrize(
    "updates",
    [
        {"version": 2},
        {"version": True},
        {"compression": "pickle"},
        {"dtype": "O"},
        {"dtype": ["<f4"]},
        {"shape": [1, 3]},
        {"shape": [True]},
        {"shape": [-1]},
        {"shape": [2**40]},
        {"sha256": "0" * 64},
        {"unexpected": 1},
    ],
)
def test_invalid_metadata_is_rejected_before_array_use(updates: dict) -> None:
    with pytest.raises(TraceArrayError):
        decode_trace_array(_replace_header(encode_trace_array([0, 0.5, 0]), **updates))


def test_checksum_covers_dtype_shape_and_samples() -> None:
    original = encode_trace_array([0.5, 0.25])
    assert trace_array_storage_info(original)["dtype"] == "<f4"
    # Same uncompressed byte count, different interpretation: still corruption.
    with pytest.raises(TraceArrayError, match="checksum"):
        decode_trace_array(_replace_header(original, dtype="<f8", shape=[1]))
    _metadata, compressed = _parts(original)
    raw = bytearray(zlib.decompress(compressed))
    raw[0] ^= 1
    header_size = struct.unpack(">I", original[8:12])[0]
    corrupted = original[: 12 + header_size] + zlib.compress(raw)
    with pytest.raises(TraceArrayError, match="checksum"):
        decode_trace_array(corrupted)


@pytest.mark.parametrize(
    "variation",
    [
        "cut_header",
        "cut_stream",
        "bad_magic",
        "trailing",
        "second_stream",
        "bad_length",
        "duplicate_key",
        "wrong_shape",
        "excess_data",
    ],
)
def test_corrupt_and_noncanonical_payloads_fail(variation: str) -> None:
    payload = encode_trace_array([0, 0.5, 0])
    header, compressed = _parts(payload)
    if variation == "cut_header":
        payload = payload[:20]
    elif variation == "cut_stream":
        payload = payload[:-1]
    elif variation == "bad_magic":
        payload = b"X" + payload[1:]
    elif variation == "trailing":
        payload += b"ignored"
    elif variation == "second_stream":
        payload += zlib.compress(b"other")
    elif variation == "bad_length":
        payload = payload[:8] + struct.pack(">I", 4097) + payload[12:]
    elif variation == "duplicate_key":
        encoded = json.dumps(header).encode()[:-1] + b',"version":1}'
        payload = payload[:8] + struct.pack(">I", len(encoded)) + encoded + compressed
    elif variation == "wrong_shape":
        payload = _replace_header(payload, shape=[2])
    else:
        length = struct.unpack(">I", payload[8:12])[0]
        payload = payload[: 12 + length] + zlib.compress(b"\x00" * 10000)
    with pytest.raises(TraceArrayError):
        decode_trace_array(payload)


def test_encoding_limit_is_checked_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codec, "_MAX_RAW_BYTES", 16)
    with pytest.raises(TraceArrayError, match="64 MiB"):
        encode_trace_array([0, 0, 0])


def _version_ten(path: Path, payloads: list[str | bytes]) -> None:
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE spike_trace (id INTEGER PRIMARY KEY,"
            '"values" JSON NOT NULL, noise FLOAT)'
        )
        connection.exec_driver_sql(
            "CREATE TABLE trace (id INTEGER PRIMARY KEY,inferred_spikes JSON)"
        )
        connection.exec_driver_sql("INSERT INTO trace VALUES (1,'[0,0.1,0]')")
        connection.exec_driver_sql(
            "CREATE TABLE spike_inference_run (id INTEGER PRIMARY KEY,"
            "backend_version VARCHAR,dtype VARCHAR)"
        )
        connection.exec_driver_sql(
            "INSERT INTO spike_inference_run VALUES (1,NULL,NULL)"
        )
        for index, payload in enumerate(payloads):
            connection.exec_driver_sql(
                "INSERT INTO spike_trace VALUES (?,?,?)", (index - 2, payload, 0.123)
            )
        connection.exec_driver_sql("PRAGMA user_version=10")
    engine.dispose()


def test_migration_streams_all_rows_and_preserves_legacy_and_unknown_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v10.cali"
    original = [[0, index / 13, 0.25] for index in range(70)]
    _version_ten(
        path, [json.dumps(row) for row in original] + [encode_trace_array([0, 0.1, 0])]
    )
    engine = create_cali_engine(f"sqlite:///{path}")
    try:
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 11
            rows = connection.exec_driver_sql(
                'SELECT typeof("values"),"values",noise FROM spike_trace ORDER BY id'
            ).all()
            assert [decode_trace_array(row[1]) for row in rows] == [
                *original,
                [0, 0.1, 0],
            ]
            assert all(row[0] == "blob" and row[2] == 0.123 for row in rows)
            assert (
                connection.exec_driver_sql("SELECT inferred_spikes FROM trace").scalar()
                == "[0,0.1,0]"
            )
            assert connection.exec_driver_sql(
                "SELECT backend_version,dtype FROM spike_inference_run"
            ).one() == (None, None)
            before = rows
        with patch.object(_engine, "_MIGRATIONS", ()):
            ensure_schema_current(engine)
        with engine.connect() as connection:
            assert (
                connection.exec_driver_sql(
                    'SELECT typeof("values"),"values",noise '
                    "FROM spike_trace ORDER BY id"
                ).all()
                == before
            )
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "bad_payload", ["null", "[true]", "broken", encode_trace_array([0])[:-1]]
)
def test_invalid_row_rolls_back_previous_arrays_and_version(
    tmp_path: Path, bad_payload: str | bytes
) -> None:
    path = tmp_path / "bad.cali"
    first = "[0,0.1,0]"
    _version_ten(path, [first, bad_payload])
    engine = create_engine(f"sqlite:///{path}")
    try:
        with pytest.raises(ValueError, match="Spike trace -1"):
            ensure_schema_current(engine)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 10
            assert connection.exec_driver_sql(
                'SELECT "values" FROM spike_trace ORDER BY id'
            ).scalars().all() == [first, bad_payload]
        with engine.begin() as connection:
            connection.exec_driver_sql(
                'UPDATE spike_trace SET "values"=? WHERE id=-1', ("[1,2,3]",)
            )
        ensure_schema_current(engine)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 11
    finally:
        engine.dispose()


def test_interrupted_conversion_is_atomic_and_retryable(tmp_path: Path) -> None:
    path = tmp_path / "interrupted.cali"
    values = ["[0,0.1,0]", "[1,2,3]"]
    _version_ten(path, values)
    engine = create_engine(f"sqlite:///{path}")

    def interrupt(*args: object) -> None:
        if str(args[2]).startswith("UPDATE spike_trace"):
            raise RuntimeError("interrupted codec write")

    try:
        event.listen(engine, "after_cursor_execute", interrupt)
        with pytest.raises(RuntimeError, match="interrupted codec"):
            ensure_schema_current(engine)
        event.remove(engine, "after_cursor_execute", interrupt)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 10
            assert (
                connection.exec_driver_sql(
                    'SELECT "values" FROM spike_trace ORDER BY id'
                )
                .scalars()
                .all()
                == values
            )
        ensure_schema_current(engine)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar() == 11
    finally:
        engine.dispose()


@pytest.mark.parametrize("methods", [("oasis",), ("cascade",), ("oasis", "cascade")])
def test_orm_json_snapshots_csv_and_mixed_legacy_reads(
    tmp_path: Path, methods: tuple
) -> None:
    path = tmp_path / "consumer.cali"
    engine = create_cali_engine(f"sqlite:///{path}")
    create_database_and_tables(engine)
    values = np.array([0, 0.25, 0.125, 0], dtype=np.float32).astype(float).tolist()
    try:
        with Session(engine) as session:
            experiment = Experiment(name="codec consumer")
            fov = FOV(name="A1_0000", position_index=0, rois=[ROI(label_value=1)])
            session.add_all([experiment, fov])
            session.flush()
            result = CaliResult(experiment=experiment.id)
            session.add(result)
            session.flush()
            trace = Traces(
                roi_id=fov.rois[0].id,
                analysis_result_id=result.id,
                raw_trace=[1, 2, 3, 4],
                dff=values,
                den_dff=values,
                spike_traces=[
                    SpikeTrace(
                        values=values,
                        valid_start=1,
                        valid_stop=3,
                        inference_run=SpikeInferenceRun(
                            method=method,
                            units="spikes/frame" if method == "cascade" else "a.u.",
                            backend_package="cascade2p"
                            if method == "cascade"
                            else "oasis-deconv",
                        ),
                    )
                    for method in methods
                ],
            )
            session.add(trace)
            session.commit()
            run_id = result.id
            session.expunge_all()
            reloaded = session.exec(select(Traces)).one()
            for method in methods:
                assert reloaded.get_spike_values(method) == values
            snapshot = Traces.model_validate_json(reloaded.model_dump_json())
            assert [child.values for child in snapshot.spike_traces] == [values] * len(
                methods
            )
            if len(methods) == 2:
                with pytest.raises(ValueError, match="Multiple spike outputs"):
                    _ = reloaded.inferred_spikes
            else:
                assert reloaded.inferred_spikes == values
        with engine.begin() as connection:
            blobs = connection.exec_driver_sql(
                'SELECT id,typeof("values"),"values" FROM spike_trace'
            ).all()
            assert all(row[1] == "blob" for row in blobs)
            identifier = blobs[0][0]
            connection.exec_driver_sql(
                'UPDATE spike_trace SET "values"=? WHERE id=?',
                (json.dumps(values), identifier),
            )
        with Session(engine) as session:
            assert session.get(SpikeTrace, identifier).values == values
            # Replacing a legacy list writes the codec through the same public API.
            child = session.get(SpikeTrace, identifier)
            child.values = [*values, 0.0625]
            child.valid_stop = 4
            session.commit()
            child.values = values
            child.valid_stop = 3
            session.commit()
        export_traces_to_csv(
            engine,
            {
                INFERRED_SPIKES_TRACES: "oasis" in methods,
                CASCADE_EXPECTED_SPIKES_TRACES: "cascade" in methods,
            },
            run_id,
            path,
        )
        target = tmp_path / "consumer_exports" / f"run_{run_id}"
        if "cascade" in methods:
            exported = pd.read_csv(target / "cascade_expected_spikes.csv")
            samples = exported.iloc[:, 0].to_numpy(dtype=float)
            assert np.isnan(samples[0]) and np.isnan(samples[-1])
            np.testing.assert_array_equal(samples[1:3], values[1:3])
        metadata = json.loads((target / "trace_metadata.json").read_text())
        assert metadata is not None
    finally:
        engine.dispose()
