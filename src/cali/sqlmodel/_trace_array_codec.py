"""Versioned, lossless spike-array storage with backward-compatible JSON reads.

SQLite permits BLOBs in existing JSON-affinity columns, so the public values field
and its consumers remain unchanged. Schema 11 gates v1 BLOBs; schema 13 gates the
v2 byte-shuffle option. Checksums cover metadata and original uncompressed bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy import LargeBinary
from sqlalchemy.types import TypeDecorator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.engine import Dialect

_MAGIC = b"CALIARR\x00"
_PREFIX_SIZE = len(_MAGIC) + 4
_MAX_HEADER_BYTES = 4096
_MAX_RAW_BYTES = 64 * 1024**2
_DTYPES = {"<f4": 4, "<f8": 8}


class TraceArrayError(ValueError):
    """An array cannot be encoded losslessly or its stored payload is invalid."""


def _numeric_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    if (isinstance(values, np.ndarray) and values.ndim != 1) or len(
        values
    ) > _MAX_RAW_BYTES // 8:
        raise TraceArrayError("Trace must be one-dimensional and at most 64 MiB.")
    for value in values:
        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
            raise TraceArrayError("Trace values must be a flat array of real numbers.")
        if isinstance(value, Integral):
            try:
                if int(float(value)) != value:
                    raise TraceArrayError("Trace integer is not lossless as float64.")
            except OverflowError as error:
                raise TraceArrayError("Trace integer exceeds float64 range.") from error
        elif isinstance(value, np.floating) and value.dtype.itemsize > 8:
            if not math.isnan(value) and value != float(value):
                raise TraceArrayError("Trace value is not lossless as float64.")
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1 or array.nbytes > _MAX_RAW_BYTES:
        raise TraceArrayError("Trace must be one-dimensional and at most 64 MiB.")
    return array


def _metadata_bytes(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")


def _checksum(metadata: dict[str, Any], raw: bytes) -> str:
    digest = hashlib.sha256(_MAGIC + _metadata_bytes(metadata))
    digest.update(raw)
    return digest.hexdigest()


def encode_trace_array(
    values: Sequence[float] | np.ndarray, *, version: int = 2
) -> bytes:
    """Encode losslessly, choosing float32 only for exact round trips.

    Existing float32 CASCADE output remains float32. OASIS/imported doubles stay
    float64 whenever float32 would change even one sample. Nonfinite historical
    JSON numbers retain float64; encoding does not invent new scientific values.
    """
    array = _numeric_array(values)
    if np.all(np.isfinite(array)):
        with np.errstate(over="ignore", invalid="ignore"):
            compact = array.astype("<f4")
        if np.array_equal(compact.astype("<f8"), array):
            array = compact
    return _encode_array(array, version=version)


def _encode_array(array: np.ndarray, *, version: int) -> bytes:
    """Retain the original dtype/bits, selecting the smaller full v2 payload."""
    if type(version) is not int or version not in (1, 2):
        raise TraceArrayError("Unsupported trace-array codec version.")
    raw = array.tobytes()
    sources = [("zlib", raw)]
    if version == 2:
        shuffled = (
            np.frombuffer(raw, dtype="u1")
            .reshape(-1, array.dtype.itemsize)
            .T.copy()
            .tobytes()
        )
        sources.append(("zlib-byte-shuffle", shuffled))
    choices = []
    for compression, source in sources:
        metadata: dict[str, Any] = {
            "version": version,
            "compression": compression,
            "dtype": array.dtype.str,
            "shape": [len(array)],
        }
        metadata["sha256"] = _checksum(metadata, raw)
        header = _metadata_bytes(metadata)
        choices.append(
            _MAGIC
            + struct.pack(">I", len(header))
            + header
            + zlib.compress(source, level=6)
        )
    return min(choices, key=len)


def _unique_metadata(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in pairs:
        if key in metadata:
            raise TraceArrayError("Duplicate trace metadata key.")
        metadata[key] = value
    return metadata


def _decode_blob(payload: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    if not payload.startswith(_MAGIC) or len(payload) < _PREFIX_SIZE:
        raise TraceArrayError("Invalid trace-array signature.")
    header_size = struct.unpack(">I", payload[len(_MAGIC) : _PREFIX_SIZE])[0]
    header_stop = _PREFIX_SIZE + header_size
    if not 0 < header_size <= _MAX_HEADER_BYTES or header_stop >= len(payload):
        raise TraceArrayError("Invalid trace-array header length.")
    try:
        metadata = json.loads(
            payload[_PREFIX_SIZE:header_stop], object_pairs_hook=_unique_metadata
        )
    except (ValueError, UnicodeError) as error:
        raise TraceArrayError("Invalid trace-array metadata.") from error
    if not isinstance(metadata, dict) or set(metadata) != {
        "version",
        "compression",
        "dtype",
        "shape",
        "sha256",
    }:
        raise TraceArrayError("Invalid trace-array metadata fields.")
    if type(metadata["version"]) is not int or metadata["version"] not in (1, 2):
        raise TraceArrayError("Unsupported trace-array codec version.")
    dtype = metadata["dtype"]
    shape = metadata["shape"]
    checksum = metadata["sha256"]
    if (
        metadata["compression"]
        not in (
            ("zlib",) if metadata["version"] == 1 else ("zlib", "zlib-byte-shuffle")
        )
        or not isinstance(dtype, str)
        or dtype not in _DTYPES
        or not isinstance(shape, list)
        or len(shape) != 1
        or type(shape[0]) is not int
        or shape[0] < 0
        or shape[0] > _MAX_RAW_BYTES // 8
        or not isinstance(checksum, str)
        or len(checksum) != 64
    ):
        raise TraceArrayError("Invalid trace-array dtype, shape or checksum.")
    expected_bytes = shape[0] * _DTYPES[dtype]
    if expected_bytes > _MAX_RAW_BYTES:
        raise TraceArrayError("Trace-array decoded size exceeds 64 MiB.")
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(payload[header_stop:], expected_bytes + 1)
    except zlib.error as error:
        raise TraceArrayError("Invalid compressed trace-array payload.") from error
    if (
        len(raw) != expected_bytes
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise TraceArrayError("Trace-array payload length does not match its shape.")
    if metadata["compression"] == "zlib-byte-shuffle":
        raw = (
            np.frombuffer(raw, dtype="u1")
            .reshape(_DTYPES[dtype], shape[0])
            .T.copy()
            .tobytes()
        )
    checked_metadata = {
        key: value for key, value in metadata.items() if key != "sha256"
    }
    if _checksum(checked_metadata, raw) != checksum:
        raise TraceArrayError("Trace-array checksum mismatch.")
    return np.frombuffer(raw, dtype=dtype), metadata


def decode_trace_array(payload: str | bytes | memoryview) -> list[float]:
    """Read a checked v1/v2 BLOB or a historical numeric JSON array."""
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if isinstance(payload, bytes) and payload.startswith(_MAGIC):
        array, _ = _decode_blob(payload)
        return list(array.tolist())
    try:
        values = json.loads(payload)
    except (TypeError, ValueError, UnicodeError) as error:
        raise TraceArrayError("Invalid legacy trace-array JSON or BLOB.") from error
    if not isinstance(values, list):
        raise TraceArrayError("Legacy trace JSON must contain a numeric array.")
    return list(_numeric_array(values).tolist())


def _transcode_trace_array(payload: str | bytes, *, version: int) -> bytes:
    """Upgrade BLOBs without converting dtype or any original sample bits."""
    if isinstance(payload, bytes) and payload.startswith(_MAGIC):
        array, _ = _decode_blob(payload)
        return _encode_array(array, version=version)
    return encode_trace_array(decode_trace_array(payload), version=version)


def trace_array_storage_info(payload: str | bytes) -> dict[str, Any]:
    """Inspect storage metadata after validating its complete contents."""
    if isinstance(payload, bytes) and payload.startswith(_MAGIC):
        _, metadata = _decode_blob(payload)
        return metadata
    return {
        "version": 0,
        "encoding": "json",
        "shape": [len(decode_trace_array(payload))],
    }


class TraceArrayType(TypeDecorator[list[float]]):
    """Keep every ORM consumer on one checked JSON/BLOB decoding boundary."""

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> bytes:
        """Write new arrays using the lossless v2 BLOB representation."""
        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TraceArrayError("Trace values must be a numeric array.")
        return encode_trace_array(value)

    def process_result_value(self, value: Any, dialect: Dialect) -> list[float]:
        """Decode current BLOBs and unconverted historical JSON rows."""
        return decode_trace_array(value)
