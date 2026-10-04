"""Evaluate a byte-shuffle candidate without modifying production databases.

The proposed v2 representation is NOT accepted by cali's current v1 reader.
Measure full payloads, including proposed metadata/checksums, on saved benchmark
NPZ samples. A versioned reader/migration is required before using these bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np

from cali.sqlmodel._trace_array_codec import decode_trace_array, encode_trace_array

MAGIC = b"CALIARR\x00"
PREFIX_SIZE = len(MAGIC) + 4


def metadata_bytes(metadata: dict) -> bytes:
    """Match the production metadata canonicalization for the proposed version."""
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")


def candidate(payload: bytes) -> tuple[bytes, str]:
    """Preserve every original sample bit, selecting the smaller v2 encoding."""
    decode_trace_array(payload)  # Validate the original production representation.
    size = struct.unpack(">I", payload[len(MAGIC) : PREFIX_SIZE])[0]
    metadata = json.loads(payload[PREFIX_SIZE : PREFIX_SIZE + size])
    if metadata["version"] != 1 or metadata["compression"] != "zlib":
        raise ValueError("The candidate comparison requires production v1 input.")
    raw = zlib.decompress(payload[PREFIX_SIZE + size :])
    width = np.dtype(metadata["dtype"]).itemsize
    shuffled = np.frombuffer(raw, dtype="u1").reshape(-1, width).T.copy().tobytes()
    choices = []
    for compression, source in (("zlib", raw), ("zlib-byte-shuffle", shuffled)):
        proposed = {
            "version": 2,
            "compression": compression,
            "dtype": metadata["dtype"],
            "shape": metadata["shape"],
        }
        digest = hashlib.sha256(MAGIC + metadata_bytes(proposed))
        digest.update(raw)
        proposed["sha256"] = digest.hexdigest()
        header = metadata_bytes(proposed)
        encoded = (
            MAGIC + struct.pack(">I", len(header)) + header + zlib.compress(source, 6)
        )
        # Independently reverse the proposed layout and validate its checksum.
        restored = zlib.decompress(encoded[PREFIX_SIZE + len(header) :])
        if compression == "zlib-byte-shuffle":
            restored = (
                np.frombuffer(restored, dtype="u1")
                .reshape(width, -1)
                .T.copy()
                .tobytes()
            )
        if restored != raw:
            raise ValueError("Byte-shuffle candidate changed original sample bits.")
        checked = {name: value for name, value in proposed.items() if name != "sha256"}
        digest = hashlib.sha256(MAGIC + metadata_bytes(checked))
        digest.update(restored)
        if digest.hexdigest() != proposed["sha256"]:
            raise ValueError("Byte-shuffle candidate checksum failed.")
        choices.append((encoded, compression))
    return min(choices, key=lambda choice: len(choice[0]))


def main() -> None:
    """Record lossless candidate sizes; no timing or database acceptance is implied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = {}
    legacy_json = 0
    with np.load(args.input, allow_pickle=False) as arrays:
        for method in ("oasis", "cascade"):
            values = arrays[method]
            original_bytes = candidate_bytes = shuffled_rows = 0
            for row in values:
                original = encode_trace_array(row.tolist())
                proposed, compression = candidate(original)
                original_bytes += len(original)
                candidate_bytes += len(proposed)
                shuffled_rows += compression == "zlib-byte-shuffle"
                if method == "oasis":
                    legacy_json += len(json.dumps(row.tolist()).encode())
            sizes[method] = {
                "shape": list(values.shape),
                "v1_payload_bytes_per_fov": original_bytes,
                "candidate_payload_bytes_per_fov": candidate_bytes,
                "byte_shuffled_rows": shuffled_rows,
                "all_original_sample_bits_preserved": True,
            }
    original = sum(item["v1_payload_bytes_per_fov"] for item in sizes.values())
    proposed = sum(item["candidate_payload_bytes_per_fov"] for item in sizes.values())
    result = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "scope": (
            "proposed v2 payload sizes only; current cali cannot read this candidate"
        ),
        "rows": sizes,
        "hypothetical_legacy_oasis_json_bytes_per_fov": legacy_json,
        "v1_with_legacy_96_fov_mib": (original + legacy_json) * 96 / 1024**2,
        "candidate_with_legacy_96_fov_mib": (proposed + legacy_json) * 96 / 1024**2,
        "budget_mib": 512,
        "production_gate": (
            "failed v1 workload; v2 requires reader and migration acceptance"
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
