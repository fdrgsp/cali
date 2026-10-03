"""Regenerate the small pretrained oracle from a pinned upstream real example.

Run in an environment with cali[cascade], with OMP_NUM_THREADS=1 and
MKL_NUM_THREADS=1 for bounded CPU execution. This calls external upstream
APIs directly; it does not use cali's reference adapter to generate its oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat

from cali._cascade_models import load_cascade_model
from cali._cascade_package import load_cascade_package

MODEL = "Global_EXC_30Hz_smoothing25ms"
MODEL_MANIFEST = "ac8954174ba0a01a2d929a7e8b3fc7e3a4365d5c01f5e262d2b597822fcae184"
SOURCE_SHA256 = "ba82413d65fc3dfac622da559fdef0e379f7ed072316b08b1980c7c7d4739f3b"
SOURCE_REVISION = "c6978d5ff33edad8792c76e040412f0636913092"
SOURCE_PATH = (
    "Example_datasets/Allen-Brain-Observatory-Visual-Coding-30Hz/"
    "Experiment_552195520_excerpt.mat"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example_mat", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.example_mat.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("The example MAT file does not match the pinned source checksum")
    package = load_cascade_package()
    model = load_cascade_model(
        MODEL, args.model_dir, expected_manifest=MODEL_MANIFEST
    )
    dff = loadmat(args.example_mat)["dF_traces"][:2, :256].copy()
    noise = package.utils.calculate_noise_levels(dff, model.sampling_rate)
    expected = package.cascade.predict(
        MODEL, dff, model_folder=str(model.directory.parent), threshold=0,
        padding=0, trace_noise_levels=noise, verbosity=0,
        device=package.torch.device("cpu"),
    )
    if not np.isfinite(expected).all() or not np.any(expected > 0):
        raise ValueError("The golden example must have finite, nonzero predictions")
    folder = Path(__file__).resolve().parents[1] / "tests/fixtures/cascade_reference"
    folder.mkdir(parents=True, exist_ok=True)
    fixture = folder / "real_excerpt.npz"
    np.savez_compressed(fixture, dff=dff, expected_spikes=expected, noise=noise)
    metadata = {
        "schema_version": 1,
        "model_name": MODEL,
        "model_manifest_sha256": model.manifest_sha256,
        "catalogue_revision": model.catalogue_revision,
        "package_revision": package.package_revision,
        "package_source_sha256": package.source_manifest_sha256,
        "source_url": (
            "https://github.com/PTRRupprecht/CascadeTorch/blob/"
            + SOURCE_REVISION + "/" + SOURCE_PATH
        ),
        "source_sha256": SOURCE_SHA256,
        "source_array": "dF_traces",
        "roi_slice": [0, 2],
        "frame_slice": [0, 256],
        "sampling_rate_hz": model.sampling_rate,
        "valid_interval": list(model.valid_interval(256)),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "device": "cpu",
        "input_dtype": str(dff.dtype),
        "prediction_dtype": str(expected.dtype),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": package.torch.__version__,
        "torch_threads": package.torch.get_num_threads(),
        "threshold": 0,
        "padding": 0,
    }
    (folder / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
