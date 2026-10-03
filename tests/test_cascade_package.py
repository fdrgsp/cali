"""Installed-wheel smoke tests for the optional CASCADE package.

Run with ``pytest --noconftest tests/test_cascade_package.py`` in the clean
wheel-install job. The generated constant-output ensemble checks packaging and
upstream loading/inference; scientific pretrained-model equivalence is P3a.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cascade2p") is None,
    reason="Optional cali[cascade] dependency is not installed",
)


def test_installed_cascade_distribution_contains_importable_package() -> None:
    import cascade2p
    from cascade2p import cascade, config, utils

    installed = distribution("CascadeTorch")
    assert installed.version == "2.0"
    files = {str(file) for file in installed.files or []}
    assert "cascade2p/__init__.py" in files
    assert Path(cascade2p.__file__).is_file()
    assert callable(cascade.predict)
    assert callable(config.read_config)
    assert callable(utils.define_model)
    assert callable(utils.calculate_noise_levels)


def test_package_import_is_warning_clean_and_defers_plotting_and_torch(
    tmp_path: Path,
) -> None:
    code = (
        "import sys; import cascade2p; from cascade2p import cascade, config, utils; "
        "assert 'torch' not in sys.modules; "
        "assert 'matplotlib.pyplot' not in sys.modules; "
        "assert 'seaborn' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-I", "-W", "error", "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_cali_wheel_defers_optional_imports_and_declares_the_exact_pin(
    tmp_path: Path,
) -> None:
    code = (
        "import sys; import cali; "
        "from cali._cascade_package import CASCADE_PACKAGE_URL; "
        "from importlib.metadata import distribution; "
        "assert 'torch' not in sys.modules; assert 'cascade2p' not in sys.modules; "
        "assert any(CASCADE_PACKAGE_URL in item and 'cascade' in item "
        "for item in distribution('cali').requires)"
    )
    subprocess.run(
        [sys.executable, "-I", "-W", "error", "-c", code],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_installed_cali_loader_verifies_the_package_source_commit() -> None:
    from cali._cascade_package import CASCADE_PACKAGE_REVISION, load_cascade_package

    package = load_cascade_package()
    assert package.package_version == "2.0"
    assert package.package_revision == CASCADE_PACKAGE_REVISION
    assert len(package.source_manifest_sha256) == 64


def test_upstream_predictor_loads_generated_ensemble_and_preserves_padding(
    tmp_path: Path,
) -> None:
    import torch
    from cascade2p import cascade, config, utils
    from ruamel.yaml import YAML

    name = "Packaging_smoke_30Hz"
    folder = tmp_path / name
    folder.mkdir()
    cfg = {
        "model_name": name,
        "sampling_rate": 30,
        "training_datasets": [],
        "ensemble_size": 2,
        "batch_size": 32,
        "before_frac": 0.5,
        "windowsize": 64,
        "noise_levels": [2],
        "smoothing": 0.025,
        "causal_kernel": 0,
        "verbose": 0,
        "filter_sizes": [3, 3, 3],
        "filter_numbers": [2, 2, 2],
        "dense_expansion": 2,
        "loss_function": "mean_squared_error",
        "optimizer": "Adagrad",
    }
    with (folder / "config.yaml").open("w") as output:
        YAML().dump(cfg, output)
    assert config.read_config(folder / "config.yaml")["sampling_rate"] == 30
    for ensemble, prediction in enumerate((0.5, 1.0)):
        model = utils.define_model(
            **{
                key: cfg[key]
                for key in (
                    "filter_sizes",
                    "filter_numbers",
                    "dense_expansion",
                    "windowsize",
                    "loss_function",
                    "optimizer",
                )
            }
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.dense2.bias.fill_(prediction)
        torch.save(
            model.state_dict(), folder / f"Model_NoiseLevel_2_Ensemble_{ensemble}.pth"
        )
    dff = np.zeros((2, 96))
    noise = utils.calculate_noise_levels(dff, cfg["sampling_rate"])
    prediction = cascade.predict(
        name,
        dff,
        model_folder=str(tmp_path),
        threshold=0,
        padding=0,
        trace_noise_levels=noise,
        verbosity=0,
        device="cpu",
    )
    assert prediction.shape == dff.shape
    assert np.all(np.isfinite(prediction))
    np.testing.assert_array_equal(prediction[:, :32], 0)
    np.testing.assert_array_equal(prediction[:, -32:], 0)
    np.testing.assert_allclose(prediction[:, 32:-32], 0.75, rtol=0, atol=0)
