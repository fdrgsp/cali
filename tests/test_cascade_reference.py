"""Network-free reference-adapter contracts and optional real-model oracle."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from cali._cascade_models import CascadeModel, CascadeModelError
from cali._cascade_package import CascadePackage
from cali.extraction._frame_window import TimingDescriptor
from cali.extraction._spike_inference import InferenceCancelled
from cali.extraction._spike_inference import _cascade_reference as reference


def _timing(count: int = 96, rate: float = 10) -> TimingDescriptor:
    return TimingDescriptor(
        (np.arange(count) * 1000 / rate).tolist(), "runner_time", True
    )


@pytest.fixture
def fake_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[reference.CascadeReferenceBackend, CascadePackage, list[tuple]]:
    model = CascadeModel(
        "Test_10Hz",
        tmp_path / "Test_10Hz",
        "catalogue",
        "manifest",
        10,
        0.2,
        False,
        64,
        0.5,
        (2, 3, 4),
        2,
        ("weight.pth",),
        "config-hash",
    )
    cascade = ModuleType("cascade")
    utils = ModuleType("utils")
    torch = ModuleType("torch")
    torch.device = lambda device: device
    torch.cuda = MagicMock()
    torch.cuda.is_available.return_value = False
    torch.backends = MagicMock()
    torch.backends.mps.is_available.return_value = False
    calls = []

    def noise(dff: np.ndarray, rate: float) -> np.ndarray:
        calls.append(("noise", rate))
        return np.median(np.abs(np.diff(dff, axis=1)), axis=1) / np.sqrt(rate) * 100

    def predict(name: str, dff: np.ndarray, **kwargs: Any) -> np.ndarray:
        calls.append(("predict", name, kwargs))
        result = np.zeros_like(dff)
        result[:, 32:-32] = 0.25
        return result

    utils.calculate_noise_levels = noise
    cascade.predict = predict
    package = CascadePackage(
        cascade, ModuleType("config"), utils, torch, "2.0", "package-pin", "source-hash"
    )
    monkeypatch.setattr(reference, "load_cascade_model", lambda *a, **kw: model)
    monkeypatch.setattr(reference, "load_cascade_package", lambda: package)
    return (
        reference.CascadeReferenceBackend(model.name, tmp_path, device="cpu"),
        package,
        calls,
    )


def test_once_per_fov_model_rate_noise_and_provenance(fake_reference: tuple) -> None:
    backend, _package, calls = fake_reference
    dff = np.random.default_rng(17).normal(0, 0.08, (2, 96))
    original = dff.copy()
    result = backend.infer_all(dff, 10, timing=_timing(rate=9.95))
    assert [call[0] for call in calls] == ["noise", "predict"]
    assert calls[0][1] == 10  # Model rate, even when accepted observed rate differs.
    kwargs = calls[1][2]
    assert kwargs["threshold"] == 0 and kwargs["threshold"] is not False
    assert kwargs["padding"] == 0
    assert kwargs["verbosity"] == 0
    assert kwargs["device"] == "cpu"
    np.testing.assert_array_equal(kwargs["trace_noise_levels"], result.noise_by_roi)
    np.testing.assert_array_equal(original, dff)
    assert result.spikes.dtype == np.float32
    assert result.spikes.shape == dff.shape
    assert (result.valid_start, result.valid_stop) == (32, 64)
    assert not result.spikes[:, :32].any() and not result.spikes[:, 64:].any()
    assert np.all(result.spikes[:, 32:64] == 0.25)
    assert result.observed_frame_rate == pytest.approx(9.95)
    assert result.model.manifest_sha256 == "manifest"
    assert result.model.config_sha256 == "config-hash"
    assert result.package_revision == "package-pin"
    assert result.package_source_sha256 == "source-hash"
    assert result.units == "spikes/frame" and result.resolved_device == "cpu"
    assert backend.minimum_frames == 65


@pytest.mark.parametrize(
    "dff",
    [
        np.zeros(96),
        np.empty((0, 96)),
        np.empty((1, 0)),
        np.zeros((1, 64)),
        np.full((1, 96), np.nan),
        np.full((1, 96), np.inf),
        np.full((1, 96), 1e40),
        np.ones((1, 96), dtype=complex),
        np.full((1, 96), "invalid"),
    ],
)
def test_invalid_inputs_never_call_upstream(
    dff: np.ndarray, fake_reference: tuple
) -> None:
    backend, _, calls = fake_reference
    with pytest.raises(ValueError):
        backend.infer_all(dff, 10, timing=_timing())
    assert calls == []


@pytest.mark.parametrize(
    "case", ["length", "exposure", "irregular", "rate", "settings"]
)
def test_invalid_timing_never_calls_upstream(case: str, fake_reference: tuple) -> None:
    backend, _, calls = fake_reference
    timing = _timing()
    settings_rate = 10
    if case == "length":
        timing = _timing(count=95)
    elif case == "exposure":
        timing = TimingDescriptor(timing.timestamps_ms, "exposure", False)
    elif case == "irregular":
        timing.timestamps_ms[10] += 10
    elif case == "rate":
        timing = _timing(rate=11)
    else:
        settings_rate = 11
    with pytest.raises(ValueError):
        backend.infer_all(np.zeros((1, 96)), settings_rate, timing=timing)
    assert calls == []


@pytest.mark.parametrize(
    ("rate", "matches"),
    [(10.099, True), (10.1005, False), (9.901, True), (9.899, False)],
)
def test_catalogue_and_preflight_agree_near_rate_boundary(
    rate: float, matches: bool, fake_reference: tuple
) -> None:
    from cali._cascade_models import CatalogueEntry, compatible_cascade_models

    backend, _, _ = fake_reference
    entry = CatalogueEntry("model_10Hz", "https://example.test", "", 10)
    assert bool(compatible_cascade_models((entry,), rate)) is matches
    if matches:
        backend.prepare(rate)
    else:
        with pytest.raises(ValueError, match="does not match"):
            backend.prepare(rate)


def test_noise_selection_ties_and_coverage_warning(
    fake_reference: tuple, caplog: pytest.LogCaptureFixture
) -> None:
    backend, package, _ = fake_reference
    package.utils.calculate_noise_levels = lambda dff, rate: np.array([2.5, 0, 9])
    result = backend.infer_all(np.zeros((3, 96)), 10, timing=_timing())
    np.testing.assert_array_equal(result.selected_noise_levels_by_roi, [2, 2, 4])
    np.testing.assert_array_equal(result.noise_in_model_range, [True, False, False])
    assert "2/3 ROI noise estimates are outside model coverage" in caplog.text


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_reference_cancellation_discards_complete_result(
    stage: int, fake_reference: tuple
) -> None:
    backend, _, calls = fake_reference
    checks = iter(index == stage for index in range(4))
    with pytest.raises(InferenceCancelled):
        backend.infer_all(
            np.zeros((1, 96)), 10, timing=_timing(), cancel=lambda: next(checks)
        )
    assert sum(call[0] == "predict" for call in calls) == (stage == 3)


def test_modified_manifest_refuses_inference(
    fake_reference: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, _, calls = fake_reference

    def changed(*args: Any, **kwargs: Any) -> None:
        assert kwargs["expected_manifest"] == "manifest"
        raise CascadeModelError("changed model")

    monkeypatch.setattr(reference, "load_cascade_model", changed)
    with pytest.raises(CascadeModelError, match="changed model"):
        backend.infer_all(np.zeros((1, 96)), 10, timing=_timing())
    assert calls == []


@pytest.mark.parametrize("damage", ["shape", "nan", "negative", "edges"])
def test_invalid_upstream_output_rejected(damage: str, fake_reference: tuple) -> None:
    backend, package, _ = fake_reference
    output = np.zeros((1, 96))
    if damage == "shape":
        output = output[0]
    elif damage == "nan":
        output[0, 40] = np.nan
    elif damage == "negative":
        output[0, 40] = -1
    else:
        output[0, 0] = 1
    package.cascade.predict = lambda *args, **kwargs: output
    with pytest.raises(ValueError, match="CASCADE returned"):
        backend.infer_all(np.zeros((1, 96)), 10, timing=_timing())


@pytest.mark.parametrize("noise", [np.array([np.nan]), np.array([2, 3])])
def test_invalid_noise_rejected(noise: np.ndarray, fake_reference: tuple) -> None:
    backend, package, calls = fake_reference
    package.utils.calculate_noise_levels = lambda dff, rate: noise
    with pytest.raises(ValueError, match="noise estimates"):
        backend.infer_all(np.zeros((1, 96)), 10, timing=_timing())
    assert calls == []


@pytest.mark.parametrize("available", ["cpu", "mps", "cuda"])
def test_auto_device_priority_and_resolution(
    available: str, fake_reference: tuple
) -> None:
    _, package, _ = fake_reference
    package.torch.cuda.is_available.return_value = available == "cuda"
    package.torch.cuda.current_device.return_value = 1
    package.torch.backends.mps.is_available.return_value = available in {"mps", "cuda"}
    expected = "cuda:1" if available == "cuda" else available
    assert reference.resolve_cascade_device(package, "auto") == expected


@pytest.mark.parametrize("device", ["mps", "cuda", "bad"])
def test_explicit_unavailable_device_does_not_fallback(
    device: str, fake_reference: tuple
) -> None:
    _, package, _ = fake_reference
    with pytest.raises(CascadeModelError):
        reference.resolve_cascade_device(package, device)


@pytest.mark.skipif(
    os.environ.get("CALI_CASCADE_REFERENCE_TESTS") != "1",
    reason="Pretrained CASCADE oracle runs in the dedicated optional-dependency job",
)
@pytest.mark.parametrize(
    "device", os.environ.get("CALI_CASCADE_TEST_DEVICES", "cpu").split(",")
)
def test_pinned_pretrained_real_and_synthetic_oracle(device: str) -> None:
    """Opt in requires the exact real model; absent dependencies/models fail the job."""
    fixture = Path(__file__).parent / "fixtures" / "cascade_reference"
    metadata = json.loads((fixture / "manifest.json").read_text())
    package = reference.load_cascade_package()
    assert device in {"cpu", "mps", "cuda"}
    assert package.package_revision == metadata["package_revision"]
    assert package.source_manifest_sha256 == metadata["package_source_sha256"]
    backend = reference.CascadeReferenceBackend(
        metadata["model_name"],
        expected_manifest=metadata["model_manifest_sha256"],
        device=device,
    )
    with np.load(fixture / "real_excerpt.npz", allow_pickle=False) as saved:
        real = saved["dff"]
        expected = saved["expected_spikes"]
        expected_noise = saved["noise"]
    synthetic = (
        np.random.default_rng(732)
        .normal(0, np.array([0.12, 0.3])[:, None], (2, 128))
        .astype(np.float32)
    )
    synthetic[:, 50:] += 0.4 * np.exp(-np.arange(78) / 8)
    for dff in (real, synthetic):
        timing = _timing(len(dff[0]), backend.model.sampling_rate)
        result = backend.infer_all(dff, backend.model.sampling_rate, timing=timing)
        implicit = package.cascade.predict(
            backend.model.name,
            dff,
            model_folder=str(backend.model.directory.parent),
            threshold=0,
            padding=0,
            trace_noise_levels=None,
            verbosity=0,
            device=package.torch.device(result.resolved_device),
        )
        np.testing.assert_allclose(result.spikes, implicit, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(result.spikes, implicit.astype(np.float32))
        assert result.resolved_device.split(":")[0] == device
        if device != "cpu":
            cpu = reference.CascadeReferenceBackend(
                metadata["model_name"],
                expected_manifest=metadata["model_manifest_sha256"],
                device="cpu",
            ).infer_all(dff, backend.model.sampling_rate, timing=timing)
            np.testing.assert_allclose(result.spikes, cpu.spikes, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(result.noise_by_roi, cpu.noise_by_roi)
            np.testing.assert_array_equal(
                result.selected_noise_levels_by_roi, cpu.selected_noise_levels_by_roi
            )
        if dff is real:
            np.testing.assert_allclose(result.spikes, expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(result.noise_by_roi, expected_noise)
        else:
            assert len(np.unique(result.selected_noise_levels_by_roi)) == 2
        assert result.spikes.dtype == np.float32
        assert not result.spikes[:, : result.valid_start].any()
        assert not result.spikes[:, result.valid_stop :].any()


def test_pretrained_fixture_identity_and_valid_edges() -> None:
    fixture = Path(__file__).parent / "fixtures" / "cascade_reference"
    metadata = json.loads((fixture / "manifest.json").read_text())
    assert (
        hashlib.sha256((fixture / "real_excerpt.npz").read_bytes()).hexdigest()
        == (metadata["fixture_sha256"])
    )
    with np.load(fixture / "real_excerpt.npz", allow_pickle=False) as saved:
        assert saved["dff"].shape == saved["expected_spikes"].shape == (2, 256)
        assert saved["dff"].dtype == np.float32
        assert np.isfinite(saved["expected_spikes"]).all()
        assert np.any(saved["expected_spikes"] > 0)
        assert not saved["expected_spikes"][:, :32].any()
        assert not saved["expected_spikes"][:, 224:].any()
        assert saved["noise"].shape == (2,)


def test_backend_import_defers_torch_and_image_readers(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-Werror",
            "-c",
            (
                "import sys; "
                "import cali.extraction._spike_inference._cascade_reference; "
                "import cali.extraction._spike_inference._cascade_service; "
                "assert 'torch' not in sys.modules; "
                "assert 'cascade2p' not in sys.modules; "
                "assert 'cali.readers' not in sys.modules; "
                "assert 'numcodecs' not in sys.modules"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_lazy_extraction_runner_keeps_existing_public_export() -> None:
    from cali import extraction
    from cali.extraction._extraction_runner import ExtractionRunner

    assert extraction.ExtractionRunner is ExtractionRunner
    with pytest.raises(AttributeError):
        _ = extraction.missing_attribute
