"""Network-free window/cache contracts and optional pretrained equivalence."""

from __future__ import annotations

import gc
import json
import os
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from cali._cascade_models import CascadeModel, CascadeModelError
from cali._cascade_package import CascadePackage
from cali.extraction._frame_window import TimingDescriptor
from cali.extraction._spike_inference import InferenceCancelled
from cali.extraction._spike_inference import _cascade_reference as reference
from cali.extraction._spike_inference._cascade_cached import (
    CachedCascadePredictor,
    iter_cascade_windows,
)
from cali.extraction._spike_inference._cascade_service import CascadeInferenceService


def _timing(count: int, rate: float = 10) -> TimingDescriptor:
    return TimingDescriptor(
        (np.arange(count) * 1000 / rate).tolist(), "runner_time", True
    )


@pytest.fixture
def toy_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CascadeModel, CascadePackage, dict[str, Any]]:
    folder = tmp_path / "Test_10Hz"
    folder.mkdir()
    names = tuple(
        f"Model_NoiseLevel_{noise}_Ensemble_{index}.pth"
        for noise in (2, 3)
        for index in (0, 1)
    )
    for name in names:
        (folder / name).write_text(name)
    model = CascadeModel(
        "Test_10Hz",
        folder,
        "catalogue",
        "manifest",
        10,
        0.2,
        False,
        8,
        0.5,
        (2, 3),
        2,
        names,
        "config-hash",
    )
    state: dict[str, Any] = {
        "loads": [],
        "calls": [],
        "modules": [],
        "inference": False,
    }

    class Tensor:
        def __init__(self, array: np.ndarray) -> None:
            self.array = array

        def to(self, device: str) -> Tensor:
            return self

        def cpu(self) -> Tensor:
            return self

        def numpy(self) -> np.ndarray:
            return self.array

    class Network:
        def __init__(self, **kwargs: Any) -> None:
            state["modules"].append(weakref.ref(self))
            self.bias = 0.0
            self.training = True
            self.location = None

        def parameters(self) -> list:
            return [SimpleNamespace(numel=lambda: 16, element_size=lambda: 4)]

        def buffers(self) -> list:
            return []

        def load_state_dict(self, weights: dict) -> None:
            self.bias = weights["bias"]

        def to(self, **kwargs: Any) -> Network:
            self.location = kwargs
            return self

        def eval(self) -> Network:
            self.training = False
            return self

        def __call__(self, tensor: Tensor) -> Tensor:
            assert not self.training and state["inference"]
            assert tensor.array.dtype == np.float32
            state["calls"].append((threading.get_ident(), len(tensor.array)))
            callback = state.get("on_forward")
            if callback:
                callback()
            return Tensor(
                (tensor.array.mean(axis=(1, 2)) + np.float32(self.bias))[:, None]
            )

    def load(path: Path, **kwargs: Any) -> dict:
        assert kwargs["weights_only"] is True
        state["loads"].append(
            (path.name, threading.get_ident(), kwargs["map_location"])
        )
        if state.get("fail_load") == path.name:
            raise ValueError("broken weights")
        noise, index = path.stem.split("_")[2::2]
        return {"bias": float(noise) / 10 + float(index) / 4 - 0.5}

    @contextmanager
    def inference() -> Any:
        assert not state["inference"]
        state["inference"] = True
        try:
            yield
        finally:
            state["inference"] = False

    package = CascadePackage(
        *(ModuleType(name) for name in ("cascade", "config", "utils", "torch")),
        "2.0",
        "package-pin",
        "source-hash",
    )
    package.torch.device = lambda device: device
    package.torch.cuda = SimpleNamespace(is_available=lambda: False)
    package.torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )
    package.torch.float32 = "float32"
    package.torch.from_numpy = Tensor
    package.torch.load = load
    package.torch.inference_mode = inference
    package.config.read_config = lambda path: {
        "filter_sizes": [3, 2, 1],
        "filter_numbers": [2, 2, 2],
        "dense_expansion": 2,
        "windowsize": 8,
        "loss_function": "mean_squared_error",
        "optimizer": "Adagrad",
    }
    package.utils.define_model = Network
    package.utils.calculate_noise_levels = lambda dff, rate: np.resize(
        [2.0, 3.0], len(dff)
    )
    monkeypatch.setattr(reference, "load_cascade_package", lambda: package)
    monkeypatch.setattr(reference, "load_cascade_model", lambda *args, **kwargs: model)
    return model, package, state


@pytest.mark.parametrize("chunk", [1, 3, 10, 1024])
@pytest.mark.parametrize("before", [0.25, 0.5, 0.75])
def test_window_coordinates_order_and_allocation(
    chunk: int, before: float, toy_package: tuple
) -> None:
    model, _, _ = toy_package
    model = replace(model, before_fraction=before)
    dff = np.arange(3 * 19, dtype=float).reshape(3, 19)
    original = dff.copy()
    records = []
    for rows, frames, windows in iter_cascade_windows(
        dff, np.array([2, 0]), model, chunk
    ):
        assert windows.dtype == np.float32 and windows.flags.c_contiguous
        assert len(windows) <= chunk
        assert windows.nbytes <= chunk * model.window_size * 4
        for row, frame, window in zip(rows, frames, windows, strict=True):
            begin = frame - int(before * model.window_size - 1)
            np.testing.assert_array_equal(window[:, 0], dff[row, begin : begin + 8])
            records.append((int(row), int(frame)))
    start, stop = model.valid_interval(19)
    assert records == [(row, frame) for row in (2, 0) for frame in range(start, stop)]
    np.testing.assert_array_equal(dff, original)


def test_fractional_upstream_alignment_is_not_silently_repaired(
    toy_package: tuple,
) -> None:
    model, _, _ = toy_package
    with pytest.raises(CascadeModelError, match="incomplete upstream"):
        list(
            iter_cascade_windows(
                np.zeros((1, 19)),
                np.array([0]),
                replace(model, before_fraction=0.4),
                10,
            )
        )


@pytest.mark.parametrize("chunk", [1, 5, 64])
def test_ensemble_average_nonnegative_edges_and_reuse(
    chunk: int, toy_package: tuple
) -> None:
    model, _, state = toy_package
    backend = CachedCascadePredictor(
        model.name, model.directory.parent, device="cpu", max_windows=chunk
    )
    dff = np.linspace(-0.3, 0.7, 3 * 19).reshape(3, 19)
    actual = backend.infer_all(dff, 10, timing=_timing(19))
    expected = np.zeros_like(dff, dtype=np.float32)
    for row in range(3):
        noise = (2, 3, 2)[row]
        for frame in range(4, 15):
            mean = dff[row, frame - 3 : frame + 5].astype(np.float32).mean()
            terms = [
                np.float32(mean + np.float32(noise / 10 + index / 4 - 0.5)) / 2
                for index in (0, 1)
            ]
            expected[row, frame] = max(sum(float(value) for value in terms), 0)
    np.testing.assert_array_equal(actual.spikes, expected)
    assert len(state["loads"]) == 4
    np.testing.assert_array_equal(
        backend.infer_all(dff, 10, timing=_timing(19)).spikes, expected
    )
    assert len(state["loads"]) == 4
    assert backend.stats.ensembles == 2
    assert backend.stats.parameter_bytes == 256
    assert backend.stats.largest_chunk_windows <= chunk
    assert all(
        module().location == {"device": "cpu", "dtype": "float32"}
        for module in state["modules"]
    )
    backend.clear_cache()
    assert backend.stats.ensembles == backend.stats.parameter_bytes == 0
    assert all(module() is None for module in state["modules"])
    backend.infer_all(dff, 10, timing=_timing(19))
    assert len(state["loads"]) == 8
    backend.close()
    assert backend.stats.ensembles == backend.stats.parameter_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        backend.infer_all(dff, 10, timing=_timing(19))


@pytest.mark.skipif(
    os.environ.get("CALI_CASCADE_REFERENCE_TESTS") != "1",
    reason="Real device cancellation runs in optional pretrained validation",
)
@pytest.mark.parametrize(
    "device", os.environ.get("CALI_CASCADE_TEST_DEVICES", "cpu").split(",")
)
def test_pretrained_cancellation_stops_after_one_chunk_and_can_retry(
    device: str,
) -> None:
    fixture = Path(__file__).parent / "fixtures/cascade_reference"
    metadata = json.loads((fixture / "manifest.json").read_text())
    with np.load(fixture / "real_excerpt.npz", allow_pickle=False) as saved:
        dff, golden = saved["dff"], saved["expected_spikes"]
    backend = CachedCascadePredictor(
        metadata["model_name"],
        expected_manifest=metadata["model_manifest_sha256"],
        device=device,
        max_windows=37,
    )
    try:
        baseline = backend.infer_all(dff, 30, timing=_timing(dff.shape[1], 30))
        chunks, loads = backend.stats.chunks, backend.stats.model_loads
        with pytest.raises(InferenceCancelled):
            backend.infer_all(
                dff,
                30,
                timing=_timing(dff.shape[1], 30),
                cancel=lambda: backend.stats.chunks > chunks,
            )
        assert backend.stats.chunks == chunks + 1
        assert backend.stats.model_loads == loads
        retry = backend.infer_all(dff, 30, timing=_timing(dff.shape[1], 30))
        assert retry.resolved_device.split(":")[0] == device
        np.testing.assert_array_equal(retry.spikes, baseline.spikes)
        np.testing.assert_allclose(retry.spikes, golden, rtol=1e-5, atol=1e-6)
        assert backend.stats.model_loads == loads
    finally:
        backend.close()
    assert backend.stats.ensembles == backend.stats.parameter_bytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        backend.infer_all(dff, 30, timing=_timing(dff.shape[1], 30))


def test_cache_budget_evicts_and_refuses_oversized_ensemble(toy_package: tuple) -> None:
    model, _, state = toy_package
    backend = CachedCascadePredictor(model.name, max_cache_bytes=150, device="cpu")
    backend.infer_all(np.zeros((2, 19)), 10, timing=_timing(19))
    assert backend.stats.ensembles == 1 and backend.stats.parameter_bytes == 128
    assert sum(module() is not None for module in state["modules"]) == 2
    backend.infer_all(np.zeros((1, 19)), 10, timing=_timing(19))
    assert len(state["loads"]) == 6
    backend.close()
    too_small = CachedCascadePredictor(model.name, max_cache_bytes=64, device="cpu")
    with pytest.raises(CascadeModelError, match="cache budget"):
        too_small.infer_all(np.zeros((1, 19)), 10, timing=_timing(19))
    assert too_small.stats.ensembles == 0


def test_failed_ensemble_never_publishes_partial_models(toy_package: tuple) -> None:
    model, _, state = toy_package
    backend = CachedCascadePredictor(model.name, device="cpu")
    state["fail_load"] = model.weight_files[1]
    with pytest.raises(ValueError, match="broken weights"):
        backend.infer_all(np.zeros((1, 19)), 10, timing=_timing(19))
    assert backend.stats.ensembles == 0
    assert all(module() is None for module in state["modules"])


def test_validated_numeric_weight_names_are_not_treated_as_missing(
    toy_package: tuple,
) -> None:
    model, package, state = toy_package
    aliases = []
    for name in model.weight_files:
        alias = name.replace("NoiseLevel_2_", "NoiseLevel_02_")
        (model.directory / name).rename(model.directory / alias)
        aliases.append(alias)
    backend = CachedCascadePredictor(model.name, device="cpu")
    backend._ensemble(
        replace(model, weight_files=tuple(aliases)), package, 2, "cpu", None
    )
    assert len(state["loads"]) == 2
    backend.close()


def test_cancel_after_chunk_and_single_thread_ownership(toy_package: tuple) -> None:
    model, _, state = toy_package
    backend = CachedCascadePredictor(model.name, device="cpu", max_windows=3)
    cancelled = threading.Event()
    state["on_forward"] = cancelled.set
    with pytest.raises(InferenceCancelled):
        backend.infer_all(
            np.zeros((1, 100)), 10, timing=_timing(100), cancel=cancelled.is_set
        )
    assert backend.stats.chunks == 1
    assert len(state["calls"]) == 2  # Finish the current chunk's ensemble, then stop.
    errors = []

    def wrong_owner() -> None:
        try:
            backend.clear_cache()
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=wrong_owner)
    thread.start()
    thread.join(5)
    assert len(errors) == 1 and "owning worker" in errors[0]
    backend.close()


def test_cache_identity_includes_manifest_device_index_and_dtype(
    toy_package: tuple,
) -> None:
    model, package, state = toy_package
    backend = CachedCascadePredictor(model.name, device="cpu")
    backend._ensemble(model, package, 2, "cpu", None)
    backend._ensemble(model, package, 2, "cpu", None)
    assert len(state["loads"]) == 2
    backend._ensemble(
        replace(model, manifest_sha256="replacement"), package, 2, "cpu", None
    )
    backend._ensemble(model, package, 2, "cuda:0", None)
    backend._ensemble(model, package, 2, "cuda:1", None)
    assert len(state["loads"]) == 8
    assert len(backend._model_cache) == 4
    assert {key.dtype for key in backend._model_cache} == {"float32"}
    assert {key.device_index for key in backend._model_cache} == {None, 0, 1}
    backend.clear_cache()


def _wait_pending(service: CascadeInferenceService) -> None:
    deadline = time.monotonic() + 3
    while service.pending_count != 1 and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    assert service.pending_count == 1


def test_service_single_owner_bounded_queue_and_waiting_cancel(
    toy_package: tuple,
) -> None:
    model, _, state = toy_package
    started, release, cancelled, third_started = (threading.Event() for _ in range(4))

    def pause() -> None:
        started.set()
        assert release.wait(5)

    state["on_forward"] = pause
    service = CascadeInferenceService(model.name, device="cpu", max_windows=3)
    dff = np.zeros((2, 19))
    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            first = pool.submit(service.infer_all, dff, 10, timing=_timing(19))
            assert started.wait(3)
            second = pool.submit(service.infer_all, dff, 10, timing=_timing(19))
            _wait_pending(service)

            def third() -> Any:
                third_started.set()
                return service.infer_all(
                    dff, 10, timing=_timing(19), cancel=cancelled.is_set
                )

            blocked = pool.submit(third)
            assert third_started.wait(3) and not blocked.done()
            cancelled.set()
            with pytest.raises(InferenceCancelled):
                blocked.result(timeout=3)
            assert service.pending_count == 1
            release.set()
            np.testing.assert_array_equal(
                first.result(timeout=3).spikes, second.result(timeout=3).spikes
            )
            assert {call[0] for call in state["calls"]} == {service._worker.ident}
            assert {load[1] for load in state["loads"]} == {service._worker.ident}
            assert len(state["loads"]) == 4
            assert service.stats.model_loads == 4
            service.clear_cache()
            assert service.stats.ensembles == 0
            assert all(module() is None for module in state["modules"])
        finally:
            release.set()
            service.close()
    assert not service._worker.is_alive()


def test_service_close_cancels_current_and_queued_requests(toy_package: tuple) -> None:
    model, _, state = toy_package
    started, release = threading.Event(), threading.Event()

    def pause() -> None:
        started.set()
        assert release.wait(5)

    state["on_forward"] = pause
    service = CascadeInferenceService(model.name, device="cpu", max_windows=3)
    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            first = pool.submit(
                service.infer_all, np.zeros((1, 100)), 10, timing=_timing(100)
            )
            assert started.wait(3)
            second = pool.submit(
                service.infer_all, np.zeros((1, 100)), 10, timing=_timing(100)
            )
            _wait_pending(service)
            closing = pool.submit(service.close)
            assert service._closing.wait(3)
            release.set()
            for pending in (first, second):
                with pytest.raises(InferenceCancelled):
                    pending.result(timeout=3)
            closing.result(timeout=3)
            assert service.stats.chunks == 1
            assert service.stats.ensembles == 0
            assert not service._worker.is_alive()
        finally:
            release.set()
            service.close()
    service.close()  # Idempotent, including concurrent closers.
    with pytest.raises(RuntimeError, match="closed"):
        service.infer_all(np.zeros((1, 19)), 10, timing=_timing(19))


def test_queued_cancel_does_not_wait_for_another_fov(toy_package: tuple) -> None:
    model, _, state = toy_package
    started, release, cancelled = (threading.Event() for _ in range(3))

    def pause() -> None:
        started.set()
        assert release.wait(5)

    state["on_forward"] = pause
    service = CascadeInferenceService(model.name, device="cpu", max_windows=3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            active = pool.submit(
                service.infer_all, np.zeros((1, 100)), 10, timing=_timing(100)
            )
            assert started.wait(3)
            queued = pool.submit(
                service.infer_all,
                np.zeros((1, 100)),
                10,
                timing=_timing(100),
                cancel=cancelled.is_set,
            )
            _wait_pending(service)
            cancelled.set()
            with pytest.raises(InferenceCancelled):
                queued.result(timeout=1)
            assert not active.done()
            release.set()
            active.result(timeout=3)
            service.clear_cache()
            assert len(state["loads"]) == 2
        finally:
            release.set()
            service.close()


@pytest.mark.parametrize("error", [ValueError, TimeoutError])
def test_service_error_propagation_recovery_and_input_release(
    error: type[Exception], toy_package: tuple
) -> None:
    model, _, state = toy_package

    def fail() -> None:
        raise error("model failure")

    with CascadeInferenceService(model.name, device="cpu") as service:
        state["on_forward"] = fail
        with ThreadPoolExecutor(max_workers=1) as pool:
            failing = pool.submit(
                service.infer_all, np.zeros((1, 19)), 10, timing=_timing(19)
            )
            with pytest.raises(error, match="model failure"):
                failing.result(timeout=3)
        state.pop("on_forward")
        dff = np.zeros((1, 19))
        input_ref = weakref.ref(dff)
        result = service.infer_all(dff, 10, timing=_timing(19))
        del dff
        service.clear_cache()  # Barrier: the last prediction request has been released.
        gc.collect()
        assert input_ref() is None
        assert result.spikes.shape == (1, 19)


def test_service_reentrant_submission_rejected_and_close_before_use(
    toy_package: tuple,
) -> None:
    model, _, state = toy_package
    service = CascadeInferenceService(model.name, device="cpu")
    state["on_forward"] = service.clear_cache
    try:
        with pytest.raises(RuntimeError, match="submit to itself"):
            service.infer_all(np.zeros((1, 19)), 10, timing=_timing(19))
    finally:
        service.close()
    unused = CascadeInferenceService(model.name, device="cpu")
    unused.close()
    assert unused.stats.ensembles == 0
    assert not unused._worker.is_alive()


@pytest.mark.skipif(
    os.environ.get("CALI_CASCADE_REFERENCE_TESTS") != "1",
    reason="Real cached service equivalence runs in CASCADE CI",
)
@pytest.mark.parametrize(
    "device", os.environ.get("CALI_CASCADE_TEST_DEVICES", "cpu").split(",")
)
def test_pretrained_service_reuses_models_across_concurrent_fovs(device: str) -> None:
    fixture = Path(__file__).parent / "fixtures/cascade_reference"
    metadata = json.loads((fixture / "manifest.json").read_text())
    with np.load(fixture / "real_excerpt.npz", allow_pickle=False) as saved:
        dff, golden = saved["dff"], saved["expected_spikes"]
    with CascadeInferenceService(
        metadata["model_name"],
        expected_manifest=metadata["model_manifest_sha256"],
        device=device,
        max_windows=37,
    ) as service:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(service.infer_all, dff, 30, timing=_timing(256, 30))
                for _ in range(3)
            ]
            for future in futures:
                result = future.result(timeout=60)
                assert result.resolved_device.split(":")[0] == device
                np.testing.assert_allclose(result.spikes, golden, rtol=1e-5, atol=1e-6)
        assert service.stats.model_loads == 5
        assert service.stats.ensembles == 1
    assert service.stats.ensembles == service.stats.parameter_bytes == 0


@pytest.mark.skipif(
    os.environ.get("CALI_CASCADE_REFERENCE_TESTS") != "1",
    reason="Real cached predictor equivalence runs in CASCADE CI",
)
@pytest.mark.parametrize("chunk", [1, 37, 1024])
@pytest.mark.parametrize(
    "device", os.environ.get("CALI_CASCADE_TEST_DEVICES", "cpu").split(",")
)
def test_cached_pretrained_matches_upstream_and_golden(chunk: int, device: str) -> None:
    fixture = Path(__file__).parent / "fixtures/cascade_reference"
    metadata = json.loads((fixture / "manifest.json").read_text())
    backend = CachedCascadePredictor(
        metadata["model_name"],
        expected_manifest=metadata["model_manifest_sha256"],
        device=device,
        max_windows=chunk,
    )
    oracle = reference.CascadeReferenceBackend(
        metadata["model_name"],
        expected_manifest=metadata["model_manifest_sha256"],
        device=device,
    )
    with np.load(fixture / "real_excerpt.npz", allow_pickle=False) as saved:
        real, golden = saved["dff"], saved["expected_spikes"]
    synthetic = (
        np.random.default_rng(732)
        .normal(0, np.array([0.12, 0.3])[:, None], (2, 128))
        .astype(np.float32)
    )
    synthetic[:, 50:] += 0.4 * np.exp(-np.arange(78) / 8)
    for dff in (real, synthetic):
        expected = oracle.infer_all(dff, 30, timing=_timing(dff.shape[1], 30))
        actual = backend.infer_all(dff, 30, timing=_timing(dff.shape[1], 30))
        assert actual.resolved_device.split(":")[0] == device
        np.testing.assert_allclose(actual.spikes, expected.spikes, rtol=1e-5, atol=1e-6)
        if dff is real:
            np.testing.assert_allclose(actual.spikes, golden, rtol=1e-5, atol=1e-6)
        if device != "cpu":
            cpu = reference.CascadeReferenceBackend(
                metadata["model_name"],
                expected_manifest=metadata["model_manifest_sha256"],
                device="cpu",
            ).infer_all(dff, 30, timing=_timing(dff.shape[1], 30))
            np.testing.assert_allclose(actual.spikes, cpu.spikes, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(actual.noise_by_roi, expected.noise_by_roi)
        np.testing.assert_array_equal(
            actual.selected_noise_levels_by_roi, expected.selected_noise_levels_by_roi
        )
        loads = backend.stats.model_loads
        np.testing.assert_array_equal(
            backend.infer_all(dff, 30, timing=_timing(dff.shape[1], 30)).spikes,
            actual.spikes,
        )
        assert backend.stats.model_loads == loads
        assert backend.stats.largest_chunk_windows <= chunk
    backend.close()
