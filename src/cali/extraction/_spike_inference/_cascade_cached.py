"""Bounded CASCADE windows and reusable external model ensembles."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from cali._cascade_models import CascadeModelError

from ._base import InferenceCancelled
from ._cascade_reference import CascadeReferenceBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from cali._cascade_models import CascadeModel
    from cali._cascade_package import CascadePackage
    from cali.extraction._frame_window import TimingDescriptor

    from ._cascade_reference import CascadeResult


@dataclass(frozen=True)
class ModelCacheKey:
    """Keep model identity, noise ensemble, device index, and dtype independent."""

    model_dir: Path
    model_manifest_sha256: str
    noise_level: int
    device_type: str
    device_index: int | None
    dtype: str


@dataclass(frozen=True)
class CascadeCacheStats:
    """Small immutable diagnostics without exposing Torch-owned cache objects."""

    ensembles: int
    parameter_bytes: int
    model_loads: int
    chunks: int
    largest_chunk_windows: int


def _check_cancel(cancel: Callable[[], bool] | None) -> None:
    if cancel is not None and cancel():
        raise InferenceCancelled("CASCADE chunked batch cancelled.")


def _window_bounds(count: int, model: CascadeModel) -> tuple[int, int, int]:
    start, stop = model.valid_interval(count)
    preprocessing_start = int(model.before_fraction * model.window_size - 1)
    if stop - preprocessing_start + model.window_size - 1 > count:
        raise CascadeModelError(
            "The model's fractional window alignment leaves an incomplete "
            "upstream prediction inside its valid interval."
        )
    return start, stop, preprocessing_start


def iter_cascade_windows(
    dff: np.ndarray,
    roi_indices: np.ndarray,
    model: CascadeModel,
    max_windows: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield float32 windows in ROI/frame order, allocating at most one chunk.

    For a prediction at frame f, the first input frame is
    f - (valid_start - 1), per the documented upstream alignment. Only the
    predictor's valid interval is materialized; the preceding preprocessing
    sample and padded edges are excluded.
    """
    if type(max_windows) is not int or max_windows <= 0:
        raise ValueError("CASCADE chunk size must be a positive integer.")
    valid_start, valid_stop, preprocessing_start = _window_bounds(dff.shape[1], model)
    valid_count = valid_stop - valid_start
    for first in range(0, len(roi_indices) * valid_count, max_windows):
        flat = np.arange(
            first, min(first + max_windows, len(roi_indices) * valid_count)
        )
        rows = roi_indices[flat // valid_count]
        frames = valid_start + flat % valid_count
        windows = np.empty((len(flat), model.window_size, 1), dtype=np.float32)
        # Each contiguous piece refers to one ROI; sliding_window_view is a view,
        # and assignment converts straight into the bounded float32 buffer.
        breaks = np.r_[0, np.flatnonzero(np.diff(rows)) + 1, len(rows)]
        for begin, end in zip(breaks[:-1], breaks[1:]):
            source = np.lib.stride_tricks.sliding_window_view(
                dff[rows[begin]], model.window_size
            )
            start = frames[begin] - preprocessing_start
            windows[begin:end, :, 0] = source[start : start + end - begin]
        yield rows, frames, windows


class CachedCascadePredictor(CascadeReferenceBackend):
    """A single-owner predictor using external model definitions and noise QC."""

    def __init__(
        self,
        model_name: str,
        model_dir: str | Path | None = None,
        *,
        expected_manifest: str | None = None,
        device: str = "auto",
        max_windows: int = 1024,
        max_cache_bytes: int = 128 * 1024 * 1024,
    ) -> None:
        if type(max_windows) is not int or max_windows <= 0:
            raise ValueError("CASCADE chunk size must be a positive integer.")
        if type(max_cache_bytes) is not int or max_cache_bytes <= 0:
            raise ValueError("CASCADE cache budget must be a positive integer.")
        super().__init__(
            model_name, model_dir, expected_manifest=expected_manifest, device=device
        )
        self.max_windows = max_windows
        self.max_cache_bytes = max_cache_bytes
        self._model_cache: OrderedDict[ModelCacheKey, tuple[tuple[Any, ...], int]] = (
            OrderedDict()
        )
        self._owner_thread: threading.Thread | None = None
        self._closed = False
        self._model_loads = 0
        self._chunks = 0
        self._largest_chunk_windows = 0

    @property
    def stats(self) -> CascadeCacheStats:
        """Return counts and parameter storage, excluding transient activations."""
        return CascadeCacheStats(
            len(self._model_cache),
            sum(size for _, size in self._model_cache.values()),
            self._model_loads,
            self._chunks,
            self._largest_chunk_windows,
        )

    def _claim_owner(self) -> None:
        current = threading.current_thread()
        if self._owner_thread is None:
            self._owner_thread = current
        elif self._owner_thread is not current:
            raise RuntimeError(
                "CASCADE predictor must run on its owning worker thread."
            )

    def infer_all(
        self,
        dff: np.ndarray,
        frame_rate: float,
        *,
        timing: TimingDescriptor,
        cancel: Callable[[], bool] | None = None,
    ) -> CascadeResult:
        """Reuse ensembles on their owner, retaining the oracle's validation path."""
        self._claim_owner()
        if self._closed:
            raise RuntimeError("CASCADE predictor is closed.")
        return super().infer_all(dff, frame_rate, timing=timing, cancel=cancel)

    def clear_cache(self) -> None:
        """Release every cached module on the owning worker."""
        self._claim_owner()
        self._model_cache.clear()

    def close(self) -> None:
        """Release modules deterministically and reject further inference."""
        self.clear_cache()
        self._closed = True

    def _ensemble(
        self,
        model: CascadeModel,
        package: CascadePackage,
        noise_level: int,
        device: str,
        cancel: Callable[[], bool] | None,
    ) -> tuple[Any, ...]:
        device_type, _, device_index = device.partition(":")
        key = ModelCacheKey(
            model.directory.resolve(),
            model.manifest_sha256,
            noise_level,
            device_type,
            int(device_index) if device_index else None,
            "float32",
        )
        if key in self._model_cache:
            self._model_cache.move_to_end(key)
            return self._model_cache[key][0]
        names = [
            name
            for name in model.weight_files
            if int(name.split("_")[2]) == noise_level
        ]
        if len(names) != model.ensemble_size:
            raise CascadeModelError("Verified model is missing its selected ensemble.")
        file_bytes = sum((model.directory / name).stat().st_size for name in names)
        if file_bytes > self.max_cache_bytes:
            raise CascadeModelError(
                "Selected ensemble exceeds the CASCADE cache budget."
            )
        while (
            self._model_cache
            and self.stats.parameter_bytes + file_bytes > self.max_cache_bytes
        ):
            self._model_cache.popitem(last=False)
        cfg = package.config.read_config(str(model.directory / "config.yaml"))
        modules = []
        parameter_bytes = 0
        for name in names:
            _check_cancel(cancel)
            module = package.utils.define_model(
                filter_sizes=cfg["filter_sizes"],
                filter_numbers=cfg["filter_numbers"],
                dense_expansion=cfg["dense_expansion"],
                windowsize=cfg["windowsize"],
                loss_function=cfg["loss_function"],
                optimizer=cfg["optimizer"],
            )
            parameter_bytes += sum(
                value.numel() * value.element_size()
                for value in (*module.parameters(), *module.buffers())
            )
            if parameter_bytes > self.max_cache_bytes:
                raise CascadeModelError(
                    "Selected ensemble exceeds the CASCADE cache budget."
                )
            while (
                self._model_cache
                and self.stats.parameter_bytes + parameter_bytes > self.max_cache_bytes
            ):
                self._model_cache.popitem(last=False)
            module.to(device=package.torch.device(device), dtype=package.torch.float32)
            state = package.torch.load(
                model.directory / name,
                map_location=package.torch.device(device),
                weights_only=True,
            )
            self._model_loads += 1
            module.load_state_dict(state)
            del state
            module.eval()
            modules.append(module)
        _check_cancel(cancel)
        ensemble = tuple(modules)
        self._model_cache[key] = ensemble, parameter_bytes
        return ensemble

    def _predict(
        self,
        dff: np.ndarray,
        model: CascadeModel,
        package: CascadePackage,
        noise: np.ndarray,
        selected: np.ndarray,
        device: str,
        cancel: Callable[[], bool] | None,
    ) -> np.ndarray:
        _window_bounds(dff.shape[1], model)
        prediction = np.zeros(dff.shape, dtype=np.float32)
        for noise_level in model.noise_levels:
            rows = np.flatnonzero(selected == noise_level)
            if not len(rows):
                continue
            _check_cancel(cancel)
            # Validate alignment before loading weights; never emulate missing
            # windows with zero or expose a sample upstream would pad.
            ensemble = self._ensemble(model, package, noise_level, device, cancel)
            for chunk_rows, frames, windows in iter_cascade_windows(
                dff, rows, model, self.max_windows
            ):
                _check_cancel(cancel)
                self._chunks += 1
                self._largest_chunk_windows = max(
                    self._largest_chunk_windows, len(windows)
                )
                tensor = package.torch.from_numpy(windows).to(
                    package.torch.device(device)
                )
                average = np.zeros(len(windows), dtype=np.float64)
                with package.torch.inference_mode():
                    for module in ensemble:
                        values = np.asarray(module(tensor).cpu().numpy()).reshape(-1)
                        if values.shape != average.shape:
                            raise ValueError("CASCADE returned an invalid chunk shape.")
                        # NumPy float32 division, then float64 accumulation,
                        # preserves the oracle's ensemble averaging semantics.
                        average += values / len(ensemble)
                if not np.all(np.isfinite(average)) or np.any(
                    average > np.finfo(np.float32).max
                ):
                    raise ValueError("CASCADE returned invalid chunk values.")
                prediction[chunk_rows, frames] = np.maximum(average, 0)
                del tensor, average, values, windows
                _check_cancel(cancel)
            del ensemble, module
        return prediction
