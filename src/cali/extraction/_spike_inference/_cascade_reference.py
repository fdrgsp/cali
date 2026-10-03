"""Once-per-FOV upstream CASCADE oracle, before any cache or custom windowing."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cali._cascade_models import CascadeModelError, load_cascade_model
from cali._cascade_package import load_cascade_package
from cali.extraction._frame_window import validate_model_timing
from cali.logger import cali_logger

from ._base import InferenceCancelled

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from cali._cascade_models import CascadeModel
    from cali._cascade_package import CascadePackage
    from cali.extraction._frame_window import TimingDescriptor


@dataclass(frozen=True)
class CascadeResult:
    """Expected spikes/frame, valid edges, and exact model/package diagnostics."""

    spikes: np.ndarray
    noise_by_roi: np.ndarray
    selected_noise_levels_by_roi: np.ndarray
    noise_in_model_range: np.ndarray
    valid_start: int
    valid_stop: int
    model: CascadeModel
    package_version: str
    package_revision: str
    package_source_sha256: str
    resolved_device: str
    observed_frame_rate: float
    dtype: str = "float32"
    units: str = "spikes/frame"


def resolve_cascade_device(package: CascadePackage, requested: str) -> str:
    """Resolve auto once, or reject an explicitly unavailable device."""
    torch = package.torch
    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested not in {"cpu", "mps", "cuda"}:
        raise CascadeModelError("CASCADE device must be auto, cpu, cuda, or mps.")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise CascadeModelError(
                "The explicitly selected CUDA device is unavailable."
            )
        return f"cuda:{torch.cuda.current_device()}"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise CascadeModelError("The explicitly selected MPS device is unavailable.")
    return requested


class CascadeReferenceBackend:
    """Call the pinned upstream predictor once for a validated complete FOV."""

    name = "cascade"

    def __init__(
        self,
        model_name: str,
        model_dir: str | Path | None = None,
        *,
        expected_manifest: str | None = None,
        device: str = "auto",
    ) -> None:
        self.model = load_cascade_model(
            model_name, model_dir, expected_manifest=expected_manifest
        )
        self._requested_device = device
        self._resolved_device: str | None = None
        self._prediction_lock = threading.Lock()

    @property
    def minimum_frames(self) -> int:
        """Expose the selected model's receptive-field preflight requirement."""
        return self.model.minimum_frames

    def prepare(self, frame_rate: float) -> None:
        """Check package/device and configured rate before any ROI work."""
        if (
            not math.isfinite(frame_rate)
            or frame_rate <= 0
            or abs(frame_rate - self.model.sampling_rate) / self.model.sampling_rate
            > 0.01
        ):
            raise CascadeModelError(
                f"Extraction rate {frame_rate:g} Hz does not match CASCADE model "
                f"rate {self.model.sampling_rate:g} Hz (allowed 1%)."
            )
        if self._resolved_device is None:
            self._resolved_device = resolve_cascade_device(
                load_cascade_package(), self._requested_device
            )

    def infer_all(
        self,
        dff: np.ndarray,
        frame_rate: float,
        *,
        timing: TimingDescriptor,
        cancel: Callable[[], bool] | None = None,
    ) -> CascadeResult:
        """Validate retained inputs, compute model-rate noise, and call upstream."""
        if dff.ndim != 2 or dff.shape[0] == 0:
            raise ValueError("CASCADE requires a nonempty (ROIs, frames) DFF matrix.")
        if (
            not np.issubdtype(dff.dtype, np.number)
            or np.issubdtype(dff.dtype, np.complexfloating)
            or not np.all(np.isfinite(dff))
        ):
            raise ValueError("CASCADE requires finite real DFF values.")
        if np.any(dff > np.finfo(np.float32).max) or np.any(
            dff < -np.finfo(np.float32).max
        ):
            raise ValueError("CASCADE DFF values must be representable as float32.")
        valid_start, valid_stop = self.model.valid_interval(dff.shape[1])
        if len(timing.timestamps_ms) != dff.shape[1]:
            raise ValueError("CASCADE timing must match the retained DFF length.")
        observed_rate = validate_model_timing(
            timing,
            settings_frame_rate=frame_rate,
            model_frame_rate=self.model.sampling_rate,
        )
        if cancel is not None and cancel():
            raise InferenceCancelled("CASCADE reference batch cancelled.")
        # Never reuse a descriptor after a config/weight replacement.
        model = load_cascade_model(
            self.model.name,
            self.model.directory.parent,
            expected_manifest=self.model.manifest_sha256,
        )
        package = load_cascade_package()
        if self._resolved_device is None:
            self._resolved_device = resolve_cascade_device(
                package, self._requested_device
            )
        noise = np.asarray(
            package.utils.calculate_noise_levels(dff, model.sampling_rate)
        )
        if noise.shape != (len(dff),) or not np.all(np.isfinite(noise)):
            raise ValueError(
                "CASCADE noise estimates must be finite and match DFF rows."
            )
        levels = np.asarray(model.noise_levels)
        selected = levels[np.argmin(np.abs(noise[:, None] - levels[None, :]), axis=1)]
        coverage = (noise >= levels.min()) & (noise <= levels.max())
        if not np.all(coverage):
            cali_logger.warning(
                f"CASCADE {model.name}: {np.count_nonzero(~coverage)}/{len(dff)} ROI "
                f"noise estimates are outside model coverage "
                f"[{levels.min()}, {levels.max()}]; nearest noise ensembles are used."
            )
        if cancel is not None and cancel():
            raise InferenceCancelled("CASCADE reference batch cancelled.")
        prediction = np.asarray(
            self._predict(
                dff, model, package, noise, selected, self._resolved_device, cancel
            )
        )
        if cancel is not None and cancel():
            raise InferenceCancelled("CASCADE reference batch cancelled.")
        if (
            prediction.shape != dff.shape
            or not np.all(np.isfinite(prediction))
            or np.any(prediction < 0)
            or np.any(prediction > np.finfo(np.float32).max)
        ):
            raise ValueError("CASCADE returned invalid expected spikes/frame.")
        if np.any(prediction[:, :valid_start]) or np.any(prediction[:, valid_stop:]):
            raise ValueError("CASCADE returned nonzero receptive-field padding.")
        return CascadeResult(
            prediction.astype(np.float32, copy=False),
            noise.copy(),
            selected,
            coverage,
            valid_start,
            valid_stop,
            model,
            package.package_version,
            package.package_revision,
            package.source_manifest_sha256,
            self._resolved_device,
            observed_rate,
        )

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
        """Keep upstream prediction as the oracle behind shared input validation."""
        # The reference fallback serializes Torch calls across the FOV pool.
        # Waiting callers can still cancel; upstream itself is not interruptible.
        while not self._prediction_lock.acquire(timeout=0.05):
            if cancel is not None and cancel():
                raise InferenceCancelled("CASCADE reference submission cancelled.")
        try:
            if cancel is not None and cancel():
                raise InferenceCancelled("CASCADE reference submission cancelled.")
            return np.asarray(
                package.cascade.predict(
                    model.name,
                    dff,
                    model_folder=str(model.directory.parent),
                    threshold=0,
                    padding=0,
                    trace_noise_levels=noise,
                    verbosity=0,
                    device=package.torch.device(device),
                )
            )
        finally:
            self._prediction_lock.release()
