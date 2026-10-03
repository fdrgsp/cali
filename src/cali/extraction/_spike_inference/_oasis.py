"""OASIS inference with one synchronous batch call per field of view."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from oasis.functions import GetSn, deconvolve, estimate_parameters

from cali.logger import cali_logger

from ._base import InferenceCancelled, OasisResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class OasisBackend:
    """Preserve the legacy per-ROI OASIS computation inside a batch interface."""

    name = "oasis"
    # Welch frequencies must include a bin strictly inside (0.25, 0.5).
    # Five is the smallest length after which every longer input has such a bin;
    # four frames produces an empty noise band and a NaN estimate.
    minimum_frames = 5

    def infer_all(
        self,
        dff: np.ndarray,
        frame_rate: float,
        *,
        decay_constant: float | None = None,
        roi_labels: Sequence[int] | None = None,
        fov_name: str = "",
        cancel: Callable[[], bool] | None = None,
    ) -> OasisResult:
        """Infer rows in order, checking cancellation before and after each ROI."""
        if dff.ndim != 2:
            raise ValueError("OASIS requires a (ROIs, frames) DFF matrix.")
        if dff.shape[1] < self.minimum_frames:
            raise ValueError(
                f"OASIS noise estimation requires at least {self.minimum_frames} "
                f"frames; received {dff.shape[1]}."
            )
        if not np.all(np.isfinite(dff)):
            raise ValueError("OASIS requires finite DFF values.")
        if roi_labels is not None and len(roi_labels) != len(dff):
            raise ValueError("ROI labels must match the number of DFF rows.")
        den_dff = np.empty(dff.shape, dtype=float)
        spikes = np.empty(dff.shape, dtype=float)
        noise = np.empty(len(dff), dtype=float)
        coefficients = np.empty((len(dff), 1), dtype=float)
        tau = decay_constant or 0.0
        for index, row in enumerate(dff):
            if cancel is not None and cancel():
                raise InferenceCancelled("OASIS batch cancelled.")
            label = roi_labels[index] if roi_labels is not None else index
            den_dff[index], spikes[index], noise[index], g = self._infer_row(
                row, frame_rate, tau, label, fov_name
            )
            coefficients[index] = g
            if cancel is not None and cancel():
                raise InferenceCancelled("OASIS batch cancelled.")
        return OasisResult(den_dff, spikes, noise, coefficients)

    def _infer_row(
        self,
        dff: np.ndarray,
        frame_rate: float,
        tau: float,
        label_value: int,
        fov_name: str,
    ) -> tuple[np.ndarray, np.ndarray, float, tuple[float, ...]]:
        """Run the legacy OASIS algorithm unchanged for one input row."""
        # run OASIS deconvolution on the dff trace
        if tau > 0.0:
            # User-provided decay constant τ → AR(1) coefficient g
            g1 = float(np.exp(-1.0 / (frame_rate * tau)))  # AR(1) coefficient
            g: tuple[float] | tuple[float, float] = (g1,)  # OASIS expects a tuple
            optimize_g = 0  # do NOT re-optimize g, user fixed it
            # Estimate noise from DFF trace
            sn = GetSn(dff, range_ff=[0.25, 0.5], method="median")
        else:
            # Estimate AR parameters + noise from ORIGINAL dff trace
            try:
                g_arr, sn = estimate_parameters(
                    dff,
                    p=1,  # AR(1)
                    range_ff=[0.25, 0.5],  # default
                    method="median",  # mean or logmexp
                    lags=10,
                    fudge_factor=0.98,
                )
                # make sure g is a tuple
                g = tuple(np.atleast_1d(g_arr))
            except (np.linalg.LinAlgError, ValueError) as e:
                # estimate_parameters can fail with LinAlgError on edge-case traces
                # (e.g., very short traces or numerical issues in autocorrelation)
                cali_logger.warning(
                    f"⚠️ OASIS parameter estimation failed for ROI {label_value} in "
                    f"{fov_name}: {e}. Using default AR1 coefficient (0.95,)."
                )
                g = (0.95,)  # fallback stable AR(1) coefficient
                sn = GetSn(dff, range_ff=[0.25, 0.5], method="median")
            optimize_g = 0  # set >0 only if you really want refine

        # Deconvolve with error handling for invalid AR coefficients
        effective_g = g
        try:
            den_dff, spikes, _b, _g_fit, _lam = deconvolve(
                dff,
                g=g,
                sn=sn,
                penalty=1,  # L1 sparsity (standard OASIS)
                optimize_g=optimize_g,
            )
        except (ValueError, RuntimeError) as e:
            # If OASIS fails due to invalid AR coefficients, fall back to stable default
            cali_logger.warning(
                f"⚠️ OASIS deconvolution failed for ROI {label_value} in {fov_name}: "
                f"{e}. Retrying with stable default AR1 coefficient (0.95,)."
            )
            optimize_g = 0
            effective_g = (0.95,)
            den_dff, spikes, _b, _g_fit, _lam = deconvolve(
                dff,
                g=(0.95,),  # fallback stable AR(1) coefficient
                sn=sn,
                penalty=1,  # L1 sparsity (standard OASIS)
                optimize_g=optimize_g,
            )

        cali_logger.debug(f"OASIS params ROI {label_value}: g={g}, sn={sn}")

        # Convert to float
        den_dff = den_dff.astype(float)
        spikes = spikes.astype(float)

        return den_dff, spikes, sn, effective_g
