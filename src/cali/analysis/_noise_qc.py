"""Summaries and advisory comparisons on separate calcium/model noise scales."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from cali.logger import cali_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cali.sqlmodel import FOVAnalysis


def summarize_noise(values: Iterable[float]) -> tuple[float | None, float | None, int]:
    """Summarize known finite nonnegative estimates, never treating missing as zero."""
    known = [value for value in values if math.isfinite(value) and value >= 0]
    if not known:
        return None, None, 0
    q1, median, q3 = np.percentile(known, [25, 50, 75], method="linear")
    return float(median), float(q3 - q1), len(known)


class NoiseQCCollection:
    """Retain only scalar summaries for a completed batch's advisory noise checks."""

    def __init__(self) -> None:
        self._groups: dict[tuple, list[tuple[str, float]]] = {}

    def add(self, name: str, result: FOVAnalysis) -> None:
        """Separate calcium from CASCADE and separate verified model identities."""
        if result.calcium_noise_median is not None:
            self._groups.setdefault(("calcium/GetSn",), []).append(
                (name, result.calcium_noise_median)
            )
        child = result.get_spike_analysis("cascade")
        if child is None or child.model_noise_median is None:
            return
        run = child.inference_run
        if run is None or any(
            value is None
            for value in (
                run.resolved_model,
                run.weights_manifest_sha256,
                run.model_sampling_rate_hz,
            )
        ):
            return  # Unknown model identities cannot establish a comparison group.
        key = (
            "cascade/model noise",
            run.resolved_model,
            run.weights_manifest_sha256,
            run.model_sampling_rate_hz,
        )
        self._groups.setdefault(key, []).append((name, child.model_noise_median))

    def warn_outliers(self) -> None:
        """Warn beyond Q1/Q3 ± 3 IQR for groups of at least four known FOVs.

        This descriptive check never modifies results, flags, thresholds or model
        selection. A zero-IQR group has no useful spread estimate and is skipped.
        """
        for key, records in self._groups.items():
            if len(records) < 4:
                continue
            values = np.array([value for _, value in records], dtype=float)
            if not np.all(np.isfinite(values)):
                continue
            q1, q3 = np.percentile(values, [25, 75], method="linear")
            spread = q3 - q1
            if spread <= 0:
                continue
            for name, value in records:
                if value < q1 - 3 * spread or value > q3 + 3 * spread:
                    cali_logger.warning(
                        f"Noise QC: {name} {key[0]} median {value:.6g} is outside "
                        f"the batch's Q1/Q3 ± 3 IQR range "
                        f"[{q1 - 3 * spread:.6g}, {q3 + 3 * spread:.6g}]. "
                        "Review imaging quality; analysis results are unchanged."
                    )
