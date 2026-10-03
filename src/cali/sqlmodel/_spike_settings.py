"""Validation shared by constructors, database writes, and runner preflight."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from cali._constants import (
    DEFAULT_BURST_GAUSS_SIGMA,
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CCG_N_SHUFFLES,
    DEFAULT_ENABLE_RISING_EDGE_ANALYSIS,
    DEFAULT_MIN_BURST_DURATION,
    DEFAULT_SPIKE_SYNC_JITTER_WINDOW,
    DEFAULT_SPIKE_SYNCHRONY_MAX_LAG,
    DEFAULT_SPIKE_THRESHOLD,
)

SpikeMethod = Literal["oasis", "cascade"]
CASCADE_AP_THRESHOLD = "cascade_ap"


def canonical_spike_methods(value: Any) -> tuple[SpikeMethod, ...]:
    """Accept list/tuple selections and return a nonempty, immutable ordering."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("spike_methods must be a nonempty list or tuple.")
    if any(method not in ("oasis", "cascade") for method in value):
        raise ValueError("spike_methods may contain only 'oasis' and 'cascade'.")
    ordered_methods: tuple[SpikeMethod, ...] = ("oasis", "cascade")
    return tuple(method for method in ordered_methods if method in value)


def require_available_spike_methods(methods: Any) -> None:
    """Keep stored CASCADE selections from running through the OASIS-only pipeline."""
    if "cascade" in canonical_spike_methods(methods):
        raise NotImplementedError(
            "CASCADE execution is not available yet. Select OASIS until the "
            "CASCADE backend and method-bound analysis are enabled."
        )


class ExtractionOutputSettings(BaseModel):
    """Validate extraction-owned output selection without loading a backend."""

    spike_methods: tuple[SpikeMethod, ...] = ("oasis",)
    cascade_model: str | None = None
    cascade_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    frame_rate_verified: bool = False
    discard_initial_value: float = 0.0
    discard_initial_unit: Literal["frames", "seconds"] = "frames"

    @model_validator(mode="before")
    @classmethod
    def normalize_selection(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            methods = canonical_spike_methods(data.get("spike_methods", ("oasis",)))
            data["spike_methods"] = methods
            if "cascade" not in methods:
                data["cascade_model"] = None
                data["cascade_device"] = "auto"
        return data

    @model_validator(mode="after")
    def validate_selection(self) -> ExtractionOutputSettings:
        if "cascade" in self.spike_methods:
            if not self.cascade_model or not self.cascade_model.strip():
                raise ValueError("CASCADE requires an explicit cascade_model.")
            self.cascade_model = self.cascade_model.strip()
        value = self.discard_initial_value
        if not math.isfinite(value) or value < 0:
            raise ValueError("Discard at Start must be finite and non-negative.")
        if self.discard_initial_unit == "frames" and not value.is_integer():
            raise ValueError("Discard at Start must be a whole number in Frames mode.")
        # Seconds-mode timing is validated against each source FOV at extraction.
        return self


class SpikeAnalysisParameters(BaseModel):
    """One method's threshold, bursts, synchrony, and CCG configuration."""

    method: SpikeMethod = "oasis"
    threshold_mode: Literal["global", "multiplier", "cascade_ap"] = "multiplier"
    threshold_value: float | None = DEFAULT_SPIKE_THRESHOLD
    cascade_ap_threshold_fraction: float | None = None
    burst_threshold: float = DEFAULT_BURST_THRESHOLD
    burst_min_duration: float = DEFAULT_MIN_BURST_DURATION
    burst_gaussian_sigma: float = DEFAULT_BURST_GAUSS_SIGMA
    spikes_sync_cross_corr_lag: float = DEFAULT_SPIKE_SYNCHRONY_MAX_LAG
    spikes_sync_jitter_window: float = DEFAULT_SPIKE_SYNC_JITTER_WINDOW
    ccg_n_shuffles: int = DEFAULT_CCG_N_SHUFFLES
    enable_rising_edge_analysis: bool = DEFAULT_ENABLE_RISING_EDGE_ANALYSIS

    @model_validator(mode="before")
    @classmethod
    def method_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("method") == "cascade":
            data = dict(data)
            data.setdefault("threshold_mode", CASCADE_AP_THRESHOLD)
            data.setdefault("threshold_value", None)
            data.setdefault("cascade_ap_threshold_fraction", 1 / math.e)
        return data

    @field_validator("threshold_value", "cascade_ap_threshold_fraction")
    @classmethod
    def finite_threshold(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("Spike thresholds must be finite and non-negative.")
        return value

    @model_validator(mode="after")
    def validate_threshold(self) -> SpikeAnalysisParameters:
        allowed = (
            {"global", "multiplier"}
            if self.method == "oasis"
            else {"global", CASCADE_AP_THRESHOLD}
        )
        if self.threshold_mode not in allowed:
            raise ValueError(
                f"Invalid threshold mode {self.threshold_mode!r} for {self.method}."
            )
        if self.threshold_mode == CASCADE_AP_THRESHOLD:
            fraction = self.cascade_ap_threshold_fraction
            if fraction is None or not 0 < fraction <= 1:
                raise ValueError("CASCADE AP threshold fraction must be in (0, 1].")
            self.threshold_value = None
        else:
            if self.threshold_value is None:
                raise ValueError(
                    "A global/multiplier threshold needs an explicit value."
                )
            self.cascade_ap_threshold_fraction = None
        return self


# Legacy public settings names remain OASIS compatibility accessors during migration.
LEGACY_SPIKE_SETTING_NAMES = {
    "spike_threshold_value": "threshold_value",
    "spike_threshold_mode": "threshold_mode",
    **{
        name: name
        for name in (
            "burst_threshold",
            "burst_min_duration",
            "burst_gaussian_sigma",
            "spikes_sync_cross_corr_lag",
            "spikes_sync_jitter_window",
            "ccg_n_shuffles",
            "enable_rising_edge_analysis",
        )
    },
}
