"""Batch results shared by trace extraction and spike inference backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class InferenceCancelled(RuntimeError):
    """A batch was cancelled; callers must discard its entire FOV."""


@dataclass(frozen=True)
class OasisResult:
    """Denoised calcium, spikes, and diagnostics in input ROI order."""

    den_dff: np.ndarray
    spikes: np.ndarray
    sn_by_roi: np.ndarray
    g_by_roi: np.ndarray
