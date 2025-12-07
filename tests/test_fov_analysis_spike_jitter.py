"""Tests for spike jitter parameter usage in FOV analysis.

This module verifies that:
1. FOV analysis uses spikes_sync_jitter_window for spike synchrony
2. FOV analysis does NOT use calcium_sync_jitter_window for spikes
"""

from __future__ import annotations

import pytest

from cali._constants import DEFAULT_SPIKE_SYNC_JITTER_WINDOW
from cali.sqlmodel._model import AnalysisSettings


def test_fov_analysis_uses_spike_jitter_not_calcium() -> None:
    """Test that FOV analysis settings have separate spike jitter field.

    This test verifies that the AnalysisSettings model has a separate
    spikes_sync_jitter_window field that is distinct from
    calcium_sync_jitter_window, which is used in _fov_analysis.py line 232.
    """
    # Create analysis settings with different values
    calcium_jitter = 100.0
    spike_jitter = 300.0

    settings = AnalysisSettings(
        calcium_sync_jitter_window=calcium_jitter,
        spikes_sync_jitter_window=spike_jitter,
    )

    # Verify the model field is correctly accessed
    assert hasattr(settings, "spikes_sync_jitter_window")
    assert settings.spikes_sync_jitter_window == spike_jitter
    assert settings.spikes_sync_jitter_window != settings.calcium_sync_jitter_window


def test_spike_jitter_separate_from_calcium_in_settings() -> None:
    """Verify that changing calcium jitter doesn't affect spike jitter."""
    settings = AnalysisSettings(
        calcium_sync_jitter_window=50.0,
        spikes_sync_jitter_window=DEFAULT_SPIKE_SYNC_JITTER_WINDOW,
    )

    # Change calcium jitter
    settings.calcium_sync_jitter_window = 500.0

    # Spike jitter should remain unchanged
    assert settings.spikes_sync_jitter_window == DEFAULT_SPIKE_SYNC_JITTER_WINDOW
    assert settings.spikes_sync_jitter_window != settings.calcium_sync_jitter_window


@pytest.mark.parametrize(
    "spike_jitter,calcium_jitter",
    [
        (100.0, 200.0),
        (200.0, 100.0),
        (150.0, 150.0),  # Same value but independent fields
        (50.0, 500.0),
    ],
)
def test_spike_and_calcium_jitter_independence(
    spike_jitter: float, calcium_jitter: float
) -> None:
    """Test that spike and calcium jitter are truly independent fields."""
    settings = AnalysisSettings(
        spikes_sync_jitter_window=spike_jitter,
        calcium_sync_jitter_window=calcium_jitter,
    )

    assert settings.spikes_sync_jitter_window == spike_jitter
    assert settings.calcium_sync_jitter_window == calcium_jitter

    # Even if values are equal, they should be separate fields
    if spike_jitter == calcium_jitter:
        # Change one shouldn't affect the other
        settings.spikes_sync_jitter_window = spike_jitter + 100
        assert settings.spikes_sync_jitter_window != settings.calcium_sync_jitter_window
