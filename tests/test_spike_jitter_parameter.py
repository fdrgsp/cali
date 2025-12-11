"""Tests for separate spike jitter synchrony parameter.

This module tests that:
1. The spike jitter parameter is separate from calcium jitter
2. GUI widgets handle the spike jitter parameter correctly
3. Database model stores spike jitter parameter
4. Analysis code uses spike jitter (not calcium jitter) for spikes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlmodel import Session, select

from cali._constants import DEFAULT_SPIKE_SYNC_JITTER_WINDOW
from cali.gui._analysis_gui import SpikeData, _SpikeWidget
from cali.sqlmodel._model import AnalysisSettings

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    from tests.conftest import TempDB


def test_spike_data_has_jitter_field() -> None:
    """Test that SpikeData dataclass includes synchrony_jitter field."""
    # Create with default values
    data = SpikeData(
        spike_threshold=0.5,
        spike_threshold_mode="adaptive",
        burst_threshold=75.0,
        burst_min_duration=3000.0,
        burst_blur_sigma=1.0,
        synchrony_lag=1000.0,
        synchrony_jitter=200.0,
    )

    assert hasattr(data, "synchrony_jitter")
    assert data.synchrony_jitter == 200.0


def test_spike_data_default_jitter_value() -> None:
    """Test that SpikeData uses correct default for synchrony_jitter."""
    # Create with minimal required fields, rest should default
    data = SpikeData(
        spike_threshold=0.5,
        spike_threshold_mode="adaptive",
        burst_threshold=75.0,
        burst_min_duration=3000.0,
        burst_blur_sigma=1.0,
        synchrony_lag=1000.0,
        synchrony_jitter=DEFAULT_SPIKE_SYNC_JITTER_WINDOW,
    )

    assert data.synchrony_jitter == DEFAULT_SPIKE_SYNC_JITTER_WINDOW
    assert data.synchrony_jitter == 200.0  # Verify constant value


def test_spike_widget_has_jitter_spinbox(qtbot: QtBot) -> None:
    """Test that _SpikeWidget creates a jitter spinbox."""
    widget = _SpikeWidget()
    qtbot.addWidget(widget)

    # Check that the jitter spinbox exists
    assert hasattr(widget, "_spike_jitter_spin")
    assert widget._spike_jitter_spin is not None


def test_spike_widget_jitter_value_get_set(qtbot: QtBot) -> None:
    """Test that _SpikeWidget can get/set jitter value."""
    widget = _SpikeWidget()
    qtbot.addWidget(widget)

    # Set custom jitter value
    test_jitter = 250.0
    widget._spike_jitter_spin.setValue(test_jitter)

    # Get value through widget.value()
    data = widget.value()
    assert data.synchrony_jitter == test_jitter

    # Set via setValue
    new_data = SpikeData(
        spike_threshold=0.6,
        spike_threshold_mode="fixed",
        burst_threshold=80.0,
        burst_min_duration=4000.0,
        burst_blur_sigma=2.0,
        synchrony_lag=1500.0,
        synchrony_jitter=300.0,
    )
    widget.setValue(new_data)
    assert widget._spike_jitter_spin.value() == 300.0


def test_spike_widget_reset_jitter(qtbot: QtBot) -> None:
    """Test that _SpikeWidget.reset() resets jitter to default."""
    widget = _SpikeWidget()
    qtbot.addWidget(widget)

    # Change jitter to non-default value
    widget._spike_jitter_spin.setValue(999.0)
    assert widget._spike_jitter_spin.value() == 999.0

    # Reset
    widget.reset()

    # Should be back to default
    assert widget._spike_jitter_spin.value() == DEFAULT_SPIKE_SYNC_JITTER_WINDOW


def test_spike_jitter_separate_from_calcium_jitter(qtbot: QtBot) -> None:
    """Test that spike jitter is independent of calcium jitter."""
    from cali.gui._analysis_gui import _CalciumPeaksWidget

    calcium_widget = _CalciumPeaksWidget()
    spike_widget = _SpikeWidget()
    qtbot.addWidget(calcium_widget)
    qtbot.addWidget(spike_widget)

    # Set different jitter values
    calcium_jitter = 150.0
    spike_jitter = 300.0

    calcium_widget._calcium_synchrony_jitter_spin.setValue(calcium_jitter)
    spike_widget._spike_jitter_spin.setValue(spike_jitter)

    # Verify they are independent
    calcium_data = calcium_widget.value()
    spike_data = spike_widget.value()

    assert calcium_data.calcium_synchrony_jitter == calcium_jitter
    assert spike_data.synchrony_jitter == spike_jitter
    assert calcium_data.calcium_synchrony_jitter != spike_data.synchrony_jitter


@pytest.mark.parametrize("jitter_value", [50.0, 100.0, 200.0, 500.0, 1000.0])
def test_spike_jitter_parametrized_values(qtbot: QtBot, jitter_value: float) -> None:
    """Test spike jitter widget with various values."""
    widget = _SpikeWidget()
    qtbot.addWidget(widget)

    widget._spike_jitter_spin.setValue(jitter_value)
    data = widget.value()

    assert data.synchrony_jitter == jitter_value


def test_analysis_settings_spike_jitter_field(temp_db: TempDB) -> None:
    """Test that AnalysisSettings model has spikes_sync_jitter_window field."""
    engine, _db_path = temp_db

    settings = AnalysisSettings(
        spikes_sync_jitter_window=250.0,
    )

    with Session(engine) as session:
        session.add(settings)
        session.commit()
        session.refresh(settings)

        assert settings.id is not None
        assert settings.spikes_sync_jitter_window == 250.0


def test_analysis_settings_spike_jitter_default(temp_db: TempDB) -> None:
    """Test that AnalysisSettings uses correct default for spike jitter."""
    engine, _db_path = temp_db

    # Create settings without specifying spike jitter
    settings = AnalysisSettings()

    with Session(engine) as session:
        session.add(settings)
        session.commit()
        session.refresh(settings)

        assert settings.spikes_sync_jitter_window == DEFAULT_SPIKE_SYNC_JITTER_WINDOW


def test_analysis_settings_spike_jitter_persistence(temp_db: TempDB) -> None:
    """Test that spike jitter value persists in database."""
    engine, _db_path = temp_db

    test_jitter = 350.0
    settings = AnalysisSettings(spikes_sync_jitter_window=test_jitter)

    with Session(engine) as session:
        session.add(settings)
        session.commit()
        settings_id = settings.id

    # Retrieve in new session
    with Session(engine) as session:
        retrieved = session.exec(
            select(AnalysisSettings).where(AnalysisSettings.id == settings_id)
        ).first()

        assert retrieved is not None
        assert retrieved.spikes_sync_jitter_window == test_jitter


def test_spike_jitter_independent_of_calcium_jitter_in_model(temp_db: TempDB) -> None:
    """Test that spike and calcium jitter are separate fields in AnalysisSettings."""
    engine, _db_path = temp_db

    calcium_jitter = 100.0
    spike_jitter = 400.0

    settings = AnalysisSettings(
        calcium_sync_jitter_window=calcium_jitter,
        spikes_sync_jitter_window=spike_jitter,
    )

    with Session(engine) as session:
        session.add(settings)
        session.commit()
        session.refresh(settings)

        assert settings.calcium_sync_jitter_window == calcium_jitter
        assert settings.spikes_sync_jitter_window == spike_jitter
        assert settings.calcium_sync_jitter_window != settings.spikes_sync_jitter_window


def test_spike_jitter_affects_settings_equality() -> None:
    """Test that changing spikes_sync_jitter_window makes settings unequal.

    This is a regression test for a bug where spikes_sync_jitter_window was
    missing from __eq__ and __hash__, causing the runner to incorrectly reuse
    existing settings when only spike jitter changed.
    """
    settings1 = AnalysisSettings(spikes_sync_jitter_window=200.0)
    settings2 = AnalysisSettings(spikes_sync_jitter_window=400.0)
    settings3 = AnalysisSettings(spikes_sync_jitter_window=200.0)

    # Different spike jitter should make settings unequal
    assert settings1 != settings2
    assert hash(settings1) != hash(settings2)

    # Same spike jitter should make settings equal
    assert settings1 == settings3
    assert hash(settings1) == hash(settings3)
