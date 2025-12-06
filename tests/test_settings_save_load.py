"""Tests for save and load settings functionality in CaliGui."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from pytestqt.qtbot import QtBot


@pytest.fixture
def settings_file(tmp_path: Path) -> Path:
    """Provide a temporary path for settings file."""
    return tmp_path / "test_settings.json"


@pytest.fixture
def cali_gui(qtbot: QtBot) -> Generator[Any, None, None]:
    """Create a CaliGui instance for testing."""
    from cali.gui import CaliGui

    gui = CaliGui()
    qtbot.addWidget(gui)
    yield gui
    gui.close()


def test_save_settings_creates_valid_json(
    cali_gui: Any, settings_file: Path, qtbot: QtBot
) -> None:
    """Test that saving settings creates a valid JSON file with all expected fields."""
    # Mock the file dialog to return our test file path
    from unittest.mock import patch

    with patch(
        "cali.gui._cali_gui.QFileDialog.getSaveFileName",
        return_value=(str(settings_file), ""),
    ):
        cali_gui._on_save_settings()

    # Verify file was created
    assert settings_file.exists()

    # Load and verify JSON structure
    with open(settings_file) as f:
        settings = json.load(f)

    # Verify top-level structure
    assert "detection" in settings
    assert "extraction" in settings
    assert "analysis" in settings

    # Verify detection settings exist (checking for Cellpose default fields)
    detection = settings["detection"]
    assert "model_type" in detection
    assert "cellprob_threshold" in detection
    assert "flow_threshold" in detection

    # Verify extraction settings exist
    extraction = settings["extraction"]
    assert "trace_extraction_data" in extraction
    trace_data = extraction["trace_extraction_data"]
    assert "frame_rate" in trace_data
    assert "decay_constant" in trace_data
    assert "dff_window_size" in trace_data

    # Verify analysis settings exist
    analysis = settings["analysis"]
    assert "calcium_peaks_data" in analysis
    assert "spikes_data" in analysis
    assert "experiment_type_data" in analysis


def test_load_settings_restores_gui_state(
    cali_gui: Any, settings_file: Path, qtbot: QtBot
) -> None:
    """Test that loading settings correctly restores GUI widget values."""
    # Create a settings file with known values in nested format
    test_settings = {
        "detection": {
            "model_type": "cyto3",
            "model_path": None,
            "diameter": 30.0,
            "cellprob_threshold": 0.3,
            "flow_threshold": 0.5,
            "min_size": 15,
            "normalize": False,
            "batch_size": 16,
        },
        "extraction": {
            "trace_extraction_data": {
                "dff_window_size": 2500.0,
                "decay_constant": 0.6,
                "frame_rate": 25.0,
                "neuropil_inner_radius": 4,
                "neuropil_min_pixels": 150,
                "neuropil_correction_factor": 0.8,
            }
        },
        "analysis": {
            "calcium_peaks_data": {
                "peaks_height": 4.0,
                "peaks_height_mode": "global",
                "peaks_distance": 300.0,
                "peaks_prominence_multiplier": 0.4,
                "calcium_synchrony_jitter": 250.0,
                "calcium_peaks_max_lag": 1200.0,
                "calcium_network_threshold": 0.4,
            },
            "spikes_data": {
                "spike_threshold": 5.0,
                "spike_threshold_mode": "global",
                "burst_threshold": 75.0,
                "burst_min_duration": 600.0,
                "burst_blur_sigma": 0.06,
                "synchrony_lag": 60.0,
            },
            "experiment_type_data": {
                "experiment_type": "Spontaneous Activity",
                "led_power_equation": None,
                "led_pulse_duration": None,
                "led_pulse_powers": None,
                "led_pulse_on_frames": None,
                "stimulation_area_path": None,
            },
        },
    }

    with open(settings_file, "w") as f:
        json.dump(test_settings, f)

    # Mock the file dialog to return our test file path
    from unittest.mock import patch

    with patch(
        "cali.gui._cali_gui.QFileDialog.getOpenFileName",
        return_value=(str(settings_file), ""),
    ):
        cali_gui._on_load_settings()

    # Verify settings were loaded by checking the widget values
    detection_value = cali_gui._detection_wdg.value()
    extraction_value = cali_gui._extraction_wdg.value()
    analysis_value = cali_gui._analysis_wdg.value()

    # Verify detection settings
    assert detection_value.diameter == 30.0
    assert detection_value.cellprob_threshold == 0.3
    assert detection_value.flow_threshold == 0.5

    # Verify extraction settings
    assert extraction_value.trace_extraction_data is not None
    assert extraction_value.trace_extraction_data.frame_rate == 25.0
    assert extraction_value.trace_extraction_data.decay_constant == 0.6
    assert extraction_value.trace_extraction_data.dff_window_size == 2500.0

    # Verify analysis settings
    assert analysis_value.calcium_peaks_data is not None
    assert analysis_value.calcium_peaks_data.peaks_height == 4.0
    assert analysis_value.calcium_peaks_data.peaks_distance == 300.0
    assert analysis_value.spikes_data is not None
    assert analysis_value.spikes_data.spike_threshold == 5.0
    assert analysis_value.spikes_data.burst_threshold == 75.0


def test_save_and_load_roundtrip(
    cali_gui: Any, settings_file: Path, qtbot: QtBot
) -> None:
    """Test that save followed by load preserves all settings."""
    # Get original values
    original_detection = cali_gui._detection_wdg.value()
    original_extraction = cali_gui._extraction_wdg.value()
    original_analysis = cali_gui._analysis_wdg.value()

    # Save settings
    from unittest.mock import patch

    with patch(
        "cali.gui._cali_gui.QFileDialog.getSaveFileName",
        return_value=(str(settings_file), ""),
    ):
        cali_gui._on_save_settings()

    # Verify file was created
    assert settings_file.exists()

    # Load settings back
    with patch(
        "cali.gui._cali_gui.QFileDialog.getOpenFileName",
        return_value=(str(settings_file), ""),
    ):
        cali_gui._on_load_settings()

    # Verify values match after round trip
    loaded_detection = cali_gui._detection_wdg.value()
    loaded_extraction = cali_gui._extraction_wdg.value()
    loaded_analysis = cali_gui._analysis_wdg.value()

    # Check detection settings
    assert loaded_detection.model_type == original_detection.model_type
    assert loaded_detection.diameter == original_detection.diameter

    # Check extraction settings
    assert loaded_extraction.trace_extraction_data is not None
    assert original_extraction.trace_extraction_data is not None
    assert (
        loaded_extraction.trace_extraction_data.frame_rate
        == original_extraction.trace_extraction_data.frame_rate
    )
    assert (
        loaded_extraction.trace_extraction_data.decay_constant
        == original_extraction.trace_extraction_data.decay_constant
    )

    # Check analysis settings
    assert loaded_analysis.calcium_peaks_data is not None
    assert original_analysis.calcium_peaks_data is not None
    assert (
        loaded_analysis.calcium_peaks_data.peaks_height
        == original_analysis.calcium_peaks_data.peaks_height
    )
    assert loaded_analysis.spikes_data is not None
    assert original_analysis.spikes_data is not None
    assert (
        loaded_analysis.spikes_data.spike_threshold
        == original_analysis.spikes_data.spike_threshold
    )


def test_load_settings_with_missing_fields_uses_defaults(
    cali_gui: Any, settings_file: Path, qtbot: QtBot
) -> None:
    """Test that loading settings with missing fields uses defaults."""
    # Create settings with only detection data (missing extraction and analysis)
    partial_settings = {
        "detection": {
            "model_type": "cpsam",  # Use default model type
            "diameter": 25.0,
            "cellprob_threshold": -2.0,
            "flow_threshold": 0.8,
        }
    }

    settings_file.write_text(json.dumps(partial_settings))

    # Load settings
    from unittest.mock import patch

    # Mock show_error_dialog to prevent blocking modal and suppress error logs
    with (
        patch(
            "cali.gui._cali_gui.QFileDialog.getOpenFileName",
            return_value=(str(settings_file), ""),
        ),
        patch("cali.gui._cali_gui.show_error_dialog"),
        patch("cali.gui._cali_gui.cali_logger.error"),
    ):
        cali_gui._on_load_settings()

    # Verify detection settings were loaded
    loaded_detection = cali_gui._detection_wdg.value()
    assert loaded_detection.model_type == "cpsam"
    assert loaded_detection.diameter == 25.0
    assert loaded_detection.cellprob_threshold == -2.0
    assert loaded_detection.flow_threshold == 0.8

    # Verify extraction and analysis use defaults (should not be None)
    loaded_extraction = cali_gui._extraction_wdg.value()
    assert loaded_extraction.trace_extraction_data is not None

    loaded_analysis = cali_gui._analysis_wdg.value()
    assert loaded_analysis.calcium_peaks_data is not None
    assert loaded_analysis.spikes_data is not None


def test_load_settings_handles_invalid_json(
    cali_gui: Any, settings_file: Path, qtbot: QtBot
) -> None:
    """Test that loading invalid JSON shows error and doesn't crash."""
    # Create invalid JSON file
    with open(settings_file, "w") as f:
        f.write("{ invalid json content }")

    from unittest.mock import patch

    # Mock show_error_dialog to prevent blocking modal and suppress error logs
    with (
        patch(
            "cali.gui._cali_gui.QFileDialog.getOpenFileName",
            return_value=(str(settings_file), ""),
        ),
        patch("cali.gui._cali_gui.show_error_dialog"),
        patch("cali.gui._cali_gui.cali_logger.error"),
    ):
        # Should not raise exception, just log error
        cali_gui._on_load_settings()


def test_load_settings_cancel_does_nothing(cali_gui: Any, qtbot: QtBot) -> None:
    """Test that canceling the load dialog doesn't change settings."""
    original_detection = cali_gui._detection_wdg.value()
    original_extraction = cali_gui._extraction_wdg.value()
    original_analysis = cali_gui._analysis_wdg.value()

    from unittest.mock import patch

    # Mock user canceling the dialog (returns empty string)
    with patch(
        "cali.gui._cali_gui.QFileDialog.getOpenFileName",
        return_value=("", ""),
    ):
        cali_gui._on_load_settings()

    # Verify nothing changed
    loaded_detection = cali_gui._detection_wdg.value()
    loaded_extraction = cali_gui._extraction_wdg.value()
    loaded_analysis = cali_gui._analysis_wdg.value()

    assert loaded_detection.diameter == original_detection.diameter
    assert (
        loaded_extraction.trace_extraction_data.frame_rate
        == original_extraction.trace_extraction_data.frame_rate
    )
    assert (
        loaded_analysis.calcium_peaks_data.peaks_height
        == original_analysis.calcium_peaks_data.peaks_height
    )


def test_save_settings_cancel_does_nothing(
    cali_gui: Any, tmp_path: Path, qtbot: QtBot
) -> None:
    """Test that canceling the save dialog doesn't create a file."""
    from unittest.mock import patch

    # Mock user canceling the dialog (returns empty string)
    with patch(
        "cali.gui._cali_gui.QFileDialog.getSaveFileName",
        return_value=("", ""),
    ):
        cali_gui._on_save_settings()

    # Verify no file was created
    assert not any(tmp_path.glob("*.json"))


def test_load_settings_with_evoked_experiment_data(
    cali_gui: Any, settings_file: Path, qtbot: QtBot
) -> None:
    """Test loading settings with evoked experiment configuration."""
    evoked_settings = {
        "detection": {
            "model_type": "cyto3",
            "diameter": 30.0,
            "cellprob_threshold": 0.0,
            "flow_threshold": 0.4,
        },
        "extraction": {
            "trace_extraction_data": {
                "dff_window_size": 3000.0,
                "decay_constant": 0.4,
                "frame_rate": 30.0,
                "neuropil_inner_radius": 2,
                "neuropil_min_pixels": 100,
                "neuropil_correction_factor": 0.7,
            }
        },
        "analysis": {
            "calcium_peaks_data": {
                "peaks_height": 2.0,
                "peaks_height_mode": "multiplier",
                "peaks_distance": 200.0,
                "peaks_prominence_multiplier": 0.33,
                "calcium_synchrony_jitter": 200.0,
                "calcium_peaks_max_lag": 1000.0,
                "calcium_network_threshold": 0.3,
            },
            "spikes_data": {
                "spike_threshold": 3.0,
                "spike_threshold_mode": "multiplier",
                "burst_threshold": 65.0,
                "burst_min_duration": 500.0,
                "burst_blur_sigma": 0.05,
                "synchrony_lag": 50.0,
            },
            "experiment_type_data": {
                "experiment_type": "Evoked Activity",
                "led_power_equation": "y = 2.5*x + 10",
                "led_pulse_duration": 5.0,
                "led_pulse_powers": [10.0, 20.0, 30.0],
                "led_pulse_on_frames": [10, 20, 30],
                "stimulation_area_path": r"/path/to/stim/mask.tif",
            },
        },
    }

    settings_file.write_text(json.dumps(evoked_settings))

    from unittest.mock import patch

    with patch(
        "cali.gui._cali_gui.QFileDialog.getOpenFileName",
        return_value=(str(settings_file), ""),
    ):
        cali_gui._on_load_settings()

    # Verify evoked-specific fields were loaded
    loaded_analysis = cali_gui._analysis_wdg.value()
    assert loaded_analysis.experiment_type_data is not None
    assert loaded_analysis.experiment_type_data.experiment_type == "Evoked Activity"
    assert loaded_analysis.experiment_type_data.led_power_equation == "y = 2.5*x + 10"
    assert loaded_analysis.experiment_type_data.led_pulse_duration == 5.0
    assert loaded_analysis.experiment_type_data.led_pulse_powers == [10.0, 20.0, 30.0]
    assert loaded_analysis.experiment_type_data.led_pulse_on_frames == [10, 20, 30]
    # Normalize path for cross-platform comparison (Windows uses backslashes)
    assert (
        loaded_analysis.experiment_type_data.stimulation_area_path.replace("\\", "/")
        == "/path/to/stim/mask.tif"
    )
