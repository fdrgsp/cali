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
    assert "use_gpu" in detection

    # Verify extraction settings exist
    extraction = settings["extraction"]
    assert "trace_extraction_data" in extraction
    trace_data = extraction["trace_extraction_data"]
    assert "decay_constant" in trace_data
    assert "dff_window_size" in trace_data
    assert "dff_percentile" in trace_data
    assert "metadata_data" in extraction
    metadata_data = extraction["metadata_data"]
    assert "frame_rate" in metadata_data

    # Verify export options are saved
    assert "export_options" in extraction
    assert "export_enabled" in extraction
    assert isinstance(extraction["export_options"], dict)

    # Verify analysis settings exist
    analysis = settings["analysis"]
    assert "calcium_peaks_data" in analysis
    assert "spikes_data" in analysis
    assert "experiment_type_data" in analysis
    calcium_peaks = analysis["calcium_peaks_data"]
    assert "cluster_n_clusters" in calcium_peaks
    assert "cluster_max_k" in calcium_peaks

    # Verify analysis export options are saved
    assert "export_options" in analysis
    assert "export_enabled" in analysis
    assert isinstance(analysis["export_options"], dict)


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
            "use_gpu": False,
        },
        "extraction": {
            "trace_extraction_data": {
                "dff_window_size": 2.5,
                "dff_percentile": 15,
                "decay_constant": 0.6,
                "frame_rate": 25.0,
                "neuropil_inner_radius": 4,
                "neuropil_min_pixels": 150,
                "neuropil_correction_factor": 0.8,
            },
            "export_options": {
                "Raw Calcium Traces": [False, 0, 0],
                "\u0394F/F Traces": [True, 3, 0],
            },
            "export_enabled": False,
        },
        "analysis": {
            "calcium_peaks_data": {
                "peaks_height": 4.0,
                "peaks_height_mode": "global",
                "peaks_distance": 300.0,
                "peaks_prominence_multiplier": 0.4,
                "burst_threshold": 65.0,
                "burst_min_duration": 600.0,
                "burst_blur_sigma": 0.06,
                "cluster_n_clusters": 3,
                "cluster_max_k": 8,
            },
            "spikes_data": {
                "spike_threshold": 5.0,
                "spike_threshold_mode": "global",
                "burst_threshold": 75.0,
                "burst_min_duration": 600.0,
                "burst_blur_sigma": 0.06,
                "synchrony_lag": 600.0,
                "synchrony_jitter": 250.0,
                "ccg_n_shuffles": 50,
                "enable_rising_edge_analysis": True,
            },
            "experiment_type_data": {
                "experiment_type": "Spontaneous Activity",
                "led_power_equation": None,
                "led_pulse_duration": None,
                "led_pulse_powers": None,
                "led_pulse_on_frames": None,
                "stimulation_area_path": None,
            },
            "export_options": {
                "\u0394F/F Correlation Matrix": [True, 1, 0],
            },
            "export_enabled": True,
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
    assert detection_value.use_gpu is False

    # Verify extraction settings
    assert extraction_value.trace_extraction_data is not None
    assert extraction_value.trace_extraction_data.frame_rate == 10.0
    assert extraction_value.trace_extraction_data.decay_constant == 0.6
    assert extraction_value.trace_extraction_data.dff_window_size == 2.5
    assert extraction_value.trace_extraction_data.dff_percentile == 15

    # Verify analysis settings
    assert analysis_value.calcium_peaks_data is not None
    assert analysis_value.calcium_peaks_data.peaks_height == 4.0
    assert analysis_value.calcium_peaks_data.peaks_distance == 300.0
    assert analysis_value.calcium_peaks_data.burst_threshold == 65.0
    assert analysis_value.calcium_peaks_data.cluster_n_clusters == 3
    assert analysis_value.calcium_peaks_data.cluster_max_k == 8
    assert analysis_value.spikes_data is not None
    assert analysis_value.spikes_data.spike_threshold == 5.0
    assert analysis_value.spikes_data.burst_threshold == 75.0
    assert analysis_value.spikes_data.synchrony_lag == 600.0
    assert analysis_value.spikes_data.synchrony_jitter == 250.0
    assert analysis_value.spikes_data.ccg_n_shuffles == 50
    assert analysis_value.spikes_data.enable_rising_edge_analysis is True

    # Verify extraction export options were loaded
    assert extraction_value.export_enabled is False
    assert extraction_value.export_options is not None
    raw_checked = extraction_value.export_options.get("Raw Calcium Traces")
    assert raw_checked is not None
    assert raw_checked[0] is False  # checked state
    dff_checked = extraction_value.export_options.get("\u0394F/F Traces")
    assert dff_checked is not None
    assert dff_checked[0] is True

    # Verify analysis export options were loaded
    assert analysis_value.export_enabled is True
    assert analysis_value.export_options is not None
    corr_checked = analysis_value.export_options.get("\u0394F/F Correlation Matrix")
    assert corr_checked is not None
    assert corr_checked[0] is True


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
    assert loaded_detection.use_gpu == original_detection.use_gpu

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
    assert (
        loaded_extraction.trace_extraction_data.dff_percentile
        == original_extraction.trace_extraction_data.dff_percentile
    )

    # Check analysis settings
    assert loaded_analysis.calcium_peaks_data is not None
    assert original_analysis.calcium_peaks_data is not None
    assert (
        loaded_analysis.calcium_peaks_data.peaks_height
        == original_analysis.calcium_peaks_data.peaks_height
    )
    assert (
        loaded_analysis.calcium_peaks_data.cluster_n_clusters
        == original_analysis.calcium_peaks_data.cluster_n_clusters
    )
    assert (
        loaded_analysis.calcium_peaks_data.cluster_max_k
        == original_analysis.calcium_peaks_data.cluster_max_k
    )
    assert loaded_analysis.spikes_data is not None
    assert original_analysis.spikes_data is not None
    assert (
        loaded_analysis.spikes_data.spike_threshold
        == original_analysis.spikes_data.spike_threshold
    )
    assert (
        loaded_analysis.spikes_data.synchrony_jitter
        == original_analysis.spikes_data.synchrony_jitter
    )
    assert (
        loaded_analysis.spikes_data.ccg_n_shuffles
        == original_analysis.spikes_data.ccg_n_shuffles
    )
    assert (
        loaded_analysis.spikes_data.enable_rising_edge_analysis
        == original_analysis.spikes_data.enable_rising_edge_analysis
    )

    # Check extraction export options preserved in round trip
    assert loaded_extraction.export_enabled == original_extraction.export_enabled
    assert loaded_extraction.export_options == original_extraction.export_options

    # Check analysis export options preserved in round trip
    assert loaded_analysis.export_enabled == original_analysis.export_enabled
    assert loaded_analysis.export_options == original_analysis.export_options


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
            # Note: use_gpu is intentionally omitted to test default handling
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
    # use_gpu should use default value (True) when not specified
    assert loaded_detection.use_gpu is True

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
            "use_gpu": True,
        },
        "extraction": {
            "trace_extraction_data": {
                "dff_window_size": 3.0,
                "dff_percentile": 20,
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
                "burst_threshold": 65.0,
                "burst_min_duration": 500.0,
                "burst_blur_sigma": 0.05,
            },
            "spikes_data": {
                "spike_threshold": 3.0,
                "spike_threshold_mode": "multiplier",
                "burst_threshold": 65.0,
                "burst_min_duration": 500.0,
                "burst_blur_sigma": 0.05,
                "synchrony_lag": 500.0,
                "synchrony_jitter": 200.0,
                "ccg_n_shuffles": 40,
                "enable_rising_edge_analysis": True,
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

    # Verify spike settings including ccg and rising edge
    assert loaded_analysis.spikes_data is not None
    assert loaded_analysis.spikes_data.ccg_n_shuffles == 40
    assert loaded_analysis.spikes_data.enable_rising_edge_analysis is True


def test_run_selection_loads_all_settings(
    cali_gui: Any, qtbot: QtBot, test_db_copy: Path
) -> None:
    """Test that selecting run loads all extraction, analysis, and detection
    settings."""
    # Use the test database which has runs with dff_percentile=10
    data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"

    # Initialize the GUI from the test database
    cali_gui._initialize_from_database(str(test_db_copy), data_path)

    # Set a different value first to verify it changes
    from cali.gui._extraction_gui import (
        ExtractionSettingsData,
        MetadataData,
        TraceExtractionData,
    )

    cali_gui._extraction_wdg.setValue(
        ExtractionSettingsData(
            trace_extraction_data=TraceExtractionData(
                dff_window_size=5.0,
                dff_percentile=50,  # Different from the db value
                decay_constant=1.0,
            ),
            metadata_data=MetadataData(frame_rate=20.0),
        )
    )

    # Verify the value was set
    assert cali_gui._extraction_wdg.value().trace_extraction_data.dff_percentile == 50

    # Simulate selecting run 1 (which has dff_percentile=10)
    cali_gui._on_run_item_selected(1)

    # Verify all extraction settings were loaded from the database
    extraction_value = cali_gui._extraction_wdg.value()
    assert extraction_value.trace_extraction_data is not None
    assert extraction_value.trace_extraction_data.dff_percentile == 10
    assert extraction_value.trace_extraction_data.dff_window_size == 10.0
    assert extraction_value.trace_extraction_data.decay_constant == 0.0
    assert extraction_value.trace_extraction_data.neuropil_inner_radius == 2
    assert extraction_value.trace_extraction_data.neuropil_min_pixels == 200
    assert extraction_value.trace_extraction_data.neuropil_correction_factor == 0.6
    assert extraction_value.metadata_data is not None
    assert extraction_value.metadata_data.frame_rate == 10.0
    assert extraction_value.metadata_data.pixel_size == 0.65
    assert extraction_value.threads == 3

    # Verify all analysis settings were loaded from the database
    analysis_value = cali_gui._analysis_wdg.value()
    assert analysis_value.calcium_peaks_data is not None
    assert analysis_value.calcium_peaks_data.peaks_height == 2.0
    assert analysis_value.calcium_peaks_data.peaks_height_mode == "multiplier"
    assert analysis_value.calcium_peaks_data.peaks_distance == 200.0
    assert analysis_value.calcium_peaks_data.peaks_prominence_multiplier == 1.0
    assert analysis_value.calcium_peaks_data.burst_threshold == 50.0
    assert analysis_value.calcium_peaks_data.burst_min_duration == 100.0
    assert analysis_value.calcium_peaks_data.burst_blur_sigma == 0.01
    assert analysis_value.calcium_peaks_data.cluster_n_clusters == 0
    assert analysis_value.calcium_peaks_data.cluster_max_k == 10
    assert analysis_value.spikes_data is not None
    assert analysis_value.spikes_data.spike_threshold == 1.0
    assert analysis_value.spikes_data.spike_threshold_mode == "multiplier"
    assert analysis_value.spikes_data.burst_threshold == 50.0
    assert analysis_value.spikes_data.burst_min_duration == 100.0
    assert analysis_value.spikes_data.burst_blur_sigma == 0.01
    assert analysis_value.spikes_data.synchrony_lag == 500.0
    assert analysis_value.spikes_data.synchrony_jitter == 200.0
    assert analysis_value.spikes_data.ccg_n_shuffles == 30
    assert analysis_value.spikes_data.enable_rising_edge_analysis is True
    assert analysis_value.experiment_type_data is not None
    assert analysis_value.experiment_type_data.experiment_type == "Evoked Activity"
    assert analysis_value.experiment_type_data.led_power_equation == ""
    assert analysis_value.experiment_type_data.led_pulse_duration == 100.0
    assert analysis_value.experiment_type_data.led_pulse_powers == [2.0, 4.0, 6.0]
    assert analysis_value.experiment_type_data.led_pulse_on_frames == [3, 53, 103]
    expected_path = (
        "/Users/fdrgsp/Documents/git/cali/tests/test_data/data_and_db_for_tests/"
        "stimulation_mask.tif"
    )
    assert (
        analysis_value.experiment_type_data.stimulation_area_path.replace("\\", "/")
        == expected_path
    )
    assert analysis_value.threads == 3


def test_run_selection_preserves_export_options_and_enabled(
    cali_gui: Any, qtbot: QtBot, test_db_copy: Path
) -> None:
    """Selecting a run preserves extraction/analysis export state.

    Loading a run's settings must not reset the export_options or
    export_enabled values the user had configured.
    """
    data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"

    cali_gui._initialize_from_database(str(test_db_copy), data_path)

    # Set export-enabled to the non-default state for each widget:
    # extraction defaults True → set False
    # analysis defaults False → set True
    cali_gui._extraction_wdg._export_group.setChecked(False)
    cali_gui._analysis_wdg._export_group.setChecked(True)

    cali_gui._on_run_item_selected(1)

    # Both export states must survive the settings reload
    assert cali_gui._extraction_wdg.value().export_enabled is False
    assert cali_gui._analysis_wdg.value().export_enabled is True


def test_load_settings_legacy_cluster_data_key(
    cali_gui: Any, settings_file: Path, qtbot: QtBot
) -> None:
    """Loading settings with a legacy 'cluster_data' key migrates values into
    calcium_peaks_data.

    Legacy settings files stored cluster_n_clusters/cluster_max_k under a
    separate 'cluster_data' dict in analysis.  The loader must copy those values
    into calcium_peaks_data when the modern key is missing.
    """
    # Build a settings dict that uses the OLD 'cluster_data' layout
    legacy_settings = {
        "detection": {},
        "extraction": {},
        "analysis": {
            "calcium_peaks_data": {
                "peaks_height": 3.0,
                "peaks_height_mode": "global",
                "peaks_distance": 200.0,
                "peaks_prominence_multiplier": 0.3,
                "burst_threshold": 60.0,
                "burst_min_duration": 500.0,
                "burst_blur_sigma": 0.05,
                # NOTE: no 'cluster_n_clusters' / 'cluster_max_k' here
            },
            # Legacy key that pre-dates the inline cluster fields
            "cluster_data": {
                "cluster_n_clusters": 4,
                "cluster_max_k": 12,
            },
        },
    }

    with open(settings_file, "w") as f:
        json.dump(legacy_settings, f)

    from unittest.mock import patch

    with patch(
        "cali.gui._cali_gui.QFileDialog.getOpenFileName",
        return_value=(str(settings_file), ""),
    ):
        cali_gui._on_load_settings()

    analysis_value = cali_gui._analysis_wdg.value()
    assert analysis_value.calcium_peaks_data is not None
    # Values should have been migrated from the legacy 'cluster_data' key
    assert analysis_value.calcium_peaks_data.cluster_n_clusters == 4
    assert analysis_value.calcium_peaks_data.cluster_max_k == 12


# ---------------------------------------------------------------------------
# Pure-JSON serialisation round-trip (no Qt required)
# ---------------------------------------------------------------------------


def test_settings_json_round_trip(tmp_path: Path) -> None:
    """JSON save/load of all settings dataclasses preserves every field."""
    import json
    import os
    from dataclasses import asdict

    from cali._constants import DEFAULT_FRAME_RATE
    from cali.gui._analysis_gui import (
        AnalysisSettingsData,
        CalciumPeaksData,
        ExperimentTypeData,
        SpikeData,
    )
    from cali.gui._detection_gui import CellposeSettingsData
    from cali.gui._extraction_gui import (
        ExtractionSettingsData,
        MetadataData,
        TraceExtractionData,
    )

    detection = CellposeSettingsData(
        model_type="cyto3", diameter=30.0, cellprob_threshold=0.5, use_gpu=False
    )
    extraction = ExtractionSettingsData(
        trace_extraction_data=TraceExtractionData(dff_window_size=5.0, frame_rate=20.0),
        metadata_data=MetadataData(pixel_size=0.5, frame_rate=20.0),
        threads=2,
    )
    analysis = AnalysisSettingsData(
        calcium_peaks_data=CalciumPeaksData(peaks_height=5.0),
        spikes_data=SpikeData(spike_threshold=2.0),
        experiment_type_data=ExperimentTypeData(
            experiment_type="Evoked Activity",
            led_pulse_powers=[1.0, 2.0],
            led_pulse_on_frames=[5, 10],
        ),
        frame_rate=20.0,
        threads=2,
        n_processes=4,
    )

    full_settings = {
        "detection": asdict(detection),
        "extraction": asdict(extraction),
        "analysis": asdict(analysis),
    }
    json_path = tmp_path / "settings.json"
    with open(json_path, "w") as f:
        json.dump(full_settings, f)

    with open(json_path) as f:
        settings = json.load(f)

    det_loaded = CellposeSettingsData(**settings["detection"])
    assert det_loaded == detection

    ext = settings["extraction"]
    ext_export_options = ext.get("export_options")
    if ext_export_options is not None:
        ext_export_options = {k: tuple(v) for k, v in ext_export_options.items()}
    ext_loaded = ExtractionSettingsData(
        trace_extraction_data=(
            TraceExtractionData(**ext["trace_extraction_data"])
            if ext.get("trace_extraction_data")
            else None
        ),
        metadata_data=(
            MetadataData(**ext["metadata_data"]) if ext.get("metadata_data") else None
        ),
        threads=ext.get("threads", max((os.cpu_count() or 1) - 2, 1)),
        export_options=ext_export_options,
        export_enabled=ext.get("export_enabled", True),
    )
    assert ext_loaded.threads == extraction.threads
    assert ext_loaded.trace_extraction_data == extraction.trace_extraction_data
    assert ext_loaded.metadata_data == extraction.metadata_data

    ana = settings["analysis"]
    ana_export_options = ana.get("export_options")
    if ana_export_options is not None:
        ana_export_options = {k: tuple(v) for k, v in ana_export_options.items()}
    _default_threads = max((os.cpu_count() or 1) - 2, 1)
    ana_loaded = AnalysisSettingsData(
        calcium_peaks_data=(
            CalciumPeaksData(**ana["calcium_peaks_data"])
            if ana.get("calcium_peaks_data")
            else None
        ),
        spikes_data=(
            SpikeData(**ana["spikes_data"]) if ana.get("spikes_data") else None
        ),
        experiment_type_data=(
            ExperimentTypeData(**ana["experiment_type_data"])
            if ana.get("experiment_type_data")
            else None
        ),
        frame_rate=ana.get("frame_rate", DEFAULT_FRAME_RATE),
        threads=ana.get("threads", _default_threads),
        n_processes=ana.get("n_processes", _default_threads),
        export_options=ana_export_options,
        export_enabled=ana.get("export_enabled", False),
    )
    assert ana_loaded.frame_rate == analysis.frame_rate
    assert ana_loaded.threads == analysis.threads
    assert ana_loaded.n_processes == analysis.n_processes
    assert ana_loaded.calcium_peaks_data == analysis.calcium_peaks_data
    assert ana_loaded.spikes_data == analysis.spikes_data
    assert ana_loaded.experiment_type_data == analysis.experiment_type_data
