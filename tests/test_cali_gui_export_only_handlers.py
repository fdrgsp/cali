"""Tests for CaliGui export only handlers and edge cases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from cali.gui._cali_gui import CaliGui

if TYPE_CHECKING:
    from pytestqt.qtbot import QtBot

    pass


@pytest.fixture
def test_db_path() -> Path:
    """Return path to test database."""
    return Path("tests/test_data/data_and_db_for_tests/test_db.cali")


@pytest.fixture
def test_data_path() -> Path:
    """Return path to test data directory."""
    return Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")


def test_handle_export_only_no_database(qtbot: QtBot) -> None:
    """Test _handle_export_only with no database loaded."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Explicitly set database_path to None
    gui._database_path = None

    # Mock show_error_dialog
    with patch("cali.gui._cali_gui.show_error_dialog") as mock_error:
        gui._handle_export_only(run_id=1)
        mock_error.assert_called_once()
        assert "No database loaded" in mock_error.call_args[0][1]


def test_handle_export_only_no_exports_selected(
    qtbot: QtBot,
    test_db_path: Path,
    test_data_path: Path,
) -> None:
    """Test _handle_export_only with no export options selected."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Initialize GUI with database
    gui._initialize_from_database(str(test_db_path), str(test_data_path))
    qtbot.wait(100)

    # Disable all export options
    gui._extraction_wdg._export_group.setChecked(False)
    gui._analysis_wdg._export_group.setChecked(False)

    # Mock show_error_dialog
    with patch("cali.gui._cali_gui.show_error_dialog") as mock_error:
        gui._handle_export_only(run_id=1)
        mock_error.assert_called_once()
        assert "No export options selected" in mock_error.call_args[0][1]


def test_handle_export_only_export_failure(
    qtbot: QtBot,
    test_db_path: Path,
    test_data_path: Path,
) -> None:
    """Test _handle_export_only handles export failure gracefully."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Initialize GUI with database
    gui._initialize_from_database(str(test_db_path), str(test_data_path))
    qtbot.wait(100)

    # Enable some export options
    gui._extraction_wdg._export_group.setChecked(True)

    # Mock export function to raise exception
    with (
        patch("cali.gui._cali_gui.show_error_dialog") as mock_error,
        patch(
            "cali.util._database_to_csv.export_traces_to_csv",
            side_effect=RuntimeError("Test error"),
        ),
    ):
        gui._handle_export_only(run_id=1)
        # Should show error dialog
        mock_error.assert_called()
        assert "Export failed" in mock_error.call_args[0][1]


def test_get_run_option_export_only(qtbot: QtBot) -> None:
    """Test _get_run_option returns correct value for Export Only mode."""
    from cali.gui._run_widget import CaliRunSettings

    gui = CaliGui()
    qtbot.addWidget(gui)

    # Create settings with run_id (Export Only mode)
    settings = CaliRunSettings(
        positions=[],
        run_detection=False,
        run_extraction=False,
        run_analysis=False,
        detection_settings_id=None,
        extraction_settings_id=None,
        run_id=1,  # This indicates Export Only mode
    )

    result = gui._get_run_option(settings)
    assert result == 6  # Export Only option index


def test_get_run_option_detection_extraction_analysis(qtbot: QtBot) -> None:
    """Test _get_run_option returns correct value for full pipeline."""
    from cali.gui._run_widget import CaliRunSettings

    gui = CaliGui()
    qtbot.addWidget(gui)

    # Create settings for full pipeline
    settings = CaliRunSettings(
        positions=[],
        run_detection=True,
        run_extraction=True,
        run_analysis=True,
        detection_settings_id=None,
        extraction_settings_id=None,
        run_id=None,
    )

    result = gui._get_run_option(settings)
    assert result == 0  # Detection, Extraction and Analysis


def test_get_run_option_detection_extraction(qtbot: QtBot) -> None:
    """Test _get_run_option returns correct value for detection + extraction."""
    from cali.gui._run_widget import CaliRunSettings

    gui = CaliGui()
    qtbot.addWidget(gui)

    settings = CaliRunSettings(
        positions=[],
        run_detection=True,
        run_extraction=True,
        run_analysis=False,
        detection_settings_id=None,
        extraction_settings_id=None,
        run_id=None,
    )

    result = gui._get_run_option(settings)
    assert result == 1  # Detection and Extraction


def test_on_scene_well_changed_no_data(qtbot: QtBot) -> None:
    """Test _on_scene_well_changed when no data is loaded."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Should handle gracefully when no data
    gui._on_scene_well_changed()
    # No error should be raised
