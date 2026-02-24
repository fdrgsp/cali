"""Test CaliGui database-only loading mode (no raw imaging data)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt

from cali.gui import CaliGui
from cali.gui._init_dialog import InputDialogData, _InputDialog
from cali.gui._run_widget import _RunCaliWidget
from cali.sqlmodel import Experiment


@pytest.fixture
def gui(qtbot: QtBot) -> CaliGui:
    """Create a CaliGui instance."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    return gui


@pytest.fixture
def db_with_extraction(tmp_path: Path, mock_detection_runner: MagicMock) -> Path:
    """Create a .cali database with extraction results."""
    from cali.runner import CaliRunner
    from cali.sqlmodel import AnalysisSettings, DetectionSettings, ExtractionSettings

    data_path = Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")
    db_name = "test_db_only.cali"

    exp = Experiment.create_from_data(
        name="Test",
        data_path=data_path,
        description="Test experiment",
    )
    runner = CaliRunner()
    runner.run(
        experiment=exp,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(),
        global_position_indices=[0],
        database_name=db_name,
        output_path=tmp_path,
    )
    return tmp_path / db_name


@pytest.fixture
def db_without_extraction(tmp_path: Path) -> Path:
    """Create a .cali database without extraction results."""
    from cali.sqlmodel import save_experiment_to_database

    exp = Experiment(name="Test", description="Test experiment")
    save_experiment_to_database(exp, tmp_path, database_name="no_extraction.cali")
    return tmp_path / "no_extraction.cali"
    return tmp_path / "no_extraction.cali"


# --- _InputDialog tests ---


def test_input_dialog_database_only(qtbot: QtBot) -> None:
    """Dialog returns data_path=None when only database path is set."""
    dialog = _InputDialog()
    qtbot.addWidget(dialog)

    # Switch to "From Database" tab (index 0)
    dialog._tab_widget.setCurrentIndex(0)

    # Set only database path, leave data path empty
    dialog._browse_database._path.setText("/some/path/test.cali")
    dialog._browse_data_db._path.setText("")

    value = dialog.value()
    assert value.database_path is not None
    assert value.data_path is None


def test_input_dialog_database_with_data(qtbot: QtBot) -> None:
    """Dialog returns both paths when both are set."""
    dialog = _InputDialog()
    qtbot.addWidget(dialog)

    dialog._tab_widget.setCurrentIndex(0)
    dialog._browse_database._path.setText("/some/path/test.cali")
    dialog._browse_data_db._path.setText("/some/data/path")

    value = dialog.value()
    assert value.database_path is not None
    assert value.data_path is not None


def test_input_dialog_data_path_label_has_asterisk(qtbot: QtBot) -> None:
    """Data path label in database tab should show asterisk for optional."""
    dialog = _InputDialog()
    qtbot.addWidget(dialog)

    assert "Data Path*" in dialog._browse_data_db._label.text()


# --- Initialization tests ---


def test_initialize_from_database_only_success(
    gui: CaliGui, db_with_extraction: Path
) -> None:
    """Database-only initialization works with extraction results."""
    gui._initialize_from_database_only(db_with_extraction)

    assert gui._database_path == str(db_with_extraction)
    assert gui._data is None
    assert gui._data_path is None
    assert gui._output_path == str(db_with_extraction.parent)
    assert not gui._loading_bar.isVisible()


def test_initialize_from_database_only_no_extraction(
    gui: CaliGui, db_without_extraction: Path
) -> None:
    """Database-only initialization fails without both detection and extraction."""
    with patch("cali.gui._cali_gui.show_error_dialog") as mock_dialog:
        gui._initialize_from_database_only(db_without_extraction)

        mock_dialog.assert_called_once()
        args = mock_dialog.call_args[0]
        assert "both detection and extraction" in args[1].lower()


def test_initialize_from_database_only_not_exists(gui: CaliGui) -> None:
    """Database-only initialization fails when file doesn't exist."""
    with patch("cali.gui._cali_gui.show_error_dialog") as mock_dialog:
        gui._initialize_from_database_only("nonexistent.cali")

        mock_dialog.assert_called_once()
        args = mock_dialog.call_args[0]
        assert "not found" in args[1].lower()


# --- Run options tests ---


def test_run_options_disabled_without_data(qtbot: QtBot) -> None:
    """Options requiring raw data are disabled when has_data=False."""
    wdg = _RunCaliWidget()
    qtbot.addWidget(wdg)

    # Simulate having detections and extractions in DB using proper API
    wdg.populate_detection_settings([(1, "cellpose")])
    wdg.populate_extraction_settings([1])

    wdg.set_has_data(False)

    from qtpy.QtGui import QStandardItemModel

    model = wdg._run_options_combo.model()
    assert isinstance(model, QStandardItemModel)

    # Options 0-4 (detection/extraction) should be disabled
    for idx in range(5):
        item = model.item(idx)
        assert item is not None
        assert not (item.flags() & Qt.ItemFlag.ItemIsEnabled), (
            f"Option {idx} should be disabled without data"
        )

    # Option 5 (Analysis Only) should be enabled (has detections + extractions)
    analysis_item = model.item(5)
    assert analysis_item is not None
    assert analysis_item.flags() & Qt.ItemFlag.ItemIsEnabled


def test_run_options_re_enabled_with_data(qtbot: QtBot) -> None:
    """Options are re-enabled when has_data switches back to True."""
    wdg = _RunCaliWidget()
    qtbot.addWidget(wdg)

    wdg.set_has_data(False)
    wdg.set_has_data(True)

    from qtpy.QtGui import QStandardItemModel

    model = wdg._run_options_combo.model()
    assert isinstance(model, QStandardItemModel)

    # Options 0, 1, 3 (full pipeline, det+ext, det only) should be re-enabled
    for idx in (0, 1, 3):
        item = model.item(idx)
        assert item is not None
        assert item.flags() & Qt.ItemFlag.ItemIsEnabled, (
            f"Option {idx} should be enabled with data"
        )


# --- _show_data_input_dialog routing tests ---


def test_show_dialog_routes_to_database_only(gui: CaliGui, tmp_path: Path) -> None:
    """Dialog routes to _initialize_from_database_only when data_path is None."""
    db_path = tmp_path / "test.cali"

    mock_value = InputDialogData(
        data_path=None,
        output_path=None,
        database_path=str(db_path),
        database_name=None,
    )

    with (
        patch("cali.gui._cali_gui._InputDialog.exec", return_value=True),
        patch("cali.gui._cali_gui._InputDialog.value", return_value=mock_value),
        patch.object(gui, "_initialize_from_database_only") as mock_init,
    ):
        gui._show_data_input_dialog()
        mock_init.assert_called_once_with(str(db_path))


def test_database_only_shows_labels_in_viewer(
    gui: CaliGui, db_with_extraction: Path, qtbot: QtBot
) -> None:
    """Test that labels are shown in image viewer even without data (database-only)."""
    gui._initialize_from_database_only(db_with_extraction)

    # Ensure GUI is fully initialized
    qtbot.waitUntil(lambda: not gui._loading_bar.isVisible(), timeout=5000)

    # Verify data is None (database-only mode)
    assert gui._data is None, "Should be in database-only mode (data=None)"

    # Select a FOV in the table
    if gui._fov_table.rowCount() > 0:
        gui._fov_table.selectRow(0)
        qtbot.wait(100)  # Give time for selection to propagate

        # Verify that labels exist in the image viewer
        # Even though data is None, labels should be loaded
        viewer = gui._image_viewer._viewer
        assert viewer.labels_image is not None, (
            "Labels should be shown in viewer even without image data"
        )

        # Verify the labels button is enabled
        assert gui._image_viewer._labels.isEnabled(), (
            "Labels button should be enabled when labels exist, "
            "even in database-only mode"
        )
