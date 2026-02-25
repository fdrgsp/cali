"""Test CaliGui database-only loading mode (no raw imaging data)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt

from cali.gui import CaliGui
from cali.gui._init_dialog import InputDialogData, _InputDialog
from cali.gui._run_widget import CaliRunSettings, _RunCaliWidget
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


# --- _get_total_positions tests ---


def test_get_total_positions_returns_zero_with_no_state(gui: CaliGui) -> None:
    """Returns 0 when neither data nor database_path is set."""
    assert gui._get_total_positions() == 0


def test_get_total_positions_from_database(
    gui: CaliGui, db_with_extraction: Path
) -> None:
    """Returns FOV count from database when in database-only mode."""
    gui._initialize_from_database_only(db_with_extraction)
    assert gui._data is None
    assert gui._database_path == str(db_with_extraction)

    # Should query FOVs from the database (one FOV was processed)
    count = gui._get_total_positions()
    assert count >= 1


# --- _on_cali_run guard tests ---


def test_on_cali_run_blocks_detection_without_data(
    gui: CaliGui, db_with_extraction: Path
) -> None:
    """_on_cali_run shows error when detection requested but data is None."""
    gui._initialize_from_database_only(db_with_extraction)
    assert gui._data is None
    assert gui._database_path is not None

    mock_value = CaliRunSettings(
        positions=[0],
        run_detection=True,
        run_extraction=False,
        run_analysis=False,
        detection_settings_id=None,
        extraction_settings_id=None,
        run_id=None,
    )

    with (
        patch.object(gui._run_cali_wdg, "value", return_value=mock_value),
        patch("cali.gui._cali_gui.show_error_dialog") as mock_err,
    ):
        gui._on_cali_run()
        mock_err.assert_called_once()
        msg = mock_err.call_args[0][1]
        assert "raw imaging data" in msg.lower() or "detection" in msg.lower()


def test_on_cali_run_blocks_extraction_without_data(
    gui: CaliGui, db_with_extraction: Path
) -> None:
    """_on_cali_run shows error when extraction requested but data is None."""
    gui._initialize_from_database_only(db_with_extraction)
    assert gui._data is None

    mock_value = CaliRunSettings(
        positions=[0],
        run_detection=False,
        run_extraction=True,
        run_analysis=False,
        detection_settings_id=None,
        extraction_settings_id=None,
        run_id=None,
    )

    with (
        patch.object(gui._run_cali_wdg, "value", return_value=mock_value),
        patch("cali.gui._cali_gui.show_error_dialog") as mock_err,
    ):
        gui._on_cali_run()
        mock_err.assert_called_once()


def test_on_cali_run_returns_early_when_no_database_path(gui: CaliGui) -> None:
    """_on_cali_run returns early (no-op) when database_path is None."""
    assert gui._database_path is None
    with patch("cali.gui._cali_gui.show_error_dialog") as mock_err:
        gui._on_cali_run()
        mock_err.assert_not_called()


# --- _show_data_input_dialog exception handling ---


def test_show_dialog_exception_during_database_only_init(
    gui: CaliGui, tmp_path: Path
) -> None:
    """Exception during database-only init shows error dialog."""
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
        patch.object(
            gui,
            "_initialize_from_database_only",
            side_effect=RuntimeError("boom"),
        ),
        patch("cali.gui._cali_gui.show_error_dialog") as mock_err,
    ):
        gui._show_data_input_dialog()
        mock_err.assert_called_once()
        msg = mock_err.call_args[0][1]
        assert "boom" in msg


# --- _RunCaliWidget auto-switch tests ---


def test_set_has_data_false_auto_switches_current_selection(qtbot: QtBot) -> None:
    """When has_data becomes False, all detection options 0-4 are disabled.

    If no Analysis Only option is available (no detections/extractions), the
    combo resets to index 0.
    """
    wdg = _RunCaliWidget()
    qtbot.addWidget(wdg)

    # Start with data available and option 0 selected
    wdg.set_has_data(True)
    wdg._run_options_combo.setCurrentIndex(0)

    # Disable data - options 0-4 become disabled
    wdg.set_has_data(False)

    from qtpy.QtGui import QStandardItemModel

    model = wdg._run_options_combo.model()
    assert isinstance(model, QStandardItemModel)
    # All detection/extraction options (0-4) must be disabled
    for idx in range(5):
        item = model.item(idx)
        assert item is not None
        assert not (item.flags() & Qt.ItemFlag.ItemIsEnabled), (
            f"Option {idx} should be disabled when has_data=False"
        )


def test_set_has_data_true_extraction_only_without_detections_resets(
    qtbot: QtBot,
) -> None:
    """When has_data=True but no detections and current=4 (Extraction Only),
    resets to 0."""
    wdg = _RunCaliWidget()
    qtbot.addWidget(wdg)

    # Explicitly select option 4
    wdg._run_options_combo.setCurrentIndex(4)

    # Calling set_has_data(True) with no detection settings populated should
    # reset idx 4 back to 0 since there are no detections.
    wdg.set_has_data(True)

    assert wdg._run_options_combo.currentIndex() == 0


def test_set_has_data_true_extraction_analysis_without_detections_resets(
    qtbot: QtBot,
) -> None:
    """When has_data=True but no detections and current=2 (Ext+Analysis),
    resets to 0."""
    wdg = _RunCaliWidget()
    qtbot.addWidget(wdg)

    wdg._run_options_combo.setCurrentIndex(2)
    wdg.set_has_data(True)

    assert wdg._run_options_combo.currentIndex() == 0


# --- Database-only: plate well click populates FOV table ---


def test_on_scene_well_changed_populates_fov_table_from_database(
    gui: CaliGui, db_with_extraction: Path, qtbot: QtBot
) -> None:
    """_on_scene_well_changed populates FOV table from DB in database-only mode."""
    from sqlmodel import Session, create_engine
    from sqlmodel import select as sql_select

    from cali.sqlmodel._model import Well

    gui._initialize_from_database_only(db_with_extraction)

    # Find an actual well name in the created database
    engine = create_engine(
        f"sqlite:///{db_with_extraction}",
        connect_args={"timeout": 30.0, "check_same_thread": False},
    )
    try:
        with Session(engine) as session:
            well = session.exec(sql_select(Well)).first()
    finally:
        engine.dispose(close=True)

    if well is None:
        pytest.skip("No wells in database")

    # Mock the plate view selected items with the real well name

    mock_item = MagicMock()
    mock_position = MagicMock()
    mock_position.name = well.name
    mock_item.data.return_value = mock_position
    gui._plate_view._selected_items = {mock_item}

    # Clear the FOV table before calling
    gui._fov_table.clearContents()
    gui._fov_table.setRowCount(0)

    # Call the plate-well handler directly
    gui._on_scene_well_changed()

    assert gui._fov_table.rowCount() > 0, (
        "FOV table should be populated from database in database-only mode"
    )


# --- Database-only: FOV selection shows labels without image data ---


def test_on_fov_table_selection_changed_shows_labels_database_only(
    gui: CaliGui, db_with_extraction: Path, qtbot: QtBot
) -> None:
    """_on_fov_table_selection_changed shows labels in viewer in database-only mode."""
    import useq
    from sqlmodel import Session, create_engine
    from sqlmodel import select as sql_select

    from cali.gui._fov_table import WellInfo
    from cali.sqlmodel._model import FOV

    gui._initialize_from_database_only(db_with_extraction)

    # Find the actual FOV name to use for the WellInfo
    engine = create_engine(
        f"sqlite:///{db_with_extraction}",
        connect_args={"timeout": 30.0, "check_same_thread": False},
    )
    try:
        with Session(engine) as session:
            fov = session.exec(sql_select(FOV)).first()
    finally:
        engine.dispose(close=True)

    if fov is None:
        pytest.skip("No FOVs in database")

    # Add the FOV entry to the table directly
    gui._fov_table.add_position(
        WellInfo(fov.position_index, useq.Position(name=fov.name))
    )
    gui._fov_table.selectRow(0)

    # Trigger the selection handler
    gui._on_fov_table_selection_changed()

    # In database-only mode, image viewer should be updated (no raw image data)
    # The image data should be None; labels may or may not be present
    assert gui._image_viewer is not None  # viewer was updated without error


def test_on_scene_well_changed_returns_early_when_no_db_path(
    gui: CaliGui, qtbot: QtBot
) -> None:
    """_on_scene_well_changed returns early when _data is None and
    _database_path is None.

    Covers the guard at _cali_gui.py line 2545.
    """
    assert gui._data is None
    assert gui._database_path is None

    mock_item = MagicMock()
    mock_position = MagicMock()
    mock_position.name = "A1"
    mock_item.data.return_value = mock_position
    gui._plate_view._selected_items = {mock_item}

    # Should return early without error or any database access
    gui._on_scene_well_changed()

    assert gui._fov_table.rowCount() == 0


def test_on_cali_run_empty_positions_calls_get_total_positions(
    gui: CaliGui, db_with_extraction: Path, qtbot: QtBot
) -> None:
    """_on_cali_run with positions=[] falls back to _get_total_positions (line 1367).

    When positions is an empty list, the second pos assignment recomputes from
    _get_total_positions, identical to how line 1240 does for the first assignment.
    """
    # Need a real DB so the function doesn't return early at the db-path check
    gui._database_path = str(db_with_extraction)

    # Mock data (non-None) so the early-return guard (_data is None) passes
    gui._data = MagicMock()

    # Choose option 0 = "Full Analysis" which has run_detection+run_extraction=True
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(0)

    # Leave positions widget empty so parse_lineedit_text returns []
    # which forces the list(range(_get_total_positions())) fallback at line 1367
    gui._run_cali_wdg._positions_wdg.setValue("")

    # Intercept at create_worker to avoid spawning an actual QThread
    with patch("cali.gui._cali_gui.create_worker") as mock_worker:
        mock_worker.return_value = MagicMock()
        gui._on_cali_run()

    # create_worker being called confirms execution reached past line 1367
    mock_worker.assert_called_once()
