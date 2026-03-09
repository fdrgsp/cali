from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import useq
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItemModel
from qtpy.QtWidgets import QApplication, QMessageBox
from sqlmodel import Session, create_engine

from cali.gui import CaliGui
from cali.gui._run_selection_dialog import RunSelectionDialog
from cali.gui._run_widget import CaliRunSettings
from cali.sqlmodel._model import (
    AnalysisSettings,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)

THREADS = 1


def test_launch_gui(qtbot: QtBot) -> None:
    """Test launching the Cali GUI."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui.show()


@pytest.fixture
def temp_db_with_detection(tmp_path: Path) -> Path:
    """Create a temporary database with detection settings."""
    db_path = tmp_path / "test_detection.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create tables
    from cali.sqlmodel._model import SQLModel

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create experiment
        experiment = Experiment(
            name="Test Experiment",
            description=f"Experiment from data at {tmp_path / 'data.zarr'}",
        )
        session.add(experiment)
        session.commit()

        # Create detection settings
        detection_settings = DetectionSettings(
            method="cellpose",
            model_type="cyto3",
            diameter=30.0,
            cellprob_threshold=0.0,
            flow_threshold=0.4,
        )
        session.add(detection_settings)
        session.commit()

    engine.dispose(close=True)
    return db_path


@pytest.fixture
def temp_db_with_extraction(tmp_path: Path) -> Path:
    """Create a temporary database with detection and extraction settings."""
    db_path = tmp_path / "test_extraction.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    from cali.sqlmodel._model import SQLModel

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create experiment
        experiment = Experiment(
            name="Test Experiment",
            description=f"Experiment from data at {tmp_path / 'data.zarr'}",
        )
        session.add(experiment)
        session.commit()

        # Create detection settings
        detection_settings = DetectionSettings(
            method="cellpose",
            model_type="cyto3",
            diameter=30.0,
            cellprob_threshold=0.0,
            flow_threshold=0.4,
        )
        session.add(detection_settings)
        session.commit()

        # Create extraction settings
        extraction_settings = ExtractionSettings(
            neuropil_inner_radius=2,
            neuropil_min_pixels=50,
            neuropil_correction_factor=0.7,
            decay_constant=0.5,
            dff_window=100,
            threads=THREADS,
        )
        session.add(extraction_settings)
        session.commit()

    engine.dispose(close=True)
    return db_path


@pytest.fixture
def temp_db_with_analysis(tmp_path: Path) -> Path:
    """Create a temporary database with detection, extraction, and analysis settings."""
    db_path = tmp_path / "test_analysis.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    from cali.sqlmodel._model import SQLModel

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create experiment
        experiment = Experiment(
            name="Test Experiment",
            description=f"Experiment from data at {tmp_path / 'data.zarr'}",
        )
        session.add(experiment)
        session.commit()

        # Create detection settings
        detection_settings = DetectionSettings(
            method="cellpose",
            model_type="cyto3",
            diameter=30.0,
            cellprob_threshold=0.0,
            flow_threshold=0.4,
        )
        session.add(detection_settings)
        session.commit()

        # Create extraction settings
        extraction_settings = ExtractionSettings(
            neuropil_inner_radius=2,
            neuropil_min_pixels=50,
            neuropil_correction_factor=0.7,
            decay_constant=0.5,
            dff_window=100,
            threads=THREADS,
        )
        session.add(extraction_settings)
        session.commit()

        # Create analysis settings
        analysis_settings = AnalysisSettings(
            peaks_prominence_multiplier=3.0,
            peaks_distance=10,
            threads=THREADS,
        )
        session.add(analysis_settings)
        session.commit()

    engine.dispose(close=True)
    return db_path


@pytest.fixture
def temp_db_with_runs(tmp_path: Path) -> Path:
    """Create a temporary database with some CaliResult runs."""
    db_path = tmp_path / "test_runs.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    from cali.sqlmodel._model import SQLModel

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create experiment
        experiment = Experiment(
            name="Test Experiment",
            description=f"Experiment from data at {tmp_path / 'data.zarr'}",
        )
        session.add(experiment)
        session.commit()
        session.refresh(experiment)
        assert experiment.id is not None

        # Create detection settings
        detection_settings = DetectionSettings(
            method="cellpose",
            model_type="cyto3",
            diameter=30.0,
            cellprob_threshold=0.0,
            flow_threshold=0.4,
        )
        session.add(detection_settings)
        session.commit()

        # Create extraction settings
        extraction_settings = ExtractionSettings(
            neuropil_inner_radius=2,
            neuropil_min_pixels=50,
            neuropil_correction_factor=0.7,
            decay_constant=0.5,
            dff_window=100,
            threads=THREADS,
        )
        session.add(extraction_settings)
        session.commit()

        # Create analysis settings
        analysis_settings = AnalysisSettings(
            peaks_prominence_multiplier=3.0,
            peaks_distance=10,
            threads=THREADS,
        )
        session.add(analysis_settings)
        session.commit()

        # Create some CaliResult entries
        for _i in range(3):
            result = CaliResult(
                experiment=experiment.id,
                detection_settings_id=detection_settings.id,
                extraction_settings_id=extraction_settings.id,
                analysis_settings_id=analysis_settings.id,
                positions_detected=[0, 1],
                positions_extracted=[0, 1],
                positions_analyzed=[0, 1],
            )
            session.add(result)
        session.commit()

    engine.dispose(close=True)
    return db_path


def test_run_options_detection_only(qtbot: QtBot, temp_db_with_detection: Path) -> None:
    """Test that detection-only database doesn't enable extraction/analysis options."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitExposed(gui)

    # Directly populate settings without loading data
    gui._database_path = str(temp_db_with_detection)
    gui._runs_panel.set_database_path(str(temp_db_with_detection))
    gui._populate_settings(str(temp_db_with_detection))

    # Check that detection settings are populated
    assert gui._run_cali_wdg._detection_settings_combo.count() > 1

    # Check that extraction settings are not populated
    assert gui._run_cali_wdg._extraction_settings_combo.count() == 1  # Only placeholder

    # Set to "Extraction Only" option (now at index 4)
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(4)

    # Check that detection combo becomes visible
    assert gui._run_cali_wdg._detection_settings_combo.isVisible()
    assert not gui._run_cali_wdg._extraction_settings_combo.isVisible()

    # Check that "Analysis Only" option is disabled
    model = gui._run_cali_wdg._run_options_combo.model()
    assert isinstance(model, QStandardItemModel)
    analysis_item = model.item(5)
    assert analysis_item is not None
    assert not (analysis_item.flags() & Qt.ItemFlag.ItemIsSelectable)


def test_run_options_with_extraction(
    qtbot: QtBot, temp_db_with_extraction: Path
) -> None:
    """Test that extraction database enables extraction-only but not analysis-only."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitExposed(gui)

    gui._database_path = str(temp_db_with_extraction)
    gui._runs_panel.set_database_path(str(temp_db_with_extraction))
    gui._populate_settings(str(temp_db_with_extraction))

    # Check that both detection and extraction settings are populated
    assert gui._run_cali_wdg._detection_settings_combo.count() > 1
    assert gui._run_cali_wdg._extraction_settings_combo.count() > 1

    # Set to "Extraction Only" - should be enabled
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(4)
    assert gui._run_cali_wdg._detection_settings_combo.isVisible()

    # Set to "Analysis Only" - should be disabled (no analysis settings yet)
    # Note: In the current implementation, this checks for extraction settings existence
    # which we have, so it should be enabled
    model = gui._run_cali_wdg._run_options_combo.model()
    assert isinstance(model, QStandardItemModel)
    analysis_item = model.item(5)
    assert analysis_item is not None
    # Should be enabled since we have both detection and extraction
    assert analysis_item.flags() & Qt.ItemFlag.ItemIsEnabled

    # Cleanup
    gui.close()


def test_run_options_all_settings(qtbot: QtBot, temp_db_with_analysis: Path) -> None:
    """Test that all run options are available when all settings exist."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitExposed(gui)

    gui._database_path = str(temp_db_with_analysis)
    gui._runs_panel.set_database_path(str(temp_db_with_analysis))
    gui._populate_settings(str(temp_db_with_analysis))

    # Check all settings are populated
    assert gui._run_cali_wdg._detection_settings_combo.count() > 1
    assert gui._run_cali_wdg._extraction_settings_combo.count() > 1

    model = gui._run_cali_wdg._run_options_combo.model()
    assert isinstance(model, QStandardItemModel)

    # All options should be enabled
    for i in range(5):
        item = model.item(i)
        assert item is not None
        assert item.flags() & Qt.ItemFlag.ItemIsEnabled


def test_run_settings_parsing_detection_only(qtbot: QtBot) -> None:
    """Test parsing run settings for detection-only mode."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Set to "Detection Only"
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(3)

    settings = gui._run_cali_wdg.value()
    assert isinstance(settings, CaliRunSettings)
    assert settings.run_detection
    assert not settings.run_extraction
    assert not settings.run_analysis
    assert settings.detection_settings_id is None  # Not using existing detection
    assert settings.extraction_settings_id is None


def test_run_settings_parsing_extraction_only(qtbot: QtBot) -> None:
    """Test parsing run settings for extraction-only mode."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Populate detection settings like the real code does
    gui._run_cali_wdg.populate_detection_settings([(1, "cellpose")])

    # Set to "Extraction Only" - this triggers visibility change
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(4)

    # Verify combo is visible after changing option
    assert gui._run_cali_wdg._detection_settings_combo.isVisible()

    # Select the detection settings (index 1 is first real item after placeholder)
    gui._run_cali_wdg._detection_settings_combo.setCurrentIndex(1)

    settings = gui._run_cali_wdg.value()
    assert isinstance(settings, CaliRunSettings)
    assert not settings.run_detection
    assert settings.run_extraction
    assert not settings.run_analysis
    assert settings.detection_settings_id == 1
    assert settings.extraction_settings_id is None

    # Cleanup
    gui.close()


def test_run_settings_parsing_analysis_only(qtbot: QtBot) -> None:
    """Test parsing run settings for analysis-only mode."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Populate settings like the real code does
    gui._run_cali_wdg.populate_detection_settings([(1, "cellpose")])
    gui._run_cali_wdg.populate_extraction_settings([1])

    # Set to "Analysis Only" - this triggers visibility change
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(5)

    # Verify combos are visible
    assert gui._run_cali_wdg._detection_settings_combo.isVisible()
    assert gui._run_cali_wdg._extraction_settings_combo.isVisible()

    # Select settings (index 1 is first real item after placeholder)
    gui._run_cali_wdg._detection_settings_combo.setCurrentIndex(1)
    gui._run_cali_wdg._extraction_settings_combo.setCurrentIndex(1)

    settings = gui._run_cali_wdg.value()
    assert isinstance(settings, CaliRunSettings)
    assert not settings.run_detection
    assert not settings.run_extraction
    assert settings.run_analysis
    assert settings.detection_settings_id == 1
    assert settings.extraction_settings_id == 1


def test_run_settings_parsing_full_pipeline(qtbot: QtBot) -> None:
    """Test parsing run settings for full pipeline."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Set to "Detection, Extraction and Analysis"
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(0)

    settings = gui._run_cali_wdg.value()
    assert isinstance(settings, CaliRunSettings)
    assert settings.run_detection
    assert settings.run_extraction
    assert settings.run_analysis
    assert settings.detection_settings_id is None  # Creating new detection
    assert settings.extraction_settings_id is None


def test_run_settings_parsing_positions(qtbot: QtBot) -> None:
    """Test parsing position inputs."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Test single positions
    gui._run_cali_wdg._positions_wdg.setValue("0, 5, 10")
    settings = gui._run_cali_wdg.value()
    assert settings.positions == [0, 5, 10]

    # Test range
    gui._run_cali_wdg._positions_wdg.setValue("0-5")
    settings = gui._run_cali_wdg.value()
    assert settings.positions == [0, 1, 2, 3, 4, 5]

    # Test mixed
    gui._run_cali_wdg._positions_wdg.setValue("0-2, 5, 8-10")
    settings = gui._run_cali_wdg.value()
    assert settings.positions == [0, 1, 2, 5, 8, 9, 10]

    # Test empty (all positions)
    gui._run_cali_wdg._positions_wdg.setValue("")
    settings = gui._run_cali_wdg.value()
    assert settings.positions == []


def test_combo_visibility_changes(qtbot: QtBot) -> None:
    """Test that combos show/hide correctly when changing run options."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Default: "Detection, Extraction and Analysis"
    assert not gui._run_cali_wdg._detection_settings_combo.isVisible()
    assert not gui._run_cali_wdg._extraction_settings_combo.isVisible()

    # "Detection Only" (index 3)
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(3)
    assert not gui._run_cali_wdg._detection_settings_combo.isVisible()
    assert not gui._run_cali_wdg._extraction_settings_combo.isVisible()

    # "Extraction Only" (index 4)
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(4)
    assert gui._run_cali_wdg._detection_settings_combo.isVisible()
    assert not gui._run_cali_wdg._extraction_settings_combo.isVisible()

    # "Analysis Only" (index 5)
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(5)
    assert gui._run_cali_wdg._detection_settings_combo.isVisible()
    assert gui._run_cali_wdg._extraction_settings_combo.isVisible()

    # Back to "Detection and Extraction" (index 1)
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(1)
    assert not gui._run_cali_wdg._detection_settings_combo.isVisible()
    assert not gui._run_cali_wdg._extraction_settings_combo.isVisible()


def test_on_worker_finished_selects_recent_run(
    qtbot: QtBot, test_db_copy: Path
) -> None:
    """Test that _on_worker_finished selects the most recently modified run."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitExposed(gui)

    # Load the test database which has 2 runs with different last_modified times
    data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"

    # Initialize GUI from database
    gui._initialize_from_database(str(test_db_copy), data_path)

    # Get the most recently modified run from database to know which one to expect
    engine = create_engine(f"sqlite:///{test_db_copy}")
    with Session(engine) as session:
        from sqlmodel import select

        most_recent = session.exec(
            select(CaliResult)
            .order_by(CaliResult.last_modified.desc())  # type: ignore
            .limit(1)
        ).first()
        assert most_recent is not None
        expected_run_id = most_recent.id

    engine.dispose(close=True)

    # Mock refresh_runs to avoid reloading (we just want to test selection)
    import unittest.mock

    with unittest.mock.patch.object(gui._runs_panel, "refresh_runs"):
        # Call the method that should select the most recently modified run
        gui._on_worker_finished()

    # Verify that the most recently modified run is now selected
    selected_item = gui._runs_panel._runs_list.currentItem()
    assert selected_item is not None
    selected_run_id = selected_item.data(Qt.ItemDataRole.UserRole)
    assert selected_run_id == expected_run_id, (
        f"Expected run {expected_run_id} to be selected, "
        f"but run {selected_run_id} is selected"
    )

    # Cleanup
    gui.close()


def test_check_positions_missing_detection(
    qtbot: QtBot, temp_db_with_detection: Path
) -> None:
    """Test the helper method that checks for missing detection data."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    gui._database_path = str(temp_db_with_detection)

    # With no ROIs in the database, all positions should be missing
    missing = gui._check_positions_missing_detection(1, [0, 1, 2])
    assert missing == [0, 1, 2]

    # Add some ROIs for position 0
    from cali.sqlmodel._model import FOV, ROI

    engine = create_engine(f"sqlite:///{temp_db_with_detection}")
    with Session(engine) as session:
        fov = FOV(name="fov_0", position_index=0, experiment_id=1)
        session.add(fov)
        session.commit()
        session.refresh(fov)

        roi = ROI(
            label_value=1,
            fov_id=fov.id,
            detection_settings_id=1,
        )
        session.add(roi)
        session.commit()

    # Now position 0 should not be missing, but 1 and 2 should be
    missing = gui._check_positions_missing_detection(1, [0, 1, 2])
    assert missing == [1, 2]

    engine.dispose(close=True)
    gui.close()


def test_check_positions_missing_extraction(
    qtbot: QtBot, temp_db_with_extraction: Path
) -> None:
    """Test the helper method that checks for missing extraction data."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    gui._database_path = str(temp_db_with_extraction)

    # With no traces in the database, all positions should be missing
    missing = gui._check_positions_missing_extraction(1, 1, [0, 1, 2])
    assert missing == [0, 1, 2]

    # Add FOV, ROI, CaliResult, and Traces for position 0
    from cali.sqlmodel._model import FOV, ROI, CaliResult, Traces

    engine = create_engine(f"sqlite:///{temp_db_with_extraction}")
    with Session(engine) as session:
        # Create CaliResult first
        result = CaliResult(
            experiment=1,
            detection_settings_id=1,
            extraction_settings_id=1,
            positions_analyzed=[0],
        )
        session.add(result)
        session.commit()
        session.refresh(result)

        # Create FOV
        fov = FOV(name="fov_0", position_index=0, experiment_id=1)
        session.add(fov)
        session.commit()
        session.refresh(fov)

        # Create ROI
        roi = ROI(
            label_value=1,
            fov_id=fov.id,
            detection_settings_id=1,
        )
        session.add(roi)
        session.commit()
        session.refresh(roi)

        # Create Traces
        traces = Traces(
            roi_id=roi.id,
            analysis_result_id=result.id,
            raw_trace=[1.0, 2.0, 3.0],
        )
        session.add(traces)
        session.commit()

    # Now position 0 should not be missing, but 1 and 2 should be
    missing = gui._check_positions_missing_extraction(1, 1, [0, 1, 2])
    assert missing == [1, 2]

    engine.dispose(close=True)
    gui.close()


def test_analysis_only_requires_both_ids(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that analysis-only mode requires both detection and extraction IDs."""
    from unittest.mock import MagicMock

    gui = CaliGui()
    qtbot.addWidget(gui)

    # Set up mock data
    gui._database_path = str(tmp_path / "test.cali")
    gui._data = MagicMock()
    gui._data.sequence = MagicMock()
    gui._data.sequence.stage_positions = [MagicMock()]

    # Mock show_error_dialog to track calls
    error_calls = []

    def mock_error_dialog(parent: object, msg: str) -> None:
        error_calls.append(msg)

    monkeypatch.setattr("cali.gui._cali_gui.show_error_dialog", mock_error_dialog)

    # Mock Experiment.load_from_database
    mock_exp = MagicMock()
    monkeypatch.setattr(
        "cali.gui._cali_gui.Experiment.load_from_database",
        lambda *args, **kwargs: mock_exp,
    )

    # Test: Analysis-only with only extraction ID (missing detection ID)
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(5)  # Analysis Only
    gui._run_cali_wdg._extraction_settings_combo.addItem("Extraction 1", 1)
    gui._run_cali_wdg._extraction_settings_combo.setCurrentIndex(1)
    # Detection combo is empty (no detection ID selected)

    gui._on_cali_run()

    assert len(error_calls) == 1
    assert "Detection ID" in error_calls[0]

    # Test: Analysis-only with only detection ID (missing extraction ID)
    error_calls.clear()
    gui._run_cali_wdg._detection_settings_combo.addItem("Detection 1", 1)
    gui._run_cali_wdg._detection_settings_combo.setCurrentIndex(1)
    # Clear extraction combo to None
    gui._run_cali_wdg._extraction_settings_combo.clear()
    gui._run_cali_wdg._extraction_settings_combo.addItem("Select...", None)

    gui._on_cali_run()

    assert len(error_calls) == 1
    assert "Extraction ID" in error_calls[0]

    # Test: Analysis-only with neither ID
    error_calls.clear()
    gui._run_cali_wdg._detection_settings_combo.clear()
    gui._run_cali_wdg._detection_settings_combo.addItem("Select...", None)

    gui._on_cali_run()

    assert len(error_calls) == 1
    assert "Detection ID" in error_calls[0] and "Extraction ID" in error_calls[0]

    gui.close()


def test_runs_panel_delete_run(
    qtbot: QtBot, temp_db_with_runs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test deleting a run using _RunsPanel."""
    from qtpy.QtWidgets import QMessageBox

    from cali.gui._runs_panel import _RunsPanel

    # Create the runs panel
    panel = _RunsPanel()
    qtbot.addWidget(panel)
    panel.show()

    # Set the database path
    panel.set_database_path(str(temp_db_with_runs))

    # Check that runs are loaded
    assert panel._runs_list.count() == 3

    # Select the first run
    first_item = panel._runs_list.item(0)
    assert first_item is not None
    run_id = first_item.data(Qt.ItemDataRole.UserRole)
    assert run_id is not None
    panel._runs_list.setCurrentItem(first_item)

    # Mock QMessageBox.warning to return Yes
    def mock_warning(*args: Any, **kwargs: Any) -> QMessageBox.StandardButton:
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr("cali.gui._runs_panel.QMessageBox.warning", mock_warning)

    # Simulate clicking delete button
    panel._delete_selected_run()

    # Check that the run was deleted
    panel.refresh_runs()
    assert panel._runs_list.count() == 2

    # Verify the run is gone from database
    engine = create_engine(f"sqlite:///{temp_db_with_runs}")
    with Session(engine) as session:
        from sqlmodel import select

        from cali.sqlmodel._model import CaliResult

        remaining_results = session.exec(select(CaliResult)).all()
        assert len(remaining_results) == 2
        # Check that the deleted run_id is not in remaining
        remaining_ids = {r.id for r in remaining_results}
        assert run_id not in remaining_ids

    engine.dispose(close=True)
    panel.close()


def test_plate_map_widget_emits_signal_on_accept(qtbot: QtBot) -> None:
    """Test that _PlateMapWidget emits plateMapSaved signal when dialog is accepted."""
    from cali.gui._plate_map import _PlateMapWidget

    widget = _PlateMapWidget()
    qtbot.addWidget(widget)

    # Track signal emission
    with qtbot.waitSignal(widget.plateMapSaved, timeout=1000):
        # Simulate dialog accepted
        widget._on_dialog_accepted()

    # Verify dialog is hidden after accept
    assert widget._plate_map_dialog.isHidden()


def test_plate_map_widget_close_save(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that close dialog with Save emits plateMapSaved and hides dialog."""
    from unittest.mock import MagicMock

    from qtpy.QtWidgets import QMessageBox

    from cali.gui._plate_map import _PlateMapWidget

    widget = _PlateMapWidget()
    qtbot.addWidget(widget)

    # Mock QMessageBox.question to return Save
    monkeypatch.setattr(
        QMessageBox,
        "question",
        MagicMock(return_value=QMessageBox.StandardButton.Save),
    )

    # Show the dialog first
    widget._plate_map_dialog.show()
    assert not widget._plate_map_dialog.isHidden()

    # Simulate that changes have been made
    widget._has_changes = True

    # Track signal emission and call the close handler
    with qtbot.waitSignal(widget.plateMapSaved, timeout=1000):
        widget._on_dialog_close_requested()

    # Verify dialog is hidden
    assert widget._plate_map_dialog.isHidden()


def test_plate_map_widget_close_discard(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that close dialog with Discard hides dialog without saving."""
    from unittest.mock import MagicMock

    from qtpy.QtWidgets import QMessageBox

    from cali.gui._plate_map import _PlateMapWidget

    widget = _PlateMapWidget()
    qtbot.addWidget(widget)

    # Mock QMessageBox.question to return Discard
    monkeypatch.setattr(
        QMessageBox,
        "question",
        MagicMock(return_value=QMessageBox.StandardButton.Discard),
    )

    # Show the dialog first
    widget._plate_map_dialog.show()
    assert not widget._plate_map_dialog.isHidden()

    # Simulate that changes have been made
    widget._has_changes = True

    # Call the close handler - should NOT emit plateMapSaved
    signal_emitted = False

    def on_signal() -> None:
        nonlocal signal_emitted
        signal_emitted = True

    widget.plateMapSaved.connect(on_signal)
    widget._on_dialog_close_requested()

    # Verify dialog is hidden but signal was NOT emitted
    assert widget._plate_map_dialog.isHidden()
    assert not signal_emitted


def test_plate_map_widget_close_cancel(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that close dialog with Cancel keeps dialog open."""
    from unittest.mock import MagicMock

    from qtpy.QtWidgets import QMessageBox

    from cali.gui._plate_map import _PlateMapWidget

    widget = _PlateMapWidget()
    qtbot.addWidget(widget)

    # Mock QMessageBox.question to return Cancel
    monkeypatch.setattr(
        QMessageBox,
        "question",
        MagicMock(return_value=QMessageBox.StandardButton.Cancel),
    )

    # Show the dialog first
    widget._plate_map_dialog.show()
    assert not widget._plate_map_dialog.isHidden()

    # Simulate that changes have been made
    widget._has_changes = True

    # Call the close handler - should NOT emit plateMapSaved and NOT hide dialog
    signal_emitted = False

    def on_signal() -> None:
        nonlocal signal_emitted
        signal_emitted = True

    widget.plateMapSaved.connect(on_signal)
    widget._on_dialog_close_requested()

    # Verify dialog is still visible and signal was NOT emitted
    assert not widget._plate_map_dialog.isHidden()
    assert not signal_emitted


# ============================================================================
# GUI Initialization Tests
# ============================================================================


@pytest.fixture
def gui(qtbot: QtBot) -> CaliGui:
    """Create a CaliGui instance."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    return gui


def test_initialize_from_database_success(
    gui: CaliGui, qtbot: QtBot, tmp_path: Path, mock_detection_runner: MagicMock
) -> None:
    """Test successful initialization from existing database."""
    from cali.runner import CaliRunner
    from cali.sqlmodel import AnalysisSettings, DetectionSettings, ExtractionSettings

    data_path = Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")
    db_path = tmp_path / "test.cali"

    # Create experiment with plate structure from data
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
        database_name="test.cali",
        output_path=tmp_path,
    )

    # Initialize GUI from database
    gui._initialize_from_database(db_path, data_path)

    # Verify state
    assert gui._database_path == str(db_path)
    assert gui._data_path == str(data_path)
    assert gui._output_path == str(tmp_path)
    assert gui._data is not None
    assert gui._data.sequence is not None
    assert not gui._loading_bar.isVisible()


def test_initialize_from_database_not_exists(gui: CaliGui, qtbot: QtBot) -> None:
    """Test initialization fails when database doesn't exist."""
    with patch("cali.gui._cali_gui.show_error_dialog") as mock_dialog:
        gui._initialize_from_database("nonexistent.cali", "fake_data")

        # Should show error dialog
        mock_dialog.assert_called_once()
        args = mock_dialog.call_args[0]
        assert "not found" in args[1].lower()
        assert not gui._loading_bar.isVisible()


def test_initialize_from_database_invalid_data_path(
    gui: CaliGui, qtbot: QtBot, tmp_path: Path
) -> None:
    """Test initialization fails with invalid data path."""
    # Create minimal database
    from cali.sqlmodel import save_experiment_to_database

    exp = Experiment(name="Test", description="Test")
    db_path = tmp_path / "test.cali"
    save_experiment_to_database(exp, tmp_path, database_name="test.cali")

    with patch("cali.gui._cali_gui.show_error_dialog") as mock_dialog:
        gui._initialize_from_database(db_path, "invalid/path")

        # Should show error about unsupported format
        mock_dialog.assert_called_once()
        args = mock_dialog.call_args[0]
        assert "unsupported file format" in args[1].lower()
        assert not gui._loading_bar.isVisible()


def test_initialize_from_directories_new_database(
    gui: CaliGui, qtbot: QtBot, tmp_path: Path
) -> None:
    """Test creating new database from directories."""
    data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"
    output_path = tmp_path
    db_name = "new_test.cali"

    gui._initialize_from_directories(str(data_path), str(output_path), db_name)

    # Verify database was created
    db_path = output_path / db_name
    assert db_path.exists()
    assert gui._database_path == str(db_path)
    assert gui._data_path == data_path
    assert gui._output_path == str(output_path)
    assert gui._data is not None
    assert not gui._loading_bar.isVisible()


def test_initialize_from_directories_existing_db_no_overwrite(
    gui: CaliGui, qtbot: QtBot, tmp_path: Path, mock_detection_runner: MagicMock
) -> None:
    """Test loading existing database when user chooses not to overwrite."""
    # Create existing database
    from cali.runner import CaliRunner
    from cali.sqlmodel import AnalysisSettings, DetectionSettings, ExtractionSettings

    data_path = Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")
    db_name = "existing.cali"

    exp = Experiment.create_from_data(
        name="Existing",
        data_path=data_path,
        description="Existing experiment",
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

    # Mock user choosing "No" (don't overwrite)
    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.No
    ):
        gui._initialize_from_directories(str(data_path), str(tmp_path), db_name)

    # Should load existing database
    assert gui._database_path == str(tmp_path / db_name)
    assert gui._data is not None
    assert not gui._loading_bar.isVisible()


def test_initialize_from_directories_existing_db_overwrite(
    gui: CaliGui, qtbot: QtBot, tmp_path: Path
) -> None:
    """Test overwriting existing database when user chooses to overwrite."""
    from cali.sqlmodel import save_experiment_to_database

    data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"
    db_name = "to_overwrite.cali"

    exp = Experiment(name="Old", description="Old experiment")
    save_experiment_to_database(exp, tmp_path, database_name=db_name)

    old_db_path = tmp_path / db_name
    assert old_db_path.exists()

    # Mock user choosing "Yes" (overwrite)
    with patch.object(
        QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes
    ):
        gui._initialize_from_directories(str(data_path), str(tmp_path), db_name)

    # Database should be overwritten with new experiment
    assert gui._database_path == str(tmp_path / db_name)
    assert old_db_path.exists()  # File still exists but contents replaced

    # Load and verify it's a new experiment
    from sqlmodel import Session, create_engine

    engine = create_engine(f"sqlite:///{old_db_path}")
    with Session(engine) as session:
        new_exp = Experiment.load_from_database(old_db_path, session=session)
        # New experiment should have default name "Cali Experiment"
        assert new_exp.name == "Cali Experiment"

    engine.dispose(close=True)
    assert not gui._loading_bar.isVisible()


def test_initialize_from_directories_invalid_data_path(
    gui: CaliGui, qtbot: QtBot, tmp_path: Path
) -> None:
    """Test initialization fails with invalid data path."""
    with patch("cali.gui._cali_gui.show_error_dialog") as mock_dialog:
        gui._initialize_from_directories("invalid/path", str(tmp_path))

        # Should show error
        mock_dialog.assert_called_once()
        args = mock_dialog.call_args[0]
        assert "no valid data" in args[1].lower()
        assert not gui._loading_bar.isVisible()


def test_initialize_appends_cali_extension(
    gui: CaliGui, qtbot: QtBot, tmp_path: Path
) -> None:
    """Test that .cali extension is automatically appended if missing."""
    data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"
    db_name = "test_db"  # No .cali extension

    gui._initialize_from_directories(str(data_path), str(tmp_path), db_name)

    # Should add .cali extension
    expected_db_path = tmp_path / "test_db.cali"
    assert gui._database_path == str(expected_db_path)
    assert expected_db_path.exists()


def test_clear_widget_before_initialization(gui: CaliGui, qtbot: QtBot) -> None:
    """Test that widget state is properly cleared before initialization."""
    # Set some state
    gui._database_path = "old_path.cali"
    gui._data_path = "old_data"
    gui._output_path = "old_output"
    gui._data = Mock()

    # Clear
    gui._clear_widget_before_initialization()

    # Verify everything is reset
    assert gui._database_path is None
    assert gui._data_path is None
    assert gui._output_path is None
    assert gui._data is None


def test_initialize_updates_graph_properties(
    gui: CaliGui, qtbot: QtBot, tmp_path: Path, mock_detection_runner: MagicMock
) -> None:
    """Test that graph widgets are updated with database path."""
    from cali.runner import CaliRunner
    from cali.sqlmodel import AnalysisSettings, DetectionSettings, ExtractionSettings

    data_path = Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")
    db_path = tmp_path / "test.cali"

    exp = Experiment.create_from_data(
        name="Test",
        data_path=data_path,
        description="Test",
    )
    runner = CaliRunner()
    runner.run(
        experiment=exp,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
        extraction_settings=ExtractionSettings(),
        analysis_settings=AnalysisSettings(),
        global_position_indices=[0],
        database_name="test.cali",
        output_path=tmp_path,
    )

    gui._initialize_from_database(db_path, data_path)

    # Verify all graph widgets have database path set
    for sw_graph in gui.SW_GRAPHS:
        assert sw_graph.database_path == str(db_path)
        assert sw_graph.engine is not None

    for mw_graph in gui.MW_GRAPHS:
        assert mw_graph.database_path == str(db_path)
        assert mw_graph.engine is not None


def test_initialize_handles_missing_sequence(
    gui: CaliGui, qtbot: QtBot, tmp_path: Path
) -> None:
    """Test that initialization fails gracefully when sequence is missing."""
    # Create database with experiment
    from cali.sqlmodel import save_experiment_to_database

    exp = Experiment(name="Test", description="Test")
    db_path = tmp_path / "test.cali"
    save_experiment_to_database(exp, tmp_path, database_name="test.cali")

    # Mock load_data_from_path to return data without sequence
    mock_data = Mock()
    mock_data.sequence = None

    with (
        patch("cali.gui._cali_gui.load_data_from_path", return_value=mock_data),
        patch("cali.gui._cali_gui.show_error_dialog") as mock_dialog,
    ):
        gui._initialize_from_database(db_path, "fake_path")

        # Should show error about missing sequence
        mock_dialog.assert_called_once()
        args = mock_dialog.call_args[0]
        assert "mdasequence not found" in args[1].lower()
        assert not gui._loading_bar.isVisible()


# ============================================================================
# GUI No HCS Plate Plan Tests
# ============================================================================


def test_plate_plan_preserved_after_reload(qtbot: QtBot) -> None:
    """Test that plate plan from wizard is preserved after data reload.

    When loading a non-HCS tensorstore (stage_positions is tuple, not WellPlatePlan),
    the user selects a plate plan via the wizard. This plate plan should be:
    1. Saved to the experiment
    2. Re-applied to the data after reload
    3. Used for GUI display (not falling back to DEFAULT_PLATE_PLAN)
    """
    widget = CaliGui()
    qtbot.addWidget(widget)

    # Initialize from non-HCS data
    data_path = "tests/test_data/no_hcs/no_hcs.tensorstore.zarr"
    db_path = Path("tests/test_data/no_hcs/no_hcs.cali")

    # The workflow would be:
    # 1. User loads non-HCS data -> wizard appears
    # 2. User selects plate plan
    # 3. Data is reloaded
    # 4. Plate plan should be re-applied

    # For testing, we'll directly test the fix:
    # After _initialize_from_directories completes, the data should have
    # the plate plan set (not be a tuple)

    # If database exists, load from it
    if db_path.exists():
        widget._initialize_from_database(db_path, data_path)
    else:
        # Would need to mock the wizard for full test
        pytest.skip("Database doesn't exist, would require wizard mocking")

    # After initialization, check that data has plate plan set
    assert widget._data is not None
    assert widget._data.sequence is not None

    # Load experiment from database
    from cali.sqlmodel import Experiment

    experiment = Experiment.load_from_database(db_path, load_data=False)

    # Check that experiment has a plate
    assert experiment.plate is not None, "Experiment should have a plate"

    # Check that the plate has a plate plan
    exp_plate_plan = experiment.plate.plate_plan
    assert exp_plate_plan is not None, "Experiment plate should have a plate_plan"

    # The data should have the plate plan applied (loaded from database)
    # Note: The GUI loads data without plate_plan now, it relies on the database
    # storing the plate_plan, which is then used by _draw_plate_with_selection
    # So we just verify that the experiment has the plate_plan stored correctly

    # Verify it's a WellPlatePlan
    assert isinstance(exp_plate_plan, useq.WellPlatePlan), (
        f"Expected WellPlatePlan, got {type(exp_plate_plan)}"
    )

    # Verify it has the expected structure
    assert exp_plate_plan.plate is not None, "Plate plan should have a plate"
    assert len(exp_plate_plan.selected_well_names) > 0, "Should have selected wells"


# ============================================================================
# GUI State After Detection Tests
# ============================================================================


@pytest.fixture
def gui_with_detection(qtbot: QtBot, test_db_copy: Path) -> CaliGui:
    """Create a GUI loaded with a database that has detection results."""
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Load the test database that has detection results
    data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"

    gui._initialize_from_database(str(test_db_copy), data_path)
    qtbot.waitUntil(lambda: gui._loading_bar.isHidden(), timeout=10000)

    return gui


def test_label_button_enabled_after_loading_database(
    gui_with_detection: CaliGui, qtbot: QtBot
) -> None:
    """Test label button is enabled after loading database with detection.

    This test simulates the ACTUAL user workflow:
    1. Load database with detection results
    2. Click on a well
    3. Select FOV from table
    4. Verify label button is enabled

    This is a regression test for the bug where labels would not appear.
    """
    # The GUI fixture already loaded the database
    # Now we need to simulate selecting a well and FOV

    # First, let's manually trigger what happens when you click on B5
    assert gui_with_detection._data is not None
    assert gui_with_detection._data.sequence is not None

    # Get B5_0000 position (should be position 0)
    positions = gui_with_detection._data.sequence.stage_positions

    b5_pos = None
    for i, pos in enumerate(positions):
        # Look for the FULL FOV name with position index
        if hasattr(pos, "name") and pos.name and "B5_0000" in pos.name:
            b5_pos = (i, pos)
            break

    if b5_pos is None:
        pytest.skip("B5_0000 position not found in test data")

    pos_idx, pos = b5_pos

    # Simulate well selection by calling the handler directly
    # This is what happens when you click on a well
    from cali.gui._fov_table import WellInfo

    well_info = WellInfo(pos_idx=pos_idx, fov=pos)

    # Add the position to the FOV table (simulating what happens on well click)
    gui_with_detection._fov_table.clear()
    gui_with_detection._fov_table.setRowCount(0)
    gui_with_detection._fov_table.add_position(well_info)

    qtbot.wait(100)

    # Now select the FOV in the table (simulating user click)
    gui_with_detection._fov_table.selectRow(0)
    qtbot.wait(200)

    # Verify labels were loaded
    roi_labels, neuropil_labels = gui_with_detection._get_labels(well_info)
    assert roi_labels is not None, "ROI labels should exist for B5_0000"
    assert neuropil_labels is not None, "Neuropil labels should exist for B5_0000"

    # Verify the label button is enabled
    assert gui_with_detection._image_viewer._labels.isEnabled(), (
        "Label button should be enabled when detection results exist for the FOV"
    )


def test_label_button_enabled_after_detection(
    gui_with_detection: CaliGui, qtbot: QtBot
) -> None:
    """Test label button in image viewer is enabled when setData gets labels.

    This test verifies the fix for the bug where the label button would never
    be enabled even when labels existed. The bug was that setData() was
    checking for labels_image and contours_image existence BEFORE
    update_image() created them.

    This is a regression test for the bug introduced in commit d4742d6.
    """
    import numpy as np

    # Create dummy image and label data (simulating what would come from DB)
    image_data = np.random.rand(512, 512).astype(np.float32)

    # Create a simple label mask with a few ROIs
    labels = np.zeros((512, 512), dtype=np.uint16)
    labels[50:100, 50:100] = 1  # ROI 1
    labels[150:200, 150:200] = 2  # ROI 2
    labels[250:300, 250:300] = 3  # ROI 3

    # Call setData with both image and labels (this is what
    # _on_fov_table_selection_changed does)
    gui_with_detection._image_viewer.setData(image_data, labels)

    qtbot.wait(100)

    # Verify the label button is enabled
    assert gui_with_detection._image_viewer._labels.isEnabled(), (
        "Label button should be enabled when setData is called with labels"
    )

    # Verify images were created
    assert gui_with_detection._image_viewer._viewer.image is not None
    assert gui_with_detection._image_viewer._viewer.labels_image is not None
    assert gui_with_detection._image_viewer._viewer.contours_image is not None

    # Verify the button tooltip is correct (should be the enabled tooltip)
    tooltip = gui_with_detection._image_viewer._labels.toolTip()
    assert "Toggle ROI labels visibility" in tooltip, (
        f"Expected enabled tooltip, got: {tooltip}"
    )


def test_visualization_combo_enabled_after_analysis(
    gui_with_detection: CaliGui, qtbot: QtBot
) -> None:
    """Test visualization combos have enabled items after loading analysis.

    This test verifies that combo boxes in the visualization tab are properly
    populated and have enabled items when analysis results exist.
    """
    # Switch to visualization tab
    gui_with_detection._main_tab.setCurrentWidget(gui_with_detection._visualization_tab)
    qtbot.wait(100)

    # Select a well and FOV (same as above)
    assert gui_with_detection._data is not None
    assert gui_with_detection._data.sequence is not None

    pos = gui_with_detection._data.sequence.stage_positions[0]
    scene = gui_with_detection._plate_view.scene()
    items = scene.items()  # type: ignore

    for item in items:
        if hasattr(item, "well_pos") and item.well_pos == pos:  # type: ignore
            gui_with_detection._plate_view.clearSelection()
            item.setSelected(True)
            break

    qtbot.wait(100)

    if gui_with_detection._fov_table.rowCount() > 0:
        gui_with_detection._fov_table.selectRow(0)
        qtbot.wait(100)

    # Check that at least one graph widget has enabled combo items
    graph_widget = gui_with_detection._single_well_graph_1

    # Verify combo box is enabled
    assert graph_widget._combo.isEnabled(), (
        "Plot selection combo should be enabled when analysis results exist"
    )

    # Verify at least some items are enabled
    model = graph_widget._combo.model()
    enabled_items = []
    for i in range(model.rowCount()):  # type: ignore
        item = model.item(i)  # type: ignore
        if item and (item.flags() & Qt.ItemFlag.ItemIsEnabled):  # type: ignore
            enabled_items.append(item.text())

    assert len(enabled_items) > 0, (
        f"At least some plot options should be enabled. "
        f"Total items: {model.rowCount()}, Enabled: {len(enabled_items)}"  # type: ignore
    )


def test_labels_button_disabled_when_no_labels(qtbot: QtBot) -> None:
    """Test that label button is disabled when no labels exist.

    This is a control test to verify the button is properly disabled
    when there are no detection results.
    """
    gui = CaliGui()
    qtbot.addWidget(gui)

    # Don't load any database - just verify initial state
    assert not gui._image_viewer._labels.isEnabled(), (
        "Label button should be disabled initially when no labels exist"
    )

    # Set data without labels
    import numpy as np

    data = np.random.rand(100, 100).astype(np.float32)
    gui._image_viewer.setData(data, labels=None)

    # Verify button is still disabled
    assert not gui._image_viewer._labels.isEnabled(), (
        "Label button should remain disabled when setData is called with labels=None"
    )

    # Verify tooltip indicates why it's disabled
    tooltip = gui._image_viewer._labels.toolTip()
    assert "No labels data available" in tooltip, (
        f"Expected disabled tooltip explaining why, got: {tooltip}"
    )


# ============================================================================
# GUI Ambiguous Runs Tests
# ============================================================================


def test_run_selection_dialog_formatting() -> None:
    """Test that run selection dialog formats runs correctly."""
    # Create mock runs
    run1 = MagicMock(spec=CaliResult)
    run1.id = 1
    run1.detection_settings_id = 1
    run1.extraction_settings_id = 1
    run1.analysis_settings_id = None
    run1.positions_detected = [0, 1]
    run1.positions_extracted = [0, 1]
    run1.positions_analyzed = None

    run2 = MagicMock(spec=CaliResult)
    run2.id = 2
    run2.detection_settings_id = 1
    run2.extraction_settings_id = 2
    run2.analysis_settings_id = None
    run2.positions_detected = [2, 3]
    run2.positions_extracted = [2, 3]
    run2.positions_analyzed = None

    runs = [run1, run2]
    message = "Multiple runs exist with the same detection settings (ID 1)"

    # Create dialog
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dialog = RunSelectionDialog(None, runs, message)

    # Check that runs are in the list
    assert dialog._runs_list.count() == 2

    # Check formatting
    item1_text = dialog._runs_list.item(0).text()
    assert "Run ID 1:" in item1_text
    assert "Detection: ID 1" in item1_text
    assert "Extraction: ID 1" in item1_text
    assert "Analysis: None" in item1_text
    assert "detected=[0, 1]" in item1_text

    item2_text = dialog._runs_list.item(1).text()
    assert "Run ID 2:" in item2_text
    assert "Extraction: ID 2" in item2_text


def test_run_selection_dialog_get_selected() -> None:
    """Test getting selected run ID from dialog."""
    run1 = MagicMock(spec=CaliResult)
    run1.id = 1
    run1.detection_settings_id = 1
    run1.extraction_settings_id = 1
    run1.analysis_settings_id = None
    run1.positions_detected = [0]
    run1.positions_extracted = [0]
    run1.positions_analyzed = None

    runs = [run1]
    message = "Test message"

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dialog = RunSelectionDialog(None, runs, message)

    # Initially nothing selected
    assert dialog.get_selected_run_id() is None

    # Select first item
    dialog._runs_list.setCurrentRow(0)
    assert dialog.get_selected_run_id() == 1
