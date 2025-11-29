from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItemModel
from sqlmodel import Session, create_engine

from cali.gui import CaliGui
from cali.gui._run_widget import CaliRunSettings
from cali.sqlmodel._model import (
    AnalysisSettings,
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

    engine.dispose()
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

    engine.dispose()
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

    engine.dispose()
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


def test_on_worker_finished_selects_last_run(qtbot: QtBot) -> None:
    """Test that _on_worker_finished selects the last run in the runs panel."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui.show()
    qtbot.waitExposed(gui)

    # Add some items to the runs list to simulate existing runs
    gui._runs_panel._runs_list.addItem("Run 1")
    gui._runs_panel._runs_list.addItem("Run 2")
    gui._runs_panel._runs_list.addItem("Run 3")

    # Set run IDs in the item data (assuming Qt.ItemDataRole.UserRole stores the run ID)
    from qtpy.QtCore import Qt

    item0 = gui._runs_panel._runs_list.item(0)
    assert item0 is not None
    item0.setData(Qt.ItemDataRole.UserRole, 1)

    item1 = gui._runs_panel._runs_list.item(1)
    assert item1 is not None
    item1.setData(Qt.ItemDataRole.UserRole, 2)

    item2 = gui._runs_panel._runs_list.item(2)
    assert item2 is not None
    item2.setData(Qt.ItemDataRole.UserRole, 3)

    # Mock refresh_runs to do nothing so items remain
    import unittest.mock

    with unittest.mock.patch.object(gui._runs_panel, "refresh_runs"):
        # Call the method that should select the last run
        gui._on_worker_finished()

    # Check that the last item (index 2) is selected
    assert gui._runs_panel._runs_list.currentRow() == 2

    # Check that the graphs have been updated with the last run ID
    # Assuming there are graphs in SW_GRAPHS
    if gui.SW_GRAPHS:
        assert gui.SW_GRAPHS[0].run_id == 3
    if gui.MW_GRAPHS:
        assert gui.MW_GRAPHS[0].run_id == 3

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

    engine.dispose()
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
            detection_settings=1,
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

    engine.dispose()
    gui.close()


def test_analysis_only_requires_both_ids(
    qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that analysis-only mode requires both detection and extraction IDs."""
    from unittest.mock import MagicMock

    gui = CaliGui()
    qtbot.addWidget(gui)

    # Set up mock data
    gui._database_path = "test.cali"
    gui._data = MagicMock()
    gui._data.sequence = MagicMock()
    gui._data.sequence.stage_positions = [MagicMock()]

    # Mock show_error_dialog to track calls
    error_calls = []

    def mock_error_dialog(parent: object, msg: str) -> None:
        error_calls.append(msg)

    monkeypatch.setattr("cali.gui._cali_gui.show_error_dialog", mock_error_dialog)

    # Mock Experiment.load_from_db
    mock_exp = MagicMock()
    monkeypatch.setattr(
        "cali.gui._cali_gui.Experiment.load_from_db", lambda *args, **kwargs: mock_exp
    )

    # Test: Analysis-only with only extraction ID (missing detection ID)
    gui._run_cali_wdg._run_options_combo.setCurrentIndex(5)  # Analysis Only
    gui._run_cali_wdg._extraction_settings_combo.addItem("Extraction 1", 1)
    gui._run_cali_wdg._extraction_settings_combo.setCurrentIndex(1)
    # Detection combo is empty (no detection ID selected)

    gui._on_cali_run_clicked()

    assert len(error_calls) == 1
    assert "Detection ID" in error_calls[0]

    # Test: Analysis-only with only detection ID (missing extraction ID)
    error_calls.clear()
    gui._run_cali_wdg._detection_settings_combo.addItem("Detection 1", 1)
    gui._run_cali_wdg._detection_settings_combo.setCurrentIndex(1)
    # Clear extraction combo to None
    gui._run_cali_wdg._extraction_settings_combo.clear()
    gui._run_cali_wdg._extraction_settings_combo.addItem("Select...", None)

    gui._on_cali_run_clicked()

    assert len(error_calls) == 1
    assert "Extraction ID" in error_calls[0]

    # Test: Analysis-only with neither ID
    error_calls.clear()
    gui._run_cali_wdg._detection_settings_combo.clear()
    gui._run_cali_wdg._detection_settings_combo.addItem("Select...", None)

    gui._on_cali_run_clicked()

    assert len(error_calls) == 1
    assert "Detection ID" in error_calls[0] and "Extraction ID" in error_calls[0]

    gui.close()
