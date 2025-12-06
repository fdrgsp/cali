"""Test CaliGui initialization from database and directories."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QMessageBox

from cali.gui import CaliGui
from cali.sqlmodel import FOV, ROI, Experiment, Mask


def create_mock_fov(
    position_index: int = 0, num_rois: int = 3, name: str | None = None
) -> FOV:
    """Create a mock FOV with ROIs for testing without running cellpose."""
    if name is None:
        name = "B5_0000" if position_index == 0 else "B6_0000"
    fov = FOV(position_index=position_index, name=name)

    rois = []
    for i in range(1, num_rois + 1):
        mask_data = np.zeros((256, 256), dtype=np.uint8)
        cy, cx = 50 + i * 20, 50 + i * 20
        y, x = np.ogrid[:256, :256]
        mask_region = ((x - cx) ** 2 + (y - cy) ** 2) <= 100
        mask_data[mask_region] = 1

        coords = np.where(mask_data)
        coords_y = coords[0].tolist()
        coords_x = coords[1].tolist()

        mask = Mask(
            mask_type="roi",
            coords_y=coords_y,
            coords_x=coords_x,
            height=256,
            width=256,
        )

        roi = ROI(label_value=i, roi_mask=mask)
        rois.append(roi)

    fov.rois = rois
    return fov


@pytest.fixture
def mock_cellpose() -> Iterator[MagicMock]:
    """Mock cellpose detection to avoid slow model loading."""
    with patch(
        "cali.detection._detection_runner.DetectionRunner._run_cellpose"
    ) as mock:

        def mock_detection(
            dataset: Any,
            detection_settings: Any,
            position_indices: list[int],
            *args: Any,
            **kwargs: Any,
        ) -> Iterator[FOV]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection
        yield mock


@pytest.fixture
def gui(qtbot: QtBot) -> CaliGui:
    """Create a CaliGui instance."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    return gui


def test_initialize_from_database_success(
    gui: CaliGui, qtbot: QtBot, tmp_path: Path, mock_cellpose: MagicMock
) -> None:
    """Test successful initialization from existing database."""
    from cali.runner import CaliRunner
    from cali.sqlmodel import AnalysisSettings, DetectionSettings, ExtractionSettings

    data_path = Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")
    db_path = tmp_path / "test.cali"

    # Create experiment with database
    exp = Experiment(name="Test", description="Test experiment")
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
    gui: CaliGui, qtbot: QtBot, tmp_path: Path, mock_cellpose: MagicMock
) -> None:
    """Test loading existing database when user chooses not to overwrite."""
    # Create existing database
    from cali.runner import CaliRunner
    from cali.sqlmodel import AnalysisSettings, DetectionSettings, ExtractionSettings

    data_path = Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")
    db_name = "existing.cali"

    exp = Experiment(name="Existing", description="Existing experiment")
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
        new_exp = Experiment.load_from_db(old_db_path, session=session)
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
    gui: CaliGui, qtbot: QtBot, tmp_path: Path, mock_cellpose: MagicMock
) -> None:
    """Test that graph widgets are updated with database path."""
    from cali.runner import CaliRunner
    from cali.sqlmodel import AnalysisSettings, DetectionSettings, ExtractionSettings

    data_path = Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")
    db_path = tmp_path / "test.cali"

    exp = Experiment(name="Test", description="Test")
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
