"""Test that custom Cellpose model settings persist when switching databases."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot

from cali.gui import CaliGui
from cali.gui._detection_gui import CellposeSettingsData
from cali.sqlmodel import Experiment


@pytest.fixture
def gui(qtbot: QtBot, tmp_path: Path) -> CaliGui:
    """Create a CaliGui instance."""
    widget = CaliGui()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def test_database(tmp_path: Path) -> Path:
    """Create a minimal test database."""
    from cali.sqlmodel._model import SQLModel

    db_path = tmp_path / "test_runners.cali"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create tables
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Create minimal experiment
        experiment = Experiment(
            name="Test Experiment",
            description="Test experiment for custom model testing",
        )
        session.add(experiment)
        session.commit()

    engine.dispose(close=True)
    return db_path


def test_custom_model_combo_uses_findtext_for_selection(
    gui: CaliGui,
    qtbot: QtBot,
    test_database: Path,
    data_path: Path,
    tmp_path: Path,
) -> None:
    """Test that custom model selection uses findText to ensure robustness.

    This tests the fix for the issue where loading a second database with custom
    model settings would fail because setCurrentText didn't work reliably.
    The fix uses findText + setCurrentIndex instead.
    """
    # Initialize from database
    gui._initialize_from_database(str(test_database), str(data_path))
    qtbot.wait(100)

    detection_wdg = gui._detection_wdg._cellpose_wdg

    # Test that setValue properly uses findText for "custom"
    custom_model_path = str(tmp_path / "custom_model.pth")
    custom_settings = CellposeSettingsData(
        model_type="custom",
        model_path=custom_model_path,
        diameter=30,
        cellprob_threshold=-0.5,
    )

    # This should use findText internally
    detection_wdg.setValue(custom_settings)
    qtbot.wait(50)

    # Verify it worked
    assert detection_wdg._models_combo.currentText() == "custom"
    assert detection_wdg._browse_custom_model.value() == custom_model_path
    assert detection_wdg._browse_custom_model.isVisible()
    assert detection_wdg._diameter_spin.value() == 30
    assert detection_wdg._cellprob_threshold_spin.value() == -0.5

    # Now test with a different model type
    default_settings = CellposeSettingsData()
    detection_wdg.setValue(default_settings)
    qtbot.wait(50)

    # Should revert to default model
    assert detection_wdg._models_combo.currentText() != "custom"
    assert not detection_wdg._browse_custom_model.isVisible()

    # Switch back to custom
    detection_wdg.setValue(custom_settings)
    qtbot.wait(50)

    # Should work again
    assert detection_wdg._models_combo.currentText() == "custom"
    assert detection_wdg._browse_custom_model.isVisible()


def test_custom_model_combo_item_always_available(gui: CaliGui, qtbot: QtBot) -> None:
    """Test that 'custom' is always available in the models combo box."""
    detection_wdg = gui._detection_wdg._cellpose_wdg

    # Check that "custom" exists in combo
    custom_idx = detection_wdg._models_combo.findText("custom")
    assert custom_idx >= 0, "'custom' should always be in the combo box"

    # Verify we can select it
    detection_wdg._models_combo.setCurrentIndex(custom_idx)
    qtbot.wait(50)

    assert detection_wdg._models_combo.currentText() == "custom"
    assert detection_wdg._browse_custom_model.isVisible()
