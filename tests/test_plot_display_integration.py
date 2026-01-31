"""Integration test to verify plots actually display correctly."""

import pytest
from pytestqt.qtbot import QtBot
from sqlmodel import Session, select

from cali.gui import CaliGui
from cali.sqlmodel._model import FOV


@pytest.fixture
def gui_with_test_data(qtbot: QtBot) -> CaliGui:
    """Create a CaliGui instance with test data."""
    gui = CaliGui()
    qtbot.addWidget(gui)
    gui._database_path = "tests/test_data/data_and_db_for_tests/test_db.cali"
    gui._data_path = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"
    gui._initialize_from_database(gui._database_path, gui._data_path)
    return gui


def test_single_well_plot_displays(gui_with_test_data: CaliGui, qtbot: QtBot) -> None:
    """Test that selecting a plot type actually renders plot items."""
    widget = gui_with_test_data.SW_GRAPHS[0]

    # Set FOV
    with Session(widget._engine) as session:
        fov = session.exec(select(FOV).limit(1)).first()
        assert fov is not None, "No FOV found in test database"

        widget.fov = fov.name

    # Select a known plot
    plot_name = "Calcium Raw Traces"
    idx = widget._combo.findText(plot_name)
    assert idx >= 0, f"Plot '{plot_name}' not found in combo"

    widget._combo.setCurrentIndex(idx)

    # Verify plot has items
    plot = widget.plot_item
    assert plot is not None, "Plot item is None"
    assert len(plot.items) > 0, f"Plot has no items after selecting '{plot_name}'"

    # Verify combo shows correct selection
    assert widget._combo.currentText() == plot_name


def test_combo_preserves_selection_on_fov_change(
    gui_with_test_data: CaliGui, qtbot: QtBot
) -> None:
    """Test that combo selection persists when changing FOV."""
    widget = gui_with_test_data.SW_GRAPHS[0]

    # Get two FOVs
    with Session(widget._engine) as session:
        fovs = list(session.exec(select(FOV).limit(2)).all())
        assert len(fovs) >= 2, "Need at least 2 FOVs for this test"

        # Set first FOV
        widget.fov = fovs[0].name

    # Select a plot
    plot_name = "Calcium Raw Traces"
    widget._combo.setCurrentText(plot_name)
    assert widget._combo.currentText() == plot_name

    # Change to second FOV
    widget.fov = fovs[1].name

    # Combo selection should persist
    assert widget._combo.currentText() == plot_name, (
        "Combo selection did not persist after FOV change"
    )


def test_combo_preserves_enabled_selection_on_run_change(
    gui_with_test_data: CaliGui, qtbot: QtBot
) -> None:
    """Test combo selection persists when changing run ID if still available."""
    widget = gui_with_test_data.SW_GRAPHS[0]

    # Set FOV
    with Session(widget._engine) as session:
        fov = session.exec(select(FOV).limit(1)).first()
        widget.fov = fov.name

    # Select a plot
    plot_name = "Calcium Raw Traces"
    widget._combo.setCurrentText(plot_name)
    original_selection = widget._combo.currentText()

    # Change run_id (simulate changing analysis run)
    widget.run_id = 1  # Set to specific run

    # If the plot is still available, selection should persist
    # If not available, it should reset to "None"
    current_selection = widget._combo.currentText()
    assert current_selection in [
        original_selection,
        "None",
    ], f"Unexpected selection '{current_selection}' after run change"


@pytest.mark.parametrize(
    "plot_name",
    [
        "Calcium Raw Traces",
        "Calcium ΔF/F0 Traces",
        "Cell Size",
    ],
)
def test_multiple_plot_types_render(
    gui_with_test_data: CaliGui, qtbot: QtBot, plot_name: str
) -> None:
    """Test that different plot types can be rendered."""
    widget = gui_with_test_data.SW_GRAPHS[0]

    # Set FOV
    with Session(widget._engine) as session:
        fov = session.exec(select(FOV).limit(1)).first()
        widget.fov = fov.name

    # Try to select the plot
    idx = widget._combo.findText(plot_name)
    if idx < 0:
        pytest.skip(f"Plot '{plot_name}' not available in test data")

    widget._combo.setCurrentIndex(idx)

    # Verify plot renders
    plot = widget.plot_item
    assert len(plot.items) > 0, f"Plot '{plot_name}' did not render any items"


def test_plot_clears_when_selecting_none(
    gui_with_test_data: CaliGui, qtbot: QtBot
) -> None:
    """Test that plot clears when selecting 'None'."""
    widget = gui_with_test_data.SW_GRAPHS[0]

    # Set FOV and select a plot
    with Session(widget._engine) as session:
        fov = session.exec(select(FOV).limit(1)).first()
        widget.fov = fov.name

    widget._combo.setCurrentText("Calcium Raw Traces")
    assert len(widget.plot_item.items) > 0, "Plot should have items"

    # Select "None"
    widget._combo.setCurrentText("None")

    # Plot should be cleared
    assert len(widget.plot_item.items) == 0, (
        "Plot should be empty after selecting 'None'"
    )
