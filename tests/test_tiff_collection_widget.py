"""Tests for TiffCollectionWidget auto-assign and reset functionality."""

from pathlib import Path

import pytest
import tifffile
from pytestqt.qtbot import QtBot
from useq import WellPlatePlan

from cali.gui._tiff_collection_widget import TiffCollectionWidget


@pytest.fixture
def tiff_files_96well(tmp_path: Path) -> list[Path]:
    """Create dummy TIFF files named after 96-well plate wells."""
    import numpy as np

    files = []
    for well in ("A1", "A2", "B1"):
        for i in range(2):
            p = tmp_path / f"{well}_fov{i:04d}.tif"
            tifffile.imwrite(p, np.zeros((5, 5), dtype=np.uint16))
            files.append(p)
    return files


@pytest.fixture
def tiff_widget(qtbot: QtBot) -> TiffCollectionWidget:
    """Create a TiffCollectionWidget with a 96-well plate plan set."""
    widget = TiffCollectionWidget()
    qtbot.addWidget(widget)
    plate_plan = WellPlatePlan(plate="96-well", a1_center_xy=(0, 0))
    widget._plate_widget.setValue(plate_plan)
    return widget


def test_auto_assign_files_to_wells(
    tiff_widget: TiffCollectionWidget, tiff_files_96well: list[Path]
) -> None:
    """Files matching well names are auto-assigned on set_tiff_files call."""
    tiff_widget.set_tiff_files(tiff_files_96well)

    # A1 and A2 each had 2 files, B1 had 2 files
    assert (0, 0) in tiff_widget._file_map  # A1
    assert (0, 1) in tiff_widget._file_map  # A2
    assert (1, 0) in tiff_widget._file_map  # B1
    assert len(tiff_widget._file_map[(0, 0)]) == 2
    assert len(tiff_widget._file_map[(0, 1)]) == 2
    assert len(tiff_widget._file_map[(1, 0)]) == 2


def test_auto_assign_no_files_leaves_empty_map(
    tiff_widget: TiffCollectionWidget,
) -> None:
    """Auto-assign with no files leaves the file map empty."""
    tiff_widget.set_tiff_files([])
    assert tiff_widget._file_map == {}


def test_auto_assign_no_matching_files(
    tiff_widget: TiffCollectionWidget, tmp_path: Path
) -> None:
    """Files with no well-name pattern produce an empty file map."""
    import numpy as np

    files = [tmp_path / "random_image_001.tif", tmp_path / "random_image_002.tif"]
    for f in files:
        tifffile.imwrite(f, np.zeros((5, 5), dtype=np.uint16))

    tiff_widget.set_tiff_files(files)
    assert tiff_widget._file_map == {}


def test_auto_assign_plate_change_clears_and_reassigns(
    tiff_widget: TiffCollectionWidget, tiff_files_96well: list[Path]
) -> None:
    """Changing plate type clears existing assignments and re-runs auto-assign."""
    tiff_widget.set_tiff_files(tiff_files_96well)
    assert tiff_widget._file_map  # some matches expected

    # Switch to 6-well plate (A1 well still exists, others may not)
    new_plan = WellPlatePlan(plate="6-well", a1_center_xy=(0, 0))
    tiff_widget._plate_widget.setValue(new_plan)

    # After plate change, file map should be rebuilt (not contain previous state)
    # 6-well has A1..B3; at most A1 would match
    for well_key in tiff_widget._file_map:
        assert well_key in {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}


def test_reset_clears_all_assignments(
    tiff_widget: TiffCollectionWidget, tiff_files_96well: list[Path]
) -> None:
    """Clicking reset clears all file-to-well assignments."""
    tiff_widget.set_tiff_files(tiff_files_96well)
    assert tiff_widget._file_map  # populated by auto-assign

    tiff_widget._reset_btn.click()

    assert tiff_widget._file_map == {}
    assert tiff_widget._assigned_list.count() == 0


def test_reset_with_empty_map_is_noop(tiff_widget: TiffCollectionWidget) -> None:
    """Clicking reset when no assignments exist does not raise."""
    tiff_widget._reset_btn.click()
    assert tiff_widget._file_map == {}


def test_auto_assign_direct_call_no_files(tiff_widget: TiffCollectionWidget) -> None:
    """Calling _auto_assign_to_wells directly with no _tiff_files is a no-op."""
    tiff_widget._tiff_files = []
    tiff_widget._auto_assign_to_wells()
    assert tiff_widget._file_map == {}


def test_auto_assign_direct_call_no_plate(
    qtbot: QtBot, tiff_files_96well: list[Path]
) -> None:
    """_auto_assign_to_wells exits early when plate plan has no plate object."""
    from unittest.mock import MagicMock, patch

    widget = TiffCollectionWidget()
    qtbot.addWidget(widget)
    # Set files directly without triggering auto-assign
    widget._tiff_files = tiff_files_96well

    # Make the plate widget return a plan with plate=None
    mock_plan = MagicMock()
    mock_plan.plate = None
    with patch.object(widget._plate_widget, "value", return_value=mock_plan):
        widget._auto_assign_to_wells()

    assert widget._file_map == {}
