"""Tests for natural_sort_key and its correct use across sort call-sites."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from cali._constants import natural_sort_key

if TYPE_CHECKING:
    from pathlib import Path

    from pytestqt.qtbot import QtBot


# ---------------------------------------------------------------------------
# natural_sort_key unit tests
# ---------------------------------------------------------------------------


def test_natural_sort_key_pure_strings() -> None:
    """Alphabetic-only strings sort lexicographically as expected."""
    assert sorted(["B", "A", "C"], key=natural_sort_key) == ["A", "B", "C"]


def test_natural_sort_key_numeric_suffix() -> None:
    """Numbers in names are compared as integers, not digit-by-digit."""
    assert sorted(["B10", "B2", "B1", "B9"], key=natural_sort_key) == [
        "B1",
        "B2",
        "B9",
        "B10",
    ]


def test_natural_sort_key_fov_names() -> None:
    """Well_FOV names like B2_0000 sort numerically on both parts."""
    names = ["B10_0000", "B2_0000", "B1_0000", "A10_0000", "A2_0000"]
    assert sorted(names, key=natural_sort_key) == [
        "A2_0000",
        "A10_0000",
        "B1_0000",
        "B2_0000",
        "B10_0000",
    ]


def test_natural_sort_key_fov_index() -> None:
    """FOV index part also sorts numerically (0010 > 0002)."""
    names = ["A1_0010", "A1_0002", "A1_0001"]
    assert sorted(names, key=natural_sort_key) == ["A1_0001", "A1_0002", "A1_0010"]


def test_natural_sort_key_empty_string() -> None:
    """Empty string does not raise and sorts before non-empty strings."""
    assert sorted(["B1", "", "A1"], key=natural_sort_key) == ["", "A1", "B1"]


@pytest.mark.parametrize(
    "unsorted_input, expected",
    [
        (["B10", "B2", "B1"], ["B1", "B2", "B10"]),
        (["A10_0000", "A2_0000", "A1_0000"], ["A1_0000", "A2_0000", "A10_0000"]),
        (["well_10", "well_2", "well_1"], ["well_1", "well_2", "well_10"]),
    ],
)
def test_natural_sort_key_parametrized(
    unsorted_input: list[str], expected: list[str]
) -> None:
    assert sorted(unsorted_input, key=natural_sort_key) == expected


# ---------------------------------------------------------------------------
# TiffCollectionWidget: files sorted naturally
# ---------------------------------------------------------------------------


def test_tiff_files_sorted_naturally_from_directory(
    tmp_path: Path, qtbot: QtBot
) -> None:
    """TIFF files from directory are ordered naturally, not lexicographically."""
    from cali.gui._tiff_collection_widget import TiffCollectionWidget

    for name in ("B10_0000.tif", "B2_0000.tif", "B1_0000.tif"):
        tifffile.imwrite(tmp_path / name, np.zeros((5, 5), dtype=np.uint16))

    widget = TiffCollectionWidget()
    qtbot.addWidget(widget)
    widget.set_tiff_files(tmp_path)

    names = [f.name for f in widget._tiff_files]
    assert names == ["B1_0000.tif", "B2_0000.tif", "B10_0000.tif"]


def test_tiff_files_sorted_naturally_from_list(tmp_path: Path, qtbot: QtBot) -> None:
    """TIFF files provided as a list are ordered naturally."""
    from cali.gui._tiff_collection_widget import TiffCollectionWidget

    files = []
    for name in ("B10_0000.tif", "B2_0000.tif", "B1_0000.tif"):
        p = tmp_path / name
        tifffile.imwrite(p, np.zeros((5, 5), dtype=np.uint16))
        files.append(p)

    widget = TiffCollectionWidget()
    qtbot.addWidget(widget)
    widget.set_tiff_files(files)

    names = [f.name for f in widget._tiff_files]
    assert names == ["B1_0000.tif", "B2_0000.tif", "B10_0000.tif"]


def test_file_map_sorted_naturally_per_well(tmp_path: Path, qtbot: QtBot) -> None:
    """Matched files stored per well in _file_map are ordered naturally."""
    from useq import WellPlatePlan

    from cali.gui._tiff_collection_widget import TiffCollectionWidget

    files = []
    for name in ("A1_0010.tif", "A1_0002.tif", "A1_0001.tif"):
        p = tmp_path / name
        tifffile.imwrite(p, np.zeros((5, 5), dtype=np.uint16))
        files.append(p)

    widget = TiffCollectionWidget()
    qtbot.addWidget(widget)
    plate_plan = WellPlatePlan(plate="96-well", a1_center_xy=(0, 0))
    widget._plate_widget.setValue(plate_plan)
    widget.set_tiff_files(files)

    a1_files = widget._file_map.get((0, 0), [])
    assert len(a1_files) == 3
    assert [f.name for f in a1_files] == ["A1_0001.tif", "A1_0002.tif", "A1_0010.tif"]


# ---------------------------------------------------------------------------
# _ImportLabelsDialog: folder scan sorted naturally
# ---------------------------------------------------------------------------


def test_import_labels_folder_sorted_naturally(
    tmp_path: Path, qtbot: QtBot, populated_db: Path
) -> None:
    """Label files from a folder scan are ordered naturally, not lexicographically."""
    from cali.gui._import_labels_dialog import _ImportLabelsDialog

    for name in ("B10_labels.tif", "B2_labels.tif", "B1_labels.tif"):
        tifffile.imwrite(tmp_path / name, np.zeros((5, 5), dtype=np.uint16))

    dialog = _ImportLabelsDialog(str(populated_db))
    qtbot.addWidget(dialog)
    dialog._on_folder_selected(str(tmp_path))

    names = [f.name for f in dialog._label_files]
    assert names == ["B1_labels.tif", "B2_labels.tif", "B10_labels.tif"]


# ---------------------------------------------------------------------------
# _database_to_csv: FOV column ordering in trace CSV
# ---------------------------------------------------------------------------


def test_export_traces_fov_columns_natural_order(tmp_path: Path) -> None:
    """CSV columns from _export_trace_data are ordered naturally by FOV name."""
    import pandas as pd
    from sqlalchemy import create_engine
    from sqlmodel import Session

    from cali._constants import RAW_CALCIUM_TRACES
    from cali.sqlmodel._model import (
        FOV,
        ROI,
        AnalysisSettings,
        CaliResult,
        DataAnalysis,
        Experiment,
        Traces,
    )
    from cali.sqlmodel._util import create_database_and_tables
    from cali.util._database_to_csv import _export_trace_data

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    # Insert in non-natural order — without natural_sort_key the CSV columns
    # would come out as B1_0000, B10_0000, B2_0000 (lexicographic, wrong).
    fov_names = ["B10_0000", "B2_0000", "B1_0000"]

    with Session(engine) as session:
        exp = Experiment(name="test_exp")
        session.add(exp)
        session.flush()

        settings = AnalysisSettings(frame_rate=10.0)
        session.add(settings)
        session.flush()

        run = CaliResult(experiment=exp.id, analysis_settings_id=settings.id)
        session.add(run)
        session.flush()
        run_id: int = run.id  # type: ignore[assignment]

        for i, fov_name in enumerate(fov_names):
            fov = FOV(name=fov_name, position_index=i)
            session.add(fov)
            session.flush()

            roi = ROI(fov_id=fov.id, label_value=1, stimulated=True)
            session.add(roi)
            session.flush()

            da = DataAnalysis(roi_id=roi.id, analysis_result_id=run_id)
            session.add(da)

            traces = Traces(
                roi_id=roi.id,
                analysis_result_id=run_id,
                raw_trace=[1.0, 2.0, 3.0],
            )
            session.add(traces)

        session.commit()

    output = tmp_path / "traces.csv"
    _export_trace_data(engine, output, RAW_CALCIUM_TRACES, run_id=run_id)

    df = pd.read_csv(output)
    # Column names look like "B1_0000_ROI_1_stim" — extract the FOV prefix
    fov_prefixes = [col.rsplit("_ROI_", 1)[0] for col in df.columns]
    assert fov_prefixes == ["B1_0000", "B2_0000", "B10_0000"]
