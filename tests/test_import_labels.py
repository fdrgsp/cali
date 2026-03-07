"""Tests for import_labels_to_database in cali.util."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import tifffile
from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel._model import (
    FOV,
    ROI,
    DetectionSettings,
)
from cali.util import import_labels_to_database

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock


@pytest.fixture
def populated_db(
    tmp_path: Path,
    test_experiment: Any,
    data_path: Path,
    mock_detection_runner: MagicMock,
) -> Path:
    """Create a database with an Experiment and FOVs via a mocked detection run."""
    db_path = tmp_path / "import_test.cali"
    runner = CaliRunner(commit_batch_size=1)
    detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        database_name=db_path.name,
        output_path=db_path.parent,
        global_position_indices=[0],
    )
    return db_path


@pytest.fixture
def label_tiff(tmp_path: Path) -> Path:
    """Create a simple 2D label TIFF with 3 labelled regions."""
    arr = np.zeros((256, 256), dtype=np.uint16)
    arr[10:30, 10:30] = 1
    arr[50:70, 50:70] = 2
    arr[100:120, 100:120] = 3
    p = tmp_path / "labels.tif"
    tifffile.imwrite(p, arr)
    return p


def _get_first_fov_name(db_path: Path) -> str:
    """Return the name of the first FOV in the database."""
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            fov = session.exec(select(FOV)).first()
            assert fov is not None
            return fov.name
    finally:
        engine.dispose(close=True)


def test_import_labels_creates_detection_settings(
    populated_db: Path, label_tiff: Path
) -> None:
    """import_labels_to_database creates DetectionSettings(method='imported_labels')."""
    fov_name = _get_first_fov_name(populated_db)
    det_id = import_labels_to_database(str(populated_db), {fov_name: label_tiff})

    engine = create_engine(f"sqlite:///{populated_db}")
    try:
        with Session(engine) as session:
            ds = session.get(DetectionSettings, det_id)
            assert ds is not None
            assert ds.method == "imported_labels"
    finally:
        engine.dispose(close=True)


def test_import_labels_creates_rois(populated_db: Path, label_tiff: Path) -> None:
    """import_labels_to_database creates ROIs for each label in the TIFF."""
    fov_name = _get_first_fov_name(populated_db)
    import_labels_to_database(str(populated_db), {fov_name: label_tiff})

    engine = create_engine(f"sqlite:///{populated_db}")
    try:
        with Session(engine) as session:
            rois = session.exec(select(ROI)).all()
            assert any(r.label_value in (1, 2, 3) for r in rois)
    finally:
        engine.dispose(close=True)


def test_import_labels_idempotent_detection_settings(
    populated_db: Path, label_tiff: Path
) -> None:
    """Calling import_labels_to_database twice reuses the same DetectionSettings."""
    fov_name = _get_first_fov_name(populated_db)
    det_id1 = import_labels_to_database(str(populated_db), {fov_name: label_tiff})
    det_id2 = import_labels_to_database(str(populated_db), {fov_name: label_tiff})
    assert det_id1 == det_id2


def test_import_labels_skips_unknown_fov_name(
    populated_db: Path, label_tiff: Path
) -> None:
    """import_labels_to_database silently skips FOV names not found in DB."""
    import_labels_to_database(str(populated_db), {"DOES_NOT_EXIST_9999": label_tiff})


def test_import_labels_skips_non_2d_label(populated_db: Path, tmp_path: Path) -> None:
    """import_labels_to_database skips label TIFFs that are not 2D."""
    arr_3d = np.zeros((5, 256, 256), dtype=np.uint16)
    arr_3d[0, 10:30, 10:30] = 1
    label_3d = tmp_path / "labels_3d.tif"
    tifffile.imwrite(label_3d, arr_3d)

    fov_name = _get_first_fov_name(populated_db)
    import_labels_to_database(str(populated_db), {fov_name: label_3d})


def test_import_labels_skips_empty_label(populated_db: Path, tmp_path: Path) -> None:
    """import_labels_to_database skips label TIFFs with no foreground labels."""
    empty_arr = np.zeros((256, 256), dtype=np.uint16)
    empty_label = tmp_path / "labels_empty.tif"
    tifffile.imwrite(empty_label, empty_arr)

    fov_name = _get_first_fov_name(populated_db)
    import_labels_to_database(str(populated_db), {fov_name: empty_label})


def test_import_labels_raises_on_missing_experiment(
    tmp_path: Path, label_tiff: Path
) -> None:
    """import_labels_to_database raises ValueError when no Experiment in DB."""
    from cali.sqlmodel._model import SQLModel

    db_path = tmp_path / "no_experiment.cali"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    engine.dispose(close=True)

    with pytest.raises(ValueError, match="No experiment found in database"):
        import_labels_to_database(str(db_path), {"A1_0000": label_tiff})
