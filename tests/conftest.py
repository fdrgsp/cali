"""Pytest configuration for cali tests.

CRITICAL: This file is imported BEFORE pytest plugins initialize.
On Windows, we must import torch before PyQt6 to avoid DLL conflicts.
The pytest-qt plugin auto-imports PyQt6 during initialization, so we
preload torch here to ensure proper DLL loading order.
"""

import shutil
import sys
import tempfile
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlmodel import Session
from useq import register_well_plates

from cali.sqlmodel import (
    FOV,
    Condition,
    Experiment,
    FOVAnalysis,
    Plate,
    Well,
)
from cali.sqlmodel._model import (
    ROI,
    AnalysisSettings,
    CaliResult,
    DataAnalysis,
    Mask,
)
from cali.sqlmodel._util import create_database_and_tables

# Import torch before pytest-qt initializes on Windows
if sys.platform == "win32":
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        # Torch/cellpose might not be installed, that's ok
        pass


TempDB = tuple[Engine, Path]


register_well_plates(
    {
        "dish-35mm-round": {
            "rows": 1,
            "columns": 1,
            "well_spacing": 0.0,
            "well_size": 35.0,
            "circular_wells": True,
            "name": "dish-35mm-round",
        },
    }
)


@pytest.fixture(autouse=True)
def _mock_pyconify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Mock pyconify.svg_path to avoid network requests in tests."""
    svg_dir = tmp_path / "icons"
    svg_dir.mkdir()
    _counter = 0

    def mock_svg_path(*key: str, color: str | None = None, **kwargs: object) -> Path:
        nonlocal _counter
        fill = color or "currentColor"
        svg_content = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            f'<rect width="24" height="24" fill="{fill}"/></svg>'
        )
        svg_file = svg_dir / f"icon_{_counter}.svg"
        _counter += 1
        svg_file.write_text(svg_content)
        return svg_file

    monkeypatch.setattr("pyconify.api.svg_path", mock_svg_path)
    monkeypatch.setattr("superqt.iconify.svg_path", mock_svg_path, raising=False)
    yield


@pytest.fixture
def temp_db() -> Generator[TempDB, None, None]:
    """Create a temporary SQLite database for testing."""
    import gc

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    engine = create_engine(f"sqlite:///{db_path}")
    create_database_and_tables(engine)

    yield engine, db_path

    # Cleanup - dispose engine before deleting file
    # Dispose with close=True to close all checked-in connections (Python 3.13)
    engine.dispose(close=True)
    # Force garbage collection to ensure connections are closed
    gc.collect()
    # Delete the database file
    try:
        db_path.unlink(missing_ok=True)
    except PermissionError:
        # On Windows, file might still be locked
        pass


_TEST_DB = Path("tests/test_data/data_and_db_for_tests/test_db.cali")


@pytest.fixture
def test_db_copy(tmp_path: Path) -> Path:
    """Return a disposable copy of test_db.cali so the original is never modified."""
    dest = tmp_path / "test_db.cali"
    shutil.copy2(_TEST_DB, dest)
    return dest


@pytest.fixture
def data_path() -> Path:
    """Return path to test data."""
    path = Path("tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr")
    if not path.exists():
        pytest.skip(f"Test data not found at {path}")
    return path


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Create a test database path."""
    return tmp_path / "test_runners.cali"


@pytest.fixture
def test_experiment(data_path: Path) -> Experiment:
    """Create a test experiment."""
    exp = Experiment.create_from_data(
        name="Test Runner Experiment",
        data_path=str(data_path),
    )
    return exp


def create_mock_fov(position_index: int = 0, num_rois: int = 3) -> FOV:
    """Create a mock FOV with ROIs for testing without running cellpose."""
    fov = FOV(position_index=position_index, name=f"A1_{position_index:04d}")

    rois = []
    for i in range(1, num_rois + 1):
        # Create a simple circular mask matching dataset dims (256x256)
        mask_data = np.zeros((256, 256), dtype=np.uint8)
        cy, cx = 50 + i * 20, 50 + i * 20
        y, x = np.ogrid[:256, :256]
        mask_region = ((x - cx) ** 2 + (y - cy) ** 2) <= 100
        mask_data[mask_region] = 1

        # Get coordinates from mask
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

        roi = ROI(
            label_value=i,
            roi_mask=mask,
        )
        rois.append(roi)

    fov.rois = rois
    return fov


@pytest.fixture
def populated_db(
    tmp_path: Path,
    test_experiment: Any,
    data_path: Path,
    mock_detection_runner: MagicMock,
) -> Path:
    """Create a database with an Experiment and FOVs via a mocked detection run."""
    from cali.runner import CaliRunner
    from cali.sqlmodel._model import DetectionSettings

    db_path = tmp_path / "test_populated.cali"
    runner = CaliRunner()
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=DetectionSettings(method="cellpose", model_type="cpsam"),
        database_name=db_path.name,
        output_path=db_path.parent,
        global_position_indices=[0],
    )
    return db_path


@pytest.fixture
def label_tiff(tmp_path: Path) -> Path:
    """Create a simple 2D label TIFF with 3 labelled regions."""
    import tifffile

    arr = np.zeros((256, 256), dtype=np.uint16)
    arr[10:30, 10:30] = 1
    arr[50:70, 50:70] = 2
    arr[100:120, 100:120] = 3
    p = tmp_path / "A1_0000_labels.tif"
    tifffile.imwrite(p, arr)
    return p


@pytest.fixture()
def mock_detection_runner() -> Iterator[MagicMock]:
    """Fixture that patches DetectionRunner to return mock FOVs quickly.

    This avoids slow Cellpose model loading and execution during tests.
    """
    with patch(
        "cali.detection._detection_runner.DetectionRunner._run_cellpose"
    ) as mock:

        def mock_detection(
            dataset: Any,
            detection_settings: Any,
            position_indices: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Iterator[FOV]:
            for pos_idx in position_indices:
                yield create_mock_fov(pos_idx)

        mock.side_effect = mock_detection
        yield mock


# ---------------------------------------------------------------------------
# Shared rich in-memory DB: 2 conditions x 2 FOVs with ROIs, burst stats,
# synchrony, correlation, CCG, and stim data. Reused across multi-well tests.
# ---------------------------------------------------------------------------


def _build_full_db() -> tuple[Engine, int]:
    """In-memory DB with rich data for synchrony, correlation, PCA tests."""
    import gc

    engine = create_engine("sqlite:///:memory:")
    create_database_and_tables(engine)

    with Session(engine) as session:
        exp = Experiment(name="full_exp")
        session.add(exp)
        session.flush()

        settings = AnalysisSettings(frame_rate=10.0, enable_rising_edge_analysis=True)
        session.add(settings)
        session.flush()

        run = CaliResult(experiment=exp.id, analysis_settings_id=settings.id)
        session.add(run)
        session.flush()
        run_id: int = run.id  # type: ignore[assignment]

        plate = Plate(experiment=exp, name="P1", plate_type="6-well")
        session.add(plate)
        session.flush()

        rng = np.random.default_rng(42)

        for cond_name, row_idx in [("WT", 0), ("KO", 1)]:
            cond = Condition(name=cond_name, condition_type="genotype")
            for fov_idx in range(2):
                well = Well(
                    plate=plate,
                    name=f"{cond_name}_W{fov_idx}",
                    row=row_idx,
                    column=fov_idx,
                    conditions=[cond],
                )
                session.add(well)
                session.flush()

                fov = FOV(
                    name=f"fov_{cond_name.lower()}_{fov_idx}",
                    position_index=fov_idx,
                    well_id=well.id,
                )
                session.add(fov)
                session.flush()

                corr = rng.uniform(0.2, 0.8, (3, 3))
                np.fill_diagonal(corr, 1.0)
                corr = ((corr + corr.T) / 2).tolist()
                corr_arr = np.array(corr)
                n = corr_arr.shape[0]
                mask_arr = ~np.eye(n, dtype=bool)
                global_corr = float(np.mean(corr_arr[mask_arr]))

                fa = FOVAnalysis(
                    fov_id=fov.id,
                    analysis_result_id=run.id,
                    active_roi_labels=[1, 2, 3],
                    spike_burst_count=3 + fov_idx,
                    spike_burst_avg_duration=0.5 + 0.1 * fov_idx,
                    spike_burst_avg_interval=2.0 + 0.2 * fov_idx,
                    spike_population_activity=[0.0] * 600,
                    global_spike_jitter_synchrony=0.3 + 0.1 * fov_idx,
                    global_spike_max_lag_correlation=global_corr,
                    spike_max_lag_correlation_matrix=corr,
                    global_calcium_dff_correlation=0.4 + 0.05 * fov_idx,
                    global_calcium_den_dff_correlation=0.5 + 0.05 * fov_idx,
                    global_spike_jitter_synchrony_rising_edges=(0.25 + 0.1 * fov_idx),
                    global_spike_max_lag_correlation_rising_edges=(global_corr * 0.9),
                    fraction_significant_ccg_pairs=0.6 + 0.05 * fov_idx,
                    fraction_significant_ccg_pairs_rising_edges=(0.5 + 0.05 * fov_idx),
                )
                session.add(fa)

                for roi_idx in range(3):
                    roi = ROI(
                        label_value=roi_idx + 1,
                        active=True,
                        fov_id=fov.id,
                        cell_size=float(rng.uniform(50, 200)),
                    )
                    session.add(roi)
                    session.flush()

                    da = DataAnalysis(
                        roi_id=roi.id,
                        analysis_result_id=run.id,
                        peaks_amplitudes_den_dff=rng.uniform(0.5, 3.0, 5).tolist(),
                        den_dff_frequency=float(rng.uniform(0.1, 2.0)),
                        iei=rng.uniform(0.5, 5.0, 4).tolist(),
                        inferred_spikes_frequency=float(rng.uniform(0.1, 1.0)),
                        inferred_spikes_rising_edge_frequency=float(
                            rng.uniform(0.05, 0.5)
                        ),
                    )
                    session.add(da)

        session.commit()

    gc.collect()
    return engine, run_id


@pytest.fixture
def full_db() -> Generator[tuple[Engine, int], None, None]:
    """In-memory DB with 2 conditions x 2 FOVs, burst/sync/corr/PCA data."""
    engine, run_id = _build_full_db()
    yield engine, run_id
    engine.dispose(close=True)
