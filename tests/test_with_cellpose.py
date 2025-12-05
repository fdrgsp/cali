import gc
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlmodel import Session, create_engine, select

from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    DetectionSettings,
    Experiment,
)

THREADS = 1
MODEL = "cpsam"  # cellpose4
# MODEL = "cyto3"  # cellpose3


@pytest.fixture(autouse=True)
def cleanup_gc() -> Iterator[None]:
    """Force garbage collection after each test to close DB connections."""
    yield
    gc.collect()


@pytest.fixture
def runner() -> CaliRunner:
    return CaliRunner(commit_batch_size=1)


@pytest.mark.skip()
@pytest.mark.skipif(sys.platform == "win32", reason="Test takes too long on Windows")
def test_cali_runner_real_cellpose(
    test_db_path: Path, test_experiment: Experiment, data_path: Path
) -> None:
    """Test running real cellpose detection (slow, for coverage).

    This test runs the actual cellpose model to ensure coverage of the
    detection code path. It is marked as slow and should be skipped in
    fast CI runs using: pytest -m "not slow"
    """
    runner = CaliRunner(commit_batch_size=1)

    detection_settings = DetectionSettings(
        method="cellpose",
        model_type=MODEL,
        diameter=30.0,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
    )

    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            ds = session.exec(select(DetectionSettings)).first()
            assert ds is not None
            assert ds.method == "cellpose"

            fovs = session.exec(select(FOV)).all()
            assert len(fovs) > 0

            rois = session.exec(select(ROI)).all()
            assert len(rois) > 0
    finally:
        engine.dispose()
