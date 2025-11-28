"""Tests for the CaliRunner and internal runners.

Optimized for CI speed while maintaining >90% coverage.
Key optimizations:
1. Mock cellpose inference in most tests (cellpose is slow)
2. Only 1-2 tests run real cellpose to cover that code path
3. Use fixtures to share database setup where possible
"""

import gc
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlmodel import Session, create_engine, select

from cali.analysis._analysis_runner import AnalysisRunner
from cali.readers._tensorstore_zarr_reader import TensorstoreZarrReader
from cali.runner import CaliRunner
from cali.sqlmodel import (
    FOV,
    ROI,
    AnalysisSettings,
    CaliResult,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
    Mask,
    Traces,
)
from cali.util import load_data_from_path

THREADS = 1
MODEL = "cpsam"  # cellpose4
# MODEL = "cyto3"  # cellpose3


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
def mock_detection_runner():
    """Fixture that patches DetectionRunner to return mock FOVs quickly."""
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
        print("👺 Using mocked DetectionRunner")
        yield mock


@pytest.fixture(autouse=True)
def cleanup_gc() -> Iterator[None]:
    """Force garbage collection after each test to close DB connections."""
    yield
    gc.collect()


@pytest.fixture
def runner() -> CaliRunner:
    return CaliRunner(commit_batch_size=1)


# =============================================================================
# FAST TESTS (using mocked cellpose)
# =============================================================================


def test_cali_runner_detection_only_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,
) -> None:
    """Test running detection only (mocked cellpose for speed)."""
    runner = CaliRunner(commit_batch_size=1)

    detection_settings = DetectionSettings(
        method="cellpose",
        model_type=MODEL,
        diameter=30.0,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
    )

    # Run detection
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify results in database
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

            result = session.exec(select(CaliResult)).first()
            assert result is not None
            assert result.detection_settings == ds.id
            assert result.extraction_settings is None
            assert result.analysis_settings is None
    finally:
        engine.dispose()


def test_cali_runner_full_pipeline_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,
) -> None:
    """Test running full pipeline with mocked detection."""
    runner = CaliRunner(commit_batch_size=1)

    detection_settings = DetectionSettings(
        method="cellpose",
        model_type=MODEL,
        diameter=30.0,
    )

    extraction_settings = ExtractionSettings(
        neuropil_inner_radius=2,
        neuropil_min_pixels=50,
        dff_window=100,
        threads=THREADS,
    )

    analysis_settings = AnalysisSettings(
        peaks_prominence_multiplier=3.0,
        threads=THREADS,
    )

    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            result = session.exec(select(CaliResult)).first()
            assert result is not None
            assert result.detection_settings is not None
            assert result.extraction_settings is not None
            assert result.analysis_settings is not None
    finally:
        engine.dispose()


# @pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_cali_runner_incremental_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,
) -> None:
    """Test incremental running (detection first, then extraction) with mocked detection."""
    runner = CaliRunner(commit_batch_size=1)

    detection_settings = DetectionSettings(
        method="cellpose",
        model_type=MODEL,
        diameter=30.0,
    )

    # 1. Run Detection Only
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
            ds_id = ds.id
            assert ds_id is not None
    finally:
        engine.dispose()

    # 2. Run Extraction Only (using existing detection)
    extraction_settings = ExtractionSettings(
        neuropil_inner_radius=2,
        neuropil_min_pixels=50,
        threads=THREADS,
    )

    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_id,
        extraction_settings=extraction_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            result = session.exec(
                select(CaliResult).where(CaliResult.extraction_settings.is_not(None))  # type: ignore
            ).first()
            assert result is not None
            assert result.detection_settings == ds_id
            assert result.extraction_settings is not None
    finally:
        engine.dispose()


# =============================================================================
# SLOW TEST (runs real cellpose for coverage)
# =============================================================================


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


# =============================================================================
# UNIT TESTS (no data dependencies)
# =============================================================================


def test_analysis_runner_direct() -> None:
    """Test AnalysisRunner directly with mocked data."""
    runner = AnalysisRunner()

    analysis_settings = AnalysisSettings(
        peaks_prominence_multiplier=3.0,
        threads=THREADS,
    )

    # Create dummy FOV with ROI and Traces
    fov = FOV(position_index=0, name="Pos0")
    roi = ROI(
        id=1,
        roi_mask_id=1,
        detection_settings_id=1,
        label_value=1,
        fov_id=1,
    )

    # Create dummy traces with peaks
    trace_data = np.zeros(100)
    trace_data[20] = 10  # Peak
    trace_data[50] = 10  # Peak

    traces = Traces(
        raw_trace=trace_data.tolist(),
        neuropil_trace=np.zeros(100).tolist(),
        dff=trace_data.tolist(),
        dec_dff=trace_data.tolist(),
        inferred_spikes=np.zeros(100).tolist(),
        analysis_result_id=1,
        x_axis=np.arange(100, dtype=float).tolist(),
        x_axis_units="frames",
    )

    roi.traces_history = [traces]
    fov.rois = [roi]

    results = runner.run([fov], analysis_settings, as_generator=False)
    assert isinstance(results, list)
    assert len(results) == 1

    res_roi = results[0].rois[0]
    assert hasattr(res_roi, "_new_data_analysis")
    assert len(res_roi._new_data_analysis) > 0
    da = res_roi._new_data_analysis[0]  # type: ignore
    assert da.peaks_dec_dff is not None
    assert len(da.peaks_dec_dff) > 0


def test_cali_runner_cancel() -> None:
    """Test cancellation of CaliRunner."""
    runner = CaliRunner()
    runner.cancel()
    assert runner._detection_runner._cancellation_event.is_set()
    assert runner._extraction_runner._cancellation_event.is_set()


def test_detection_runner_error(data_path: Path) -> None:
    """Test DetectionRunner with invalid method."""
    from cali.detection._detection_runner import DetectionRunner
    from cali.readers import TensorstoreZarrReader

    runner = DetectionRunner()
    settings = DetectionSettings(method="invalid_method")
    dataset = TensorstoreZarrReader(data_path)

    with pytest.raises(ValueError, match="Unknown detection method"):
        list(runner.run(dataset, settings, [0]))


# @pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_cali_runner_overwrite_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,
) -> None:
    """Test overwriting existing database with mocked detection."""
    runner = CaliRunner(commit_batch_size=1)
    detection_settings = DetectionSettings(
        method="cellpose",
        model_type=MODEL,
        diameter=30.0,
    )

    # First run
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Second run with overwrite=True
    detection_settings_2 = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings_2,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
        overwrite=True,
    )

    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            fovs = session.exec(select(FOV)).all()
            assert len(fovs) > 0
    finally:
        engine.dispose()


def test_cali_runner_validation_error_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,
) -> None:
    """Test validation error when experiment mismatch."""
    runner = CaliRunner(commit_batch_size=1)
    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    diff_experiment = Experiment.create_from_data(
        name="Different Experiment",
        data_path=str(data_path),
    )

    with pytest.raises(ValueError, match="does not match the one in the database"):
        runner.run(
            experiment=diff_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
            overwrite=False,
        )


def test_cali_runner_skipping_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,
) -> None:
    """Test skipping detection if already exists."""
    runner = CaliRunner(commit_batch_size=1)
    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    # First run
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Second run - should skip detection
    detection_settings_2 = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings_2,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )


def test_cali_runner_upgrading_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,
) -> None:
    """Test upgrading result from detection-only to full analysis."""
    runner = CaliRunner(commit_batch_size=1)
    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    # 1. Detection Only
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
            ds_id = ds.id
    finally:
        engine.dispose()

    # 2. Extraction Only (Upgrade)
    extraction_settings = ExtractionSettings(
        neuropil_inner_radius=2, neuropil_min_pixels=50, threads=THREADS
    )
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_id,
        extraction_settings=extraction_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # 3. Analysis (Upgrade)
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            es = session.exec(select(ExtractionSettings)).first()
            assert es is not None
            es_id = es.id
    finally:
        engine.dispose()

    analysis_settings = AnalysisSettings(
        peaks_prominence_multiplier=3.0, threads=THREADS
    )
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_id,
        extraction_settings=es_id,
        analysis_settings=analysis_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )


def test_analysis_runner_error() -> None:
    """Test error handling in AnalysisRunner."""
    # Mock ROI and Traces
    roi = ROI(
        id=1,
        label_value=1,
        fov_id=1,
    )
    traces = Traces(
        id=1,
        roi_id=1,
        raw_trace=[1, 2, 3],
        neuropil_trace=[0, 0, 0],
        dff=[0.1, 0.2, 0.3],
        dec_dff=[0.01, 0.02, 0.03],
        x_axis=[0, 1, 2],
        x_axis_units="s",
    )
    roi.traces_history = [traces]

    # Mock FOV
    fov = FOV(
        id=1,
        name="FOV_01",
        position_index=0,
        rois=[roi],
    )

    runner = AnalysisRunner()

    # Mock _analyze_roi_traces to raise exception
    from unittest.mock import patch

    with patch.object(
        runner, "_analyze_roi_traces", side_effect=ValueError("Test Error")
    ):
        # Should not raise, but log error
        runner.run([fov], analysis_settings=AnalysisSettings(threads=THREADS))

    # Verify no analysis added
    assert not hasattr(roi, "_new_data_analysis") or len(roi._new_data_analysis) == 0


# @pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_cali_runner_batching_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test batching logic in CaliRunner with mocked detection."""
    runner = CaliRunner(commit_batch_size=2)
    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],  # Only 1 pos available in test data
    )

    # Verify results
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            assert len(results) == 1
    finally:
        engine.dispose()


def test_cali_runner_commit_error_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test error handling during batch commit with mocked detection."""
    runner = CaliRunner(commit_batch_size=1)
    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    with patch(
        "cali.runner._cali_runner.commit_fov_result",
        side_effect=Exception("Commit Error"),
    ):
        try:
            runner.run(
                experiment=test_experiment,
                dataset_path=data_path,
                detection_settings=detection_settings,
                database_name=test_db_path.name,
                output_path=test_db_path.parent,
                global_position_indices=[0],
            )
        except Exception:
            pass


def test_cali_runner_process_batch_error_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test error handling in _run_detection with mocked detection."""
    runner = CaliRunner(commit_batch_size=1)
    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    # Patch _run_detection to raise exception
    with patch.object(
        runner, "_run_detection", side_effect=Exception("Detection Error")
    ):
        try:
            runner.run(
                experiment=test_experiment,
                dataset_path=data_path,
                detection_settings=detection_settings,
                database_name=test_db_path.name,
                output_path=test_db_path.parent,
                global_position_indices=[0],
            )
        except Exception:
            pass


def test_cali_runner_settings_reuse_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test reusing existing settings in CaliRunner with mocked detection."""
    runner = CaliRunner(commit_batch_size=1)
    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    # First run
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Get settings ID
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            ds = session.exec(select(DetectionSettings)).first()
            assert ds is not None
    finally:
        engine.dispose()

    # Second run with same settings object (should reuse)
    # We need to pass a new object with same values to trigger lookup by value
    detection_settings_2 = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings_2,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify only 1 settings object exists
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            ds_count = len(session.exec(select(DetectionSettings)).all())
            assert ds_count == 1
    finally:
        engine.dispose()


def test_extraction_runner_error() -> None:
    """Test error handling in ExtractionRunner."""
    from cali.extraction._extraction_runner import ExtractionRunner
    from cali.readers import TensorstoreZarrReader

    runner = ExtractionRunner()

    # Mock FOV
    roi = ROI(id=1, label_value=1, fov_id=1)
    fov = FOV(id=1, name="FOV_01", position_index=0, rois=[roi])

    # Mock dataset
    from unittest.mock import MagicMock

    dataset = MagicMock(spec=TensorstoreZarrReader)

    # Mock _analyze_position to raise exception
    with patch.object(
        runner, "_analyze_position", side_effect=Exception("Extraction Error")
    ):
        # Should log error
        list(runner.run(dataset, ExtractionSettings(threads=THREADS), [fov]))


def test_cali_runner_stimulation_mask_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    tmp_path: Path,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test loading stimulation mask from file with mocked detection."""
    import tifffile

    # Create dummy mask file
    mask_path = tmp_path / "stim_mask.tif"
    mask_data = np.zeros((256, 256), dtype=np.uint8)
    mask_data[10:20, 10:20] = 1
    tifffile.imwrite(mask_path, mask_data)

    runner = CaliRunner(commit_batch_size=1)

    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    extraction_settings = ExtractionSettings(
        neuropil_inner_radius=2, neuropil_min_pixels=50, threads=THREADS
    )

    analysis_settings = AnalysisSettings(
        peaks_height_value=1.0,
        peaks_height_mode="std",
        peaks_distance=10,
        stimulation_mask_path=str(mask_path),
        threads=THREADS,
    )

    # Run full pipeline
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify mask was loaded
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            ans = session.exec(select(AnalysisSettings)).first()
            assert ans is not None
            assert ans.stimulation_mask_id is not None

            from cali.sqlmodel._model import Mask

            mask = session.get(Mask, ans.stimulation_mask_id)
            assert mask is not None
            assert mask.mask_type == "stimulation"

    finally:
        engine.dispose()


def test_detection_runner_2d_data(
    test_db_path: Path, test_experiment: Experiment
) -> None:
    """Test detection on 2D data (no time dimension)."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    from cali.readers import TensorstoreZarrReader

    runner = CaliRunner(commit_batch_size=1)

    # Mock dataset with 2D data
    dataset = MagicMock(spec=TensorstoreZarrReader)
    dataset.sequence = MagicMock()
    dataset.sequence.stage_positions = [0]

    # Return 2D array (y, x)
    dataset.isel.return_value = (np.zeros((100, 100)), {})

    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    # Patch CellposeModel to avoid loading real model
    # We patch where it is defined/imported
    with patch("cellpose.models.CellposeModel") as MockModel:
        mock_instance = MockModel.return_value
        # Mock eval to return masks (list of arrays), flows, styles
        # masks should be same shape as input image
        mock_instance.eval.return_value = (
            [np.zeros((100, 100), dtype=np.uint16)],
            None,
            None,
        )

        # Patch load_data_from_path to return our mock dataset
        with patch(
            "cali.runner._cali_runner.load_data_from_path", return_value=dataset
        ):
            # Run detection
            # We expect it to run without error, even if it finds nothing
            runner.run(
                experiment=test_experiment,
                dataset_path="dummy_path",
                detection_settings=detection_settings,
                database_name=test_db_path.name,
                output_path=test_db_path.parent,
                global_position_indices=[0],
            )


def test_detection_runner_debug(
    test_db_path: Path, test_experiment: Experiment, data_path: Path, monkeypatch
) -> None:
    """Test detection with debug logging enabled."""
    monkeypatch.setenv("CELLPOSE_DEBUG", "1")

    runner = CaliRunner(commit_batch_size=1)

    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    # Run detection
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )


def test_extraction_runner_cancel_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test cancellation during extraction with mocked detection."""
    import threading
    import time
    from typing import Any

    runner = CaliRunner(commit_batch_size=1)

    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    extraction_settings = ExtractionSettings(
        neuropil_inner_radius=2, neuropil_min_pixels=50, threads=THREADS
    )

    # Start run in a thread
    def run_task() -> None:
        original_analyze = runner._extraction_runner._analyze_position

        def slow_analyze(*args: Any, **kwargs: Any) -> Any:
            time.sleep(1.0)
            return original_analyze(*args, **kwargs)

        with patch.object(
            runner._extraction_runner, "_analyze_position", side_effect=slow_analyze
        ):
            runner.run(
                experiment=test_experiment,
                dataset_path=data_path,
                detection_settings=detection_settings,
                extraction_settings=extraction_settings,
                database_name=test_db_path.name,
                output_path=test_db_path.parent,
                global_position_indices=[0],
            )

    t = threading.Thread(target=run_task)
    t.start()

    # Wait a bit for it to start and enter the loop
    time.sleep(0.2)

    # Cancel
    runner.cancel()

    t.join()

    # Verify cancellation logged


def test_cali_runner_settings_errors_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test error handling for settings lookup with mocked detection."""
    runner = CaliRunner(commit_batch_size=1)

    # 1. Non-existent DetectionSettings ID
    with pytest.raises(ValueError, match="DetectionSettings with ID 999 not found"):
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=999,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )

    # 2. DetectionSettings object with non-existent ID
    ds = DetectionSettings(method="cellpose", model_type=MODEL, id=999)
    # This should create it, not raise error (based on code reading)
    # Wait, lines 855-871 handle this case:
    # existing = session.get(DetectionSettings, detection_settings.id)
    # if existing is not None: ... return existing
    # else: session.add(detection_settings) ...

    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify it was created
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            saved_ds = session.get(DetectionSettings, 999)
            assert saved_ds is not None
    finally:
        engine.dispose()

    # 3. Non-existent ExtractionSettings ID
    ds_valid = DetectionSettings(method="cellpose")
    with pytest.raises(ValueError, match="ExtractionSettings ID 999 not found"):
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=ds_valid,
            extraction_settings=999,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )


def test_analysis_runner_no_traces() -> None:
    """Test analysis runner with ROI that has no traces."""
    runner = AnalysisRunner()

    # Mock FOV with ROI but no traces
    roi = ROI(id=1, label_value=1, fov_id=1)
    roi.traces_history = []
    fov = FOV(id=1, name="FOV_01", position_index=0, rois=[roi])

    analysis_settings = AnalysisSettings(threads=THREADS)

    # Should log warning and continue
    results = list(runner.run([fov], analysis_settings))
    assert len(results) == 1
    assert results[0] == fov
    # No data analysis added
    assert len(roi.data_analysis_history) == 0


def test_analysis_runner_cancel() -> None:
    """Test cancellation during analysis."""
    import threading
    import time

    import numpy as np

    runner = AnalysisRunner()

    # Mock FOV
    roi = ROI(id=1, label_value=1, fov_id=1)
    # Mock traces
    traces = Traces(
        id=1,
        roi_id=1,
        raw_trace=np.zeros(100).tolist(),
        neuropil_trace=np.zeros(100).tolist(),
        dec_dff=np.zeros(100).tolist(),
        dff=np.zeros(100).tolist(),
    )
    roi.traces_history = [traces]
    fov = FOV(id=1, name="FOV_01", position_index=0, rois=[roi])

    analysis_settings = AnalysisSettings(threads=THREADS)

    # Patch _analyze_roi_traces to be slow
    original_analyze = runner._analyze_roi_traces

    def slow_analyze(*args, **kwargs):
        time.sleep(1.0)
        return original_analyze(*args, **kwargs)

    with patch.object(runner, "_analyze_roi_traces", side_effect=slow_analyze):
        # Start run in a thread
        def run_task() -> None:
            list(runner.run([fov], analysis_settings))

        t = threading.Thread(target=run_task)
        t.start()

        # Wait a bit for it to start
        time.sleep(0.2)

        # Cancel
        runner.cancel()

        t.join()


def test_extraction_runner_cancel_before_start_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test cancellation before extraction starts with mocked detection."""
    runner = CaliRunner(commit_batch_size=1)

    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    extraction_settings = ExtractionSettings(
        neuropil_inner_radius=2,
        neuropil_min_pixels=50,
    )

    # Set cancel event
    runner.cancel()

    # Patch clear to do nothing so the event stays set
    with patch.object(runner._extraction_runner._cancellation_event, "clear"):
        # Run should return immediately (or handle it gracefully)
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )


def test_cali_runner_update_result_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test updating an existing analysis result with mocked detection."""
    runner = CaliRunner(commit_batch_size=1)

    detection_settings = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0
    )

    # 1. Run on position 0
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=detection_settings,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Get the ID from DB
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            ds = session.exec(select(DetectionSettings)).first()
            assert ds is not None
            assert ds.id is not None
            ds_id = ds.id
    finally:
        engine.dispose()

    # 2. Run on position 0 again (should update existing result)
    # We pass the ID to avoid DetachedInstanceError with the previous object
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_id,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify only one result exists
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            assert len(results) == 1
            assert results[0].positions_analyzed == [0]
    finally:
        engine.dispose()


def test_settings_deduplication_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    runner: CaliRunner,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test that identical settings are reused with mocked detection."""
    ds1 = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)
    ds2 = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)

    # Run 1
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds1,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Run 2 with identical but new object
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds2,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify only one DetectionSettings exists
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            settings = session.exec(select(DetectionSettings)).all()
            assert len(settings) == 1
    finally:
        engine.dispose()


def test_settings_by_id_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    runner: CaliRunner,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test passing settings by ID with mocked detection."""
    from sqlmodel import SQLModel

    ds = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)
    es = ExtractionSettings(neuropil_inner_radius=10)
    as_ = AnalysisSettings(peaks_height_value=10.0)

    # Pre-populate DB
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        SQLModel.metadata.create_all(engine)  # Create tables first!

        with Session(engine) as session:
            # We must add the experiment too, otherwise runner complains
            session.add(test_experiment)
            session.add(ds)
            session.add(es)
            session.add(as_)
            session.commit()
            session.refresh(test_experiment)
            session.refresh(ds)
            session.refresh(es)
            session.refresh(as_)
            ds_id = ds.id
            es_id = es.id
            as_id = as_.id
            assert ds_id is not None
    finally:
        engine.dispose()

    # Run using IDs
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_id,
        extraction_settings=es_id,
        analysis_settings=as_id,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify result created
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            assert len(results) == 1
            assert results[0].detection_settings == ds_id
            assert results[0].extraction_settings == es_id
            assert results[0].analysis_settings == as_id
    finally:
        engine.dispose()


def test_settings_object_with_id_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    runner: CaliRunner,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test passing settings object that already has an ID with mocked detection."""
    ds = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)

    # Run 1 to create settings in DB
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Get the ID from DB
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            ds_db = session.exec(select(DetectionSettings)).first()
            assert ds_db is not None
            ds_id = ds_db.id
            assert ds_id is not None
    finally:
        engine.dispose()

    # Create a NEW object but with the SAME ID (simulating loading from elsewhere)
    ds_with_id = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0, id=ds_id
    )

    # Run 2 with object having ID
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_with_id,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify still only one settings
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            settings = session.exec(select(DetectionSettings)).all()
            assert len(settings) == 1
    finally:
        engine.dispose()


def test_result_upgrade_flow_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    runner: CaliRunner,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test upgrading result from detection -> extraction -> analysis."""
    ds = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)
    es = ExtractionSettings(neuropil_inner_radius=10)
    as_ = AnalysisSettings(peaks_height_value=10.0)

    # 1. Detection only
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            assert len(results) == 1
            assert results[0].extraction_settings is None
            ds_id = results[0].detection_settings
            assert ds_id is not None
    finally:
        engine.dispose()

    # 2. Upgrade to Extraction
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_id,
        extraction_settings=es,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            assert len(results) == 1
            assert results[0].extraction_settings is not None
            assert results[0].analysis_settings is None
            es_id = results[0].extraction_settings
            assert es_id is not None
    finally:
        engine.dispose()

    # 3. Upgrade to Analysis
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_id,
        extraction_settings=es_id,
        analysis_settings=as_,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            assert len(results) == 1
            assert results[0].analysis_settings is not None
    finally:
        engine.dispose()


def test_load_fovs_filtering_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    runner: CaliRunner,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test that _load_fovs_from_db filters by detection settings."""
    ds1 = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)
    ds2 = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=40.0
    )  # Different

    # Run detection with DS1 on pos 0
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds1,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Run detection with DS2 on pos 0
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds2,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
        overwrite=False,
    )

    # Now we want to run extraction ONLY for DS1
    es = ExtractionSettings(neuropil_inner_radius=10)

    # We need to get the ID of DS1
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            ds1_db = session.exec(
                select(DetectionSettings).where(DetectionSettings.diameter == 30.0)
            ).first()
            assert ds1_db is not None
            ds1_id = ds1_db.id
            assert ds1_id is not None

            # Verify we have ROIs for both settings
            rois = session.exec(select(ROI)).all()
            assert len(rois) > 0
            ds_ids = {r.detection_settings_id for r in rois}
            assert len(ds_ids) == 2
    finally:
        engine.dispose()

    # Run extraction for DS1
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds1_id,
        extraction_settings=es,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Verify that traces were only created for ROIs belonging to DS1
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            # Get all ROIs with traces
            # We need to check if ROIs from DS2 have traces (they shouldn't)

            # Get DS2 ID
            ds2_db = session.exec(
                select(DetectionSettings).where(DetectionSettings.diameter == 40.0)
            ).first()
            assert ds2_db is not None
            ds2_id = ds2_db.id
            assert ds2_id is not None

            rois_ds2 = session.exec(
                select(ROI).where(ROI.detection_settings_id == ds2_id)
            ).all()
            for roi in rois_ds2:
                assert len(roi.traces_history) == 0

            rois_ds1 = session.exec(
                select(ROI).where(ROI.detection_settings_id == ds1_id)
            ).all()
            for roi in rois_ds1:
                assert len(roi.traces_history) > 0
    finally:
        engine.dispose()


def test_run_as_generator_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    runner: CaliRunner,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test running as a generator with mocked detection."""
    ds = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)

    # Run as generator
    gen = runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
        as_generator=True,
    )

    assert gen is not None

    # Consume generator
    messages = list(gen)
    assert len(messages) > 0
    assert any("Running Detection" in m for m in messages)

    # Verify result created
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            assert len(results) == 1
    finally:
        engine.dispose()


def test_extraction_analysis_settings_with_id_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    runner: CaliRunner,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test passing extraction/analysis settings objects with IDs."""
    ds = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)
    es = ExtractionSettings(neuropil_inner_radius=10)
    as_ = AnalysisSettings(peaks_height_value=10.0)

    # Run 1 to create settings in DB
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds,
        extraction_settings=es,
        analysis_settings=as_,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
    )

    # Get IDs
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            ds_db = session.exec(select(DetectionSettings)).first()
            es_db = session.exec(select(ExtractionSettings)).first()
            as_db = session.exec(select(AnalysisSettings)).first()
            assert ds_db is not None
            assert es_db is not None
            assert as_db is not None
            ds_id = ds_db.id
            es_id = es_db.id
            as_id = as_db.id
    finally:
        engine.dispose()

    # Create new objects with same IDs
    ds_new = DetectionSettings(
        method="cellpose", model_type=MODEL, diameter=30.0, id=ds_id
    )
    es_new = ExtractionSettings(neuropil_inner_radius=10, id=es_id)
    as_new = AnalysisSettings(peaks_height_value=10.0, id=as_id)

    # Run 2
    runner.run(
        experiment=test_experiment,
        dataset_path=data_path,
        detection_settings=ds_new,
        extraction_settings=es_new,
        analysis_settings=as_new,
        database_name=test_db_path.name,
        output_path=test_db_path.parent,
        global_position_indices=[0],
        overwrite=False,
    )

    # Verify no duplicates
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            assert len(session.exec(select(DetectionSettings)).all()) == 1
            assert len(session.exec(select(ExtractionSettings)).all()) == 1
            assert len(session.exec(select(AnalysisSettings)).all()) == 1
    finally:
        engine.dispose()


def test_result_update_existing_mocked(
    test_db_path: Path,
    test_experiment: Experiment,
    data_path: Path,
    runner: CaliRunner,
    mock_detection_runner,  # type: ignore[no-untyped-def]
) -> None:
    """Test updating an existing result (exact match) with mocked detection."""
    from typing import Any

    # We need to mock the dataset to have 2 positions
    real_dataset = load_data_from_path(data_path)

    with patch("cali.runner._cali_runner.load_data_from_path") as mock_load:
        mock_dataset = MagicMock(spec=TensorstoreZarrReader)
        mock_dataset.sequence.stage_positions = [0, 1]

        def side_effect(p: int = 0, metadata: bool = False, **kwargs: Any) -> Any:
            return real_dataset.isel(p=0, metadata=metadata)  # type: ignore

        mock_dataset.isel.side_effect = side_effect
        mock_load.return_value = mock_dataset

        ds = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)

        # Run 1: Position 0
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=ds,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[0],
        )

        # Run 2: Position 1 (should update existing result to include pos 1)
        ds2 = DetectionSettings(method="cellpose", model_type=MODEL, diameter=30.0)
        runner.run(
            experiment=test_experiment,
            dataset_path=data_path,
            detection_settings=ds2,
            database_name=test_db_path.name,
            output_path=test_db_path.parent,
            global_position_indices=[1],
        )

    # Verify only one result exists with both positions
    engine = create_engine(f"sqlite:///{test_db_path}")
    try:
        with Session(engine) as session:
            results = session.exec(select(CaliResult)).all()
            assert len(results) == 1
            positions = results[0].positions_analyzed
            assert positions is not None
            assert sorted(positions) == [0, 1]
    finally:
        engine.dispose()
