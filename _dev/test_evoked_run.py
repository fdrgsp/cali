"""Test running evoked experiment with stimulation mask to verify the fix."""

from pathlib import Path

from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)


def test_evoked_run() -> None:
    """Test evoked experiment run with stimulation mask."""
    # Use the evoked test data
    data_path = Path("tests/test_data/evoked/evk.tensorstore.zarr")
    db_path = Path("tests/test_data/evoked/test_evoked_run.cali")
    mask_path = Path("tests/test_data/evoked/mask.tif")

    # Clean up any existing test database
    if db_path.exists():
        db_path.unlink()

    # Create experiment
    experiment = Experiment.create_from_data(
        name="Test Evoked",
        data_path=str(data_path),
        description="Test evoked experiment",
    )

    # Create runner
    runner = CaliRunner()

    # Create settings
    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings(threads=5)
    analysis_settings = AnalysisSettings(
        experiment_type="Evoked Activity",
        stimulation_mask_path=str(mask_path),
        led_pulse_duration=0.02,
        led_pulse_powers=[100.0],
        led_pulse_on_frames=[10],
        threads=5,
    )

    # Run full pipeline
    print("Running evoked experiment with stimulation mask...")
    runner.run(
        experiment=experiment,
        dataset_path=str(data_path),
        detection_settings=detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        global_position_indices=[0],
        database_name=db_path.name,
        output_path=db_path.parent,
    )

    print("✅ Test completed successfully!")

    # Clean up
    if db_path.exists():
        db_path.unlink()


if __name__ == "__main__":
    test_evoked_run()
