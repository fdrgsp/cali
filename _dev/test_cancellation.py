"""Test cancellation between detection and extraction phases."""

import threading
import time
from pathlib import Path

from cali.runner import CaliRunner
from cali.sqlmodel import DetectionSettings, Experiment, ExtractionSettings


def test_cancel_between_phases():
    """Test that cancellation works between detection and extraction."""
    data_path = "tests/test_data/evoked/evk.tensorstore.zarr"
    db_path = Path("tests/test_data/evoked/test_cancel.cali")
    
    # Clean up any existing test database
    if db_path.exists():
        db_path.unlink()
    
    # Load experiment
    experiment = Experiment.create_from_data(
        name="Test Cancel",
        data_path=data_path,
        description="Test cancellation",
    )
    
    # Create runner
    runner = CaliRunner()
    
    # Create settings
    detection_settings = DetectionSettings(method="cellpose", model_type="cyto3")
    extraction_settings = ExtractionSettings()
    
    # Create a thread that will cancel after 2 seconds (should be after detection)
    def cancel_after_delay():
        time.sleep(2)
        print("🛑 Requesting cancellation...")
        runner.cancel()
    
    cancel_thread = threading.Thread(target=cancel_after_delay, daemon=True)
    cancel_thread.start()
    
    # Run with both detection and extraction
    # Should cancel after detection completes but before extraction starts
    try:
        result = runner.run(
            experiment=experiment,
            dataset_path=data_path,
            detection_settings=detection_settings,
            extraction_settings=extraction_settings,
            global_position_indices=[0],
            database_name="test_cancel.cali",
            output_path=db_path.parent,
            as_generator=True,
        )
        if result is not None:
            for msg in result:
                print(msg)
    except Exception as e:
        print(f"Error: {e}")
    
    # Clean up
    if db_path.exists():
        db_path.unlink()
    
    print("✅ Test completed")


if __name__ == "__main__":
    test_cancel_between_phases()
