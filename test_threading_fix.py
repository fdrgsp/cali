"""Test script to verify numba threading fix."""

from pathlib import Path

from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)

# Use the test data
data_path = "tests/test_data/test_for_plot/evk.tensorstore.zarr"
db_path = Path("/tmp/test_threading.cali")

# Clean up if exists
if db_path.exists():
    db_path.unlink()

print("Testing with 10 threads to test crashing...")

exp = Experiment.create_from_data("test_exp", data_path)
detection_settings = DetectionSettings(
    method="cellpose",
    model_type="cpsam",
)
extraction_settings = ExtractionSettings(threads=10)  # Use 10 threads
analysis_settings = AnalysisSettings(experiment_type="evoked")

runner = CaliRunner()

try:
    runner.run(
        exp,
        data_path,
        detection_settings,
        extraction_settings=extraction_settings,
        analysis_settings=analysis_settings,
        output_path=db_path.parent,
        database_name=db_path.name,
        overwrite=True,
    )
    print("\n✅ SUCCESS! No threading crashes occurred.")
    print(f"Database created at: {db_path}")
except Exception as e:
    print(f"\n❌ FAILED with error: {e}")
    raise
finally:
    # Clean up
    if db_path.exists():
        db_path.unlink()
