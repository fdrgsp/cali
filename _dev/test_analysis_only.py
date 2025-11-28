"""Test Analysis Only mode with existing detection and extraction settings."""

from pathlib import Path

from cali.runner import CaliRunner
from cali.sqlmodel import AnalysisSettings, Experiment

# Load experiment
db_path = Path("tests/test_data/evoked/results.cali")
exp = Experiment.load_from_db(db_path, load_data=False)

# Create analysis settings
ana_settings = AnalysisSettings(
    led_pulse_duration=0.5,
    led_pulse_powers=[10.0],
    led_pulse_on_frames=[100],
)

# Create runner and run analysis-only mode
# Should use existing DetectionSettings ID 1 and ExtractionSettings ID 1
runner = CaliRunner()

try:
    runner.run(
        experiment=exp,
        dataset_path="tests/test_data/evoked/evk.tensorstore.zarr",
        detection_settings=1,  # Use existing detection settings ID
        extraction_settings=1,  # Use existing extraction settings ID
        analysis_settings=ana_settings,
        global_position_indices=[0],  # Just test first position
        database_name=db_path.name,
        output_path=db_path.parent,
    )
    print("✅ Analysis Only mode succeeded!")
except ValueError as e:
    print(f"❌ Error: {e}")
