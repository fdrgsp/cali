"""
Example of how to use DetectionRunner and AnalysisRunner individually.

This script demonstrates how to run the pipeline components manually without
using the unified CaliRunner class. This gives you more control over the
execution flow, database interactions, and object lifecycle.
"""

from pathlib import Path

from rich import print
from sqlmodel import Session, create_engine

from cali.analysis._analysis_runner import AnalysisRunner
from cali.detection import DetectionRunner
from cali.sqlmodel import (
    DetectionSettings,
    Experiment,
    save_experiment_to_database,
)
from cali.sqlmodel._model import AnalysisSettings
from cali.util import (
    get_fovs_by_detection_id,
    load_data,
    save_results_to_database,
)
from cali.util._util import commit_fov_result

# 1. Setup paths and data
# -----------------------
# Update these paths to match your data
dataset_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt.tensorstore.zarr"
db_path = Path("manual_run.cali")
# Process just a few positions for this example
positions_to_process = [17, 18]

# Clean up previous run for this example
if db_path.exists():
    db_path.unlink()

# 2. Setup Database and Experiment
# --------------------------------
exp = Experiment.create_from_data("manual_exp", dataset_path)
save_experiment_to_database(exp, db_path.parent, database_name=db_path.name)

engine = create_engine(f"sqlite:///{db_path}")

# 3. Run Detection
# ----------------
print("🔍 Running Detection...")
detection_runner = DetectionRunner()
detection_settings = DetectionSettings(
    method="cellpose",
    model_type="custom",
    custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",
)
# By default, run() returns a list of FOVs (as_generator=False)
fovs_detected = detection_runner.run(
    dataset=dataset_path,
    detection_settings=detection_settings,
    global_position_indices=positions_to_process,
)

print("✅ Saved FOVs to database.")


# 4. Run Analysis
# ---------------
print("📈 Running Analysis...")

# Load FOVs from database
analysis_runner = AnalysisRunner()
analysis_settings = AnalysisSettings(
    dff_window=150, threads=10, peaks_height_value=2
)

# Run analysis
# By default, run() returns a list of FOVs (as_generator=False)
fovs_analyzed = analysis_runner.run(
    dataset=dataset_path,
    settings=analysis_settings,
    fovs=fovs_to_analyze,
)

# Save analysis results
save_results_to_database(
    db_path,
    "manual_exp",
    fovs_analyzed,
    detection_settings=detection_id,
    analysis_settings=analysis_settings,
    positions_processed=positions_to_process,
)

print("🎉 Done!")
