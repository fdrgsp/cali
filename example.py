from datetime import datetime
from pathlib import Path

from sqlmodel import create_engine

from cali.analysis import AnalysisRunner
from cali.detection import CellposeSettings, DetectionRunner
from cali.sqlmodel import AnalysisSettings, Experiment, data_to_plate
from cali.sqlmodel._visualize_experiment import (
    print_all_analysis_results,
)

exp = Experiment(
    id=0,
    name="New Experiment",
    description="A Test Experiment.",
    created_at=datetime.now(),
    database_name="cali_cp.db",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    analysis_path="analysis_results",
)

# Create plate with plate_maps and conditions from the data useq metadata
plate = data_to_plate(
    exp.data_path,
    exp,
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)
if plate is None:
    raise ValueError("Failed to create plate from data.")
exp.plate = plate

# delete existing database if it exists
if Path(exp.db_path).exists():
    Path(exp.db_path).unlink()

# DETECTION STEP
detection = DetectionRunner()
cp_settings = CellposeSettings()
detection.run_cellpose(exp, "cpsam", cp_settings, global_position_indices=[0])

# ANALYSIS STEP
analysis = AnalysisRunner()

# Run 1: Conservative dff_window (100 frames) - New
settings1 = AnalysisSettings(threads=4, dff_window=100)
analysis.run(exp, settings1, global_position_indices=[0])

# Run 2: Wider dff_window (150 frames) - New
settings2 = AnalysisSettings(threads=4, dff_window=150, id=54)
analysis.run(exp, settings2, global_position_indices=[0])

# Run 3: Back to 100 frames - should REUSE settings1 and UPDATE AnalysisResult #1
settings3 = AnalysisSettings(threads=4, dff_window=100)
analysis.run(exp, settings3, global_position_indices=[0])
# Run 4: Even wider dff_window (200 frames) - New
settings4 = AnalysisSettings(threads=4, dff_window=200)
analysis.run(exp, settings4, global_position_indices=[0])

# Visualize the complete experiment tree with analysis results
engine = create_engine(f"sqlite:///{exp.db_path}")
print_all_analysis_results(
    engine,
    experiment_name=None,
    show_settings=True,
    max_experiment_level="roi",
)
