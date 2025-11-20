from pathlib import Path

from sqlmodel import create_engine

from cali._constants import EVOKED
from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner
from cali.sqlmodel import AnalysisSettings, Experiment
from cali.sqlmodel._model import DetectionSettings
from cali.sqlmodel._visualize_experiment import (
    print_all_analysis_results,
)

exp = Experiment.create_from_data(
    name="New Experiment",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    analysis_path="/Users/fdrgsp/Desktop/cali_test",
    database_name="evk.tensorstore.zarr.db",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
    experiment_type=EVOKED,
)

# delete existing database if it exists
if Path(exp.db_path).exists():
    Path(exp.db_path).unlink()

# DETECTION STEP
detection = DetectionRunner()
dsettings1 = DetectionSettings(method="cellpose", model_type="cpsam")
detection.run(exp, dsettings1, global_position_indices=[0])

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
