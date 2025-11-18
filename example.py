from pathlib import Path

from sqlmodel import create_engine

from cali._constants import EVOKED
from cali.analysis import AnalysisRunner
from cali.detection import CellposeSettings, DetectionRunner
from cali.sqlmodel import AnalysisSettings, Experiment
from cali.sqlmodel._model import DetectionSettings
from cali.sqlmodel._visualize_experiment import (
    print_all_analysis_results,
    print_database_tree,
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
d_settings = DetectionSettings(method="cellpose", model_type="cpsam")
detection.run_cellpose(exp, d_settings, global_position_indices=[0])

# ANALYSIS STEP
# (detection_settings_id is automatically retrieved from the database)
analysis = AnalysisRunner()
a_settings = AnalysisSettings(threads=4, dff_window=100)
analysis.run(exp, a_settings, global_position_indices=[0])

a_settings1 = AnalysisSettings(threads=4, dff_window=150)
analysis.run(exp, a_settings1, global_position_indices=[0])

# Visualize the complete experiment tree with analysis results
engine = create_engine(f"sqlite:///{exp.db_path}")
print_all_analysis_results(
    engine,
    experiment_name=None,
    show_settings=True,
    max_experiment_level="roi",
)
print_database_tree(engine)
