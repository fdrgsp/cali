from datetime import datetime
from pathlib import Path

from sqlmodel import create_engine

from cali.analysis import AnalysisRunner
from cali.detection import DetectionRunner
from cali.readers import TensorstoreZarrReader
from cali.sqlmodel import AnalysisSettings, Experiment, useq_plate_plan_to_db
from cali.sqlmodel._model import DetectionSettings
from cali.sqlmodel._visualize_experiment import print_all_analysis_results

# ###########################################

exp = Experiment(
    id=0,
    name="New Experiment",
    description="A Test Experiment.",
    created_at=datetime.now(),
    database_name="cali_new.db",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    analysis_path="analysis_results",
)
exp1 = Experiment(
    id=1,
    name="New Experiment",
    description="A Test Experiment.",
    created_at=datetime.now(),
    database_name="cali_new.db",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    analysis_path="analysis_results",
)

# ----------------------------------- FIXME: create isolated function for this
# load the data and get the useq plate plan from the sequence
# Create plate with plate_maps and conditions in one step
data = TensorstoreZarrReader(exp.data_path)
exp.plate = useq_plate_plan_to_db(
    (plate_plan := data.sequence.stage_positions),
    exp,
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)
exp1.plate = useq_plate_plan_to_db(
    (plate_plan := data.sequence.stage_positions),
    exp,
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)
# -----------------------------------

# delete existing database if it exists
if Path(exp.db_path).exists():
    Path(exp.db_path).unlink()

# DETECTION STEP
detection = DetectionRunner()
dsettings1 = DetectionSettings(method="cellpose", model_type="cpsam")
detection.run(exp, dsettings1, global_position_indices=[0])

# ANALYSIS STEP
analysis = AnalysisRunner()

# Run 1 - new
settings = AnalysisSettings(threads=4, dff_window=100)
analysis.run(exp, settings, global_position_indices=list(range(len(plate_plan))))

# Run 2 - new
settings = AnalysisSettings(threads=4, dff_window=100)
analysis.run(exp1, settings, global_position_indices=list(range(len(plate_plan))))

# Visualize the complete experiment tree with analysis results
engine = create_engine(f"sqlite:///{exp.db_path}")
print_all_analysis_results(
    engine,
    experiment_name=None,  # or experiment name string
    show_settings=False,  # show detailed settings
    max_experiment_level="roi",  # show down to FOV level
)
