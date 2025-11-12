from datetime import datetime
from unittest import runner

import useq

# from cali.analysis import AnalysisRunner
from cali.analysis._analysis_runner2 import AnalysisRunner
from cali.readers import TensorstoreZarrReader
from cali.sqlmodel import AnalysisSettings, Experiment, useq_plate_plan_to_db
from cali.sqlmodel._visualize_experiment import print_experiment_tree

# ###########################################

exp = Experiment(
    id=0,
    name="New Experiment",
    description="A Test Experiment.",
    created_at=datetime.now(),
    database_name="cali_new.db",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    labels_path="tests/test_data/evoked/evk_labels",
    analysis_path="analysis_results",
)

# load the data and get the useq plate plan from the sequence
data = TensorstoreZarrReader(exp.data_path)
assert data.sequence is not None
plate_plan = data.sequence.stage_positions
assert isinstance(plate_plan, useq.WellPlatePlan)


# Define plate maps for conditions (optional)
plate_maps = {
    "genotype": {"B5": "WT"},
    "treatment": {"B5": "Vehicle"},
}

# Create plate with plate_maps and conditions in one step
plate = useq_plate_plan_to_db(plate_plan, exp, plate_maps)
exp.plate = plate


settings = AnalysisSettings(threads=4)


# ###########################################

# run the analysis
analysis = AnalysisRunner()
analysis.run(exp, settings, positions=list(range(len(plate_plan))))

print_experiment_tree(analysis._runner.experiment())
