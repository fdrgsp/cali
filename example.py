from datetime import datetime

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
# -----------------------------------

settings = AnalysisSettings(threads=4)
# run the analysis
analysis = AnalysisRunner()
analysis.run(exp, settings, global_position_indices=list(range(len(plate_plan))))

loaded_exp = Experiment.load_from_db(exp.db_path, exp.id)
print_experiment_tree(loaded_exp)
