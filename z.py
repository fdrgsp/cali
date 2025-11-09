import time
from datetime import datetime

import useq
from rich import print

from cali._plate_viewer._analysis_new import AnalysisRunner
from cali.readers import TensorstoreZarrReader
from cali.sqlmodel._models import Experiment
from cali.sqlmodel._useq_plate_to_db import useq_plate_plan_to_db
from cali.sqlmodel._visualize_experiment import print_experiment_tree

# database_path = (
#     "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis/cali.db"
# )
# exp = load_experiment_from_db(database_path)
# assert exp is not None
# print_experiment_tree(exp)

# runner = AnalysisRunner()
# runner.set_experiment(exp)
# data = TensorstoreZarrReader(exp.data_path)
# runner.set_data(data)
# print(runner.get_settings())
# runner.run()

# # Wait for the worker to finish
# if runner._worker is not None:
#     print("Waiting for analysis to complete...")
#     while runner._worker.is_running:
#         time.sleep(0.1)
#     print("Analysis complete!")
#     print_experiment_tree(exp)
# else:
#     print("No worker was created")


database_path = (
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis/cali_new.db"
)
data_path = (
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
)
analysis_path = "/Users/fdrgsp/Desktop/cali_test"
labels_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_labels"


new_exp = Experiment(
    id=0,
    name="New Experiment",
    description="A Test Experiment.",
    created_at=datetime.now(),
    database_path=database_path,
    data_path=data_path,
    labels_path=labels_path,
    analysis_path=analysis_path,
)
print(new_exp)

# load the data and get the useq plate plan from the sequence
data = TensorstoreZarrReader(new_exp.data_path)
# Define plate maps for conditions
plate_maps = {"genotype": {"B5": "WT"}, "treatment": {"B5": "Vehicle"}}
plate_plan = data.sequence.stage_positions
assert isinstance(plate_plan, useq.WellPlatePlan)

# Create plate with plate_maps and conditions in one step
plate = useq_plate_plan_to_db(plate_plan, new_exp, plate_maps=plate_maps)
new_exp.plate = plate

print_experiment_tree(new_exp)

# Now when we set the experiment in the runner, conditions will be applied
runner = AnalysisRunner()
runner.set_data(data)
runner.set_experiment(new_exp)



def _p(msg: str) -> None:
    print("ANALYSIS INFO:", msg)
runner.analysisInfo.connect(_p)

runner.run()

while runner._worker is not None and runner._worker.is_running:
    time.sleep(0.1)

print_experiment_tree(new_exp)
