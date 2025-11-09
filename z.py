from pathlib import Path

import useq

from cali._plate_viewer._analysis_with_sqlmodel import AnalysisRunner
from cali.readers import TensorstoreZarrReader
from cali.sqlmodel._json_to_db import (
    load_analysis_from_json,
    load_experiment_from_db,
    save_experiment_to_db,
)
from cali.sqlmodel._visualize_experiment import print_experiment_tree

# recreate the experiment from json files
print("Loading experiment from JSON...")
data_path = (
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
)
labels_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_labels"
analysis_dir = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis"
plate = useq.WellPlate.from_str("96-well")
exp_form_json = load_analysis_from_json(
    str(data_path), str(labels_path), str(analysis_dir), plate
)
database_path = Path(analysis_dir) / "cali.db"
exp_form_json.database_path = str(database_path)
print("Saving experiment to database...")
save_experiment_to_db(exp_form_json, database_path, overwrite=True)
print_experiment_tree(exp_form_json)


# load the experiment from the newly created database
print("Loading experiment from database...")
exp = load_experiment_from_db(database_path)
assert exp is not None
print_experiment_tree(exp)

# delete existing rois
print("Deleting existing ROIs...")
for well in exp.plate.wells:
    for fov in well.fovs:
        fov.rois = []

# save database after deletion
print("Saving experiment after ROI deletion...")
save_experiment_to_db(exp, database_path, overwrite=True)
print_experiment_tree(exp)

# run analysis
print("Running analysis to recreate ROIs...")
runner = AnalysisRunner()
runner.set_experiment(exp)
data = TensorstoreZarrReader(exp.data_path)
runner.set_data(data)
runner.run()
print_experiment_tree(exp)

# reopen database to verify the state after analysis
print("Loading experiment from database after analysis...")
exp = load_experiment_from_db(database_path)
assert exp is not None
print_experiment_tree(exp)


# database_path = (
#     "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis/cali_new.db"
# )
# data_path = (
#     "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
# )
# analysis_path = "/Users/fdrgsp/Desktop/cali_test"
# labels_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_labels"


# new_exp = Experiment(
#     id=0,
#     name="New Experiment",
#     description="A Test Experiment.",
#     created_at=datetime.now(),
#     database_path=database_path,
#     data_path=data_path,
#     labels_path=labels_path,
#     analysis_path=analysis_path,
# )
# print(new_exp)

# # load the data and get the useq plate plan from the sequence
# data = TensorstoreZarrReader(new_exp.data_path)
# # Define plate maps for conditions
# plate_maps = {"genotype": {"B5": "WT"}, "treatment": {"B5": "Vehicle"}}
# plate_plan = data.sequence.stage_positions
# assert isinstance(plate_plan, useq.WellPlatePlan)

# # Create plate with plate_maps and conditions in one step
# plate = useq_plate_plan_to_db(plate_plan, new_exp, plate_maps=plate_maps)
# new_exp.plate = plate

# print_experiment_tree(new_exp)

# # Now when we set the experiment in the runner, conditions will be applied
# runner = AnalysisRunner()
# runner.set_data(data)
# runner.set_experiment(new_exp)


# def _p(msg: str) -> None:
#     print("ANALYSIS INFO:", msg)


# runner.analysisInfo.connect(_p)

# runner.run()

# if runner._worker is not None:
#     while runner._worker.is_running:
#         time.sleep(0.1)

# print_experiment_tree(new_exp)
