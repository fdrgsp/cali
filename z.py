"""Testing."""

from pathlib import Path

import useq

from cali._plate_viewer._analysis_with_sqlmodel import AnalysisRunner
from cali.readers import TensorstoreZarrReader
from cali.sqlmodel import (
    load_analysis_from_json,
    load_experiment_from_database,
    save_experiment_to_database,
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
save_experiment_to_database(exp_form_json, database_path, overwrite=True)
print_experiment_tree(exp_form_json)
print(exp_form_json.plate.wells)

# load the experiment from the newly created database
print("Loading experiment from database...")
exp = load_experiment_from_database(database_path)
assert exp is not None
print_experiment_tree(exp)
print(exp.plate.wells)

# # delete existing rois
# print("Deleting existing ROIs...")
# for well in exp.plate.wells:
#     for fov in well.fovs:
#         fov.rois = []

# # save database after deletion
# print("Saving experiment after ROI deletion...")
# save_experiment_to_database(exp, database_path, overwrite=True)
# print_experiment_tree(exp)

# # run analysis
# print("Running analysis to recreate ROIs...")
# runner = AnalysisRunner()
# runner.set_experiment(exp)
# data = TensorstoreZarrReader(exp.data_path)
# runner.set_data(data)
# runner.run()
# print_experiment_tree(exp)

# # reopen database to verify the state after analysis
# print("Loading experiment from database after analysis...")
# exp = load_experiment_from_database(database_path)
# assert exp is not None
# print_experiment_tree(exp)
