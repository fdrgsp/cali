"""Test script for loading analysis from JSON and converting to useq.WellPlatePlan."""

from pathlib import Path

from cv2 import exp
import useq
from rich import print

from cali.sqlmodel import (
    experiment_to_useq_plate_plan,
    load_analysis_from_json,
    print_experiment_tree,
    save_experiment_to_db,
)

data_path = (
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
    # "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont.tensorstore.zarr"
)

labels_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_labels"
# labels_path = (
    # "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_labels"
# )

analysis_dir = Path(
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis"
    # "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_analysis"
)

plate = useq.WellPlate.from_str("96-well")
experiment = load_analysis_from_json(
    str(data_path), str(labels_path), str(analysis_dir), plate
)

db_path = analysis_dir / "cali.db"
experiment.database_path = str(db_path)

from rich import print
print(experiment)
print(experiment.plate)
print(experiment.analysis_settings)

session = save_experiment_to_db(experiment, db_path, overwrite=True, keep_session=True)
# print_experiment_tree(experiment, max_level="roi")

plate_plan = experiment_to_useq_plate_plan(experiment)
# print(plate_plan)
