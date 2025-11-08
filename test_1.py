"""Test script for loading analysis from JSON and converting to useq.WellPlatePlan."""

from pathlib import Path

import useq
from rich import print

from cali.sqlmodel import (
    experiment_to_useq_plate_plan,
    load_analysis_from_json,
    print_model_tree,
    save_experiment_to_db,
)

data_path = (
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
    # "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont.tensorstore.zarr"
)

labels_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_labels"
labels_path = (
    # "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_labels"
)

analysis_dir = Path(
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis"
    # "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_analysis"
)

plate = useq.WellPlate.from_str("96-well")
experiment = load_analysis_from_json(
    str(data_path), str(labels_path), str(analysis_dir), plate
)
experiment_name = experiment.name


db_path = analysis_dir / "cali.db"
session = save_experiment_to_db(experiment, db_path, overwrite=True, keep_session=True)
print_model_tree(experiment, max_level="roi")

plate_plan = experiment_to_useq_plate_plan(experiment)
print(plate_plan)


# from sqlmodel import create_engine
# from cali.slqmodel import print_experiment_tree
# experiment = load_analysis_from_json(analysis_dir, useq_plate=plate)
# experiment_name = experiment.name
# db_path = Path("db_from_json.db")
# session = save_experiment_to_db(experiment, db_path, overwrite=True)
# engine = create_engine(f"sqlite:///{db_path}")
# print_experiment_tree(experiment_name, engine)
