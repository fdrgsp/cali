"""Example script to load an experiment from JSON files and save it to a database."""

from pathlib import Path

import useq

from cali.sqlmodel import (
    load_analysis_from_json,
    print_experiment_tree,
    save_experiment_to_db,
)

# Set paths for data, labels, and analysis directory
data_path = "tests/test_data/evoked/evk.tensorstore.zarr"
# data_path = "tests/test_data/spontaneous/spont.tensorstore.zarr"

labels_path = "tests/test_data/evoked/evk_labels"
# labels_path = "tests/test_data/spontaneous/spont_labels"

analysis_dir = "tests/test_data/evoked/evk_analysis"
# analysis_dir = "tests/test_data/spontaneous/spont_analysis"

# Create useq.WellPlate that matches the experiment
plate = useq.WellPlate.from_str("96-well")

# Load experiment from JSON files
experiment = load_analysis_from_json(
    str(data_path), str(labels_path), str(analysis_dir), plate
)

# save experiment to database
db_path = Path(analysis_dir) / "cali.db"
experiment.database_path = str(db_path)
save_experiment_to_db(experiment, db_path, overwrite=True)

# view experiment tree
print_experiment_tree(experiment, max_level="roi")
