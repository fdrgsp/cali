"""Example script to load an experiment from JSON files and save it to a database."""

import useq

from cali.sqlmodel import (
    load_analysis_from_json,
    print_experiment_tree,
    save_experiment_to_database,
)

# Set paths for data, labels, and analysis directory
# data_path = (
#     "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
# )
# labels_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_labels"
# analysis_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk_analysis"
# data_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont.tensorstore.zarr"  # noqa: E501
# labels_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_labels"  # noqa: E501
# analysis_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/spontaneous/spont_analysis"  # noqa: E501

data_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt.tensorstore.zarr"
labels_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt_labels"
analysis_path = "/Users/fdrgsp/Desktop/cali_test"

# Create useq.WellPlate that matches the experiment
plate = useq.WellPlate.from_str("96-well")

# Load experiment from JSON files
experiment = load_analysis_from_json(
    str(data_path), str(labels_path), str(analysis_path), plate
)

# save experiment to database in the specified above analysis directory with the
# default name "cali.db". specify database_name to use a different name.
save_experiment_to_database(experiment, overwrite=True)

# view experiment tree
print_experiment_tree(experiment, max_level="fov")

from rich import print

print(experiment)
print(experiment.analysis_settings)
print(experiment.plate)
