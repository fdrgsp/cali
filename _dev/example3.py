"""Example showing the simplified Experiment.create_with_plate API."""

from cali._constants import EVOKED
from cali.sqlmodel import Experiment, print_experiment_tree
from cali.sqlmodel._util import save_experiment_to_database

# Create experiment with plate structure in one call - much simpler!
exp = Experiment.create(
    name="New Experiment",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    analysis_path="/Users/fdrgsp/Desktop/cali_test",
    database_name="evk.tensorstore.zarr.db",
    plate_type="96-well",
    well_names=["B5", "B6", "C5"],
    fovs_per_well=2,
    plate_maps={
        "genotype": {"B5": "WT", "B6": "KO", "C5": "WT"},
        "treatment": {"B5": "Vehicle", "B6": "Vehicle", "C5": "Drug"},
    },
    experiment_type=EVOKED,
)

# Save to database
save_experiment_to_database(exp, overwrite=True)
print_experiment_tree(exp)


# Create experiment with plate structure in one call - much simpler!
exp = Experiment.create_from_data(
    name="New Experiment",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    analysis_path="/Users/fdrgsp/Desktop/cali_test",
    database_name="evk.tensorstore.zarr.db",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
    experiment_type=EVOKED,
)

# Save to database
save_experiment_to_database(exp, overwrite=True)
print_experiment_tree(exp)
