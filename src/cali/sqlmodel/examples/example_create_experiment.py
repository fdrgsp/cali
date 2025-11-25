"""Example showing the simplified Experiment.create_with_plate API."""

from cali.sqlmodel import Experiment
from cali.sqlmodel._util import save_experiment_to_database

# Create experiment with plate structure in one call - much simpler!
exp = Experiment.create(
    name="New Experiment",
    plate_type="96-well",
    well_names=["B5", "B6", "C5"],
    fovs_per_well=3,
    plate_maps={
        "genotype": {"B5": "WT", "B6": "KO", "C5": "WT"},
        "treatment": {"B5": "Vehicle", "B6": "Vehicle", "C5": "Drug"},
    },
)

# Save to database
out = "/Users/fdrgsp/Desktop/cali_test"
save_experiment_to_database(exp, out, overwrite=True)


# Create experiment with plate structure in one call - much simpler!
exp = Experiment.create_from_data(
    name="New Experiment",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
)

# Save to database
save_experiment_to_database(exp, out, overwrite=True)
