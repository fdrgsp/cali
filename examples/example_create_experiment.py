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


# NOTE: if you used TensorstoreZarrReader or OMEZarrReader from
# micromanager-gui without HCS but with a list of positions, you have to also pass
# the plate_plan argument to Experiment.create_from_data to set the plate structure.
# This is because the data metadata lacks plate information in this case.
# e.g.:
# if you selected manually 1 position per well in a 96-well plate for wells from
# G2 to G11 andf have 2 fovs per well:
import useq

pp = useq.WellPlatePlan(
    plate=useq.WellPlate.from_str("96-well"),
    a1_center_xy=(0.0, 0.0),
    selected_wells=((6,), (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
    well_points_plan=useq.RandomPoints(num_points=2),
)
# If you instead do not want to use a plate plan, you still have to create a plan with
# a simple rectangular of circular dish/coverslip with the number of positions you have.
# e.g.:
# if you have 5 positions:
# pp = useq.WellPlatePlan(
#     plate=useq.WellPlate.from_str("dish-35mm-round"),
#     a1_center_xy=(0.0, 0.0),
#     selected_wells=((0,), (0,)),
#     well_points_plan=useq.RandomPoints(num_points=5)
# )
# Then you pass this plate_plan to Experiment.create_from_data:
exp = Experiment.create_from_data(
    name="New Experiment",
    data_path="tests/test_data/evoked/evk.tensorstore.zarr",
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
    plate_plan=pp,
)
save_experiment_to_database(exp, out, overwrite=True)
print(exp.plate)
print(exp.plate.wells)
