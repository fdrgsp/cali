from pathlib import Path

from rich import print

from cali.sqlmodel import Experiment, save_experiment_to_database

data_path = "tests/test_data/evoked/evk.tensorstore.zarr"
output_path = "tests/test_data/evoked/"

print(list(Path(output_path).glob("*.cali")))
assert Path("tests/test_data/evoked/results.cali").exists()


# CREATE THE EXPERIMENT BASED ON DATA -------------------------------------
experiment = Experiment.create_from_data(
    name="Cali Experiment",
    data_path=data_path,
    description=f"Experiment from data at {data_path}.",
)

# SAVE THE EXPERIMENT TO A NEW DATABASE------------------------------------
save_experiment_to_database(
    experiment, output_path, database_name="results_new.cali", overwrite=True
)
