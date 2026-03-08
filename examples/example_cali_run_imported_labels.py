"""Example of how to run the full cali pipeline with imported labels.

This is the same as example_cali_run.py but uses pre-existing label TIFFs
instead of Cellpose for detection. No Cellpose or GPU is required.
"""

from pathlib import Path

from sqlmodel import create_engine

from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    Experiment,
    ExtractionSettings,
    save_experiment_to_database,
)
from cali.sqlmodel._visualize_experiment import print_cali_results
from cali.util import import_labels_to_database

database_name = "results.cali"
database_path = f"{database_name}"
dataset = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"

# Clean up previous run for this example
if Path(database_path).exists():
    Path(database_path).unlink()

exp = Experiment.create_from_data("exp", dataset)

# ---- 2. Create the database and set up the FOV structure ----
# needed before importing labels so we have FOVs to link the labels to
save_experiment_to_database(
    exp, Path(database_path).parent, database_name=database_name
)

# ---- 3. Import pre-existing label TIFFs ----
# Folder containing label TIFFs
labels_folder = Path("/path/to/label_tiffs")

# Build a label_map: FOV name -> label TIFF path
# FOV names follow the pattern: {well}_{fov_index}, e.g. "A1_0000", "A1_0001"
label_map = {
    "A1_0000": labels_folder / "A1_0000_labels.tif",
    "A1_0001": labels_folder / "A1_0001_labels.tif",
    "B1_0000": labels_folder / "B1_0000_labels.tif",
    "B1_0001": labels_folder / "B1_0001_labels.tif",
    # ...
}

# import the labels into the database and get the assigned detection_settings_id
detection_id = import_labels_to_database(database_path, label_map)

# initialize the runner
runner = CaliRunner()

# None to process all positions or list of global indices e.g. [0, 2, 5]
positions_to_process = None

# ---- 4. Run extraction ----
extraction_settings = ExtractionSettings(dff_window=10, threads=3)
runner.run(
    experiment=exp,
    dataset_path=dataset,
    detection_settings=detection_id,
    extraction_settings=extraction_settings,
    global_position_indices=positions_to_process,
    output_path=Path(database_path).parent,
    database_name=database_name,
)

# ---- 5. Run analysis ----
analysis_settings = AnalysisSettings(peaks_height_value=2)
runner.run(
    experiment=exp,
    dataset_path=dataset,
    detection_settings=detection_id,
    extraction_settings=extraction_settings,
    analysis_settings=analysis_settings,
    global_position_indices=positions_to_process,
    output_path=Path(database_path).parent,
    database_name=database_name,
)

# Print summary of results
engine = create_engine(f"sqlite:///{database_path}")
print_cali_results(engine)

# Clean up engine
engine.dispose(close=True)
