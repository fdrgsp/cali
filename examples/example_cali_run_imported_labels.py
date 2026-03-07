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
)
from cali.sqlmodel._visualize_experiment import print_cali_results
from cali.util import import_labels_to_database

runner = CaliRunner()

database_name = "results.cali"
database_path = f"{database_name}"
dataset = "tests/test_data/data_and_db_for_tests/evk.tensorstore.zarr"

# None to process all positions or list of global indices e.g. [0, 2, 5]
positions_to_process = None

exp = Experiment.create_from_data("exp", dataset)

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
det_id = import_labels_to_database(database_path, label_map)

# Use det_id (the detection_settings_id from import)
extraction_settings = ExtractionSettings(dff_window=10, threads=3)
runner.run(
    experiment=exp,
    dataset_path=dataset,
    detection_settings=det_id,
    extraction_settings=extraction_settings,
    global_position_indices=positions_to_process,
    output_path=Path(database_path).parent,
    database_name=database_name,
)

analysis_settings = AnalysisSettings(peaks_height_value=2)
runner.run(
    experiment=exp,
    dataset_path=dataset,
    detection_settings=det_id,
    extraction_settings=1,
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
