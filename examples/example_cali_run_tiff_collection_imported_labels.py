"""Example of how to run the full cali pipeline from tiff files and imported labels.

This combines TiffCollectionReader (for loading TIFF image data) with
import_labels_to_database (for using pre-existing label TIFFs instead of
Cellpose). No Cellpose or GPU is required.
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

# ---- 1. Set up the TIFF collection data ----
# Folder containing your TIFF time-series files
tiff_folder = Path("/path/to/tiffs")

# Build a file_map: well name -> list of TIFF file paths (one per FOV)
# Files are named as: A1_0000.tif, A1_0001.tif, B1_0000.tif, etc.
# Each TIFF is a time-series stack for one FOV.
file_map = {
    "A1": [
        tiff_folder / "A1_0000.tif",
        tiff_folder / "A1_0001.tif",
    ],
    "B1": [
        tiff_folder / "B1_0000.tif",
        tiff_folder / "B1_0001.tif",
    ],
    # ...
}

# None to process all positions or list of global indices e.g. [0, 2, 5]
positions_to_process = None

# ---- 2. Create experiment with TIFF collection settings ----
exp = Experiment.create_from_data(
    "exp",
    str(tiff_folder),
    tiff_file_map=file_map,
    tiff_plate_type="96-well",
    tiff_metadata={"exposure_ms": 100.0, "pixel_size_um": 0.65},
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

det_id = import_labels_to_database(database_path, label_map)

# ---- 4. Run extraction ----
extraction_settings = ExtractionSettings(dff_window=10, threads=3)
runner.run(
    experiment=exp,
    dataset_path=str(tiff_folder),
    detection_settings=det_id,
    extraction_settings=extraction_settings,
    global_position_indices=positions_to_process,
    output_path=Path(database_path).parent,
    database_name=database_name,
)

# ---- 5. Run analysis ----
analysis_settings = AnalysisSettings(peaks_height_value=2)
runner.run(
    experiment=exp,
    dataset_path=str(tiff_folder),
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
