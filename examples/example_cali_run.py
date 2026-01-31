"""Example of how to run the full CALI pipeline."""

from pathlib import Path

from sqlmodel import create_engine

from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    ExtractionSettings,
)
from cali.sqlmodel._visualize_experiment import print_cali_results

runner = CaliRunner()

database_name = "results.cali"
database_path = f"{database_name}"
dataset = "tests/test_data/evoked/evk.tensorstore.zarr"

# None to process all positions or list of global indices e.g. [0, 2, 5]
positions_to_process = None

exp = Experiment.create_from_data("exp", dataset)

# this will create detection settings with ID 1
detection_settings = DetectionSettings(method="cellpose", model_type="cpsam")
runner.run(
    experiment=exp,
    dataset_path=dataset,
    detection_settings=detection_settings,
    global_position_indices=positions_to_process,
    output_path=Path(database_path).parent,
    database_name=database_name,
    overwrite=True,
)

# this will create extraction settings with ID 1
extraction_settings = ExtractionSettings(dff_window=10, threads=3)
runner.run(
    experiment=exp,
    dataset_path=dataset,
    detection_settings=1,
    extraction_settings=extraction_settings,
    global_position_indices=positions_to_process,
    output_path=Path(database_path).parent,
    database_name=database_name,
)

analysis_settings = AnalysisSettings(peaks_height_value=2)
runner.run(
    experiment=exp,
    dataset_path=dataset,
    detection_settings=1,
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
