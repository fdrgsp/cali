from pathlib import Path

from sqlmodel import create_engine

from cali.runner import CaliRunner
from cali.sqlmodel import (
    AnalysisSettings,
    DetectionSettings,
    Experiment,
    print_cali_results,
    save_experiment_to_database,
)

data_path = (
    "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/evk.tensorstore.zarr"
)
output_path = "/Users/fdrgsp/Documents/git/cali/tests/test_data/evoked/"

# create a new experiment using the data in the specified directory
exp = Experiment.create_from_data(
    name="My Experiment",
    data_path=data_path,
    plate_maps={
        "genotype": {"B5": "WT"},
        "treatment": {"B5": "Vehicle"},
    },
    description=f"Experiment from {data_path}",
)
# save the experiment to a new database
save_experiment_to_database(
    exp, output_path, database_name="results_2.cali", overwrite=True
)

# Print initial state (no analysis results yet)
engine = create_engine(f"sqlite:///{Path(output_path) / 'results_2.cali'}")
print_cali_results(engine, show_settings=False)
# engine.dispose()

# initialize CaliRunner
runner = CaliRunner()

# specify dataset path, detection settings, and analysis settings
detection_settings = DetectionSettings(
    method="cellpose",
    model_type="custom",
    custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",
)
analysis_settings = AnalysisSettings(
    dff_window=130,
    neuropil_min_pixels=100,
    neuropil_correction_factor=0.7,
    neuropil_inner_radius=2,
)

# run analysis using new settings on existing detected ROIs (detection_id parameter)
runner.run(
    exp,
    data_path,
    detection_settings,
    analysis_settings=analysis_settings,
    global_position_indices=[0],
    output_path=output_path,
    database_name="results_2.cali",
    overwrite=True,
)

# Create a fresh engine connection to see the results
# engine = create_engine(f"sqlite:///{Path(output_path) / 'results_2.cali'}")
print_cali_results(engine, show_settings=True)
# engine.dispose()

analysis_settings = AnalysisSettings(
    dff_window=150,
    neuropil_min_pixels=100,
    neuropil_correction_factor=0.7,
    neuropil_inner_radius=2,
)

runner.run(
    exp,
    data_path,
    1,
    analysis_settings=analysis_settings,
    global_position_indices=[0],
    output_path=output_path,
    database_name="results_2.cali",
)

print_cali_results(engine, show_settings=True)
