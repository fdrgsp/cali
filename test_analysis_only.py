from pathlib import Path

from sqlmodel import create_engine

from cali.runner import CaliRunner
from cali.sqlmodel import AnalysisSettings, DetectionSettings
from cali.sqlmodel._model import Experiment
from cali.sqlmodel._visualize_experiment import print_cali_results

runner = CaliRunner()

database_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
dataset = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/TSC_hSynLAM77_ACTX250730_D36_DIV54_250923_jRCaMP1b_Spt.tensorstore.zarr"

exp = Experiment.create_from_data("exp", dataset)
detection_settings = DetectionSettings(
    method="cellpose",
    model_type="custom",
    custom_model="/Users/fdrgsp/Documents/git/cali/src/cali/detection/cellpose_models/cp3_img8_epoch7000_py",
)
# Different analysis settings to test DB loading
analysis_settings = AnalysisSettings(dff_window=200, threads=8, peaks_height_value=3)

# Run analysis only (detection already exists)
runner.run(
    exp,
    dataset,
    detection_settings,
    analysis_settings=analysis_settings,
    global_position_indices=[16, 17, 18],
    output_path=Path(database_path).parent,
    database_name="results.cali",
    overwrite=False,  # Don't overwrite - use existing DB
)

engine = create_engine(f"sqlite:///{database_path}")
print_cali_results(engine, show_settings=False, max_experiment_level="fov")
